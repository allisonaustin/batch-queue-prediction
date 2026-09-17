import numpy as np
import xgboost as xgb
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
import torch
import gc
from eval.helper import _empty_gpu

# config
_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if _GPU else "cpu")
XGB_DEV = "cuda" if _GPU else "cpu"
CB_TASK = "GPU" if _GPU else "CPU"
LGBM_DEV = "cpu"

def sk_gain(m, nfeat):
    if "lightgbm" in type(m).__module__:
        v = np.asarray(m.booster_.feature_importance("gain"), float)
    elif "xgboost" in type(m).__module__:
        v = np.asarray(m.feature_importances_, float)
    else:                                              # CatBoost
        v = np.asarray(m.get_feature_importance(), float)
    if len(v) < nfeat:
        v = np.concatenate([v, np.zeros(nfeat - len(v))])
    s = v.sum(); return v / s if s > 0 else v

def _xgb_cls(spw, seed=42):
    return xgb.XGBClassifier(
        tree_method="hist", device=XGB_DEV, n_estimators=200, max_depth=8,
        learning_rate=0.1, scale_pos_weight=spw, eval_metric="logloss",
        random_state=seed
    )

def _xgb_reg(seed=42):
    return xgb.XGBRegressor(
        tree_method="hist", device=XGB_DEV, n_estimators=200, max_depth=8,
        learning_rate=0.1, random_state=seed
    )

# Quantiles the wait-time distributional heads are fitted at. The mixture fit
# leaves a conditional SD of ~1.17 in log space even when the component is known,
# so a point estimate cannot be the deliverable; these five heads give a median
# plus 50% and 80% central intervals.
WAIT_TAUS = (0.1, 0.25, 0.5, 0.75, 0.9)


def _xgb_quantile(alphas=WAIT_TAUS, seed=42):
    """Multi-quantile XGBoost head: predicts all of `alphas` from one model.

    XGBoost >= 2.0 fits `reg:quantileerror` with a vector `quantile_alpha`, so
    predict() returns an (n, len(alphas)) array in the order given. Fitting the
    quantiles jointly keeps them from crossing as readily as independent fits do.
    """
    return xgb.XGBRegressor(
        tree_method="hist", device=XGB_DEV, n_estimators=200, max_depth=8,
        learning_rate=0.1, objective="reg:quantileerror",
        quantile_alpha=np.asarray(alphas, dtype=float), random_state=seed
    )


def _lgbm_quantile(alpha, seed=42):
    """Single-quantile LightGBM head; call once per tau."""
    return LGBMRegressor(
        device=LGBM_DEV, objective="quantile", alpha=float(alpha),
        n_estimators=200, num_leaves=255, max_depth=8, learning_rate=0.1,
        n_jobs=-1, verbose=-1, random_state=seed
    )


def _xgb_aft(Xtr, y_lo, y_hi, seed=42, dist="normal", scale=1.0, rounds=200,
             depth=8, lr=0.1):
    """Fits an accelerated-failure-time model and returns the raw Booster.

    `survival:aft` with `aft_loss_distribution="normal"` IS a conditional
    lognormal -- the same family the FermiGrid mixture fit identified -- so the
    model form matches the data rather than being imposed on it. It also accepts
    RIGHT-CENSORED labels, which is what lets never-ran jobs stay in the training
    set: their wait is not missing, it is known only to exceed the time at which
    they left the queue.

    `y_lo` and `y_hi` are in ORIGINAL TIME UNITS (seconds), not log space -- AFT
    takes the log itself. For an observed wait pass y_lo == y_hi == wait; for a
    censored one pass y_lo == wait-so-far and y_hi == np.inf. Predictions come
    back in seconds.
    """
    d = xgb.DMatrix(_xgb_prep(Xtr))
    d.set_float_info("label_lower_bound", np.asarray(y_lo, dtype=np.float32))
    d.set_float_info("label_upper_bound", np.asarray(y_hi, dtype=np.float32))
    params = {
        "objective": "survival:aft", "eval_metric": "aft-nloglik",
        "aft_loss_distribution": dist, "aft_loss_distribution_scale": float(scale),
        "tree_method": "hist", "device": XGB_DEV, "max_depth": depth,
        "learning_rate": lr, "seed": seed,
    }
    return xgb.train(params, d, num_boost_round=rounds)


def _xgb_prep(arr):
    """Converts CPU NumPy arrays to GPU PyTorch tensors if CUDA is enabled."""
    if torch.cuda.is_available() and XGB_DEV.startswith("cuda"):
        return torch.from_numpy(arr).to(XGB_DEV)
    return arr

def _xgb_prep(arr):
    """Ensures contiguous float32 NumPy array on CPU for XGBoost GPU engine."""
    if isinstance(arr, torch.Tensor):
        arr = arr.cpu().numpy()
    return np.ascontiguousarray(arr, dtype=np.float32)

def _sk_cls(lib, spw, seed=42):
    if lib == "lightgbm":
        import os
        avail_cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 4
        
        return LGBMClassifier(
            n_estimators=100,
            num_leaves=31,          # Standard leaf limit (fast & effective)
            max_depth=-1,           # Eliminates depth truncation & "best gain: -inf" warnings
            learning_rate=0.1,
            n_jobs=avail_cores,
            subsample=0.1,          # Bagging: samples 10% (4.8M rows) per tree for high speed
            subsample_freq=1,
            colsample_bytree=0.8,
            min_child_samples=1000, # Stops micro-splits on 48M dataset
            scale_pos_weight=spw,
            # subsample=0.1 above makes this fit genuinely stochastic, so the seed
            # is load-bearing here rather than cosmetic.
            random_state=seed,
            verbose=-1,
            verbosity=-1            # Explicitly quets C++ core warnings
        )
    elif lib == "catboost":
        kw = dict(
            task_type=CB_TASK, 
            devices="0", 
            iterations=200, 
            depth=8, 
            learning_rate=0.1,
            verbose=False, 
            allow_writing_files=False, 
            scale_pos_weight=spw,
            random_seed=seed
        )
        if CB_TASK == "GPU":
            kw["gpu_ram_part"] = 0.5
        return CatBoostClassifier(**kw)

def _sk_reg(lib, seed=42):
    if lib == "lightgbm":
        return LGBMRegressor(device=LGBM_DEV, n_estimators=200, num_leaves=255, max_depth=8,
                             learning_rate=0.1, n_jobs=-1, verbose=-1, random_state=seed)
    elif lib == "catboost":
        kw = dict(task_type=CB_TASK, devices="0", iterations=200, depth=8, learning_rate=0.1,
              verbose=False, allow_writing_files=False, random_seed=seed)
        if CB_TASK == "GPU":
            kw["gpu_ram_part"] = 0.5
        return CatBoostRegressor(**kw)

def _sk_fit(m, Xtr, ytr, sample_weight=None):
    def _do(mm):
        return mm.fit(Xtr, ytr) if sample_weight is None else mm.fit(Xtr, ytr, sample_weight=sample_weight)
    try:
        _do(m); return m
    except Exception as e:
        if getattr(m, "get_params", lambda: {})().get("task_type") == "GPU":
            print(f"  [fit failed: {e}]", flush=True)
            _empty_gpu(); gc.collect()
            p = dict(m.get_params()); p["task_type"] = "CPU"; p.pop("devices", None); p.pop("gpu_ram_part", None)
            m2 = type(m)(**p); _do(m2); return m2
        raise