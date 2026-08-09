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

def _xgb_cls(spw):
    return xgb.XGBClassifier(
        tree_method="hist", device=XGB_DEV, n_estimators=200, max_depth=8,
        learning_rate=0.1, scale_pos_weight=spw, eval_metric="logloss"
    )

def _xgb_reg():
    return xgb.XGBRegressor(
        tree_method="hist", device=XGB_DEV, n_estimators=200, max_depth=8,
        learning_rate=0.1
    )

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

def _sk_cls(lib, spw):
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
            scale_pos_weight=spw
        )
        if CB_TASK == "GPU":
            kw["gpu_ram_part"] = 0.5
        return CatBoostClassifier(**kw)

def _sk_reg(lib):
    if lib == "lightgbm":
        return LGBMRegressor(device=LGBM_DEV, n_estimators=200, num_leaves=255, max_depth=8,
                             learning_rate=0.1, n_jobs=-1, verbose=-1)
    elif lib == "catboost":
        kw = dict(task_type=CB_TASK, devices="0", iterations=200, depth=8, learning_rate=0.1,
              verbose=False, allow_writing_files=False)
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