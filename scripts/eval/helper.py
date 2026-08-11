import argparse
import os
import glob
import json
import warnings
import sys
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix, precision_recall_fscore_support,
    accuracy_score, roc_auc_score, average_precision_score, r2_score,
    matthews_corrcoef
)
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_DISPLAY_NAMES = {
    "xgboost": "XGBoost",
    "xgb": "XGBoost",
    "lightgbm": "LightGBM",
    "lgb": "LightGBM",
    "catboost": "CatBoost",
    "cat": "CatBoost",
    "mlp": "MLP",
    "tabnet": "TabNet",
    "saint": "SAINT",
    "ft": "FT-Transformer",
    "ft_transformer": "FT-Transformer",
    "tsmixer": "TSMixer",
    "tabr": "TabR",
    "hierarchical": "Hierarchical",
}

PREFERRED_MODEL_ORDER = [
    "xgboost",
    "lightgbm",
    "catboost",
    "mlp",
    "tabnet",
    "saint",
    "ft",
    "ft_transformer",
    "tsmixer",
    "tabr",
    "hierarchical"
]

# ---- Memory Helpers ----
def _empty_gpu():
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def _nfeat(parts):
    return int(sum(1 if getattr(p, "ndim", 1) == 1 else p.shape[1] for p in parts))


def _get_slice(parts, idxs):
    """Zero-disk row extraction across list of memory arrays."""
    idxs = np.asarray(idxs)
    if len(parts) == 1:
        return parts[0][idxs]
    cols = [p[idxs] if p.ndim == 2 else p[idxs, None] for p in parts]
    return np.hstack(cols)


# ---- Model Metric & Thresholding Helpers ----
THR_GRID = np.linspace(0.05, 0.95, 91)


def pick_thr(y_true, y_prob, target=1):
    """Find the threshold on P(y=1) that maximizes F1 for class `target`.

    A prediction is positive when `y_prob >= thr`, so raising the threshold trades
    positive-class recall for precision and does the reverse for the negative class.
    The two classes therefore peak at different cuts. `target=0` tunes for the
    negative class's own F1, which is the operating point to use when that class
    names a task of its own rather than "everything else".
    """
    best_f1, best_thr = 0.0, 0.5
    for thr in THR_GRID:
        preds = (y_prob >= thr).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            y_true, preds, average=None, labels=[0, 1], zero_division=0
        )
        if f1[target] > best_f1:
            best_f1, best_thr = f1[target], thr
    return float(best_thr)


def cls_metrics(y_true, y_prob, thr=0.5):
    """Compute standard binary classification evaluation metrics."""
    preds = (y_prob >= thr).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_true, preds, average="binary", zero_division=0
    )
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(pr),
        "recall": float(rc),
        "f1": float(f1),
        "threshold": float(thr),
    }

def cls_metrics_per_class(
    y_true, y_prob, thr=0.5, thr_neg=None, pos_name="hardware", neg_name="payload"
):
    """Binary metrics split out per class, for tasks where both classes are of interest.

    E3 asks two questions of one classifier -- "did we catch the hardware faults?"
    and "did we catch the payload faults?" -- and a single positive-class row only
    answers the first.

    `thr` is the cut tuned for the positive class; `thr_neg`, when given, is the cut
    tuned separately for the negative class. Each class block then reports its
    threshold-dependent metrics at its *own* operating point, since the two tasks
    peak at different cuts. Left as None, both classes are scored at `thr` and the
    two blocks describe one shared decision rule.

    The threshold-free metrics -- ROC AUC and each class's PR AUC -- are unaffected
    by either choice, and are the fairer basis for comparing models.
    """
    y_true = np.asarray(y_true).astype(np.int8)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    n = int(len(y_true))

    def _prfs(t):
        # index 0 -> negative class (neg_name), index 1 -> positive class (pos_name)
        return precision_recall_fscore_support(
            y_true, (y_prob >= t).astype(np.int8), average=None, labels=[0, 1],
            zero_division=0,
        )

    pr, rc, f1, sup = _prfs(thr)
    if thr_neg is None:
        thr_neg = thr
        pr_n, rc_n, f1_n = pr, rc, f1
    else:
        pr_n, rc_n, f1_n, _ = _prfs(thr_neg)

    # ROC AUC is invariant under swapping which class is "positive", so one value
    # describes the ranking for both. Average precision is not -- the negative
    # class needs its own labels and scores inverted.
    roc = float(roc_auc_score(y_true, y_prob))
    ap_pos = float(average_precision_score(y_true, y_prob))
    ap_neg = float(average_precision_score(1 - y_true, 1.0 - y_prob))

    per_class = {
        pos_name: {
            "threshold": float(thr),
            "pr_auc": ap_pos,
            "precision": float(pr[1]),
            "recall": float(rc[1]),
            "f1": float(f1[1]),
            "support": int(sup[1]),
            "prevalence": float(sup[1] / n) if n else 0.0,
        },
        neg_name: {
            "threshold": float(thr_neg),
            "pr_auc": ap_neg,
            "precision": float(pr_n[0]),
            "recall": float(rc_n[0]),
            "f1": float(f1_n[0]),
            "support": int(sup[0]),
            "prevalence": float(sup[0] / n) if n else 0.0,
        },
    }

    return {
        "roc_auc": roc,
        # The remaining shared metrics describe the single decision rule at `thr`,
        # the only one of the two cuts a deployed classifier could actually run.
        "accuracy": float(accuracy_score(y_true, (y_prob >= thr).astype(np.int8))),
        # MCC uses all four confusion-matrix cells and is symmetric between the two
        # classes, so one value scores both tasks and a trivial majority-class
        # classifier gets 0 rather than the ~0.9 accuracy flatters it with.
        "mcc": float(matthews_corrcoef(y_true, (y_prob >= thr).astype(np.int8))),
        "balanced_accuracy": float((rc[0] + rc[1]) / 2.0),
        "macro_f1": float((f1[0] + f1[1]) / 2.0),
        # Each task at its own best cut. Not reachable by one rule, so read it as a
        # per-task ceiling rather than a deployable operating point.
        "macro_f1_tuned": float((f1_n[0] + f1[1]) / 2.0),
        "threshold": float(thr),
        "n_test": n,
        "positive_class": pos_name,
        **per_class,
        # Flat aliases for the positive class, matching the pre-split schema.
        "pr_auc": ap_pos,
        "precision": float(pr[1]),
        "recall": float(rc[1]),
        "f1": float(f1[1]),
    }


def reg_metrics(y_true_log, pred_log):
    """Computes honest wait time regression metrics on both log and original second scales."""
    wt_true = np.expm1(y_true_log)
    wt_pred = np.clip(np.expm1(pred_log), 0, None)
    ae = np.abs(wt_pred - wt_true)

    b_short = wt_true < 600
    b_med = (wt_true >= 600) & (wt_true < 7200)
    b_long = wt_true >= 7200

    # sMAPE on the raw second scale, symmetric so a 10x over- and
    # under-prediction cost the same. Jobs where both true and predicted wait
    # round to ~0 have no meaningful relative error, so they score 0 rather
    # than dividing by a vanishing denominator.
    denom = (np.abs(wt_true) + np.abs(wt_pred)) / 2.0
    nonzero = denom > 1e-8
    smape_vals = np.zeros_like(wt_true, dtype=np.float64)
    if np.any(nonzero):
        smape_vals[nonzero] = np.abs(wt_true[nonzero] - wt_pred[nonzero]) / denom[nonzero]

    return {
        "r2_log": float(r2_score(y_true_log, pred_log)),
        # Scale-free error in log1p space -- the space the models train in, so
        # this is the loss they actually optimize rather than a raw-second
        # figure dominated by the longest queues.
        "mae_log1p": float(np.mean(np.abs(y_true_log - pred_log))),
        "smape_pct": float(100.0 * np.mean(smape_vals)),
        "median_ae_s": float(np.median(ae)),
        "within2x": float(
            np.mean((wt_pred <= 2 * wt_true + 1) & (wt_true <= 2 * wt_pred + 1))
        ),
        "mae_raw_s": float(np.mean(ae)),
        "mae_10m": float(ae[b_short].mean()) if b_short.any() else np.nan,
        "mae_2h": float(ae[b_med].mean()) if b_med.any() else np.nan,
        "mae_long": float(ae[b_long].mean()) if b_long.any() else np.nan,
        # Median AE per bin, alongside the mean: wait times are heavily
        # right-skewed within every bin too, so mean AE can look far worse
        # than the error a typical job in that bin actually sees.
        "median_ae_10m": float(np.median(ae[b_short])) if b_short.any() else np.nan,
        "median_ae_2h": float(np.median(ae[b_med])) if b_med.any() else np.nan,
        "median_ae_long": float(np.median(ae[b_long])) if b_long.any() else np.nan,
    }


def thr_sample(a, max_n=500_000, seed=42):
    """Subsample index array for fast threshold calculation if large."""
    a = np.asarray(a)
    if len(a) > max_n:
        return np.random.default_rng(seed).choice(a, max_n, replace=False)
    return a


VAL_FRAC = 0.10


def holdout_split(tri, frac=VAL_FRAC, order=None, seed=42):
    """Carve a threshold-selection slice out of the training indices.

    The operating point has to be chosen on predictions the model has not fit, or
    it inherits however much that model memorized its training rows -- a bias that
    lands unevenly across model families and so distorts cross-model comparison.
    The returned index arrays are disjoint: fit on the first, pick the cut on the
    second.

    `order` is a per-row sort key (queue-start time, for this dataset). Given one,
    the slice is the most recent `frac` of the training window, so the validation
    rows sit between the fitting rows and the test period exactly as the temporal
    protocol intends. Without it the slice is drawn at random, which is the
    matching choice for the random split protocol.
    """
    tri = np.asarray(tri)
    n_val = int(round(frac * len(tri)))
    if n_val < 1 or n_val >= len(tri):
        raise ValueError(f"frac={frac} yields {n_val} validation rows from {len(tri)}")

    if order is None:
        cut = np.random.default_rng(seed).permutation(len(tri))
    else:
        # argpartition, not a full sort -- only the boundary position matters.
        cut = np.argpartition(np.asarray(order)[tri], len(tri) - n_val)

    # Sorted so downstream mmap slicing stays sequential.
    return np.sort(tri[cut[:-n_val]]), np.sort(tri[cut[-n_val:]])


def log_result(exp_name, model, split, **kwargs):
    pass  # Placeholder for custom logging callbacks


# ---- Model Instantiation & Importance Extraction Helpers ----
class MLPAdapter(nn.Module):
    """Wraps train.mlp.MLP so generic evaluators calling model(X) work seamlessly."""

    def __init__(self, model, n_cat):
        super().__init__()
        self.model = model
        self.n_cat = n_cat

    def forward(self, x):
        if self.n_cat > 0:
            xc = x[:, : self.n_cat].long()
            xn = x[:, self.n_cat :].float()
        else:
            xc = torch.zeros((x.shape[0], 0), dtype=torch.long, device=x.device)
            xn = x.float()
        return self.model(xc, xn)


def predict_proba_model(m_lower, path, X_data, batch_size=32768, device=None):
    """Generates predicted probabilities P(y=1) across tree models, TabNet, and PyTorch architectures."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    X_mat = np.asarray(X_data, dtype=np.float32)

    # 1. Tree Models
    if m_lower in ("xgboost", "xgb"):
        import xgboost as xgb

        try:
            bst = xgb.Booster()
            bst.load_model(path)
            probs = bst.predict(xgb.DMatrix(X_mat))
        except Exception:
            import joblib

            model = joblib.load(path)
            probs = (
                model.predict_proba(X_mat)[:, 1]
                if hasattr(model, "predict_proba")
                else model.predict(X_mat)
            )
        return np.asarray(probs, dtype=np.float32)

    elif m_lower in ("lightgbm", "lgb"):
        import lightgbm as lgb

        try:
            bst = lgb.Booster(model_file=path)
            probs = bst.predict(X_mat)
        except Exception:
            import joblib

            model = joblib.load(path)
            probs = (
                model.predict_proba(X_mat)[:, 1]
                if hasattr(model, "predict_proba")
                else model.predict(X_mat)
            )
        return np.asarray(probs, dtype=np.float32)

    elif m_lower in ("catboost", "cat"):
        from catboost import CatBoostClassifier

        cb = CatBoostClassifier()
        cb.load_model(path)
        return np.asarray(cb.predict_proba(X_mat)[:, 1], dtype=np.float32)

    # 2. TabNet
    elif m_lower == "tabnet":
        from pytorch_tabnet.tab_model import TabNetClassifier

        clf = TabNetClassifier()
        clf.load_model(path)
        return np.asarray(clf.predict_proba(X_mat)[:, 1], dtype=np.float32)

    # 3. PyTorch Deep Learning Models (MLP, SAINT, FT, TSMixer, TabR)
    elif path.endswith((".pt", ".pth")):
        loaded_obj = torch.load(path, map_location=device)
        state_dict = (
            loaded_obj["state_dict"]
            if isinstance(loaded_obj, dict) and "state_dict" in loaded_obj
            else loaded_obj
        )
        if isinstance(state_dict, dict):
            state_dict = {
                k.replace("module.", "").replace("model.", ""): v
                for k, v in state_dict.items()
            }

        net = _instantiate_pytorch_model(
            m_lower, X_mat.shape[1], state_dict=state_dict
        )
        if net is None:
            raise ValueError(f"Could not instantiate model for {m_lower}")

        net = net.to(device)
        if isinstance(state_dict, dict) and m_lower != "mlp":
            try:
                net.load_state_dict(state_dict, strict=True)
            except Exception:
                net.load_state_dict(state_dict, strict=False)
        net.eval()

        probs = []
        with torch.no_grad():
            for start in range(0, len(X_mat), batch_size):
                bx = torch.tensor(
                    X_mat[start : start + batch_size],
                    dtype=torch.float32,
                    device=device,
                )
                try:
                    out = net(bx)
                except TypeError:
                    x_cat = torch.empty(
                        (len(bx), 0), dtype=torch.long, device=device
                    )
                    out = net(x_cat, bx)

                if isinstance(out, tuple):
                    out = out[0]
                if hasattr(out, "logits"):
                    out = out.logits

                if out.ndim > 1 and out.shape[1] > 1:
                    p = torch.softmax(out, dim=1)[:, 1]
                else:
                    p = torch.sigmoid(out.squeeze())

                probs.append(p.cpu().numpy())
        return np.concatenate(probs)

    raise ValueError(f"Unrecognized model path format: {path}")


def evaluate_protocol_models(
    X_all,
    y_all,
    rtr,
    rte,
    tri_t,
    tei_t,
    model_dirs=(
        "/mnt/scratch/fast0/amaustin/dl-tabular-models",
        "/mnt/scratch/fast0/amaustin/tree-models",
        "/mnt/scratch/fast0/amaustin/models",
    ),
    preferred_models=PREFERRED_MODEL_ORDER,
    batch_size=32768,
    output_path="results/protocol_eval_results.json",
):
    """Recomputes exact training thresholds on `trs` training sub-samples, evaluates test sets, and exports JSON results."""
    aliases = {
        "xgboost": ["xgboost", "xgb"],
        "lightgbm": ["lightgbm", "lgb"],
        "catboost": ["catboost", "cat"],
        "mlp": ["mlp"],
        "tabnet": ["tabnet"],
        "saint": ["saint"],
        "ft": ["ft", "ft_transformer"],
        "tsmixer": ["tsmixer"],
        "tabr": ["tabr"],
    }

    results = {
        "models": [],
        "roc_auc_r": [],
        "pr_auc_r": [],
        "prec_r": [],
        "rec_r": [],
        "f1_r": [],
        "thr_r": [],
        "roc_auc_t": [],
        "pr_auc_t": [],
        "prec_t": [],
        "rec_t": [],
        "f1_t": [],
        "thr_t": [],
    }

    # Prepare threshold estimation sub-samples (trs) matching training code
    trs_r = thr_sample(rtr)
    trs_t = thr_sample(tri_t)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for m in preferred_models:
            m_lower = m.lower()
            search_names = aliases.get(m_lower, [m_lower])
            matched_paths = {}

            for split in ["random", "temporal"]:
                for d in model_dirs:
                    if not os.path.exists(d):
                        continue
                    for fname in sorted(os.listdir(d)):
                        fn_lower = fname.lower()
                        if (
                            any(s in fn_lower for s in search_names)
                            and split in fn_lower
                            and fn_lower.endswith(
                                (
                                    ".zip",
                                    ".pt",
                                    ".pth",
                                    ".json",
                                    ".txt",
                                    ".bin",
                                    ".cbm",
                                )
                            )
                        ):
                            matched_paths[split] = os.path.join(d, fname)
                            break
                    if split in matched_paths:
                        break

            if "random" in matched_paths and "temporal" in matched_paths:
                try:
                    print(f"\n[Evaluating] {m.upper()}")

                    # --- 1. Random Split Evaluation ---
                    # Predict on training sample trs to compute exact threshold
                    p_trs_r = predict_proba_model(
                        m_lower, matched_paths["random"], X_all[trs_r], batch_size=batch_size
                    )
                    thr_r = pick_thr(y_all[trs_r], p_trs_r)

                    # Predict on test set rte
                    p_te_r = predict_proba_model(
                        m_lower, matched_paths["random"], X_all[rte], batch_size=batch_size
                    )
                    m_rnd = cls_metrics(y_all[rte], p_te_r, thr=thr_r)

                    # --- 2. Temporal Split Evaluation ---
                    # Predict on training sample trs to compute exact threshold
                    p_trs_t = predict_proba_model(
                        m_lower, matched_paths["temporal"], X_all[trs_t], batch_size=batch_size
                    )
                    thr_t = pick_thr(y_all[trs_t], p_trs_t)

                    # Predict on test set tei_t
                    p_te_t = predict_proba_model(
                        m_lower, matched_paths["temporal"], X_all[tei_t], batch_size=batch_size
                    )
                    m_tmp = cls_metrics(y_all[tei_t], p_te_t, thr=thr_t)

                    display_name = MODEL_DISPLAY_NAMES.get(m_lower, m)
                    results["models"].append(display_name)

                    # Store Metrics
                    results["roc_auc_r"].append(m_rnd["roc_auc"])
                    results["pr_auc_r"].append(m_rnd["pr_auc"])
                    results["prec_r"].append(m_rnd["precision"])
                    results["rec_r"].append(m_rnd["recall"])
                    results["f1_r"].append(m_rnd["f1"])
                    results["thr_r"].append(m_rnd["threshold"])

                    results["roc_auc_t"].append(m_tmp["roc_auc"])
                    results["pr_auc_t"].append(m_tmp["pr_auc"])
                    results["prec_t"].append(m_tmp["precision"])
                    results["rec_t"].append(m_tmp["recall"])
                    results["f1_t"].append(m_tmp["f1"])
                    results["thr_t"].append(m_tmp["threshold"])

                    print(
                        f"  Random   | ROC: {m_rnd['roc_auc']:.3f} | PR: {m_rnd['pr_auc']:.3f} | P: {m_rnd['precision']:.3f} | R: {m_rnd['recall']:.3f} | F1: {m_rnd['f1']:.3f} @ thr={thr_r:.4f}"
                    )
                    print(
                        f"  Temporal | ROC: {m_tmp['roc_auc']:.3f} | PR: {m_tmp['pr_auc']:.3f} | P: {m_tmp['precision']:.3f} | R: {m_tmp['recall']:.3f} | F1: {m_tmp['f1']:.3f} @ thr={thr_t:.4f}"
                    )

                except Exception as e:
                    print(f"[Error] Failed evaluating model {m}: {e}")

    if output_path:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[Saved] Complete evaluation results saved to {output_path}")

    return results


def compute_permutation_importance(
    model, X_eval, y_eval, device="cuda", batch_size=32768, sample_size=None
):
    """Computes Permutation Feature Importance (% ROC-AUC drop) for PyTorch DL models.

    Uses 100% of X_eval if sample_size is None or if len(X_eval) <= sample_size.
    """
    if hasattr(model, "eval"):
        model.eval()

    if sample_size is not None and len(X_eval) > sample_size:
        idx = np.random.choice(len(X_eval), size=sample_size, replace=False)
        X_sub, y_sub = X_eval[idx], y_eval[idx]
    else:
        X_sub, y_sub = X_eval.copy(), y_eval.copy()

    def _predict(X_data):
        preds = []
        for i in range(0, len(X_data), batch_size):
            bx = torch.from_numpy(X_data[i : i + batch_size]).float().to(device)
            with torch.no_grad():
                out = model(bx)
                if hasattr(out, "logits"):
                    out = out.logits
                if out.ndim > 1:
                    out = out.squeeze(-1)
                preds.append(torch.sigmoid(out.float()).cpu().numpy())
        return np.concatenate(preds)

    base_preds = _predict(X_sub)
    base_auc = roc_auc_score(y_sub, base_preds)

    n_features = X_sub.shape[1]
    importance_scores = np.zeros(n_features)

    for f_idx in range(n_features):
        X_perm = X_sub.copy()
        np.random.shuffle(X_perm[:, f_idx])

        perm_preds = _predict(X_perm)
        perm_auc = roc_auc_score(y_sub, perm_preds)

        importance_scores[f_idx] = max(0.0, base_auc - perm_auc)

    total_imp = importance_scores.sum()
    if total_imp > 0:
        importance_scores = importance_scores / total_imp

    return importance_scores


def extract_tree_importance(model_name, file_path, n_feats):
    """Loads saved decision tree models and extracts normalized feature gain importances (summing to 1.0)."""
    m_name = model_name.lower()

    if m_name in ("lightgbm", "lgb"):
        import lightgbm as lgb

        bst = lgb.Booster(model_file=file_path)
        imp = bst.feature_importance(importance_type="gain")

    elif m_name in ("catboost", "cat"):
        from catboost import CatBoostClassifier

        cb = CatBoostClassifier()
        cb.load_model(file_path)
        imp = cb.get_feature_importance()

    elif m_name in ("xgboost", "xgb"):
        import xgboost as xgb

        m = xgb.XGBClassifier()
        m.load_model(file_path)
        imp = m.feature_importances_

    else:
        raise ValueError(f"Unsupported tree model identifier: {model_name}")

    imp = np.nan_to_num(np.asarray(imp, dtype=np.float32))

    if len(imp) < n_feats:
        imp = np.pad(imp, (0, n_feats - len(imp)))
    elif len(imp) > n_feats:
        imp = imp[:n_feats]

    total = np.sum(imp)
    return (imp / total) if total > 0 else imp


def _instantiate_pytorch_model(m_name, n_feats, state_dict=None):
    """Dynamically instantiates PyTorch models using checkpoint dimensions."""
    m_name = m_name.lower()

    if m_name == "mlp":
        from train.mlp import MLP

        if state_dict is not None:
            emb_keys = sorted(
                [
                    k
                    for k in state_dict
                    if k.startswith("embs.") and k.endswith(".weight")
                ],
                key=lambda k: int(k.split(".")[1]),
            )
            cards_f = [state_dict[k].shape[0] - 1 for k in emb_keys]
            sum_emb_dim = sum(state_dict[k].shape[1] for k in emb_keys)

            first_linear_in = state_dict["body.0.weight"].shape[1]
            n_num = first_linear_in - sum_emb_dim
            out_dim = state_dict["head.weight"].shape[0]

            hidden = []
            idx = 0
            while f"body.{idx}.weight" in state_dict:
                hidden.append(state_dict[f"body.{idx}.weight"].shape[0])
                idx += 4

            net = MLP(cards_f=cards_f, n_num=n_num, out=out_dim, hidden=tuple(hidden))
            net.load_state_dict(state_dict)
            return MLPAdapter(net, len(cards_f))
        else:
            net = MLP(cards_f=[], n_num=n_feats)
            return MLPAdapter(net, 0)

    elif m_name in ("ft", "ft_transformer"):
        from train.ft import FTTransformer

        return FTTransformer(num_features=n_feats, d_token=32, depth=2, heads=4)

    elif m_name == "tsmixer":
        import train.tsmixer as tsm_mod

        for cls_name in ("TSMixer", "TSMixerModel"):
            if hasattr(tsm_mod, cls_name):
                cls = getattr(tsm_mod, cls_name)
                for kw in [
                    {"num_features": n_feats},
                    {"in_features": n_feats},
                    {"d_in": n_feats},
                ]:
                    try:
                        return cls(**kw)
                    except Exception:
                        pass

    if m_name == "saint":
        try:
            try:
                from train.saint import SAINT
            except ImportError:
                sys.path.append(os.getcwd())
                from train.saint import SAINT

            d_token = 32
            depth = 2
            heads = 4
            cat_dims = []
            n_num = n_feats

            # Inspect state_dict to reconstruct exact model dimensions
            if isinstance(state_dict, dict):
                if "num_embed" in state_dict:
                    n_num = state_dict["num_embed"].shape[0]
                    d_token = state_dict["num_embed"].shape[1]

                # Infer depth from layer indices
                layer_indices = [
                    int(k.split(".")[1])
                    for k in state_dict.keys()
                    if k.startswith("layers.")
                ]
                if layer_indices:
                    depth = max(layer_indices) + 1

                # Reconstruct cat_dims if categorical features were used
                if "cat_offsets" in state_dict and "cat_embed.weight" in state_dict:
                    offsets = state_dict["cat_offsets"].cpu().tolist()
                    total_embeds = state_dict["cat_embed.weight"].shape[0]
                    cat_dims = []
                    for i in range(len(offsets)):
                        next_off = (
                            offsets[i + 1]
                            if i + 1 < len(offsets)
                            else total_embeds
                        )
                        cat_dims.append(next_off - offsets[i])

            return SAINT(
                n_num=n_num,
                cat_dims=cat_dims,
                d_token=d_token,
                depth=depth,
                heads=heads,
            )
        except Exception as e:
            print(f"[Error Instantiating SAINT] {e}")
            return None

    elif m_name == "tabr":
        import train.tabr as tabr_mod

        for cls_name in ("TabR", "TabRModel"):
            if hasattr(tabr_mod, cls_name):
                cls = getattr(tabr_mod, cls_name)
                for kw in [{"num_features": n_feats}, {"in_features": n_feats}]:
                    try:
                        return cls(**kw)
                    except Exception:
                        pass

    return None


def load_saved_importances(
    model_dirs=(
        "/mnt/scratch/fast0/amaustin/models",
        "/mnt/scratch/fast0/amaustin/tree-models",
        "/mnt/scratch/fast0/amaustin/dl-tabular-models",
    ),
    models=(
        "xgboost",
        "lightgbm",
        "catboost",
        "mlp",
        "tabnet",
        "saint",
        "ft",
        "tabr",
        "tsmixer",
    ),
    splits=("random", "temporal"),
    n_feats=44,
    X_eval=None,
    y_eval=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """Scans directories, loads saved Tree, TabNet, and PyTorch DL models, and returns.

    an imp_dict keyed by (model, split) with normalized importances.
    """
    aliases = {
        "xgboost": ["xgboost", "xgb"],
        "lightgbm": ["lightgbm", "lgb"],
        "catboost": ["catboost", "cat"],
        "mlp": ["mlp"],
        "tabnet": ["tabnet"],
        "saint": ["saint"],
        "ft": ["ft", "ft_transformer"],
        "tabr": ["tabr"],
        "tsmixer": ["tsmixer"],
    }
    imp_dict = {}

    for m in models:
        m_lower = m.lower()
        search_names = aliases.get(m_lower, [m_lower])

        for split in splits:
            matched_path = None

            for d in model_dirs:
                if not os.path.exists(d):
                    continue
                for fname in os.listdir(d):
                    fn_lower = fname.lower()
                    if any(s in fn_lower for s in search_names) and split in fn_lower:
                        matched_path = os.path.join(d, fname)
                        break
                if matched_path:
                    break

            if not matched_path:
                print(f"[Missing] No saved model found for {m}/{split}")
                continue

            # 1. Decision Trees
            if m_lower in ("xgboost", "xgb", "lightgbm", "lgb", "catboost", "cat"):
                try:
                    imp_dict[(m, split)] = extract_tree_importance(
                        m_lower, matched_path, n_feats
                    )
                    print(f"[Loaded Tree Model] {m:10s} ({split:8s}) <- {matched_path}")
                except Exception as e:
                    print(f"[Error] Failed loading tree model {m}/{split} from {matched_path}: {e}")

            elif m_lower == "tabnet":
                try:
                    from pytorch_tabnet.tab_model import TabNetClassifier

                    clf = TabNetClassifier()
                    clf.load_model(matched_path)

                    if X_eval is not None:
                        X_mat = (
                            X_eval.values
                            if hasattr(X_eval, "values")
                            else np.asarray(X_eval)
                        ).astype(np.float32)

                        # TabNet explain computes attention mask importance across X_eval
                        M_explain, _ = clf.explain(X_mat)
                        imp = M_explain.sum(axis=0)
                        imp = np.nan_to_num(imp)
                        total = np.sum(imp)
                        imp_dict[(m, split)] = (imp / total) if total > 0 else imp
                        print(f"[Loaded TabNet Model] {m:10s} ({split:8s}) computed feature importances successfully.")
                    else:
                        print(f"[Warning] Pass X_eval to compute TabNet feature importances: {m}/{split}")
                except Exception as e:
                    print(f"[Error] Failed loading TabNet model {m}/{split}: {e}")

            # 3. PyTorch Deep Learning Models
            elif matched_path.endswith((".pt", ".pth")):
                if X_eval is None or y_eval is None:
                    print(f"[Skipping Permutation Imp] {m:10s} ({split:8s}): Pass X_eval and y_eval.")
                    continue

                try:
                    state_dict = torch.load(matched_path, map_location=device)
                    net = _instantiate_pytorch_model(
                        m_lower, n_feats, state_dict=state_dict
                    )

                    if net is None:
                        print(f"[Warning] Could not auto-instantiate class for {m}. Skipping.")
                        continue

                    net = net.to(device)
                    if m_lower != "mlp":
                        net.load_state_dict(state_dict)

                    imp = compute_permutation_importance(
                        net, X_eval, y_eval, device=device
                    )
                    imp_dict[(m, split)] = imp
                    print(f"[Computed Permutation Imp] {m:10s} ({split:8s}) on {device}")
                except Exception as e:
                    print(f"[Error] Failed permutation importance for {m}/{split}: {e}")

    return imp_dict

def predict_reg_model(path, X_data, batch_size=32768, device=None):
    """Loads and predicts regression targets across Tree models, TabNet, and PyTorch architectures."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    fname = os.path.basename(path).lower()
    X_mat = np.asarray(X_data, dtype=np.float32)

    # 1. Tree Models
    if "xgboost" in fname or "xgb" in fname:
        import xgboost as xgb
        try:
            model = xgb.XGBRegressor()
            model.load_model(path)
            return model.predict(X_mat)
        except Exception:
            bst = xgb.Booster()
            bst.load_model(path)
            return bst.predict(xgb.DMatrix(X_mat))

    elif "lightgbm" in fname or "lgb" in fname:
        import lightgbm as lgb
        model = lgb.Booster(model_file=path)
        return model.predict(X_mat)

    elif "catboost" in fname or "cb" in fname:
        from catboost import CatBoostRegressor
        model = CatBoostRegressor()
        model.load_model(path)
        return model.predict(X_mat)

    # 2. TabNet
    elif "tabnet" in fname or fname.endswith(".zip"):
        from pytorch_tabnet.tab_model import TabNetRegressor
        model = TabNetRegressor()
        model.load_model(path)
        return model.predict(X_mat).squeeze()

    # 3. PyTorch Deep Learning Models (.pt / .pth)
    elif path.endswith((".pt", ".pth")):
        m_lower = None
        for key in ("mlp", "saint", "ft_transformer", "ft", "tsmixer", "tabr"):
            if key in fname:
                m_lower = key
                break

        if m_lower is None:
            raise ValueError(f"Could not infer model architecture type from filename: '{fname}'")

        loaded_obj = torch.load(path, map_location=device)

        # Extract or infer categorical vs continuous column split
        cards_f = loaded_obj.get("cards_f", None) if isinstance(loaded_obj, dict) else None
        n_cat = len(cards_f) if cards_f is not None else 0

        # Extract scaling statistics or fallback to feature standardization
        mean = loaded_obj.get("mean", None) if isinstance(loaded_obj, dict) else None
        std = loaded_obj.get("std", None) if isinstance(loaded_obj, dict) else None

        X_cat = X_mat[:, :n_cat].astype(np.int64)
        X_num = X_mat[:, n_cat:].astype(np.float32)

        if mean is not None and std is not None:
            mean_np = np.asarray(mean, dtype=np.float32)
            std_np = np.asarray(std, dtype=np.float32)
            std_np = np.where(std_np == 0, 1.0, std_np)
            X_num_scaled = np.nan_to_num((X_num - mean_np) / std_np)
        else:
            # Fallback scaling for raw state_dict checkpoints missing training stats
            num_mean = np.mean(X_num, axis=0)
            num_std = np.std(X_num, axis=0)
            num_std = np.where(num_std == 0, 1.0, num_std)
            X_num_scaled = np.nan_to_num((X_num - num_mean) / num_std)

        X_mat_proc = np.hstack([X_cat, X_num_scaled]).astype(np.float32)

        if isinstance(loaded_obj, torch.nn.Module):
            net = loaded_obj.to(device)
        else:
            state_dict = (
                loaded_obj["state_dict"]
                if isinstance(loaded_obj, dict) and "state_dict" in loaded_obj
                else loaded_obj
            )
            if isinstance(state_dict, dict):
                state_dict = {
                    k.replace("module.", "").replace("model.", ""): v
                    for k, v in state_dict.items()
                }

            net = _instantiate_pytorch_model(
                m_lower, X_mat_proc.shape[1], state_dict=state_dict
            )
            if net is None:
                raise ValueError(
                    f"Could not instantiate PyTorch model for '{fname}' (identifier: '{m_lower}')"
                )

            net = net.to(device)
            if isinstance(state_dict, dict) and m_lower != "mlp":
                try:
                    net.load_state_dict(state_dict, strict=True)
                except Exception:
                    net.load_state_dict(state_dict, strict=False)

        net.eval()

        preds = []
        with torch.no_grad():
            for start in range(0, len(X_mat_proc), batch_size):
                bx = torch.tensor(
                    X_mat_proc[start : start + batch_size],
                    dtype=torch.float32,
                    device=device,
                )
                try:
                    out = net(bx)
                except TypeError:
                    if n_cat > 0:
                        xc = bx[:, :n_cat].long()
                        xn = bx[:, n_cat:].float()
                        out = net(xc, xn)
                    else:
                        xc = torch.zeros((len(bx), 0), dtype=torch.long, device=device)
                        out = net(xc, bx)

                if isinstance(out, tuple):
                    out = out[0]
                if hasattr(out, "logits"):
                    out = out.logits

                preds.append(out.squeeze().cpu().numpy())

        return np.concatenate(preds).squeeze()

    # 4. Joblib / Pickle / Generic
    else:
        import joblib
        model = joblib.load(path)
        if hasattr(model, "predict"):
            import inspect
            sig = inspect.signature(model.predict)
            if "n_cat" in sig.parameters:
                n_cat = globals().get("NCAT_MATCH", 0)
                return model.predict(X_mat, n_cat=n_cat).squeeze()
            return model.predict(X_mat).squeeze()

    raise ValueError(f"Unrecognized model file format: {fname}")


def campaign_breakdown(y_true, y_pred, campaign_ids, min_n=100, group_cols=None):
    """Per-campaign wait-time error, for spotting which campaigns are easier
    or harder to predict than the model's overall average. Campaigns with
    fewer than `min_n` test jobs are dropped -- their error estimates are too
    noisy to compare against campaigns with thousands of jobs.

    `group_cols`: optional {name: row-aligned array} of extra categorical
    columns to summarize per campaign (majority value among that campaign's
    jobs) -- e.g. for coloring a campaign-level plot by some coarser
    category. Since a campaign's own metadata columns are normally constant
    across its jobs, majority-vote just recovers that constant value.
    """
    campaign_ids = np.asarray(campaign_ids)
    ae = np.abs(np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64))
    valid = ~np.isnan(campaign_ids) & (campaign_ids >= 0)
    group_cols = group_cols or {}

    out = {}
    for cid in np.unique(campaign_ids[valid]):
        m = valid & (campaign_ids == cid)
        n = int(m.sum())
        if n < min_n:
            continue
        entry = {
            "n_jobs": n,
            "median_ae_s": float(np.median(ae[m])),
            "mean_ae_s": float(np.mean(ae[m])),
        }
        for name, arr in group_cols.items():
            vals, counts = np.unique(np.asarray(arr)[m], return_counts=True)
            entry[name] = float(vals[np.argmax(counts)])
        out[int(cid)] = entry
    return out


def evaluate_wait_models(
    model_paths, X_test, y_test_raw, is_log_pred=True,
    campaign_ids=None, min_campaign_n=100, campaign_group_cols=None,
):
    """Evaluates saved regression models already on disk (no retraining --
    loads each cached model and scores it on X_test/y_test_raw). Reuses
    `reg_metrics` so results match the same schema `eval/harness.py` writes
    to `results/wait_time_results.json`, plus a per-campaign error breakdown
    when `campaign_ids` (row-aligned with X_test) is given. `campaign_group_cols`
    (optional {name: row-aligned array}) is forwarded to `campaign_breakdown`
    to also tag each campaign with the majority value of another column
    (e.g. for coloring a campaign-level plot by some coarser category).

    Returns (df_eval, results_by_model, campaign_by_model):
      - df_eval: flat summary table, one row per model, sorted by r2_log.
      - results_by_model: {raw_key: reg_metrics(...) dict} -- same shape as
        wait_time_results.json's per-model/per-split entries.
      - campaign_by_model: {raw_key: {campaign_id: {...}}} or {} if
        campaign_ids wasn't provided.
    """
    y_true = np.asarray(y_test_raw, dtype=np.float64).ravel()
    rows = []
    results_by_model = {}
    campaign_by_model = {}

    for path in model_paths:
        model_name = os.path.basename(path)
        raw_key = model_name.split("_")[0].lower()

        try:
            preds = predict_reg_model(path, X_test)
            preds = np.asarray(preds, dtype=np.float64).ravel()

            # reg_metrics expects both true and predicted values already in
            # log1p space; clip to prevent np.expm1 overflow downstream.
            pred_log = np.clip(preds, -20.0, 20.0) if is_log_pred else np.log1p(np.maximum(0, preds))
            y_true_clean = np.maximum(0, y_true)
            y_true_log = np.log1p(y_true_clean)

            valid_mask = np.isfinite(y_true_log) & np.isfinite(pred_log)
            if not np.any(valid_mask):
                print(f"[SKIP] {model_name}: No valid finite predictions found.")
                continue

            mm = reg_metrics(y_true_log[valid_mask], pred_log[valid_mask])
            results_by_model[raw_key] = mm
            rows.append({"Model": raw_key, "Model File": model_name, **mm})

            if campaign_ids is not None:
                y_pred = np.maximum(0, np.expm1(pred_log[valid_mask]))
                group_cols = {
                    name: np.asarray(arr)[valid_mask]
                    for name, arr in (campaign_group_cols or {}).items()
                }
                campaign_by_model[raw_key] = campaign_breakdown(
                    y_true_clean[valid_mask],
                    y_pred,
                    np.asarray(campaign_ids)[valid_mask],
                    min_n=min_campaign_n,
                    group_cols=group_cols,
                )

            print(f"[OK] {model_name}")
        except Exception as e:
            print(f"[FAILED] {model_name}: {e}")

    df_eval = pd.DataFrame(rows)
    if not df_eval.empty:
        df_eval = df_eval.sort_values("r2_log", ascending=False).reset_index(drop=True)
    return df_eval, results_by_model, campaign_by_model

# ---- Visualizations ----
def _klabel(k):
    """Converts key identifiers like ('ft', 'random') or 'ft' into clean display labels."""
    if isinstance(k, tuple):
        model_part = MODEL_DISPLAY_NAMES.get(str(k[0]).lower(), str(k[0]))
        rest = [str(x) for x in k[1:]]
        return " ".join([model_part] + rest)
    return MODEL_DISPLAY_NAMES.get(str(k).lower(), str(k))


def plot_confusions(cm_dict, class_labels, title, ncols=None, normalize=True):
    keys = list(cm_dict)
    if not keys:
        print(f"[skip] {title}: no data (run the producing cell first)")
        return
    ncols = ncols or len(keys)
    nrows = int(np.ceil(len(keys) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.8 * ncols, 4.5 * nrows), squeeze=False
    )
    for ax in axes.flat:
        ax.axis("off")
    for i, k in enumerate(keys):
        ax = axes.flat[i]
        ax.axis("on")
        cm = np.asarray(cm_dict[k], float)
        M = cm / np.maximum(cm.sum(1, keepdims=True), 1) if normalize else cm
        sns.heatmap(
            M,
            annot=True,
            fmt=".2f" if normalize else ".0f",
            cmap="Blues",
            vmin=0,
            vmax=1 if normalize else None,
            cbar=False,
            square=True,
            xticklabels=class_labels,
            yticklabels=class_labels,
            annot_kws={"size": 12},
            ax=ax,
        )
        ax.set_title(_klabel(k), pad=10)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.suptitle(title, y=1.03)
    plt.tight_layout()
    plt.show()


def imp_heatmap_x(imp_dict, feats, row_keys=None, title="Feature Importances"):
    if row_keys is None:
        row_keys = sorted(list(imp_dict.keys()))
    row_keys = [k for k in row_keys if k in imp_dict]
    if not row_keys:
        print(f"[skip] {title}: no importances found")
        return
    M = np.vstack([np.asarray(imp_dict[k]) for k in row_keys]) * 100
    df = pd.DataFrame(M, index=[_klabel(k) for k in row_keys], columns=feats)
    df = df[df.mean(axis=0).sort_values(ascending=False).index]
    fig, ax = plt.subplots(
        figsize=(4 + 0.65 * len(df.columns), 2.5 + 0.9 * len(row_keys))
    )
    sns.heatmap(
        df,
        cmap="rocket_r",
        annot=True,
        fmt=".0f",
        cbar_kws={"label": "share of gain / importance (%)", "shrink": 0.8},
        annot_kws={"size": 11},
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )
    ax.tick_params(length=0)
    ax.set_title(title, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    plt.show()


def imp_diff_heatmap(
    imp_dict,
    feats,
    models=None,
    title="Feature Gain Shift: Random vs. Temporal Split Protocol on Failure Prediction Task",
    per_model_scale=True,
):
    """Renders heatmaps comparing (Temporal - Random) importance shift."""
    all_keys = list(imp_dict.keys())
    available_models = (
        set(k[0] for k in all_keys if isinstance(k, tuple))
        if models is None
        else set(models)
    )

    # Order rows according to PREFERRED_MODEL_ORDER
    ordered_models = []
    for pref in PREFERRED_MODEL_ORDER:
        for m in available_models:
            if m.lower() == pref and m not in ordered_models:
                if (m, "random") in imp_dict and (m, "temporal") in imp_dict:
                    ordered_models.append(m)

    for m in sorted(available_models):
        if (
            m not in ordered_models
            and (m, "random") in imp_dict
            and (m, "temporal") in imp_dict
        ):
            ordered_models.append(m)

    rows = {}
    for m in ordered_models:
        delta = (
            np.asarray(imp_dict[(m, "temporal")])
            - np.asarray(imp_dict[(m, "random")])
        ) * 100
        disp_name = MODEL_DISPLAY_NAMES.get(m.lower(), m)
        rows[disp_name] = delta

    if not rows:
        print(
            f"[skip] {title}: need both (random & temporal) importances for target models"
        )
        return

    df = pd.DataFrame(rows).T
    df.columns = feats

    # Filter out negligible features (< 0.2%) and sort by largest absolute shift
    df = df.loc[:, df.abs().max(axis=0) >= 0.2]
    sorted_cols = df.abs().max(axis=0).sort_values(ascending=False).index
    df = df[sorted_cols]

    n_models = len(df.index)

    sns.plotting_context("talk")
    
    if per_model_scale:
        fig, axes = plt.subplots(
            n_models,
            1,
            figsize=(4 + 0.9 * len(df.columns), 1.5 * n_models + 1.2),
            sharex=True,
            squeeze=False,
        )

        for i, model_name in enumerate(df.index):
            ax = axes[i, 0]
            row_df = df.loc[[model_name]]
            lim = float(np.nanmax(np.abs(row_df.values))) or 1.0
            is_last = i == n_models - 1

            sns.heatmap(
                row_df,
                cmap="vlag",
                center=0,
                vmin=-lim,
                vmax=lim,
                annot=True,
                fmt=".2f",
                annot_kws={"size": 16},
                cbar_kws={
                    "label": "Δ (%)",
                    "shrink": 0.85,
                    "pad": 0.01,
                },
                linewidths=0.5,
                linecolor="white",
                xticklabels=df.columns if is_last else False,
                ax=ax,
            )
            ax.tick_params(length=0)
            ax.set_ylabel("")
            ax.set_yticklabels(
                ax.get_yticklabels(), rotation=0, fontweight="bold", fontsize=20
            )

            if is_last:
                ax.set_xticklabels(
                    ax.get_xticklabels(), rotation=45, ha="right", fontsize=20
                )
            else:
                ax.set_xlabel("")

        fig.suptitle(title, y=1.02, fontsize=30)
        plt.tight_layout()
        plt.show()

    else:
        lim = float(np.nanmax(np.abs(df.values))) or 1.0
        fig, ax = plt.subplots(
            figsize=(4 + 0.95 * len(df.columns), 2.5 + 0.8 * len(rows))
        )
        sns.heatmap(
            df,
            cmap="vlag",
            center=0,
            vmin=-lim,
            vmax=lim,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 13},
            cbar_kws={
                "label": "temporal - random gain (%)",
                "shrink": 0.8,
                "pad": 0.01,
            },
            linewidths=0.5,
            linecolor="white",
            ax=ax,
        )
        ax.tick_params(length=0)
        ax.set_title(title, pad=12)
        ax.set_xlabel("")
        ax.set_ylabel("")

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        plt.tight_layout()

        notebook_dir = os.getcwd()
        figures_dir = os.path.join(notebook_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)

        pdf_path = os.path.join(figures_dir, "imp_diff_heatmap.pdf")
        png_path = os.path.join(figures_dir, "imp_diff_heatmap.png")

        plt.savefig(pdf_path, bbox_inches="tight")
        plt.savefig(png_path, bbox_inches="tight", dpi=300)
        plt.show()


if __name__ == "__main__":
    DATA_DIR = "/mnt/scratch/fast0/amaustin/datasets/fife/"
    targets = np.load(os.path.join(DATA_DIR, "targets_and_masks.npz"))

    Xmatch = np.load(os.path.join(DATA_DIR, "Xmatch.npy"), mmap_mode="r")
    Xsub = np.load(os.path.join(DATA_DIR, "Xsub.npy"), mmap_mode="r")

    failed = np.load(os.path.join(DATA_DIR, "failed.npy"), mmap_mode="r")
    hw = np.load(os.path.join(DATA_DIR, "hw.npy"), mmap_mode="r")
    wait_sv = np.load(os.path.join(DATA_DIR, "wait_sv.npy"), mmap_mode="r")
    tr_mask = np.load(os.path.join(DATA_DIR, "tr_mask.npy"), mmap_mode="r")
    te_mask = np.load(os.path.join(DATA_DIR, "te_mask.npy"), mmap_mode="r")

    yv = failed
    idx = np.arange(len(yv))

    # Temporal split indices
    tri_t = np.where(tr_mask)[0]
    tei_t = np.where(te_mask)[0]

    # Random split indices
    rng = np.random.default_rng(0)
    perm = rng.permutation(idx)
    rte = np.sort(perm[: len(tei_t)])
    rtr = np.sort(perm[len(tei_t) :])

    print(f"Data loaded: Total {len(yv):,} rows | Feature matrix {Xmatch.shape}")
    print(f"Random split: {len(rtr):,} train / {len(rte):,} test")
    print(f"Temporal split: {len(tri_t):,} train / {len(tei_t):,} test")

    evaluate_protocol_models(
        X_all=Xmatch,
        y_all=yv,
        rtr=rtr,
        rte=rte,
        tri_t=tri_t,
        tei_t=tei_t
    )