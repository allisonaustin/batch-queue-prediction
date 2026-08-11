import os
import gc
import json
import time
import traceback
import numpy as np
import psutil
import torch
import argparse
import joblib
from sklearn.metrics import confusion_matrix
from eval.helper import (
    _empty_gpu, _nfeat, _get_slice, log_result, thr_sample, pick_thr,
    cls_metrics, cls_metrics_per_class, reg_metrics, THR_GRID, holdout_split,
)
from train.tree import _xgb_reg, _sk_reg, _xgb_cls, _xgb_prep, sk_gain, _sk_cls, _sk_fit
from train.mlp import mlp_fit_eval
from train.tabnet import tabnet_fit_eval
from train.saint import saint_fit_eval
from train.ft import ft_fit_eval
from train.tabr import tabr_fit_eval
from train.tsmixer import tsmixer_fit_eval
from train.hierarchical import hierarchical_fit_eval

def mem_gb():
    return psutil.Process().memory_info().rss / (1024 ** 3)

def save_experiment_results(exp_name, lib, got_metrics, output_dir=None):
    """Loads existing experiment JSON, updates the entries for the model, and saves back to disk."""
    output_dir = output_dir if output_dir is not None else os.getcwd() + "/results"
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{exp_name}_results.json")

    data = {}
    if os.path.exists(json_path) and os.path.getsize(json_path) > 0:
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = {}

    # Merge new split metrics for this model library
    if lib not in data:
        data[lib] = {}
    data[lib].update(got_metrics)

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"[{exp_name.upper()}] Appended results for model '{lib}' to {json_path}", flush=True)

def fit_eval_binary(
    lib, parts, tri, tei, trs, yv, spw, want_imp=False, ncat=None, split=None, exp_tag=None,
    class_names=None,
):
    """Fits the requested binary model library and returns metrics, importance, and CM.

    `class_names` as a (positive, negative) pair switches the metric block to the
    per-class form -- used by E3, where both classes name a real task. Left as None
    (E1) the metrics stay positive-class only.
    """
    yva = np.asarray(yv)
    nfeat = _nfeat(parts)
    imp = None
    model_dir = "/mnt/scratch/fast0/amaustin/models/"

    if lib == "mlp":
        p_te, p_trs = mlp_fit_eval(
            parts, tri, tei, ncat, "bin", yv, spw=spw, trs=trs, split=split, exp_tag=exp_tag
        )

    elif lib == "tabnet":
        p_te, p_trs, imp = tabnet_fit_eval(
            parts, tri, tei, ncat, "bin", yv, spw=spw, trs=trs, want_imp=want_imp, split=split, exp_tag=exp_tag
        )

    elif lib == "saint":
        p_te, p_trs, imp = saint_fit_eval(
            parts, tri, tei, ncat, "bin", yv, spw=spw, trs=trs, want_imp=want_imp, split=split, exp_tag=exp_tag
        )

    elif lib == "ft":
        p_te, p_trs, imp = ft_fit_eval(
            parts, tri, tei, ncat, "bin", yv, spw=spw, trs=trs, want_imp=want_imp, split=split, exp_tag=exp_tag
        )

    elif lib == "tabr":
        p_te, p_trs, imp = tabr_fit_eval(
            parts, tri, tei, ncat, "bin", yv, spw=spw, trs=trs, want_imp=want_imp, split=split, exp_tag=exp_tag
        )

    elif lib == "tsmixer":
        p_te, p_trs, imp = tsmixer_fit_eval(
            parts, tri, tei, ncat, "bin", yv, spw=spw, trs=trs, want_imp=want_imp, split=split, exp_tag=exp_tag
        )


    elif lib == "xgboost":
        Xtr = _get_slice(parts, tri)
        m = _xgb_cls(spw)
        m.fit(Xtr, yva[tri])

        X_trs = _xgb_prep(_get_slice(parts, trs))
        X_tei = _xgb_prep(_get_slice(parts, tei))

        p_trs = m.predict_proba(X_trs)[:, 1]
        p_te = m.predict_proba(X_tei)[:, 1]

        if hasattr(p_trs, "cpu"):
            p_trs = p_trs.cpu().numpy()
            p_te = p_te.cpu().numpy()

        if want_imp:
            imp = sk_gain(m, nfeat)

        if split is not None:
            model_dir = "/mnt/scratch/fast0/amaustin/tree-models/"
            os.makedirs(model_dir, exist_ok=True)
            save_path = os.path.join(model_dir, f"{lib}_bin_{exp_tag}_{split}.json")
            m.save_model(save_path)
            print(f"[{lib}] Saved model to {save_path}", flush=True)

        del m, Xtr, X_trs, X_tei

    else:
        Xtr = _get_slice(parts, tri)
        m = _sk_fit(_sk_cls(lib, spw), Xtr, yva[tri])
        p_trs = m.predict_proba(_get_slice(parts, trs))[:, 1]
        p_te = m.predict_proba(_get_slice(parts, tei))[:, 1]
        if want_imp:
            imp = sk_gain(m, nfeat)

        if split is not None:
            os.makedirs(model_dir, exist_ok=True)
            save_path = os.path.join(model_dir, f"{lib}_bin_{split}.txt")
            if lib in ("lightgbm", "lgb"):
                m.booster_.save_model(save_path)
            elif lib in ("catboost", "cb"):
                m.save_model(save_path)
            else:
                joblib.dump(m, save_path)
            print(f"[{lib}] Saved model to {save_path}", flush=True)

        del m, Xtr

    _empty_gpu()
    gc.collect()

    thr = pick_thr(yva[trs], p_trs)
    if class_names is not None:
        # Each task gets its own operating point, tuned on the same held-out sample.
        thr_neg = pick_thr(yva[trs], p_trs, target=0)
        mm = cls_metrics_per_class(
            yva[tei], p_te, thr, thr_neg=thr_neg,
            pos_name=class_names[0], neg_name=class_names[1],
        )
    else:
        mm = cls_metrics(yva[tei], p_te, thr)
    cm = confusion_matrix(
        yva[tei].astype(np.int8), (p_te >= thr).astype(np.int8), labels=[0, 1]
    )
    return mm, imp, cm

def fit_eval_reg(
    lib, parts, tri, tei, trs, yv, want_imp=False, ncat=None, split=None, exp_tag=None
):
    """Fits the requested regression model library and returns wait time regression metrics and importance."""
    yva = np.asarray(yv)
    nfeat = _nfeat(parts)
    imp = None
    model_dir = "/mnt/scratch/fast0/amaustin/models/"

    if lib == "mlp":
        p_te, p_trs = mlp_fit_eval(
            parts, tri, tei, ncat, "reg", yv, trs=trs, split=split, exp_tag=exp_tag
        )

    elif lib == "tabnet":
        p_te, p_trs, imp = tabnet_fit_eval(
            parts, tri, tei, ncat, "reg", yv, trs=trs, want_imp=want_imp, split=split, exp_tag=exp_tag
        )

    elif lib == "saint":
        p_te, p_trs, imp = saint_fit_eval(
            parts, tri, tei, ncat, "reg", yv, trs=trs, want_imp=want_imp, split=split, exp_tag=exp_tag, is_regression=True
        )

    elif lib == "ft":
        p_te, p_trs, imp = ft_fit_eval(
            parts, tri, tei, ncat, "reg", yv, trs=trs, want_imp=want_imp, split=split, exp_tag=exp_tag, is_regression=True
        )

    elif lib == "tabr":
        p_te, p_trs, imp = tabr_fit_eval(
            parts, tri, tei, ncat, "reg", yv, trs=trs, want_imp=want_imp, split=split, exp_tag=exp_tag, is_regression=True
        )

    elif lib == "tsmixer":
        p_te, p_trs, imp = tsmixer_fit_eval(
            parts, tri, tei, ncat, "reg", yv, trs=trs, want_imp=want_imp, split=split, exp_tag=exp_tag, is_regression=True
        )

    elif lib == "hierarchical":
        p_te, p_trs, imp = hierarchical_fit_eval(
            parts, tri, tei, ncat, "reg", yv, trs=trs, want_imp=want_imp, split=split, exp_tag=exp_tag
        )

    elif lib in ("xgboost", "xgb"):
        Xtr = _get_slice(parts, tri)
        m = _xgb_reg()
        m.fit(Xtr, yva[tri])

        X_trs = _xgb_prep(_get_slice(parts, trs))
        X_tei = _xgb_prep(_get_slice(parts, tei))

        p_trs = m.predict(X_trs)
        p_te = m.predict(X_tei)

        if hasattr(p_trs, "cpu"):
            p_trs = p_trs.cpu().numpy()
            p_te = p_te.cpu().numpy()

        if want_imp:
            imp = sk_gain(m, nfeat)

        if split is not None:
            model_dir = "/mnt/scratch/fast0/amaustin/tree-models/"
            os.makedirs(model_dir, exist_ok=True)
            save_path = os.path.join(model_dir, f"{lib}_reg_{exp_tag}_{split}.json")
            m.save_model(save_path)
            print(f"[{lib}] Saved regression model to {save_path}", flush=True)

        del m, Xtr, X_trs, X_tei

    else:
        Xtr = _get_slice(parts, tri)
        m = _sk_fit(_sk_reg(lib), Xtr, yva[tri])
        p_trs = m.predict(_get_slice(parts, trs))
        p_te = m.predict(_get_slice(parts, tei))
        if want_imp:
            imp = sk_gain(m, nfeat)

        if split is not None:
            os.makedirs(model_dir, exist_ok=True)
            save_path = os.path.join(model_dir, f"{lib}_reg_{exp_tag}_{split}.txt")
            if lib in ("lightgbm", "lgb"):
                m.booster_.save_model(save_path)
            elif lib in ("catboost", "cb"):
                m.save_model(save_path)
            else:
                joblib.dump(m, save_path)
            print(f"[{lib}] Saved regression model to {save_path}", flush=True)

        del m, Xtr

    _empty_gpu()
    gc.collect()

    mm = reg_metrics(yva[tei], p_te)
    return mm, imp, p_te


def run_e1_model(lib, Xm, yv, splits, imp_store, cm_store, ncat, order=None):
    """Executes evaluation across all splits for a given model architecture.

    `order` is the per-row time key used to carve the threshold-selection slice off
    the end of the training window on temporal splits; see `holdout_split`.
    """
    yva = np.asarray(yv)
    got = {}
    for split, a, b in splits:
        try:
            fit_i, val_i = holdout_split(a, order=order if split == "temporal" else None)
            trs = thr_sample(val_i)
            spw = float((yva[fit_i] == 0).sum() / max((yva[fit_i] == 1).sum(), 1))
            print(
                f"[{lib}/{split}] fit {len(fit_i):,} | threshold-selection holdout "
                f"{len(val_i):,} ({len(val_i) / len(a):.0%}"
                f"{', most recent' if order is not None and split == 'temporal' else ', random'})",
                flush=True,
            )
            t_start = time.perf_counter()
            mm, imp, cm = fit_eval_binary(
                lib,
                [Xm],
                fit_i,
                b,
                trs,
                yv,
                spw,
                want_imp=True,
                ncat=ncat,
                split=split
            )
            if imp is not None:
                imp_store[(lib, split)] = imp

            cm_store[(lib, split)] = cm
            got[split] = mm
            # log_result("E1", model=lib, split=split, **mm)
            print(
                f"{'model':9s} {'split':9s}  ROC    PR     F1     Prec   Rec\n"
                f"{lib:9s} {split:9s}  {mm['roc_auc']:.3f}  {mm['pr_auc']:.3f}  {mm['f1']:.3f}  "
                f"{mm['precision']:.3f}  {mm['recall']:.3f}",
                flush=True,
            )
            t_end = time.perf_counter()
            print(f"Total time: {t_end - t_start:.2f} seconds")
        except Exception as e:
            traceback.print_exc()
            print(f"  [skip] {lib}/{split}: {e}", flush=True)
            _empty_gpu()
            gc.collect()

    if "random" in got and "temporal" in got:
        d, t = got["random"], got["temporal"]
        print(
            f"{lib:9s} {'Delta':9s}  {t['roc_auc'] - d['roc_auc']:+.3f}  {t['pr_auc'] - d['pr_auc']:+.3f}  "
            f"{t['f1'] - d['f1']:+.3f}  {t['precision'] - d['precision']:+.3f}  {t['recall'] - d['recall']:+.3f}"
        )

    if got:
        save_experiment_results("protocol_eval", lib, got)

    return got


def run_e2_model(lib, Xsub, wait_log, splits, imp_store, ncat):
    """Executes submit-time wait regression across requested splits for a given model architecture."""
    got = {}
    for split, a, b in splits:
        try:
            # Mask out unobserved or NaN wait times
            vw_a = ~np.isnan(wait_log[a])
            vw_b = ~np.isnan(wait_log[b])
            tri = a[vw_a]
            tei = b[vw_b]
            trs = thr_sample(tri)

            t_start = time.perf_counter()
            mm, imp, _ = fit_eval_reg(
                lib,
                [Xsub],
                tri,
                tei,
                trs,
                wait_log,
                want_imp=True,
                ncat=ncat,
                split=split,
                exp_tag="e2"
            )
            if imp is not None:
                imp_store[(lib, split)] = imp

            got[split] = mm
            log_result("E2", model=lib, split=split, **mm)
            print(
                f"{'model':9s} {'split':9s}  R2(log)  Med-AE(s)  Within-2x  MAE(<10m)  MAE(10m-2h) MAE(>2h)\n"
                f"{lib:9s} {split:9s}  {mm['r2_log']:.3f}    {mm['median_ae_s']:7.0f}s  {mm['within2x']:.3f}     "
                f"{mm['mae_10m']:8.0f}s  {mm['mae_2h']:8.0f}s   {mm['mae_long']:8.0f}s",
                flush=True,
            )
            t_end = time.perf_counter()
            print(f"Total time: {t_end - t_start:.2f} seconds")
        except Exception as e:
            traceback.print_exc()
            print(f"  [skip] {lib}/{split}: {e}", flush=True)
            _empty_gpu()
            gc.collect()

    if got:
        save_experiment_results("wait_time", lib, got)

    return got

E3_TASKS = ("hardware", "payload")
_E3_HDR = f"{'model':10s} {'split':10s} {'task':9s}  Thr   PR     F1     Prec   Rec     Support"


def _fmt_e3_block(lib, split, mm):
    """One row per attribution task at its own tuned threshold, then shared metrics."""
    lines = [_E3_HDR]
    for task in E3_TASKS:
        t = mm[task]
        # A cut pinned to the end of the search grid means F1 for that task was still
        # climbing -- the operating point is a grid artifact, not an optimum.
        edge = " *" if t["threshold"] <= THR_GRID[0] or t["threshold"] >= THR_GRID[-1] else ""
        lines.append(
            f"{lib:10s} {split:10s} {task:9s}  {t['threshold']:.2f}  {t['pr_auc']:.3f}  "
            f"{t['f1']:.3f}  {t['precision']:.3f}  {t['recall']:.3f}  {t['support']:>9,d}{edge}"
        )
    lines.append(
        f"{'':10s} {'':10s} {'shared':9s}  ROC {mm['roc_auc']:.3f}  MCC {mm['mcc']:.3f}  "
        f"Acc {mm['accuracy']:.3f}  BalAcc {mm['balanced_accuracy']:.3f}  "
        f"MacroF1 {mm['macro_f1']:.3f} (tuned {mm['macro_f1_tuned']:.3f})"
    )
    if any(
        mm[t]["threshold"] <= THR_GRID[0] or mm[t]["threshold"] >= THR_GRID[-1]
        for t in E3_TASKS
    ):
        lines.append(
            f"{'':10s} {'':10s} * threshold at the edge of the "
            f"[{THR_GRID[0]:.2f}, {THR_GRID[-1]:.2f}] search grid"
        )
    return "\n".join(lines)


def _fmt_e3_delta(lib, rnd, tmp):
    """Random -> temporal shift, per task. Negative means the temporal split is harder."""
    lines = [f"{'model':10s} {'split':10s} {'task':9s}  dThr   dPR    dF1    dPrec  dRec"]
    for task in E3_TASKS:
        d, t = rnd[task], tmp[task]
        lines.append(
            f"{lib:10s} {'Delta':10s} {task:9s}  {t['threshold'] - d['threshold']:+.2f}  "
            f"{t['pr_auc'] - d['pr_auc']:+.3f}  "
            f"{t['f1'] - d['f1']:+.3f}  {t['precision'] - d['precision']:+.3f}  "
            f"{t['recall'] - d['recall']:+.3f}"
        )
    lines.append(
        f"{'':10s} {'':10s} {'shared':9s}  dROC {tmp['roc_auc'] - rnd['roc_auc']:+.3f}  "
        f"dMCC {tmp['mcc'] - rnd['mcc']:+.3f}  "
        f"dBalAcc {tmp['balanced_accuracy'] - rnd['balanced_accuracy']:+.3f}  "
        f"dMacroF1 {tmp['macro_f1'] - rnd['macro_f1']:+.3f}"
    )
    return "\n".join(lines)


def run_e3_model(lib, Xm, failed, hw, splits, imp_store, cm_store, ncat, order=None):
    """Executes fault attribution evaluation (hardware vs. payload failure) conditioned on job failure.

    `order` is the per-row time key used to carve the threshold-selection slice off
    the end of the training window on temporal splits; see `holdout_split`.
    """
    hw_a = np.asarray(hw)
    failed_a = np.asarray(failed)
    got = {}

    total_failed = (failed_a == 1).sum()
    total_hw = ((failed_a == 1) & (hw_a == 1)).sum()
    total_payload = ((failed_a == 1) & (hw_a == 0)).sum()
    
    print("\n" + "=" * 60, flush=True)
    print(f"  Total Failed Jobs: {total_failed:,}", flush=True)
    print(f"  Hardware Faults  : {total_hw:,} ({total_hw / total_failed * 100:.2f}%)", flush=True)
    print(f"  Payload Faults   : {total_payload:,} ({total_payload / total_failed * 100:.2f}%)", flush=True)
    print("=" * 60 + "\n", flush=True)

    for split, a, b in splits:
        try:
            # Mask to evaluate ONLY jobs that failed (failed == 1)
            tri = a[failed_a[a] == 1]
            tei = b[failed_a[b] == 1]

            # Fit and threshold-selection rows must be disjoint, so the operating
            # point is chosen on predictions this model has not already seen.
            fit_i, val_i = holdout_split(tri, order=order if split == "temporal" else None)
            trs = thr_sample(val_i)
            spw = float((hw_a[fit_i] == 0).sum() / max((hw_a[fit_i] == 1).sum(), 1))
            print(
                f"[{lib}/{split}] fit {len(fit_i):,} | threshold-selection holdout "
                f"{len(val_i):,} ({len(val_i) / len(tri):.0%}"
                f"{', most recent' if order is not None and split == 'temporal' else ', random'}"
                f", {(hw_a[val_i] == 1).mean() * 100:.2f}% hardware)",
                flush=True,
            )

            t_start = time.perf_counter()
            mm, imp, cm = fit_eval_binary(
                lib,
                [Xm],
                fit_i,
                tei,
                trs,
                hw,
                spw,
                want_imp=True,
                ncat=ncat,
                split=split,
                exp_tag="e3_fault",
                class_names=("hardware", "payload"),
            )
            if imp is not None:
                imp_store[(lib, split)] = imp

            cm_store[(lib, split)] = cm
            got[split] = mm
            log_result("E3", model=lib, split=split, **mm)
            print(_fmt_e3_block(lib, split, mm), flush=True)
            t_end = time.perf_counter()
            print(f"Total time: {t_end - t_start:.2f} seconds")
        except Exception as e:
            traceback.print_exc()
            print(f"  [skip] {lib}/{split}: {e}", flush=True)
            _empty_gpu()
            gc.collect()

    if "random" in got and "temporal" in got:
        print(_fmt_e3_delta(lib, got["random"], got["temporal"]), flush=True)

    if got:
        save_experiment_results("fault_attr", lib, got)

    return got

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Tabular Model Evaluation Harness (E1/E2)"
    )
    parser.add_argument(
        "experiment",
        type=str,
        choices=["e1", "e2", "e3"],
        help="Experiment identifier: 'e1' (job failure classification) or 'e2' (wait time regression)",
    )
    parser.add_argument(
        "model",
        type=str,
        help="Model identifier (e.g., ft, saint, tsmixer, xgb, lgb, cat, mlp, tabnet)",
    )
    parser.add_argument(
        "split",
        type=str,
        nargs="?",
        default="both",
        choices=["random", "temporal", "both"],
        help="Split protocol to run: 'random', 'temporal', or 'both' (default: both)",
    )
    args = parser.parse_args()

    SAVE_DIR = "/mnt/scratch/fast0/amaustin/datasets/fife/"
    targets = np.load(os.path.join(SAVE_DIR, "targets_and_masks.npz"))

    # Queue-start time: the key the temporal split is cut on, and so the ordering
    # used to hold out the most recent slice of training for threshold selection.
    QS = targets["qs"]

    Xmatch = np.load(os.path.join(SAVE_DIR, "Xmatch.npy"), mmap_mode="r")
    Xsub = np.load(os.path.join(SAVE_DIR, "Xsub.npy"), mmap_mode="r")

    failed = np.load(os.path.join(SAVE_DIR, "failed.npy"), mmap_mode="r")
    hw = np.load(os.path.join(SAVE_DIR, "hw.npy"), mmap_mode="r")
    wait_sv = np.load(os.path.join(SAVE_DIR, "wait_sv.npy"), mmap_mode="r")
    tr_mask = np.load(os.path.join(SAVE_DIR, "tr_mask.npy"), mmap_mode="r")
    te_mask = np.load(os.path.join(SAVE_DIR, "te_mask.npy"), mmap_mode="r")

    # Compute log-transformed target for wait time regression
    wait_log = np.log1p(np.maximum(wait_sv, 0))

    NCAT_MATCH = globals().get("NCAT_MATCH", None)
    NCAT_SUB = globals().get("NCAT_SUB", None)

    print(f"Loaded Xmatch {Xmatch.shape} and Xsub {Xsub.shape}")
    print(
        f"Train split: {tr_mask.sum():,} rows | Test split: {te_mask.sum():,} rows"
    )

    # Index setup
    idx = np.arange(len(tr_mask))

    # 1. Temporal split indices
    tri_t = np.where(tr_mask)[0]
    tei_t = np.where(te_mask)[0]

    # 2. Random split indices
    rng = np.random.default_rng(0)
    perm = rng.permutation(idx)
    rte = np.sort(perm[: len(tei_t)])
    rtr = np.sort(perm[len(tei_t) :])

    SPLITS = []
    if args.split in ("random", "both"):
        SPLITS.append(("random", rtr, rte))
    if args.split in ("temporal", "both"):
        SPLITS.append(("temporal", tri_t, tei_t))

    IMP = {}
    print(
        f"Splits prepared: Random ({len(rtr):,} train / {len(rte):,} test) | "
        f"Temporal ({len(tri_t):,} train / {len(tei_t):,} test)"
    )

    if args.experiment == "e1":
        print(f"Running Experiment E1 (Job Failure Classification) [{args.model}]")
        CM = {}
        run_e1_model(
            args.model, Xmatch, failed, SPLITS, IMP, CM, NCAT_MATCH, order=QS
        )
    elif args.experiment == "e2":
        print(f"Running Experiment E2 (Wait Time Regression) [{args.model}]")
        run_e2_model(
            args.model, Xsub, wait_log, SPLITS, IMP, NCAT_SUB
        )

    elif args.experiment == "e3":
        print(f"Running Experiment E3 (Fault Attribution: Hardware vs. Payload) [{args.model}]")
        CM = {}
        run_e3_model(
            args.model, Xmatch, failed, hw, SPLITS, IMP, CM, NCAT_MATCH, order=QS
        )
