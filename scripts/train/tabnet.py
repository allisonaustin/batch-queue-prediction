import os
import gc
import numpy as np
import torch
from eval.helper import _empty_gpu, _get_slice
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

model_dir = "/mnt/scratch/fast0/amaustin/dl-tabular-models"
os.makedirs(model_dir, exist_ok=True)


def tabnet_fit_eval(
    parts, tri, tei, ncat, kind, y, spw=None, trs=None, want_imp=False, split=None, exp_tag=None
):
    split = split if split is not None else "default"
    DEV = "cuda" if torch.cuda.is_available() else "cpu"

    pin_mem = DEV != "cpu"
    ya = np.asarray(y)

    print(f"device={DEV} | split={split} | kind={kind}", flush=True)

    Xtr = _get_slice(parts, tri)
    X_te = _get_slice(parts, tei)
    X_trs = _get_slice(parts, trs) if trs is not None else None

    Xtr_np = np.asarray(Xtr, dtype=np.float32)
    X_te_np = np.asarray(X_te, dtype=np.float32)
    X_trs_np = np.asarray(X_trs, dtype=np.float32) if X_trs is not None else None

    cat_idxs, cat_dims = [], []
    if isinstance(ncat, int) and ncat > 0:
        cat_idxs = list(range(ncat))
        cat_dims = [
            int(max(Xtr_np[:, i].max(), X_te_np[:, i].max()) + 1)
            for i in cat_idxs
        ]

    eval_size = min(50000, len(X_te_np))
    eval_idx = np.random.choice(len(X_te_np), size=eval_size, replace=False)
    X_eval_sub = X_te_np[eval_idx]
    y_eval_sub = ya[tei][eval_idx]

    y_tr = ya[tri]

    # TabNetRegressor requires 2D targets shape (N, 1)
    if kind == "reg":
        y_tr = y_tr.reshape(-1, 1)
        y_eval_sub = y_eval_sub.reshape(-1, 1)

    tabnet_params = dict(
        device_name=DEV,
        n_d=8,
        n_a=8,
        n_steps=3,
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size": 5, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type="sparsemax",
        verbose=1,
    )

    if kind == "reg":
        clf = TabNetRegressor(**tabnet_params)
        eval_metric = ["mse"]
    else:
        clf = TabNetClassifier(**tabnet_params)
        eval_metric = ["auc"]

    clf.fit(
        X_train=Xtr_np,
        y_train=y_tr,
        eval_set=[(X_eval_sub, y_eval_sub)],
        eval_name=["test"],
        eval_metric=eval_metric,
        max_epochs=10,
        patience=3,
        batch_size=16384,
        virtual_batch_size=2048,
        num_workers=0,
        pin_memory=pin_mem,
        drop_last=False,
        compute_importance=False,
    )

    save_path = os.path.join(model_dir, f"tabnet_{kind}_{exp_tag}_{split}")
    clf.save_model(save_path)
    print(f"--> Generating predictions for test splits... | split={split}", flush=True)

    if kind == "reg":
        p_te = clf.predict(X_te_np).reshape(-1)
        p_trs = (
            clf.predict(X_trs_np).reshape(-1) if X_trs_np is not None else None
        )
    else:
        p_te = clf.predict_proba(X_te_np)[:, 1]
        p_trs = (
            clf.predict_proba(X_trs_np)[:, 1] if X_trs_np is not None else None
        )

    imp = getattr(clf, "feature_importances_", None) if want_imp else None

    del clf, Xtr, X_te, X_trs, Xtr_np, X_te_np, X_trs_np, X_eval_sub
    _empty_gpu()
    gc.collect()

    return p_te, p_trs, imp