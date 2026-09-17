"""Filesystem layout for datasets, saved models and saved predictions.

Every path in the project resolves through here. Each root takes an environment
override so a run can be pointed elsewhere without editing code:

    FIFE_DATA_ROOT   feature matrices and targets
    FIFE_MODEL_ROOT  saved models
    FIFE_PRED_ROOT   saved test predictions

Everything lives on the fast local NVMe: training mmaps tens of GB out of it
repeatedly, and /media/storage0 is NFS.

Model layout is one directory per experiment and model:

    models/e1/xgboost/xgboost_bin_temporal.json
    models/e2/lightgbm/lightgbm_reg_random.txt
    models/e3/saint/saint_bin_temporal.pt

A multi-seed run writes the same path each time, so the file left on disk is the
LAST seed. That is deliberate: per-seed metrics are what the variance report needs
and they are kept in the results JSON under `<split>__seed<n>` keys, while keeping
one model per seed would multiply the on-disk footprint for artifacts nothing reads.
"""

import os

DATA_ROOT = os.environ.get("FIFE_DATA_ROOT", "/mnt/scratch/fast0/amaustin/datasets/fife")
MODEL_ROOT = os.environ.get("FIFE_MODEL_ROOT", "/mnt/scratch/fast0/amaustin/models")
PRED_ROOT = os.environ.get("FIFE_PRED_ROOT", "/mnt/scratch/fast0/amaustin/predictions")

# exp_tag values used by the harness -> the directory they belong in.
_EXP_ALIASES = {"e3_fault": "e3", "e2_dist": "e2dist"}

def experiment_dir_name(exp_tag):
    if not exp_tag:
        return "misc"
    return _EXP_ALIASES.get(str(exp_tag), str(exp_tag))


def model_dir(exp_tag, lib, create=True):
    """<MODEL_ROOT>/<experiment>/<model>/"""
    d = os.path.join(MODEL_ROOT, experiment_dir_name(exp_tag), str(lib))
    if create:
        os.makedirs(d, exist_ok=True)
    return d


def model_path(exp_tag, lib, kind, split, ext="", create=True):
    """Full path for one saved model. `ext` includes the dot, or is empty for
    libraries that append their own (TabNet). Not seed-qualified: a multi-seed run
    overwrites, leaving the last seed's model."""
    return os.path.join(model_dir(exp_tag, lib, create=create),
                        f"{lib}_{kind}_{split}{ext}")


def all_model_dirs():
    """Every per-model directory under MODEL_ROOT, for post-hoc loaders that scan
    for saved models. Includes MODEL_ROOT itself so flat legacy layouts still load."""
    out = [MODEL_ROOT]
    if not os.path.isdir(MODEL_ROOT):
        return out
    for exp in sorted(os.listdir(MODEL_ROOT)):
        p = os.path.join(MODEL_ROOT, exp)
        if not os.path.isdir(p):
            continue
        out.append(p)
        out.extend(os.path.join(p, m) for m in sorted(os.listdir(p))
                   if os.path.isdir(os.path.join(p, m)))
    return out


def data_path(name):
    return os.path.join(DATA_ROOT, name)
