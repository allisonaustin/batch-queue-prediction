import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap

# Sentinel entity_col_name meaning "no aggregation -- every job row is its
# own point" (the "Jobs" entity in the explorer).
RAW_JOB_ENTITY = "__RAW_JOB__"


def _grouped_sums(inv_indices, n_entities, values):
    """Sum `values` (n_rows, n_features) into n_entities groups via a sparse
    indicator matmul. Equivalent to but much faster than looping
    `np.add.at` per column -- O(nnz) sparse matmul vs. a Python-level loop
    over every feature column.
    """
    n_rows = len(inv_indices)
    indicator = sp.csr_matrix(
        (np.ones(n_rows), (inv_indices, np.arange(n_rows))),
        shape=(n_entities, n_rows),
    )
    return np.asarray(indicator @ values)


def _feature_importance(feat_means, fail_rates, feature_names):
    """Cheap, method-agnostic importance: |corr| of each feature's
    per-entity mean with the entity's failure rate. Computed on the small
    (n_entities x n_features) table, so it's essentially free regardless of
    how many raw rows fed into it.
    """
    out = []
    with np.errstate(invalid="ignore", divide="ignore"):
        for j, name in enumerate(feature_names):
            col = feat_means[:, j]
            if np.nanstd(col) == 0 or np.nanstd(fail_rates) == 0:
                score = 0.0
            else:
                c = np.corrcoef(col, fail_rates)[0, 1]
                score = 0.0 if np.isnan(c) else abs(float(c))
            out.append((name, score))
    out.sort(key=lambda t: t[1], reverse=True)
    return out


def prepare_entity_data(
    Xmatch,
    failed,
    hw,
    xmatch_cols,
    entity_col_name,
    label_prefix=None,
    dr_method="PCA",
    split_mask=None,
    time_mask=None,
    feature_cols=None,
    extra_entity_cols=None,
    max_points=None,
    random_state=42,
    n_neighbors=None,
    perplexity=None,
):
    """Aggregate jobs into DR-embedded entities (or, for `RAW_JOB_ENTITY`,
    treat each job row as its own point).

    `feature_cols`: names from `xmatch_cols` to use for the DR/importance
    computation (defaults to all of them). Lets the caller restrict to a
    user-editable feature list instead of averaging every column, including
    the categorical code columns, which is meaningless for those.

    `extra_entity_cols`: {name: row-aligned array} for entity groupings not
    present in `xmatch_cols` (e.g. `ClusterId`), keyed and looked up the same
    way as an `xmatch_cols` column.

    `max_points`: only enforced for `RAW_JOB_ENTITY` -- caps how many job
    rows are randomly sampled before DR runs, since there's no aggregation
    to bound the point count otherwise. Aggregated entities (Owner/Group/
    Site/ClusterId/...) are already cheap regardless of row count.

    `n_neighbors` (UMAP) / `perplexity` (t-SNE): user-adjustable DR
    hyperparameters. Each is still clamped to `n_samples - 1` internally,
    so an oversized value degrades gracefully instead of erroring on a
    small entity count.
    """
    extra_entity_cols = extra_entity_cols or {}
    raw_mode = entity_col_name == RAW_JOB_ENTITY

    if raw_mode:
        col_idx = None
    elif entity_col_name in extra_entity_cols:
        col_idx = None
    elif entity_col_name not in xmatch_cols:
        return None
    else:
        col_idx = xmatch_cols.index(entity_col_name)

    if label_prefix is None:
        label_prefix = "Job" if raw_mode else entity_col_name

    # Exclude the entity's own grouping column from the feature set: e.g. for
    # entity_col_name="Owner" (Users), every row in a group shares the same
    # Owner value, so its per-entity mean is just that entity's own ID with
    # zero within-group variance -- a degenerate "feature" that can swamp
    # real signal once standardized alongside everything else.
    if feature_cols:
        feat_idx = [
            xmatch_cols.index(c)
            for c in feature_cols
            if c in xmatch_cols and c != entity_col_name
        ]
        feat_names = [xmatch_cols[j] for j in feat_idx]
    else:
        feat_idx = [j for j, c in enumerate(xmatch_cols) if c != entity_col_name]
        feat_names = [xmatch_cols[j] for j in feat_idx]
    if not feat_idx:
        return None

    # Combine train/test mask and date range mask (kept as row indices from
    # here on, so raw-mode subsampling below is cheap)
    if split_mask is not None and time_mask is not None:
        active_mask = split_mask & time_mask
    elif split_mask is not None:
        active_mask = split_mask
    elif time_mask is not None:
        active_mask = time_mask
    else:
        active_mask = None

    idx_pool = np.where(active_mask)[0] if active_mask is not None else np.arange(Xmatch.shape[0])
    if len(idx_pool) == 0:
        return None

    if raw_mode and max_points and len(idx_pool) > max_points:
        rng = np.random.default_rng(random_state)
        idx_pool = np.sort(rng.choice(idx_pool, size=max_points, replace=False))

    if raw_mode:
        entity_codes = idx_pool.astype(np.int64)
    elif entity_col_name in extra_entity_cols:
        entity_codes = np.asarray(extra_entity_cols[entity_col_name])[idx_pool]
    else:
        entity_codes = Xmatch[idx_pool, col_idx]

    fail_vec = np.asarray(failed[idx_pool]).astype(np.float64)
    hw_vec = np.asarray(hw[idx_pool]).astype(np.float64)
    Xm_sub = np.asarray(Xmatch[np.ix_(idx_pool, feat_idx)])

    # Filter out unassigned entity codes (< 0 or NaN)
    entity_codes = np.asarray(entity_codes, dtype=np.float64)
    valid_mask = ~np.isnan(entity_codes) & (entity_codes >= 0)
    if not np.any(valid_mask):
        return None

    entity_codes = entity_codes[valid_mask].astype(np.int64)
    fail_vec = fail_vec[valid_mask]
    hw_vec = hw_vec[valid_mask]
    Xm_sub = Xm_sub[valid_mask]

    # Unique entity accumulation
    unique_entities, inv_indices, counts = np.unique(
        entity_codes, return_inverse=True, return_counts=True
    )
    n_entities = len(unique_entities)

    if n_entities == 0:
        return None

    feat_sums = _grouped_sums(inv_indices, n_entities, Xm_sub.astype(np.float64))
    fail_sums = np.zeros(n_entities, dtype=np.float64)
    hw_sums = np.zeros(n_entities, dtype=np.float64)

    np.add.at(fail_sums, inv_indices, fail_vec)
    np.add.at(hw_sums, inv_indices, hw_vec)

    feat_means = feat_sums / counts[:, None]
    fail_rates = fail_sums / counts
    hw_rates = hw_sums / counts

    # Sort by job count (descending) once, up front, and apply the same
    # permutation to everything -- computing this order separately from a
    # pandas .sort_values() call risks desyncing feat_means from df_raw on
    # tied counts (pandas' sort isn't guaranteed to match np.argsort's).
    order = np.argsort(-counts, kind="stable")
    unique_entities, counts = unique_entities[order], counts[order]
    fail_rates, hw_rates = fail_rates[order], hw_rates[order]
    feat_means = feat_means[order]

    df_raw = pd.DataFrame({
        "entity_code": unique_entities,
        "jobs": counts,
        "failure_rate": fail_rates,
        "hw_rate": hw_rates,
    })

    df_raw["Entity_Anon"] = (
        [f"{label_prefix} {code}" for code in df_raw["entity_code"]]
        if raw_mode
        else [f"{label_prefix} {i+1:02d}" for i in range(len(df_raw))]
    )
    df_raw["failure_rate_pct"] = df_raw["failure_rate"] * 100.0
    df_raw["hw_rate_pct"] = df_raw["hw_rate"] * 100.0
    df_raw["payload_rate_pct"] = np.maximum(
        0.0, df_raw["failure_rate_pct"] - df_raw["hw_rate_pct"]
    )
    df_raw["hw_share"] = np.where(
        df_raw["failure_rate"] > 0,
        np.clip(df_raw["hw_rate"] / df_raw["failure_rate"], 0.0, 1.0),
        0.0,
    )
    # Share of *failures* (not all jobs) that were payload vs. hardware --
    # guarded the same way as hw_share so a zero-failure entity reports 0%
    # for both rather than payload_share defaulting to 100%.
    df_raw["payload_share"] = np.where(
        df_raw["failure_rate"] > 0, 1.0 - df_raw["hw_share"], 0.0
    )

    max_fail = max(df_raw["failure_rate"].max(), 1e-5)
    df_raw["opacity"] = (
        0.35 + 0.65 * (df_raw["failure_rate"] / max_fail)
    ).fillna(0.85)

    # DR Execution
    features = np.nan_to_num(feat_means, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    n_samples = len(df_raw)
    if n_samples < 3:
        df_raw["DR1"] = X_scaled[:, 0] if X_scaled.shape[1] > 0 else 0.0
        df_raw["DR2"] = X_scaled[:, 1] if X_scaled.shape[1] > 1 else 0.0
    else:
        if np.all(np.std(X_scaled, axis=0) == 0):
            X_scaled += np.random.normal(0, 1e-5, X_scaled.shape)

        if dr_method == "t-SNE":
            perplexity = min(perplexity or 30, max(2, n_samples - 1))
            reducer = TSNE(
                n_components=2, perplexity=perplexity, random_state=random_state,
                init="pca", n_jobs=-1,
            )
            embedding = reducer.fit_transform(X_scaled)
        elif dr_method == "UMAP":
            n_neighbors = min(n_neighbors or 15, max(2, n_samples - 1))
            # umap-learn pins n_jobs=1 whenever random_state is set (needed for
            # reproducibility), so there's no parallelism knob to turn here.
            # Per-job feature vectors (raw_mode) form a sparse, disconnected
            # neighbor graph, which makes UMAP's default spectral init fall
            # back to an expensive multi-component embedding (~4x slower at
            # 50k points, confirmed via benchmark). Aggregated entities have
            # far fewer/denser points and never hit this, so only raw_mode
            # needs the cheaper random init.
            reducer = umap.UMAP(
                n_components=2, n_neighbors=n_neighbors, random_state=random_state,
                init="random" if raw_mode else "spectral",
            )
            embedding = reducer.fit_transform(X_scaled)
        else:
            reducer = PCA(n_components=2, random_state=random_state)
            embedding = reducer.fit_transform(X_scaled)

        df_raw["DR1"] = embedding[:, 0]
        df_raw["DR2"] = embedding[:, 1]

    df_raw.attrs["feature_names"] = feat_names
    df_raw.attrs["feature_importance"] = _feature_importance(feat_means, fail_rates, feat_names)

    return df_raw
