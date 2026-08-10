import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap


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
):
  if entity_col_name not in xmatch_cols:
    return None

  if label_prefix is None:
    label_prefix = entity_col_name

  col_idx = xmatch_cols.index(entity_col_name)

  # Combine train/test mask and date range mask
  if split_mask is not None and time_mask is not None:
    active_mask = split_mask & time_mask
  elif split_mask is not None:
    active_mask = split_mask
  elif time_mask is not None:
    active_mask = time_mask
  else:
    active_mask = None

  if active_mask is not None:
    entity_codes = Xmatch[active_mask, col_idx]
    fail_vec = failed[active_mask]
    hw_vec = hw[active_mask]
    Xm_sub = Xmatch[active_mask]
  else:
    entity_codes = Xmatch[:, col_idx]
    fail_vec = failed
    hw_vec = hw
    Xm_sub = Xmatch

  # Filter out unassigned entity codes (< 0 or NaN)
  valid_mask = ~np.isnan(entity_codes) & (entity_codes >= 0)
  if not np.any(valid_mask):
    return None

  entity_codes = entity_codes[valid_mask].astype(np.int64)
  fail_vec = np.asarray(fail_vec[valid_mask]).astype(np.float64)
  hw_vec = np.asarray(hw_vec[valid_mask]).astype(np.float64)
  Xm_sub = Xm_sub[valid_mask]

  # Unique entity accumulation
  unique_entities, inv_indices, counts = np.unique(
      entity_codes, return_inverse=True, return_counts=True
  )
  n_entities = len(unique_entities)

  if n_entities == 0:
    return None

  n_features = Xm_sub.shape[1]
  feat_sums = np.zeros((n_entities, n_features), dtype=np.float64)
  fail_sums = np.zeros(n_entities, dtype=np.float64)
  hw_sums = np.zeros(n_entities, dtype=np.float64)

  np.add.at(fail_sums, inv_indices, fail_vec)
  np.add.at(hw_sums, inv_indices, hw_vec)

  for j in range(n_features):
    np.add.at(feat_sums[:, j], inv_indices, Xm_sub[:, j])

  feat_means = feat_sums / counts[:, None]
  fail_rates = fail_sums / counts
  hw_rates = hw_sums / counts

  df_raw = (
      pd.DataFrame({
          "entity_code": unique_entities,
          "jobs": counts,
          "failure_rate": fail_rates,
          "hw_rate": hw_rates,
      })
      .sort_values(by="jobs", ascending=False)
      .reset_index(drop=True)
  )

  df_raw["Entity_Anon"] = [
      f"{label_prefix} {i+1:02d}" for i in range(len(df_raw))
  ]
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
      perplexity = min(30, max(2, n_samples - 1))
      reducer = TSNE(
          n_components=2, perplexity=perplexity, random_state=42, init="pca"
      )
      embedding = reducer.fit_transform(X_scaled)
    elif dr_method == "UMAP":
      n_neighbors = min(15, max(2, n_samples - 1))
      reducer = umap.UMAP(
          n_components=2, n_neighbors=n_neighbors, random_state=42
      )
      embedding = reducer.fit_transform(X_scaled)
    else:
      reducer = PCA(n_components=2, random_state=42)
      embedding = reducer.fit_transform(X_scaled)

    df_raw["DR1"] = embedding[:, 0]
    df_raw["DR2"] = embedding[:, 1]

  return df_raw