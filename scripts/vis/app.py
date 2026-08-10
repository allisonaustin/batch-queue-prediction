"""Interactive DR data explorer widget.

Builds the date-range / entity / DR-method dashboard on top of
`vis.dr.prepare_entity_data`. Operates directly on the `Xmatch` feature
matrix reloaded from the saved training-data arrays (see
`data-explorer.ipynb`), so it needs the job-start timestamps (`jst` in
`targets_and_masks.npz`) passed in separately since that column isn't
one of the `XMATCH_COLS` feature columns.

Defaults (date window, DR method, entity, point cap) live in `config.yaml`;
the feature columns used for DR + the importance panel live in
`features.yaml`. Both are re-read on every `build_explorer()` call, so
editing them takes effect on the next cell run -- no code change needed.
"""

import os
from datetime import date, datetime

import ipywidgets as widgets
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml

import vis.dr as dr

_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_ENTITY_COL_MAP = {"Users": "Owner", "Groups": "Group", "Sites": "MatchSite", "Campaigns": "CampaignId"}

# Payload (blue) <-> Hardware (red) diverging scale for hw_share, with a mid
# gray instead of RdBu_r's near-white midpoint -- against a dark plot
# background near-white marker fills were nearly invisible.
FAULT_COLORSCALE = [[0.0, "#2b7fd6"], [0.5, "#8a8a8a"], [1.0, "#e0483e"]]
FAULT_GRADIENT_CSS = "linear-gradient(to right, #2b7fd6, #8a8a8a, #e0483e)"

def _load_yaml(env_var, default_filename):
    path = os.environ.get(env_var, os.path.join(_DIR, default_filename))
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


def _as_date(v):
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    return date.fromisoformat(v)


def load_config():
    cfg = _load_yaml("FIFE_VIS_CONFIG", "config.yaml")
    defaults = {**(cfg.get("defaults") or {})}
    return {
        "defaults": defaults,
        "max_points": cfg.get("max_points", 5000)
    }


def load_feature_list():
    cfg = _load_yaml("FIFE_VIS_FEATURES", "features.yaml")
    return list(cfg.get("features") or [])


def _legend_html(v_small=None, v_med=None, v_large=None):
    job_rows = []
    if v_small is not None:
        job_rows.append(f"""
            <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 4px;">
                <span style="display: inline-block; width: 6px; height: 6px; background-color: #ccc; border-radius: 50%;"></span>
                <span style="color: #ccc;">{v_small:,}</span>
            </div>""")
    if v_med is not None:
        job_rows.append(f"""
            <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 4px;">
                <span style="display: inline-block; width: 10px; height: 10px; background-color: #ccc; border-radius: 50%;"></span>
                <span style="color: #ccc;">{v_med:,}</span>
            </div>""")
    if v_large is not None:
        job_rows.append(f"""
            <div style="display: flex; align-items: center; gap: 6px;">
                <span style="display: inline-block; width: 14px; height: 14px; background-color: #ccc; border-radius: 50%;"></span>
                <span style="color: #ccc;">{v_large:,}</span>
            </div>""")

    job_volume_html = (
        "".join(job_rows)
        if job_rows
        else '<div style="color: #888;">--</div>'
    )

    return f"""
    <div style="font-family: sans-serif; font-size: 11px; color: #ddd; margin-top: 3px; padding: 8px; border: 1px solid #444; border-radius: 6px; background-color: #1e1e1e; width: 230px; box-sizing: border-box;">
        <div style="font-weight: bold; margin-bottom: 4px; text-align: center; color: #eee;">Fault Attribution</div>
        <div style="height: 10px; background: {FAULT_GRADIENT_CSS}; border-radius: 2px;"></div>
        <div style="display: flex; justify-content: space-between; font-size: 10px; color: #aaa; margin-top: 2px; margin-bottom: 12px;">
            <span>Payload</span>
            <span style="text-align: right;">Hardware</span>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div>
                <div style="font-weight: bold; margin-bottom: 6px; color: #eee;">Job Volume</div>
                {job_volume_html}
            </div>
            <div style="margin-right: 10px;">
                <div style="font-weight: bold; margin-bottom: 6px; color: #eee;">Failure Rate</div>
                <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 4px;">
                    <span style="display: inline-block; width: 10px; height: 10px; background-color: #ccc; opacity: 0.35; border-radius: 50%;"></span>
                    <span style="color: #ccc;">Low</span>
                </div>
                <div style="display: flex; align-items: center; gap: 6px;">
                    <span style="display: inline-block; width: 10px; height: 10px; background-color: #ccc; opacity: 1.0; border-radius: 50%;"></span>
                    <span style="color: #ccc;">High</span>
                </div>
            </div>
        </div>
    </div>
    """


def _fig_layout(dr_method, selected_entity, n=None):
    title_text = f"{dr_method} Embedding - {selected_entity}"
    if n is not None:
        title_text += f"  (n={n:,})"
    return dict(
        title=dict(
            text=title_text,
            x=0.0,
            xanchor="left",
            font=dict(size=14, color="#e8e8e8"),
        ),
        xaxis=dict(
            title=f"{dr_method} 1",
            gridcolor="#3a3a3a",
            zeroline=False,
            showline=True,
            linecolor="#5a5a5a",
        ),
        yaxis=dict(
            title=f"{dr_method} 2",
            gridcolor="#3a3a3a",
            zeroline=False,
            showline=True,
            linecolor="#5a5a5a",
        ),
        template="plotly_dark",
        paper_bgcolor="#1e1e1e",
        plot_bgcolor="#1e1e1e",
        width=750,
        height=620,
        margin=dict(l=40, r=20, t=40, b=40),
    )


def _importance_html(df):
  names = df.attrs.get("feature_names", []) if df is not None else []
  imp = (df.attrs.get("feature_importance", []) if df is not None else [])

  if not imp:
    rows_html = (
        '<div style="color:#999;">No features configured -- edit'
        " vis/features.yaml.</div>"
    )
  else:
    max_score = max((s for _, s in imp), default=0.0) or 1.0
    rows_html = "".join(
        f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:3px;">'
        f'<span style="width:120px;font-size:10px;color:#ccc;white-space:nowrap;'
        f'overflow:hidden;text-overflow:ellipsis;" title="{name}">{name}</span>'
        f'<div style="flex:1;background:#3a3a3a;border-radius:2px;height:8px;overflow:hidden;">'
        f'<div style="width:{100 * score / max_score:.0f}%;background:#4da3ff;height:100%;"></div>'
        f"</div>"
        f'<span style="width:28px;text-align:right;font-size:10px;color:#aaa;">{score:.2f}</span>'
        f"</div>"
        for name, score in imp
    )

  return f"""
    <div style="font-family: sans-serif; font-size: 11px; color: #ddd; margin-top: 3px; padding: 8px; border: 1px solid #444; border-radius: 6px; background-color: #1e1e1e; width: 230px; box-sizing: border-box;">
        <div style="font-weight: bold; margin-bottom: 2px; color: #eee;">Features used ({len(names)})</div>
        <div style="color: #999; margin-bottom: 6px;">abs(corr) with failure rate</div>
        <div style="max-height: 180px; overflow-y: auto; padding-right: 4px;">
            {rows_html}
        </div>
    </div>
    """


def build_explorer(
    Xmatch,
    XMATCH_COLS,
    failed,
    hw,
    job_starts,
    split_mask=None,
    entity_col_map=None,
    cluster_ids=None,
):
    """Assemble the interactive DR dashboard.

    `job_starts` must be the per-row job-start epoch-seconds array
    (`targets_and_masks.npz["jst"]`), aligned to `Xmatch`'s rows -- it's
    not part of `XMATCH_COLS` so it has to come in separately.
    `split_mask` (e.g. `tr_mask`/`te_mask`) is optional and ANDed with
    the date-range mask.
    `cluster_ids`: optional per-row `ClusterId` array (also not part of
    `XMATCH_COLS`). If given, a "Clusters" entity option is added; if not,
    it's silently omitted from the dropdown rather than erroring, since
    `ClusterId` isn't currently persisted alongside Xmatch/Xsub.

    Returns the ipywidgets layout; call `display()` on the result.
    """
    cfg = load_config()
    feature_cols = load_feature_list()
    max_points = cfg["max_points"]
    d = cfg["defaults"]

    entity_col_map = dict(entity_col_map or BASE_ENTITY_COL_MAP)
    if cluster_ids is not None:
        entity_col_map["Clusters"] = "ClusterId"
    extra_entity_cols = {"ClusterId": np.asarray(cluster_ids)} if cluster_ids is not None else {}

    job_starts = np.asarray(job_starts)

    start_date_w = widgets.DatePicker(
        value=_as_date(d["start_date"]),
        description="Start Date",
        style={"description_width": "75px"},
        layout=widgets.Layout(width="210px", margin="1px 0px"),
    )
    end_date_w = widgets.DatePicker(
        value=_as_date(d["end_date"]),
        description="End Date",
        style={"description_width": "75px"},
        layout=widgets.Layout(width="210px", margin="1px 0px"),
    )
    dr_method_w = widgets.Dropdown(
        options=["PCA", "t-SNE", "UMAP"],
        value=d["dr_method"],
        description="DR Method",
        style={"description_width": "75px"},
        layout=widgets.Layout(width="210px", margin="1px 0px"),
    )
    n_neighbors_w = widgets.IntSlider(
        value=d.get("n_neighbors", 15),
        min=2,
        max=100,
        description="n_neighbors",
        style={"description_width": "75px"},
        layout=widgets.Layout(width="240px", margin="1px 0px"),
    )
    perplexity_w = widgets.IntSlider(
        value=d.get("perplexity", 30),
        min=2,
        max=100,
        description="perplexity",
        style={"description_width": "75px"},
        layout=widgets.Layout(width="240px", margin="1px 0px"),
    )

    def _sync_param_visibility(change=None):
        n_neighbors_w.layout.display = "" if dr_method_w.value == "UMAP" else "none"
        perplexity_w.layout.display = "" if dr_method_w.value == "t-SNE" else "none"

    _sync_param_visibility()
    dr_method_w.observe(_sync_param_visibility, names="value")

    entity_w = widgets.Dropdown(
        options=list(entity_col_map.keys()),
        value=d["entity"] if d["entity"] in entity_col_map else next(iter(entity_col_map)),
        description="Entity",
        style={"description_width": "75px"},
        layout=widgets.Layout(width="210px", margin="1px 0px"),
    )

    legend_w = widgets.HTML(value=_legend_html(), layout=widgets.Layout(margin="1px 0px"))
    features_w = widgets.HTML(value=_importance_html(None), layout=widgets.Layout(margin="1px 0px"))
    out = widgets.Output()
    state = {"fig": None}

    export_btn = widgets.Button(
        description="Export",
        icon="camera",
        button_style="primary",
        layout=widgets.Layout(width="100px"),
    )

    def export_screenshot(b):
        if state["fig"] is not None:
            filename = f"dr_embedding_{entity_w.value}_{start_date_w.value}_to_{end_date_w.value}.png"
            try:
                # Requires kaleido package (`pip install kaleido`)
                state["fig"].write_image(filename, scale=2)
                with out:
                    print(f"Screenshot successfully saved to '{filename}'!")
            except Exception as e:
                with out:
                    print(
                        f"Export error: {e}\n(Make sure `kaleido` is installed: `pip install kaleido`)"
                    )

    export_btn.on_click(export_screenshot)

    def update_dashboard(change=None):
        start_date = start_date_w.value
        end_date = end_date_w.value
        dr_method = dr_method_w.value
        selected_entity = entity_w.value

        fig = go.Figure()

        def show_message(text):
            fig.add_annotation(
                text=text,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=14, color="#999999"),
            )
            fig.update_layout(**_fig_layout(dr_method, selected_entity))
            legend_w.value = _legend_html()
            features_w.value = _importance_html(None)
            with out:
                out.clear_output(wait=True)
                fig.show(config={"responsive": True})

        if not start_date or not end_date or start_date > end_date:
            show_message("Please select a valid date range.")
            return

        # Job Start Date filter, ANDed with the optional train/test split mask.
        # `end_date` is inclusive, so the window extends through end-of-day.
        start_ts = pd.Timestamp(start_date).timestamp()
        end_ts = pd.Timestamp(end_date).timestamp() + 86400
        time_mask = (job_starts >= start_ts) & (job_starts < end_ts)

        col_name = entity_col_map[selected_entity]
        label_prefix = selected_entity[:-1] if selected_entity.endswith("s") else selected_entity
        df = dr.prepare_entity_data(
            Xmatch=Xmatch,
            failed=failed,
            hw=hw,
            xmatch_cols=XMATCH_COLS,
            entity_col_name=col_name,
            label_prefix=label_prefix,
            dr_method=dr_method,
            split_mask=split_mask,
            time_mask=time_mask,
            feature_cols=feature_cols,
            extra_entity_cols=extra_entity_cols,
            max_points=max_points,
            n_neighbors=n_neighbors_w.value if dr_method == "UMAP" else None,
            perplexity=perplexity_w.value if dr_method == "t-SNE" else None,
        )

        if df is None or df.empty:
            show_message(f"No job data found for {selected_entity} in selected date range.")
            return

        max_jobs = df["jobs"].max()
        sizeref = 2.0 * max_jobs / (32**2)
        if max_jobs <= 2:
            sizeref = 1.0

        fig.add_trace(
            go.Scatter(
                x=df["DR1"],
                y=df["DR2"],
                mode="markers",
                name=selected_entity,
                showlegend=False,
                customdata=np.stack(
                    (
                        df["Entity_Anon"],
                        df["jobs"],
                        df["failure_rate_pct"],
                        df["hw_share"] * 100.0,
                        df["payload_share"] * 100.0,
                    ),
                    axis=-1,
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Total Jobs: %{customdata[1]:,}<br>"
                    "Overall Failure Rate: %{customdata[2]:.2f}%<br>"
                    "Hardware Faults: %{customdata[3]:.1f}% of failures<br>"
                    "Payload Faults: %{customdata[4]:.1f}% of failures<br>"
                    "<extra></extra>"
                ),
                marker=dict(
                    size=df["jobs"],
                    sizemode="area",
                    sizeref=sizeref,
                    sizemin=5,
                    color=df["hw_share"],
                    colorscale=FAULT_COLORSCALE,
                    cmin=0.0,
                    cmax=1.0,
                    opacity=df["opacity"].tolist(),
                    line=dict(width=0.8, color="rgba(220,220,220,0.6)"),
                ),
            )
        )

        fig.update_layout(**_fig_layout(dr_method, selected_entity, n=len(df)))

        # Job Volume legend reports the actual min/median/max job count
        # across the currently-plotted entities (not a synthetic 10%/50% of
        # the max), so it reflects the real data range.
        min_jobs = int(df["jobs"].min())
        med_jobs = int(df["jobs"].median())
        max_jobs_i = int(max_jobs)
        if min_jobs == max_jobs_i:
            v_small, v_med, v_large = None, None, max_jobs_i
        elif min_jobs == med_jobs or med_jobs == max_jobs_i:
            v_small, v_med, v_large = min_jobs, None, max_jobs_i
        else:
            v_small, v_med, v_large = min_jobs, med_jobs, max_jobs_i

        legend_w.value = _legend_html(v_small, v_med, v_large)
        features_w.value = _importance_html(df)

        state["fig"] = fig

        with out:
            out.clear_output(wait=True)
            fig.show(
                config={
                    "responsive": True,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": f"dr_embedding_{selected_entity}",
                        "height": 620,
                        "width": 750,
                        "scale": 2,
                    },
                    "displayModeBar": True,
                }
            )

    start_date_w.observe(update_dashboard, names="value")
    end_date_w.observe(update_dashboard, names="value")
    dr_method_w.observe(update_dashboard, names="value")
    entity_w.observe(update_dashboard, names="value")
    n_neighbors_w.observe(update_dashboard, names="value")
    perplexity_w.observe(update_dashboard, names="value")

    controls_box = widgets.VBox(
        [
            start_date_w, end_date_w, dr_method_w, n_neighbors_w, perplexity_w,
            entity_w, legend_w, features_w,
        ],
        layout=widgets.Layout(margin="0px 0px 0px 15px", width="340px"),
    )

    # Export button anchored top-right of the plot itself (not the whole
    # dashboard row), so it sits next to the plot's own toolbar.
    plot_header = widgets.HBox(
        [export_btn],
        layout=widgets.Layout(width="750px", justify_content="flex-end", margin="0px 0px 4px 0px"),
    )
    plot_area = widgets.VBox([plot_header, out])

    app_layout = widgets.HBox(
        [plot_area, controls_box], layout=widgets.Layout(align_items="flex-start")
    )

    update_dashboard()
    return app_layout
