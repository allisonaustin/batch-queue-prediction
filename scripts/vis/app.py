from datetime import date

import ipywidgets as widgets
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import vis.dr as dr

# Entity dropdown label -> column name in XMATCH_COLS
ENTITY_COL_MAP = {"Users": "Owner", "Groups": "Group", "Sites": "MatchSite", "Jobs": "JobId"}


def _legend_html(v_small=None, v_med=None, v_large=None):
    s_str = f"{v_small:,}" if v_small is not None else "--"
    m_str = f"{v_med:,}" if v_med is not None else "--"
    l_str = f"{v_large:,}" if v_large is not None else "--"

    return f"""
    <div style="font-family: sans-serif; font-size: 11px; color: #333; margin-top: 10px; padding: 10px; border: 1px solid #e0e0e0; border-radius: 6px; background-color: #fafafa; width: 230px; box-sizing: border-box;">

        <!-- Fault Attribution Colorbar -->
        <div style="font-weight: bold; margin-bottom: 4px; text-align: center;">Fault Attribution</div>
        <div style="height: 10px; width: 100%; background: linear-gradient(to right, #0571b0, #f7f7f7, #ca0020); border-radius: 2px;"></div>
        <div style="display: flex; justify-content: space-between; font-size: 10px; color: #666; margin-top: 2px; margin-bottom: 12px;">
            <span>Payload</span>
           <span style="text-align: right;">Hardware</span>
        </div>

        <!-- Job Volume & Failure Rate Side-by-Side -->
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <!-- Job Volume Column -->
            <div>
                <div style="font-weight: bold; margin-bottom: 6px;">Job Volume</div>
                <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 4px;">
                    <span style="display: inline-block; width: 6px; height: 6px; background-color: #666; border-radius: 50%;"></span>
                    <span style="color: #444;">{s_str}</span>
                </div>
                <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 4px;">
                    <span style="display: inline-block; width: 10px; height: 10px; background-color: #666; border-radius: 50%;"></span>
                    <span style="color: #444;">{m_str}</span>
                </div>
                <div style="display: flex; align-items: center; gap: 6px;">
                    <span style="display: inline-block; width: 14px; height: 14px; background-color: #666; border-radius: 50%;"></span>
                    <span style="color: #444;">{l_str}</span>
                </div>
            </div>

            <!-- Failure Rate Column -->
            <div style="margin-right: 10px;">
                <div style="font-weight: bold; margin-bottom: 6px;">Failure Rate</div>
                <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 4px;">
                    <span style="display: inline-block; width: 10px; height: 10px; background-color: #666; opacity: 0.35; border-radius: 50%;"></span>
                    <span style="color: #444;">Low</span>
                </div>
                <div style="display: flex; align-items: center; gap: 6px;">
                    <span style="display: inline-block; width: 10px; height: 10px; background-color: #666; opacity: 1.0; border-radius: 50%;"></span>
                    <span style="color: #444;">High</span>
                </div>
            </div>
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
):
    """Assemble the interactive DR dashboard.

    `job_starts` must be the per-row job-start epoch-seconds array
    (`targets_and_masks.npz["jst"]`), aligned to `Xmatch`'s rows -- it's
    not part of `XMATCH_COLS` so it has to come in separately.
    `split_mask` (e.g. `tr_mask`/`te_mask`) is optional and ANDed with
    the date-range mask.

    Returns the ipywidgets layout; call `display()` on the result.
    """
    entity_col_map = entity_col_map or ENTITY_COL_MAP
    job_starts = np.asarray(job_starts)

    start_date_w = widgets.DatePicker(
        value=date(2025, 2, 1),
        description="Start Date",
        style={"description_width": "75px"},
        layout=widgets.Layout(width="210px"),
    )
    end_date_w = widgets.DatePicker(
        value=date(2025, 2, 14),
        description="End Date",
        style={"description_width": "75px"},
        layout=widgets.Layout(width="210px"),
    )
    dr_method_w = widgets.Dropdown(
        options=["PCA", "t-SNE", "UMAP"],
        value="UMAP",
        description="DR Method",
        style={"description_width": "75px"},
        layout=widgets.Layout(width="210px"),
    )
    entity_w = widgets.Dropdown(
        options=list(entity_col_map.keys()),
        value=next(iter(entity_col_map)),
        description="Entity",
        style={"description_width": "75px"},
        layout=widgets.Layout(width="210px"),
    )

    legend_w = widgets.HTML(value=_legend_html())
    out = widgets.Output()
    state = {"fig": None}

    export_btn = widgets.Button(
        description="Export",
        icon="camera",
        button_style="primary",
        layout=widgets.Layout(margin="0px 0px 10px 0px", width="100px"),
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

        fig_layout_defaults = dict(
            title=dict(
                text=f"{dr_method} Embedding - {selected_entity}",
                x=0.0,
                xanchor="left",
                font=dict(size=14, color="#222222"),
            ),
            xaxis=dict(
                title=f"{dr_method} 1",
                gridcolor="#f0f0f0",
                zeroline=False,
                showline=True,
                linecolor="#cccccc",
            ),
            yaxis=dict(
                title=f"{dr_method} 2",
                gridcolor="#f0f0f0",
                zeroline=False,
                showline=True,
                linecolor="#cccccc",
            ),
            template="plotly_white",
            width=750,
            height=620,
            margin=dict(l=40, r=20, t=40, b=40),
        )

        def show_message(text):
            fig.add_annotation(
                text=text,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=14, color="#888888"),
            )
            fig.update_layout(**fig_layout_defaults)
            legend_w.value = _legend_html()
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
        )

        if df is None or df.empty:
            show_message(f"No job data found for {selected_entity} in selected date range.")
            return

        max_jobs = df["jobs"].max()
        sizeref = 2.0 * max_jobs / (32**2)

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
                        df["hw_rate_pct"],
                        df["payload_rate_pct"],
                        df["hw_share"] * 100.0,
                    ),
                    axis=-1,
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Total Jobs: %{customdata[1]:,}<br>"
                    "Overall Failure Rate: %{customdata[2]:.2f}%<br>"
                    "Hardware Fault Rate: %{customdata[3]:.2f}%<br>"
                    "Payload Failure Rate: %{customdata[4]:.2f}%<br>"
                    "HW Share of Failures: %{customdata[5]:.1f}%<br>"
                    "<b>DR Coord:</b> (%{x:.2f}, %{y:.2f})"
                    "<extra></extra>"
                ),
                marker=dict(
                    size=df["jobs"],
                    sizemode="area",
                    sizeref=sizeref,
                    sizemin=5,
                    color=df["hw_share"],
                    colorscale="RdBu_r",
                    cmin=0.0,
                    cmax=1.0,
                    opacity=df["opacity"].tolist(),
                    line=dict(width=0.8, color="rgba(40,40,40,0.5)"),
                ),
            )
        )

        fig.update_layout(**fig_layout_defaults)

        v_small = max(1, int(max_jobs * 0.10))
        v_med = max(2, int(max_jobs * 0.50))
        v_large = int(max_jobs)
        legend_w.value = _legend_html(v_small, v_med, v_large)

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

    controls_box = widgets.VBox(
        [start_date_w, end_date_w, dr_method_w, entity_w, legend_w],
        layout=widgets.Layout(margin="0px 0px 0px 15px", width="240px"),
    )
    dashboard_layout = widgets.HBox(
        [out, controls_box], layout=widgets.Layout(align_items="center")
    )
    app_layout = widgets.VBox(
        [export_btn, dashboard_layout],
        layout=widgets.Layout(align_items="flex-start"),
    )

    update_dashboard()
    return app_layout
