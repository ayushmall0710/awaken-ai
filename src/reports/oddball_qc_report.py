"""P300 Oddball QC HTML report — parquet-based, embeds pre-generated plot PNGs."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.data_loading import config
from src.reports import style_utils

# Re-export for table formatting
ICON_TRUE = style_utils.ICON_TRUE
ICON_FALSE = style_utils.ICON_FALSE


class OddballQCReport:
    """Build session HTML fragments and standalone QC report from parquet-sliced data."""

    def __init__(
        self,
        patient_id: str,
        session_id: str,
        clinical_row: pd.Series,
        detail_df: pd.DataFrame,
        mapping_row: pd.Series,
        output_dir: Optional[Path] = None,
    ):
        self.patient_id = patient_id
        self.session_id = session_id
        self.clinical_row = clinical_row
        self.detail_df = detail_df
        self.mapping_row = mapping_row
        self.output_dir = output_dir or (config.REPORTS_DIR / patient_id / session_id / "oddball")
        self.output_dir = Path(self.output_dir)
        self.report_file = self.output_dir / f"{session_id}_oddball_qc.html"

    def build_session_html(self) -> str:
        """Collapsible <details> fragment for the combined report (used by runner)."""
        panel = style_utils.build_session_panel(self.session_id, collapsible=True)
        content = self._build_content_html()
        return f"<details class='session-wrapper' open>{panel}<div class='session-content'>{content}</div></details>"

    def generate(self) -> Path:
        """Write a standalone single-session HTML file to disk."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        html = style_utils.build_html_header(
            "P300 Oddball QC Report",
            patient_id=self.patient_id,
            session_id=self.session_id,
            extra_css=self._build_css_extensions(),
        )
        html += self._build_content_html()
        html += style_utils.build_html_footer("P300 Oddball Pipeline")
        self.report_file.write_text(html, encoding="utf-8")
        return self.report_file

    def _build_content_html(self) -> str:
        parts = [
            self._build_results_overview(),
            self._build_plots_section(),
            self._build_clinical_table(),
            self._build_legend_box(),
            self._build_electrode_table(),
            self._build_mapping_table(),
        ]
        return "\n".join(parts)

    def _build_results_overview(self) -> str:
        qc_pass = self.clinical_row.get("qc_pass", False)
        badge_text = "P300 QC Pass" if qc_pass else "P300 QC Fail"
        badge_color = style_utils.BG_SUCCESS if qc_pass else style_utils.BG_DANGER
        text_color = style_utils.TEXT_SUCCESS if qc_pass else style_utils.TEXT_DANGER
        badge = style_utils.build_status_badge(badge_text, bg_color=badge_color, text_color=text_color)

        subtype = self._format_cell("p300_subtype", self.clinical_row.get("p300_subtype"))
        n_rare = self.clinical_row.get("n_rare_epochs")
        n_std = self.clinical_row.get("n_standard_epochs")

        pz_amp = self._format_cell(
            "p300_diff_amplitude_Pz_uV",
            self.clinical_row.get("p300_diff_amplitude_Pz_uV"),
        )
        pz_lat = self._format_cell(
            "p300_diff_latency_Pz_ms",
            self.clinical_row.get("p300_diff_latency_Pz_ms"),
        )
        baseline = self._format_baseline(self.clinical_row.get("baseline_std_uV"))
        cards = [
            {"title": "QC Status", "value": badge, "desc": "Pass = ≥2 valid electrodes, subtype ≠ absent"},
            {"title": "Subtype", "value": subtype, "desc": "P3b (Pz-max), P3a (Fz-max), mixed, absent"},
            {"title": "Pz Amplitude (µV)", "value": pz_amp, "desc": "Difference wave peak at Pz"},
            {"title": "Pz Latency (ms)", "value": pz_lat, "desc": "Timing of the peak at Pz"},
            {
                "title": "Rare Epochs",
                "value": str(n_rare) if n_rare is not None else "N/A",
                "desc": "Included rare events",
            },
            {
                "title": "Standard Epochs",
                "value": str(n_std) if n_std is not None else "N/A",
                "desc": "Included standard events",
            },
            {"title": "Baseline σ (µV)", "value": baseline, "desc": "Pre-stimulus noise level"},
        ]
        return style_utils.build_metric_cards(cards)

    def _build_clinical_table(self) -> str:
        row = self.clinical_row
        headers = [
            "Best Electrode",
            "Valid Electrodes",
            "QC Notes",
        ]
        cells = [
            self._format_cell("p300_best_electrode", row.get("p300_best_electrode")),
            self._format_cell("p300_n_valid_electrodes", row.get("p300_n_valid_electrodes")),
            self._format_cell("qc_notes", row.get("qc_notes", "")),
        ]
        return style_utils.build_metric_table(headers, [cells], title="Additional Metadata")

    def _build_electrode_table(self) -> str:
        if self.detail_df.empty:
            empty_table = style_utils.build_metric_table(
                ["Electrode", "Valid", "Amplitude (µV)", "Latency (ms)", "Flag"], [], title=""
            )
            return (
                f"<details class='tech-details'>"
                f"<summary>Technical Diagnostics: Electrode Detail</summary>"
                f"{empty_table}</details>"
            )
        headers = ["Electrode", "Valid", "Amplitude (µV)", "Latency (ms)", "Flag"]
        rows = []
        for _, r in self.detail_df.iterrows():
            rows.append(
                [
                    r.get("electrode", ""),
                    self._format_cell("is_valid", r.get("is_valid")),
                    self._format_cell("p300_amplitude_uV", r.get("p300_amplitude_uV")),
                    self._format_cell("p300_latency_ms", r.get("p300_latency_ms")),
                    self._format_cell("flagged_reason", r.get("flagged_reason")),
                ]
            )
        table_html = style_utils.build_metric_table(headers, rows, title="")
        return (
            f"<details class='tech-details'>"
            f"<summary>Technical Diagnostics: Electrode Detail</summary>"
            f"{table_html}</details>"
        )

    def _build_mapping_table(self) -> str:
        r = self.mapping_row
        headers = [
            "Rare candidate",
            "Rare mapped",
            "Rare unmapped",
            "Rare boundary-clipped",
            "Rare mapping rate",
            "Standard candidate",
            "Standard mapped",
        ]
        rate = r.get("rare_mapping_rate")
        is_nan = isinstance(rate, float) and math.isnan(rate)
        rate_str = f"{float(rate):.1%}" if rate is not None and not is_nan else "N/A"
        cells = [
            self._format_cell("n_rare_events_candidate", r.get("n_rare_events_candidate")),
            self._format_cell("n_rare_mapped", r.get("n_rare_mapped")),
            self._format_cell("n_rare_unmapped", r.get("n_rare_unmapped")),
            self._format_cell("n_rare_boundary_clipped", r.get("n_rare_boundary_clipped")),
            rate_str,
            self._format_cell("n_standard_events_candidate", r.get("n_standard_events_candidate")),
            self._format_cell("n_standard_mapped", r.get("n_standard_mapped")),
        ]
        table_html = style_utils.build_metric_table(headers, [cells], title="")
        return (
            f"<details class='tech-details'>"
            f"<summary>Technical Diagnostics: Mapping Forensics</summary>"
            f"{table_html}</details>"
        )

    def _build_legend_box(self) -> str:
        items = [
            {
                "term": "qc_pass",
                "desc": (
                    "Pass requires ≥2 valid electrodes (positive amplitude, latency 250–600 ms) and subtype ≠ absent."
                ),
                "ranges": [
                    ("legend-excellent", "Pass"),
                    ("legend-bad", "Fail"),
                ],
            },
            {
                "term": "p300_subtype",
                "desc": "P3b (Pz-max), P3a (Fz-max), mixed (Cz-max), absent.",
                "ranges": None,
            },
            {
                "term": "baseline_std_uV",
                "desc": "Pre-stimulus noise level (µV). Lower is better.",
                "ranges": [
                    ("legend-excellent", "&lt; 10 µV &mdash; Good"),
                    ("legend-ok", "10–20 µV &mdash; Marginal"),
                    ("legend-bad", "&gt; 20 µV &mdash; Poor"),
                ],
            },
        ]
        return style_utils.build_legend_box("Legend", items)

    def _build_plots_section(self) -> str:
        paths = self._resolve_plot_paths()
        labels = {"erp": "ERP (4-panel)", "topomap": "Difference Topomap", "erp_image": "ERP Image (Pz)"}

        def card(key: str) -> str:
            src = paths.get(key)
            if src is None:
                return ""
            lbl = labels.get(key, key)
            return (
                f"<div class='plot-card'><p class='metric-card-title'>{lbl}</p>"
                f"<img src='{src}' alt='{lbl}' class='plot-img'/></div>"
            )

        # Row 1: ERP (4-panel) + ERP image side by side
        row1_parts = [c for c in (card("erp"), card("erp_image")) if c]
        row1 = f"<div class='plot-row'>\n{''.join(row1_parts)}\n</div>" if row1_parts else ""

        # Row 2: Topomap full width
        topo = paths.get("topomap")
        row2 = (
            (
                f"<div class='plot-row plot-row-full'>\n"
                f"<div class='plot-card plot-card-full'><p class='metric-card-title'>{labels['topomap']}</p>"
                f"<img src='{topo}' alt='{labels['topomap']}' class='plot-img'/></div>\n</div>"
            )
            if topo
            else ""
        )

        return f"{row1}\n{row2}"

    def _resolve_plot_paths(self) -> Dict[str, Optional[str]]:
        """Return base64-encoded data URIs for plot PNGs so images are embedded inline."""
        import base64

        base = f"{self.patient_id}_{self.session_id}_oddball"
        plots_dir = config.ERP_PLOTS_DIR
        out = {}
        for key, suffix in [("erp", "erp"), ("topomap", "topomap"), ("erp_image", "erp_image")]:
            p = plots_dir / f"{base}_{suffix}.png"
            if p.exists():
                data = base64.b64encode(p.read_bytes()).decode("ascii")
                out[key] = f"data:image/png;base64,{data}"
            else:
                out[key] = None
        return out

    def _format_cell(self, col: str, val: Any) -> str:
        if col == "is_valid":
            if val is True:
                return ICON_TRUE
            if val is False:
                return ICON_FALSE
            return "N/A"
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return "N/A"
        if isinstance(val, float) and col != "qc_notes":
            return f"{val:.2f}"
        return str(val)

    def _format_baseline(self, val: Any) -> str:
        """Format baseline noise with color warnings for high values."""
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return "N/A"
        try:
            num = float(val)
            formatted = f"{num:.2f}"
            if num > 20:
                return f"<span style='color: {style_utils.TEXT_DANGER}; font-weight: bold;'>{formatted} ⚠</span>"
            elif num > 10:
                return f"<span style='color: {style_utils.TEXT_WARNING}; font-weight: bold;'>{formatted} ⚠</span>"
            return f"<span style='color: {style_utils.TEXT_SUCCESS};'>{formatted}</span>"
        except (ValueError, TypeError):
            return str(val)

    def _build_css_extensions(self) -> str:
        border = style_utils.BORDER_LIGHT
        legend_bg = style_utils.LEGEND_BG
        text_muted = style_utils.TEXT_MUTED
        return f"""
    .plot-row {{ display: flex; flex-wrap: wrap; gap: 1.5rem; margin: 1rem 0; }}
    .plot-row-full {{ margin-top: 0.5rem; }}
    .plot-card {{ flex: 1; min-width: 300px; }}
    .plot-card-full {{ flex: 1 1 100%; min-width: 0; }}
    .plot-img {{ max-width: 100%; height: auto; border: 1px solid {border}; border-radius: 8px; }}
    .tech-details {{ margin: 1rem 0; padding: 0.5rem; background: {legend_bg};
        border: 1px solid {border}; border-radius: 4px; }}
    .tech-details summary {{ font-weight: 500; cursor: pointer; color: {text_muted}; }}
    .tech-details summary:hover {{ opacity: 0.8; }}
    """
