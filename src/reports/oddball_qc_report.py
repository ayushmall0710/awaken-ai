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
    """Build session HTML fragments and standalone summary report from parquet-sliced data."""

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
        return style_utils.wrap_session_fragment(self.session_id, self._build_content_html())

    def generate(self) -> Path:
        """Write a standalone single-session HTML file to disk."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        html = style_utils.build_html_header(
            "P300 Oddball Summary Report",
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
            self._build_p300_electrode_table(),
            self._build_mmn_electrode_table(),
            self._build_clinical_table(),
            self._build_legend_box(),
            self._build_mapping_table(),
        ]
        return "\n".join(parts)

    def _p300_detected(self) -> bool:
        """Heuristic: P300 is 'detected' if a finite Pz peak exists within the P300 window."""
        amp = self.clinical_row.get("p300_rare_amplitude_Pz_uV")
        lat = self.clinical_row.get("p300_rare_latency_Pz_ms")
        try:
            amp_f = float(amp)
            lat_f = float(lat)
        except Exception:
            return False
        if math.isnan(amp_f) or math.isnan(lat_f):
            return False
        # P300 window (ms) matches ERP_CONFIG["p300_window"] = 300–600ms
        return (amp_f > 0.0) and (300.0 <= lat_f <= 600.0)

    def _build_results_overview(self) -> str:
        detected = self._p300_detected()
        badge_text = "P300 Detected" if detected else "P300 Not Detected"
        badge_color = style_utils.BG_SUCCESS if detected else style_utils.BG_DANGER
        text_color = style_utils.TEXT_SUCCESS if detected else style_utils.TEXT_DANGER
        badge = style_utils.build_status_badge(badge_text, bg_color=badge_color, text_color=text_color)

        # --- Row 1 (P300 focus) ---
        p300_amp_pz = self._format_cell("p300_rare_amplitude_Pz_uV", self.clinical_row.get("p300_rare_amplitude_Pz_uV"))
        p300_lat_pz = self._format_cell("p300_rare_latency_Pz_ms", self.clinical_row.get("p300_rare_latency_Pz_ms"))
        baseline = self.clinical_row.get("baseline_std_uV")
        amp_raw = self.clinical_row.get("p300_rare_amplitude_Pz_uV")
        try:
            snr = float(amp_raw) / float(baseline) if amp_raw is not None and baseline not in (None, 0) else None
            snr_str = f"{snr:.2f}×" if snr is not None and not (isinstance(snr, float) and math.isnan(snr)) else "N/A"
        except Exception:
            snr_str = "N/A"

        row1 = [
            {"title": "P300 Status", "value": badge, "desc": "Detected if Pz peak is finite and within 300–600 ms"},
            {"title": "Signal/Noise", "value": snr_str, "desc": "P300 Amp (Pz) ÷ baseline σ"},
            {"title": "P300 Amplitude (Pz)", "value": p300_amp_pz, "desc": "Rare-only ERP peak at Pz (µV)"},
            {"title": "P300 Latency (Pz)", "value": p300_lat_pz, "desc": "Rare-only ERP peak latency at Pz (ms)"},
        ]

        # --- Row 2 (MMN focus) ---
        n_rare = self.clinical_row.get("n_rare_epochs")
        n_std = self.clinical_row.get("n_standard_epochs")
        mmn_amp_fz = self._format_cell("diff_mmn_amplitude_Fz_uV", self.clinical_row.get("diff_mmn_amplitude_Fz_uV"))
        mmn_lat_fz = self._format_cell("diff_mmn_latency_Fz_ms", self.clinical_row.get("diff_mmn_latency_Fz_ms"))
        row2 = [
            {"title": "Rare Epochs", "value": str(n_rare) if n_rare is not None else "N/A", "desc": "Included rare epochs"},
            {"title": "Standard Epochs", "value": str(n_std) if n_std is not None else "N/A", "desc": "Included standard epochs"},
            {"title": "MMN Amplitude (Fz)", "value": mmn_amp_fz, "desc": "Difference wave negative peak at Fz (µV)"},
            {"title": "MMN Latency (Fz)", "value": mmn_lat_fz, "desc": "Difference wave MMN latency at Fz (ms)"},
        ]

        return f"{style_utils.build_metric_cards(row1)}\n{style_utils.build_metric_cards(row2)}"

    def _build_clinical_table(self) -> str:
        row = self.clinical_row
        baseline = self._format_baseline(row.get("baseline_std_uV"))
        headers = [
            "Best Electrode",
            "Valid Electrodes",
            "Baseline σ (µV)",
            "QC Notes",
        ]
        cells = [
            self._format_cell("p300_best_electrode", row.get("p300_best_electrode")),
            self._format_cell("p300_n_valid_electrodes", row.get("p300_n_valid_electrodes")),
            baseline,
            self._format_cell("qc_notes", row.get("qc_notes", "")),
        ]
        return style_utils.build_metric_table(headers, [cells], title="Additional Metadata")

    def _build_p300_electrode_table(self) -> str:
        headers = [
            "Electrode",
            "Valid",
            "Rare Amp (uV)",
            "Rare Lat (ms)",
            "Diff Amp (uV)",
            "Diff Lat (ms)",
            "Flag",
        ]
        if self.detail_df.empty:
            return style_utils.build_metric_table(headers, [], title="P300 Breakdown")
        rows = []
        for _, r in self.detail_df.iterrows():
            rows.append(
                [
                    r.get("electrode", ""),
                    self._format_cell("is_valid", r.get("is_valid")),
                    self._format_cell("p300_amplitude_uV", r.get("p300_amplitude_uV")),
                    self._format_cell("p300_latency_ms", r.get("p300_latency_ms")),
                    self._format_cell("diff_amplitude_uV", r.get("diff_amplitude_uV")),
                    self._format_cell("diff_latency_ms", r.get("diff_latency_ms")),
                    self._format_cell("flagged_reason", r.get("flagged_reason")),
                ]
            )
        return style_utils.build_metric_table(headers, rows, title="P300 Breakdown")

    def _build_mmn_electrode_table(self) -> str:
        headers = [
            "Electrode",
            "Valid",
            "Rare Amp (uV)",
            "Rare Lat (ms)",
            "Diff Amp (uV)",
            "Diff Lat (ms)",
            "Flag",
        ]
        if self.detail_df.empty:
            return style_utils.build_metric_table(headers, [], title="MMN Breakdown")

        rows = []
        for _, r in self.detail_df.iterrows():
            elec = r.get("electrode", "")
            # MMN validity: diff MMN amplitude should be negative, latency within 100–250ms.
            mmn_amp = r.get("diff_mmn_amplitude_uV")
            mmn_lat = r.get("diff_mmn_latency_ms")

            mmn_valid = True
            mmn_flag = ""
            if mmn_amp is None or (isinstance(mmn_amp, float) and math.isnan(mmn_amp)):
                mmn_valid = False
                mmn_flag = "missing"
            elif float(mmn_amp) >= 0:
                mmn_valid = False
                mmn_flag = "non_negative"
            elif mmn_lat is None or (isinstance(mmn_lat, float) and math.isnan(mmn_lat)):
                mmn_valid = False
                mmn_flag = "missing_latency"
            else:
                try:
                    lat = float(mmn_lat)
                    if not (100 <= lat <= 250):
                        mmn_valid = False
                        mmn_flag = "out_of_range"
                except Exception:
                    mmn_valid = False
                    mmn_flag = "bad_latency"

            rows.append(
                [
                    elec,
                    ICON_TRUE if mmn_valid else ICON_FALSE,
                    self._format_cell("p300_amplitude_uV", r.get("p300_amplitude_uV")),
                    self._format_cell("p300_latency_ms", r.get("p300_latency_ms")),
                    self._format_cell("diff_mmn_amplitude_uV", mmn_amp),
                    self._format_cell("diff_mmn_latency_ms", mmn_lat),
                    mmn_flag or "—",
                ]
            )

        return style_utils.build_metric_table(headers, rows, title="MMN Breakdown")

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
            {
                "term": "mmn_validity",
                "desc": "MMN Valid: difference-wave MMN amplitude < 0 µV and latency within 100–250 ms.",
                "ranges": None,
            },
        ]
        return style_utils.build_legend_box("Legend", items)

    def _build_plots_section(self) -> str:
        paths = self._resolve_plot_paths()
        labels = {
            "erp": "ERP Waveforms",
            "topomap": "Scalp Topography (Difference Wave)",
            "erp_image": "Single-Trial ERP Image (Pz)",
        }

        def card(key: str) -> str:
            src = paths.get(key)
            if src is None:
                return ""
            lbl = labels.get(key, key)
            return (
                f"<div class='plot-card'><p class='metric-card-title'>{lbl}</p>"
                f"<img src='{src}' alt='{lbl}' class='plot-img'/></div>"
            )

        def card_full(key: str) -> str:
            src = paths.get(key)
            if src is None:
                return ""
            lbl = labels.get(key, key)
            return (
                f"<div class='plot-row plot-row-full'>\n"
                f"<div class='plot-card plot-card-full'><p class='metric-card-title'>{lbl}</p>"
                f"<img src='{src}' alt='{lbl}' class='plot-img'/></div>\n</div>"
            )

        row_p300 = card_full("p300")
        row_mmn = card_full("mmn")
        row_erp = card_full("erp") if not row_p300 and not row_mmn else ""

        # Bottom row: Topomap + Single trial side by side
        row3_parts = [card(k) for k in ("topomap", "erp_image") if card(k)]
        row3 = f"<div class='plot-row'>\n{''.join(row3_parts)}\n</div>" if row3_parts else ""

        return f"{row_p300}\n{row_mmn}\n{row_erp}\n{row3}"

    def _resolve_plot_paths(self) -> Dict[str, Optional[str]]:
        """Return base64-encoded data URIs for plot PNGs so images are embedded inline."""
        import base64

        base = f"{self.patient_id}_{self.session_id}_oddball"
        plots_dir = config.ERP_PLOTS_DIR
        out = {}
        for key, suffix in [("p300", "p300"), ("mmn", "mmn"), ("erp", "erp"), ("topomap", "topomap"), ("erp_image", "erp_image")]:
            p_png = plots_dir / f"{base}_{suffix}.png"
            p_gif = plots_dir / f"{base}_{suffix}.gif"
            if p_gif.exists():
                data = base64.b64encode(p_gif.read_bytes()).decode("ascii")
                out[key] = f"data:image/gif;base64,{data}"
            elif p_png.exists():
                data = base64.b64encode(p_png.read_bytes()).decode("ascii")
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
    .plot-card {{ flex: 1 1 0; min-width: 300px; }}
    .plot-card-full {{ flex: 1 1 100%; min-width: 0; }}
    /* Keep side-by-side plot cards visually consistent */
    .plot-row > .plot-card {{ display: flex; flex-direction: column; }}
    .plot-row > .plot-card .plot-img {{ width: 100%; height: 520px; object-fit: contain; }}
    .plot-img {{ max-width: 100%; height: auto; border: 1px solid {border}; border-radius: 8px; }}
    .tech-details {{ margin: 1rem 0; padding: 0.5rem; background: {legend_bg};
        border: 1px solid {border}; border-radius: 4px; }}
    .tech-details summary {{ font-weight: 500; cursor: pointer; color: {text_muted}; }}
    .tech-details summary:hover {{ opacity: 0.8; }}
    """
