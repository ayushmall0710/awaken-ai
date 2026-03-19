"""P300 Oddball QC HTML report — parquet-based, embeds pre-generated plots."""

from __future__ import annotations

from datetime import datetime
import math
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.data_loading import config
from src.reports import style_utils

# Re-export for table formatting
ICON_TRUE = style_utils.ICON_TRUE
ICON_FALSE = style_utils.ICON_FALSE


P300_WINDOW_MS = (300.0, 600.0)
MMN_WINDOW_MS = (100.0, 250.0)


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
        self._setup_output_dir(output_dir)

    def _setup_output_dir(self, output_dir: Optional[Path]) -> None:
        """Setup a timestamped session report directory unless one is provided explicitly."""
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = config.REPORTS_DIR / self.patient_id / self.session_id / "oddball" / timestamp

        self.output_dir = Path(output_dir)
        self.report_file = self.output_dir / "oddball_qc.html"

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
            self._build_mapping_table(),
            self._build_confidence_interpretation_box(),
            self._build_legend_box(),
        ]
        return "\n".join(parts)

    @staticmethod
    def _coerce_float(value: Any) -> Optional[float]:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(out):
            return None
        return out

    @staticmethod
    def _coerce_int(value: Any) -> Optional[int]:
        numeric = OddballQCReport._coerce_float(value)
        if numeric is None:
            return None
        return int(round(numeric))

    @staticmethod
    def _in_range(value: Optional[float], window_ms: tuple[float, float]) -> bool:
        if value is None:
            return False
        return window_ms[0] <= value <= window_ms[1]

    def _detail_value(self, electrode: str, column: str) -> Any:
        if self.detail_df.empty or "electrode" not in self.detail_df.columns:
            return None
        mask = self.detail_df["electrode"].astype(str).str.upper() == electrode.upper()
        if not mask.any():
            return None
        return self.detail_df.loc[mask, column].iloc[0] if column in self.detail_df.columns else None

    def _difference_support(
        self,
        diff_amp_pz: Optional[float],
        diff_lat_pz: Optional[float],
    ) -> str:
        if diff_amp_pz is None or diff_lat_pz is None:
            return "unavailable"
        if diff_amp_pz > 0 and self._in_range(diff_lat_pz, P300_WINDOW_MS):
            return "supportive"
        return "weak"

    def _stats_support(
        self,
        p_val: Optional[float],
        n_rare: Optional[int],
        n_std: Optional[int],
    ) -> str:
        if p_val is None or n_rare is None or n_std is None or n_rare < 2 or n_std < 2:
            return "unavailable"
        if p_val < 0.05:
            return "supportive"
        if p_val < 0.20:
            return "weak"
        return "not_supported"

    @staticmethod
    def _trial_count_tier(n_rare_epochs: Optional[int]) -> str:
        if n_rare_epochs is None or n_rare_epochs < 10:
            return "poor"
        if n_rare_epochs < 20:
            return "borderline"
        return "good"

    @staticmethod
    def _snr_tier(snr: Optional[float]) -> str:
        if snr is None:
            return "unavailable"
        if snr < 1.25:
            return "poor"
        if snr < 2.0:
            return "borderline"
        return "good"

    @staticmethod
    def _data_quality_tier(trial_count_tier: str, snr_tier: str) -> str:
        if trial_count_tier == "poor" or snr_tier == "poor":
            return "poor"
        if trial_count_tier == "borderline" or snr_tier == "borderline":
            return "borderline"
        if trial_count_tier == "good" and snr_tier == "good":
            return "good"
        return "unknown"

    @staticmethod
    def _topography_label(subtype: Any) -> str:
        subtype_str = str(subtype).strip()
        if subtype_str == "P3b":
            return "P3b (Pz)"
        if subtype_str == "P3a":
            return "P3a (Fz)"
        if subtype_str == "mixed":
            return "Mixed (Cz)"
        return "Absent / unclear"

    @staticmethod
    def _join_limiters(limiters: list[str]) -> str:
        if not limiters:
            return ""
        if len(limiters) == 1:
            return limiters[0]
        if len(limiters) == 2:
            return f"{limiters[0]} and {limiters[1]}"
        return f"{', '.join(limiters[:-1])}, and {limiters[-1]}"

    def _interpret_p300_summary(self) -> Dict[str, Any]:
        rare_amp_pz = self._coerce_float(self.clinical_row.get("p300_rare_amplitude_Pz_uV"))
        rare_lat_pz = self._coerce_float(self.clinical_row.get("p300_rare_latency_Pz_ms"))

        diff_amp_pz = self._coerce_float(self.clinical_row.get("p300_diff_amplitude_Pz_uV"))
        if diff_amp_pz is None:
            diff_amp_pz = self._coerce_float(self._detail_value("Pz", "diff_amplitude_uV"))

        diff_lat_pz = self._coerce_float(self.clinical_row.get("p300_diff_latency_Pz_ms"))
        if diff_lat_pz is None:
            diff_lat_pz = self._coerce_float(self._detail_value("Pz", "diff_latency_ms"))

        mmn_amp_fz = self._coerce_float(self.clinical_row.get("diff_mmn_amplitude_Fz_uV"))
        if mmn_amp_fz is None:
            mmn_amp_fz = self._coerce_float(self._detail_value("Fz", "diff_mmn_amplitude_uV"))

        mmn_lat_fz = self._coerce_float(self.clinical_row.get("diff_mmn_latency_Fz_ms"))
        if mmn_lat_fz is None:
            mmn_lat_fz = self._coerce_float(self._detail_value("Fz", "diff_mmn_latency_ms"))

        baseline = self._coerce_float(self.clinical_row.get("baseline_std_uV"))
        snr = None
        if rare_amp_pz is not None and baseline not in (None, 0):
            snr = rare_amp_pz / baseline

        n_rare_epochs = self._coerce_int(self.clinical_row.get("n_rare_epochs"))
        n_standard_epochs = self._coerce_int(self.clinical_row.get("n_standard_epochs"))
        p_val = self._coerce_float(self.clinical_row.get("p300_p_value"))
        t_stat = self._coerce_float(self.clinical_row.get("p300_t_stat"))
        p300_n_rare = self._coerce_int(self.clinical_row.get("p300_n_rare"))
        p300_n_standard = self._coerce_int(self.clinical_row.get("p300_n_standard"))
        best_electrode = self.clinical_row.get("p300_best_electrode")
        subtype = self.clinical_row.get("p300_subtype")

        pz_available = rare_amp_pz is not None and rare_lat_pz is not None
        if not pz_available:
            candidate_reason = "pz_unavailable"
        elif rare_amp_pz <= 0:
            candidate_reason = "no_positive_peak"
        elif not self._in_range(rare_lat_pz, P300_WINDOW_MS):
            candidate_reason = "latency_out_of_range"
        else:
            candidate_reason = "present"

        candidate_present = candidate_reason == "present"
        difference_support = self._difference_support(diff_amp_pz, diff_lat_pz)
        stats_support = self._stats_support(p_val, p300_n_rare, p300_n_standard)
        trial_count_tier = self._trial_count_tier(n_rare_epochs)
        snr_tier = self._snr_tier(snr)
        data_quality_tier = self._data_quality_tier(trial_count_tier, snr_tier)
        mmn_valid = mmn_amp_fz is not None and mmn_amp_fz < 0 and self._in_range(mmn_lat_fz, MMN_WINDOW_MS)
        topography_label = self._topography_label(subtype)

        if (
            candidate_present
            and difference_support == "supportive"
            and trial_count_tier == "good"
            and snr_tier in {"good", "borderline"}
            and stats_support in {"supportive", "weak"}
            and data_quality_tier != "poor"
        ):
            confidence_label = "Detected"
        elif candidate_present and data_quality_tier != "poor":
            confidence_label = "Low-confidence detected"
        else:
            confidence_label = "No reliable P300 detected"

        limiter_phrases: list[str] = []
        if not pz_available:
            limiter_phrases.append("Pz metrics unavailable")
        elif candidate_reason == "latency_out_of_range":
            limiter_phrases.append("peak outside 300-600 ms")
        elif candidate_reason == "no_positive_peak":
            limiter_phrases.append("no positive Pz peak")

        if trial_count_tier == "poor":
            limiter_phrases.append("insufficient rare-trial count")
        elif trial_count_tier == "borderline":
            limiter_phrases.append("borderline rare-trial count")

        if snr_tier == "poor":
            limiter_phrases.append("poor signal-to-noise")
        elif snr_tier == "borderline":
            limiter_phrases.append("borderline signal-to-noise")

        if stats_support == "not_supported":
            limiter_phrases.append("non-significant rare-standard contrast")
        elif stats_support == "unavailable":
            limiter_phrases.append("rare-standard test unavailable")

        if difference_support == "weak":
            limiter_phrases.append("weak difference-wave support")
        elif difference_support == "unavailable":
            limiter_phrases.append("difference-wave support unavailable")

        return {
            "rare_amp_pz": rare_amp_pz,
            "rare_lat_pz": rare_lat_pz,
            "diff_amp_pz": diff_amp_pz,
            "diff_lat_pz": diff_lat_pz,
            "mmn_amp_fz": mmn_amp_fz,
            "mmn_lat_fz": mmn_lat_fz,
            "baseline": baseline,
            "snr": snr,
            "n_rare_epochs": n_rare_epochs,
            "n_standard_epochs": n_standard_epochs,
            "p_val": p_val,
            "t_stat": t_stat,
            "p300_n_rare": p300_n_rare,
            "p300_n_standard": p300_n_standard,
            "best_electrode": best_electrode,
            "subtype": subtype,
            "pz_available": pz_available,
            "candidate_present": candidate_present,
            "candidate_reason": candidate_reason,
            "difference_support": difference_support,
            "stats_support": stats_support,
            "trial_count_tier": trial_count_tier,
            "snr_tier": snr_tier,
            "data_quality_tier": data_quality_tier,
            "mmn_valid": mmn_valid,
            "topography_label": topography_label,
            "confidence_label": confidence_label,
            "limiter_phrases": limiter_phrases,
        }

    @staticmethod
    def _tier_badge(text: str, tier: str) -> str:
        if tier in {"Detected", "Separated", "Good", "supportive"}:
            return style_utils.build_status_badge(text, style_utils.BG_SUCCESS, style_utils.TEXT_SUCCESS)
        if tier in {"Low-confidence detected", "Trend only", "Borderline", "weak"}:
            return style_utils.build_status_badge(text, style_utils.BG_WARNING, style_utils.TEXT_WARNING)
        if tier in {"No reliable P300 detected", "Not separated", "Poor", "not_supported"}:
            return style_utils.build_status_badge(text, style_utils.BG_DANGER, style_utils.TEXT_DANGER)
        return style_utils.build_status_badge(text, style_utils.BG_INFO, style_utils.TEXT_INFO)

    @staticmethod
    def _format_amp_latency(amplitude: Optional[float], latency: Optional[float]) -> str:
        if amplitude is None or latency is None:
            return "N/A"
        return f"{amplitude:.2f} µV at {latency:.0f} ms"

    @staticmethod
    def _format_count(value: Optional[int]) -> str:
        return str(value) if value is not None else "N/A"

    def _build_results_overview(self) -> str:
        summary = self._interpret_p300_summary()

        if summary["candidate_present"]:
            candidate_value = self._format_amp_latency(summary["rare_amp_pz"], summary["rare_lat_pz"])
            candidate_desc = "Positive rare-only Pz peak in the 300-600 ms window."
            candidate_border = style_utils.TEXT_SUCCESS
        else:
            candidate_value = "No reliable candidate"
            if summary["candidate_reason"] == "pz_unavailable":
                candidate_desc = "Pz was unavailable or could not be quantified."
            elif summary["candidate_reason"] == "latency_out_of_range":
                candidate_desc = "Largest positive Pz peak fell outside the 300-600 ms window."
            else:
                candidate_desc = "No positive Pz peak was detected in the target window."
            candidate_border = style_utils.TEXT_DANGER

        confidence_text = summary["confidence_label"]
        confidence_badge = self._tier_badge(confidence_text, confidence_text)
        limiters = self._join_limiters(summary["limiter_phrases"])
        if confidence_text == "Detected":
            confidence_desc = "Positive Pz peak in 300-600 ms with supportive QC and condition contrast."
        elif confidence_text == "Low-confidence detected":
            confidence_desc = f"Positive Pz peak in 300-600 ms, but support is limited by {limiters}."
        elif limiters:
            confidence_desc = f"Interpretation is limited by {limiters}."
        else:
            confidence_desc = "No reliable positive Pz peak in 300-600 ms, or available data were insufficient."

        stats_map = {
            "supportive": "Separated",
            "weak": "Trend only",
            "not_supported": "Not separated",
            "unavailable": "Unavailable",
        }
        stats_text = stats_map[summary["stats_support"]]
        stats_badge = self._tier_badge(stats_text, summary["stats_support"])
        if summary["stats_support"] == "unavailable":
            stats_desc = "Insufficient rare or standard trials for the Welch test."
        else:
            stats_desc = (
                f"Welch p={summary['p_val']:.3f} (n_rare={summary['p300_n_rare']}, n_std={summary['p300_n_standard']})"
            )

        quality_text = summary["data_quality_tier"].capitalize()
        quality_badge = self._tier_badge(quality_text, quality_text)
        if summary["n_rare_epochs"] is not None and summary["snr"] is not None:
            quality_desc = f"Rare epochs: {summary['n_rare_epochs']}; SNR: {summary['snr']:.2f}×"
        elif summary["n_rare_epochs"] is not None:
            quality_desc = f"Rare epochs: {summary['n_rare_epochs']}; SNR unavailable"
        elif summary["snr"] is not None:
            quality_desc = f"Rare epochs unavailable; SNR: {summary['snr']:.2f}×"
        else:
            quality_desc = "Rare-count and SNR metrics unavailable"

        row1 = [
            {
                "title": "P300 Candidate at Pz",
                "value": candidate_value,
                "desc": candidate_desc,
                "border_color": candidate_border,
            },
            {
                "title": "Confidence",
                "value": confidence_badge,
                "desc": confidence_desc,
                "border_color": style_utils.UW_PURPLE,
            },
            {
                "title": "Rare vs Standard Support",
                "value": stats_badge,
                "desc": stats_desc,
                "border_color": style_utils.TEXT_INFO,
            },
            {
                "title": "Data Quality",
                "value": quality_badge,
                "desc": quality_desc,
                "border_color": style_utils.TEXT_WARNING if quality_text == "Borderline" else style_utils.UW_PURPLE,
            },
        ]

        if summary["mmn_valid"]:
            mmn_value = self._format_amp_latency(summary["mmn_amp_fz"], summary["mmn_lat_fz"])
            mmn_desc = "Difference-wave MMN candidate at Fz."
        else:
            mmn_value = "Not reliable"
            mmn_desc = "No reliable MMN candidate at Fz."

        topography_desc = (
            f"Best electrode: {summary['best_electrode']}"
            if summary["best_electrode"] not in (None, "", "nan")
            else "Best electrode unavailable"
        )

        row2 = [
            {
                "title": "Rare Epochs",
                "value": self._format_count(summary["n_rare_epochs"]),
                "desc": "Included rare epochs",
            },
            {
                "title": "Standard Epochs",
                "value": self._format_count(summary["n_standard_epochs"]),
                "desc": "Included standard epochs",
            },
            {
                "title": "MMN at Fz",
                "value": mmn_value,
                "desc": mmn_desc,
                "border_color": style_utils.TEXT_INFO,
            },
            {
                "title": "Topography",
                "value": summary["topography_label"],
                "desc": topography_desc,
                "border_color": style_utils.TEXT_INFO,
            },
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
                    if not (MMN_WINDOW_MS[0] <= lat <= MMN_WINDOW_MS[1]):
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
            f"<details class='report-details'>"
            f"<summary>Technical Diagnostics: Mapping Forensics</summary>"
            f"{table_html}</details>"
        )

    def _build_legend_box(self) -> str:
        items = [
            {
                "term": "P300 Candidate at Pz",
                "desc": "Positive rare-only Pz peak inside 300–600 ms.",
                "ranges": None,
            },
            {
                "term": "Confidence",
                "desc": "Summarizes morphology, QC, difference-wave support, and Welch support.",
                "ranges": [
                    ("legend-excellent", "Detected"),
                    ("legend-ok", "Low-confidence detected"),
                    ("legend-bad", "No reliable P300"),
                ],
            },
            {
                "term": "Rare vs Standard Support",
                "desc": "Welch t-test on single-trial mean amplitude at Pz (300–600 ms).",
                "ranges": [
                    ("legend-excellent", "Separated"),
                    ("legend-good", "Trend only"),
                    ("legend-bad", "Not separated"),
                ],
            },
            {
                "term": "Data Quality",
                "desc": "Uses rare-epoch count and P300 signal/noise (Pz amp ÷ baseline σ).",
                "ranges": [
                    ("legend-excellent", "Good"),
                    ("legend-ok", "Borderline"),
                    ("legend-bad", "Poor"),
                ],
            },
            {
                "term": "MMN validity",
                "desc": "MMN is reliable only when Fz difference-wave amplitude is negative and latency is 100–250 ms.",
                "ranges": None,
            },
            {
                "term": "Topography",
                "desc": "P3b (Pz-max), P3a (Fz-max), mixed (Cz-max), or absent/unclear.",
                "ranges": None,
            },
        ]
        legend_html = style_utils.build_legend_box("Legend", items)
        return (
            f"<details class='report-details'><summary>Legend and metric definitions</summary>{legend_html}</details>"
        )

    def _build_confidence_interpretation_box(self) -> str:
        return """
        <section class='confidence-interpretation'>
            <h3>Confidence Interpretation</h3>
            <p>
                Confidence combines Pz morphology, rare-trial count, signal-to-noise, rare-vs-standard Welch support,
                and difference-wave support.
            </p>
            <dl class='confidence-terms'>
                <dt>Detected</dt>
                <dd>Positive Pz peak in 300-600 ms with supportive QC and condition contrast.</dd>
                <dt>Low-confidence detected</dt>
                <dd>P300-like peak present, but limited by trial count, noise, or weak support metrics.</dd>
                <dt>No reliable P300 detected</dt>
                <dd>No usable positive Pz peak in 300-600 ms, or the available evidence is insufficient.</dd>
            </dl>
            <p class='confidence-note'>
                A non-significant Welch test does not by itself mean no P300-like morphology was observed.
            </p>
            <div class='confidence-thresholds'>
                <p class='confidence-thresholds-title'>Key thresholds</p>
                <ul>
                    <li>P300 candidate window: 300-600 ms</li>
                    <li>MMN validity window: 100-250 ms</li>
                    <li>Rare-trial count: &gt;=20 good, 10-19 borderline, &lt;10 poor</li>
                    <li>Signal-to-noise: &gt;=2.0 good, 1.25-1.99 borderline, &lt;1.25 poor</li>
                    <li>Welch support: p&lt;0.05 supportive, 0.05-0.19 weak, &gt;=0.20 not supportive</li>
                </ul>
            </div>
        </section>
        """

    def _build_plots_section(self) -> str:
        paths = self._resolve_plot_paths()
        labels = {
            "p300": "P300 Focus (Pz)",
            "mmn": "MMN Focus (Fz)",
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
        row3_parts = [card(k) for k in ("topomap", "erp_image") if card(k)]
        row3 = f"<div class='plot-row'>\n{''.join(row3_parts)}\n</div>" if row3_parts else ""

        return f"{row_p300}\n{row_mmn}\n{row3}"

    def _resolve_plot_paths(self) -> Dict[str, Optional[str]]:
        """Return base64-encoded data URIs for plot PNGs/GIFs so images are embedded inline."""
        import base64

        base = f"{self.patient_id}_{self.session_id}_oddball"
        plots_dir = config.ERP_PLOTS_DIR
        out = {}
        for key, suffix in [
            ("p300", "p300"),
            ("mmn", "mmn"),
            ("erp", "erp"),
            ("topomap", "topomap"),
            ("erp_image", "erp_image"),
        ]:
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
            if num > 10:
                return f"<span style='color: {style_utils.TEXT_WARNING}; font-weight: bold;'>{formatted} ⚠</span>"
            return f"<span style='color: {style_utils.TEXT_SUCCESS};'>{formatted}</span>"
        except (ValueError, TypeError):
            return str(val)

    def _build_css_extensions(self) -> str:
        border = style_utils.BORDER_LIGHT
        legend_bg = style_utils.LEGEND_BG
        text_muted = style_utils.TEXT_MUTED
        purple = style_utils.UW_PURPLE
        return f"""
    .plot-row {{ display: flex; flex-wrap: wrap; gap: 1.5rem; margin: 1rem 0; }}
    .plot-row-full {{ margin-top: 0.5rem; }}
    .plot-card {{ flex: 1 1 0; min-width: 300px; }}
    .plot-card-full {{ flex: 1 1 100%; min-width: 0; }}
    .plot-row > .plot-card {{ display: flex; flex-direction: column; }}
    .plot-row > .plot-card .plot-img {{ width: 100%; height: 520px; object-fit: contain; }}
    .plot-img {{ max-width: 100%; height: auto; border: 1px solid {border}; border-radius: 8px; }}
    .report-details {{ margin: 1rem 0; padding: 0.6rem 0.85rem; background: {legend_bg};
        border: 1px solid {border}; border-radius: 6px; }}
    .report-details summary {{ font-weight: 600; cursor: pointer; color: {text_muted}; }}
    .report-details summary:hover {{ opacity: 0.8; }}
    .metric-card-desc {{ white-space: normal; }}
    .confidence-interpretation {{ margin: 1.5rem 0 1rem; padding: 1rem 1.1rem; background: {legend_bg};
        border-left: 4px solid {purple}; border-radius: 6px; }}
    .confidence-interpretation h3 {{ margin: 0 0 0.6rem; }}
    .confidence-interpretation p {{ margin: 0.5rem 0; }}
    .confidence-terms {{ margin: 0.75rem 0; }}
    .confidence-terms dt {{ font-weight: 700; color: {purple}; margin-top: 0.45rem; }}
    .confidence-terms dd {{ margin: 0.15rem 0 0 1rem; }}
    .confidence-note {{ font-style: italic; }}
    .confidence-thresholds {{ margin-top: 0.75rem; }}
    .confidence-thresholds-title {{ font-weight: 700; color: {purple}; margin-bottom: 0.25rem; }}
    .confidence-thresholds ul {{ margin: 0.25rem 0 0 1.1rem; padding: 0; }}
    """
