"""
ENG-06: QC Report Generation

Aggregates per-session QC metadata produced by ENG-03 (artifact rejection) and
generates an HTML dashboard showing artifact rejection rates, SNR metrics,
epoch counts, and ICA summaries per patient and trial type.

Three focused classes:
- ``QCDataCollector``  — discovers and loads ``eng03_qc.parquet`` files.
- ``QCMetricsCalculator`` — computes derived metrics (drop rates, SNR, ICA stats).
- ``QCReportGenerator`` — renders HTML report + optional summary CSV.

Convenience function:
- ``generate_qc_report()`` — one-call entry point that chains the three classes.

Output schema:
- ``data/processed/reports/qc_report.html``
- ``data/processed/reports/qc_summary.csv``
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data_loading import config

logger = logging.getLogger(__name__)


# ── QCDataCollector ──────────────────────────────────────────────────────────


class QCDataCollector:
    """Discover and load ENG-03 QC parquet files from the processed QC directory."""

    def __init__(self, qc_dir: Path | None = None) -> None:
        self.qc_dir = Path(qc_dir) if qc_dir is not None else config.QC_DIR

    def discover_qc_files(self) -> list[Path]:
        """Glob for all ``eng03_qc.parquet`` files under the QC directory."""
        if not self.qc_dir.exists():
            logger.warning("QC directory does not exist: %s", self.qc_dir)
            return []
        files = sorted(self.qc_dir.glob("**/eng03_qc.parquet"))
        logger.info("Discovered %d QC parquet file(s) under %s", len(files), self.qc_dir)
        return files

    def load_all(self) -> pd.DataFrame:
        """Read and concatenate all QC parquets into a single DataFrame.

        Returns an empty DataFrame with the expected columns when no files exist.
        """
        files = self.discover_qc_files()
        if not files:
            return _empty_qc_dataframe()
        frames = [pd.read_parquet(f) for f in files]
        return pd.concat(frames, ignore_index=True)

    def load_session(self, patient_id: str, date: str) -> pd.DataFrame:
        """Load QC data for a specific patient session."""
        qc_path = self.qc_dir / patient_id / date / "eng03_qc.parquet"
        if not qc_path.exists():
            logger.warning("QC file not found: %s", qc_path)
            return _empty_qc_dataframe()
        return pd.read_parquet(qc_path)

    def get_available_sessions(self) -> list[tuple[str, str]]:
        """Return a list of (patient_id, date) pairs that have QC data."""
        files = self.discover_qc_files()
        return sorted((f.parent.parent.name, f.parent.name) for f in files)


def _empty_qc_dataframe() -> pd.DataFrame:
    """Return an empty DataFrame with the standard ENG-03 QC columns."""
    cols = [
        "patient_id",
        "date",
        "trial_type",
        "window_sec",
        "reject_ptp_percentile",
        "reject_ptp_threshold_uv",
        "n_epochs_total",
        "n_epochs_dropped",
        "n_epochs_kept",
        "drop_reason",
        "ica",
        "notes",
        "ptp_uv_p50",
        "ptp_uv_p95",
        "ptp_uv_p99",
        "ptp_uv_max",
        "ptp_uv_mean",
    ]
    return pd.DataFrame(columns=cols)


# ── QCMetricsCalculator ─────────────────────────────────────────────────────


class QCMetricsCalculator:
    """Compute derived QC metrics from raw ENG-03 QC data.

    All ``compute_*`` methods return a *new* DataFrame (the original is not mutated).
    ``compute_all_metrics()`` chains every computation and returns a single enriched DF.
    """

    def __init__(self, qc_df: pd.DataFrame) -> None:
        self._df = qc_df.copy()

    @property
    def raw_df(self) -> pd.DataFrame:
        """The original (immutable) QC DataFrame passed at init."""
        return self._df.copy()

    def compute_drop_rates(self) -> pd.DataFrame:
        """Add ``drop_rate`` column: fraction of epochs dropped (0.0 – 1.0)."""
        df = self._df.copy()
        totals = pd.to_numeric(df["n_epochs_total"], errors="coerce").fillna(0)
        dropped = pd.to_numeric(df["n_epochs_dropped"], errors="coerce").fillna(0)
        df["drop_rate"] = np.where(totals > 0, dropped / totals, np.nan)
        return df

    def compute_snr_estimates(self) -> pd.DataFrame:
        """Derive a signal-to-noise ratio estimate from PTP statistics.

        SNR is defined as ``20 * log10(ptp_uv_p50 / ptp_uv_p95)``.  This gives
        a negative dB value where values closer to 0 indicate less noisy data.
        """
        df = self._df.copy()
        if "ptp_uv_p50" not in df.columns or "ptp_uv_p95" not in df.columns:
            df["snr_db"] = np.nan
            return df
        p50 = pd.to_numeric(df["ptp_uv_p50"], errors="coerce")
        p95 = pd.to_numeric(df["ptp_uv_p95"], errors="coerce")
        valid = p50.notna() & p95.notna() & (p95 > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(valid, p50 / p95, np.nan)
            df["snr_db"] = np.where(np.isfinite(ratio) & (ratio > 0), 20 * np.log10(ratio), np.nan)
        return df

    def parse_ica_summaries(self) -> pd.DataFrame:
        """Extract key ICA fields from the ``ica`` JSON column.

        Adds columns: ``ica_method``, ``ica_classification_method``,
        ``ica_n_components_excluded``, ``ica_n_eog``, ``ica_n_ecg``,
        ``ica_n_muscle``, ``ica_n_line_noise``, ``ica_n_channel_noise``.
        """
        df = self._df.copy()
        if "ica" not in df.columns:
            return df
        parsed_df = pd.DataFrame(df["ica"].apply(_parse_single_ica_json).tolist(), index=df.index)
        return pd.concat([df, parsed_df], axis=1)

    def compute_retention_rate(self) -> pd.DataFrame:
        """Add ``retention_rate`` column: fraction of epochs kept (0.0 – 1.0).

        Complements ``drop_rate`` — easier to read at a glance
        ("94% of epochs retained").
        """
        df = self._df.copy()
        totals = pd.to_numeric(df["n_epochs_total"], errors="coerce").fillna(0)
        kept = pd.to_numeric(df["n_epochs_kept"], errors="coerce").fillna(0)
        df["retention_rate"] = np.where(totals > 0, kept / totals, np.nan)
        return df

    def compute_data_coverage(self) -> pd.DataFrame:
        """Add ``estimated_recording_min`` column.

        Approximates total recording time per row as
        ``window_sec × n_epochs_total / 60``.  Useful for checking protocol
        adherence (e.g., "we expected ~18 min of language trials").
        """
        df = self._df.copy()
        window = pd.to_numeric(df.get("window_sec", pd.Series(dtype=float)), errors="coerce")
        totals = pd.to_numeric(df.get("n_epochs_total", pd.Series(dtype=float)), errors="coerce")
        df["estimated_recording_min"] = (window * totals) / 60.0
        return df

    def flag_usable_sessions(self, min_retention: float = 0.50, min_epochs_kept: int = 1) -> pd.DataFrame:
        """Add boolean ``is_usable`` column.

        A session row is flagged usable when:
        - ``retention_rate >= min_retention`` (default 50 %)
        - ``n_epochs_kept >= min_epochs_kept`` (default 1)

        Both thresholds are keyword arguments so callers can override easily.

        Raises:
            ValueError: If ``retention_rate`` column is missing — call
                ``compute_retention_rate()`` first (or use ``compute_all_metrics()``).
        """
        df = self._df.copy()
        if "retention_rate" not in df.columns:
            raise ValueError(
                "'retention_rate' column not found. "
                "Call compute_retention_rate() before flag_usable_sessions(), "
                "or use compute_all_metrics() which chains them in order."
            )
        retention = pd.to_numeric(df["retention_rate"], errors="coerce")
        kept_vals = pd.to_numeric(df.get("n_epochs_kept", pd.Series(dtype=float)), errors="coerce").fillna(0)
        df["is_usable"] = (retention >= min_retention) & (kept_vals >= min_epochs_kept)
        return df

    def parse_drop_reasons(self) -> pd.DataFrame:
        """Add ``primary_drop_reason`` column from the ``drop_reason`` field.

        ``drop_reason`` in ENG-03 is a single string label per row (e.g.
        ``"ENG03_PTP_GT_P95"``).  This method normalises it to a clean label
        and stores it as ``primary_drop_reason`` for easy grouping.
        """
        df = self._df.copy()
        if "drop_reason" not in df.columns:
            df["primary_drop_reason"] = "none"
            return df
        df["primary_drop_reason"] = (
            df["drop_reason"].fillna("none").astype(str).str.strip().str.lower().replace("", "none")
        )
        return df

    def compute_all_metrics(self) -> pd.DataFrame:
        """Chain all metric computations and return a fully enriched DataFrame."""
        self._df = self.compute_drop_rates()
        self._df = self.compute_retention_rate()
        self._df = self.compute_snr_estimates()
        self._df = self.parse_ica_summaries()
        self._df = self.compute_data_coverage()
        self._df = self.flag_usable_sessions()
        self._df = self.parse_drop_reasons()
        return self._df.copy()

    def summary_by_patient(self) -> pd.DataFrame:
        """Aggregated statistics grouped by ``patient_id``."""
        df = self.compute_all_metrics()
        return _aggregate_summary(df, group_cols=["patient_id"])

    def summary_by_trial_type(self) -> pd.DataFrame:
        """Aggregated statistics grouped by ``trial_type``."""
        df = self.compute_all_metrics()
        return _aggregate_summary(df, group_cols=["trial_type"])


def _parse_single_ica_json(raw_json: Any) -> dict[str, Any]:
    """Parse one ICA JSON string into flat metric fields.

    When ICLabel was used, the per-type component lists (``eog_components``, etc.)
    are empty because ICLabel puts everything into ``excluded``.  In that case we
    derive per-type counts from ``iclabel_labels`` + ``excluded`` indices.
    """
    defaults: dict[str, Any] = {
        "ica_method": None,
        "ica_classification_method": None,
        "ica_n_components_excluded": 0,
        "ica_n_eog": 0,
        "ica_n_ecg": 0,
        "ica_n_muscle": 0,
        "ica_n_line_noise": 0,
        "ica_n_channel_noise": 0,
    }
    if pd.isna(raw_json) or raw_json is None:
        return defaults
    try:
        d = json.loads(raw_json) if isinstance(raw_json, str) else raw_json
    except (json.JSONDecodeError, TypeError):
        return defaults

    excluded = d.get("excluded", [])
    counts = _count_iclabel_artifact_types(d, excluded)

    return {
        "ica_method": d.get("method"),
        "ica_classification_method": d.get("classification_method"),
        "ica_n_components_excluded": len(excluded),
        "ica_n_eog": counts["eog"],
        "ica_n_ecg": counts["ecg"],
        "ica_n_muscle": counts["muscle"],
        "ica_n_line_noise": counts["line_noise"],
        "ica_n_channel_noise": counts["channel_noise"],
    }


# ICLabel label-string -> our short category key
_ICLABEL_TO_CATEGORY: dict[str, str] = {
    "eye blink": "eog",
    "eye": "eog",
    "heart beat": "ecg",
    "heart": "ecg",
    "muscle artifact": "muscle",
    "muscle": "muscle",
    "line noise": "line_noise",
    "line_noise": "line_noise",
    "channel noise": "channel_noise",
    "channel_noise": "channel_noise",
}


def _count_iclabel_artifact_types(
    d: dict[str, Any],
    excluded: list[int],
) -> dict[str, int]:
    """Count excluded components per artifact type.

    Strategy:
    1. If ``iclabel_labels`` exists and the per-type lists are empty, derive
       counts from the labels of the *excluded* component indices.
    2. Otherwise fall back to the per-type lists stored by ENG-03.
    """
    counts: dict[str, int] = {"eog": 0, "ecg": 0, "muscle": 0, "line_noise": 0, "channel_noise": 0}
    labels = d.get("iclabel_labels")

    has_per_type = any(
        len(d.get(k, [])) > 0
        for k in (
            "eog_components",
            "ecg_components",
            "muscle_components",
            "line_noise_components",
            "channel_noise_components",
        )
    )

    if labels and not has_per_type:
        for idx in excluded:
            if idx < len(labels):
                cat = _ICLABEL_TO_CATEGORY.get(labels[idx].lower().strip())
                if cat:
                    counts[cat] += 1
    else:
        counts["eog"] = len(d.get("eog_components", []))
        counts["ecg"] = len(d.get("ecg_components", []))
        counts["muscle"] = len(d.get("muscle_components", []))
        counts["line_noise"] = len(d.get("line_noise_components", []))
        counts["channel_noise"] = len(d.get("channel_noise_components", []))

    return counts


def _aggregate_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Build a summary table with key aggregate statistics."""
    if df.empty:
        return pd.DataFrame()

    numeric_cols = {
        "n_epochs_total": "sum",
        "n_epochs_dropped": "sum",
        "n_epochs_kept": "sum",
    }
    agg_dict: dict[str, Any] = {}
    for col, func in numeric_cols.items():
        if col in df.columns:
            agg_dict[col] = pd.NamedAgg(column=col, aggfunc=func)

    if "drop_rate" in df.columns:
        agg_dict["mean_drop_rate"] = pd.NamedAgg(column="drop_rate", aggfunc="mean")
    if "retention_rate" in df.columns:
        agg_dict["mean_retention_rate"] = pd.NamedAgg(column="retention_rate", aggfunc="mean")
    if "snr_db" in df.columns:
        agg_dict["mean_snr_db"] = pd.NamedAgg(column="snr_db", aggfunc="mean")
    if "ptp_uv_mean" in df.columns:
        agg_dict["mean_ptp_uv"] = pd.NamedAgg(column="ptp_uv_mean", aggfunc="mean")
    if "estimated_recording_min" in df.columns:
        agg_dict["total_recording_min"] = pd.NamedAgg(column="estimated_recording_min", aggfunc="sum")
    if "is_usable" in df.columns:
        agg_dict["n_usable"] = pd.NamedAgg(column="is_usable", aggfunc="sum")

    agg_dict["n_sessions"] = pd.NamedAgg(column=group_cols[0], aggfunc="count")
    return df.groupby(group_cols, dropna=False).agg(**agg_dict).reset_index()


# ── QCReportGenerator ────────────────────────────────────────────────────────


class QCReportGenerator:
    """Render an HTML QC dashboard and optional CSV summary from enriched QC data."""

    def __init__(
        self,
        metrics_df: pd.DataFrame,
        output_dir: Path | None = None,
        active_filters: dict[str, list[str]] | None = None,
    ) -> None:
        self.df = metrics_df.copy()
        self.output_dir = Path(output_dir) if output_dir is not None else config.REPORTS_DIR
        # active_filters: {"patient_id": [...], "date": [...]} — empty means no filter
        self.active_filters: dict[str, list[str]] = active_filters or {}

    def generate(self, filename: str = "qc_report.html") -> Path:
        """Write full HTML report to disk and return the output path."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        html = self._render_html()
        out_path = self.output_dir / filename
        out_path.write_text(html, encoding="utf-8")
        logger.info("QC report written to %s", out_path)
        return out_path

    def save_summary_csv(self, filename: str = "qc_summary.csv") -> Path:
        """Save enriched metrics DataFrame as CSV for downstream consumption."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / filename
        self.df.to_csv(out_path, index=False)
        logger.info("QC summary CSV written to %s", out_path)
        return out_path

    # ── HTML assembly ────────────────────────────────────────────────────

    def _render_html(self) -> str:
        """Assemble the complete HTML document."""
        patient_ids = sorted(self.df["patient_id"].dropna().unique()) if "patient_id" in self.df.columns else []
        patient_sections = "\n".join(self._render_patient_section(pid) for pid in patient_ids)
        return _HTML_TEMPLATE.format(
            css=_render_css(),
            timestamp=datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            filter_banner=self._render_filter_banner(),
            summary_table=self._render_summary_table(),
            trial_type_overview=self._render_trial_type_overview(),
            patient_sections=patient_sections,
        )

    def _render_filter_banner(self) -> str:
        """Render an info banner describing which filters are active.

        Returns an empty string when no filters are applied (full report).
        """
        if not self.active_filters:
            return ""
        parts: list[str] = []
        if self.active_filters.get("patient_id"):
            ids = ", ".join(sorted(self.active_filters["patient_id"]))
            parts.append(f"<strong>Patient&thinsp;ID:</strong>&ensp;{ids}")
        if self.active_filters.get("date"):
            dates = ", ".join(sorted(self.active_filters["date"]))
            parts.append(f"<strong>Session&thinsp;Date:</strong>&ensp;{dates}")
        if not parts:
            return ""
        detail = "&emsp;|&emsp;".join(parts)
        return (
            f'<div class="filter-banner">'
            f'<span class="filter-icon">&#x1F50D;</span> '
            f"<strong>Filtered Report</strong>&ensp;&mdash;&ensp;{detail}"
            f"</div>"
        )

    def _render_summary_table(self) -> str:
        """Build the executive summary HTML table."""
        if self.df.empty:
            return "<p>No QC data available.</p>"

        n_patients = self.df["patient_id"].nunique() if "patient_id" in self.df.columns else 0
        n_sessions = (
            len(self.df.groupby(["patient_id", "date"])) if {"patient_id", "date"}.issubset(self.df.columns) else 0
        )
        total_epochs = int(pd.to_numeric(self.df.get("n_epochs_total", 0), errors="coerce").sum())
        total_dropped = int(pd.to_numeric(self.df.get("n_epochs_dropped", 0), errors="coerce").sum())
        overall_drop_rate = (total_dropped / total_epochs * 100) if total_epochs > 0 else 0.0
        mean_snr = self.df["snr_db"].mean() if "snr_db" in self.df.columns else float("nan")
        # Count usable session rows
        n_usable = int(self.df["is_usable"].sum()) if "is_usable" in self.df.columns else "N/A"
        # Total estimated recording time
        total_rec = (
            self.df["estimated_recording_min"].sum() if "estimated_recording_min" in self.df.columns else float("nan")
        )

        rows = [
            ("Total Patients", str(n_patients)),
            ("Total Sessions", str(n_sessions)),
            ("Usable Sessions (≥50% retention)", str(n_usable)),
            ("Total Epochs", str(total_epochs)),
            ("Total Dropped", str(total_dropped)),
            ("Overall Drop Rate", f"{overall_drop_rate:.1f}%"),
            ("Mean SNR (dB)", f"{mean_snr:.2f}" if np.isfinite(mean_snr) else "N/A"),
            ("Est. Total Recording Time (min)", f"{total_rec:.1f}" if np.isfinite(total_rec) else "N/A"),
        ]
        return _build_html_table(["Metric", "Value"], rows)

    def _render_patient_section(self, patient_id: str) -> str:
        """Render a per-patient HTML section with trial-type breakdown."""
        pdf = self.df[self.df["patient_id"] == patient_id]
        if pdf.empty:
            return ""

        dates = sorted(pdf["date"].dropna().unique())
        session_label = ", ".join(str(d) for d in dates)

        header = f'<h2 id="{_safe_id(patient_id)}">{patient_id}</h2>\n<p>Sessions: {session_label}</p>\n'
        table = self._render_trial_type_table(pdf)
        ica_section = self._render_ica_summary(pdf)
        notes_section = self._render_session_notes(pdf)
        return f'<div class="patient-section">\n{header}{table}\n{ica_section}\n{notes_section}\n</div>\n'

    def _render_trial_type_table(self, pdf: pd.DataFrame) -> str:
        """Build per-trial-type metrics table for one patient."""
        cols = [
            "trial_type",
            "n_epochs_total",
            "n_epochs_kept",
            "retention_rate",
            "ptp_uv_p50",
            "ptp_uv_p95",
            "snr_db",
            "estimated_recording_min",
            "is_usable",
        ]
        available = [c for c in cols if c in pdf.columns]
        headers = _FRIENDLY_COL_NAMES

        # Vectorised: apply _format_cell per column, then zip rows into lists
        formatted = {c: pdf[c].apply(lambda v, col=c: _format_cell(col, v)) for c in available}
        rows = list(zip(*(formatted[c] for c in available)))

        table_html = _build_html_table([headers.get(c, c) for c in available], rows)
        footnote = (
            '<dl class="legend-box">'
            "<dt>SNR&thinsp;(dB)</dt>"
            "<dd>Relative signal quality&thinsp;=&thinsp;20&middot;log<sub>10</sub>"
            "(median&thinsp;PTP&thinsp;/&thinsp;P95&thinsp;PTP). "
            "Higher (less negative) = cleaner signal.<br>"
            '<span class="legend-range legend-excellent">&gt;&minus;3&thinsp;dB &mdash; Excellent</span>'
            '&ensp;<span class="legend-range legend-good">&minus;3 to &minus;6&thinsp;dB &mdash; Good</span>'
            '&ensp;<span class="legend-range legend-ok">&minus;6 to &minus;10&thinsp;dB &mdash; Acceptable</span>'
            '&ensp;<span class="legend-range legend-bad">&lt;&minus;10&thinsp;dB &mdash; Noisy (review)</span>'
            "</dd>"
            "<dt>&#x2705; Usable</dt>"
            "<dd>Session retained &ge;50&thinsp;% of epochs <em>and</em> kept &ge;1 epoch.</dd>"
            "<dt>&#x274c; Not Usable</dt>"
            "<dd>&gt;50&thinsp;% epochs dropped <em>or</em> zero epochs kept &mdash; "
            "exclude from group-level analysis.</dd>"
            "</dl>"
        )
        return table_html + footnote

    def _render_ica_summary(self, pdf: pd.DataFrame) -> str:
        """Render ICA component exclusion summary for a patient.

        ICA is fitted on the whole EDF (session-level), so component counts
        are identical across trial types within the same session.  We therefore
        deduplicate by session date and show one row per session rather than
        repeating the same numbers for every trial type.
        """
        ica_cols = [c for c in pdf.columns if c.startswith("ica_")]
        if not ica_cols:
            return ""

        # Deduplicate by session date — ICA is session-level, not trial-level
        ica_cols = [
            "date",
            "ica_classification_method",
            "ica_n_components_excluded",
            "ica_n_eog",
            "ica_n_ecg",
            "ica_n_muscle",
        ]
        available_ica = [c for c in ica_cols if c in pdf.columns]
        deduped = pdf[available_ica].drop_duplicates(subset="date") if "date" in pdf.columns else pdf[available_ica]

        int_cols = ["ica_n_components_excluded", "ica_n_eog", "ica_n_ecg", "ica_n_muscle"]
        for col in int_cols:
            if col in deduped.columns:
                deduped = deduped.copy()
                deduped[col] = pd.to_numeric(deduped[col], errors="coerce").fillna(0).astype(int)
        if all(c in deduped.columns for c in ["ica_n_components_excluded", "ica_n_eog", "ica_n_ecg", "ica_n_muscle"]):
            deduped = deduped.copy()
            deduped["ica_n_other"] = (
                deduped["ica_n_components_excluded"]
                - deduped["ica_n_eog"]
                - deduped["ica_n_ecg"]
                - deduped["ica_n_muscle"]
            ).clip(lower=0)

        col_map = {
            "date": "Session Date",
            "ica_classification_method": "Method",
            "ica_n_components_excluded": "Excluded",
            "ica_n_eog": "EOG",
            "ica_n_ecg": "ECG",
            "ica_n_muscle": "Muscle",
            "ica_n_other": "Others",
        }
        display_cols = [c for c in col_map if c in deduped.columns]
        rows = [list(map(str, r)) for r in deduped[display_cols].itertuples(index=False)]
        headers = [col_map[c] for c in display_cols]
        return "<h3>Artifact Rejection Summary</h3>\n" + _build_html_table(headers, rows)

    def _render_trial_type_overview(self) -> str:
        """Cross-patient trial-type comparison table."""
        if self.df.empty or "trial_type" not in self.df.columns:
            return "<p>No trial type data available.</p>"

        summary = _aggregate_summary(self.df, group_cols=["trial_type"])
        if summary.empty:
            return "<p>No aggregated data available.</p>"

        cols = [
            "trial_type",
            "n_sessions",
            "n_usable",
            "n_epochs_total",
            "n_epochs_dropped",
            "n_epochs_kept",
            "mean_drop_rate",
            "mean_retention_rate",
            "mean_snr_db",
            "total_recording_min",
        ]
        available = [c for c in cols if c in summary.columns]
        headers_map = {
            "trial_type": "Trial Type",
            "n_sessions": "Sessions",
            "n_usable": "Usable",
            "n_epochs_total": "Total Epochs",
            "n_epochs_dropped": "Dropped",
            "n_epochs_kept": "Kept",
            "mean_drop_rate": "Avg Drop Rate",
            "mean_retention_rate": "Avg Retention",
            "mean_snr_db": "Avg SNR (dB)",
            "total_recording_min": "Est. Recording (min)",
        }
        rows: list[list[str]] = [[_format_cell(c, row.get(c)) for c in available] for _, row in summary.iterrows()]
        return _build_html_table([headers_map.get(c, c) for c in available], rows)

    def _render_session_notes(self, pdf: pd.DataFrame) -> str:
        """Render ENG-03 session notes as a small HTML list.

        The ``notes`` column stores a JSON array of strings (e.g.
        ``["noisy session", "audio on left side only"]``).
        """
        if "notes" not in pdf.columns:
            return ""
        parsed = pdf["notes"].apply(_parse_notes_json)
        all_notes = [note for note_list in parsed for note in note_list]
        if not all_notes:
            return ""
        items_html = "\n".join(f"<li>{note}</li>" for note in all_notes)
        return f"<h3>Session Notes</h3>\n<ul>\n{items_html}\n</ul>"


# ── Convenience function ─────────────────────────────────────────────────────


def generate_qc_report(
    qc_dir: Path | None = None,
    output_dir: Path | None = None,
    patient_ids: list[str] | None = None,
    dates: list[str] | None = None,
) -> Path:
    """One-call entry point: collect QC data -> compute metrics -> generate HTML.

    Args:
        qc_dir:      Directory containing ``eng03_qc.parquet`` files.  Defaults
                     to ``config.QC_DIR``.
        output_dir:  Directory to write the HTML report and CSV.  Defaults to
                     ``config.REPORTS_DIR``.
        patient_ids: Optional list of patient IDs to include (e.g.
                     ``["CON008"]``).  When *None* all patients are included.
        dates:       Optional list of session dates to include in ``YYYY-MM-DD``
                     format (e.g. ``["2025-08-14"]``).  When *None* all sessions
                     are included.

    Returns:
        Path to the generated HTML report.
    """
    collector = QCDataCollector(qc_dir=qc_dir)
    qc_df = collector.load_all()

    # --- apply filters ---
    active_filters: dict[str, list[str]] = {}
    if patient_ids:
        active_filters["patient_id"] = list(patient_ids)
        mask = (
            qc_df["patient_id"].isin(patient_ids)
            if "patient_id" in qc_df.columns
            else pd.Series(True, index=qc_df.index)
        )
        qc_df = qc_df[mask].copy()
        logger.info("Filter applied — patient_id in %s. Rows remaining: %d", patient_ids, len(qc_df))
    if dates:
        active_filters["date"] = list(dates)
        mask = qc_df["date"].astype(str).isin(dates) if "date" in qc_df.columns else pd.Series(True, index=qc_df.index)
        qc_df = qc_df[mask].copy()
        logger.info("Filter applied — date in %s. Rows remaining: %d", dates, len(qc_df))

    calculator = QCMetricsCalculator(qc_df)
    metrics_df = calculator.compute_all_metrics()
    generator = QCReportGenerator(metrics_df, output_dir=output_dir, active_filters=active_filters)
    report_path = generator.generate()
    generator.save_summary_csv()
    return report_path


# ── HTML helpers ─────────────────────────────────────────────────────────────

_FRIENDLY_COL_NAMES: dict[str, str] = {
    "trial_type": "Trial Type",
    "n_epochs_total": "Total Epochs",
    "n_epochs_dropped": "Dropped",
    "n_epochs_kept": "Kept",
    "drop_rate": "Drop Rate",
    "retention_rate": "Retention Rate",
    "reject_ptp_threshold_uv": "Reject Threshold (\u00b5V)",
    "ptp_uv_p50": "Median PTP (\u00b5V)",
    "ptp_uv_p95": "P95 PTP (\u00b5V)",
    "snr_db": "SNR (dB)",
    "estimated_recording_min": "Est. Duration (min)",
    "is_usable": "Usable",
}


def _parse_notes_json(raw: Any) -> list[str]:
    """Parse a single ``notes`` cell into a list of non-empty strings.

    Handles JSON-encoded arrays, plain strings, ``None``, and ``NaN`` safely.
    """
    if raw is None or (isinstance(raw, float) and not np.isfinite(raw)):
        return []
    try:
        items = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(items, list):
            return [str(n) for n in items if str(n).strip()]
        return [str(items)] if str(items).strip() else []
    except (json.JSONDecodeError, TypeError):
        return [str(raw)] if str(raw).strip() else []


def _format_cell(col: str, val: Any) -> str:
    """Format a cell value for HTML display."""
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        return "N/A"
    if col in ("drop_rate", "mean_drop_rate"):
        return f"{float(val) * 100:.1f}%"
    if col in ("retention_rate", "mean_retention_rate"):
        return f"{float(val) * 100:.1f}%"
    if col in ("snr_db", "mean_snr_db"):
        return f"{float(val):.2f}"
    if col in ("reject_ptp_threshold_uv", "ptp_uv_p50", "ptp_uv_p95", "mean_ptp_uv"):
        return f"{float(val):.1f}"
    if col in ("estimated_recording_min", "total_recording_min"):
        return f"{float(val):.1f}"
    if col == "is_usable":
        return "✅" if val else "❌"
    if isinstance(val, float):
        return f"{val:.2f}"
    return str(val)


def _safe_id(text: str) -> str:
    """Convert text to a safe HTML id attribute."""
    return text.replace(" ", "-").replace("/", "-").lower()


def _build_html_table(headers: list[str], rows: list[Any]) -> str:
    """Build a simple HTML <table> from headers and rows."""
    parts = ["<table>\n<thead>\n<tr>"]
    for h in headers:
        parts.append(f"<th>{h}</th>")
    parts.append("</tr>\n</thead>\n<tbody>\n")
    for row in rows:
        parts.append("<tr>")
        cells = row if isinstance(row, (list, tuple)) else list(row)
        for cell in cells:
            parts.append(f"<td>{cell}</td>")
        parts.append("</tr>\n")
    parts.append("</tbody>\n</table>")
    return "".join(parts)


def _render_css() -> str:
    """Inline CSS for a clean, readable report."""
    # UW purple: #4b2e83  (https://www.washington.edu/brand/graphic-elements/primary-color-palette/)
    return """
    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        margin: 2rem auto;
        max-width: 1400px;
        padding: 0 1.5rem;
        color: #1a1a1a;
        background: #fafafa;
        line-height: 1.5;
    }
    h1 { color: #4b2e83; border-bottom: 2px solid #4b2e83; padding-bottom: 0.3rem; }
    h2 { color: #4b2e83; margin-top: 2rem; }
    h3 { color: #6a4caa; margin-top: 1rem; }
    .table-wrapper {
        width: 100%;
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
        margin: 1rem 0;
    }
    table {
        border-collapse: collapse;
        min-width: 600px;
        width: 100%;
        background: #fff;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        font-size: 0.875rem;
    }
    th, td {
        padding: 0.45rem 0.75rem;
        text-align: left;
        border-bottom: 1px solid #e0e0e0;
        white-space: nowrap;
    }
    th { background: #4b2e83; color: #fff; font-weight: 600; }
    tbody tr:nth-child(even) { background: #f5f0fb; }
    tbody tr:hover { background: #e8dff5; }
    .patient-section {
        background: #fff;
        padding: 1.5rem;
        margin: 1.5rem 0;
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        overflow-x: auto;
    }
    .filter-banner {
        background: #fdf6e3; border-left: 4px solid #856404;
        padding: 0.6rem 1rem; margin: 0.5rem 0 1.5rem; border-radius: 4px;
        font-size: 0.9rem; color: #333; display: flex; align-items: center; gap: 0.5rem;
    }
    .filter-icon { font-size: 1.1rem; }
    .summary-section, .overview-section { margin-bottom: 2rem; }
    .legend-box { font-size: 0.8rem; color: #333; background: #f9f7fd; border-left: 3px solid #4b2e83;
                  padding: 0.6rem 1rem; margin: 0.5rem 0 1.2rem; line-height: 1.7; }
    .legend-box dt { font-weight: 700; color: #4b2e83; margin-top: 0.35rem; }
    .legend-box dd { margin: 0 0 0 1rem; }
    .legend-range { display: inline-block; padding: 0.1rem 0.45rem; border-radius: 3px;
                    font-size: 0.75rem; font-weight: 600; margin-top: 0.25rem; }
    .legend-excellent { background: #d4edda; color: #155724; }
    .legend-good      { background: #cce5ff; color: #004085; }
    .legend-ok        { background: #fff3cd; color: #856404; }
    .legend-bad       { background: #f8d7da; color: #721c24; }
    footer { margin-top: 3rem; color: #888; font-size: 0.85rem; border-top: 1px solid #ddd; padding-top: 1rem; }
    """


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EEG QC Report — AwakenAI Pipeline</title>
<style>{css}</style>
</head>
<body>
<h1>EEG Quality Control Report</h1>
<p>Generated: {timestamp}</p>
{filter_banner}

<div class="summary-section">
<h2>Executive Summary</h2>
{summary_table}
</div>

<div class="overview-section">
<h2>Trial Type Overview</h2>
{trial_type_overview}
</div>

<h2>Per-Patient Details</h2>
{patient_sections}

<footer>
<p>Report generated by ENG-06 QC Report Pipeline — AwakenAI Capstone</p>
</footer>
</body>
</html>"""
