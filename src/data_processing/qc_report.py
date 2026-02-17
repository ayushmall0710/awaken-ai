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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data_loading import config

logger = logging.getLogger(__name__)


# ── QCDataCollector ──────────────────────────────────────────────────────────


class QCDataCollector:
    """Discover and load ENG-03 QC parquet files from the processed QC directory."""

    def __init__(self, qc_dir: Optional[Path] = None) -> None:
        self.qc_dir = Path(qc_dir) if qc_dir is not None else config.QC_DIR

    def discover_qc_files(self) -> List[Path]:
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

    def get_available_sessions(self) -> List[Tuple[str, str]]:
        """Return a list of (patient_id, date) pairs that have QC data."""
        files = self.discover_qc_files()
        sessions: List[Tuple[str, str]] = []
        for f in files:
            date = f.parent.name
            patient_id = f.parent.parent.name
            sessions.append((patient_id, date))
        return sorted(sessions)


def _empty_qc_dataframe() -> pd.DataFrame:
    """Return an empty DataFrame with the standard ENG-03 QC columns."""
    cols = [
        "patient_id", "date", "trial_type", "window_sec",
        "reject_ptp_percentile", "reject_ptp_threshold_uv",
        "n_epochs_total", "n_epochs_dropped", "n_epochs_kept",
        "drop_reason", "ica", "notes",
        "ptp_uv_p50", "ptp_uv_p95", "ptp_uv_p99", "ptp_uv_max", "ptp_uv_mean",
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
        parsed = df["ica"].apply(_parse_single_ica_json) if "ica" in df.columns else pd.DataFrame()
        if not parsed.empty:
            parsed_df = pd.DataFrame(parsed.tolist(), index=df.index)
            for col in parsed_df.columns:
                df[col] = parsed_df[col]
        return df

    def compute_all_metrics(self) -> pd.DataFrame:
        """Chain all metric computations and return a fully enriched DataFrame."""
        self._df = self.compute_drop_rates()
        self._df = self.compute_snr_estimates()
        self._df = self.parse_ica_summaries()
        return self._df.copy()

    def summary_by_patient(self) -> pd.DataFrame:
        """Aggregated statistics grouped by ``patient_id``."""
        df = self.compute_all_metrics()
        return _aggregate_summary(df, group_cols=["patient_id"])

    def summary_by_trial_type(self) -> pd.DataFrame:
        """Aggregated statistics grouped by ``trial_type``."""
        df = self.compute_all_metrics()
        return _aggregate_summary(df, group_cols=["trial_type"])


def _parse_single_ica_json(raw_json: Any) -> Dict[str, Any]:
    """Parse one ICA JSON string into flat metric fields.

    When ICLabel was used, the per-type component lists (``eog_components``, etc.)
    are empty because ICLabel puts everything into ``excluded``.  In that case we
    derive per-type counts from ``iclabel_labels`` + ``excluded`` indices.
    """
    defaults: Dict[str, Any] = {
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
_ICLABEL_TO_CATEGORY: Dict[str, str] = {
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
    d: Dict[str, Any],
    excluded: List[int],
) -> Dict[str, int]:
    """Count excluded components per artifact type.

    Strategy:
    1. If ``iclabel_labels`` exists and the per-type lists are empty, derive
       counts from the labels of the *excluded* component indices.
    2. Otherwise fall back to the per-type lists stored by ENG-03.
    """
    counts = {"eog": 0, "ecg": 0, "muscle": 0, "line_noise": 0, "channel_noise": 0}
    labels = d.get("iclabel_labels")

    has_per_type = any(
        len(d.get(k, [])) > 0
        for k in ("eog_components", "ecg_components", "muscle_components",
                   "line_noise_components", "channel_noise_components")
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


def _aggregate_summary(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """Build a summary table with key aggregate statistics."""
    if df.empty:
        return pd.DataFrame()

    numeric_cols = {
        "n_epochs_total": "sum",
        "n_epochs_dropped": "sum",
        "n_epochs_kept": "sum",
    }
    agg_dict: Dict[str, Any] = {}
    for col, func in numeric_cols.items():
        if col in df.columns:
            agg_dict[col] = pd.NamedAgg(column=col, aggfunc=func)

    if "drop_rate" in df.columns:
        agg_dict["mean_drop_rate"] = pd.NamedAgg(column="drop_rate", aggfunc="mean")
    if "snr_db" in df.columns:
        agg_dict["mean_snr_db"] = pd.NamedAgg(column="snr_db", aggfunc="mean")
    if "ptp_uv_mean" in df.columns:
        agg_dict["mean_ptp_uv"] = pd.NamedAgg(column="ptp_uv_mean", aggfunc="mean")

    agg_dict["n_sessions"] = pd.NamedAgg(column=group_cols[0], aggfunc="count")
    return df.groupby(group_cols, dropna=False).agg(**agg_dict).reset_index()


# ── QCReportGenerator ────────────────────────────────────────────────────────


class QCReportGenerator:
    """Render an HTML QC dashboard and optional CSV summary from enriched QC data."""

    def __init__(
        self,
        metrics_df: pd.DataFrame,
        output_dir: Optional[Path] = None,
    ) -> None:
        self.df = metrics_df.copy()
        self.output_dir = Path(output_dir) if output_dir is not None else config.REPORTS_DIR

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
            summary_table=self._render_summary_table(),
            trial_type_overview=self._render_trial_type_overview(),
            patient_sections=patient_sections,
        )

    def _render_summary_table(self) -> str:
        """Build the executive summary HTML table."""
        if self.df.empty:
            return "<p>No QC data available.</p>"

        n_patients = self.df["patient_id"].nunique() if "patient_id" in self.df.columns else 0
        n_sessions = len(self.df.groupby(["patient_id", "date"])) if {"patient_id", "date"}.issubset(self.df.columns) else 0
        total_epochs = int(pd.to_numeric(self.df.get("n_epochs_total", 0), errors="coerce").sum())
        total_dropped = int(pd.to_numeric(self.df.get("n_epochs_dropped", 0), errors="coerce").sum())
        overall_drop_rate = (total_dropped / total_epochs * 100) if total_epochs > 0 else 0.0
        mean_snr = self.df["snr_db"].mean() if "snr_db" in self.df.columns else float("nan")

        rows = [
            ("Total Patients", str(n_patients)),
            ("Total Sessions", str(n_sessions)),
            ("Total Epochs", str(total_epochs)),
            ("Total Dropped", str(total_dropped)),
            ("Overall Drop Rate", f"{overall_drop_rate:.1f}%"),
            ("Mean SNR (dB)", f"{mean_snr:.2f}" if np.isfinite(mean_snr) else "N/A"),
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
        return f'<div class="patient-section">\n{header}{table}\n{ica_section}\n</div>\n'

    def _render_trial_type_table(self, pdf: pd.DataFrame) -> str:
        """Build per-trial-type metrics table for one patient."""
        cols = ["trial_type", "n_epochs_total", "n_epochs_dropped", "n_epochs_kept", "drop_rate", "reject_ptp_threshold_uv", "snr_db"]
        available = [c for c in cols if c in pdf.columns]
        headers = _FRIENDLY_COL_NAMES.copy()

        rows: List[List[str]] = []
        for _, row in pdf.iterrows():
            cells = []
            for c in available:
                val = row.get(c)
                cells.append(_format_cell(c, val))
            rows.append(cells)

        return _build_html_table([headers.get(c, c) for c in available], rows)

    def _render_ica_summary(self, pdf: pd.DataFrame) -> str:
        """Render ICA component exclusion summary for a patient."""
        ica_cols = [c for c in pdf.columns if c.startswith("ica_")]
        if not ica_cols:
            return ""

        rows: List[List[str]] = []
        for _, row in pdf.iterrows():
            tt = str(row.get("trial_type", ""))
            method = str(row.get("ica_classification_method", "N/A"))
            n_excl = int(row.get("ica_n_components_excluded", 0))
            n_eog = int(row.get("ica_n_eog", 0))
            n_ecg = int(row.get("ica_n_ecg", 0))
            n_muscle = int(row.get("ica_n_muscle", 0))
            rows.append([tt, method, str(n_excl), str(n_eog), str(n_ecg), str(n_muscle)])

        headers = ["Trial Type", "Method", "Excluded", "EOG", "ECG", "Muscle"]
        return "<h3>ICA Component Summary</h3>\n" + _build_html_table(headers, rows)

    def _render_trial_type_overview(self) -> str:
        """Cross-patient trial-type comparison table."""
        if self.df.empty or "trial_type" not in self.df.columns:
            return "<p>No trial type data available.</p>"

        summary = _aggregate_summary(self.df, group_cols=["trial_type"])
        if summary.empty:
            return "<p>No aggregated data available.</p>"

        cols = ["trial_type", "n_sessions", "n_epochs_total", "n_epochs_dropped", "n_epochs_kept", "mean_drop_rate", "mean_snr_db"]
        available = [c for c in cols if c in summary.columns]
        headers_map = {
            "trial_type": "Trial Type",
            "n_sessions": "Sessions",
            "n_epochs_total": "Total Epochs",
            "n_epochs_dropped": "Dropped",
            "n_epochs_kept": "Kept",
            "mean_drop_rate": "Avg Drop Rate",
            "mean_snr_db": "Avg SNR (dB)",
        }
        rows: List[List[str]] = []
        for _, row in summary.iterrows():
            cells = [_format_cell(c, row.get(c)) for c in available]
            rows.append(cells)

        return _build_html_table([headers_map.get(c, c) for c in available], rows)


# ── Convenience function ─────────────────────────────────────────────────────


def generate_qc_report(
    qc_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """One-call entry point: collect QC data -> compute metrics -> generate HTML.

    Returns the path to the generated HTML report.
    """
    collector = QCDataCollector(qc_dir=qc_dir)
    qc_df = collector.load_all()
    calculator = QCMetricsCalculator(qc_df)
    metrics_df = calculator.compute_all_metrics()
    generator = QCReportGenerator(metrics_df, output_dir=output_dir)
    report_path = generator.generate()
    generator.save_summary_csv()
    return report_path


# ── HTML helpers ─────────────────────────────────────────────────────────────

_FRIENDLY_COL_NAMES: Dict[str, str] = {
    "trial_type": "Trial Type",
    "n_epochs_total": "Total Epochs",
    "n_epochs_dropped": "Dropped",
    "n_epochs_kept": "Kept",
    "drop_rate": "Drop Rate",
    "reject_ptp_threshold_uv": "PTP Threshold (\u00b5V)",
    "snr_db": "SNR (dB)",
}


def _format_cell(col: str, val: Any) -> str:
    """Format a cell value for HTML display."""
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        return "N/A"
    if col == "drop_rate" or col == "mean_drop_rate":
        return f"{float(val) * 100:.1f}%"
    if col in ("snr_db", "mean_snr_db"):
        return f"{float(val):.2f}"
    if col in ("reject_ptp_threshold_uv", "mean_ptp_uv"):
        return f"{float(val):.1f}"
    if isinstance(val, float):
        return f"{val:.2f}"
    return str(val)


def _safe_id(text: str) -> str:
    """Convert text to a safe HTML id attribute."""
    return text.replace(" ", "-").replace("/", "-").lower()


def _build_html_table(headers: List[str], rows: List[Any]) -> str:
    """Build a simple HTML <table> from headers and rows."""
    parts = ['<table>\n<thead>\n<tr>']
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
    """Minimal inline CSS for a clean, readable report."""
    return """
    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        margin: 2rem auto;
        max-width: 1100px;
        color: #1a1a1a;
        background: #fafafa;
        line-height: 1.6;
    }
    h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 0.3rem; }
    h2 { color: #2c3e50; margin-top: 2rem; }
    h3 { color: #555; margin-top: 1rem; }
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 1rem 0;
        background: #fff;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    th, td { padding: 0.6rem 1rem; text-align: left; border-bottom: 1px solid #e0e0e0; }
    th { background: #3498db; color: #fff; font-weight: 600; }
    tr:hover { background: #f5f5f5; }
    .patient-section {
        background: #fff;
        padding: 1.5rem;
        margin: 1.5rem 0;
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .summary-section, .overview-section { margin-bottom: 2rem; }
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
