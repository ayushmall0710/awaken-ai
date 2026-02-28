"""QC Report runner — dispatches to ENG-06 generate_qc_report."""

from __future__ import annotations

from pathlib import Path


def run(
    patient_ids: list[str] | None,
    session: str | None,
    output_dir: Path | None,
) -> Path:
    """Generate an HTML QC report and return its path.

    Args:
        patient_ids: Patients to include; ``None`` means all patients.
        session:     Session date (YYYY-MM-DD) to restrict to; ``None`` means all sessions.
        output_dir:  Where to write the report; ``None`` falls back to config.REPORTS_DIR.

    Returns:
        Path to the generated ``qc_report.html`` file.
    """
    from src.data_processing.qc_report import generate_qc_report

    dates = [session] if session else None

    return generate_qc_report(
        output_dir=output_dir,
        patient_ids=patient_ids,
        dates=dates,
    )
