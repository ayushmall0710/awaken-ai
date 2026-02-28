"""
data_processing package.

Important: keep imports here lightweight.

Some submodules (e.g., ENG-02/ENG-03) depend on heavy optional runtime deps like `mne`.
Tests and lightweight ETL steps (DAT-03) should still be importable without requiring
those heavy deps at package import time.

Runtime imports are deferred via ``__getattr__`` so that
``from src.data_processing import ArtifactRejector`` works at runtime without
eagerly pulling in MNE.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover – keeps type-checkers & IDEs happy
    from src.data_processing.artifact_rejection import ArtifactRejector
    from src.data_processing.qc_report import (
        QCDataCollector,
        QCMetricsCalculator,
        QCReportGenerator,
        generate_qc_report,
    )
    from src.data_processing.timestamp_aligner import TimestampAligner

__all__ = [
    "TimestampAligner",
    "ArtifactRejector",
    "QCDataCollector",
    "QCMetricsCalculator",
    "QCReportGenerator",
    "generate_qc_report",
]

# Lazy-import mapping: attribute name -> (module_name, attribute_name)
_LAZY_IMPORTS = {
    "ArtifactRejector": "artifact_rejection",
    "TimestampAligner": "timestamp_aligner",
    "QCDataCollector": "qc_report",
    "QCMetricsCalculator": "qc_report",
    "QCReportGenerator": "qc_report",
    "generate_qc_report": "qc_report",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(f"src.data_processing.{_LAZY_IMPORTS[name]}")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
