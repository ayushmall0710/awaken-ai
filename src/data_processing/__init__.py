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
    from src.data_processing.timestamp_aligner import TimestampAligner

__all__ = ["TimestampAligner", "ArtifactRejector"]


def __getattr__(name: str):
    if name == "ArtifactRejector":
        from src.data_processing.artifact_rejection import ArtifactRejector

        return ArtifactRejector
    if name == "TimestampAligner":
        from src.data_processing.timestamp_aligner import TimestampAligner

        return TimestampAligner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
