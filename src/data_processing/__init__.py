"""
data_processing package.

Important: keep imports here lightweight.

Some submodules (e.g., ENG-02/ENG-03) depend on heavy optional runtime deps like `mne`.
Tests and lightweight ETL steps (DAT-03) should still be importable without requiring
those heavy deps at package import time.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from src.data_processing.artifact_rejection import ArtifactRejector
    from src.data_processing.timestamp_aligner import TimestampAligner

__all__ = ["TimestampAligner", "ArtifactRejector"]
