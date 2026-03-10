"""
Base Pipeline ABC

All analysis pipelines (Command Following, Oddball, Language) inherit from this class.
Defines the template method pattern: load → preprocess → analyze.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

import pandas as pd

from src.data_loading import UnifiedDataLoader

logger = logging.getLogger(__name__)


class BasePipeline(ABC):
    """Base class for all EEG analysis pipelines.

    Template method ``run()`` enforces: load → preprocess → analyze.
    Subclasses implement the individual steps.
    """

    def __init__(self, loader: Optional[UnifiedDataLoader] = None):
        self.loader = loader or UnifiedDataLoader()
        self.patient_id: Optional[str] = None
        self.session_id: Optional[str] = None
        self.aligned_events: Optional[pd.DataFrame] = None
        self.results: Optional[pd.DataFrame] = None
        self._qc_metrics: dict[str, Any] = {}

    def run(self, patient_id: str, session_id: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """Template method: load → preprocess → analyze."""
        self.patient_id = patient_id
        self.session_id = session_id

        events = self.loader.load_aligned_events(patient_id)
        if session_id:
            events = events[events["session_id"] == session_id]
        self.aligned_events = events
        self.load()
        self.preprocess()
        self.results = self.analyze(**kwargs)
        return self.results

    @abstractmethod
    def load(self) -> Any:
        """Load pipeline-specific data (epochs, pairs, etc.)."""
        ...

    @abstractmethod
    def preprocess(self) -> Any:
        """Apply pipeline-specific preprocessing (filtering, channel selection, etc.)."""
        ...

    @abstractmethod
    def analyze(self, **kwargs) -> pd.DataFrame:
        """
        Run the core analysis and return results as a DataFrame.

        Delegation pattern for ``analyze()``:
        Each subclass keeps its domain-specific analysis method
        (e.g. ``calculate_erd``, ``_compute_itpc``) and implements
        ``analyze()`` as a one-line delegation::

            def analyze(self, **kwargs):
                return self.calculate_erd(**kwargs)

        This provides a consistent entry-point for CLI orchestration
        while preserving descriptive, domain-readable method names.
        """
        ...

    @abstractmethod
    def generate_summary(self) -> Any:
        """Generate a classification/summary dict from results."""
        ...

    @property
    def qc_metrics(self) -> Any:
        """Standardized QC metrics populated during analyze(). Consumed by QCReport."""
        return self._qc_metrics
