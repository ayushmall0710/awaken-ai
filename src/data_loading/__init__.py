"""
Data Loading Module

Runtime imports are deferred via ``__getattr__`` so that
``from src.data_loading import UnifiedDataLoader`` works without eagerly
importing heavy dependencies like MNE.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from src.data_loading.inventory import DataInventory
    from src.data_loading.patient_data import PatientData
    from src.data_loading.unified_data_loader import UnifiedDataLoader, UnifiedDataLoadingError

__all__ = [
    "UnifiedDataLoader",
    "UnifiedDataLoadingError",
    "PatientData",
    "DataInventory",
]


def __getattr__(name: str):
    if name in ("UnifiedDataLoader", "UnifiedDataLoadingError"):
        from src.data_loading.unified_data_loader import UnifiedDataLoader, UnifiedDataLoadingError

        _map = {"UnifiedDataLoader": UnifiedDataLoader, "UnifiedDataLoadingError": UnifiedDataLoadingError}
        return _map[name]
    if name == "PatientData":
        from src.data_loading.patient_data import PatientData

        return PatientData
    if name == "DataInventory":
        from src.data_loading.inventory import DataInventory

        return DataInventory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
