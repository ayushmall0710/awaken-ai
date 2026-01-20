"""
Data Loading Module

This module provides functionality for loading and processing EEG data:
- EEGDataLoader: Base class for loading EDF files and CSV stimulus logs (ENG-01)
- TimestampAligner: DC channel-based timestamp alignment (ENG-02)
"""

from .eeg_data_loader import EEGDataLoader
from .timestamp_alignment import TimestampAligner

__all__ = ['EEGDataLoader', 'TimestampAligner']
