"""
Patient Data View Module

Provides the PatientData class for focused single-patient workflows.
Acts as a lightweight view over the unified data for one patient.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List, Callable
import warnings

import pandas as pd
import mne


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)


class PatientData:
    """
    Focused interface for working with a single patient's data.
    
    Provides convenient access to trial data and lazy-loads EDF when needed.
    Acts as a lightweight view over the main unified DataFrame.
    
    Attributes:
        patient_id (str): Patient identifier
        trials_df (pd.DataFrame): All trials for this patient
        raw (mne.io.Raw): Lazy-loaded EEG data (None until accessed)
    
    Example:
        >>> patient = loader.get_patient('CON008')
        >>> language_trials = patient.get_trials_by_type('language')
        >>> raw = patient.raw  # Triggers EDF loading
        >>> print(f"Channels: {len(raw.ch_names)}")
    """
    
    def __init__(
        self,
        patient_id: str,
        trials_df: pd.DataFrame,
        edf_loader_func: Callable,
        data_root: Path
    ):
        """
        Initialize PatientData view.
        
        Args:
            patient_id: Patient identifier
            trials_df: DataFrame with trials for this patient (will be copied)
            edf_loader_func: Function to call for loading EDF (from UnifiedDataLoader)
            data_root: Root directory for data files
        """
        self.patient_id = patient_id
        self.trials_df = trials_df.copy()  # Defensive copy
        self._edf_loader = edf_loader_func
        self._data_root = data_root
        self._raw = None
    
    @property
    def raw(self) -> mne.io.Raw:
        """
        Lazy-load and return EEG data.
        
        EDF file is only loaded on first access to this property.
        Subsequent accesses return the cached Raw object.
        
        Returns:
            MNE Raw object containing EEG data
        
        Raises:
            Exception: If EDF loading fails
        """
        if self._raw is None:
            logger.info(f"Lazy loading EDF for {self.patient_id}")
            self._raw = self._edf_loader(self.patient_id, use_clipped=True)
        return self._raw
    
    def get_trials_by_type(self, trial_type: str) -> pd.DataFrame:
        """
        Get trials of specific type for this patient.
        
        Args:
            trial_type: Type of trial (e.g., 'language', 'oddball', 'left_command')
        
        Returns:
            DataFrame with matching trials (defensive copy)
        
        Example:
            >>> oddball_trials = patient.get_trials_by_type('oddball')
        """
        filtered = self.trials_df[self.trials_df['trial_type'] == trial_type]
        
        if len(filtered) == 0:
            available_types = self.get_trial_types()
            warnings.warn(
                f"No trials found for type '{trial_type}' in patient {self.patient_id}. "
                f"Available types: {available_types}"
            )
        
        return filtered.copy()
    
    def get_trial(self, trial_idx: int) -> pd.Series:
        """
        Get specific trial by index (within this patient's trials).
        
        Args:
            trial_idx: Index of trial in this patient's trial DataFrame
        
        Returns:
            Series with trial metadata
        
        Raises:
            IndexError: If trial_idx out of range
        
        Example:
            >>> first_trial = patient.get_trial(0)
            >>> print(f"Trial type: {first_trial['trial_type']}")
        """
        if trial_idx < 0 or trial_idx >= len(self.trials_df):
            raise IndexError(
                f"Trial index {trial_idx} out of range [0, {len(self.trials_df)-1}] "
                f"for patient {self.patient_id}"
            )
        
        return self.trials_df.iloc[trial_idx]
    
    def get_trial_types(self) -> List[str]:
        """
        Get list of trial types available for this patient.
        
        Returns:
            Sorted list of trial type strings
        """
        return sorted(self.trials_df['trial_type'].unique().tolist())
    
    def get_eeg_info(self) -> Dict:
        """
        Get EEG metadata for this patient.
        
        Note: This triggers EDF loading if not already loaded.
        
        Returns:
            Dictionary with EEG recording metadata:
            - patient_id: Patient identifier
            - n_channels: Number of EEG channels
            - channel_names: List of channel names
            - sampling_rate: Sampling rate in Hz
            - duration_seconds: Recording duration in seconds
            - duration_minutes: Recording duration in minutes
            - measurement_date: Recording start date
            - channel_types: List of channel types
        
        Example:
            >>> info = patient.get_eeg_info()
            >>> print(f"Sampling rate: {info['sampling_rate']} Hz")
        """
        raw = self.raw  # Triggers loading if needed
        
        return {
            'patient_id': self.patient_id,
            'n_channels': len(raw.ch_names),
            'channel_names': raw.ch_names,
            'sampling_rate': raw.info['sfreq'],
            'duration_seconds': raw.times[-1],
            'duration_minutes': raw.times[-1] / 60,
            'measurement_date': raw.info['meas_date'],
            'channel_types': raw.get_channel_types(),
        }
    
    def validate(self) -> Dict[str, bool]:
        """
        Validate data quality for this patient.
        
        Returns:
            Dictionary with validation results:
            - has_trials: Patient has trial data
            - edf_loadable: EDF file can be loaded
            - timestamps_complete: No missing timestamp values
            - sentences_valid: Sentences have correct format
            - timestamp_alignment: CSV times within EDF duration (if EDF loaded)
        
        Example:
            >>> validation = patient.validate()
            >>> if not validation['timestamp_alignment']:
            ...     print("Need DC channel alignment (ENG-02)")
        """
        validation = {
            'has_trials': len(self.trials_df) > 0,
            'edf_loadable': False,
            'timestamps_complete': True,
            'sentences_valid': True,
            'timestamp_alignment': None  # None = couldn't check
        }
        
        # Check timestamp completeness
        null_counts = self.trials_df[['start_time', 'end_time', 'duration']].isnull().sum()
        if null_counts.any():
            validation['timestamps_complete'] = False
            warnings.warn(
                f"Patient {self.patient_id} has trials with missing timing data:\n{null_counts}"
            )
        
        # Check sentences validity
        for sentences in self.trials_df['sentences']:
            if not isinstance(sentences, list):
                validation['sentences_valid'] = False
                warnings.warn(
                    f"Patient {self.patient_id} has invalid sentence format "
                    f"(expected List[Dict], got {type(sentences)})"
                )
                break
            if sentences and not all(isinstance(s, dict) for s in sentences):
                validation['sentences_valid'] = False
                warnings.warn(
                    f"Patient {self.patient_id} has invalid sentence items "
                    f"(expected dict elements)"
                )
                break
        
        # Try to load EDF
        try:
            raw = self.raw
            validation['edf_loadable'] = True
            
            # Check timestamp alignment if EDF loaded
            if raw.info['meas_date'] is not None:
                edf_duration = raw.times[-1]
                edf_start_unix = raw.info['meas_date'].timestamp()
                
                csv_min_time = self.trials_df['start_time'].min()
                csv_max_time = self.trials_df['end_time'].max()
                
                csv_start_relative = csv_min_time - edf_start_unix
                csv_end_relative = csv_max_time - edf_start_unix
                
                # Check alignment
                if csv_start_relative < 0:
                    validation['timestamp_alignment'] = False
                    warnings.warn(
                        f"Patient {self.patient_id}: CSV trials start before EDF recording "
                        f"(offset: {csv_start_relative:.1f}s)"
                    )
                elif csv_end_relative > edf_duration:
                    validation['timestamp_alignment'] = False
                    warnings.warn(
                        f"Patient {self.patient_id}: CSV trials extend beyond EDF recording "
                        f"(EDF duration: {edf_duration:.1f}s, CSV end: {csv_end_relative:.1f}s)"
                    )
                else:
                    validation['timestamp_alignment'] = True
            else:
                warnings.warn(
                    f"Patient {self.patient_id}: EDF missing measurement date. "
                    f"Cannot validate timestamp alignment. "
                    f"Will rely on DC audio channel alignment (ENG-02)."
                )
        except Exception as e:
            warnings.warn(f"Patient {self.patient_id}: Failed to load EDF: {e}")
        
        return validation
    
    def __repr__(self) -> str:
        """String representation of PatientData."""
        edf_status = "EDF loaded" if self._raw is not None else "EDF not loaded"
        return (
            f"PatientData("
            f"patient={self.patient_id}, "
            f"{len(self.trials_df)} trials, "
            f"{edf_status}"
            f")"
        )
