"""
EEG Data Loader Module

Provides the EEGDataLoader class for loading and managing EEG recordings (EDF)
and their associated stimulus timing files (CSV).

This module serves as the foundation for the EEG data processing pipeline,
enabling downstream tasks like timestamp alignment and epoch extraction.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List, Union
import warnings

import pandas as pd
import numpy as np
import mne

from . import config


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)


class EEGDataLoadingError(Exception):
    """Custom exception for EEG data loading errors"""
    pass


class EEGDataLoader:
    """
    Loads and manages EEG recordings (EDF files) and stimulus timing data (CSV files).
    
    This class provides a unified interface for accessing patient EEG data and associated
    experimental metadata, supporting trial-level access and data validation.
    
    Attributes:
        patient_id (str): Patient identifier (e.g., 'CON008')
        edf_path (Path): Path to EDF file
        stimulus_csv_path (Path): Path to stimulus timing CSV
        raw (mne.io.Raw): MNE Raw object containing EDF data
        stimulus_df (pd.DataFrame): DataFrame with stimulus timing information
        edf_loaded (bool): Whether EDF has been loaded
        csv_loaded (bool): Whether CSV has been loaded
    
    Example:
        >>> # Explicit paths (full control)
        >>> loader = EEGDataLoader(
        ...     patient_id="CON008",
        ...     edf_path="data/EEG Project Data/EEG/edf/CON008_clipped.EDF",
        ...     stimulus_csv_path="data/EEG Project Data/EEG/CON008_2025-08-14_stimulus_results.csv"
        ... )
        >>> loader.load()
        
        >>> # Auto-discovery (convenience)
        >>> loader = EEGDataLoader.from_patient_id("CON008", use_clipped=True)
        >>> trials = loader.get_trials(trial_type='language')
        >>> print(f"Found {len(trials)} language trials")
    """
    
    def __init__(
        self,
        patient_id: str,
        edf_path: Union[str, Path],
        stimulus_csv_path: Union[str, Path],
        preload: bool = True,
        verbose: bool = False
    ):
        """
        Initialize EEGDataLoader with patient data file paths.
        
        Args:
            patient_id: Patient identifier (e.g., 'CON008', 'CON009')
            edf_path: Path to EDF file (raw or clipped)
            stimulus_csv_path: Path to stimulus timing CSV file
            preload: Whether to preload EDF data into memory (default: True)
            verbose: Whether to show detailed loading messages (default: False)
        
        Raises:
            EEGDataLoadingError: If files don't exist or paths are invalid
        """
        self.patient_id = patient_id
        self.edf_path = Path(edf_path)
        self.stimulus_csv_path = Path(stimulus_csv_path)
        self.preload = preload
        self.verbose = verbose
        
        # Data containers
        self.raw: Optional[mne.io.Raw] = None
        self.stimulus_df: Optional[pd.DataFrame] = None
        
        # Status flags
        self.edf_loaded = False
        self.csv_loaded = False
        
        # Validate paths on initialization
        self._validate_paths()
        
        # Set MNE logging level
        if not verbose:
            mne.set_log_level('WARNING')
    
    @classmethod
    def from_patient_id(
        cls,
        patient_id: str,
        data_root: Optional[Union[str, Path]] = None,
        use_clipped: bool = True,
        date: Optional[str] = None,
        preload: bool = True,
        verbose: bool = False
    ) -> 'EEGDataLoader':
        """
        Factory method to create loader by auto-discovering file paths.
        
        Automatically finds EDF and CSV files for a given patient ID, avoiding
        the need to manually construct paths.
        
        Args:
            patient_id: Patient identifier (e.g., 'CON008', 'CON009')
            data_root: Root directory for data. If None, uses config.LOCAL_DATA_ROOT
            use_clipped: Whether to prefer clipped EDF files (default: True)
            date: Specific date for CSV file (YYYY-MM-DD). If None, auto-discovers
            preload: Whether to preload EDF data into memory (default: True)
            verbose: Whether to show detailed loading messages (default: False)
        
        Returns:
            EEGDataLoader instance
        
        Raises:
            EEGDataLoadingError: If files cannot be found
        
        Example:
            >>> # Simple usage
            >>> loader = EEGDataLoader.from_patient_id("CON008")
            
            >>> # With options
            >>> loader = EEGDataLoader.from_patient_id(
            ...     "CON009",
            ...     use_clipped=True,
            ...     date="2025-08-26"
            ... )
        """
        if data_root is None:
            data_root = config.LOCAL_DATA_ROOT
        data_root = Path(data_root)
        
        # Find EDF file
        edf_dir = data_root / "EEG Project Data" / "EEG" / "edf"
        edf_path = cls._find_edf(edf_dir, patient_id, use_clipped)
        
        # Find CSV file
        csv_dir = data_root / "EEG Project Data" / "EEG"
        stimulus_csv_path = cls._find_csv(csv_dir, patient_id, date)
        
        logger.info(f"Auto-discovered files for {patient_id}:")
        logger.info(f"  EDF: {edf_path.name}")
        logger.info(f"  CSV: {stimulus_csv_path.name}")
        
        return cls(
            patient_id=patient_id,
            edf_path=edf_path,
            stimulus_csv_path=stimulus_csv_path,
            preload=preload,
            verbose=verbose
        )
    
    @staticmethod
    def _find_edf(edf_dir: Path, patient_id: str, use_clipped: bool) -> Path:
        """Find EDF file for patient."""
        # Try clipped first if preferred
        if use_clipped:
            clipped_path = edf_dir / f"{patient_id}_clipped.EDF"
            if clipped_path.exists():
                return clipped_path
        
        # Try raw EDF
        raw_path = edf_dir / f"{patient_id}.EDF"
        if raw_path.exists():
            return raw_path
        
        # Try in old stimulus software subfolder
        old_dir = edf_dir / "old stimulus software"
        if old_dir.exists():
            clipped_old = old_dir / f"{patient_id}_clipped.EDF"
            if clipped_old.exists():
                return clipped_old
            
            raw_old = old_dir / f"{patient_id}.EDF"
            if raw_old.exists():
                return raw_old
        
        raise EEGDataLoadingError(
            f"Could not find EDF file for patient {patient_id} in {edf_dir}\n"
            f"Tried: {patient_id}_clipped.EDF, {patient_id}.EDF, and old stimulus software folder"
        )
    
    @staticmethod
    def _find_csv(csv_dir: Path, patient_id: str, date: Optional[str]) -> Path:
        """Find stimulus CSV file for patient."""
        # If date specified, try exact match first
        if date:
            exact_path = csv_dir / f"{patient_id}_{date}_stimulus_results.csv"
            if exact_path.exists():
                return exact_path
        
        # Search for any matching CSV
        pattern = f"{patient_id}_*_stimulus_results.csv"
        matches = list(csv_dir.glob(pattern))
        
        if not matches:
            # Try stimuli_record subfolder
            stimuli_record_dir = csv_dir / "stimuli_record"
            if stimuli_record_dir.exists():
                matches = list(stimuli_record_dir.glob(pattern))
        
        if not matches:
            raise EEGDataLoadingError(
                f"Could not find stimulus CSV for patient {patient_id} in {csv_dir}\n"
                f"Pattern: {pattern}"
            )
        
        if len(matches) > 1:
            if date:
                raise EEGDataLoadingError(
                    f"Found multiple CSV files for {patient_id}:\n" +
                    "\n".join(f"  - {m.name}" for m in matches) +
                    f"\nSpecified date '{date}' not found."
                )
            else:
                warnings.warn(
                    f"Found multiple CSV files for {patient_id}:\n" +
                    "\n".join(f"  - {m.name}" for m in matches) +
                    f"\nUsing most recent: {matches[-1].name}\n"
                    f"Specify 'date' parameter to choose a specific file."
                )
        
        return matches[-1]  # Return most recent if multiple
    
    def _validate_paths(self) -> None:
        """Validate that specified file paths exist."""
        if not self.edf_path.exists():
            raise EEGDataLoadingError(
                f"EDF file not found: {self.edf_path}"
            )
        
        if not self.stimulus_csv_path.exists():
            raise EEGDataLoadingError(
                f"Stimulus CSV file not found: {self.stimulus_csv_path}"
            )
        
        logger.info(f"Initialized EEGDataLoader for patient {self.patient_id}")
        logger.info(f"  EDF: {self.edf_path.name}")
        logger.info(f"  CSV: {self.stimulus_csv_path.name}")
    
    def load_edf(self) -> mne.io.Raw:
        """
        Load EDF file using MNE-Python.
        
        Returns:
            MNE Raw object containing EEG data
        
        Raises:
            EEGDataLoadingError: If EDF loading fails
        """
        try:
            logger.info(f"Loading EDF file: {self.edf_path.name}")
            
            # Load EDF with MNE
            self.raw = mne.io.read_raw_edf(
                self.edf_path,
                preload=self.preload,
                verbose=self.verbose
            )
            
            self.edf_loaded = True
            
            # Log basic info
            duration_min = self.raw.times[-1] / 60
            logger.info(f"  Channels: {len(self.raw.ch_names)}")
            logger.info(f"  Sampling rate: {self.raw.info['sfreq']} Hz")
            logger.info(f"  Duration: {duration_min:.1f} minutes")
            
            return self.raw
            
        except Exception as e:
            raise EEGDataLoadingError(
                f"Failed to load EDF file {self.edf_path}: {str(e)}"
            )
    
    def load_stimulus_timing(self) -> pd.DataFrame:
        """
        Load and validate stimulus timing CSV file.
        
        Returns:
            DataFrame with stimulus timing information
        
        Raises:
            EEGDataLoadingError: If CSV loading or validation fails
        """
        try:
            logger.info(f"Loading stimulus CSV: {self.stimulus_csv_path.name}")
            
            # Load CSV
            self.stimulus_df = pd.read_csv(self.stimulus_csv_path)
            
            # Validate schema
            self._validate_csv_schema()
            
            self.csv_loaded = True
            
            # Log basic info
            logger.info(f"  Total trials: {len(self.stimulus_df)}")
            trial_counts = self.stimulus_df['trial_type'].value_counts()
            for trial_type, count in trial_counts.items():
                logger.info(f"    {trial_type}: {count}")
            
            return self.stimulus_df
            
        except Exception as e:
            raise EEGDataLoadingError(
                f"Failed to load stimulus CSV {self.stimulus_csv_path}: {str(e)}"
            )
    
    def _validate_csv_schema(self) -> None:
        """
        Validate that CSV has expected columns and data types.
        
        Raises:
            EEGDataLoadingError: If schema validation fails
        """
        required_columns = [
            'patient_id', 'date', 'trial_type', 'sentences',
            'start_time', 'end_time', 'duration'
        ]
        
        missing_columns = [col for col in required_columns if col not in self.stimulus_df.columns]
        
        if missing_columns:
            raise EEGDataLoadingError(
                f"CSV missing required columns: {missing_columns}"
            )
        
        # Validate patient_id matches
        csv_patient_ids = self.stimulus_df['patient_id'].unique()
        if len(csv_patient_ids) > 1:
            warnings.warn(
                f"CSV contains multiple patient IDs: {csv_patient_ids}. "
                f"Expected only {self.patient_id}"
            )
        elif csv_patient_ids[0] != self.patient_id:
            warnings.warn(
                f"CSV patient_id ({csv_patient_ids[0]}) doesn't match "
                f"specified patient_id ({self.patient_id})"
            )
    
    def load(self) -> 'EEGDataLoader':
        """
        Load both EDF and CSV files.
        
        Returns:
            Self for method chaining
        """
        if not self.edf_loaded:
            self.load_edf()
        
        if not self.csv_loaded:
            self.load_stimulus_timing()
        
        # Perform validation after both are loaded
        self.validate()
        
        return self
    
    def validate(self) -> Dict[str, bool]:
        """
        Validate data integrity and alignment feasibility.
        
        Returns:
            Dictionary with validation results
        
        Raises:
            EEGDataLoadingError: If validation fails critically
        """
        if not self.edf_loaded or not self.csv_loaded:
            raise EEGDataLoadingError(
                "Cannot validate: both EDF and CSV must be loaded first"
            )
        
        validation_results = {
            'timestamp_alignment': True,
            'duration_match': True,
            'complete_trials': True
        }
        
        # Get EDF duration in seconds
        edf_duration = self.raw.times[-1]
        edf_start_unix = self.raw.info['meas_date'].timestamp() if self.raw.info['meas_date'] else None
        
        # Check if CSV timestamps fall within EDF duration
        if edf_start_unix is not None:
            csv_min_time = self.stimulus_df['start_time'].min()
            csv_max_time = self.stimulus_df['end_time'].max()
            
            # Calculate relative times
            csv_start_relative = csv_min_time - edf_start_unix
            csv_end_relative = csv_max_time - edf_start_unix
            
            if csv_start_relative < 0:
                warnings.warn(
                    f"CSV trials start before EDF recording "
                    f"(offset: {csv_start_relative:.1f}s)"
                )
                validation_results['timestamp_alignment'] = False
            
            if csv_end_relative > edf_duration:
                warnings.warn(
                    f"CSV trials extend beyond EDF recording "
                    f"(EDF duration: {edf_duration:.1f}s, "
                    f"CSV end: {csv_end_relative:.1f}s)"
                )
                validation_results['timestamp_alignment'] = False
        else:
            warnings.warn(
                "EDF file missing measurement date timestamp. "
                "Cannot validate timestamp alignment. "
                "Will rely on DC audio channel alignment (ENG-02)."
            )
            validation_results['timestamp_alignment'] = None
        
        # Check for incomplete trials (missing data)
        null_counts = self.stimulus_df[['start_time', 'end_time', 'duration']].isnull().sum()
        if null_counts.any():
            warnings.warn(f"Found trials with missing timing data:\n{null_counts}")
            validation_results['complete_trials'] = False
        
        logger.info("Validation complete")
        return validation_results
    
    def get_trials(self, trial_type: Optional[str] = None) -> pd.DataFrame:
        """
        Get trials, optionally filtered by trial type.
        
        Args:
            trial_type: Filter by trial type (e.g., 'language', 'oddball+p', 'left_command+p')
                       If None, returns all trials.
        
        Returns:
            DataFrame with trial metadata
        
        Raises:
            EEGDataLoadingError: If CSV not loaded
        """
        if not self.csv_loaded:
            raise EEGDataLoadingError("CSV must be loaded first. Call load() or load_stimulus_timing()")
        
        if trial_type is None:
            return self.stimulus_df.copy()
        
        filtered = self.stimulus_df[self.stimulus_df['trial_type'] == trial_type].copy()
        
        if len(filtered) == 0:
            available_types = self.stimulus_df['trial_type'].unique()
            warnings.warn(
                f"No trials found for type '{trial_type}'. "
                f"Available types: {list(available_types)}"
            )
        
        return filtered
    
    def get_trial(self, trial_idx: int) -> pd.Series:
        """
        Get specific trial by index.
        
        Args:
            trial_idx: Index of trial in stimulus DataFrame
        
        Returns:
            Series with trial metadata
        
        Raises:
            EEGDataLoadingError: If CSV not loaded or index invalid
        """
        if not self.csv_loaded:
            raise EEGDataLoadingError("CSV must be loaded first. Call load() or load_stimulus_timing()")
        
        if trial_idx < 0 or trial_idx >= len(self.stimulus_df):
            raise EEGDataLoadingError(
                f"Trial index {trial_idx} out of range [0, {len(self.stimulus_df)-1}]"
            )
        
        return self.stimulus_df.iloc[trial_idx]
    
    def get_eeg_info(self) -> Dict:
        """
        Get EDF metadata and channel information.
        
        Returns:
            Dictionary with EEG recording metadata
        
        Raises:
            EEGDataLoadingError: If EDF not loaded
        """
        if not self.edf_loaded:
            raise EEGDataLoadingError("EDF must be loaded first. Call load() or load_edf()")
        
        info = {
            'patient_id': self.patient_id,
            'n_channels': len(self.raw.ch_names),
            'channel_names': self.raw.ch_names,
            'sampling_rate': self.raw.info['sfreq'],
            'duration_seconds': self.raw.times[-1],
            'duration_minutes': self.raw.times[-1] / 60,
            'measurement_date': self.raw.info['meas_date'],
            'channel_types': self.raw.get_channel_types(),
        }
        
        return info
    
    def get_trial_types(self) -> List[str]:
        """
        Get list of unique trial types in the dataset.
        
        Returns:
            List of trial type strings
        
        Raises:
            EEGDataLoadingError: If CSV not loaded
        """
        if not self.csv_loaded:
            raise EEGDataLoadingError("CSV must be loaded first. Call load() or load_stimulus_timing()")
        
        return sorted(self.stimulus_df['trial_type'].unique().tolist())
    
    def __repr__(self) -> str:
        """String representation of loader."""
        status = []
        if self.edf_loaded:
            status.append(f"EDF loaded ({len(self.raw.ch_names)} channels)")
        else:
            status.append("EDF not loaded")
        
        if self.csv_loaded:
            status.append(f"CSV loaded ({len(self.stimulus_df)} trials)")
        else:
            status.append("CSV not loaded")
        
        status_str = ", ".join(status)
        return f"EEGDataLoader(patient={self.patient_id}, {status_str})"
