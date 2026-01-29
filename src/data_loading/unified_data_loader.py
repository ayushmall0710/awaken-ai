"""Unified data loader for Parquet file from DAT-03 with cross-patient and single-patient access."""

import logging
from pathlib import Path
from typing import Optional, Dict, List, Union
from functools import lru_cache
import warnings

import pandas as pd
import numpy as np
import mne

from . import config
from .patient_data import PatientData


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)


class UnifiedDataLoadingError(Exception):
    """Custom exception for unified data loading errors"""
    pass


class UnifiedDataLoader:
    """
    Loads unified Parquet file with all patients' trial data.
    Provides cross-patient queries and single-patient views with LRU-cached EDF loading.
    """
    
    def __init__(
        self,
        parquet_path: Union[str, Path],
        data_root: Optional[Union[str, Path]] = None,
        edf_cache_size: int = 3,
        verbose: bool = False
    ):
        self.parquet_path = Path(parquet_path)
        self.data_root = Path(data_root) if data_root else config.LOCAL_DATA_ROOT
        self.edf_cache_size = edf_cache_size
        self.verbose = verbose
        
        # Validate Parquet file exists
        if not self.parquet_path.exists():
            raise UnifiedDataLoadingError(
                f"Parquet file not found: {self.parquet_path}"
            )
        
        # Load Parquet file into memory
        logger.info(f"Loading unified data from {self.parquet_path.name}")
        try:
            self.trials_df = pd.read_parquet(self.parquet_path)
        except Exception as e:
            raise UnifiedDataLoadingError(
                f"Failed to load Parquet file {self.parquet_path}: {str(e)}"
            )
        
        # Validate schema
        self.validate_schema()
        
        # Set up LRU cache for EDF loading (min size 5 for multi-session support)
        self._load_edf_cached = lru_cache(maxsize=max(5, edf_cache_size))(self._load_edf_uncached)
        
        # Set MNE logging level
        if not verbose:
            mne.set_log_level('WARNING')
        
        logger.info(f"Loaded {len(self.trials_df)} trials from {len(self.get_patient_ids())} patients")
    
    # ==================== Cross-Patient Access ====================
    
    def get_all_trials(self) -> pd.DataFrame:
        """Get all trials from all patients."""
        return self.trials_df.copy()
    
    def get_trials_by_type(
        self,
        trial_type: str,
        patient_ids: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get trials of specific type, optionally filtered by patient(s).
        
        Args:
            trial_type: Type of trial (e.g., 'language', 'oddball', 'left_command')
            patient_ids: Optional list of patient IDs to filter. If None, returns all patients.
        
        Returns:
            DataFrame with matching trials (defensive copy)
        
        Example:
            >>> # All language trials across all patients
            >>> lang_trials = loader.get_trials_by_type('language')
            
            >>> # Oddball trials for specific patients
            >>> oddball = loader.get_trials_by_type('oddball', ['CON008', 'CON009'])
        """
        filtered = self.trials_df[self.trials_df['trial_type'] == trial_type]
        
        if patient_ids is not None:
            filtered = filtered[filtered['patient_id'].isin(patient_ids)]
        
        if len(filtered) == 0:
            warnings.warn(
                f"No trials found for type '{trial_type}'" +
                (f" in patients {patient_ids}" if patient_ids else "")
            )
        
        return filtered.copy()
    
    def get_patient_ids(self) -> List[str]:
        """Get sorted list of unique patient IDs."""
        return sorted(self.trials_df['patient_id'].unique().tolist())
    
    def get_patient_sessions(self, patient_id: str) -> List[str]:
        """Get sorted list of session dates for a patient."""
        patient_trials = self.trials_df[self.trials_df['patient_id'] == patient_id]
        
        if len(patient_trials) == 0:
            raise UnifiedDataLoadingError(
                f"Patient '{patient_id}' not found"
            )
        
        sessions = sorted(patient_trials['date'].unique().tolist())
        return sessions
    
    def get_trial_types(self) -> List[str]:
        """Get sorted list of unique trial types."""
        return sorted(self.trials_df['trial_type'].unique().tolist())
    
    def get_trial_summary(self) -> pd.DataFrame:
        """
        Get summary statistics of trials by patient and type.
        
        Returns:
            DataFrame with columns: patient_id, trial_type, count
        
        Example:
            >>> summary = loader.get_trial_summary()
            >>> print(summary)
            patient_id  trial_type      count
            CON008      language        72
            CON008      oddball         5
            CON009      language        89
            ...
        """
        summary = self.trials_df.groupby(['patient_id', 'trial_type']).size().reset_index(name='count')
        return summary.sort_values(['patient_id', 'trial_type'])
    
    # ==================== Single-Patient Access ====================
    
    def get_patient_trials(self, patient_id: str) -> pd.DataFrame:
        """
        Get all trials for a specific patient.
        
        Args:
            patient_id: Patient identifier (e.g., 'CON008')
        
        Returns:
            DataFrame with patient's trials (defensive copy)
        
        Raises:
            UnifiedDataLoadingError: If patient ID not found
        """
        patient_trials = self.trials_df[self.trials_df['patient_id'] == patient_id]
        
        if len(patient_trials) == 0:
            available = self.get_patient_ids()
            raise UnifiedDataLoadingError(
                f"Patient '{patient_id}' not found in dataset. "
                f"Available patients: {available}"
            )
        
        return patient_trials.copy()
    
    def get_patient(self, patient_id: str) -> PatientData:
        """Get PatientData view for single-patient workflows."""
        patient_trials = self.get_patient_trials(patient_id)
        
        return PatientData(
            patient_id=patient_id,
            trials_df=patient_trials,
            edf_loader_func=self.load_edf,
            data_root=self.data_root,
            find_edf_fn=self._find_edf
        )
    
    # ==================== EDF Management ====================
    
    def load_edf(
        self, 
        patient_id: Optional[str] = None,
        date: Optional[str] = None,
        filepath: Optional[Union[str, Path]] = None,
        use_clipped: bool = True
    ) -> Union[mne.io.Raw, Dict[str, mne.io.Raw]]:
        """
        Load EDF file(s). Returns Raw for single/specified session, Dict[date, Raw] for multi-session.
        
        Args:
            patient_id: Patient ID (exclusive with filepath)
            date: Session date. If None, returns all sessions for multi-session patients.
            filepath: Direct file path (exclusive with patient_id, no caching)
            use_clipped: Prefer clipped EDF files
        """
        # Validate mutually exclusive parameters
        if filepath is not None and patient_id is not None:
            raise ValueError("Cannot specify both filepath and patient_id")
        
        if filepath is None and patient_id is None:
            raise ValueError("Must specify either filepath or patient_id")
        
        # Direct filepath loading (bypass validation and caching)
        if filepath is not None:
            filepath = Path(filepath)
            if not filepath.exists():
                raise UnifiedDataLoadingError(f"EDF file not found: {filepath}")
            
            logger.info(f"Loading EDF from direct path: {filepath.name}")
            try:
                return mne.io.read_raw_edf(str(filepath), preload=False, verbose=self.verbose)
            except Exception as e:
                raise UnifiedDataLoadingError(f"Failed to load EDF from {filepath}: {str(e)}")
        
        # Patient-based loading (normal flow)
        # If date not specified, load based on session count
        if date is None:
            sessions = self.get_patient_sessions(patient_id)
            
            if len(sessions) == 0:
                raise UnifiedDataLoadingError(
                    f"Patient {patient_id} has no trial data"
                )
            elif len(sessions) == 1:
                # Single session - return Raw directly
                logger.info(f"Loading single session for {patient_id}: {sessions[0]}")
                return self._load_edf_cached(patient_id, sessions[0], use_clipped)
            else:
                # Multiple sessions - return Dict with all sessions
                logger.info(
                    f"Patient {patient_id} has {len(sessions)} sessions. "
                    f"Loading all: {sessions}"
                )
                return {
                    session: self._load_edf_cached(patient_id, session, use_clipped)
                    for session in sessions
                }
        else:
            # Specific date requested - validate and return single Raw
            sessions = self.get_patient_sessions(patient_id)
            if date not in sessions:
                raise UnifiedDataLoadingError(
                    f"Date '{date}' not found for patient {patient_id}. "
                    f"Available sessions: {sessions}"
                )
            
            logger.info(f"Loading specific session for {patient_id}: {date}")
            return self._load_edf_cached(patient_id, date, use_clipped)
    
    def _load_edf_uncached(self, patient_id: str, date: str, use_clipped: bool) -> mne.io.Raw:
        edf_path = self._find_edf(patient_id, date, use_clipped)
        
        logger.info(f"Loading EDF for {patient_id} ({date}): {edf_path.name}")
        
        try:
            raw = mne.io.read_raw_edf(
                edf_path,
                preload=True,
                verbose=self.verbose
            )
            return raw
        except Exception as e:
            raise UnifiedDataLoadingError(
                f"Failed to load EDF for {patient_id} on {date} from {edf_path}: {str(e)}"
            )
    
    def _find_edf(self, patient_id: str, date: str, use_clipped: bool) -> Path:
        """Auto-discover EDF file for patient session (tries date-specific, then fallbacks)."""
        edf_dir = self.data_root / "EEG Project Data" / "EEG" / "edf"
        
        # Format date for filename (remove hyphens)
        date_str = date.replace('-', '')
        
        # Try date-specific files first
        if use_clipped:
            # Try: CON005_20250214_clipped.EDF
            dated_clipped = edf_dir / f"{patient_id}_{date_str}_clipped.EDF"
            if dated_clipped.exists():
                return dated_clipped
        
        # Try: CON005_20250214.EDF
        dated_raw = edf_dir / f"{patient_id}_{date_str}.EDF"
        if dated_raw.exists():
            return dated_raw
        
        # Fallback: Check if this is a single-session patient
        sessions = self.get_patient_sessions(patient_id)
        if len(sessions) == 1:
            # Only one session, try non-dated files
            if use_clipped:
                clipped_path = edf_dir / f"{patient_id}_clipped.EDF"
                if clipped_path.exists():
                    return clipped_path
            
            raw_path = edf_dir / f"{patient_id}.EDF"
            if raw_path.exists():
                return raw_path
        
        # Try old stimulus software subfolder
        old_dir = edf_dir / "old stimulus software"
        if old_dir.exists():
            if use_clipped:
                dated_clipped_old = old_dir / f"{patient_id}_{date_str}_clipped.EDF"
                if dated_clipped_old.exists():
                    return dated_clipped_old
                
                # Fallback for single session
                if len(sessions) == 1:
                    clipped_old = old_dir / f"{patient_id}_clipped.EDF"
                    if clipped_old.exists():
                        return clipped_old
            
            dated_raw_old = old_dir / f"{patient_id}_{date_str}.EDF"
            if dated_raw_old.exists():
                return dated_raw_old
            
            # Fallback for single session
            if len(sessions) == 1:
                raw_old = old_dir / f"{patient_id}.EDF"
                if raw_old.exists():
                    return raw_old
        
        raise UnifiedDataLoadingError(
            f"Could not find EDF file for patient {patient_id} on {date} in {edf_dir}\n"
            f"Tried: {patient_id}_{date_str}_clipped.EDF, {patient_id}_{date_str}.EDF, "
            f"and old stimulus software folder"
        )
    
    def get_cached_edfs(self) -> List[str]:
        """
        Get list of patient IDs that have EDFs currently in cache.
        
        Returns:
            List of patient IDs with cached EDFs
        """
        cache_info = self._load_edf_cached.cache_info()
        # Note: Can't directly inspect LRU cache contents, so return cache stats instead
        return {
            'hits': cache_info.hits,
            'misses': cache_info.misses,
            'size': cache_info.currsize,
            'maxsize': cache_info.maxsize
        }
    
    def clear_edf_cache(self):
        """Clear all cached EDF files from memory."""
        self._load_edf_cached.cache_clear()
        logger.info("EDF cache cleared")
    
    # ==================== Validation ====================
    
    def validate_schema(self) -> Dict[str, bool]:
        """
        Validate Parquet file has required schema.
        
        Returns:
            Dictionary with validation results
        
        Raises:
            UnifiedDataLoadingError: If schema validation fails critically
        """
        required_columns = [
            'patient_id', 'date', 'trial_type', 'sentences',
            'start_time', 'end_time', 'duration', 'source_file'
        ]
        
        missing_columns = [col for col in required_columns if col not in self.trials_df.columns]
        
        if missing_columns:
            raise UnifiedDataLoadingError(
                f"Parquet missing required columns: {missing_columns}\n"
                f"Available columns: {list(self.trials_df.columns)}"
            )
        
        validation_results = {
            'has_required_columns': True,
            'has_data': len(self.trials_df) > 0,
            'timestamps_valid': True,
            'sentences_valid': True
        }
        
        # Check if we have data
        if len(self.trials_df) == 0:
            warnings.warn("Parquet file is empty (no trials)")
            validation_results['has_data'] = False
        
        # Check timestamp data types
        try:
            if self.trials_df['start_time'].dtype not in [np.float64, np.float32]:
                warnings.warn("start_time has unexpected data type")
                validation_results['timestamps_valid'] = False
            if self.trials_df['end_time'].dtype not in [np.float64, np.float32]:
                warnings.warn("end_time has unexpected data type")
                validation_results['timestamps_valid'] = False
            if self.trials_df['duration'].dtype not in [np.float64, np.float32]:
                warnings.warn("duration has unexpected data type")
                validation_results['timestamps_valid'] = False
        except KeyError as e:
            warnings.warn(f"Missing timestamp column: {e}")
            validation_results['timestamps_valid'] = False
        
        # Check sentences is list type
        try:
            first_sentences = self.trials_df['sentences'].iloc[0]
            if not isinstance(first_sentences, list):
                warnings.warn(f"Sentences column has unexpected type: {type(first_sentences)}")
                validation_results['sentences_valid'] = False
        except (IndexError, KeyError):
            pass
        
        logger.info("Schema validation passed")
        return validation_results
    
    def validate_patient(self, patient_id: str) -> Dict[str, bool]:
        """
        Validate data quality for a specific patient.
        
        Args:
            patient_id: Patient identifier
        
        Returns:
            Dictionary with validation results:
            - edf_exists: EDF file can be found
            - edf_loadable: EDF file can be loaded
            - has_trials: Patient has trial data
            - timestamps_complete: No missing timestamp values
            - sentences_valid: Sentences have correct List[Dict] format
        """
        validation = {
            'edf_exists': False,
            'edf_loadable': False,
            'has_trials': False,
            'timestamps_complete': True,
            'sentences_valid': True
        }
        
        # Check if patient has trials
        patient_trials = self.trials_df[self.trials_df['patient_id'] == patient_id]
        validation['has_trials'] = len(patient_trials) > 0
        
        if not validation['has_trials']:
            return validation
        
        # Check EDF file existence (try all sessions, mark valid if any exists)
        sessions = self.get_patient_sessions(patient_id)
        edf_found = False
        for session_date in sessions:
            try:
                edf_path = self._find_edf(patient_id, session_date, use_clipped=True)
                validation['edf_exists'] = True
                edf_found = True
                break
            except UnifiedDataLoadingError:
                continue
        
        if not edf_found:
            return validation
        
        # Check EDF loadability
        try:
            self.load_edf(patient_id)
            validation['edf_loadable'] = True
        except UnifiedDataLoadingError:
            return validation
        
        # Check timestamp completeness
        null_counts = patient_trials[['start_time', 'end_time', 'duration']].isnull().sum()
        if null_counts.any():
            validation['timestamps_complete'] = False
        
        # Check sentences validity
        for sentences in patient_trials['sentences']:
            if not isinstance(sentences, list):
                validation['sentences_valid'] = False
                break
            if sentences and not all(isinstance(s, dict) for s in sentences):
                validation['sentences_valid'] = False
                break
        
        return validation
    
    def validate_all_patients(self) -> pd.DataFrame:
        """
        Validate all patients and return summary DataFrame.
        
        Returns:
            DataFrame with validation results per patient
            Columns: patient_id, edf_exists, edf_loadable, has_trials,
                    timestamps_complete, sentences_valid
        
        Example:
            >>> results = loader.validate_all_patients()
            >>> print(results)
            patient_id  edf_exists  edf_loadable  has_trials  ...
            CON008      True        True          True        ...
            CON009      True        True          True        ...
        """
        results = []
        
        for patient_id in self.get_patient_ids():
            validation = self.validate_patient(patient_id)
            validation['patient_id'] = patient_id
            results.append(validation)
        
        df = pd.DataFrame(results)
        # Reorder columns to put patient_id first
        cols = ['patient_id'] + [c for c in df.columns if c != 'patient_id']
        return df[cols]
    
    # ==================== Metadata ====================
    
    def get_info(self) -> Dict:
        """
        Get overall dataset information.
        
        Returns:
            Dictionary with dataset metadata:
            - parquet_path: Path to Parquet file
            - total_trials: Total number of trials
            - total_patients: Number of unique patients
            - patient_ids: List of patient IDs
            - trial_types: List of trial types
            - date_range: Min and max dates
        """
        return {
            'parquet_path': str(self.parquet_path),
            'total_trials': len(self.trials_df),
            'total_patients': len(self.get_patient_ids()),
            'patient_ids': self.get_patient_ids(),
            'trial_types': self.get_trial_types(),
            'date_range': (
                self.trials_df['date'].min() if 'date' in self.trials_df.columns else None,
                self.trials_df['date'].max() if 'date' in self.trials_df.columns else None
            ),
            'edf_cache_size': self.edf_cache_size,
            'edf_cache_stats': self.get_cached_edfs()
        }
    
    def __repr__(self) -> str:
        """String representation of loader."""
        return (
            f"UnifiedDataLoader("
            f"{len(self.trials_df)} trials, "
            f"{len(self.get_patient_ids())} patients"
            f")"
        )
