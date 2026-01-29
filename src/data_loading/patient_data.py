"""PatientData class for single-patient workflows."""

import logging
from pathlib import Path
from typing import Optional, Dict, List, Callable, Union
import warnings

import pandas as pd
import mne


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)


class PatientData:
    """
    Single-patient data view with trial filtering and lazy EDF loading.
    
    Attributes:
        patient_id: Patient identifier
        trials_df: DataFrame with all trials for this patient
    """
    
    def __init__(
        self,
        patient_id: str,
        trials_df: pd.DataFrame,
        edf_loader_func: Callable,
        data_root: Path,
        find_edf_fn: Callable
    ):
        self.patient_id = patient_id
        self.trials_df = trials_df.copy()  # Defensive copy
        self._edf_loader = edf_loader_func
        self._data_root = data_root
        self._find_edf_fn = find_edf_fn
    
    def list_sessions(self) -> List[str]:
        """List recording session dates for this patient (sorted)."""
        return sorted(self.trials_df['date'].unique().tolist())
    
    @property
    def edf_paths(self) -> Union[Path, Dict[str, Path]]:
        """Get EDF path(s): Path for single session, Dict[date, Path] for multi-session."""
        sessions = self.list_sessions()
        if len(sessions) == 1:
            return self._find_edf_fn(self.patient_id, sessions[0], use_clipped=True)
        else:
            return {
                session: self._find_edf_fn(self.patient_id, session, use_clipped=True)
                for session in sessions
            }
    
    @property
    def edf_filenames(self) -> Union[str, Dict[str, str]]:
        """Get EDF filename(s): str for single session, Dict[date, str] for multi-session."""
        paths = self.edf_paths
        if isinstance(paths, dict):
            return {date: path.name for date, path in paths.items()}
        return paths.name
    
    def get_raw(self, date: Optional[str] = None) -> Union[mne.io.Raw, Dict[str, mne.io.Raw]]:
        """
        Get EEG data. Returns Raw for single/specified session, Dict[date, Raw] for multi-session.
        
        Args:
            date: Session date. If None, returns all sessions for multi-session patients.
        """
        return self._edf_loader(self.patient_id, date=date)
    
    @property
    def raw(self) -> Union[mne.io.Raw, Dict[str, mne.io.Raw]]:
        """
        Lazy-load EEG data. Returns Raw for single session, Dict[date, Raw] for multi-session.
        
        For multi-session, use get_raw(date) or check type with isinstance().
        """
        return self.get_raw()
    
    def get_trials_by_type(self, trial_type: str) -> pd.DataFrame:
        """Get trials of specific type (e.g., 'oddball', 'language')."""
        filtered = self.trials_df[self.trials_df['trial_type'] == trial_type]
        
        if len(filtered) == 0:
            available_types = self.get_trial_types()
            warnings.warn(
                f"No trials found for type '{trial_type}' in patient {self.patient_id}. "
                f"Available types: {available_types}"
            )
        
        return filtered.copy()
    
    def get_trial(self, trial_idx: int) -> pd.Series:
        """Get trial by index."""
        if trial_idx < 0 or trial_idx >= len(self.trials_df):
            raise IndexError(
                f"Trial index {trial_idx} out of range [0, {len(self.trials_df)-1}] "
                f"for patient {self.patient_id}"
            )
        
        return self.trials_df.iloc[trial_idx]
    
    def get_trial_types(self) -> List[str]:
        """Get sorted list of trial types for this patient."""
        return sorted(self.trials_df['trial_type'].unique().tolist())
    
    def get_eeg_info(self) -> Dict:
        """Get EEG metadata (triggers EDF loading). For multi-session, uses first session."""
        raw_data = self.raw  # Triggers loading if needed
        
        # Handle multi-session (use first session for metadata)
        if isinstance(raw_data, dict):
            raw = list(raw_data.values())[0]
        else:
            raw = raw_data
        
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
    
    def get_dc_channel(
        self, 
        channel_name: str = 'DC',
        date: Optional[str] = None
    ) -> Union[mne.io.Raw, Dict[str, mne.io.Raw]]:
        """
        Extract DC audio channel for ENG-02 timestamp alignment.
        
        Args:
            channel_name: DC channel name (DC, DC1, DC2, Audio)
            date: Session date. If None, returns all sessions for multi-session patients.
        
        Returns:
            Raw with DC channel only, or Dict[date, Raw] for multi-session
        """
        raw_data = self.raw  # May return Dict for multi-session
        
        # Handle multi-session
        if isinstance(raw_data, dict):
            if date is None:
                # Return DC for all sessions
                logger.info(f"Extracting DC channel from {len(raw_data)} sessions")
                return {
                    session_date: self._extract_dc_from_raw(session_raw, channel_name)
                    for session_date, session_raw in raw_data.items()
                }
            else:
                # Return DC for specific session
                if date not in raw_data:
                    raise ValueError(
                        f"Session {date} not found. Available: {list(raw_data.keys())}"
                    )
                return self._extract_dc_from_raw(raw_data[date], channel_name)
        else:
            # Single session
            return self._extract_dc_from_raw(raw_data, channel_name)
    
    def _extract_dc_from_raw(self, raw: mne.io.Raw, channel_name: str) -> mne.io.Raw:
        """Extract DC channel from Raw object (case-insensitive search)."""
        # Search for DC channel (case-insensitive, partial match)
        available_channels = raw.ch_names
        dc_channels = [ch for ch in available_channels 
                       if channel_name.upper() in ch.upper()]
        
        if not dc_channels:
            raise ValueError(
                f"DC channel '{channel_name}' not found in EDF. "
                f"Available channels: {available_channels[:10]}..."
            )
        
        # If multiple matches, prefer exact match
        if len(dc_channels) > 1:
            exact_match = [ch for ch in dc_channels if ch == channel_name]
            if exact_match:
                dc_channels = exact_match
            else:
                logger.warning(
                    f"Multiple DC channels found: {dc_channels}. "
                    f"Using first: {dc_channels[0]}"
                )
        
        return raw.copy().pick_channels([dc_channels[0]])
    
    def get_trial_timing_info(
        self, 
        trial_type: Optional[str] = None,
        date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get trial timing with EDF-relative times for ENG-02 alignment.
        
        Args:
            trial_type: Filter by trial type (e.g., 'oddball')
            date: Filter by session. For multi-session, specify date to get correct EDF-relative times.
        
        Returns:
            DataFrame with date, trial_type, start_time, end_time, duration,
            edf_start_time, edf_end_time, sentences columns.
        """
        # Get trials
        if trial_type:
            trials = self.get_trials_by_type(trial_type)
        else:
            trials = self.trials_df.copy()
        
        # Filter by date if specified
        if date:
            trials = trials[trials['date'] == date].copy()
            if len(trials) == 0:
                logger.warning(
                    f"No trials found for {self.patient_id} on {date} "
                    f"with trial_type={trial_type}"
                )
                return pd.DataFrame(columns=[
                    'date', 'trial_type', 'start_time', 'end_time', 'duration',
                    'edf_start_time', 'edf_end_time', 'sentences'
                ])
        else:
            trials = trials.copy()
        
        # Initialize EDF-relative time columns
        trials['edf_start_time'] = None
        trials['edf_end_time'] = None
        
        # Load Raw data (might be Dict or single)
        raw_data = self.raw
        
        # Get unique sessions in the trials
        sessions = trials['date'].unique()
        
        if len(sessions) > 1 and date is None:
            logger.warning(
                f"Patient {self.patient_id} has trials from {len(sessions)} sessions. "
                f"EDF-relative times computed per-session. "
                f"For ENG-02, process each session separately by specifying date parameter."
            )
        
        # Compute EDF-relative times per session
        for session_date in sessions:
            session_mask = trials['date'] == session_date
            
            # Get the right Raw object for this session
            if isinstance(raw_data, dict):
                if session_date not in raw_data:
                    logger.warning(
                        f"No EDF loaded for session {session_date}. "
                        f"edf_start_time will be None for these trials."
                    )
                    continue
                session_raw = raw_data[session_date]
            else:
                # Single session
                session_raw = raw_data
            
            # Compute relative times for this session's trials
            if session_raw.info['meas_date'] is not None:
                edf_start_unix = session_raw.info['meas_date'].timestamp()
                
                trials.loc[session_mask, 'edf_start_time'] = \
                    trials.loc[session_mask, 'start_time'] - edf_start_unix
                trials.loc[session_mask, 'edf_end_time'] = \
                    trials.loc[session_mask, 'end_time'] - edf_start_unix
            else:
                # EDF missing measurement date
                logger.warning(
                    f"Patient {self.patient_id} session {session_date}: "
                    f"EDF missing measurement date. "
                    f"Cannot compute EDF-relative times. "
                    f"ENG-02 must use DC audio channel for alignment."
                )
        
        # Select relevant columns (include 'date' for session tracking)
        timing_cols = ['date', 'trial_type', 'start_time', 'end_time', 'duration', 
                       'edf_start_time', 'edf_end_time', 'sentences']
        return trials[timing_cols].copy()
    
    def validate(self) -> Dict[str, bool]:
        """Validate data quality (has_trials, edf_loadable, timestamps_complete, sentences_valid)."""
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
            
            # TODO(ENG-02): Implement timestamp alignment validation
            # Will use DC audio channel for precise synchronization (<50ms accuracy)
            # Current timestamp_alignment check removed as it:
            # 1. Assumes single EDF per patient (invalid for multi-session)
            # 2. Doesn't account for DC audio channel alignment
            # 3. Will be properly implemented in ENG-02 with DC waveform matching
            validation['timestamp_alignment'] = None  # Not implemented yet
        except Exception as e:
            warnings.warn(f"Patient {self.patient_id}: Failed to load EDF: {e}")
        
        return validation
    
    def __repr__(self) -> str:
        sessions = self.list_sessions()
        session_info = f"{len(sessions)} session(s)" if len(sessions) > 1 else "single session"
        return (
            f"PatientData("
            f"patient={self.patient_id}, "
            f"{len(self.trials_df)} trials, "
            f"{session_info}"
            f")"
        )
