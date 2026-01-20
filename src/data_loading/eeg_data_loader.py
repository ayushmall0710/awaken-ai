"""
EEG Data Loader Module (ENG-01)

This module provides the base EEGDataLoader class for loading EDF files
and linking them to stimulus CSV logs.
"""

import mne
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
import warnings


class EEGDataLoader:
    """
    Base class for loading EEG data from EDF files and linking to CSV stimulus logs.
    
    This class implements ENG-01: Base Data Loader functionality.
    """
    
    def __init__(self, edf_path: str, csv_path: Optional[str] = None, preload: bool = True):
        """
        Initialize the EEG Data Loader.
        
        Parameters
        ----------
        edf_path : str
            Path to the EDF file containing EEG recordings
        csv_path : str, optional
            Path to the CSV file containing stimulus timing logs
        preload : bool, default=True
            Whether to preload the EDF data into memory
        """
        self.edf_path = Path(edf_path)
        self.csv_path = Path(csv_path) if csv_path else None
        self.preload = preload
        
        # Initialize data containers
        self.raw = None
        self.stimulus_df = None
        self.patient_id = None
        
        # Load data
        self._load_edf()
        if self.csv_path:
            self._load_csv()
    
    def _load_edf(self) -> None:
        """Load EDF file using MNE."""
        if not self.edf_path.exists():
            raise FileNotFoundError(f"EDF file not found: {self.edf_path}")
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            self.raw = mne.io.read_raw_edf(
                str(self.edf_path), 
                preload=self.preload, 
                verbose=False
            )
        
        # Extract patient ID from filename (e.g., CON008_clipped.EDF -> CON008)
        self.patient_id = self.edf_path.stem.split('_')[0]
    
    def _load_csv(self) -> None:
        """Load stimulus CSV log file."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        self.stimulus_df = pd.read_csv(self.csv_path)
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get summary information about the loaded data.
        
        Returns
        -------
        dict
            Dictionary containing EDF info and optionally CSV info
        """
        info = {
            'patient_id': self.patient_id,
            'edf_path': str(self.edf_path),
            'sampling_frequency': self.raw.info['sfreq'],
            'duration_seconds': self.raw.times[-1],
            'n_channels': len(self.raw.ch_names),
            'channel_names': self.raw.ch_names,
            'recording_start': self.raw.info['meas_date']
        }
        
        if self.stimulus_df is not None:
            info.update({
                'csv_path': str(self.csv_path),
                'n_trials': len(self.stimulus_df),
                'trial_types': self.stimulus_df['trial_type'].unique().tolist() if 'trial_type' in self.stimulus_df.columns else None
            })
        
        return info
    
    def get_channel_data(self, channel_name: str, start: float = 0, stop: Optional[float] = None):
        """
        Extract data from a specific channel.
        
        Parameters
        ----------
        channel_name : str
            Name of the channel to extract
        start : float, default=0
            Start time in seconds
        stop : float, optional
            Stop time in seconds (None = end of recording)
            
        Returns
        -------
        tuple
            (data, times) - channel data array and corresponding time points
        """
        if channel_name not in self.raw.ch_names:
            raise ValueError(f"Channel '{channel_name}' not found. Available: {self.raw.ch_names}")
        
        # Get time indices
        if stop is None:
            stop = self.raw.times[-1]
        
        # Extract channel data
        picks = mne.pick_channels(self.raw.ch_names, [channel_name])
        data, times = self.raw[picks, :]
        
        # Filter by time range
        time_mask = (times >= start) & (times <= stop)
        return data[0, time_mask], times[time_mask]
    
    def find_dc_channel(self) -> Optional[str]:
        """
        Automatically identify the DC audio channel.
        
        Returns
        -------
        str or None
            Name of the DC/audio channel if found, None otherwise
        """
        dc_keywords = ['DC', 'dc', 'AUX', 'aux', 'Audio', 'audio', 'TRIG', 'trig']
        
        for ch_name in self.raw.ch_names:
            for keyword in dc_keywords:
                if keyword in ch_name:
                    return ch_name
        
        return None
    
    def get_oddball_trials(self) -> pd.DataFrame:
        """
        Get oddball trials from the stimulus dataframe.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing only oddball trials
        """
        if self.stimulus_df is None:
            raise ValueError("No CSV data loaded. Provide csv_path during initialization.")
        
        if 'trial_type' not in self.stimulus_df.columns:
            raise ValueError("CSV does not contain 'trial_type' column")
        
        # Filter for oddball trials (may be 'oddball', 'oddball+p', etc.)
        oddball_mask = self.stimulus_df['trial_type'].str.contains('oddball', case=False, na=False)
        return self.stimulus_df[oddball_mask].copy()
