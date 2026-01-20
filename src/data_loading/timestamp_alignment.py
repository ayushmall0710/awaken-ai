"""
Timestamp Alignment Module (ENG-02)

This module provides functionality to synchronize CSV Unix timestamps with EDF 
internal clocks using the DC audio input channel for precise alignment.

The alignment strategy:
1. Extract DC audio channel from EDF
2. Detect stimulus onset peaks in the audio signal
3. Convert EDF sample times to Unix timestamps
4. Match detected onsets with CSV timestamps
5. Validate alignment precision (target: ±50ms)
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from typing import Tuple, List, Optional, Dict
import warnings


class TimestampAligner:
    """
    Aligns CSV Unix timestamps with EDF internal clocks using DC audio channel.
    
    This class implements ENG-02: Timestamp Alignment functionality.
    """
    
    def __init__(self, eeg_loader, dc_channel_name: Optional[str] = None):
        """
        Initialize the timestamp aligner.
        
        Parameters
        ----------
        eeg_loader : EEGDataLoader
            An instance of EEGDataLoader with loaded EDF data
        dc_channel_name : str, optional
            Name of the DC audio channel. If None, will attempt auto-detection.
        """
        self.loader = eeg_loader
        self.dc_channel_name = dc_channel_name
        
        # Auto-detect DC channel if not provided
        if self.dc_channel_name is None:
            self.dc_channel_name = self.loader.find_dc_channel()
            if self.dc_channel_name is None:
                raise ValueError(
                    "Could not auto-detect DC channel. Please specify dc_channel_name. "
                    f"Available channels: {self.loader.raw.ch_names}"
                )
        
        # Verify channel exists
        if self.dc_channel_name not in self.loader.raw.ch_names:
            raise ValueError(
                f"Channel '{self.dc_channel_name}' not found. "
                f"Available: {self.loader.raw.ch_names}"
            )
        
        self.sampling_freq = self.loader.raw.info['sfreq']
        self.edf_start_time = self.loader.raw.info['meas_date']
    
    def extract_dc_channel(self, start: float = 0, stop: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract DC audio channel data.
        
        Parameters
        ----------
        start : float, default=0
            Start time in seconds
        stop : float, optional
            Stop time in seconds
            
        Returns
        -------
        tuple
            (data, times) - DC channel data and time points
        """
        return self.loader.get_channel_data(self.dc_channel_name, start, stop)
    
    def detect_stimulus_onsets(
        self, 
        data: np.ndarray, 
        times: np.ndarray,
        threshold: Optional[float] = None,
        min_distance: float = 0.5,
        normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect stimulus onset times from DC audio channel using peak detection.
        
        Parameters
        ----------
        data : np.ndarray
            DC channel data array
        times : np.ndarray
            Corresponding time points in seconds (EDF time)
        threshold : float, optional
            Peak detection threshold. If None, uses median + 3*MAD
        min_distance : float, default=0.5
            Minimum time between peaks in seconds (prevents double-detection)
        normalize : bool, default=True
            Whether to normalize the data before peak detection
            
        Returns
        -------
        tuple
            (peak_times, peak_values) - times and amplitudes of detected peaks
        """
        # Normalize if requested
        if normalize:
            data = (data - np.mean(data)) / np.std(data)
        
        # Auto-detect threshold using robust statistics
        if threshold is None:
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            threshold = median + 3 * mad
        
        # Convert min_distance to samples
        min_samples = int(min_distance * self.sampling_freq)
        
        # Find peaks
        peak_indices, properties = find_peaks(
            np.abs(data),  # Use absolute value to catch both positive and negative peaks
            height=threshold,
            distance=min_samples
        )
        
        # Extract peak times and values
        peak_times = times[peak_indices]
        peak_values = data[peak_indices]
        
        return peak_times, peak_values
    
    def edf_time_to_unix(self, edf_times: np.ndarray) -> np.ndarray:
        """
        Convert EDF relative times to Unix timestamps.
        
        Parameters
        ----------
        edf_times : np.ndarray
            Array of times in seconds from EDF start
            
        Returns
        -------
        np.ndarray
            Array of Unix timestamps
        """
        if self.edf_start_time is None:
            raise ValueError("EDF file does not contain recording start time (meas_date)")
        
        # Convert recording start to Unix timestamp
        edf_start_unix = self.edf_start_time.timestamp()
        
        # Add EDF times to get Unix timestamps
        return edf_start_unix + edf_times
    
    def align_with_csv(
        self,
        peak_times_edf: np.ndarray,
        csv_timestamps: np.ndarray,
        max_offset: float = 0.5
    ) -> pd.DataFrame:
        """
        Align detected peaks with CSV timestamps.
        
        Parameters
        ----------
        peak_times_edf : np.ndarray
            Detected peak times in EDF time (seconds from start)
        csv_timestamps : np.ndarray
            Unix timestamps from CSV
        max_offset : float, default=0.5
            Maximum allowed offset in seconds for matching
            
        Returns
        -------
        pd.DataFrame
            Alignment results with columns:
            - peak_idx: Index of detected peak
            - edf_time: Peak time in EDF coordinates
            - unix_time: Converted Unix timestamp
            - csv_time: Matched CSV timestamp
            - offset_ms: Difference in milliseconds
        """
        # Convert EDF times to Unix
        peak_times_unix = self.edf_time_to_unix(peak_times_edf)
        
        # Match each detected peak to nearest CSV timestamp
        alignments = []
        
        for idx, (edf_t, unix_t) in enumerate(zip(peak_times_edf, peak_times_unix)):
            # Find nearest CSV timestamp
            offsets = np.abs(csv_timestamps - unix_t)
            min_offset_idx = np.argmin(offsets)
            min_offset = offsets[min_offset_idx]
            
            # Only include if within max_offset
            if min_offset <= max_offset:
                alignments.append({
                    'peak_idx': idx,
                    'edf_time': edf_t,
                    'unix_time': unix_t,
                    'csv_time': csv_timestamps[min_offset_idx],
                    'offset_ms': (unix_t - csv_timestamps[min_offset_idx]) * 1000
                })
        
        return pd.DataFrame(alignments)
    
    def validate_alignment(
        self, 
        alignment_df: pd.DataFrame, 
        target_precision_ms: float = 50
    ) -> Dict[str, float]:
        """
        Validate alignment precision against target.
        
        Parameters
        ----------
        alignment_df : pd.DataFrame
            DataFrame from align_with_csv()
        target_precision_ms : float, default=50
            Target alignment precision in milliseconds
            
        Returns
        -------
        dict
            Dictionary with validation metrics:
            - mean_offset_ms: Mean absolute offset
            - std_offset_ms: Standard deviation of offset
            - max_offset_ms: Maximum absolute offset
            - within_target_pct: Percentage within target precision
            - n_aligned: Number of aligned events
        """
        if len(alignment_df) == 0:
            return {
                'mean_offset_ms': np.nan,
                'std_offset_ms': np.nan,
                'max_offset_ms': np.nan,
                'within_target_pct': 0.0,
                'n_aligned': 0
            }
        
        offsets = np.abs(alignment_df['offset_ms'].values)
        
        return {
            'mean_offset_ms': np.mean(offsets),
            'std_offset_ms': np.std(offsets),
            'max_offset_ms': np.max(offsets),
            'within_target_pct': (np.sum(offsets <= target_precision_ms) / len(offsets)) * 100,
            'n_aligned': len(alignment_df)
        }
    
    def synchronize_trial(
        self,
        trial_start_unix: float,
        trial_end_unix: float,
        threshold: Optional[float] = None,
        min_distance: float = 0.5
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Complete synchronization workflow for a single trial.
        
        Parameters
        ----------
        trial_start_unix : float
            Trial start time (Unix timestamp)
        trial_end_unix : float
            Trial end time (Unix timestamp)
        threshold : float, optional
            Peak detection threshold
        min_distance : float, default=0.5
            Minimum distance between peaks in seconds
            
        Returns
        -------
        tuple
            (alignment_df, metrics) - Alignment results and validation metrics
        """
        # Convert Unix times to EDF times
        edf_start_unix = self.edf_start_time.timestamp()
        trial_start_edf = trial_start_unix - edf_start_unix
        trial_end_edf = trial_end_unix - edf_start_unix
        
        # Extract DC channel for this trial
        dc_data, dc_times = self.extract_dc_channel(trial_start_edf, trial_end_edf)
        
        # Detect stimulus onsets
        peak_times, peak_values = self.detect_stimulus_onsets(
            dc_data, dc_times, threshold=threshold, min_distance=min_distance
        )
        
        # For now, we don't have individual stimulus timestamps from CSV
        # This would be used when detailed event timing is available
        # For validation, we can check if we detected reasonable number of peaks
        
        # Convert to Unix time
        peak_times_unix = self.edf_time_to_unix(peak_times)
        
        # Create simple result DataFrame
        result = pd.DataFrame({
            'peak_idx': np.arange(len(peak_times)),
            'edf_time': peak_times,
            'unix_time': peak_times_unix,
            'peak_amplitude': peak_values
        })
        
        # Basic metrics
        metrics = {
            'n_peaks_detected': len(peak_times),
            'trial_duration': trial_end_edf - trial_start_edf,
            'mean_isi': np.mean(np.diff(peak_times)) if len(peak_times) > 1 else np.nan
        }
        
        return result, metrics
