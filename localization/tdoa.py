"""
TDOA estimation using GCC‑PHAT, tuned for 1 s chunks at 16 kHz.

Notes on performance:
- Uses power‑of‑two FFT sizes
- Operates on 1D arrays, avoids copies where possible
- Limits lag search to physically plausible window
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import List
from config.config import SAMPLE_RATE, SPEED_OF_SOUND, USE_GCC_PHAT, CORRELATION_MAX_LAG, REFERENCE_MIC_INDEX, ARRAY_SIZE


class TDOACalculator:
    # Calculates Time Difference of Arrival (TDOA) between microphones
    # using Generalized Cross-Correlation with Phase Transform (GCC-PHAT)
    
    def __init__(self, sample_rate: int = SAMPLE_RATE, reference_mic: int = REFERENCE_MIC_INDEX,
                 use_gcc_phat: bool = USE_GCC_PHAT, max_lag_seconds: float = None):
        self.sample_rate = sample_rate
        self.reference_mic = reference_mic
        self.use_gcc_phat = use_gcc_phat
        self.max_lag_seconds = max_lag_seconds  # Override for non-synchronized recordings
    
    def gcc_phat(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        # Compute time delay between two signals using GCC-PHAT
        # GCC-PHAT is more robust to noise and reverberation than standard cross-correlation
        sig1 = sig1.flatten()
        sig2 = sig2.flatten()
        
        # Next power of two for faster FFTs
        n = len(sig1) + len(sig2) - 1
        n_fft = 1 << int(np.ceil(np.log2(n)))
        
        # Compute FFTs
        fft1 = np.fft.fft(sig1, n=n_fft)
        fft2 = np.fft.fft(sig2, n=n_fft)
        
        # Cross-power spectrum
        cross_spectrum = fft1 * np.conj(fft2)
        
        if self.use_gcc_phat:
            # Phase transform: normalize by magnitude
            cross_spectrum = cross_spectrum / (np.abs(cross_spectrum) + 1e-10)
        
        # Inverse FFT to get correlation (real part only)
        correlation = np.fft.ifft(cross_spectrum).real
        
        # Find peak in correlation
        # Calculate max_lag based on actual sample rate (not config constant)
        if self.max_lag_seconds is not None:
            # User-specified max lag (for non-synchronized recordings)
            max_lag_samples = int(self.max_lag_seconds * self.sample_rate)
        else:
            # Physical constraint based on array geometry
            max_physical_tdoa = (ARRAY_SIZE * 1.414) / SPEED_OF_SOUND  # seconds
            max_lag_samples = int(max_physical_tdoa * self.sample_rate * 1.5)
        max_lag = min(max_lag_samples, len(correlation) // 2)
        
        # Rearrange correlation to have zero lag at center
        correlation = np.fft.fftshift(correlation)
        center = len(correlation) // 2
        
        # Search in window around center
        search_start = max(0, center - max_lag)
        search_end = min(len(correlation), center + max_lag)
        search_region = correlation[search_start:search_end]
        
        # Find integer‑sample peak
        peak_idx = int(np.argmax(search_region))
        lag_samples = peak_idx + search_start - center

        # Optional: 3‑point quadratic interpolation for sub‑sample refinement
        if 1 <= peak_idx < (len(search_region) - 1):
            y0 = search_region[peak_idx - 1]
            y1 = search_region[peak_idx]
            y2 = search_region[peak_idx + 1]
            denom = (y0 - 2 * y1 + y2)
            if abs(denom) > 1e-12:
                delta = 0.5 * (y0 - y2) / denom  # offset in samples
                lag_samples = lag_samples + delta
        
        # Convert lag to time delay
        time_delay = lag_samples / self.sample_rate
        
        return time_delay
    
    def calculate_tdoa(self, audio_signals: np.ndarray) -> np.ndarray:
        # Calculate TDOA for all microphones relative to reference microphone
        num_mics = audio_signals.shape[1]
        
        if num_mics < 2:
            raise ValueError("At least 2 microphones required")
        
        # Reference signal
        ref_signal = audio_signals[:, self.reference_mic]
        
        # Calculate TDOA for each microphone relative to reference
        tdoas = []
        
        for i in range(num_mics):
            if i == self.reference_mic:
                continue
            
            # Calculate time delay
            # Positive TDOA means signal arrives at mic i BEFORE reference mic
            time_delay = self.gcc_phat(audio_signals[:, i], ref_signal)
            tdoas.append(time_delay)
        
        return np.array(tdoas, dtype=float)
    
    def tdoa_to_distance_differences(self, tdoas: np.ndarray) -> np.ndarray:
        # Convert time delays to distance differences
        return tdoas * SPEED_OF_SOUND
    
    def __repr__(self):
        return (f"TDOACalculator(sample_rate={self.sample_rate}, "
                f"reference_mic={self.reference_mic}, "
                f"gcc_phat={self.use_gcc_phat})")


