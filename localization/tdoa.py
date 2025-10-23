"""
Measures time differences between when sound reaches different microphones.
Uses GCC-PHAT algorithm optimized for 1-second audio chunks at 16kHz.
Fast and memory-efficient - avoids unnecessary copying and uses optimal FFT sizes.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import List
from config.config import SAMPLE_RATE, SPEED_OF_SOUND, USE_GCC_PHAT, CORRELATION_MAX_LAG, REFERENCE_MIC_INDEX, ARRAY_SIZE


class TDOACalculator:
    """Measures timing differences between microphones using cross-correlation."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, reference_mic: int = REFERENCE_MIC_INDEX,
                 use_gcc_phat: bool = USE_GCC_PHAT, max_lag_seconds: float = None):
        """Set up TDOA calculator with audio settings."""
        self.sample_rate = sample_rate
        self.reference_mic = reference_mic
        self.use_gcc_phat = use_gcc_phat
        self.max_lag_seconds = max_lag_seconds  # For recordings that aren't perfectly synced
    
    def gcc_phat(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        """Find time delay between two audio signals using cross-correlation with phase transform."""
        # Make sure we're working with 1D arrays
        sig1 = sig1.flatten()
        sig2 = sig2.flatten()

        # Use power-of-2 FFT size for speed (next power of two after combined length)
        n = len(sig1) + len(sig2) - 1
        n_fft = 1 << int(np.ceil(np.log2(n)))  # Next power of 2

        # Convert both signals to frequency domain
        fft1 = np.fft.fft(sig1, n=n_fft)
        fft2 = np.fft.fft(sig2, n=n_fft)

        # Find where the signals match in frequency domain
        cross_spectrum = fft1 * np.conj(fft2)

        if self.use_gcc_phat:
            # Phase transform: ignore amplitude differences, focus on timing
            # Makes it more reliable in noisy environments
            cross_spectrum = cross_spectrum / (np.abs(cross_spectrum) + 1e-10)

        # Convert back to time domain to find delay
        correlation = np.fft.ifft(cross_spectrum).real

        # Only search within physically possible delay range
        if self.max_lag_seconds is not None:
            # User specified max delay (for poorly synced recordings)
            max_lag_samples = int(self.max_lag_seconds * self.sample_rate)
        else:
            # Based on how far apart mics can be (diagonal distance)
            max_physical_tdoa = (ARRAY_SIZE * 1.414) / SPEED_OF_SOUND  # seconds
            max_lag_samples = int(max_physical_tdoa * self.sample_rate * 1.5)
        max_lag = min(max_lag_samples, len(correlation) // 2)

        # Shift correlation so zero delay is in the center
        correlation = np.fft.fftshift(correlation)
        center = len(correlation) // 2

        # Look for peak correlation in the valid range
        search_start = max(0, center - max_lag)
        search_end = min(len(correlation), center + max_lag)
        search_region = correlation[search_start:search_end]

        # Find the exact sample where correlation peaks
        peak_idx = int(np.argmax(search_region))
        lag_samples = peak_idx + search_start - center

        # Fine-tune: use quadratic interpolation for sub-sample accuracy
        if 1 <= peak_idx < (len(search_region) - 1):
            y0 = search_region[peak_idx - 1]
            y1 = search_region[peak_idx]
            y2 = search_region[peak_idx + 1]
            denom = (y0 - 2 * y1 + y2)
            if abs(denom) > 1e-12:
                delta = 0.5 * (y0 - y2) / denom  # Fractional sample offset
                lag_samples = lag_samples + delta

        # Convert from samples to seconds
        time_delay = lag_samples / self.sample_rate

        return time_delay
    
    def calculate_tdoa(self, audio_signals: np.ndarray) -> np.ndarray:
        """Find timing differences for all mics compared to reference mic."""
        num_mics = audio_signals.shape[1]

        if num_mics < 2:
            raise ValueError("Need at least 2 microphones")

        # Get the reference microphone's audio
        ref_signal = audio_signals[:, self.reference_mic]

        # Compare each other mic to the reference
        tdoas = []

        for i in range(num_mics):
            if i == self.reference_mic:
                continue

            # Positive delay means this mic heard sound BEFORE reference mic
            time_delay = self.gcc_phat(audio_signals[:, i], ref_signal)
            tdoas.append(time_delay)

        return np.array(tdoas, dtype=float)

    def tdoa_to_distance_differences(self, tdoas: np.ndarray) -> np.ndarray:
        """Convert time delays to how much farther sound had to travel."""
        return tdoas * SPEED_OF_SOUND

    def __repr__(self):
        return f"TDOA Calculator: {self.sample_rate}Hz, ref mic #{self.reference_mic}, GCC-PHAT: {self.use_gcc_phat}"


