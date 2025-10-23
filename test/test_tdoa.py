#!/usr/bin/env python3
"""
Test timing differences between microphones using pre-recorded audio.

This script analyzes how sound arrives at different microphones by detecting
when the sound starts in each recording. Perfect for testing your setup!

Usage:
    python3 test_tdoa.py BL.wav BR.wav TL.wav TR.wav

What it does:
- Loads 4 audio files (Bottom-Left, Bottom-Right, Top-Left, Top-Right order)
- Finds exactly when sound starts in each recording
- Shows which microphone heard the sound first/last
- Calculates precise timing differences in microseconds
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import struct
from scipy import signal
from localization.tdoa import TDOACalculator
from config.config import SPEED_OF_SOUND, SAMPLE_RATE

def read_wav(filename):
    """Load audio from a WAV file (handles different formats automatically)"""
    with open(filename, 'rb') as f:
        f.read(4)  # RIFF
        f.read(4)  # file size
        f.read(4)  # WAVE
        
        while True:
            chunk_id = f.read(4)
            if not chunk_id:
                break
            chunk_size = struct.unpack('<I', f.read(4))[0]
            
            if chunk_id == b'fmt ':
                fmt_data = f.read(chunk_size)
                audio_format = struct.unpack('<H', fmt_data[0:2])[0]
                channels = struct.unpack('<H', fmt_data[2:4])[0]
                sample_rate = struct.unpack('<I', fmt_data[4:8])[0]
                bits_per_sample = struct.unpack('<H', fmt_data[14:16])[0]
            elif chunk_id == b'data':
                data = f.read(chunk_size)
                break
            else:
                f.read(chunk_size)
        
        if audio_format == 3 and bits_per_sample == 32:
            samples = np.frombuffer(data, dtype=np.float32)
        elif audio_format == 1 and bits_per_sample == 16:
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        
        if channels == 2:
            samples = samples.reshape(-1, 2)[:, 0]
        
        return samples, sample_rate

def find_sound_onset(audio, sample_rate, threshold_percentile=95):
    """
    Find the onset of the main sound event in an audio signal.
    Returns the sample index where significant sound begins.
    """
    # Calculate energy in sliding windows
    window_size = int(0.01 * sample_rate)  # 10ms windows
    hop_size = window_size // 2
    
    energy = []
    for i in range(0, len(audio) - window_size, hop_size):
        window = audio[i:i+window_size]
        energy.append(np.sum(window**2))
    
    energy = np.array(energy)
    
    # Find threshold based on percentile of energy values
    threshold = np.percentile(energy, threshold_percentile)
    
    # Find first window that exceeds threshold
    onset_windows = np.where(energy > threshold)[0]
    
    if len(onset_windows) == 0:
        return 0
    
    # Convert window index back to sample index
    onset_sample = onset_windows[0] * hop_size
    
    # Refine by looking backward for actual onset
    search_start = max(0, onset_sample - window_size)
    search_region = audio[search_start:onset_sample + window_size]
    
    # Find first sample above 5% of max in search region
    threshold_fine = np.max(np.abs(search_region)) * 0.05
    fine_onset = np.where(np.abs(search_region) > threshold_fine)[0]
    
    if len(fine_onset) > 0:
        onset_sample = search_start + fine_onset[0]
    
    return onset_sample

def align_recordings(signals, sample_rate, reference_idx=0):
    """
    Find when sound starts in each recording and calculate timing differences.

    Uses onset detection to find exactly when each microphone heard the sound,
    then compares all of them to see which one heard it first.
    """
    print("\nFinding when sound starts in each recording...")

    # Detect when sound begins in each microphone
    onsets = []
    for i, sig in enumerate(signals):
        onset = find_sound_onset(sig, sample_rate)
        onsets.append(onset)
        print(f"  Mic {i}: sound starts at sample {onset} ({onset/sample_rate*1000:.2f} ms)")

    # Compare everything to our reference microphone
    ref_onset = onsets[reference_idx]
    print(f"\n  Using Mic {reference_idx} as reference (heard sound at {ref_onset} samples)")

    tdoas_samples = []
    for i, onset in enumerate(onsets):
        if i == reference_idx:
            continue
        # How much earlier/later this mic heard the sound (in samples)
        tdoa = onset - ref_onset
        tdoas_samples.append(tdoa)
        if tdoa < 0:
            print(f"  Mic {i}: {tdoa:+d} samples ({tdoa/sample_rate*1e6:+.2f} μs) - heard it FIRST")
        else:
            print(f"  Mic {i}: {tdoa:+d} samples ({tdoa/sample_rate*1e6:+.2f} μs) - heard it LATER")

    # Convert from samples to seconds
    tdoas_time = np.array(tdoas_samples) / sample_rate

    return tdoas_time

def resample_audio(audio_data, original_sr, target_sr=SAMPLE_RATE):
    """Change audio sample rate if needed (e.g., from 48kHz to 16kHz)"""
    if original_sr == target_sr:
        return audio_data  # Already at the right rate

    resampled = signal.resample_poly(audio_data, target_sr, original_sr)
    return resampled

# Set up file loading (use command line args or defaults)
files = sys.argv[1:5] if len(sys.argv) >= 5 else ['BL.wav', 'BR.wav', 'TL.wav', 'TR.wav']
names = ['BL', 'BR', 'TL', 'TR']

print("=" * 80)
print("TDOA (Time Difference of Arrival) Test")
print("=" * 80)

signals = []
original_sr = None

print("\nLoading audio files...")
for fname, name in zip(files, names):
    sig, sample_rate = read_wav(fname)

    if original_sr is None:
        original_sr = sample_rate

    print(f"{name}: {len(sig)} samples @ {sample_rate} Hz ({len(sig)/sample_rate:.3f} sec)")
    signals.append(sig)

# Find timing differences by detecting when sound starts in each recording
print("\nThis method is often more accurate than cross-correlation for these types of recordings!")
tdoas = align_recordings(signals, original_sr, reference_idx=0)

print("\n" + "=" * 80)
print("TIMING RESULTS (compared to BL microphone)")
print("=" * 80)

for name, tdoa in zip(names[1:], tdoas):
    print(f"\n{name} microphone:")
    print(f"  Time delay:    {tdoa*1e6:+8.2f} μs  ({tdoa*1e3:+7.3f} ms)")
    print(f"  Distance diff: {tdoa*SPEED_OF_SOUND:+8.4f} m   ({tdoa*SPEED_OF_SOUND*100:+7.2f} cm)")

# Figure out which microphone heard the sound first
arrival_times = {'BL': 0.0}
for name, tdoa in zip(names[1:], tdoas):
    arrival_times[name] = tdoa

# Sort by who heard it earliest
sorted_arrivals = sorted(arrival_times.items(), key=lambda x: x[1])

print("\n" + "=" * 80)
print("SOUND ARRIVAL ORDER")
print("=" * 80)

print("\nFrom earliest to latest:")
for i, (name, time) in enumerate(sorted_arrivals, 1):
    print(f"  {i}. {name}  @ {time*1e6:+8.2f} μs")

calculated_order = [name for name, _ in sorted_arrivals]

print(f"\nSound traveled: {' -> '.join(calculated_order)}")

print("\n" + "=" * 80)

