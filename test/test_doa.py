#!/usr/bin/env python3
"""
Offline test: load 4 one‑second files (TL, TR, BL, BR) and print the DOA.
Use this to validate geometry and channel order without live capture.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import struct
from pathlib import Path
from localization.doa import DOACalculator
from config.config import SAMPLE_RATE, MIC_POSITIONS, SPEED_OF_SOUND


def read_wav(filename):
    """Read WAV file manually"""
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

def calculate_tdoas_from_onsets(signals, sample_rate, reference_idx=0):
    """
    Calculate TDOAs directly from onset detection.
    Returns TDOAs in seconds (relative to reference microphone).
    """
    # Find onset in each signal
    onsets = []
    for sig in signals:
        onset = find_sound_onset(sig, sample_rate)
        onsets.append(onset)
    
    # Calculate TDOAs relative to reference microphone
    ref_onset = onsets[reference_idx]
    
    tdoas_samples = []
    for i, onset in enumerate(onsets):
        if i == reference_idx:
            continue
        # Negative TDOA means signal arrives BEFORE reference
        tdoa = onset - ref_onset
        tdoas_samples.append(tdoa)
    
    # Convert to time delays
    tdoas_time = np.array(tdoas_samples) / sample_rate
    
    return tdoas_time


def test_doa_from_files(audio_files: list):
    # Run DOA on 4 files ordered [BL, BR, TL, TR]
    if len(audio_files) != 4:
        print("Error: Exactly 4 audio files required")
        print("Order: bottom_left, bottom_right, top_left, top_right")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("TDOA Direction of Arrival - Offline Test")
    print("=" * 80)
    
    # Load audio files
    print("\nLoading audio files...")
    signals = []
    original_sr = None
    positions = ["Bottom-Left", "Bottom-Right", "Top-Left", "Top-Right"]
    
    for filepath, position in zip(audio_files, positions):
        print(f"  {position}: {filepath}")
        sig, sample_rate = read_wav(filepath)
        
        if original_sr is None:
            original_sr = sample_rate
        
        print(f"    {len(sig)} samples @ {sample_rate} Hz ({len(sig)/sample_rate:.3f} sec)")
        signals.append(sig)
    
    # Get microphone positions from config (order: BL, BR, TL, TR)
    mic_positions = np.array([
        MIC_POSITIONS['bottom_left'],
        MIC_POSITIONS['bottom_right'],
        MIC_POSITIONS['top_left'],
        MIC_POSITIONS['top_right']
    ])
    
    print("\nMicrophone Array Configuration:")
    for name, pos in zip(positions, mic_positions):
        print(f"  {name}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) m")
    
    # Calculate TDOA using onset detection
    print("\nCalculating TDOA using onset detection...")
    tdoas = calculate_tdoas_from_onsets(signals, original_sr, reference_idx=0)
    
    print(f"\n  Time Delays (relative to Bottom-Left mic):")
    print(f"    Bottom-Right: {tdoas[0]*1e6:8.2f} μs ({tdoas[0]*SPEED_OF_SOUND:.4f} m)")
    print(f"    Top-Left:     {tdoas[1]*1e6:8.2f} μs ({tdoas[1]*SPEED_OF_SOUND:.4f} m)")
    print(f"    Top-Right:    {tdoas[2]*1e6:8.2f} μs ({tdoas[2]*SPEED_OF_SOUND:.4f} m)")
    
    # Initialize DOA calculator
    doa_calc = DOACalculator(mic_positions=mic_positions)
    
    # Calculate DOA
    print("\nCalculating Direction of Arrival...")
    direction_vector, azimuth, elevation = doa_calc.calculate_direction(tdoas)
    
    # Display results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nDirection Vector (Cartesian):")
    print(f"  x = {direction_vector[0]:7.4f}")
    print(f"  y = {direction_vector[1]:7.4f}")
    print(f"  z = {direction_vector[2]:7.4f}")
    print(f"  |v| = {np.linalg.norm(direction_vector):.4f}")
    
    print(f"\nDirection (Spherical):")
    print(f"  Azimuth:   {azimuth:6.2f}° (0° = East, 90° = North, 180° = West, 270° = South)")
    print(f"  Elevation: {elevation:6.2f}° (0° = Horizontal, 90° = Up, -90° = Down)")
    
    print("\n" + "=" * 80 + "\n")


def main():
    # Main function for test script
    parser = argparse.ArgumentParser(
        description="Test TDOA-based DOA calculation with pre-recorded audio files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python test_doa.py mic_tl.wav mic_tr.wav mic_bl.wav mic_br.wav
  
Audio file requirements:
  - 4 audio files in order: bottom-left, bottom-right, top-left, top-right
  - 1 second duration
  - 16 kHz sample rate (recommended)
  - Mono (single channel)
  - WAV, FLAC, or OGG format
        """
    )
    
    parser.add_argument(
        'audio_files',
        nargs=4,
        metavar='AUDIO_FILE',
        help='Audio files for: bottom-left, bottom-right, top-left, top-right'
    )
    
    args = parser.parse_args()
    
    # Validate files exist
    for filepath in args.audio_files:
        if not Path(filepath).exists():
            print(f"Error: File not found: {filepath}")
            sys.exit(1)
    
    # Run test
    test_doa_from_files(args.audio_files)


if __name__ == "__main__":
    main()


