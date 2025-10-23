#!/usr/bin/env python3
"""
Test our direction-finding system with pre-recorded audio files.

Load 4 WAV files (one per microphone) and calculate where the sound came from.
Great for testing your microphone setup without needing to make noise!
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
    Find when the main sound starts in an audio clip.

    Looks for where the audio gets loud by analyzing energy in small time windows.
    Returns the sample number where the interesting sound begins.
    """
    # Break audio into small 10ms chunks and measure how loud each one is
    window_size = int(0.01 * sample_rate)  # 10ms windows
    hop_size = window_size // 2  # Overlap windows by 50%

    energy = []
    for i in range(0, len(audio) - window_size, hop_size):
        window = audio[i:i+window_size]
        energy.append(np.sum(window**2))  # Total energy in this chunk

    energy = np.array(energy)

    # Set threshold at the 95th percentile (ignore background noise)
    threshold = np.percentile(energy, threshold_percentile)

    # Find first chunk that's louder than our threshold
    onset_windows = np.where(energy > threshold)[0]

    if len(onset_windows) == 0:
        return 0  # No sound found, start from beginning

    # Convert from chunk number back to sample number
    onset_sample = onset_windows[0] * hop_size

    # Fine-tune: look closer for the exact moment sound starts
    search_start = max(0, onset_sample - window_size)
    search_region = audio[search_start:onset_sample + window_size]

    # Find first sample that's 5% as loud as the loudest in this region
    threshold_fine = np.max(np.abs(search_region)) * 0.05
    fine_onset = np.where(np.abs(search_region) > threshold_fine)[0]

    if len(fine_onset) > 0:
        onset_sample = search_start + fine_onset[0]

    return onset_sample

def calculate_tdoas_from_onsets(signals, sample_rate, reference_idx=0):
    """
    Find timing differences by detecting when sound starts in each microphone.

    Instead of cross-correlation, this just finds when each mic first hears the sound
    and calculates how much earlier/later each one was compared to the reference mic.
    """
    # Find when sound starts in each microphone recording
    onsets = []
    for sig in signals:
        onset = find_sound_onset(sig, sample_rate)
        onsets.append(onset)

    # Compare all onsets to our reference microphone
    ref_onset = onsets[reference_idx]

    tdoas_samples = []
    for i, onset in enumerate(onsets):
        if i == reference_idx:
            continue
        # How many samples earlier/later this mic heard the sound
        # Negative means this mic heard it BEFORE the reference
        tdoa = onset - ref_onset
        tdoas_samples.append(tdoa)

    # Convert from sample counts to seconds
    tdoas_time = np.array(tdoas_samples) / sample_rate

    return tdoas_time


def test_doa_from_files(audio_files: list):
    """Test direction-finding with 4 pre-recorded audio files."""
    # Need exactly 4 files in the right order
    if len(audio_files) != 4:
        print("Error: Need exactly 4 audio files")
        print("Order: bottom_left, bottom_right, top_left, top_right")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("TDOA Direction of Arrival - Offline Test")
    print("=" * 80)

    # Load all 4 audio files
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

    # Get where each microphone is positioned (from our config)
    mic_positions = np.array([
        MIC_POSITIONS['bottom_left'],
        MIC_POSITIONS['bottom_right'],
        MIC_POSITIONS['top_left'],
        MIC_POSITIONS['top_right']
    ])
    
    print("\nMicrophone Array Setup:")
    for name, pos in zip(positions, mic_positions):
        print(f"  {name}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) m")

    # Find timing differences using onset detection
    print("\nFinding timing differences...")
    tdoas = calculate_tdoas_from_onsets(signals, original_sr, reference_idx=0)

    print(f"\n  Timing Results (compared to Bottom-Left mic):")
    print(f"    Bottom-Right: {tdoas[0]*1e6:8.2f} μs ({tdoas[0]*SPEED_OF_SOUND:.4f} m)")
    print(f"    Top-Left:     {tdoas[1]*1e6:8.2f} μs ({tdoas[1]*SPEED_OF_SOUND:.4f} m)")
    print(f"    Top-Right:    {tdoas[2]*1e6:8.2f} μs ({tdoas[2]*SPEED_OF_SOUND:.4f} m)")

    # Set up the direction calculator with our microphone positions
    doa_calc = DOACalculator(mic_positions=mic_positions)

    # Convert timing differences into a direction
    print("\nCalculating sound direction...")
    direction_vector, azimuth, elevation = doa_calc.calculate_direction(tdoas)

    # Show the final results!
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nDirection Vector (3D coordinates):")
    print(f"  x = {direction_vector[0]:7.4f} (east-west)")
    print(f"  y = {direction_vector[1]:7.4f} (north-south)")
    print(f"  z = {direction_vector[2]:7.4f} (up-down)")
    print(f"  Length = {np.linalg.norm(direction_vector):.4f} (should be 1.0)")

    print(f"\nDirection (compass + elevation):")
    print(f"  Azimuth:   {azimuth:6.2f}° (0° = east, 90° = north, 180° = west, 270° = south)")
    print(f"  Elevation: {elevation:6.2f}° (0° = horizontal, 90° = straight up, -90° = straight down)")

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


