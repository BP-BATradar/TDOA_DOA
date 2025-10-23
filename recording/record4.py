import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sounddevice as sd
from scipy.io.wavfile import write
from scipy import signal
from datetime import datetime
import numpy as np
from config.config import SAMPLE_RATE, CHUNK_DURATION
from recording.multi_device_recorder import MultiDeviceRecorder

def list_microphones():
    """Show all microphones available on your system."""
    print("=" * 70)
    print("Available microphones:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:  # Only show devices that can record
            print(f"[{i}] {device['name']} - Channels: {device['max_input_channels']}")
    print("=" * 70)

def select_microphones():
    """Let user pick which 4 microphones to use for recording."""
    list_microphones()

    mic_indices = []
    mic_names = []

    print("\nPlease select 4 microphones by entering their device numbers:")
    mic_positions = ['bottom_left', 'bottom_right', 'top_left', 'top_right']

    for position in mic_positions:
        while True:
            try:
                idx = int(input(f"Enter device number for {position} microphone: "))
                device = sd.query_devices(idx)
                if device['max_input_channels'] == 0:
                    print(f"Error: Device {idx} can't record audio. Please choose another.")
                    continue
                mic_indices.append(idx)
                mic_names.append(device['name'])
                break
            except (ValueError, sd.PortAudioError):
                print("Invalid device number. Please try again.")

    return mic_indices, mic_names

def record_synchronized(mic_indices, duration=CHUNK_DURATION, sample_rate=SAMPLE_RATE, verbose=True):
    """
    Record a short audio clip from 4 separate microphones, perfectly synchronized.

    How it works:
    - Keeps all microphone streams open during recording (no stopping/starting)
    - Uses hardware timing to align all mics to the exact same starting moment
    - Gives you one clean audio chunk per microphone, ready for TDOA analysis

    Args:
        mic_indices: Which 4 microphone devices to use (by their system numbers)
        duration: How long to record in seconds (default: 1.0)
        sample_rate: Audio quality in Hz (default: 16000)
        verbose: Show detailed info about the alignment process (default: True)

    Returns:
        List of 4 numpy arrays with the synchronized audio, one per microphone
    """
    if len(mic_indices) != 4:
        raise ValueError("Exactly 4 microphones are required")

    num_samples = int(duration * sample_rate)

    if verbose:
        print(f"\n" + "="*70)
        print(f"HARDWARE-TIMED SYNCHRONIZED RECORDING")
        print(f"="*70)
        print(f"Duration: {duration} seconds")
        print(f"Sample rate: {sample_rate} Hz")
        print(f"Target samples per mic: {num_samples}")
        print(f"Recording from {len(mic_indices)} separate devices...")

    # Set up the recorder that handles all the complex synchronization
    recorder = MultiDeviceRecorder(
        mic_indices=mic_indices,
        sample_rate=sample_rate,
        chunk_duration=duration,
        blocksize=256,
    )
    recorder.start()
    try:
        block = recorder.read_chunk(timeout=5.0)  # shape: (num_samples, 4)
    finally:
        recorder.stop()

    # Make sure we have exactly the right amount of audio
    if block.shape[0] > num_samples:
        block = block[:num_samples, :]  # Trim if too long
    elif block.shape[0] < num_samples:
        # Pad with silence if too short
        pad = np.zeros((num_samples - block.shape[0], block.shape[1]), dtype=block.dtype)
        block = np.vstack([block, pad])

    # Split the synchronized audio back into individual microphone recordings
    aligned_recordings = [block[:, i].copy() for i in range(4)]

    if verbose:
        print(f"\n" + "="*70)
        print(f"All 4 recordings perfectly aligned to {num_samples} samples!")
        print(f"="*70)
        verify_synchronization(aligned_recordings, sample_rate)

    return aligned_recordings

def verify_synchronization(recordings, sample_rate, max_expected_delay_ms=5.0):
    """
    Check how well synchronized our microphone recordings are.

    Uses cross-correlation to measure timing differences between each pair of mics.
    If everything is working well, delays should be very small (under 5ms).

    Args:
        recordings: List of 4 audio arrays from synchronized microphones
        sample_rate: How many samples per second our audio has
        max_expected_delay_ms: Max delay we consider "good" sync (default: 5ms)
    """
    print("\nChecking how well synchronized our recordings are...")

    max_delay_samples = int(max_expected_delay_ms * sample_rate / 1000)
    sync_issues = []

    # Compare every pair of microphones to check timing
    mic_pairs = [
        (0, 1, "bottom_left vs bottom_right"),
        (0, 2, "bottom_left vs top_left"),
        (0, 3, "bottom_left vs top_right"),
        (1, 2, "bottom_right vs top_left"),
        (1, 3, "bottom_right vs top_right"),
        (2, 3, "top_left vs top_right")
    ]

    for mic1, mic2, pair_name in mic_pairs:
        # Find how much one signal is delayed compared to the other
        correlation = signal.correlate(recordings[mic1], recordings[mic2], mode='full')
        lags = signal.correlation_lags(len(recordings[mic1]), len(recordings[mic2]), mode='full')

        # Find where the signals match best
        peak_idx = np.argmax(np.abs(correlation))
        delay_samples = lags[peak_idx]
        delay_ms = delay_samples * 1000 / sample_rate
        correlation_strength = correlation[peak_idx] / (np.linalg.norm(recordings[mic1]) * np.linalg.norm(recordings[mic2]))

        # Check if timing difference is small enough
        status = "GOOD" if abs(delay_samples) <= max_delay_samples else "POOR"
        print(f"  {pair_name}: {delay_ms:+6.2f}ms delay, correlation: {correlation_strength:.3f} {status}")

        if abs(delay_samples) > max_delay_samples:
            sync_issues.append((pair_name, delay_ms))

    if sync_issues:
        print(f"\nWARNING: SYNCHRONIZATION ISSUES DETECTED")
        print(f"   These microphone pairs have timing delays over {max_expected_delay_ms}ms:")
        for pair, delay in sync_issues:
            print(f"   - {pair}: {delay:+.2f}ms delay")
        print(f"   This could affect direction-finding accuracy. Try:")
        print(f"   - Using mics from the same audio interface")
        print(f"   - Lowering system audio buffer settings")
        print(f"   - Using professional audio hardware")
    else:
        print(f"\nSynchronization looks great! All delays under {max_expected_delay_ms}ms.")

def save_recordings(recordings, mic_names, sample_rate=SAMPLE_RATE):
    """
    Save each microphone's audio to its own WAV file.

    Creates a timestamped folder and saves each recording with a descriptive name
    showing which position and device it came from.

    Args:
        recordings: List of 4 numpy arrays with the audio data
        mic_names: List of 4 microphone device names
        sample_rate: Audio sample rate in Hz
    """
    # Create unique timestamp for this recording session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")

    # Make sure the recordings folder exists
    os.makedirs(output_dir, exist_ok=True)

    positions = ['bottom_left', 'bottom_right', 'top_left', 'top_right']
    filenames = []

    for idx, (recording, position, mic_name) in enumerate(zip(recordings, positions, mic_names)):
        filename = f"mic_{position}_{timestamp}.wav"
        filepath = os.path.join(output_dir, filename)

        # Convert floating-point audio to 16-bit integer format for WAV file
        audio_data = (recording * 32767).astype(np.int16)
        write(filepath, sample_rate, audio_data)

        filenames.append(filepath)
        print(f"Saved: {filepath}")
        print(f"  Position: {position}")
        print(f"  Device: {mic_name}")
        print(f"  Samples: {len(recording)}")
        print(f"  Duration: {len(recording)/sample_rate:.3f}s")
    
    return filenames

def main():
    """Record a synchronized audio clip from 4 microphones and save to files."""
    print("=" * 70)
    print("4-Microphone Synchronized Recording System")
    print("For TDOA-based Direction of Arrival")
    print("=" * 70)

    # Select microphones
    mic_indices, mic_names = select_microphones()
    
    print("\n" + "=" * 70)
    print("Selected microphones:")
    positions = ['bottom_left', 'bottom_right', 'top_left', 'top_right']
    for pos, idx, name in zip(positions, mic_indices, mic_names):
        print(f"  {pos}: [{idx}] {name}")
    print("=" * 70)

    # Confirm before recording
    input("\nPress Enter to start synchronized recording...")
    
    # Record from all 4 microphones simultaneously
    recordings = record_synchronized(mic_indices)
    
    # Save recordings
    print("\nSaving recordings...")
    filenames = save_recordings(recordings, mic_names)
    
    print("\n" + "=" * 70)
    print("Recording session complete!")
    print(f"Files saved to: {os.path.dirname(filenames[0])}")
    print("=" * 70)

if __name__ == "__main__":
    main()
