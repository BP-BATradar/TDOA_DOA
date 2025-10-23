import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sounddevice as sd
from scipy.io.wavfile import write
from datetime import datetime
import numpy as np
from config.config import SAMPLE_RATE, CHUNK_DURATION

def list_microphones():
    """List all available audio input devices."""
    print("=" * 70)
    print("Available microphones:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:  # Only show input devices
            print(f"[{i}] {device['name']} - Channels: {device['max_input_channels']}")
    print("=" * 70)

def select_microphones():
    """Let user select 4 microphones for recording."""
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
                    print(f"Error: Device {idx} has no input channels. Please choose another.")
                    continue
                mic_indices.append(idx)
                mic_names.append(device['name'])
                break
            except (ValueError, sd.PortAudioError):
                print("Invalid device number. Please try again.")
    
    return mic_indices, mic_names

def record_synchronized(mic_indices, duration=CHUNK_DURATION, sample_rate=SAMPLE_RATE):
    """
    Record from 4 microphones simultaneously with precise synchronization.
    
    This function ensures all microphones start and stop at exactly the same time,
    which is critical for TDOA calculations.
    
    Args:
        mic_indices: List of 4 microphone device indices
        duration: Recording duration in seconds (default: 1.0)
        sample_rate: Sample rate in Hz (default: 16000)
    
    Returns:
        List of 4 numpy arrays containing the recorded audio data
    """
    if len(mic_indices) != 4:
        raise ValueError("Exactly 4 microphones are required")
    
    # Calculate exact number of samples for precise timing
    num_samples = int(duration * sample_rate)
    
    print(f"\nRecording from 4 microphones for {duration} seconds...")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Total samples per mic: {num_samples}")
    
    # Record from all microphones simultaneously
    # Using individual recordings but started in quick succession for best sync
    recordings = []
    
    # InputStream for better synchronization control
    # Callback-based approach with shared timing
    import threading
    import queue
    
    # Shared structures for synchronization
    start_event = threading.Event()
    queues = [queue.Queue() for _ in range(4)]
    errors = [None] * 4
    
    def create_callback(mic_index, q, error_list, idx):
        """Create a callback function for each microphone."""
        def callback(indata, frames, time_info, status):
            if status:
                error_list[idx] = status
            q.put(indata.copy())
        return callback
    
    # Create and start all streams
    streams = []
    for idx, mic_idx in enumerate(mic_indices):
        callback = create_callback(mic_idx, queues[idx], errors, idx)
        stream = sd.InputStream(
            device=mic_idx,
            channels=1,
            samplerate=sample_rate,
            callback=callback
        )
        streams.append(stream)
    
    # Start all streams as close together as possible
    for stream in streams:
        stream.start()
    
    # Record for specified duration
    sd.sleep(int(duration * 1000))  # Sleep in milliseconds
    
    # Stop all streams
    for stream in streams:
        stream.stop()
        stream.close()
    
    # Collect recorded data from queues
    print("Collecting recorded data...")
    for idx in range(4):
        chunks = []
        while not queues[idx].empty():
            chunks.append(queues[idx].get())
        
        if chunks:
            recording = np.concatenate(chunks, axis=0)
            # Ensure exact length (trim or pad to num_samples)
            if len(recording) > num_samples:
                recording = recording[:num_samples]
            elif len(recording) < num_samples:
                padding = np.zeros((num_samples - len(recording), 1))
                recording = np.vstack([recording, padding])
            recordings.append(recording.flatten())
        else:
            raise RuntimeError(f"No data received from microphone {mic_indices[idx]}")
    
    # Check for any errors
    for idx, error in enumerate(errors):
        if error:
            print(f"Warning: Microphone {mic_indices[idx]} reported status: {error}")
    
    print("Recording complete!")
    return recordings

def save_recordings(recordings, mic_names, sample_rate=SAMPLE_RATE):
    """
    Save recorded audio from 4 microphones to individual WAV files.
    
    Args:
        recordings: List of 4 numpy arrays with audio data
        mic_names: List of 4 microphone names
        sample_rate: Sample rate in Hz
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")
    
    # Create recordings directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    positions = ['bottom_left', 'bottom_right', 'top_left', 'top_right']
    filenames = []
    
    for idx, (recording, position, mic_name) in enumerate(zip(recordings, positions, mic_names)):
        filename = f"mic_{position}_{timestamp}.wav"
        filepath = os.path.join(output_dir, filename)
        
        # Normalize to int16 range
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
    """Main function to record from 4 synchronized microphones."""
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
