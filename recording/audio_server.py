import os
import sys
from datetime import datetime

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import CHUNK_DURATION, MIC_POSITIONS, REFERENCE_MIC_INDEX, SAMPLE_RATE
from localization.doa import DOACalculator
from localization.tdoa import TDOACalculator
from recording.record4 import select_microphones
from recording.multi_device_recorder import MultiDeviceRecorder


MIC_ORDER = [
    "bottom_left",
    "bottom_right",
    "top_left",
    "top_right",
]

REFERENCE_LABEL = MIC_ORDER[REFERENCE_MIC_INDEX]  # Which mic we use as timing reference
TDOA_LABELS = [label for idx, label in enumerate(MIC_ORDER) if idx != REFERENCE_MIC_INDEX]


def build_mic_position_array(order: list[str]) -> np.ndarray:
    """Get microphone positions in the right order from config."""
    try:
        positions = [MIC_POSITIONS[label] for label in order]
    except KeyError as exc:
        raise KeyError(f"Missing microphone position for: {exc.args[0]}") from exc
    return np.asarray(positions, dtype=float)


def print_microphone_selection(mic_indices: list[int], mic_names: list[str]) -> None:
    """Show which microphones are selected and which is our timing reference."""
    print("\nSelected microphones:")
    for label, device_idx, device_name in zip(MIC_ORDER, mic_indices, mic_names):
        role = "(reference)" if label == REFERENCE_LABEL else ""
        print(f"  {label:>12}: [{device_idx}] {device_name} {role}")


def format_tdoa_output(tdoas: np.ndarray) -> str:
    """Make timing differences look nice and readable."""
    lines = ["TDOA relative to bottom_left (reference):"]
    for label, tdoa in zip(TDOA_LABELS, tdoas):
        lines.append(
            f"  {label:>12}: {tdoa:+.6f} s  ({tdoa * 1e6:+8.2f} µs)"
        )
    return "\n".join(lines)


def format_doa_output(direction_vector: np.ndarray, azimuth_deg: float) -> str:
    """Make direction results look nice and easy to understand."""
    x, y, z = direction_vector
    return (
        "Direction of Arrival (DOA):\n"
        f"  Azimuth: {azimuth_deg:6.2f}°\n"
        f"  Direction vector: [x={x:+.4f}, y={y:+.4f}, z={z:+.4f}]"
    )


def main() -> None:
    """Keep recording audio and calculating sound direction in real-time."""
    print("=" * 80)
    print("Real-Time TDOA/DOA Audio Server")
    print("=" * 80)

    # Let user pick which microphones to use
    mic_indices, mic_names = select_microphones()
    print_microphone_selection(mic_indices, mic_names)

    # Set up calculators for timing differences and direction finding
    mic_positions_array = build_mic_position_array(MIC_ORDER)
    tdoa_calculator = TDOACalculator(
        sample_rate=SAMPLE_RATE,
        reference_mic=REFERENCE_MIC_INDEX,
    )
    doa_calculator = DOACalculator(
        mic_positions=mic_positions_array,
        reference_mic=REFERENCE_MIC_INDEX,
    )

    # Make sure we're using 1-second chunks for best results
    duration = float(CHUNK_DURATION)
    if not np.isclose(duration, 1.0):
        print(
            "Warning: CHUNK_DURATION in config is not 1.0s. "
            "Overriding to 1.0s for synchronized processing."
        )
        duration = 1.0

    print("\nStarting continuous recording...")
    print("Press Ctrl+C to stop.\n")

    # Start recorder that keeps all mic streams in sync
    recorder = MultiDeviceRecorder(
        mic_indices=mic_indices,
        sample_rate=SAMPLE_RATE,
        chunk_duration=duration,
        blocksize=256,
    )
    recorder.start()

    # Do one test chunk to make sure everything is aligned properly
    _ = recorder.read_chunk(timeout=5.0)
    print("\nStreams running and initial alignment applied. Entering continuous mode...\n")

    chunk_counter = 0

    try:
        while True:
            chunk_counter += 1
            chunk_start = datetime.now()

            # Get synchronized audio from all microphones
            audio_block = recorder.read_chunk(timeout=5.0)

            # Figure out timing differences between mics
            tdoas = tdoa_calculator.calculate_tdoa(audio_block)

            # Convert timing differences to direction
            direction_vector, azimuth_deg, _ = doa_calculator.calculate_direction(tdoas)

            # Show results in real-time
            print("-" * 80)
            print(
                f"Chunk #{chunk_counter:04d} | {chunk_start.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            print(format_tdoa_output(tdoas))
            print(format_doa_output(direction_vector, azimuth_deg))
            sys.stdout.flush()

    except KeyboardInterrupt:
        print("\nStopping audio server...")
    finally:
        recorder.stop()


if __name__ == "__main__":
    main()

