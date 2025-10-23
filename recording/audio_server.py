import os
import sys
from datetime import datetime

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import CHUNK_DURATION, MIC_POSITIONS, REFERENCE_MIC_INDEX, SAMPLE_RATE
from localization.doa import DOACalculator
from localization.tdoa import TDOACalculator
from recording.record4 import record_synchronized, select_microphones


MIC_ORDER = [
    "bottom_left",
    "bottom_right",
    "top_left",
    "top_right",
]

REFERENCE_LABEL = MIC_ORDER[REFERENCE_MIC_INDEX]
TDOA_LABELS = [label for idx, label in enumerate(MIC_ORDER) if idx != REFERENCE_MIC_INDEX]


def build_mic_position_array(order: list[str]) -> np.ndarray:
    """Create an ordered array of microphone positions from config."""
    try:
        positions = [MIC_POSITIONS[label] for label in order]
    except KeyError as exc:
        raise KeyError(f"Missing microphone position for: {exc.args[0]}") from exc
    return np.asarray(positions, dtype=float)


def print_microphone_selection(mic_indices: list[int], mic_names: list[str]) -> None:
    """Display the selected microphones and their roles."""
    print("\nSelected microphones:")
    for label, device_idx, device_name in zip(MIC_ORDER, mic_indices, mic_names):
        role = "(reference)" if label == REFERENCE_LABEL else ""
        print(f"  {label:>12}: [{device_idx}] {device_name} {role}")


def format_tdoa_output(tdoas: np.ndarray) -> str:
    """Format TDOA values (seconds) into a readable multi-line string."""
    lines = ["TDOA relative to bottom_left (reference):"]
    for label, tdoa in zip(TDOA_LABELS, tdoas):
        lines.append(
            f"  {label:>12}: {tdoa:+.6f} s  ({tdoa * 1e6:+8.2f} µs)"
        )
    return "\n".join(lines)


def format_doa_output(direction_vector: np.ndarray, azimuth_deg: float) -> str:
    """Format DOA results into a readable string."""
    x, y, z = direction_vector
    return (
        "Direction of Arrival (DOA):\n"
        f"  Azimuth: {azimuth_deg:6.2f}°\n"
        f"  Direction vector: [x={x:+.4f}, y={y:+.4f}, z={z:+.4f}]"
    )


def main() -> None:
    """Run continuous synchronized recording with live TDOA/DOA estimation."""
    print("=" * 80)
    print("Real-Time TDOA/DOA Audio Server")
    print("=" * 80)

    mic_indices, mic_names = select_microphones()
    print_microphone_selection(mic_indices, mic_names)

    mic_positions_array = build_mic_position_array(MIC_ORDER)
    tdoa_calculator = TDOACalculator(
        sample_rate=SAMPLE_RATE,
        reference_mic=REFERENCE_MIC_INDEX,
    )
    doa_calculator = DOACalculator(
        mic_positions=mic_positions_array,
        reference_mic=REFERENCE_MIC_INDEX,
    )

    duration = float(CHUNK_DURATION)
    if not np.isclose(duration, 1.0):
        print(
            "Warning: CHUNK_DURATION in config is not 1.0s. "
            "Overriding to 1.0s for synchronized processing."
        )
        duration = 1.0

    print("\nStarting continuous recording...")
    print("Press Ctrl+C to stop.\n")

    chunk_counter = 0

    try:
        while True:
            chunk_counter += 1
            chunk_start = datetime.now()

            recordings = record_synchronized(
                mic_indices,
                duration=duration,
                sample_rate=SAMPLE_RATE,
            )

            audio_block = np.column_stack(recordings)
            tdoas = tdoa_calculator.calculate_tdoa(audio_block)
            direction_vector, azimuth_deg, _ = doa_calculator.calculate_direction(tdoas)

            print("-" * 80)
            print(
                f"Chunk #{chunk_counter:04d} | {chunk_start.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            print(format_tdoa_output(tdoas))
            print(format_doa_output(direction_vector, azimuth_deg))
            sys.stdout.flush()

    except KeyboardInterrupt:
        print("\nStopping audio server...")


if __name__ == "__main__":
    main()

