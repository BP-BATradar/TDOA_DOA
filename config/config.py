"""
Configuration file for TDOA-based Direction of Arrival system.
Contains all system constants and processing parameters.
"""

# Audio Configuration
SAMPLE_RATE = 16000  # Hz, matches RNN drone classifier
CHUNK_DURATION = 1.0  # seconds, aligns with RNN classifier
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)  # samples

# Physical Constants
SPEED_OF_SOUND = 343.0  # m/s at 20°C

# Microphone Array Geometry (Square Array, 1m sides)
# All microphones on ground (z=0)
ARRAY_SIZE = 1.0  # meters
MIC_POSITIONS = {
    'top_left': (0.0, ARRAY_SIZE, 0.0),
    'top_right': (ARRAY_SIZE, ARRAY_SIZE, 0.0),
    'bottom_left': (0.0, 0.0, 0.0),
    'bottom_right': (ARRAY_SIZE, 0.0, 0.0)
}

# Processing Parameters
REFERENCE_MIC_INDEX = 0  # Use first mic (bottom-left at origin 0,0,0) as reference
MAX_TDOA = ARRAY_SIZE * 1.414 / SPEED_OF_SOUND  # Maximum possible TDOA (diagonal distance)

# GCC-PHAT Parameters
USE_GCC_PHAT = True  # Use phase transform for robustness
CORRELATION_MAX_LAG = int(MAX_TDOA * SAMPLE_RATE * 1.5)  # Samples to search for correlation peak


