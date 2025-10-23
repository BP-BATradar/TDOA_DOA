"""
Settings for our drone direction-finding system using time differences between microphones.
Everything here controls how we detect where sounds are coming from.
"""

# Audio setup - matches what the drone classifier expects
SAMPLE_RATE = 16000  # How many audio samples we capture per second
CHUNK_DURATION = 1.0  # How long each audio chunk is (in seconds)
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)  # Total samples per chunk

# Physics - how fast sound travels through air
SPEED_OF_SOUND = 343.0  # meters per second (typical room temperature)

# Where our microphones are positioned (square setup, 1 meter apart)
# All mics are at ground level (z=0)
ARRAY_SIZE = 1.0  # Distance between microphones in meters
MIC_POSITIONS = {
    'top_left': (0.0, ARRAY_SIZE, 0.0),
    'top_right': (ARRAY_SIZE, ARRAY_SIZE, 0.0),
    'bottom_left': (0.0, 0.0, 0.0),
    'bottom_right': (ARRAY_SIZE, 0.0, 0.0)
}

# Processing settings
REFERENCE_MIC_INDEX = 0  # Use bottom-left mic as our timing reference
MAX_TDOA = ARRAY_SIZE * 1.414 / SPEED_OF_SOUND  # Max time delay (diagonal distance)

# Cross-correlation settings - helps us find timing differences reliably
USE_GCC_PHAT = True  # Phase transform makes correlation more robust to noise
CORRELATION_MAX_LAG = int(MAX_TDOA * SAMPLE_RATE * 1.5)  # How far to search for correlation peaks


