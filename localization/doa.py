"""
Finds where sounds are coming from using timing differences between microphones.
Works best for distant sounds (like drones far away) - assumes sound waves are basically flat.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import Tuple, Optional
from config.config import SPEED_OF_SOUND, REFERENCE_MIC_INDEX


class DOACalculator:
    """Figures out what direction a sound is coming from using timing differences between mics."""

    def __init__(self, mic_positions: np.ndarray, reference_mic: int = REFERENCE_MIC_INDEX):
        """Set up the calculator with microphone positions."""
        self.mic_positions = np.array(mic_positions)
        self.reference_mic = reference_mic
        self.num_mics = len(mic_positions)

        if self.num_mics < 3:
            raise ValueError("Need at least 3 mics to figure out direction")
    
    def calculate_direction(self, tdoas: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Figure out direction from timing differences between microphones."""
        # Convert time delays to distance differences (how much farther sound traveled)
        range_differences = tdoas * SPEED_OF_SOUND

        # Get positions relative to our reference microphone
        ref_pos = self.mic_positions[self.reference_mic]
        other_mics = [i for i in range(self.num_mics) if i != self.reference_mic]
        other_positions = self.mic_positions[other_mics]

        # Set up math problem: find direction that explains the distance differences
        A = other_positions - ref_pos  # How other mics are positioned relative to reference
        b = -range_differences         # How much sound arrived early/late

        # Solve using least squares (small 3x3 system)
        direction_vector, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        # Make sure direction vector has length 1 (unit vector)
        norm = np.linalg.norm(direction_vector)
        if norm > 0:
            direction_vector = direction_vector / norm
        else:
            # If math fails, default to straight up
            direction_vector = np.array([0.0, 0.0, 1.0])

        # Convert to compass direction and up/down angle
        azimuth, elevation = self.vector_to_angles(direction_vector)

        return direction_vector, azimuth, elevation
    
    @staticmethod
    def vector_to_angles(direction_vector: np.ndarray) -> Tuple[float, float]:
        """Convert 3D direction vector to compass bearing and up/down angle."""
        x, y, z = direction_vector

        # Compass direction: angle in horizontal plane from east
        azimuth = np.arctan2(y, x)
        azimuth_deg = np.degrees(azimuth)

        # Make it 0-360 degrees (0° = east, 90° = north, etc.)
        if azimuth_deg < 0:
            azimuth_deg += 360

        # Up/down angle: how far above/below horizontal
        horizontal_distance = np.sqrt(x**2 + y**2)
        elevation = np.arctan2(z, horizontal_distance)
        elevation_deg = np.degrees(elevation)

        return azimuth_deg, elevation_deg

    @staticmethod
    def angles_to_vector(azimuth: float, elevation: float) -> np.ndarray:
        """Convert compass bearing and up/down angle back to 3D direction vector."""
        # Convert degrees to radians for math functions
        az_rad = np.radians(azimuth)
        el_rad = np.radians(elevation)

        # Break down into x,y,z components
        x = np.cos(el_rad) * np.cos(az_rad)  # east-west component
        y = np.cos(el_rad) * np.sin(az_rad)  # north-south component
        z = np.sin(el_rad)                   # up-down component

        return np.array([x, y, z])
    
    def __repr__(self):
        return f"DOA Calculator: {self.num_mics} mics, reference mic #{self.reference_mic}"


