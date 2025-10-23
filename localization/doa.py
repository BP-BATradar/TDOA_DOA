"""
DOA estimation from TDOAs using a small least‑squares system.
The math assumes far-field (plane wave) which is a good fit for distant drones.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import Tuple, Optional
from config.config import SPEED_OF_SOUND, REFERENCE_MIC_INDEX


class DOACalculator:
    # Calculates Direction of Arrival (DOA) from Time Difference of Arrival (TDOA) measurements
    # Uses geometric solver for a square microphone array
    
    def __init__(self, mic_positions: np.ndarray, reference_mic: int = REFERENCE_MIC_INDEX):
        self.mic_positions = np.array(mic_positions)
        self.reference_mic = reference_mic
        self.num_mics = len(mic_positions)
        
        if self.num_mics < 3:
            raise ValueError("At least 3 microphones required for DOA calculation")
    
    def calculate_direction(self, tdoas: np.ndarray) -> Tuple[np.ndarray, float, float]:
        # Calculate direction of arrival from TDOA measurements
        # Uses a least-squares approach to solve for the direction vector
        range_differences = tdoas * SPEED_OF_SOUND
        
        # Get reference microphone position
        ref_pos = self.mic_positions[self.reference_mic]
        
        # Positions of non-reference mics
        other_mics = [i for i in range(self.num_mics) if i != self.reference_mic]
        other_positions = self.mic_positions[other_mics]
        
        # Build matrix A and vector b for least squares: A * d = b
        A = other_positions - ref_pos
        b = -range_differences
        
        # Solve least squares (small 3x3 system); rcond=None uses default cutoff
        direction_vector, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        
        # Normalize direction vector
        norm = np.linalg.norm(direction_vector)
        if norm > 0:
            direction_vector = direction_vector / norm
        else:
            # Default to upward direction if calculation fails
            direction_vector = np.array([0.0, 0.0, 1.0])
        
        # Convert to azimuth and elevation
        azimuth, elevation = self.vector_to_angles(direction_vector)
        
        return direction_vector, azimuth, elevation
    
    @staticmethod
    def vector_to_angles(direction_vector: np.ndarray) -> Tuple[float, float]:
        # Convert a direction vector to azimuth and elevation angles
        x, y, z = direction_vector
        
        # Azimuth: angle in XY plane from X axis
        azimuth = np.arctan2(y, x)
        azimuth_deg = np.degrees(azimuth)
        
        # Convert to 0-360 range
        if azimuth_deg < 0:
            azimuth_deg += 360
        
        # Elevation: angle from XY plane
        xy_distance = np.sqrt(x**2 + y**2)
        elevation = np.arctan2(z, xy_distance)
        elevation_deg = np.degrees(elevation)
        
        return azimuth_deg, elevation_deg
    
    @staticmethod
    def angles_to_vector(azimuth: float, elevation: float) -> np.ndarray:
        # Convert azimuth and elevation angles to a direction vector
        # Convert to radians
        az_rad = np.radians(azimuth)
        el_rad = np.radians(elevation)
        
        # Calculate direction vector
        x = np.cos(el_rad) * np.cos(az_rad)
        y = np.cos(el_rad) * np.sin(az_rad)
        z = np.sin(el_rad)
        
        return np.array([x, y, z])
    
    def __repr__(self):
        return (f"DOACalculator(num_mics={self.num_mics}, "
                f"reference_mic={self.reference_mic})")


