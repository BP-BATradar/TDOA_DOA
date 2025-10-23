# TDOA-Based Direction of Arrival (DOA) System

A real-time audio localization system using Time Difference of Arrival (TDOA) measurements with a synchronized 4-microphone array. This system calculates the direction and distance to sound sources with precise timing synchronization.

## Overview

This project implements a **TDOA (Time Difference of Arrival)** based localization system that:
- Records audio from 4 synchronized microphones arranged in a square
- Calculates time delays between microphone pairs using GCC-PHAT
- Estimates the direction of arrival (azimuth and elevation)
- Provides real-time continuous processing with 1-second chunks

### Key Features

✅ **Synchronized Recording**: All 4 microphones record exactly 1 second in perfect sync  
✅ **GCC-PHAT Processing**: Robust cross-correlation with phase transform  
✅ **Real-Time DOA**: Live azimuth and 3D direction vector estimation  
✅ **Modular Design**: Reusable components for recording, TDOA, and DOA  
✅ **Flexible Input**: Support for any available audio devices  

## System Architecture

### Microphone Array Configuration

```
        Top View (1m × 1m square)

top_left (0, 1, 0)              top_right (1, 1, 0)
    TL ━━━━━━━━━━━━━━━━━━━━ TR
    ┃                          ┃
    ┃                          ┃
    ┃                          ┃
    BL ━━━━━━━━━━━━━━━━━━━━ BR
bottom_left (0, 0, 0)      bottom_right (1, 0, 0)

Reference Microphone: bottom_left (0, 0, 0)
```

- **Array Size**: 1 meter × 1 meter square
- **Reference Mic**: Bottom-left (index 0)
- **All mics at ground level** (z = 0)

### Processing Pipeline

```
┌─────────────────┐
│ 4 Microphones   │
└────────┬────────┘
         │ Synchronized Recording (1s chunks)
         ↓
┌─────────────────┐
│ Audio Streams   │ (16 kHz, 16-bit mono)
│ (4 signals)     │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ TDOA Calculator │ (GCC-PHAT)
└────────┬────────┘
         │
         ↓
┌─────────────────────┐
│ TDOA Values (3)     │ (Relative to reference mic)
│ BR, TL, TR delays   │
└────────┬────────────┘
         │
         ↓
┌──────────────────┐
│ DOA Calculator   │ (Least-squares solver)
└────────┬─────────┘
         │
         ↓
┌────────────────────────┐
│ Direction Results      │
│ - Azimuth (degrees)    │
│ - Elevation (degrees)  │
│ - Direction Vector     │
└────────────────────────┘
```

## Installation

### Requirements

```bash
Python 3.8+
numpy
scipy
sounddevice
```

### Setup

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Verify audio devices**:
```bash
python -c "import sounddevice; print(sounddevice.query_devices())"
```

## Usage

### Option 1: Real-Time Audio Server (Continuous DOA)

For continuous live direction estimation:

```bash
python recording/audio_server.py
```

**Process**:
1. Lists all available audio input devices
2. Prompts you to select 4 microphones (bottom-left, bottom-right, top-left, top-right)
3. Begins continuous 1-second synchronized recording
4. For each chunk, displays:
   - TDOA values (relative to bottom-left reference)
   - Azimuth angle
   - 3D direction vector

**Output Example**:
```
================================================================================
Real-Time TDOA/DOA Audio Server
================================================================================

Selected microphones:
    bottom_left: [0] Built-in Microphone (reference)
  bottom_right: [1] USB Audio Device
      top_left: [2] Microphone Array
     top_right: [3] External Mic

Starting continuous recording...
Press Ctrl+C to stop.

--------------------------------------------------------------------------------
Chunk #0001 | 2025-10-23 10:15:30
TDOA relative to bottom_left (reference):
    bottom_right: +0.000234 s  (  +234.00 µs)
        top_left: -0.000145 s  (  -145.00 µs)
       top_right: +0.000089 s  (   +89.00 µs)
Direction of Arrival (DOA):
  Azimuth: 45.32°
  Direction vector: [x=+0.7065, y=+0.7076, z=-0.0015]
```

### Option 2: Single Recording Session

For a one-time 1-second recording and file save:

```bash
python recording/record4.py
```

**Features**:
- Select 4 microphones interactively
- Record exactly 1 second
- Save individual WAV files for each microphone
- Files saved to `recording/recordings/`

**Output files**: 
```
mic_bottom_left_20251023_101530.wav
mic_bottom_right_20251023_101530.wav
mic_top_left_20251023_101530.wav
mic_top_right_20251023_101530.wav
```

### Option 3: Offline Testing

Test TDOA/DOA calculation on pre-recorded files:

```bash
# Test TDOA
python test/test_tdoa.py BL.wav BR.wav TL.wav TR.wav

# Test DOA
python test/test_doa.py BL.wav BR.wav TL.wav TR.wav
```

**Requirements**:
- 4 audio files in order: bottom-left, bottom-right, top-left, top-right
- 1 second duration recommended
- 16 kHz sample rate recommended
- Mono (single channel)

## Configuration

Edit `config/config.py` to adjust system parameters:

```python
# Audio Configuration
SAMPLE_RATE = 16000        # Hz
CHUNK_DURATION = 1.0       # seconds

# Microphone Array Geometry
ARRAY_SIZE = 1.0           # meters (1m × 1m square)
MIC_POSITIONS = {
    'bottom_left': (0.0, 0.0, 0.0),
    'bottom_right': (1.0, 0.0, 0.0),
    'top_left': (0.0, 1.0, 0.0),
    'top_right': (1.0, 1.0, 0.0)
}

# Processing Parameters
REFERENCE_MIC_INDEX = 0    # Use bottom_left as reference
USE_GCC_PHAT = True        # Phase transform robustness
```

## Project Structure

```
TDOA/
├── config/
│   └── config.py                 # System configuration
├── localization/
│   ├── tdoa.py                   # TDOA calculation (GCC-PHAT)
│   └── doa.py                    # DOA calculation from TDOA
├── recording/
│   ├── __init__.py
│   ├── record4.py                # 4-mic synchronized recording
│   └── audio_server.py           # Real-time TDOA/DOA server
├── test/
│   ├── test_tdoa.py              # TDOA testing script
│   └── test_doa.py               # DOA testing script
├── requirements.txt              # Python dependencies
└── README.md                      # This file
```

## Technical Details

### TDOA Calculation (GCC-PHAT)

The system uses **Generalized Cross-Correlation with Phase Transform** (GCC-PHAT) to compute time delays:

1. Computes cross-spectrum between microphone pairs
2. Applies phase transform for noise robustness
3. Performs inverse FFT to get correlation
4. Finds correlation peak with sub-sample interpolation
5. Converts lag (samples) to time delay (seconds)

**Advantages**:
- Robust to noise and reverberation
- Handles different acoustic environments
- Sub-sample precision via quadratic interpolation

### DOA Calculation

From 3 TDOA values (BR, TL, TR relative to BL), the system:

1. Converts TDOAs to range differences (multiply by sound speed)
2. Solves least-squares system for direction vector
3. Normalizes to unit vector
4. Converts to azimuth and elevation angles

**Sound Speed**: 343 m/s (20°C)

### Angle Definitions

- **Azimuth**: 0° = East, 90° = North, 180° = West, 270° = South
- **Elevation**: 0° = Horizontal plane, 90° = Directly above, -90° = Directly below

## Important Notes

⚠️ **Synchronization is Critical**

The accuracy of TDOA and DOA estimates depends on precise 1-second synchronization:
- All 4 microphones must start/stop within microseconds of each other
- The system uses `sounddevice.InputStream` with callbacks for tight timing
- Records exactly 16,000 samples at 16 kHz (1.0 second)

⚠️ **Microphone Selection**

When running `audio_server.py` or `record4.py`:
- Select different physical devices (not channels of the same device)
- Ensure each device has at least 1 input channel
- Devices may have different latencies; choose consistently

⚠️ **Sound Speed Assumption**

The DOA calculation assumes:
- Speed of sound = 343 m/s (constant temperature ~20°C)
- Far-field approximation (plane waves, not point sources)
- No multipath reflections or reverberation

Adjust `SPEED_OF_SOUND` in config for different temperatures:
- 15°C: 340 m/s
- 20°C: 343 m/s  
- 25°C: 346 m/s
