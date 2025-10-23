import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import threading
from collections import deque
from typing import List, Optional

import numpy as np
import sounddevice as sd

from config.config import SAMPLE_RATE, CHUNK_DURATION, MAX_TDOA


class MultiDeviceRecorder:
    """
    Records from multiple microphones at once, keeping them perfectly synchronized.

    How it works:
    - Opens one audio stream per microphone and keeps them running continuously
    - Uses hardware timing (ADC times) to align all mics to the same starting point
    - Gives you fixed-size audio chunks without messing up the timing differences
      (which is crucial for TDOA calculations)

    Note: This doesn't fix clock drift between devices, but by keeping streams open
    and aligning them properly at the start, it keeps timing jitter very low compared
    to starting/stopping streams all the time.
    """

    def __init__(
        self,
        mic_indices: List[int],
        sample_rate: int = SAMPLE_RATE,
        chunk_duration: float = CHUNK_DURATION,
        blocksize: int = 256,
        max_buffer_chunks: int = 10,
    ) -> None:
        if len(mic_indices) != 4:
            raise ValueError("Exactly 4 microphones are required")

        self.mic_indices = mic_indices
        self.sample_rate = int(sample_rate)
        self.chunk_duration = float(chunk_duration)
        self.chunk_size = int(round(self.sample_rate * self.chunk_duration))
        self.blocksize = int(blocksize)
        self.max_buffer_samples = max(1, max_buffer_chunks) * self.chunk_size

        self._buffers: List[deque[np.ndarray]] = [deque() for _ in range(4)]
        self._buffer_lengths: List[int] = [0, 0, 0, 0]
        self._first_adc_times: List[Optional[float]] = [None, None, None, None]
        self._initial_trim_samples: Optional[List[int]] = None

        self._lock = threading.Lock()
        self._data_available = threading.Condition(self._lock)
        self._streams: List[sd.InputStream] = []
        self._running = False

    # -------------------------- Public API --------------------------
    def start(self) -> None:
        """Start recording from all 4 microphones."""
        if self._running:
            return
        self._streams = []
        for idx, dev in enumerate(self.mic_indices):
            cb = self._make_callback(idx)
            stream = sd.InputStream(
                device=dev,
                channels=1,
                samplerate=self.sample_rate,
                callback=cb,
                blocksize=self.blocksize,
                dtype="float32",
            )
            self._streams.append(stream)
        for s in self._streams:
            s.start()
        self._running = True

    def stop(self) -> None:
        """Stop recording and close all microphone streams."""
        if not self._running:
            return
        for s in self._streams:
            try:
                s.stop()
                s.close()
            except Exception:
                pass
        self._streams = []
        self._running = False

    def read_chunk(self, timeout: Optional[float] = None) -> np.ndarray:
        """
        Get a chunk of audio data from all 4 microphones, perfectly aligned.

        What happens:
        - Waits for all mics to start recording and get their timing info
        - Trims the early-starting mics so they all begin at the same moment
        - Returns audio chunks without using the sound content for alignment
          (this preserves the natural timing differences needed for TDOA)

        Returns: Array with shape (chunk_size, 4) - one column per microphone
        """
        with self._data_available:
            if not self._running:
                raise RuntimeError("Recorder is not running. Call start() first.")

            # Wait for first adc times from all mics
            self._wait_for_first_times_locked(timeout)

            # Compute and apply initial trim once
            if self._initial_trim_samples is None:
                self._compute_initial_trim_locked()
                self._apply_initial_trim_locked()

            # Wait until all buffers have enough samples for one chunk
            self._wait_for_min_samples_locked(self.chunk_size, timeout)

            # Extract chunk_size samples from each buffer
            aligned = []
            for ch in range(4):
                data = self._popleft_samples_locked(ch, self.chunk_size)
                if data.ndim > 1:
                    data = data.reshape(-1)
                aligned.append(data)

        # Stack outside the lock
        block = np.column_stack(aligned)
        return block

    # ----------------------- Internal helpers -----------------------
    def _make_callback(self, ch_idx: int):
        """Create a callback function for a specific microphone channel."""
        def _cb(indata, frames, time_info, status):
            if status:
                # We record status but we do not print to avoid callback overhead
                pass
            with self._data_available:
                if self._first_adc_times[ch_idx] is None:
                    self._first_adc_times[ch_idx] = time_info.inputBufferAdcTime
                # Append new data
                chunk = np.asarray(indata, dtype=np.float32).copy().reshape(-1)
                self._buffers[ch_idx].append(chunk)
                self._buffer_lengths[ch_idx] += chunk.size
                # Drop old data if buffer grows too large
                while self._buffer_lengths[ch_idx] > self.max_buffer_samples:
                    dropped = self._buffers[ch_idx].popleft()
                    self._buffer_lengths[ch_idx] -= dropped.size
                self._data_available.notify_all()
        return _cb

    def _wait_for_first_times_locked(self, timeout: Optional[float]) -> None:
        if all(t is not None for t in self._first_adc_times):
            return
        remaining = None if timeout is None else timeout
        while not all(t is not None for t in self._first_adc_times):
            if not self._data_available.wait(timeout=remaining):
                raise TimeoutError("Timed out waiting for initial ADC times")

    def _compute_initial_trim_locked(self) -> None:
        """Figure out how much to trim from each mic to align them."""
        assert all(t is not None for t in self._first_adc_times)
        # Use the latest-starting mic as our reference point
        common_start = max(self._first_adc_times)  # seconds
        trims = []
        for t0 in self._first_adc_times:
            # How much earlier this mic started (in samples)
            delta_t = common_start - float(t0)
            trims.append(int(round(delta_t * self.sample_rate)))
        self._initial_trim_samples = trims

    def _apply_initial_trim_locked(self) -> None:
        assert self._initial_trim_samples is not None
        # Ensure each buffer has at least trim samples, wait if necessary
        max_trim = max(self._initial_trim_samples)
        if max_trim > 0:
            self._wait_for_min_samples_locked(max_trim, timeout=None)
        # Perform trim
        for ch, trim in enumerate(self._initial_trim_samples):
            if trim <= 0:
                continue
            _ = self._popleft_samples_locked(ch, trim)

    def _wait_for_min_samples_locked(self, min_needed: int, timeout: Optional[float]) -> None:
        if min(self._buffer_lengths) >= min_needed:
            return
        remaining = None if timeout is None else timeout
        while min(self._buffer_lengths) < min_needed:
            if not self._data_available.wait(timeout=remaining):
                raise TimeoutError("Timed out waiting for audio data")

    def _popleft_samples_locked(self, ch: int, n: int) -> np.ndarray:
        out = np.empty(n, dtype=np.float32)
        write_idx = 0
        while write_idx < n:
            if not self._buffers[ch]:
                # Should not happen due to waiting; fill zeros defensively
                remaining = n - write_idx
                out[write_idx:] = 0.0
                self._buffer_lengths[ch] -= 0
                return out
            arr = self._buffers[ch][0]
            take = min(arr.size, n - write_idx)
            out[write_idx : write_idx + take] = arr[:take]
            if take == arr.size:
                self._buffers[ch].popleft()
            else:
                self._buffers[ch][0] = arr[take:]
            write_idx += take
            self._buffer_lengths[ch] -= take
        return out


