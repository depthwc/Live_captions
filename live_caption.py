import sys
import queue
import time
import threading
import numpy as np

# Monkey-patch numpy.fromstring to fix compatibility with soundcard on numpy 2.x
# soundcard internally uses the deprecated numpy.fromstring which was removed in numpy 2.0
_orig_fromstring = np.fromstring
np.fromstring = lambda s, dtype=float, count=-1, sep='': (
    np.frombuffer(s, dtype=dtype, count=count) if sep == ""
    else _orig_fromstring(s, dtype=dtype, count=count, sep=sep)
)

import soundcard as sc
from faster_whisper import WhisperModel


class LiveCaptioner:
    """
    Capture system audio from Windows loopback and transcribe with Faster-Whisper.

    Reliability goals:
    - Auto-pick an active output device when no explicit speaker index is set.
    - Fall back across common capture sample rates.
    - Re-scan outputs if the current device stays silent for too long.
    """

    def __init__(self, model_size="base", device="auto", compute_type="default", language=None, speaker_index=None):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.speaker_index = speaker_index
        self.model = None

        self.target_samplerate = 16000
        self.preferred_capture_rates = (48000, 44100, 32000, 16000)
        self.capture_channels = 2
        self.chunk_seconds = 2
        self.min_audio_energy = 1e-5
        self.silent_reprobe_chunks = 15
        self.speaker_probe_seconds = 0.30

        self.callbacks = []
        self.status_callbacks = []
        self.q = queue.Queue(maxsize=8)
        self.is_running = False

        self._capture_thread = None
        self._transcribe_thread = None
        self._stop_event = threading.Event()
        self._has_reported_waiting = False
        self._tried_cpu_fallback = False

    def load_model(self):
        print(f"Loading '{self.model_size}' Whisper model...", file=sys.stderr)
        try:
            self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
        except Exception as e:
            # Common Windows setup issue: CUDA runtime missing.
            if self.device != "cpu" and "cublas64_12.dll" in str(e).lower():
                print("[Captioner] CUDA runtime missing, retrying on CPU.", file=sys.stderr)
                self._emit_status("CUDA not found, switching to CPU mode...")
                self.device = "cpu"
                self.compute_type = "int8"
                self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
            else:
                raise
        print("Model loaded successfully.", file=sys.stderr)

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def add_status_callback(self, callback):
        self.status_callbacks.append(callback)

    def _emit_caption(self, text):
        for cb in self.callbacks:
            cb(text)

    def _emit_status(self, status):
        for cb in self.status_callbacks:
            cb(status)

    def _resample_to_target_rate(self, audio_chunk, source_rate):
        """Linearly resample mono float32 audio to self.target_samplerate."""
        if source_rate == self.target_samplerate:
            return audio_chunk.astype(np.float32, copy=False)

        if audio_chunk.size == 0:
            return np.empty(0, dtype=np.float32)

        target_size = int(round(audio_chunk.size * (self.target_samplerate / float(source_rate))))
        target_size = max(1, target_size)

        source_index = np.arange(audio_chunk.size, dtype=np.float64)
        target_index = np.linspace(0, audio_chunk.size - 1, num=target_size, dtype=np.float64)
        return np.interp(target_index, source_index, audio_chunk).astype(np.float32)

    def _to_mono_float32(self, data):
        if data.size == 0:
            return np.empty(0, dtype=np.float32)
        if data.ndim > 1:
            return data.mean(axis=1).astype(np.float32, copy=False)
        return data.astype(np.float32, copy=False)

    def _audio_energy(self, mono_audio):
        if mono_audio.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(mono_audio ** 2)))

    def _put_audio_chunk(self, audio_chunk):
        """Keep queue bounded; when full, drop the oldest chunk."""
        try:
            self.q.put(audio_chunk, timeout=0.1)
        except queue.Full:
            try:
                self.q.get_nowait()
            except queue.Empty:
                pass
            try:
                self.q.put_nowait(audio_chunk)
            except queue.Full:
                pass

    def _ordered_speakers(self):
        speakers = sc.all_speakers()
        if not speakers:
            raise RuntimeError("No output speakers found.")

        if self.speaker_index is not None:
            if self.speaker_index < 0 or self.speaker_index >= len(speakers):
                raise ValueError(
                    f"speaker_index={self.speaker_index} is out of range (0..{len(speakers) - 1})"
                )
            return [speakers[self.speaker_index]]

        default_speaker = sc.default_speaker()
        ordered = []
        seen_ids = set()
        for spk in [default_speaker, *speakers]:
            if spk.id in seen_ids:
                continue
            seen_ids.add(spk.id)
            ordered.append(spk)
        return ordered

    def _probe_speaker(self, speaker):
        """
        Try opening loopback recorder for a speaker and estimate short-term energy.
        Returns (energy, samplerate, channels) or (None, None, None) on failure.
        """
        mic_device = sc.get_microphone(speaker.id, include_loopback=True)
        channel_count = max(1, min(self.capture_channels, speaker.channels))
        last_error = None

        for rate in self.preferred_capture_rates:
            try:
                probe_frames = max(1024, int(rate * self.speaker_probe_seconds))
                with mic_device.recorder(samplerate=rate, channels=channel_count) as mic:
                    data = mic.record(numframes=probe_frames)
                mono = self._to_mono_float32(data)
                energy = self._audio_energy(mono)
                return energy, rate, channel_count
            except Exception as e:
                last_error = e
                continue

        print(f"[Captioner] Probe failed for '{speaker.name}': {last_error}", file=sys.stderr)
        return None, None, None

    def _select_speaker_for_capture(self):
        """
        Select best loopback source:
        - explicit speaker_index: use that speaker
        - otherwise: probe all and pick highest current energy
        """
        speakers = self._ordered_speakers()
        candidates = []

        for speaker in speakers:
            energy, rate, channels = self._probe_speaker(speaker)
            if rate is None:
                continue
            print(
                f"[Captioner] Probe '{speaker.name}': energy={energy:.8f}, rate={rate}, channels={channels}",
                file=sys.stderr,
            )
            candidates.append((energy, speaker, rate, channels))

        if not candidates:
            raise RuntimeError("Could not open loopback recorder on any output device.")

        # Explicit speaker_index bypasses active-energy selection.
        if self.speaker_index is not None:
            _, speaker, rate, channels = candidates[0]
            return speaker, rate, channels

        # Pick the currently active output (highest energy). If all are near-silent,
        # still use default/first candidate so we can keep listening.
        best_energy, best_speaker, best_rate, best_channels = max(candidates, key=lambda x: x[0])
        if best_energy < self.min_audio_energy:
            _, best_speaker, best_rate, best_channels = candidates[0]
        return best_speaker, best_rate, best_channels

    def _audio_capture_loop(self):
        """Continuously captures chunked audio blocks from speaker loopback."""
        while not self._stop_event.is_set():
            try:
                if self.speaker_index is None:
                    self._emit_status("Detecting active output device...")
                speaker, capture_rate, channel_count = self._select_speaker_for_capture()

                mic_device = sc.get_microphone(speaker.id, include_loopback=True)
                chunk_frames = int(capture_rate * self.chunk_seconds)
                silent_chunks = 0

                print(
                    f"[Captioner] Listening to: {speaker.name} (loopback, {capture_rate} Hz, {channel_count} ch)",
                    file=sys.stderr,
                )
                self._emit_status(f"Listening to: {speaker.name}")

                with mic_device.recorder(samplerate=capture_rate, channels=channel_count) as mic:
                    while not self._stop_event.is_set():
                        data = mic.record(numframes=chunk_frames)
                        mono_audio = self._to_mono_float32(data)
                        audio_chunk = self._resample_to_target_rate(mono_audio, capture_rate)
                        energy = self._audio_energy(audio_chunk)

                        if energy > self.min_audio_energy:
                            silent_chunks = 0
                            self._has_reported_waiting = False
                            self._put_audio_chunk(audio_chunk)
                        else:
                            silent_chunks += 1
                            if silent_chunks >= 5 and not self._has_reported_waiting:
                                self._emit_status("Waiting for system audio...")
                                self._has_reported_waiting = True

                            # In auto mode, re-scan output devices if the current device remains silent.
                            if self.speaker_index is None and silent_chunks >= self.silent_reprobe_chunks:
                                self._emit_status("No signal on current output, rescanning devices...")
                                break

            except Exception as e:
                print(f"[Captioner] Loopback capture error: {e}", file=sys.stderr)
                self._emit_status(f"Loopback capture error: {e}")
                if self._stop_event.is_set():
                    break
                time.sleep(1.0)

    def _transcribe_loop(self):
        """Worker thread: pulls audio chunks and runs Whisper inference."""
        while not self._stop_event.is_set() or not self.q.empty():
            try:
                audio_data = self.q.get(timeout=0.5)

                transcribe_kwargs = {
                    "beam_size": 1,
                    "condition_on_previous_text": False,
                }
                if self.language:
                    transcribe_kwargs["language"] = self.language

                segments, _info = self.model.transcribe(audio_data, **transcribe_kwargs)
                text = " ".join(seg.text for seg in segments).strip()

                if text:
                    print(f"[Transcribed]: {text}", file=sys.stderr)
                    self._emit_caption(text)

            except queue.Empty:
                continue
            except Exception as e:
                error_text = str(e)
                if (
                    self.device != "cpu"
                    and not self._tried_cpu_fallback
                    and "cublas64_12.dll" in error_text.lower()
                ):
                    self._tried_cpu_fallback = True
                    print("[Captioner] CUDA unavailable during inference, switching to CPU.", file=sys.stderr)
                    self._emit_status("CUDA unavailable, switching to CPU mode...")
                    self.device = "cpu"
                    self.compute_type = "int8"
                    self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
                    continue

                print(f"[Captioner] Transcription error: {e}", file=sys.stderr)
                self._emit_status(f"Transcription error: {e}")

    def start(self):
        """Load model and spin up capture + transcription threads."""
        if not self.model:
            self._emit_status("Loading speech model...")
            self.load_model()

        self._stop_event.clear()
        self.is_running = True
        self._has_reported_waiting = False
        self._tried_cpu_fallback = False
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break

        self._capture_thread = threading.Thread(target=self._audio_capture_loop, daemon=True)
        self._transcribe_thread = threading.Thread(target=self._transcribe_loop, daemon=True)

        self._capture_thread.start()
        self._transcribe_thread.start()

    def stop(self):
        """Signal threads to stop. Does not join because threads are daemons."""
        print("[Captioner] Stopping...", file=sys.stderr)
        self._stop_event.set()
        self.is_running = False
        # Do not join here because _capture_thread is blocked in mic.record() and can't be interrupted.
        # Daemon threads will be killed automatically when the main process exits.
