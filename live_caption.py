import sys
import queue
import threading
import numpy as np

# Monkey-patch numpy.fromstring to fix compatibility with soundcard on numpy 2.x
# soundcard internally uses the deprecated numpy.fromstring which was removed in numpy 2.0
_orig_fromstring = np.fromstring
np.fromstring = lambda s, dtype=float, count=-1, sep='': (
    np.frombuffer(s, dtype=dtype, count=count) if sep == ''
    else _orig_fromstring(s, dtype=dtype, count=count, sep=sep)
)

import soundcard as sc
from faster_whisper import WhisperModel


class LiveCaptioner:
    """
    Core component for capturing system audio (speakers loopback) and running Whisper.
    Uses 'soundcard' for reliable Windows WASAPI loopback capture.
    """
    def __init__(self, model_size="base", device="auto", compute_type="default", language="en", speaker_index=None):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.speaker_index = speaker_index
        self.model = None

        self.samplerate = 16000
        self.channels = 1

        self.callbacks = []
        self.q = queue.Queue()
        self.is_running = False

        self._capture_thread = None
        self._transcribe_thread = None
        self._stop_event = threading.Event()

    def load_model(self):
        print(f"Loading '{self.model_size}' Whisper model...", file=sys.stderr)
        self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
        print("Model loaded successfully.", file=sys.stderr)

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def _audio_capture_loop(self):
        """Continuously captures 2-second audio blocks from speaker loopback."""
        try:
            if self.speaker_index is not None:
                speakers = sc.all_speakers()
                speaker = speakers[self.speaker_index]
            else:
                speaker = sc.default_speaker()

            mic_device = sc.get_microphone(speaker.id, include_loopback=True)
            print(f"[Captioner] Hooked into: {speaker.name} (loopback)", file=sys.stderr)

            chunk_frames = self.samplerate * 2  # 2-second chunks

            with mic_device.recorder(samplerate=self.samplerate, channels=self.channels) as mic:
                while not self._stop_event.is_set():
                    data = mic.record(numframes=chunk_frames)
                    audio_chunk = data.flatten().astype(np.float32)

                    energy = np.sqrt(np.mean(audio_chunk ** 2))

                    # Skip silent/dead air chunks
                    if energy > 0.001:
                        self.q.put(audio_chunk)

        except Exception as e:
            print(f"[Captioner] Loopback capture error: {e}", file=sys.stderr)

    def _transcribe_loop(self):
        """Worker thread: pulls audio chunks and runs Whisper inference."""
        while not self._stop_event.is_set() or not self.q.empty():
            try:
                audio_data = self.q.get(timeout=0.5)

                segments, _info = self.model.transcribe(
                    audio_data,
                    beam_size=5,
                    language=self.language,
                    condition_on_previous_text=False
                )

                text = " ".join([seg.text for seg in segments]).strip()

                if text:
                    print(f"[Transcribed]: {text}", file=sys.stderr)
                    for cb in self.callbacks:
                        cb(text)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Captioner] Transcription error: {e}", file=sys.stderr)

    def start(self):
        """Load model and spin up capture + transcription threads."""
        if not self.model:
            self.load_model()

        self._stop_event.clear()
        self.is_running = True
        self.q.queue.clear()

        self._capture_thread = threading.Thread(target=self._audio_capture_loop, daemon=True)
        self._transcribe_thread = threading.Thread(target=self._transcribe_loop, daemon=True)

        self._capture_thread.start()
        self._transcribe_thread.start()

    def stop(self):
        """Signal threads to stop. Does NOT join — threads are daemon and will die with the process."""
        print("[Captioner] Stopping...", file=sys.stderr)
        self._stop_event.set()
        self.is_running = False
        # Do NOT join here — _capture_thread is blocked in mic.record() and can't be interrupted
        # Daemon threads will be killed automatically when the main process exits
