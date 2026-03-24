from __future__ import annotations

import argparse
import json
import os
import queue
import re
import sys
import threading
import time
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import sounddevice as sd
from PySide6.QtCore import QEvent, QObject, QPoint, QTimer, Qt
from PySide6.QtGui import QColor, QFont, QKeyEvent, QMouseEvent
from PySide6.QtWidgets import (
    QApplication,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizeGrip,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from vosk import KaldiRecognizer, Model, SetLogLevel

SetLogLevel(-1)

SMALL_MODEL_PAGE_URL = "https://alphacephei.com/vosk/models"
SMALL_MODEL_URL_FALLBACK: dict[str, str] = {
    "ar-tn-0.1-linto": "https://alphacephei.com/vosk/models/vosk-model-small-ar-tn-0.1-linto.zip",
    "ca-0.4": "https://alphacephei.com/vosk/models/vosk-model-small-ca-0.4.zip",
    "cn-0.22": "https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip",
    "de-0.15": "https://alphacephei.com/vosk/models/vosk-model-small-de-0.15.zip",
    "de-zamia-0.3": "https://alphacephei.com/vosk/models/vosk-model-small-de-zamia-0.3.zip",
    "en-us-0.15": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
    "es-0.42": "https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip",
    "fa-0.42": "https://alphacephei.com/vosk/models/vosk-model-small-fa-0.42.zip",
    "fa-0.5": "https://alphacephei.com/vosk/models/vosk-model-small-fa-0.5.zip",
    "fr-0.22": "https://alphacephei.com/vosk/models/vosk-model-small-fr-0.22.zip",
    "fr-pguyot-0.3": "https://alphacephei.com/vosk/models/vosk-model-small-fr-pguyot-0.3.zip",
    "gu-0.42": "https://alphacephei.com/vosk/models/vosk-model-small-gu-0.42.zip",
    "it-0.22": "https://alphacephei.com/vosk/models/vosk-model-small-it-0.22.zip",
    "ko-0.22": "https://alphacephei.com/vosk/models/vosk-model-small-ko-0.22.zip",
    "kz-0.42": "https://alphacephei.com/vosk/models/vosk-model-small-kz-0.42.zip",
    "nl-0.22": "https://alphacephei.com/vosk/models/vosk-model-small-nl-0.22.zip",
    "pl-0.22": "https://alphacephei.com/vosk/models/vosk-model-small-pl-0.22.zip",
    "pt-0.3": "https://alphacephei.com/vosk/models/vosk-model-small-pt-0.3.zip",
    "ru-0.22": "https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip",
    "te-0.42": "https://alphacephei.com/vosk/models/vosk-model-small-te-0.42.zip",
    "tg-0.22": "https://alphacephei.com/vosk/models/vosk-model-small-tg-0.22.zip",
    "tr-0.3": "https://alphacephei.com/vosk/models/vosk-model-small-tr-0.3.zip",
    "uk-v3-nano": "https://alphacephei.com/vosk/models/vosk-model-small-uk-v3-nano.zip",
    "uk-v3-small": "https://alphacephei.com/vosk/models/vosk-model-small-uk-v3-small.zip",
    "uz-0.22": "https://alphacephei.com/vosk/models/vosk-model-small-uz-0.22.zip",
    "vn-0.4": "https://alphacephei.com/vosk/models/vosk-model-small-vn-0.4.zip",
}
LANGUAGE_NAME_HINTS: dict[str, str] = {
    "ar": "Arabic",
    "ca": "Catalan",
    "cn": "Chinese",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fa": "Farsi",
    "fr": "French",
    "gu": "Gujarati",
    "it": "Italian",
    "ko": "Korean",
    "kz": "Kazakh",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "te": "Telugu",
    "tg": "Tajik",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "uz": "Uzbek",
    "vn": "Vietnamese",
}
DEFAULT_LANGUAGE = "en-us-0.15"
SMALL_MODEL_CATALOG: dict[str, dict[str, str]] = {}
DEFAULT_DOWNLOAD_TIMEOUT = 30

if os.name == "nt":
    APP_DATA_DIR = Path(os.getenv("LOCALAPPDATA", str(Path.home()))) / "py-live-captions"
else:
    APP_DATA_DIR = Path.home() / ".local" / "share" / "py-live-captions"

MODEL_CACHE_DIR = APP_DATA_DIR / "models"
DEFAULT_SETTINGS_PATH = APP_DATA_DIR / "settings.json"


def slug_to_label(slug: str) -> str:
    prefix = slug.split("-", maxsplit=1)[0]
    base = LANGUAGE_NAME_HINTS.get(prefix, prefix.upper())
    return f"{base} ({slug})"


def fetch_small_model_urls(timeout_seconds: int = 12) -> dict[str, str]:
    response = urllib.request.urlopen(SMALL_MODEL_PAGE_URL, timeout=timeout_seconds)
    html = response.read().decode("utf-8", "ignore").lower()
    names = sorted(set(re.findall(r"vosk-model-small-[a-z0-9][a-z0-9\\.-]*", html)))

    models: dict[str, str] = {}
    for matched_name in names:
        full_name = matched_name.removesuffix(".zip")
        if not full_name.startswith("vosk-model-small-"):
            continue
        slug = full_name.removeprefix("vosk-model-small-").strip()
        if not slug:
            continue
        models[slug] = f"https://alphacephei.com/vosk/models/{full_name}.zip"
    return models


def initialize_small_model_catalog(timeout_seconds: int = 12) -> None:
    global SMALL_MODEL_CATALOG

    catalog: dict[str, dict[str, str]] = {
        slug: {"label": slug_to_label(slug), "url": url}
        for slug, url in SMALL_MODEL_URL_FALLBACK.items()
    }

    try:
        fetched = fetch_small_model_urls(timeout_seconds=timeout_seconds)
    except Exception:  # noqa: BLE001
        fetched = {}

    for slug, url in fetched.items():
        catalog[slug] = {"label": slug_to_label(slug), "url": url}

    if DEFAULT_LANGUAGE not in catalog and catalog:
        first_key = sorted(catalog.keys())[0]
        catalog[DEFAULT_LANGUAGE] = catalog[first_key]

    SMALL_MODEL_CATALOG = dict(sorted(catalog.items(), key=lambda kv: kv[0]))


@dataclass
class AppSettings:
    language: str = DEFAULT_LANGUAGE
    border_color: str = "#4A4A4A"
    box_color: str = "#C81F1F1F"
    text_color: str = "#F2F2F2"
    font_family: str = "Segoe UI"
    font_size: int = 26
    border_width: int = 2
    opacity: float = 0.92
    position: str = "top"
    width_ratio: float = 0.78
    height_ratio: float = 0.17
    min_width: int = 700
    min_height: int = 180
    show_scrollbar: bool = True
    resizable: bool = True
    max_lines: int = 220
    device: int | None = None
    samplerate: int | None = None
    download_timeout: int = DEFAULT_DOWNLOAD_TIMEOUT
    partial_stale_seconds: float = 1.8

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AppSettings":
        defaults = cls()
        if not data:
            return defaults

        def parse_color(name: str, keep_alpha: bool = False) -> str:
            raw = data.get(name, getattr(defaults, name))
            default_value = getattr(defaults, name)
            if not isinstance(raw, str) or not QColor(raw).isValid():
                raw = default_value

            color = QColor(raw)
            if keep_alpha and color.alpha() == 255:
                fallback = QColor(default_value)
                color.setAlpha(fallback.alpha())
                return color.name(QColor.NameFormat.HexArgb)
            return color.name(QColor.NameFormat.HexArgb if color.alpha() < 255 else QColor.NameFormat.HexRgb)

        def parse_int(name: str, minimum: int | None = None) -> int:
            raw = data.get(name, getattr(defaults, name))
            try:
                parsed = int(raw)
            except (TypeError, ValueError):
                return getattr(defaults, name)
            if minimum is not None and parsed < minimum:
                return minimum
            return parsed

        def parse_float(name: str, minimum: float, maximum: float) -> float:
            raw = data.get(name, getattr(defaults, name))
            try:
                parsed = float(raw)
            except (TypeError, ValueError):
                return getattr(defaults, name)
            return max(minimum, min(maximum, parsed))

        language = str(data.get("language", defaults.language))
        if SMALL_MODEL_CATALOG and language not in SMALL_MODEL_CATALOG:
            language = defaults.language

        position = str(data.get("position", defaults.position))
        if position not in {"top", "bottom"}:
            position = defaults.position

        device = data.get("device", defaults.device)
        try:
            device = int(device) if device not in (None, "") else None
        except (TypeError, ValueError):
            device = defaults.device

        samplerate = data.get("samplerate", defaults.samplerate)
        try:
            samplerate = int(samplerate) if samplerate not in (None, "") else None
        except (TypeError, ValueError):
            samplerate = defaults.samplerate

        font_family = str(data.get("font_family", defaults.font_family)).strip() or defaults.font_family

        return cls(
            language=language,
            border_color=parse_color("border_color"),
            box_color=parse_color("box_color", keep_alpha=True),
            text_color=parse_color("text_color"),
            font_family=font_family,
            font_size=parse_int("font_size", minimum=8),
            border_width=parse_int("border_width", minimum=0),
            opacity=parse_float("opacity", 0.2, 1.0),
            position=position,
            width_ratio=parse_float("width_ratio", 0.2, 1.0),
            height_ratio=parse_float("height_ratio", 0.1, 0.8),
            min_width=parse_int("min_width", minimum=280),
            min_height=parse_int("min_height", minimum=100),
            show_scrollbar=bool(data.get("show_scrollbar", defaults.show_scrollbar)),
            resizable=bool(data.get("resizable", defaults.resizable)),
            max_lines=parse_int("max_lines", minimum=0),
            device=device,
            samplerate=samplerate,
            download_timeout=parse_int("download_timeout", minimum=5),
            partial_stale_seconds=parse_float("partial_stale_seconds", 0.4, 8.0),
        )


@dataclass
class CaptionState:
    max_lines: int
    committed: list[str]
    partial: str

    def __init__(self, max_lines: int) -> None:
        self.max_lines = max_lines
        self.committed = []
        self.partial = ""

    def commit(self, text: str) -> None:
        cleaned = text.strip()
        if cleaned:
            self.committed.append(cleaned)
            if self.max_lines > 0:
                self.committed = self.committed[-self.max_lines :]
        self.partial = ""

    def set_partial(self, text: str) -> None:
        self.partial = text.strip()

    def clear(self) -> None:
        self.committed = []
        self.partial = ""

    def render(self) -> str:
        lines = list(self.committed)
        if self.partial:
            lines.append(self.partial)
        if self.max_lines > 0:
            lines = lines[-self.max_lines :]
        return "\n".join(lines) if lines else "Listening..."


@dataclass(eq=True)
class RecognizerConfig:
    language: str
    model_path: Path | None
    model_url_override: str | None
    device: int | None
    samplerate: int | None
    download_timeout: int
    partial_stale_seconds: float


class SessionRestart(Exception):
    def __init__(self, new_config: RecognizerConfig) -> None:
        super().__init__("Recognition session restart requested")
        self.new_config = new_config


def load_settings(path: Path) -> AppSettings:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return AppSettings()
    except (json.JSONDecodeError, OSError):
        return AppSettings()
    return AppSettings.from_dict(payload if isinstance(payload, dict) else None)


def save_settings(path: Path, settings: AppSettings) -> None:
    payload = json.dumps(asdict(settings), indent=2)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    except OSError:
        fallback = Path.cwd() / ".live_caption_settings.json"
        try:
            fallback.parent.mkdir(parents=True, exist_ok=True)
            fallback.write_text(payload, encoding="utf-8")
        except OSError:
            pass


def resolve_settings_path(preferred_path: Path) -> Path:
    try:
        preferred_path.parent.mkdir(parents=True, exist_ok=True)
        return preferred_path
    except OSError:
        fallback = Path.cwd() / ".live_caption_settings.json"
        fallback.parent.mkdir(parents=True, exist_ok=True)
        return fallback


def color_to_css(value: str) -> str:
    color = QColor(value)
    if not color.isValid():
        color = QColor("#000000")
    return f"rgba({color.red()},{color.green()},{color.blue()},{color.alpha()})"


def get_language_label(code: str) -> str:
    if code in SMALL_MODEL_CATALOG:
        return SMALL_MODEL_CATALOG[code]["label"]
    return slug_to_label(code)


def get_model_url_candidates(language: str, override: str | None) -> list[str]:
    if override:
        return [override]
    if language in SMALL_MODEL_CATALOG:
        return [SMALL_MODEL_CATALOG[language]["url"]]
    if DEFAULT_LANGUAGE in SMALL_MODEL_CATALOG:
        return [SMALL_MODEL_CATALOG[DEFAULT_LANGUAGE]["url"]]
    return []


def resolve_samplerate(device: int | None, explicit_samplerate: int | None) -> int:
    if explicit_samplerate:
        return explicit_samplerate
    del device
    return 16000


def get_device_default_samplerate(device: int | None) -> int:
    device_info = sd.query_devices(device=device, kind="input")
    default_samplerate = int(device_info["default_samplerate"])
    return default_samplerate if default_samplerate > 0 else 16000


def _find_installed_model(cache_dir: Path, model_dir_name: str) -> Path | None:
    exact_match = cache_dir / model_dir_name
    if exact_match.exists() and (exact_match / "am").exists():
        return exact_match

    return next(
        (
            path
            for path in cache_dir.iterdir()
            if path.is_dir() and path.name.startswith(model_dir_name) and (path / "am").exists()
        ),
        None,
    )


def _download_file(url: str, destination: Path, timeout_seconds: int) -> None:
    with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
        with open(destination, "wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)


def _download_and_extract_model(cache_dir: Path, model_url: str, timeout_seconds: int) -> Path:
    model_dir_name = Path(model_url).name.removesuffix(".zip")
    archive_path = cache_dir / Path(model_url).name
    try:
        _download_file(model_url, archive_path, timeout_seconds=timeout_seconds)
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(cache_dir)
    finally:
        archive_path.unlink(missing_ok=True)

    installed = _find_installed_model(cache_dir, model_dir_name)
    if installed is None:
        raise RuntimeError(
            f"Model download succeeded but expected folder '{model_dir_name}' was not found."
        )
    return installed


def ensure_model(
    model_path: Path | None,
    model_url_candidates: list[str],
    status_callback: Callable[[str], None] | None = None,
    download_timeout: int = DEFAULT_DOWNLOAD_TIMEOUT,
) -> Path:
    def notify(message: str) -> None:
        if status_callback is not None:
            status_callback(message)

    if model_path is not None:
        if not model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")
        notify("Using custom local model")
        return model_path

    if not model_url_candidates:
        raise RuntimeError("No small model URL candidates available.")

    primary_cache_dir = MODEL_CACHE_DIR
    try:
        primary_cache_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        primary_cache_dir = Path.cwd() / ".model-cache"
        primary_cache_dir.mkdir(parents=True, exist_ok=True)
        notify(f"No access to default cache, using: {primary_cache_dir}")

    search_dirs = [primary_cache_dir]
    secondary_dir = Path.cwd() / ".model-cache"
    if secondary_dir != primary_cache_dir:
        secondary_dir.mkdir(parents=True, exist_ok=True)
        search_dirs.append(secondary_dir)

    for model_url in model_url_candidates:
        model_name = Path(model_url).name.removesuffix(".zip")
        for search_dir in search_dirs:
            cached_model = _find_installed_model(search_dir, model_name)
            if cached_model is not None:
                notify(f"Using cached model: {cached_model.name}")
                return cached_model

    download_errors: list[str] = []
    for model_url in model_url_candidates:
        model_name = Path(model_url).name.removesuffix(".zip")
        notify(f"Downloading model: {model_name}")
        try:
            installed = _download_and_extract_model(
                primary_cache_dir,
                model_url,
                timeout_seconds=download_timeout,
            )
            notify(f"Model ready: {installed.name}")
            return installed
        except Exception as exc:  # noqa: BLE001
            download_errors.append(f"{model_name}: {exc}")
            notify(f"Download failed for {model_name}, trying fallback...")

    cached_any: list[Path] = []
    for search_dir in search_dirs:
        cached_any.extend(path for path in search_dir.iterdir() if path.is_dir() and (path / "am").exists())
    if cached_any:
        available = ", ".join(sorted(path.name for path in set(cached_any))[:8])
        notify(f"Requested language model not cached. Available cached models: {available}")

    details = "; ".join(download_errors[-3:])
    raise RuntimeError(
        "Could not load selected language model. "
        f"Details: {details}. "
        "Download that language model or set --model-path."
    )


class SpeechWorker(threading.Thread):
    def __init__(self, initial_config: RecognizerConfig) -> None:
        super().__init__(daemon=True)
        self.events: queue.Queue[tuple[str, str]] = queue.Queue()
        self._commands: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._stop_event = threading.Event()
        self._paused_event = threading.Event()
        self._audio_chunks: queue.Queue[bytes] = queue.Queue(maxsize=160)
        self._config = initial_config

    def stop(self) -> None:
        self._stop_event.set()
        self._commands.put(("stop", None))

    def set_paused(self, paused: bool) -> None:
        self._commands.put(("pause", paused))

    def update_config(self, config: RecognizerConfig) -> None:
        self._commands.put(("config", config))

    def run(self) -> None:
        config = self._config
        while not self._stop_event.is_set():
            config, _changed = self._drain_commands(config)
            if self._stop_event.is_set():
                break
            try:
                config = self._run_session(config)
            except SessionRestart as restart:
                config = restart.new_config
            except Exception as exc:  # noqa: BLE001
                self.events.put(("status", "Error"))
                self.events.put(("error", str(exc)))
                time.sleep(0.4)

    def _drain_commands(self, config: RecognizerConfig) -> tuple[RecognizerConfig, bool]:
        changed = False
        while True:
            try:
                command, payload = self._commands.get_nowait()
            except queue.Empty:
                break

            if command == "stop":
                self._stop_event.set()
            elif command == "pause":
                if bool(payload):
                    self._paused_event.set()
                else:
                    self._paused_event.clear()
            elif command == "config":
                if isinstance(payload, RecognizerConfig) and payload != config:
                    config = payload
                    changed = True
        return config, changed

    def _run_session(self, config: RecognizerConfig) -> RecognizerConfig:
        self.events.put(("status", f"Preparing {get_language_label(config.language)} model..."))
        model_path = ensure_model(
            model_path=config.model_path,
            model_url_candidates=get_model_url_candidates(
                config.language,
                config.model_url_override,
            ),
            status_callback=lambda message: self.events.put(("status", message)),
            download_timeout=config.download_timeout,
        )
        self.events.put(("status", f"Loading model: {model_path.name}"))
        model = Model(str(model_path))

        samplerate = resolve_samplerate(config.device, config.samplerate)
        try:
            return self._run_stream(model, samplerate, config)
        except sd.PortAudioError as exc:
            if config.samplerate is not None:
                raise exc
            fallback_samplerate = get_device_default_samplerate(config.device)
            if fallback_samplerate == samplerate:
                raise exc
            self.events.put(("status", f"Mic fallback to {fallback_samplerate} Hz"))
            return self._run_stream(model, fallback_samplerate, config)

    def _run_stream(self, model: Model, samplerate: int, config: RecognizerConfig) -> RecognizerConfig:
        recognizer = KaldiRecognizer(model, samplerate)
        recognizer.SetWords(False)
        self.events.put(("status", f"Listening - {get_language_label(config.language)}"))

        partial_cleared = False
        last_partial_text = ""
        last_partial_change_at = time.monotonic()
        paused_status_sent = False

        def audio_callback(indata: bytes, frames: int, time_info: dict, status: sd.CallbackFlags) -> None:
            del frames, time_info
            if status:
                self.events.put(("error", str(status)))
            if self._stop_event.is_set() or self._paused_event.is_set():
                return
            try:
                self._audio_chunks.put_nowait(bytes(indata))
            except queue.Full:
                try:
                    self._audio_chunks.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._audio_chunks.put_nowait(bytes(indata))
                except queue.Full:
                    pass

        with sd.RawInputStream(
            samplerate=samplerate,
            blocksize=4000,
            device=config.device,
            dtype="int16",
            channels=1,
            callback=audio_callback,
        ):
            while not self._stop_event.is_set():
                next_config, changed = self._drain_commands(config)
                if changed:
                    self._drain_audio_queue()
                    raise SessionRestart(next_config)
                config = next_config

                if self._paused_event.is_set():
                    self._drain_audio_queue()
                    if not partial_cleared:
                        self.events.put(("partial", ""))
                        partial_cleared = True
                    if not paused_status_sent:
                        self.events.put(("status", "Paused"))
                        paused_status_sent = True
                    self._stop_event.wait(0.12)
                    continue

                partial_cleared = False
                if paused_status_sent:
                    self.events.put(("status", f"Listening - {get_language_label(config.language)}"))
                    paused_status_sent = False
                try:
                    chunk = self._audio_chunks.get(timeout=0.15)
                except queue.Empty:
                    now = time.monotonic()
                    if (
                        last_partial_text
                        and now - last_partial_change_at >= config.partial_stale_seconds
                    ):
                        self.events.put(("partial", ""))
                        last_partial_text = ""
                        last_partial_change_at = now
                    continue

                if recognizer.AcceptWaveform(chunk):
                    text = json.loads(recognizer.Result()).get("text", "").strip()
                    if text:
                        self.events.put(("final", text))
                    self.events.put(("partial", ""))
                    last_partial_text = ""
                    last_partial_change_at = time.monotonic()
                else:
                    partial = json.loads(recognizer.PartialResult()).get("partial", "").strip()
                    now = time.monotonic()
                    if partial != last_partial_text:
                        self.events.put(("partial", partial))
                        last_partial_text = partial
                        last_partial_change_at = now
                    elif partial and (now - last_partial_change_at >= config.partial_stale_seconds):
                        self.events.put(("partial", ""))
                        last_partial_text = ""
                        last_partial_change_at = now

        return config

    def _drain_audio_queue(self) -> None:
        while True:
            try:
                self._audio_chunks.get_nowait()
            except queue.Empty:
                break


class ColorButton(QPushButton):
    def __init__(
        self,
        color_hex: str,
        parent: QWidget | None = None,
        *,
        allow_alpha: bool = False,
        preserve_alpha: bool = False,
    ) -> None:
        super().__init__(parent)
        self._allow_alpha = allow_alpha
        self._preserve_alpha = preserve_alpha
        self._color_hex = color_hex
        self.clicked.connect(self._pick_color)
        self._refresh_style()

    def color_hex(self) -> str:
        return self._color_hex

    def set_color_hex(self, color_hex: str) -> None:
        incoming = QColor(color_hex)
        if not incoming.isValid():
            return
        current = QColor(self._color_hex) if QColor(self._color_hex).isValid() else QColor("#000000")

        if self._allow_alpha:
            if self._preserve_alpha:
                incoming.setAlpha(current.alpha())
            self._color_hex = incoming.name(QColor.NameFormat.HexArgb)
        else:
            self._color_hex = incoming.name(QColor.NameFormat.HexRgb)
        self._refresh_style()

    def _pick_color(self) -> None:
        chosen = QColorDialog.getColor(
            QColor(self._color_hex),
            self,
            "Choose color",
            QColorDialog.ColorDialogOption.ShowAlphaChannel if self._allow_alpha else QColorDialog.ColorDialogOption(0),
        )
        if not chosen.isValid():
            return

        if self._allow_alpha:
            if self._preserve_alpha:
                current = QColor(self._color_hex)
                chosen.setAlpha(current.alpha())
            self._color_hex = chosen.name(QColor.NameFormat.HexArgb)
        else:
            self._color_hex = chosen.name(QColor.NameFormat.HexRgb)
        self._refresh_style()

    def _refresh_style(self) -> None:
        color = QColor(self._color_hex)
        text_color = "#111111" if color.lightness() > 128 else "#F8F8F8"
        self.setText(self._color_hex.upper())
        self.setStyleSheet(
            f"QPushButton {{"
            f"background-color: {color_to_css(self._color_hex)};"
            f"color: {text_color};"
            f"border: 1px solid #5a5a5a;"
            f"padding: 4px 8px;"
            f"border-radius: 4px;"
            f"}}"
        )


class SettingsDialog(QDialog):
    def __init__(
        self,
        current_settings: AppSettings,
        settings_path: Path,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Caption Settings")
        self.setModal(True)
        self.resize(620, 700)
        self._base_settings = current_settings

        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addLayout(form)

        self.language_combo = QComboBox()
        for code, entry in SMALL_MODEL_CATALOG.items():
            self.language_combo.addItem(entry["label"], code)
        self._set_combo_data(self.language_combo, current_settings.language)
        form.addRow("Language", self.language_combo)

        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(10, 80)
        self.font_size_spin.setValue(current_settings.font_size)
        form.addRow("Font size", self.font_size_spin)

        self.border_color_button = ColorButton(current_settings.border_color)
        form.addRow("Border color", self.border_color_button)
        self.box_color_button = ColorButton(
            current_settings.box_color,
            allow_alpha=True,
            preserve_alpha=True,
        )
        form.addRow("Box color", self.box_color_button)
        self.text_color_button = ColorButton(current_settings.text_color)
        form.addRow("Text color", self.text_color_button)

        self.position_combo = QComboBox()
        self.position_combo.addItem("Top", "top")
        self.position_combo.addItem("Bottom", "bottom")
        self._set_combo_data(self.position_combo, current_settings.position)
        form.addRow("Position", self.position_combo)

        save_path_label = QLabel(str(settings_path))
        save_path_label.setWordWrap(True)
        form.addRow("Settings file", save_path_label)

        about_box = QFrame(self)
        about_box.setObjectName("aboutBox")
        about_box.setStyleSheet(
            "QFrame#aboutBox {"
            "border: 1px solid #4f4f4f;"
            "border-radius: 8px;"
            "background-color: rgba(255,255,255,0.03);"
            "}"
        )
        about_layout = QVBoxLayout(about_box)
        about_layout.setContentsMargins(10, 8, 10, 8)
        about_layout.setSpacing(6)

        about_title = QLabel("About")
        about_title.setStyleSheet("font-weight: 600; color: #eaeaea;")
        about_layout.addWidget(about_title)

        about_label = QLabel(
            "App Name: Live Caption\n"
            "Version: 1.0\n"
            "Build: 1\n\n"
            "Developer: depthwc\n\n"
            "Description:\n"
            "Real-time multi-language live captions with a frameless overlay,\n"
            "custom styling, and persistent settings.\n\n"
            "Tech: PySide6 + Vosk small models\n"
            "Model Source: https://alphacephei.com/vosk/models\n\n"
            "Email: depthwc@gmail.com"
        )
        about_label.setWordWrap(True)
        about_label.setStyleSheet("color: #d4d4d4;")
        about_layout.addWidget(about_label)
        layout.addWidget(about_box)

        layout.addStretch(1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def build_settings(self) -> AppSettings:
        merged = AppSettings.from_dict(asdict(self._base_settings))
        merged.language = str(self.language_combo.currentData())
        merged.font_size = int(self.font_size_spin.value())
        merged.border_color = self.border_color_button.color_hex()
        merged.box_color = self.box_color_button.color_hex()
        merged.text_color = self.text_color_button.color_hex()
        merged.position = str(self.position_combo.currentData())
        return merged

    @staticmethod
    def _set_combo_data(combo: QComboBox, wanted_data: Any) -> None:
        for index in range(combo.count()):
            if combo.itemData(index) == wanted_data:
                combo.setCurrentIndex(index)
                return

def settings_to_recognizer_config(
    settings: AppSettings,
    model_path_override: Path | None,
    model_url_override: str | None,
) -> RecognizerConfig:
    return RecognizerConfig(
        language=settings.language,
        model_path=model_path_override,
        model_url_override=model_url_override,
        device=settings.device,
        samplerate=settings.samplerate,
        download_timeout=settings.download_timeout,
        partial_stale_seconds=settings.partial_stale_seconds,
    )


class CaptionOverlay(QWidget):
    def __init__(
        self,
        settings: AppSettings,
        settings_path: Path,
        model_path_override: Path | None = None,
        model_url_override: str | None = None,
    ) -> None:
        super().__init__()
        self.settings = settings
        self.settings_path = settings_path
        self.model_path_override = model_path_override
        self.model_url_override = model_url_override
        self.caption_state = CaptionState(max_lines=settings.max_lines)
        self._drag_offset: QPoint | None = None
        self._paused = False
        self._closing = False
        self._force_exit_scheduled = False

        self._apply_window_flags()
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._build_ui()
        self._apply_initial_geometry()
        self._apply_style()
        self._render_text()

        self.worker = SpeechWorker(
            settings_to_recognizer_config(settings, model_path_override, model_url_override)
        )
        self.worker.start()

        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self._poll_events)
        self.poll_timer.start(50)

    def _apply_window_flags(self) -> None:
        flags = Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint
        self.setWindowFlags(flags)

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(0)

        self.panel = QFrame(self)
        self.panel.setObjectName("panel")
        root_layout.addWidget(self.panel)

        panel_layout = QVBoxLayout(self.panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(0)

        self.header = QFrame(self.panel)
        self.header.setObjectName("header")
        self.header.setFixedHeight(40)
        panel_layout.addWidget(self.header)

        header_layout = QHBoxLayout(self.header)
        header_layout.setContentsMargins(12, 6, 10, 6)
        header_layout.setSpacing(8)

        self.title_label = QLabel("Live captions", self.header)
        self.title_label.setObjectName("titleLabel")
        header_layout.addWidget(self.title_label)

        self.status_label = QLabel("Starting...", self.header)
        self.status_label.setObjectName("statusLabel")
        header_layout.addWidget(self.status_label)
        header_layout.addStretch(1)

        self.pause_button = QPushButton("Pause", self.header)
        self.pause_button.setObjectName("controlButton")
        self.pause_button.clicked.connect(self._toggle_pause)
        header_layout.addWidget(self.pause_button)

        self.clear_button = QPushButton("Clear", self.header)
        self.clear_button.setObjectName("controlButton")
        self.clear_button.clicked.connect(self._clear_captions)
        header_layout.addWidget(self.clear_button)

        self.settings_button = QPushButton("Settings", self.header)
        self.settings_button.setObjectName("controlButton")
        self.settings_button.setToolTip(f"Settings file: {self.settings_path}")
        self.settings_button.clicked.connect(self._open_settings_dialog)
        header_layout.addWidget(self.settings_button)

        self.close_button = QPushButton("X", self.header)
        self.close_button.setObjectName("controlButton")
        self.close_button.clicked.connect(self.close)
        header_layout.addWidget(self.close_button)

        self.caption_view = QTextEdit(self.panel)
        self.caption_view.setObjectName("captionView")
        self.caption_view.setReadOnly(True)
        self.caption_view.setAcceptRichText(False)
        self.caption_view.setFrameShape(QFrame.Shape.NoFrame)
        panel_layout.addWidget(self.caption_view, 1)

        footer = QFrame(self.panel)
        footer.setObjectName("footer")
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(0, 0, 8, 6)
        footer_layout.setSpacing(0)
        footer_layout.addStretch(1)
        self.size_grip = QSizeGrip(footer)
        footer_layout.addWidget(self.size_grip, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)
        panel_layout.addWidget(footer)

        self.header.installEventFilter(self)
        self.title_label.installEventFilter(self)
        self.status_label.installEventFilter(self)

    def _apply_initial_geometry(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(max(self.settings.min_width, 700), max(self.settings.min_height, 180))
            self.move(100, 100)
            return

        available = screen.availableGeometry()
        width = max(int(available.width() * self.settings.width_ratio), self.settings.min_width)
        height = max(int(available.height() * self.settings.height_ratio), self.settings.min_height)
        x = available.x() + (available.width() - width) // 2
        if self.settings.position == "top":
            y = available.y() + int(available.height() * 0.05)
        else:
            y = available.y() + available.height() - height - int(available.height() * 0.07)

        self.setMinimumSize(self.settings.min_width, self.settings.min_height)
        self.resize(width, height)
        self.move(x, y)

    def _apply_style(self) -> None:
        self.setWindowOpacity(self.settings.opacity)
        self.setMinimumSize(self.settings.min_width, self.settings.min_height)
        if self.settings.resizable:
            self.setMaximumSize(16777215, 16777215)
        else:
            self.setFixedSize(self.size())
        self.size_grip.setVisible(self.settings.resizable)

        scrollbar_policy = (
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
            if self.settings.show_scrollbar
            else Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.caption_view.setVerticalScrollBarPolicy(scrollbar_policy)
        self.caption_view.setFont(QFont(self.settings.font_family, self.settings.font_size))

        box_color_css = color_to_css(self.settings.box_color)
        border_color_css = color_to_css(self.settings.border_color)
        text_color_css = color_to_css(self.settings.text_color)

        self.setStyleSheet(
            f"""
            QFrame#panel {{
                background-color: {box_color_css};
                border: {self.settings.border_width}px solid {border_color_css};
                border-radius: 14px;
            }}
            QFrame#header {{
                background-color: {box_color_css};
                border: none;
            }}
            QFrame#footer {{
                background-color: {box_color_css};
                border: none;
                border-bottom-left-radius: 12px;
                border-bottom-right-radius: 12px;
            }}
            QLabel#titleLabel {{
                color: {text_color_css};
                font-weight: 600;
                font-size: 13px;
                background: transparent;
            }}
            QLabel#statusLabel {{
                color: {text_color_css};
                font-size: 11px;
                background: transparent;
            }}
            QTextEdit#captionView {{
                background-color: {box_color_css};
                color: {text_color_css};
                font-family: "{self.settings.font_family}";
                font-size: {self.settings.font_size}pt;
                border: none;
                padding: 10px 12px 8px 12px;
                selection-background-color: #5A78A0;
                selection-color: #FFFFFF;
            }}
            QPushButton#controlButton {{
                background-color: transparent;
                color: {text_color_css};
                border: none;
                border-radius: 6px;
                padding: 4px 8px;
            }}
            QPushButton#controlButton:hover {{
                background-color: rgba(255,255,255,0.12);
            }}
            """
        )

    def _poll_events(self) -> None:
        dirty = False
        while True:
            try:
                event_type, payload = self.worker.events.get_nowait()
            except queue.Empty:
                break

            if event_type == "final":
                self.caption_state.commit(payload)
                dirty = True
            elif event_type == "partial":
                self.caption_state.set_partial(payload)
                dirty = True
            elif event_type == "status":
                self.status_label.setText(payload)
            elif event_type == "error":
                self.caption_state.commit(f"[error] {payload}")
                self.status_label.setText("Error")
                dirty = True

        if dirty:
            self._render_text()

    def _render_text(self) -> None:
        self.caption_view.setPlainText(self.caption_state.render())
        bar = self.caption_view.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _toggle_pause(self) -> None:
        self._paused = not self._paused
        self.pause_button.setText("Resume" if self._paused else "Pause")
        self.worker.set_paused(self._paused)
        self.status_label.setText("Paused" if self._paused else "Listening")

    def _clear_captions(self) -> None:
        self.caption_state.clear()
        self._render_text()

    def _open_settings_dialog(self) -> None:
        dialog = SettingsDialog(self.settings, self.settings_path, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        old_position = self.settings.position
        was_resizable = self.settings.resizable
        self.settings = dialog.build_settings()
        self.caption_state.max_lines = self.settings.max_lines

        if not was_resizable and self.settings.resizable:
            self.setMaximumSize(16777215, 16777215)

        self._apply_style()
        self._render_text()
        if self.settings.position != old_position:
            self._apply_initial_geometry()

        save_settings(self.settings_path, self.settings)
        new_config = settings_to_recognizer_config(
            self.settings,
            self.model_path_override,
            self.model_url_override,
        )
        self.worker.update_config(new_config)
        self.caption_state.clear()
        self._render_text()
        self.status_label.setText(
            f"Reconfiguring: {get_language_label(new_config.language)}"
        )

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched in (self.header, self.title_label, self.status_label):
            if event.type() == QEvent.Type.MouseButtonPress and isinstance(event, QMouseEvent):
                if event.button() == Qt.MouseButton.LeftButton:
                    self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                    return True
            elif event.type() == QEvent.Type.MouseMove and isinstance(event, QMouseEvent):
                if self._drag_offset is not None:
                    self.move(event.globalPosition().toPoint() - self._drag_offset)
                    return True
            elif event.type() == QEvent.Type.MouseButtonRelease:
                self._drag_offset = None
                return True
        return super().eventFilter(watched, event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event: Any) -> None:
        if self._closing:
            event.accept()
            return
        self._closing = True

        save_settings(self.settings_path, self.settings)
        if self.poll_timer.isActive():
            self.poll_timer.stop()
        self.worker.stop()
        self.worker.join(timeout=1.0)

        event.accept()
        app = QApplication.instance()
        if app is not None:
            app.quit()

        # Force-kill process as a watchdog if backend threads keep Python alive.
        if not self._force_exit_scheduled:
            self._force_exit_scheduled = True
            killer = threading.Timer(1.5, lambda: os._exit(0))
            killer.daemon = True
            killer.start()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Frameless Live Caption overlay (PySide6 + Vosk).")
    parser.add_argument("--settings-file", type=Path, default=DEFAULT_SETTINGS_PATH, help="Path to settings JSON.")
    parser.add_argument("--reset-settings", action="store_true", help="Delete saved settings before launch.")
    parser.add_argument("--model-path", type=Path, default=None, help="Use local Vosk model folder.")
    parser.add_argument("--model-url", default=None, help="Override model download URL.")
    parser.add_argument("--language", default=None, help="Startup small-model key (see Settings list).")
    parser.add_argument("--device", type=int, default=None, help="Input device index override.")
    parser.add_argument("--samplerate", type=int, default=None, help="Input sample rate override.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    initialize_small_model_catalog()
    settings_path: Path = resolve_settings_path(args.settings_file)

    if args.reset_settings:
        settings_path.unlink(missing_ok=True)

    settings = load_settings(settings_path)
    if args.language is not None:
        settings.language = args.language
    if args.device is not None:
        settings.device = args.device
    if args.samplerate is not None:
        settings.samplerate = args.samplerate

    if settings.language not in SMALL_MODEL_CATALOG and SMALL_MODEL_CATALOG:
        settings.language = DEFAULT_LANGUAGE

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    app.lastWindowClosed.connect(app.quit)
    overlay = CaptionOverlay(
        settings=settings,
        settings_path=settings_path,
        model_path_override=args.model_path,
        model_url_override=args.model_url,
    )
    overlay.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
