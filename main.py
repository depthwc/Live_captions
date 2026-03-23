import sys
import os
import threading
import subprocess

def _maybe_reexec_into_project_venv():
    """
    If this script is launched with a global interpreter while .venv exists,
    re-exec using the project's virtual environment Python.
    """
    project_dir = os.path.dirname(os.path.abspath(__file__))
    if os.name == "nt":
        venv_python = os.path.join(project_dir, ".venv", "Scripts", "python.exe")
    else:
        venv_python = os.path.join(project_dir, ".venv", "bin", "python")

    if not os.path.exists(venv_python):
        return

    current = os.path.normcase(os.path.abspath(sys.executable))
    target = os.path.normcase(os.path.abspath(venv_python))

    # Prevent infinite recursion in case re-exec fails.
    if current != target and os.environ.get("LIVE_CAPTIONS_REEXEC") != "1":
        print(f"[Bootstrap] Re-launching with project venv: {venv_python}")
        child_env = os.environ.copy()
        child_env["LIVE_CAPTIONS_REEXEC"] = "1"
        result = subprocess.run([venv_python] + sys.argv, env=child_env)
        raise SystemExit(result.returncode)


_maybe_reexec_into_project_venv()

# live_caption.py applies the numpy monkey-patch before importing soundcard,
# so we must NOT import soundcard here at the top level.
from live_caption import LiveCaptioner
from gui import GUIController

from PySide6.QtWidgets import QApplication

# Suppress HuggingFace symlinks cache warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def main():
    print("Initializing PySide6 Live Captions...")

    # Must instantiate QApplication before any GUI widgets
    app = QApplication(sys.argv)

    model_size = os.environ.get("LIVE_CAPTIONS_MODEL", "tiny")
    language = os.environ.get("LIVE_CAPTIONS_LANG") or None
    speaker_index_env = os.environ.get("LIVE_CAPTIONS_SPEAKER_INDEX")
    speaker_index = None
    if speaker_index_env:
        try:
            speaker_index = int(speaker_index_env)
        except ValueError:
            print(f"[Bootstrap] Ignoring invalid LIVE_CAPTIONS_SPEAKER_INDEX='{speaker_index_env}'")

    # Use default speaker loopback automatically (no blocking input() call)
    # Force CPU mode: avoid CUDA DLL errors (cublas64_12.dll) on machines without CUDA
    captioner = LiveCaptioner(
        model_size=model_size,
        speaker_index=speaker_index,
        device="cpu",
        compute_type="int8",
        language=language,
    )

    # Initialize GUI
    gui_manager = GUIController()

    # Thread-safe callback: route transcribed text to the PySide UI thread via Signal
    captioner.add_callback(gui_manager.update_caption)
    captioner.add_status_callback(gui_manager.update_caption)

    # Start backend (model loading + audio capture) in a daemon thread
    backend_thread = threading.Thread(target=captioner.start, daemon=True)
    backend_thread.start()

    # Show the overlay window
    gui_manager.show_window()

    # Block on the Qt event loop
    exit_code = app.exec()

    # Signal backend to stop; daemon threads will be killed when the process exits
    captioner.stop()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
