import sys
import os
import threading

# live_caption.py applies the numpy monkey-patch before importing soundcard,
# so we must NOT import soundcard here at the top level.
from live_caption import LiveCaptioner
from gui import GUIController

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

# Suppress HuggingFace symlinks cache warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def main():
    print("Initializing PySide6 Live Captions...")

    # Must instantiate QApplication before any GUI widgets
    app = QApplication(sys.argv)

    # Use default speaker loopback automatically (no blocking input() call)
    # Force CPU mode: avoid CUDA DLL errors (cublas64_12.dll) on machines without CUDA
    captioner = LiveCaptioner(model_size="base", speaker_index=None, device="cpu", compute_type="int8")

    # Initialize GUI
    gui_manager = GUIController()

    # Thread-safe callback: route transcribed text to the PySide UI thread via Signal
    captioner.add_callback(gui_manager.update_caption)

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
