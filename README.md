# Live Captions Overlay

Frameless, always-on-top live captions app for Windows-style workflows.

## Features

- `PySide6` custom UI
- Drag to move, optional resize with size grip
- Scrollable caption history
- Real-time speech-to-text with `Vosk`
- Settings dialog for:
  - Language (switches recognition model)
  - Border/box/text colors (any color picker)
  - Font size
  - Position

## Install

pip install -r requirements.txt

All language choices use only `vosk-model-small-*` models.
You can change the models by changing the links in the `live_caption.py` file.


## Notes

- Settings are persisted to:
  - `%LOCALAPPDATA%\py-live-captions\settings.json` on Windows
- Models are cached under:
  - `%LOCALAPPDATA%\py-live-captions\models`
- Small-model list is fetched from `https://alphacephei.com/vosk/models` at startup, with built-in fallback list if offline.
