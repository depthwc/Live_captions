import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QMenu
from PySide6.QtCore import Qt, QPoint, Signal, Slot
from PySide6.QtGui import QColor, QFont, QPalette, QAction, QCursor

class CaptionOverlay(QMainWindow):
    """
    A PySide6 frameless window that is transparent with a darker styling and a distinct border.
    """
    # Define a Qt signal for threading safe UI updates. This allows the backend Whisper thread
    # to send text strings safely to the PySide6 main UI thread.
    update_text_signal = Signal(str)

    def __init__(self):
        super().__init__()
        
        # Frameless window, stays on top, tool window (hides from taskbar slightly better)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        
        # Tells Windows to allow the background of the window to be transparent
        self.setAttribute(Qt.WA_TranslucentBackground)

        # Default size and position near the bottom of the screen
        self.resize(1000, 120)

        # Central widget applies the darker transparent background and the border
        self.central_widget = QWidget(self)
        self.central_widget.setObjectName("mainWidget")
        self.central_widget.setStyleSheet("""
            #mainWidget {
                background-color: rgba(15, 15, 15, 210); /* Darker transparent background */
                border: 2px solid rgba(120, 120, 120, 255); /* Visible softer border */
                border-radius: 12px;
            }
        """)
        
        # Internal layout padding so text doesn't hit the border
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setContentsMargins(20, 20, 20, 20)
        
        # The text label itself
        self.label = QLabel("Waiting for system audio...")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setWordWrap(True)
        self.label.setStyleSheet("""
            color: #EEEEEE;
            font-size: 26px;
            font-family: 'Segoe UI', Helvetica, Arial, sans-serif;
            font-weight: 600;
        """)
        
        self.layout.addWidget(self.label)
        self.setCentralWidget(self.central_widget)
        
        # Connect the cross-thread signal directly to a method
        self.update_text_signal.connect(self._set_text)
        
        # Variables required for click-and-drag logic on a frameless window
        self._drag_active = False
        self._drag_pos = QPoint()
        
    @Slot(str)
    def _set_text(self, text):
        """Executes securely on the main UI thread via the Signal."""
        self.label.setText(text)
        
    def mousePressEvent(self, event):
        """Capture the mouse down relative position."""
        if event.button() == Qt.LeftButton:
            self._drag_active = True
            # We calculate where exactly inside the window the user clicked
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        """Move the window while holding left click."""
        if self._drag_active:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        """Stop dragging."""
        if event.button() == Qt.LeftButton:
            self._drag_active = False
            event.accept()

    def contextMenuEvent(self, event):
        """Right click context menu to close the app."""
        context_menu = QMenu(self)
        context_menu.setStyleSheet("""
            QMenu {
                background-color: rgb(30, 30, 30);
                color: white;
                border: 1px solid gray;
                padding: 5px;
            }
            QMenu::item {
                padding: 5px 20px 5px 20px;
            }
            QMenu::item:selected {
                background-color: rgb(70, 70, 70);
            }
        """)
        
        close_action = context_menu.addAction("Exit Live Captions")
        action = context_menu.exec(self.mapToGlobal(event.pos()))
        
        if action == close_action:
            self.close()

class GUIController:
    """Wrapper to bridge LiveCaptioner with a single PySide app."""
    def __init__(self):
        # We assume QApplication is already instantiated in main.py
        self.window = CaptionOverlay()
        
    def show_window(self):
        self.window.show()
        
    def update_caption(self, text):
        # Emitting a signal is thread-safe in PySide6 regardless of the calling thread.
        self.window.update_text_signal.emit(text)
