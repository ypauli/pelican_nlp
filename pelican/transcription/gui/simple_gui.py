from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt
import sys
import os
from pathlib import Path
from .transcription_dialog import TranscriptionDialog
from ..core.audio import AudioFile


class TranscriptionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_audio_file = None
        self.current_audio_path = None
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("PELICAN Transcription")
        self.setMinimumSize(600, 400)

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # File selection area
        file_layout = QHBoxLayout()
        self.file_label = QLabel("No audio file selected")
        file_layout.addWidget(self.file_label)
        
        select_button = QPushButton("Select Audio File")
        select_button.clicked.connect(self.select_audio_file)
        file_layout.addWidget(select_button)
        
        layout.addLayout(file_layout)

        # Transcription button
        self.transcribe_button = QPushButton("Transcribe")
        self.transcribe_button.clicked.connect(self.start_transcription)
        self.transcribe_button.setEnabled(False)  # Disabled until file is loaded
        layout.addWidget(self.transcribe_button)

        # Add stretcher to push everything to the top
        layout.addStretch()

    def select_audio_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio File",
            "",
            "Audio Files (*.wav *.mp3 *.m4a *.flac);;All Files (*.*)"
        )
        
        if file_path:
            try:
                self.current_audio_path = file_path
                self.current_audio_file = AudioFile(file_path)
                self.file_label.setText(f"Selected: {Path(file_path).name}")
                self.transcribe_button.setEnabled(True)
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to load audio file: {str(e)}"
                )

    def start_transcription(self):
        if not self.current_audio_path:
            return
            
        dialog = TranscriptionDialog(self.current_audio_path, self)
        dialog.exec()


def main():
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = TranscriptionGUI()
    window.show()
    sys.exit(app.exec()) 