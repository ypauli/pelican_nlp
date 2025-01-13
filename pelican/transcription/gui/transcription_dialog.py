from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox,
    QPushButton, QProgressBar, QTextEdit
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from pathlib import Path
import os
from typing import Optional
import sys
import os.path

# Add parent directory to Python path to make relative imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.main import process_audio


class TranscriptionWorker(QThread):
    progress_update = pyqtSignal(str, float, str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, params: dict):
        super().__init__()
        self.params = params

    def run(self):
        try:
            output_file = process_audio(
                **self.params,
                progress_callback=lambda step, progress, message: 
                    self.progress_update.emit(step, progress, message)
            )
            self.finished.emit(output_file)
        except Exception as e:
            self.error.emit(str(e))


class TranscriptionDialog(QDialog):
    def __init__(self, audio_file_path: str, parent=None):
        super().__init__(parent)
        self.audio_file_path = audio_file_path
        self.worker = None
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("Transcription Settings")
        layout = QVBoxLayout()

        # Settings
        settings_layout = QVBoxLayout()

        # HF Token
        token_layout = QHBoxLayout()
        token_layout.addWidget(QLabel("HuggingFace Token:"))
        self.token_input = QLineEdit()
        self.token_input.setEchoMode(QLineEdit.Password)
        token_layout.addWidget(self.token_input)
        settings_layout.addLayout(token_layout)

        # Number of Speakers
        speakers_layout = QHBoxLayout()
        speakers_layout.addWidget(QLabel("Number of Speakers:"))
        self.speakers_input = QSpinBox()
        self.speakers_input.setMinimum(1)
        self.speakers_input.setMaximum(10)
        self.speakers_input.setValue(2)
        speakers_layout.addWidget(self.speakers_input)
        settings_layout.addLayout(speakers_layout)

        # Model Selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self.model_input = QComboBox()
        self.model_input.addItems(["openai/whisper-large", "openai/whisper-medium", "openai/whisper-small"])
        model_layout.addWidget(self.model_input)
        settings_layout.addLayout(model_layout)

        # Language Selection
        lang_layout = QHBoxLayout()
        lang_layout.addWidget(QLabel("Language:"))
        self.lang_input = QComboBox()
        self.lang_input.addItems(["de", "en", "fr", "es"])
        lang_layout.addWidget(self.lang_input)
        settings_layout.addLayout(lang_layout)

        # Pause Threshold
        pause_layout = QHBoxLayout()
        pause_layout.addWidget(QLabel("Pause Threshold (s):"))
        self.pause_input = QDoubleSpinBox()
        self.pause_input.setMinimum(0.1)
        self.pause_input.setMaximum(10.0)
        self.pause_input.setValue(2.0)
        pause_layout.addWidget(self.pause_input)
        settings_layout.addLayout(pause_layout)

        # Max Utterance Duration
        duration_layout = QHBoxLayout()
        duration_layout.addWidget(QLabel("Max Utterance Duration (s):"))
        self.duration_input = QDoubleSpinBox()
        self.duration_input.setMinimum(5.0)
        self.duration_input.setMaximum(120.0)
        self.duration_input.setValue(30.0)
        duration_layout.addWidget(self.duration_input)
        settings_layout.addLayout(duration_layout)

        layout.addLayout(settings_layout)

        # Progress Bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # Console Output
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setMinimumHeight(200)
        layout.addWidget(self.console)

        # Buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Transcription")
        self.start_button.clicked.connect(self.start_transcription)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def start_transcription(self):
        if not self.token_input.text():
            self.console.append("Error: HuggingFace token is required")
            return

        self.start_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.console.clear()

        params = {
            "file_path": self.audio_file_path,
            "hf_token": self.token_input.text(),
            "output_dir": "output",
            "num_speakers": self.speakers_input.value(),
            "model": self.model_input.currentText(),
            "language": self.lang_input.currentText(),
            "pause_threshold": self.pause_input.value(),
            "max_utterance_duration": self.duration_input.value()
        }

        self.worker = TranscriptionWorker(params)
        self.worker.progress_update.connect(self.update_progress)
        self.worker.finished.connect(self.transcription_complete)
        self.worker.error.connect(self.transcription_error)
        self.worker.start()

    def update_progress(self, step: str, progress: float, message: str):
        self.progress_bar.setValue(int(progress * 100))
        if message:
            self.console.append(message)

    def transcription_complete(self, output_file: str):
        self.console.append(f"\nTranscription completed successfully!")
        self.console.append(f"Output saved to: {output_file}")
        self.start_button.setEnabled(True)
        self.accept()

    def transcription_error(self, error_message: str):
        self.console.append(f"\nError during transcription: {error_message}")
        self.start_button.setEnabled(True) 