from typing import List, Dict
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QComboBox, QLabel, QDialogButtonBox

class SpeakerManagementDialog(QDialog):
    """Dialog for managing speaker labels."""
    
    def __init__(self, speakers: List[str], parent=None):
        super().__init__(parent)
        self.speakers = speakers
        self.setup_ui()

    def setup_ui(self):
        """Setup the dialog UI."""
        self.setWindowTitle("Manage Speakers")
        layout = QVBoxLayout(self)

        # From speaker selection
        from_label = QLabel("Merge from speaker:")
        self.from_speaker_combo = QComboBox()
        self.from_speaker_combo.addItems(self.speakers)
        layout.addWidget(from_label)
        layout.addWidget(self.from_speaker_combo)

        # To speaker selection
        to_label = QLabel("Into speaker:")
        self.to_speaker_combo = QComboBox()
        self.to_speaker_combo.setEditable(True)
        self.to_speaker_combo.addItems(self.speakers + ["New Speaker"])
        layout.addWidget(to_label)
        layout.addWidget(self.to_speaker_combo)

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def get_updated_speakers(self) -> Dict[str, str]:
        """Get the selected speaker changes."""
        return {
            'from_speaker': self.from_speaker_combo.currentText(),
            'to_speaker': self.to_speaker_combo.currentText(),
        } 