from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QListWidget, QListWidgetItem, QComboBox, QLabel,
    QDialogButtonBox
)
from PyQt5.QtCore import Qt

class BulkEditDialog(QDialog):
    def __init__(self, words, speakers, mode="edit", parent=None):
        super().__init__(parent)
        self.words = words
        self.speakers = speakers
        self.mode = mode  # "edit" or "delete"
        self.selected_indices = []
        self.new_speaker = ""
        self.setup_ui()
        
    def setup_ui(self):
        self.setWindowTitle("Bulk Edit Words" if self.mode == "edit" else "Delete Words")
        layout = QVBoxLayout(self)
        
        # Instructions
        instruction_text = ("Select words to change speaker" if self.mode == "edit" 
                          else "Select words to delete")
        layout.addWidget(QLabel(instruction_text))
        
        # Word list
        self.word_list = QListWidget()
        self.word_list.setSelectionMode(QListWidget.ExtendedSelection)
        for i, word in enumerate(self.words):
            item = QListWidgetItem(f"{word['word']} ({word.get('speaker', '')})")
            item.setData(Qt.UserRole, i)  # Store the word index
            self.word_list.addItem(item)
        layout.addWidget(self.word_list)
        
        if self.mode == "edit":
            # Speaker selection for edit mode
            speaker_layout = QHBoxLayout()
            speaker_layout.addWidget(QLabel("New Speaker:"))
            self.speaker_combo = QComboBox()
            self.speaker_combo.addItems(self.speakers + [""])
            speaker_layout.addWidget(self.speaker_combo)
            layout.addLayout(speaker_layout)
        
        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setMinimumWidth(400)
        self.setMinimumHeight(500)
        
    def get_selected_indices(self):
        return [item.data(Qt.UserRole) for item in self.word_list.selectedItems()]
        
    def get_new_speaker(self):
        if self.mode == "edit":
            return self.speaker_combo.currentText()
        return None 