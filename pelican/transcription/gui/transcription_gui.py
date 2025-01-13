import sys
import os
import json
import tempfile
import numpy as np
import librosa
import time
import re
from typing import List, Dict

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QFileDialog, QMessageBox,
    QInputDialog, QMenu, QAction, QUndoStack, QScrollBar, QLineEdit,
    QDialog
)
from PyQt5.QtCore import Qt, QTimer, QThread
from PyQt5.QtGui import QColor, QCursor

import pyqtgraph as pg
from pydub import AudioSegment
from pydub.playback import _play_with_simpleaudio as play_audio

from core.transcript import Transcript
from core.transcription_controller import TranscriptController
from gui import (
    AudioLoader, DraggableLine, SpeakerManagementDialog,
    EditWordCommand, EditSpeakerCommand, MoveBoundaryCommand,
    SplitWordCommand, AddWordCommand, DeleteWordCommand,
    BulkEditCommand, BulkDeleteCommand,
    WaveformCanvas
)
from gui.dialogs.bulk_edit_dialog import BulkEditDialog

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PELICAn Transcription Tool")
        self.setGeometry(100, 100, 1600, 900)

        # Initialize variables
        self.audio_segment = None
        self.is_playing = False
        self.play_obj = None
        self.current_time = 0.0
        self.speakers = []
        self.undo_stack = QUndoStack(self)
        self.transcript = None
        self.controller = None  # Initialize controller as None

        # Setup UI components
        self.setup_ui()
        self.setup_signals()

        # Start the autosave timer
        self.autosave_timer = QTimer(self)
        self.autosave_timer.timeout.connect(self.autosave)
        self.autosave_timer.start(5000)  # Trigger autosave every 5 seconds

        # Load autosave if exists
        self.temp_file_path = tempfile.NamedTemporaryFile(delete=False, suffix='.json').name
        self.load_autosave()

    def setup_ui(self):
        self.waveform_widget = QWidget()
        main_layout = QVBoxLayout(self.waveform_widget)
        main_layout.setSpacing(5)
        
        # Top bar with essential controls
        top_bar = QWidget()
        top_bar_layout = QHBoxLayout(top_bar)
        top_bar_layout.setContentsMargins(5, 5, 5, 5)
        
        # Left side - File Operations
        file_group = QWidget()
        file_layout = QHBoxLayout(file_group)
        file_layout.setContentsMargins(0, 0, 0, 0)
        file_layout.setSpacing(5)
        
        # Create actions with shortcuts
        load_audio_action = QAction("Load Audio (Ctrl+O)", self)
        load_audio_action.setShortcut("Ctrl+O")
        load_audio_action.triggered.connect(self.load_audio)
        
        load_transcript_action = QAction("Load Transcript (Ctrl+T)", self)
        load_transcript_action.setShortcut("Ctrl+T")
        load_transcript_action.triggered.connect(self.load_transcript)
        
        save_action = QAction("Save Annotations (Ctrl+S)", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_annotations)
        
        # Add actions to buttons
        load_audio_button = QPushButton("Load Audio (Ctrl+O)")
        load_audio_button.clicked.connect(load_audio_action.trigger)
        
        load_transcript_button = QPushButton("Load Transcript (Ctrl+T)")
        load_transcript_button.clicked.connect(load_transcript_action.trigger)
        
        save_button = QPushButton("Save Annotations (Ctrl+S)")
        save_button.clicked.connect(save_action.trigger)
        
        file_layout.addWidget(load_audio_button)
        file_layout.addWidget(load_transcript_button)
        file_layout.addWidget(save_button)
        
        # Right side - Playback Controls
        playback_group = QWidget()
        playback_layout = QHBoxLayout(playback_group)
        playback_layout.setContentsMargins(0, 0, 0, 0)
        playback_layout.setSpacing(5)
        
        play_action = QAction("Play/Pause (Space)", self)
        play_action.setShortcut("Space")
        play_action.triggered.connect(self.toggle_playback)
        
        stop_action = QAction("Stop (Esc)", self)
        stop_action.setShortcut("Esc")
        stop_action.triggered.connect(self.stop_playback)
        
        self.play_button = QPushButton("Play (Space)")
        self.play_button.clicked.connect(play_action.trigger)
        
        self.stop_button = QPushButton("Stop (Esc)")
        self.stop_button.clicked.connect(stop_action.trigger)
        
        playback_layout.addWidget(self.play_button)
        playback_layout.addWidget(self.stop_button)
        
        # Add groups to top bar
        top_bar_layout.addWidget(file_group)
        top_bar_layout.addStretch(1)
        top_bar_layout.addWidget(playback_group)
        
        # Add top bar to main layout
        main_layout.addWidget(top_bar)
        
        # Add waveform canvas
        self.canvas = WaveformCanvas(parent=self.waveform_widget, main_window=self)
        main_layout.addWidget(self.canvas)

        # Bottom controls for editing operations
        bottom_controls = QWidget()
        bottom_layout = QHBoxLayout(bottom_controls)
        bottom_layout.setContentsMargins(5, 5, 5, 5)
        
        # Edit Operations
        edit_group = QWidget()
        edit_layout = QHBoxLayout(edit_group)
        edit_layout.setContentsMargins(0, 0, 0, 0)
        edit_layout.setSpacing(5)
        
        recalc_action = QAction("Recalculate Utterances (Ctrl+R)", self)
        recalc_action.setShortcut("Ctrl+R")
        recalc_action.triggered.connect(self.recalculate_utterances)
        
        manage_speakers_action = QAction("Manage Speakers (Ctrl+M)", self)
        manage_speakers_action.setShortcut("Ctrl+M")
        manage_speakers_action.triggered.connect(self.manage_speakers)
        
        bulk_edit_action = QAction("Bulk Edit (Ctrl+E)", self)
        bulk_edit_action.setShortcut("Ctrl+E")
        bulk_edit_action.triggered.connect(lambda: self.show_bulk_dialog("edit"))
        
        bulk_delete_action = QAction("Bulk Delete (Ctrl+D)", self)
        bulk_delete_action.setShortcut("Ctrl+D")
        bulk_delete_action.triggered.connect(lambda: self.show_bulk_dialog("delete"))
        
        recalc_utterances_button = QPushButton("Recalculate Utterances (Ctrl+R)")
        recalc_utterances_button.clicked.connect(recalc_action.trigger)
        
        manage_speakers_button = QPushButton("Manage Speakers (Ctrl+M)")
        manage_speakers_button.clicked.connect(manage_speakers_action.trigger)
        
        bulk_edit_button = QPushButton("Bulk Edit (Ctrl+E)")
        bulk_edit_button.clicked.connect(bulk_edit_action.trigger)
        
        bulk_delete_button = QPushButton("Bulk Delete (Ctrl+D)")
        bulk_delete_button.clicked.connect(bulk_delete_action.trigger)
        
        edit_layout.addWidget(recalc_utterances_button)
        edit_layout.addWidget(manage_speakers_button)
        edit_layout.addWidget(bulk_edit_button)
        edit_layout.addWidget(bulk_delete_button)
        
        # Add edit group to bottom layout
        bottom_layout.addWidget(edit_group)
        
        # Add bottom controls to main layout
        main_layout.addWidget(bottom_controls)

        # Undo/Redo at the bottom
        undo_redo_widget = QWidget()
        undo_redo_layout = QHBoxLayout(undo_redo_widget)
        undo_redo_layout.setContentsMargins(5, 0, 5, 5)
        
        undo_action = QAction("Undo (Ctrl+Z)", self)
        undo_action.setShortcut("Ctrl+Z")
        undo_action.triggered.connect(self.undo_stack.undo)
        
        redo_action = QAction("Redo (Ctrl+Shift+Z)", self)
        redo_action.setShortcut("Ctrl+Shift+Z")
        redo_action.triggered.connect(self.undo_stack.redo)
        
        undo_button = QPushButton("Undo (Ctrl+Z)")
        undo_button.clicked.connect(undo_action.trigger)
        
        redo_button = QPushButton("Redo (Ctrl+Shift+Z)")
        redo_button.clicked.connect(redo_action.trigger)
        
        undo_redo_layout.addStretch(1)
        undo_redo_layout.addWidget(undo_button)
        undo_redo_layout.addWidget(redo_button)
        undo_redo_layout.addStretch(1)
        
        main_layout.addWidget(undo_redo_widget)

        # Add all actions to the window to enable shortcuts
        self.addAction(load_audio_action)
        self.addAction(load_transcript_action)
        self.addAction(save_action)
        self.addAction(play_action)
        self.addAction(stop_action)
        self.addAction(recalc_action)
        self.addAction(manage_speakers_action)
        self.addAction(bulk_edit_action)
        self.addAction(bulk_delete_action)
        self.addAction(undo_action)
        self.addAction(redo_action)

        self.setCentralWidget(self.waveform_widget)

    def setup_signals(self):
        self.canvas.boundary_changed.connect(self.on_boundary_changed)
        self.canvas.waveform_clicked.connect(self.on_waveform_clicked)
        self.canvas.word_double_clicked.connect(self.on_word_double_clicked)
        self.canvas.word_right_clicked.connect(self.on_word_right_clicked)
        self.canvas.audio_loaded.connect(self.on_audio_loaded)
        self.canvas.loading_error.connect(self.on_audio_load_error)

    def load_audio(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Audio File",
            "",
            "Audio Files (*.wav *.mp3 *.flac *.ogg);;All Files (*)",
            options=options,
        )
        if file_path:
            self.canvas.load_audio(file_path)
            # Load audio segment for playback
            try:
                self.audio_segment = AudioSegment.from_file(file_path).set_channels(1)
            except Exception as e:
                QMessageBox.critical(self, "Audio Load Error", f"Failed to load audio for playback:\n{str(e)}")

    def on_audio_loaded(self):
        self.statusBar().showMessage("Audio loaded successfully.", 5000)
        self.canvas.adjust_view_range(self.current_time)

    def on_audio_load_error(self, error_message):
        QMessageBox.critical(self, "Audio Load Error", f"Failed to load audio file:\n{error_message}")
        self.statusBar().showMessage("Failed to load audio.", 5000)

    def load_transcript(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Transcript File",
            "",
            "JSON Files (*.json);;All Files (*)",
            options=options,
        )
        if file_path:
            try:
                self.transcript = Transcript.from_json_file(file_path)
                self.controller = TranscriptController(self.transcript)  # Initialize controller
                self.speakers = self.controller.get_speakers()  # Use controller to get speakers
                self.canvas.load_words(self.transcript.combined_data)
                self.canvas.load_utterances(self.transcript.combined_utterances)
                print(f"Loaded transcript file: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error Loading Transcript", f"Failed to load transcript:\n{str(e)}")

    def on_boundary_changed(self, idx, boundary_type, new_pos, old_pos):
        if idx == -1:
            return
        command = MoveBoundaryCommand(self, idx, boundary_type, old_pos, new_pos)
        self.undo_stack.push(command)
        # Do not automatically update utterances
        self.autosave()

    def on_waveform_clicked(self, time):
        self.current_time = time
        self.canvas.update_playtime_line(self.current_time)

    def find_word_at_time(self, time):
        words = self.transcript.combined_data
        for idx, word in enumerate(words):
            start_time = float(word['start_time'])
            end_time = float(word['end_time'])
            if start_time <= time < end_time:
                return idx, word
        return None, None
    
    def on_word_clicked(self, idx):
        """
        Triggered when a word label is clicked. Allows inline editing of the word.
        """
        word_data = self.transcript.combined_data[idx]
        word_label = self.canvas.connecting_lines[idx]['label']

        # Remove the current label from the canvas
        self.canvas.plot_widget.removeItem(word_label)

        # Calculate position for the QLineEdit
        word_pos = word_label.pos()
        word_text = word_data['word']

        # Create a QLineEdit widget for inline editing
        self.editing_line = QLineEdit(self.canvas)
        self.editing_line.setText(word_text)
        self.editing_line.setFixedWidth(150)  # Set appropriate width
        self.editing_line.setAlignment(Qt.AlignCenter)

        # Map the label's position to the scene and move QLineEdit there
        scene_pos = self.canvas.plot_widget.plotItem.vb.mapViewToScene(word_pos)
        self.editing_line.move(scene_pos.toPoint())
        self.editing_line.show()
        self.editing_line.setFocus()

        # Connect editing finished signal to finalize changes
        self.editing_line.editingFinished.connect(lambda: self.finish_editing(idx))
        
    def finish_editing(self, idx):
        """
        Finalizes word editing, updates the label, and restores functionality.
        """
        # Get the new word from QLineEdit
        new_word = self.editing_line.text()

        # Update the transcript data
        old_word = self.transcript.combined_data[idx]['word']
        if new_word != old_word:
            command = EditWordCommand(self, idx, old_word, new_word)
            self.undo_stack.push(command)

        # Remove the QLineEdit
        self.editing_line.deleteLater()
        self.editing_line = None

        # Create and add the updated label
        word_label = pg.TextItem(new_word, anchor=(0.5, 0), color='white')
        mid_x = (float(self.transcript.combined_data[idx]['start_time']) +
                float(self.transcript.combined_data[idx]['end_time'])) / 2
        word_label.setPos(mid_x, 0.6)  # Adjust label position
        word_label.mouseClickEvent = lambda ev, idx=idx: self.on_word_clicked(idx)
        self.canvas.plot_widget.addItem(word_label)

        # Update the canvas label reference
        self.canvas.connecting_lines[idx]['label'] = word_label

        # Trigger autosave or other required updates
        self.autosave()

    def on_word_double_clicked(self, time):
        idx, word = self.find_word_at_time(time)
        if word is not None:
            # Edit existing word
            new_word, ok = QInputDialog.getText(self, "Edit Word", "New word:", text=word['word'])
            if ok and new_word != word['word']:
                old_value = word['word']
                command = EditWordCommand(self, idx, old_value, new_word)
                self.undo_stack.push(command)
                self.autosave()
        else:
            # Add new word at double-clicked location
            self.add_word(time)

    def on_word_right_clicked(self, time):
        idx, word = self.find_word_at_time(time)
        if word is not None:
            menu = QMenu(self)
            
            # Speaker submenu
            speaker_menu = QMenu("Set Speaker", self)
            speakers = self.speakers + [""]  # Add empty speaker option
            for speaker in speakers:
                display_text = speaker if speaker else "(No Speaker)"
                action = QAction(display_text, self)
                action.triggered.connect(lambda checked, s=speaker: self.set_word_speaker(idx, s))
                speaker_menu.addAction(action)
            menu.addMenu(speaker_menu)
            
            # Split word action
            split_action = QAction("Split Word Here", self)
            split_action.triggered.connect(lambda: self.split_word(idx, time))
            menu.addAction(split_action)
            
            # Delete word action
            delete_action = QAction("Delete Word", self)
            delete_action.triggered.connect(lambda: self.delete_word(idx))
            menu.addAction(delete_action)
            
            menu.exec_(QCursor.pos())
        else:
            # If clicked on empty space, offer to add a new word
            menu = QMenu(self)
            add_action = QAction("Add Word Here", self)
            add_action.triggered.connect(lambda: self.add_word(time))
            menu.addAction(add_action)
            menu.exec_(QCursor.pos())

    def split_word(self, idx, split_time):
        """Split a word into two at the specified time point."""
        if not self.controller:
            return
            
        word = self.controller.get_word_data(idx)
        if not word:
            return
            
        # Validate split point is within word boundaries
        start_time = float(word['start_time'])
        end_time = float(word['end_time'])
        if not (start_time < split_time < end_time):
            return
            
        # Get text for the new word
        new_word, ok = QInputDialog.getText(
            self, 
            "Split Word", 
            f"Enter text for the new word (original: '{word['word']}')"
        )
        if ok:
            command = SplitWordCommand(self, idx, split_time, new_word)
            self.undo_stack.push(command)
            self.autosave()

    def add_word(self, time_point, duration=0.1):
        """Add a new word at the specified time point."""
        if not self.controller:
            return
            
        # Get text for the new word
        new_word, ok = QInputDialog.getText(
            self, 
            "Add Word", 
            "Enter new word:"
        )
        if ok:
            command = AddWordCommand(self, time_point, new_word, duration)
            self.undo_stack.push(command)
            self.autosave()

    def set_word_speaker(self, idx, speaker):
        if not self.controller:
            return
        word = self.controller.get_word_data(idx)
        if not word:
            return
        old_speaker = word.get('speaker', '')
        if speaker != old_speaker:
            command = EditSpeakerCommand(self, idx, old_speaker, speaker)
            self.undo_stack.push(command)
            if speaker and speaker not in self.speakers:
                self.speakers.append(speaker)
            self.autosave()

    def toggle_playback(self):
        if self.audio_segment is None:
            QMessageBox.warning(self, "No Audio", "Please load an audio file first.")
            return
        if self.is_playing:
            self.pause_playback()
        else:
            self.start_playback()

    def start_playback(self):
        if self.audio_segment is not None:
            try:
                start_ms = int(self.current_time * 1000)
                sliced_audio = self.audio_segment[start_ms:]    
                self.play_obj = play_audio(sliced_audio)
                self.is_playing = True
                self.play_button.setText("Pause")
                # Record the start time and position
                self.playback_start_time = time.time()
                self.playback_start_position = self.current_time
                self.playback_timer = QTimer()
                self.playback_timer.timeout.connect(self.update_current_time)
                self.playback_timer.start(50)  # Increased frequency for smoother updates
            except Exception as e:
                QMessageBox.critical(self, "Playback Error", f"Failed to play audio:\n{str(e)}")

    def pause_playback(self):
        if self.play_obj:
            self.play_obj.stop()
            self.play_obj = None
        self.is_playing = False
        self.play_button.setText("Play")
        if hasattr(self, 'playback_timer') and self.playback_timer.isActive():
            self.playback_timer.stop()
        # Update current_time to the actual playback position
        elapsed_time = time.time() - self.playback_start_time
        self.current_time = self.playback_start_position + elapsed_time

    def stop_playback(self):
        if self.play_obj:
            self.play_obj.stop()
            self.play_obj = None
        self.is_playing = False
        self.play_button.setText("Play")
        self.current_time = 0.0
        self.canvas.update_playtime_line(self.current_time)
        if hasattr(self, 'playback_timer') and self.playback_timer.isActive():
            self.playback_timer.stop()

    def update_current_time(self):
        elapsed_time = time.time() - self.playback_start_time
        self.current_time = self.playback_start_position + elapsed_time
        if self.current_time >= self.canvas.duration:
            self.stop_playback()
            return
        self.canvas.update_playtime_line(self.current_time)

    def recalculate_utterances(self):
        if not self.transcript:
            QMessageBox.warning(self, "No Transcript", "Please load a transcript first.")
            return
        try:
            self.transcript.aggregate_to_utterances()
            self.canvas.load_utterances(self.transcript.combined_utterances)
            self.statusBar().showMessage("Utterances recalculated successfully.", 5000)
            self.autosave()
        except Exception as e:
            QMessageBox.critical(self, "Aggregation Error", f"Failed to recalculate utterances:\n{str(e)}")

    def autosave(self):
        if self.transcript:
            try:
                self.transcript.save_as_json(self.temp_file_path)
                print(f"Autosaved annotations to {self.temp_file_path}")
            except Exception as e:
                print(f"Autosave failed: {e}")

    def load_autosave(self):
        if os.path.exists(self.temp_file_path):
            try:
                self.transcript = Transcript.from_json_file(self.temp_file_path)
                self.controller = TranscriptController(self.transcript)  # Initialize controller
                self.speakers = self.controller.get_speakers()  # Use controller to get speakers
                self.canvas.load_words(self.transcript.combined_data)
                self.canvas.load_utterances(self.transcript.combined_utterances)
                QMessageBox.information(self, "Recovery", "Recovered annotations from autosave.")
            except Exception as e:
                print(f"Failed to recover autosave: {e}")

    def closeEvent(self, event):
        reply = QMessageBox.question(
            self, 'Exit',
            "Do you want to save your annotations before exiting?",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            QMessageBox.Yes
        )

        if reply == QMessageBox.Yes:
            if not self.validate_annotations():
                event.ignore()
                return
            self.save_annotations()
            if os.path.exists(self.temp_file_path):
                os.remove(self.temp_file_path)
            event.accept()
        elif reply == QMessageBox.No:
            if os.path.exists(self.temp_file_path):
                os.remove(self.temp_file_path)
            event.accept()
        else:
            event.ignore()

    def save_annotations(self):
        if not self.validate_annotations():
            return
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Annotations",
            "",
            "JSON Files (*.json);;All Files (*)",
            options=options,
        )
        if file_path:
            try:
                self.transcript.save_as_json(file_path)
                QMessageBox.information(self, "Save Successful", f"Annotations saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save annotations:\n{str(e)}")

    def validate_annotations(self):
        words = self.transcript.combined_data
        words_sorted = sorted(words, key=lambda w: float(w['start_time']))
        for i, word in enumerate(words_sorted):
            try:
                start_time = float(word['start_time'])
                end_time = float(word['end_time'])
                if start_time > end_time:
                    QMessageBox.warning(self, "Invalid Annotation", f"Start time must be less than end time for word '{word['word']}'.")
                    return False
                if i < len(words_sorted) - 1:
                    next_word = words_sorted[i + 1]
                    next_start = float(next_word['start_time'])
                    if end_time > next_start:
                        QMessageBox.warning(
                            self, "Invalid Annotation",
                            f"Annotations for words '{word['word']}' and '{next_word['word']}' overlap."
                        )
                        return False
            except ValueError:
                QMessageBox.warning(self, "Invalid Annotation", f"Non-numeric start or end time for word '{word['word']}'.")
                return False
        return True

    def manage_speakers(self):
        """Open dialog to manage speaker labels."""
        if not self.transcript or not self.controller:
            QMessageBox.warning(self, "Warning", "Please load a transcript first.")
            return

        dialog = SpeakerManagementDialog(self.speakers, parent=self)
        if dialog.exec_() == QDialog.Accepted:
            changes = dialog.get_updated_speakers()
            from_speaker = changes['from_speaker']
            to_speaker = changes['to_speaker']

            self.controller.merge_speakers(from_speaker, to_speaker)
            self.speakers = self.controller.get_speakers()  # Update speakers list

            self.canvas.load_words(self.transcript.combined_data)
            self.canvas.load_utterances(self.transcript.combined_utterances)
            self.statusBar().showMessage(f"Merged speaker {from_speaker} into {to_speaker}.", 5000)
            self.autosave()

    def delete_word(self, idx):
        """Delete a word at the specified index."""
        if not self.controller:
            return
            
        command = DeleteWordCommand(self, idx)
        self.undo_stack.push(command)
        self.autosave()

    def show_bulk_dialog(self, mode):
        """Show dialog for bulk editing or deleting words."""
        if not self.controller or not self.transcript:
            QMessageBox.warning(self, "No Transcript", "Please load a transcript first.")
            return
            
        dialog = BulkEditDialog(
            self.transcript.combined_data,
            self.speakers,
            mode=mode,
            parent=self
        )
        
        if dialog.exec_() == QDialog.Accepted:
            indices = dialog.get_selected_indices()
            if not indices:
                return
                
            if mode == "edit":
                new_speaker = dialog.get_new_speaker()
                command = BulkEditCommand(
                    self, indices, {'speaker': new_speaker},
                    f"Set speaker to {new_speaker} for {len(indices)} words"
                )
            else:  # delete mode
                command = BulkDeleteCommand(
                    self, indices,
                    f"Delete {len(indices)} words"
                )
                
            self.undo_stack.push(command)
            self.autosave()


# --- MainExecution ---

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
