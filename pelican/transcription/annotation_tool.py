import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QPushButton, QWidget, QComboBox, QFileDialog, QHBoxLayout, QMessageBox,
    QLabel, QCheckBox, QSlider, QAbstractItemView, QInputDialog, QUndoStack, QUndoCommand,
    QMenu, QAction, QScrollBar
)
from PyQt5.QtCore import Qt, QTimer, QObject, pyqtSignal, QThread, QPoint
from PyQt5.QtGui import QPixmap, QColor
import pyqtgraph as pg
from pydub import AudioSegment
from pydub.playback import _play_with_simpleaudio as play_audio
import json
import tempfile
import os


class AddRowCommand(QUndoCommand):
    def __init__(self, tool, row_position, row_data, description="Add Row"):
        super().__init__(description)
        self.tool = tool
        self.row_position = row_position
        self.row_data = row_data

    def redo(self):
        self.tool.insert_row(self.row_position, self.row_data)

    def undo(self):
        self.tool.remove_row(self.row_position)


class DeleteRowsCommand(QUndoCommand):
    def __init__(self, tool, rows_data, row_positions, description="Delete Rows"):
        super().__init__(description)
        self.tool = tool
        self.rows_data = rows_data  # List of row data dictionaries
        self.row_positions = row_positions  # List of row indices

    def redo(self):
        # Delete rows in reverse order to avoid shifting
        for row in sorted(self.row_positions, reverse=True):
            self.tool.remove_row(row)

    def undo(self):
        # Insert rows back in original order
        for row, data in sorted(zip(self.row_positions, self.rows_data)):
            self.tool.insert_row(row, data)


class EditCellCommand(QUndoCommand):
    def __init__(self, tool, row, column, old_value, new_value, description="Edit Cell"):
        super().__init__(description)
        self.tool = tool
        self.row = row
        self.column = column
        self.old_value = old_value
        self.new_value = new_value

    def redo(self):
        self.tool.set_cell(self.row, self.column, self.new_value)

    def undo(self):
        self.tool.set_cell(self.row, self.column, self.old_value)


class BulkEditSpeakerCommand(QUndoCommand):
    def __init__(self, tool, row_positions, old_speakers, new_speaker, description="Bulk Edit Speaker"):
        super().__init__(description)
        self.tool = tool
        self.row_positions = row_positions
        self.old_speakers = old_speakers
        self.new_speaker = new_speaker

    def redo(self):
        for row in self.row_positions:
            self.tool.set_speaker(row, self.new_speaker)

    def undo(self):
        for row, speaker in zip(self.row_positions, self.old_speakers):
            self.tool.set_speaker(row, speaker)


class AudioLoader(QObject):
    finished = pyqtSignal(AudioSegment, np.ndarray, float)  # AudioSegment, waveform_data, duration
    error = pyqtSignal(str)  # Error message

    def __init__(self, file_path, downsample_factor=100):
        super().__init__()
        self.file_path = file_path
        self.downsample_factor = downsample_factor

    def run(self):
        try:
            # Load audio with pydub
            audio = AudioSegment.from_file(self.file_path).set_channels(1)  # Convert to mono
            samples = np.array(audio.get_array_of_samples()).astype(np.float32)
            duration = audio.duration_seconds

            # Normalize samples
            samples /= np.max(np.abs(samples)) if np.max(np.abs(samples)) != 0 else 1.0

            # Downsample the waveform data for plotting if necessary
            if len(samples) > 1000000:  # Threshold can be adjusted
                samples = self.downsample_waveform(samples, self.downsample_factor)

            # Emit the processed data
            self.finished.emit(audio, samples, duration)
        except Exception as e:
            self.error.emit(str(e))

    def downsample_waveform(self, samples, factor):
        """Downsample the waveform by taking the mean of every 'factor' samples."""
        num_blocks = len(samples) // factor
        downsampled = np.array([samples[i * factor:(i + 1) * factor].mean() for i in range(num_blocks)])
        return downsampled


class AnnotationTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Annotation Tool with Waveform and Transcript Synchronization")
        self.setGeometry(100, 100, 1600, 900)  # Increased width and height for better layout

        self.audio_segment = None  # To store the original AudioSegment
        self.waveform_data = None  # To store waveform data for plotting
        self.duration = 0.0
        self.play_obj = None
        self.current_time = 0.0
        self.is_playing = False
        self.updating = False  # Flag to prevent recursive updates
        self.speakers = []  # List to store unique speakers

        # Initialize QUndoStack
        self.undo_stack = QUndoStack(self)

        # Initialize autosave components
        self.autosave_timer = QTimer()
        self.autosave_timer.timeout.connect(self.autosave)
        self.autosave_timer.start(5000)  # Autosave every 5 seconds

        # Create a temporary file for autosave
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        self.temp_file_path = self.temp_file.name
        self.temp_file.close()  # We'll manage the file manually

        # Main layout
        self.layout = QVBoxLayout()

        # Undo/Redo buttons layout
        undo_redo_layout = QHBoxLayout()
        undo_button = QPushButton("Undo")
        undo_button.clicked.connect(self.undo_stack.undo)
        redo_button = QPushButton("Redo")
        redo_button.clicked.connect(self.undo_stack.redo)
        undo_redo_layout.addWidget(undo_button)
        undo_redo_layout.addWidget(redo_button)
        self.layout.addLayout(undo_redo_layout)

        # Waveform and Transcript layout
        waveform_transcript_layout = QHBoxLayout()

        # Waveform plot and scrollbar layout
        waveform_layout = QVBoxLayout()
        self.waveform_plot = pg.PlotWidget()
        self.waveform_plot.setYRange(-1, 1)
        self.waveform_plot.showGrid(x=True, y=False)
        self.waveform_plot.setLabel('bottom', 'Time', 's')
        waveform_layout.addWidget(self.waveform_plot)

        # Scrollbar under waveform
        self.waveform_scrollbar = QScrollBar(Qt.Horizontal)
        self.waveform_scrollbar.setMinimum(0)
        self.waveform_scrollbar.setMaximum(0)  # Will be set when audio is loaded
        self.waveform_scrollbar.valueChanged.connect(self.on_scrollbar_moved)
        waveform_layout.addWidget(self.waveform_scrollbar)

        waveform_transcript_layout.addLayout(waveform_layout)

        # Transcript (Table)
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Start", "End", "Word", "Speaker"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)  # Allow multiple selection
        self.table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.SelectedClicked)
        self.table.setStyleSheet("selection-background-color: lightblue;")
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.show_context_menu)
        waveform_transcript_layout.addWidget(self.table)

        self.layout.addLayout(waveform_transcript_layout)

        # Playtime indicator
        self.playtime_line = self.waveform_plot.addLine(x=0, pen='r')

        # Connect mouse click event for seeking
        self.waveform_plot.scene().sigMouseClicked.connect(self.on_waveform_clicked)

        # Audio controls
        audio_control_layout = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_playback)
        audio_control_layout.addWidget(self.play_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_playback)
        audio_control_layout.addWidget(self.stop_button)

        self.layout.addLayout(audio_control_layout)

        # Connect table signals
        self.table.cellDoubleClicked.connect(self.on_cell_double_clicked)
        self.table.itemChanged.connect(self.on_item_changed)
        self.table.cellClicked.connect(self.on_table_clicked)  # For click-to-seek
        self.old_values = {}  # To store old values before editing

        # New: Handle selection changes
        self.word_lines = []
        self.table.selectionModel().selectionChanged.connect(self.on_selection_changed)

        # Buttons for controls
        button_layout = QHBoxLayout()
        load_audio_button = QPushButton("Load Audio")
        load_audio_button.clicked.connect(self.load_audio_file)
        load_audio_button.setObjectName("Load Audio")  # Set object name for easy access
        button_layout.addWidget(load_audio_button)

        load_transcript_button = QPushButton("Load Transcript")
        load_transcript_button.clicked.connect(self.load_transcript)
        button_layout.addWidget(load_transcript_button)

        save_button = QPushButton("Save Annotations")
        save_button.clicked.connect(self.save_annotations)
        button_layout.addWidget(save_button)

        # New buttons
        add_below_button = QPushButton("Add Below")
        add_below_button.clicked.connect(self.add_below)
        button_layout.addWidget(add_below_button)

        delete_button = QPushButton("Delete Selected")
        delete_button.clicked.connect(self.delete_selected)
        button_layout.addWidget(delete_button)

        bulk_edit_button = QPushButton("Bulk Edit Speaker")
        bulk_edit_button.clicked.connect(self.bulk_edit_speaker)
        button_layout.addWidget(bulk_edit_button)

        self.layout.addLayout(button_layout)

        # Add rows of data
        self.sample_data = []
        self.populate_table(self.sample_data)

        # Auto-scroll timer
        self.auto_scroll_timer = QTimer()
        self.auto_scroll_timer.timeout.connect(self.highlight_current_row)
        self.auto_scroll_timer.start(500)  # Check every 500 ms

        # Main widget
        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

        # Initialize Worker attributes
        self.audio_loader_thread = None
        self.audio_loader_worker = None

        # Load autosave if exists
        self.load_autosave()

    def populate_table(self, data):
        """Populate the table with transcript data."""
        self.table.setRowCount(0)  # Clear existing rows
        self.speakers = []  # Reset speakers list

        # Collect unique speakers
        for entry in data:
            speaker = entry.get('speaker', "")
            if speaker and speaker not in self.speakers:
                self.speakers.append(speaker)

        for row_idx, entry in enumerate(data):
            self.table.insertRow(row_idx)

            # Round start and end times to two decimals
            try:
                start_time = round(float(entry['start_time']), 2)
                end_time = round(float(entry['end_time']), 2)
            except (ValueError, KeyError):
                start_time = 0.00
                end_time = 1.00

            word = entry.get('word', "")
            speaker = entry.get('speaker', "")

            start_item = QTableWidgetItem(f"{start_time:.2f}")
            end_item = QTableWidgetItem(f"{end_time:.2f}")
            word_item = QTableWidgetItem(word)

            # Align numbers to center
            start_item.setTextAlignment(Qt.AlignCenter)
            end_item.setTextAlignment(Qt.AlignCenter)
            word_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            word_item.setBackground(QColor("black"))  # Set background color
            word_item.setForeground(QColor("white"))  # Set text color
            self.table.setItem(row_idx, 0, start_item)
            self.table.setItem(row_idx, 1, end_item)
            self.table.setItem(row_idx, 2, word_item)

            # Dropdown for speaker selection
            speaker_dropdown = QComboBox()
            speaker_dropdown.addItems(self.speakers + [""])
            speaker_dropdown.setCurrentText(speaker)
            speaker_dropdown.currentTextChanged.connect(self.on_speaker_changed)
            self.table.setCellWidget(row_idx, 3, speaker_dropdown)

    def on_speaker_changed(self, new_speaker):
        """Handle changes to the speaker dropdown."""
        if new_speaker and new_speaker not in self.speakers:
            self.speakers.append(new_speaker)
            self.update_speaker_dropdowns()

    def update_speaker_dropdowns(self):
        """Update all speaker dropdowns with the current list of speakers."""
        for row in range(self.table.rowCount()):
            speaker_dropdown = self.table.cellWidget(row, 3)
            if speaker_dropdown:
                current_speaker = speaker_dropdown.currentText()
                speaker_dropdown.blockSignals(True)
                speaker_dropdown.clear()
                speaker_dropdown.addItems(self.speakers + [""])
                speaker_dropdown.setCurrentText(current_speaker)
                speaker_dropdown.blockSignals(False)

    def save_annotations(self):
        """Save current annotations to a JSON file."""
        if not self.validate_annotations():
            return

        annotations = []
        for row_idx in range(self.table.rowCount()):
            try:
                start = float(self.table.item(row_idx, 0).text())
                end = float(self.table.item(row_idx, 1).text())
                # Round to two decimals
                start = round(start, 2)
                end = round(end, 2)
            except (ValueError, AttributeError):
                QMessageBox.warning(self, "Invalid Input", f"Invalid start or end time at row {row_idx + 1}.")
                return

            word = self.table.item(row_idx, 2).text()
            speaker = self.table.cellWidget(row_idx, 3).currentText()
            annotations.append({
                "start_time": start,
                "end_time": end,
                "word": word,
                "speaker": speaker if speaker else None
            })

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Annotations", "", "JSON Files (*.json);;All Files (*)", options=options
        )
        if file_path:
            try:
                with open(file_path, "w") as file:
                    json.dump(annotations, file, indent=4)
                QMessageBox.information(self, "Success", "Annotations saved successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save annotations:\n{str(e)}")

    def load_transcript(self):
        """Load transcript from a JSON file."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Transcript", "", "JSON Files (*.json);;All Files (*)", options=options
        )
        if file_path:
            try:
                with open(file_path, "r") as file:
                    transcript = json.load(file)
                self.populate_table(transcript)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load transcript:\n{str(e)}")

    def load_audio_file(self):
        """Load an audio file."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Audio File", "",
            "Audio Files (*.wav *.mp3 *.flac *.ogg);;All Files (*)", options=options
        )
        if file_path:
            try:
                # Disable the load_audio_button to prevent multiple clicks
                load_audio_button = self.findChild(QPushButton, "Load Audio")
                if load_audio_button:
                    load_audio_button.setEnabled(False)

                # Show a loading message
                self.statusBar().showMessage("Loading audio...")

                # Initialize the worker and thread
                self.audio_loader_worker = AudioLoader(file_path)
                self.audio_loader_thread = QThread()
                self.audio_loader_worker.moveToThread(self.audio_loader_thread)

                # Connect signals
                self.audio_loader_thread.started.connect(self.audio_loader_worker.run)
                self.audio_loader_worker.finished.connect(self.on_audio_loaded)
                self.audio_loader_worker.finished.connect(self.audio_loader_thread.quit)
                self.audio_loader_worker.finished.connect(self.audio_loader_worker.deleteLater)
                self.audio_loader_thread.finished.connect(self.audio_loader_thread.deleteLater)
                self.audio_loader_worker.error.connect(self.on_audio_load_error)

                # Start the thread
                self.audio_loader_thread.start()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load audio file:\n{str(e)}")
                if load_audio_button:
                    load_audio_button.setEnabled(True)

    def on_audio_loaded(self, audio_segment, waveform_data, duration):
        """Handle the loaded audio data."""
        self.audio_segment = audio_segment  # Store the AudioSegment for playback
        self.waveform_data = waveform_data  # Store waveform data for plotting
        self.duration = duration

        # Update the waveform plot
        self.waveform_plot.clear()  # Clear previous plot
        self.waveform_plot.plot(
            np.linspace(0, self.duration, num=len(waveform_data)),
            waveform_data,
            pen="b",
        )
        self.playtime_line = self.waveform_plot.addLine(x=0, pen='r')
        self.current_time = 0.0

        # Set waveform plot limits
        self.waveform_plot.setLimits(xMin=0.0, xMax=self.duration)

        # Adjust initial view range
        self.adjust_view_range()

        # Update scrollbar maximum
        self.waveform_scrollbar.setMaximum(int(self.duration * 1000))  # Convert to milliseconds

        # Re-enable the load_audio_button
        load_audio_button = self.findChild(QPushButton, "Load Audio")
        if load_audio_button:
            load_audio_button.setEnabled(True)

        # Update status bar
        self.statusBar().showMessage("Audio loaded successfully.", 5000)  # Message disappears after 5 seconds

    def on_audio_load_error(self, error_message):
        """Handle errors during audio loading."""
        QMessageBox.critical(self, "Audio Load Error", f"Failed to load audio file:\n{error_message}")

        # Re-enable the load_audio_button
        load_audio_button = self.findChild(QPushButton, "Load Audio")
        if load_audio_button:
            load_audio_button.setEnabled(True)

        # Update status bar
        self.statusBar().showMessage("Failed to load audio.", 5000)

    def toggle_playback(self):
        """Toggle audio playback between play and pause."""
        if self.audio_segment is None:
            QMessageBox.warning(self, "No Audio", "Please load an audio file first.")
            return

        if self.is_playing:
            self.pause_playback()
        else:
            self.start_playback()

    def start_playback(self):
        """Start audio playback from the current_time."""
        if self.audio_segment is not None:
            try:
                # Slice the AudioSegment from current_time
                start_ms = int(self.current_time * 1000)  # Convert to milliseconds
                sliced_audio = self.audio_segment[start_ms:]

                # Play the sliced audio
                self.play_obj = play_audio(sliced_audio)
                self.is_playing = True
                self.play_button.setText("Pause")

                # Start a timer to update current_time
                self.playback_timer = QTimer()
                self.playback_timer.timeout.connect(self.update_current_time)
                self.playback_timer.start(100)  # Update every 100 ms
            except Exception as e:
                QMessageBox.critical(self, "Playback Error", f"Failed to play audio:\n{str(e)}")

    def pause_playback(self):
        """Pause audio playback."""
        if self.play_obj:
            self.play_obj.stop()
            self.play_obj = None
        self.is_playing = False
        self.play_button.setText("Play")

        # Stop the playback timer
        if hasattr(self, 'playback_timer') and self.playback_timer.isActive():
            self.playback_timer.stop()

    def stop_playback(self):
        """Stop audio playback and reset playback position."""
        if self.play_obj:
            self.play_obj.stop()
            self.play_obj = None
        self.is_playing = False
        self.play_button.setText("Play")
        self.playtime_line.setValue(0)
        self.current_time = 0.0

        # Stop the playback timer
        if hasattr(self, 'playback_timer') and self.playback_timer.isActive():
            self.playback_timer.stop()

    def update_current_time(self):
        """Update the current playback time."""
        self.current_time += 0.1  # Increment by 100 ms

        if self.current_time >= self.duration:
            self.stop_playback()
            return

        # Update the playtime indicator
        self.playtime_line.setValue(round(self.current_time, 2))

        # Adjust view range
        self.adjust_view_range()

        # Update scrollbar position
        self.waveform_scrollbar.setValue(int(self.current_time * 1000))

        # Highlight the current row in the transcript
        self.highlight_current_row()

    def add_below(self):
        """Add a new annotation below the selected row or at current playback time."""
        selected_rows = self.table.selectionModel().selectedRows()
        if selected_rows:
            selected_row = selected_rows[-1].row()  # Get the last selected row
            try:
                base_start = float(self.table.item(selected_row, 1).text())  # End time of selected row
            except (ValueError, AttributeError):
                base_start = 0.0
            new_start = round(base_start, 2)
            new_end = round(new_start + 1.0, 2)  # Default duration 1 second
            insert_position = selected_row + 1
        else:
            new_start = round(self.current_time, 2)
            new_end = round(new_start + 1.0, 2)
            insert_position = self.table.rowCount()

        # Ensure new_end does not exceed audio duration
        if self.audio_segment is not None and new_end > self.duration:
            new_end = round(self.duration, 2)

        row_data = {
            'start_time': new_start,
            'end_time': new_end,
            'word': "",
            'speaker': ""
        }

        command = AddRowCommand(self, insert_position, row_data)
        self.undo_stack.push(command)

    def delete_selected(self):
        """Delete the selected annotation rows."""
        selected_rows = sorted(set(index.row() for index in self.table.selectionModel().selectedRows()), reverse=True)
        if not selected_rows:
            QMessageBox.information(self, "No Selection", "Please select at least one row to delete.")
            return

        # Gather data
        rows_data = []
        for row in selected_rows:
            row_data = {
                'start_time': self.table.item(row, 0).text(),
                'end_time': self.table.item(row, 1).text(),
                'word': self.table.item(row, 2).text(),
                'speaker': self.table.cellWidget(row, 3).currentText()
            }
            rows_data.append(row_data)

        confirm = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete {len(selected_rows)} selected row(s)?",
            QMessageBox.Yes | QMessageBox.No
        )
        if confirm == QMessageBox.Yes:
            command = DeleteRowsCommand(self, rows_data, selected_rows)
            self.undo_stack.push(command)

    def bulk_edit_speaker(self):
        """Bulk edit the speaker field for selected rows."""
        selected_rows = sorted(set(index.row() for index in self.table.selectionModel().selectedRows()))
        if not selected_rows:
            QMessageBox.information(self, "No Selection", "Please select at least one row to edit.")
            return

        # Prompt user to select a speaker
        speaker, ok = QInputDialog.getItem(
            self,
            "Select Speaker",
            "Choose a speaker to assign to selected rows:",
            self.speakers + [""],
            0,
            False
        )

        if ok:
            # Gather old speakers
            old_speakers = [self.table.cellWidget(row, 3).currentText() for row in selected_rows]
            command = BulkEditSpeakerCommand(self, selected_rows, old_speakers, speaker)
            self.undo_stack.push(command)

            # Update speakers list if necessary
            if speaker and speaker not in self.speakers:
                self.speakers.append(speaker)
                self.update_speaker_dropdowns()

    def autosave(self):
        """Automatically save annotations to a temporary file."""
        annotations = []
        for row_idx in range(self.table.rowCount()):
            try:
                start = float(self.table.item(row_idx, 0).text())
                end = float(self.table.item(row_idx, 1).text())
                start = round(start, 2)
                end = round(end, 2)
            except (ValueError, AttributeError):
                continue  # Skip invalid rows

            word = self.table.item(row_idx, 2).text()
            speaker = self.table.cellWidget(row_idx, 3).currentText()

            annotations.append({
                "start_time": start,
                "end_time": end,
                "word": word,
                "speaker": speaker if speaker else None
            })

        try:
            with open(self.temp_file_path, 'w') as f:
                json.dump(annotations, f, indent=4)
            print(f"Autosaved annotations to {self.temp_file_path}")
        except Exception as e:
            # Optionally, log the error or notify the user
            print(f"Autosave failed: {e}")

    def load_autosave(self):
        """Load annotations from the autosave file if it exists."""
        if os.path.exists(self.temp_file_path):
            try:
                with open(self.temp_file_path, "r") as file:
                    annotations = json.load(file)
                self.populate_table(annotations)
                QMessageBox.information(self, "Recovery", "Recovered annotations from autosave.")
            except Exception as e:
                print(f"Failed to recover autosave: {e}")

    def closeEvent(self, event):
        """Handle the application close event."""
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
            # Delete temp file after saving
            if os.path.exists(self.temp_file_path):
                os.remove(self.temp_file_path)
            event.accept()
        elif reply == QMessageBox.No:
            # Delete temp file without saving
            if os.path.exists(self.temp_file_path):
                os.remove(self.temp_file_path)
            event.accept()
        else:
            event.ignore()

    def on_cell_double_clicked(self, row, column):
        """Store the old value before editing."""
        item = self.table.item(row, column)
        if item:
            self.old_values[(row, column)] = item.text()

    def on_item_changed(self, item):
        """Handle cell edits and push EditCellCommand to the undo stack."""
        if self.updating:
            return

        row = item.row()
        column = item.column()
        key = (row, column)

        old_value = self.old_values.get(key, "")
        new_value = item.text()

        if old_value != new_value:
            # Validate input if necessary
            if column in [0, 1]:  # Start or End times
                try:
                    float(new_value)
                except ValueError:
                    QMessageBox.warning(self, "Invalid Input", "Start and End times must be numeric.")
                    self.updating = True
                    item.setText(old_value)
                    self.updating = False
                    return

            command = EditCellCommand(self, row, column, old_value, new_value)
            self.undo_stack.push(command)

        # Remove the old value from the dict
        if key in self.old_values:
            del self.old_values[key]

    def insert_row(self, row_position, row_data):
        """Insert a row at the specified position with the provided data."""
        self.updating = True
        self.table.insertRow(row_position)

        # Round start and end times to two decimals
        try:
            start_time = round(float(row_data.get('start_time', 0.0)), 2)
            end_time = round(float(row_data.get('end_time', 1.0)), 2)
        except (ValueError, KeyError):
            start_time = 0.00
            end_time = 1.00

        word = row_data.get('word', "")
        speaker = row_data.get('speaker', "")

        if speaker and speaker not in self.speakers:
            self.speakers.append(speaker)
            self.update_speaker_dropdowns()

        start_item = QTableWidgetItem(f"{start_time:.2f}")
        end_item = QTableWidgetItem(f"{end_time:.2f}")
        word_item = QTableWidgetItem(word)

        # Align numbers to center
        start_item.setTextAlignment(Qt.AlignCenter)
        end_item.setTextAlignment(Qt.AlignCenter)
        word_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.table.setItem(row_position, 0, start_item)
        self.table.setItem(row_position, 1, end_item)
        self.table.setItem(row_position, 2, word_item)

        # Dropdown for speaker selection
        speaker_dropdown = QComboBox()
        speaker_dropdown.addItems(self.speakers + [""])
        speaker_dropdown.setCurrentText(speaker)
        speaker_dropdown.currentTextChanged.connect(self.on_speaker_changed)
        self.table.setCellWidget(row_position, 3, speaker_dropdown)

        self.updating = False

    def remove_row(self, row_position):
        """Remove the row at the specified position."""
        self.updating = True
        self.table.removeRow(row_position)
        self.updating = False

    def set_cell(self, row, column, value):
        """Set the value of a specific cell."""
        self.updating = True
        if column in [0, 1]:  # Start or End times
            try:
                float_val = float(value)
                value = f"{float_val:.2f}"
            except ValueError:
                pass  # Optionally, handle invalid input
        item = self.table.item(row, column)
        if item:
            item.setText(value)
        self.updating = False

    def set_speaker(self, row, speaker):
        """Set the speaker for a specific row."""
        self.updating = True
        speaker_dropdown = self.table.cellWidget(row, 3)
        if speaker_dropdown:
            speaker_dropdown.setCurrentText(speaker)
        self.updating = False

        if speaker and speaker not in self.speakers:
            self.speakers.append(speaker)
            self.update_speaker_dropdowns()

    def on_waveform_clicked(self, event):
        """Handle clicks on the waveform plot to seek playback."""
        if self.audio_segment is None:
            return

        pos = event.scenePos()
        if not self.waveform_plot.sceneBoundingRect().contains(pos):
            return

        mouse_point = self.waveform_plot.getPlotItem().vb.mapSceneToView(pos)
        clicked_time = mouse_point.x()

        # Clamp the clicked_time to the duration
        clicked_time = max(0.0, min(clicked_time, self.duration))

        self.current_time = clicked_time
        self.playtime_line.setValue(round(self.current_time, 2))

        # Adjust view range
        self.adjust_view_range()

        # Update scrollbar position
        self.waveform_scrollbar.setValue(int(self.current_time * 1000))

        # Highlight the corresponding row in the transcript
        self.highlight_current_row()

        if self.is_playing:
            self.pause_playback()
            self.start_playback()

    def on_table_clicked(self, row, column):
        """Handle clicks on the transcript table."""
        start_item = self.table.item(row, 0)
        if start_item:
            try:
                start_time = float(start_item.text())
                self.current_time = start_time
                self.playtime_line.setValue(round(self.current_time, 2))
                self.highlight_current_row()
                self.adjust_view_range()
                # Update scrollbar position
                self.waveform_scrollbar.setValue(int(self.current_time * 1000))
                # Playback is not started automatically
            except ValueError:
                pass

    def on_selection_changed(self, selected, deselected):
        """Handle changes in table selection."""
        # Remove existing green lines
        for line in getattr(self, 'word_lines', []):
            self.waveform_plot.removeItem(line)
        self.word_lines = []

        # For each selected row, add start and end lines
        selected_rows = sorted(set(index.row() for index in self.table.selectionModel().selectedRows()))
        if selected_rows:
            # Adjust view to show selected word(s)
            first_row = selected_rows[0]
            start_item = self.table.item(first_row, 0)
            end_item = self.table.item(first_row, 1)
            if start_item and end_item:
                try:
                    start_time = float(start_item.text())
                    end_time = float(end_item.text())
                    self.current_time = start_time
                    self.adjust_view_range(start_time, end_time)
                    self.playtime_line.setValue(round(self.current_time, 2))
                    self.waveform_scrollbar.setValue(int(self.current_time * 1000))
                except ValueError:
                    pass

        for row in selected_rows:
            start_item = self.table.item(row, 0)
            end_item = self.table.item(row, 1)
            if start_item and end_item:
                try:
                    start_time = float(start_item.text())
                    end_time = float(end_item.text())
                    # Add vertical lines at start_time and end_time
                    start_line = self.waveform_plot.addLine(x=start_time, pen=pg.mkPen('g', width=1))
                    end_line = self.waveform_plot.addLine(x=end_time, pen=pg.mkPen('g', width=1))
                    self.word_lines.extend([start_line, end_line])
                except ValueError:
                    continue

    def get_current_row(self):
        """Find the current row based on self.current_time."""
        for row in range(self.table.rowCount()):
            start_item = self.table.item(row, 0)
            end_item = self.table.item(row, 1)
            if start_item and end_item:
                try:
                    start_time = float(start_item.text())
                    end_time = float(end_item.text())
                    if start_time <= self.current_time < end_time:
                        return row
                except ValueError:
                    continue
        return -1

    def highlight_current_row(self):
        """Highlight the current row in the transcript based on playback position."""
        current_row = self.get_current_row()
        for row in range(self.table.rowCount()):
            for column in range(self.table.columnCount()):
                item = self.table.item(row, column)
                if item:
                    if row == current_row:
                        item.setBackground(QColor("blue"))
                    else:
                        item.setBackground(QColor("black"))

        if current_row != -1:
            # Scroll to the current row
            self.table.scrollToItem(self.table.item(current_row, 0), QAbstractItemView.PositionAtCenter)

    def adjust_view_range(self, start=None, end=None):
        """Adjust the waveform plot's X-axis range."""
        if start is None or end is None:
            window_size = 5.0  # 5 seconds
            half_window = window_size / 2.0
            start = max(0.0, self.current_time - half_window)
            end = min(self.duration, self.current_time + half_window)
        self.waveform_plot.setXRange(start, end, padding=0)

    def on_scrollbar_moved(self, value):
        """Handle scrollbar movement to adjust waveform view."""
        self.current_time = value / 1000.0  # Convert milliseconds to seconds
        self.playtime_line.setValue(round(self.current_time, 2))
        self.adjust_view_range()
        self.highlight_current_row()

    def show_context_menu(self, position):
        """Show context menu for the transcript table."""
        menu = QMenu()

        add_below_action = QAction("Add Below", self)
        add_below_action.triggered.connect(self.add_below)
        menu.addAction(add_below_action)

        delete_selected_action = QAction("Delete Selected", self)
        delete_selected_action.triggered.connect(self.delete_selected)
        menu.addAction(delete_selected_action)

        bulk_edit_action = QAction("Bulk Edit Speaker", self)
        bulk_edit_action.triggered.connect(self.bulk_edit_speaker)
        menu.addAction(bulk_edit_action)

        menu.exec_(self.table.viewport().mapToGlobal(position))

    def validate_annotations(self):
        """Validate that annotations do not overlap and start times are less than end times."""
        sorted_rows = sorted(range(self.table.rowCount()), key=lambda r: float(self.table.item(r, 0).text()) if self.table.item(r, 0).text() else 0.0)
        for i in range(len(sorted_rows)):
            row = sorted_rows[i]
            start_item = self.table.item(row, 0)
            end_item = self.table.item(row, 1)
            if not start_item or not end_item:
                QMessageBox.warning(self, "Invalid Annotation", f"Missing start or end time at row {row + 1}.")
                return False
            try:
                start_time = float(start_item.text())
                end_time = float(end_item.text())
                if start_time >= end_time:
                    QMessageBox.warning(self, "Invalid Annotation", f"Start time must be less than end time at row {row + 1}.")
                    return False
                if i < len(sorted_rows) - 1:
                    next_row = sorted_rows[i + 1]
                    next_start = float(self.table.item(next_row, 0).text())
                    if end_time > next_start:
                        QMessageBox.warning(
                            self, "Invalid Annotation",
                            f"Annotations at rows {row + 1} and {next_row + 1} overlap."
                        )
                        return False
            except ValueError:
                QMessageBox.warning(self, "Invalid Annotation", f"Non-numeric start or end time at row {row + 1}.")
                return False
        return True


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AnnotationTool()
    window.show()
    sys.exit(app.exec_())