import sys
import os
import json
import tempfile
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QFileDialog, QMessageBox,
    QInputDialog, QMenu, QAction, QScrollBar, Q
)
from PyQt5.QtCore import Qt, QTimer, QObject, pyqtSignal, QThread
from PyQt5.QtGui import QColor
import pyqtgraph as pg
from pydub import AudioSegment
from pydub.playback import _play_with_simpleaudio as play_audio
from PyQt5.QtWidgets import QUndoStack, QUndoCommand


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
        self.rows_data = rows_data
        self.row_positions = row_positions

    def redo(self):
        for row in sorted(self.row_positions, reverse=True):
            self.tool.remove_row(row)

    def undo(self):
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
    finished = pyqtSignal(AudioSegment, np.ndarray, float)
    error = pyqtSignal(str)

    def __init__(self, file_path, downsample_factor=100):
        super().__init__()
        self.file_path = file_path
        self.downsample_factor = downsample_factor

    def run(self):
        try:
            audio = AudioSegment.from_file(self.file_path).set_channels(1)
            samples = np.array(audio.get_array_of_samples()).astype(np.float32)
            duration = audio.duration_seconds
            samples /= np.max(np.abs(samples)) if np.max(np.abs(samples)) != 0 else 1.0
            if len(samples) > 1_000_000:
                samples = self.downsample_waveform(samples, self.downsample_factor)
            self.finished.emit(audio, samples, duration)
        except Exception as e:
            self.error.emit(str(e))

    def downsample_waveform(self, samples, factor):
        num_blocks = len(samples) // factor
        return np.array([samples[i * factor:(i + 1) * factor].mean() for i in range(num_blocks)])


class DraggableLine(pg.InfiniteLine):
    positionChangedFinished = pyqtSignal(object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMovable(True)
        self.setCursor(Qt.SizeHorCursor)
        self._old_value = self.value()

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        new_value = self.value()
        if self._old_value != new_value:
            self.positionChangedFinished.emit(self)
            self._old_value = new_value
    
    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        self.positionChangedFinished.emit(self)


class AnnotationTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PELICAn Transcription Tool")
        self.setGeometry(100, 100, 1600, 900)

        # Initialize variables
        self.audio_segment = None
        self.waveform_data = None
        self.duration = 0.0
        self.play_obj = None
        self.current_time = 0.0
        self.is_playing = False
        self.speakers = []
        self.undo_stack = QUndoStack(self)
        self.old_values = {}
        self.word_lines = []
        self.temp_file_path = tempfile.NamedTemporaryFile(delete=False, suffix='.json').name

        self.row_to_lines = {}

        # Setup UI components
        self.setup_ui()
        self.setup_signals()

        # Load autosave if exists
        self.load_autosave()

    def setup_ui(self):
        self.layout = QVBoxLayout()
        self.setup_undo_redo_buttons()
        self.setup_waveform_and_transcript()
        self.setup_audio_controls()
        self.setup_buttons()
        self.populate_table([])

        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

    def setup_signals(self):
        self.autosave_timer = QTimer()
        self.autosave_timer.timeout.connect(self.autosave)
        self.autosave_timer.start(5000)

        self.auto_scroll_timer = QTimer()
        self.auto_scroll_timer.timeout.connect(self.highlight_current_row)
        self.auto_scroll_timer.start(100)

        self.table.cellDoubleClicked.connect(self.on_cell_double_clicked)
        self.table.itemChanged.connect(self.on_item_changed)
        self.table.cellClicked.connect(self.on_table_clicked)
        self.table.currentCellChanged.connect(self.on_current_cell_changed)
        self.table.selectionModel().selectionChanged.connect(self.on_selection_changed)
        self.waveform_plot.scene().sigMouseClicked.connect(self.on_waveform_clicked)

    def setup_undo_redo_buttons(self):
        undo_redo_layout = QHBoxLayout()
        undo_button = QPushButton("Undo")
        undo_button.clicked.connect(self.undo_stack.undo)
        redo_button = QPushButton("Redo")
        redo_button.clicked.connect(self.undo_stack.redo)
        undo_redo_layout.addWidget(undo_button)
        undo_redo_layout.addWidget(redo_button)
        self.layout.addLayout(undo_redo_layout)

    def setup_waveform_and_transcript(self):
        waveform_transcript_layout = QHBoxLayout()
        waveform_layout = QVBoxLayout()
        self.waveform_plot = pg.PlotWidget()
        self.waveform_plot.setYRange(-1, 1)
        self.waveform_plot.showGrid(x=True, y=False)
        self.waveform_plot.setLabel('bottom', 'Time', 's')
        waveform_layout.addWidget(self.waveform_plot)

        self.waveform_scrollbar = QScrollBar(Qt.Horizontal)
        self.waveform_scrollbar.setMinimum(0)
        self.waveform_scrollbar.valueChanged.connect(self.on_scrollbar_moved)
        waveform_layout.addWidget(self.waveform_scrollbar)
        waveform_transcript_layout.addLayout(waveform_layout)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Start", "End", "Word", "Speaker"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.SelectedClicked)
        self.table.setStyleSheet("selection-background-color: lightblue;")
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.show_context_menu)
        self.table.setSortingEnabled(False)
        waveform_transcript_layout.addWidget(self.table)
        self.layout.addLayout(waveform_transcript_layout)
        self.playtime_line = self.waveform_plot.addLine(x=0, pen='r')

    def setup_audio_controls(self):
        audio_control_layout = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_playback)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_playback)
        self.return_button = QPushButton("Return to Current Selection (X)")
        self.return_button.clicked.connect(self.return_to_selection)
        audio_control_layout.addWidget(self.play_button)
        audio_control_layout.addWidget(self.stop_button)
        audio_control_layout.addWidget(self.return_button)
        self.layout.addLayout(audio_control_layout)

    def setup_buttons(self):
        button_layout = QHBoxLayout()
        load_audio_button = QPushButton("Load Audio")
        load_audio_button.clicked.connect(self.load_audio_file)
        load_audio_button.setObjectName("Load Audio")
        load_transcript_button = QPushButton("Load Transcript")
        load_transcript_button.clicked.connect(self.load_transcript)
        save_button = QPushButton("Save Annotations")
        save_button.clicked.connect(self.save_annotations)
        add_below_button = QPushButton("Add Below")
        add_below_button.clicked.connect(self.add_below)
        delete_button = QPushButton("Delete Selected")
        delete_button.clicked.connect(self.delete_selected)
        bulk_edit_button = QPushButton("Bulk Edit Speaker")
        bulk_edit_button.clicked.connect(self.bulk_edit_speaker)
        button_layout.addWidget(load_audio_button)
        button_layout.addWidget(load_transcript_button)
        button_layout.addWidget(save_button)
        button_layout.addWidget(add_below_button)
        button_layout.addWidget(delete_button)
        button_layout.addWidget(bulk_edit_button)
        self.layout.addLayout(button_layout)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_X:
            self.return_to_selection()
        else:
            super().keyPressEvent(event)

    def populate_table(self, data):
        self.table.setRowCount(0)
        self.speakers = []
        # Clear existing lines
        for item in self.word_lines:
            self.waveform_plot.removeItem(item['line'])
        self.word_lines = []
        self.row_to_lines = {}

        for row_idx, entry in enumerate(data):
            self.table.insertRow(row_idx)
            start_time = round(float(entry.get('start_time', 0.0)), 2)
            end_time = round(float(entry.get('end_time', 1.0)), 2)
            word = entry.get('word', "")
            speaker = entry.get('speaker', "")

            start_item = QTableWidgetItem(f"{start_time:.2f}")
            end_item = QTableWidgetItem(f"{end_time:.2f}")
            word_item = QTableWidgetItem(word)
            start_item.setTextAlignment(Qt.AlignCenter)
            end_item.setTextAlignment(Qt.AlignCenter)
            word_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            word_item.setBackground(QColor("black"))
            word_item.setForeground(QColor("white"))
            self.table.setItem(row_idx, 0, start_item)
            self.table.setItem(row_idx, 1, end_item)
            self.table.setItem(row_idx, 2, word_item)

            speaker_dropdown = QComboBox()
            speaker_dropdown.addItems(self.speakers + [""])
            speaker_dropdown.setCurrentText(speaker)
            speaker_dropdown.currentTextChanged.connect(self.on_speaker_changed)
            self.table.setCellWidget(row_idx, 3, speaker_dropdown)

    def add_word_lines(self, row, start_time, end_time):
        """Add draggable lines for the word boundaries."""
        start_line = DraggableLine(pos=start_time, angle=90, pen=pg.mkPen('g', width=1))
        end_line = DraggableLine(pos=end_time, angle=90, pen=pg.mkPen('g', width=1))
        self.waveform_plot.addItem(start_line)
        self.waveform_plot.addItem(end_line)
        self.word_lines.extend([
            {'line': start_line, 'row': row, 'column': 0},
            {'line': end_line, 'row': row, 'column': 1}
        ])
        self.row_to_lines.setdefault(row, []).extend([start_line, end_line])
        start_line.positionChangedFinished.connect(self.on_line_moved_finished)
        end_line.positionChangedFinished.connect(self.on_line_moved_finished)

    def on_speaker_changed(self, new_speaker):
        if new_speaker and new_speaker not in self.speakers:
            self.speakers.append(new_speaker)
            self.update_speaker_dropdowns()

    def update_speaker_dropdowns(self):
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
        if not self.validate_annotations():
            return

        annotations = []
        for row_idx in range(self.table.rowCount()):
            try:
                start = float(self.table.item(row_idx, 0).text())
                end = float(self.table.item(row_idx, 1).text())
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
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Audio File", "",
            "Audio Files (*.wav *.mp3 *.flac *.ogg);;All Files (*)", options=options
        )
        if file_path:
            load_audio_button = self.findChild(QPushButton, "Load Audio")
            if load_audio_button:
                load_audio_button.setEnabled(False)

            self.statusBar().showMessage("Loading audio...")
            self.audio_loader_worker = AudioLoader(file_path)
            self.audio_loader_thread = QThread()
            self.audio_loader_worker.moveToThread(self.audio_loader_thread)
            self.audio_loader_thread.started.connect(self.audio_loader_worker.run)
            self.audio_loader_worker.finished.connect(self.on_audio_loaded)
            self.audio_loader_worker.finished.connect(self.audio_loader_thread.quit)
            self.audio_loader_worker.finished.connect(self.audio_loader_worker.deleteLater)
            self.audio_loader_thread.finished.connect(self.audio_loader_thread.deleteLater)
            self.audio_loader_worker.error.connect(self.on_audio_load_error)
            self.audio_loader_thread.start()

    def on_audio_loaded(self, audio_segment, waveform_data, duration):
        self.audio_segment = audio_segment
        self.waveform_data = waveform_data
        self.duration = duration
        self.waveform_plot.clear()
        self.waveform_plot.plot(
            np.linspace(0, self.duration, num=len(waveform_data)),
            waveform_data,
            pen="b",
        )
        self.playtime_line = self.waveform_plot.addLine(x=0, pen='r')
        self.current_time = 0.0
        self.waveform_plot.setLimits(xMin=0.0, xMax=self.duration)
        self.adjust_view_range()
        self.waveform_scrollbar.setMaximum(int(self.duration * 1000))
        load_audio_button = self.findChild(QPushButton, "Load Audio")
        if load_audio_button:
            load_audio_button.setEnabled(True)
        self.statusBar().showMessage("Audio loaded successfully.", 5000)
        self.redraw_word_lines()
        
        
    def redraw_word_lines(self):
        """Redraw the word boundary lines based on current table data."""
        # Clear existing lines
        for item in self.word_lines:
            self.waveform_plot.removeItem(item['line'])
        self.word_lines = []
        self.row_to_lines = {}

        # Add lines for each word
        for row in range(self.table.rowCount()):
            start_item = self.table.item(row, 0)
            end_item = self.table.item(row, 1)
            if start_item and end_item:
                try:
                    start_time = float(start_item.text())
                    end_time = float(end_item.text())
                    self.add_word_lines(row, start_time, end_time)
                except ValueError:
                    continue


    def on_audio_load_error(self, error_message):
        QMessageBox.critical(self, "Audio Load Error", f"Failed to load audio file:\n{error_message}")
        load_audio_button = self.findChild(QPushButton, "Load Audio")
        if load_audio_button:
            load_audio_button.setEnabled(True)
        self.statusBar().showMessage("Failed to load audio.", 5000)

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
                self.playback_timer = QTimer()
                self.playback_timer.timeout.connect(self.update_current_time)
                self.playback_timer.start(100)
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

    def stop_playback(self):
        if self.play_obj:
            self.play_obj.stop()
            self.play_obj = None
        self.is_playing = False
        self.play_button.setText("Play")
        self.playtime_line.setValue(0)
        self.current_time = 0.0
        if hasattr(self, 'playback_timer') and self.playback_timer.isActive():
            self.playback_timer.stop()

    def update_current_time(self):
        self.current_time += 0.1
        if self.current_time >= self.duration:
            self.stop_playback()
            return
        self.playtime_line.setValue(round(self.current_time, 2))
        self.adjust_view_range()
        self.waveform_scrollbar.blockSignals(True)
        self.waveform_scrollbar.setValue(int(self.current_time * 1000))
        self.waveform_scrollbar.blockSignals(False)
        self.highlight_current_row()

    def add_below(self):
        selected_rows = self.get_selected_rows()
        if selected_rows:
            selected_row = selected_rows[-1]
            try:
                base_start = float(self.table.item(selected_row, 1).text())
            except (ValueError, AttributeError):
                base_start = 0.0
            new_start = round(base_start, 2)
            insert_position = selected_row + 1
        else:
            new_start = round(self.current_time, 2)
            insert_position = self.table.rowCount()

        new_end = round(new_start + 1.0, 2)
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
        selected_rows = self.get_selected_rows()
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

            # **Remove associated lines**
            for line in self.row_to_lines.get(row, []):
                self.waveform_plot.removeItem(line)
                self.word_lines = [wl for wl in self.word_lines if wl['line'] != line]
            self.row_to_lines.pop(row, None)

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
        selected_rows = self.get_selected_rows()
        if not selected_rows:
            QMessageBox.information(self, "No Selection", "Please select at least one row to edit.")
            return

        speaker, ok = QInputDialog.getItem(
            self,
            "Select Speaker",
            "Choose a speaker to assign to selected rows:",
            self.speakers + [""],
            0,
            False
        )

        if ok:
            old_speakers = [self.table.cellWidget(row, 3).currentText() for row in selected_rows]
            command = BulkEditSpeakerCommand(self, selected_rows, old_speakers, speaker)
            self.undo_stack.push(command)
            if speaker and speaker not in self.speakers:
                self.speakers.append(speaker)
                self.update_speaker_dropdowns()

    def autosave(self):
        annotations = []
        for row_idx in range(self.table.rowCount()):
            try:
                start = float(self.table.item(row_idx, 0).text())
                end = float(self.table.item(row_idx, 1).text())
                start = round(start, 2)
                end = round(end, 2)
            except (ValueError, AttributeError):
                continue

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
        except Exception as e:
            print(f"Autosave failed: {e}")

    def load_autosave(self):
        if os.path.exists(self.temp_file_path):
            try:
                with open(self.temp_file_path, "r") as file:
                    annotations = json.load(file)
                self.populate_table(annotations)
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

    def on_cell_double_clicked(self, row, column):
        item = self.table.item(row, column)
        if item:
            self.old_values[(row, column)] = item.text()

    def on_item_changed(self, item):
        if self.table.signalsBlocked():
            return

        row = item.row()
        column = item.column()
        key = (row, column)
        old_value = self.old_values.get(key, "")
        new_value = item.text()

        if old_value != new_value:
            if column in [0, 1]:
                try:
                    new_time = float(new_value)
                    self.update_line_position(row, column, new_time)
                except ValueError:
                    QMessageBox.warning(self, "Invalid Input", "Start and End times must be numeric.")
                    self.table.blockSignals(True)
                    item.setText(old_value)
                    self.table.blockSignals(False)
                    return

            command = EditCellCommand(self, row, column, old_value, new_value)
            self.undo_stack.push(command)

        if key in self.old_values:
            del self.old_values[key]

    def insert_row(self, row_position, row_data):
        self.table.blockSignals(True)
        self.table.insertRow(row_position)

        start_time = round(float(row_data.get('start_time', 0.0)), 2)
        end_time = round(float(row_data.get('end_time', 1.0)), 2)
        word = row_data.get('word', "")
        speaker = row_data.get('speaker', "")

        if speaker and speaker not in self.speakers:
            self.speakers.append(speaker)
            self.update_speaker_dropdowns()

        start_item = QTableWidgetItem(f"{start_time:.2f}")
        end_item = QTableWidgetItem(f"{end_time:.2f}")
        word_item = QTableWidgetItem(word)
        start_item.setTextAlignment(Qt.AlignCenter)
        end_item.setTextAlignment(Qt.AlignCenter)
        word_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.table.setItem(row_position, 0, start_item)
        self.table.setItem(row_position, 1, end_item)
        self.table.setItem(row_position, 2, word_item)

        speaker_dropdown = QComboBox()
        speaker_dropdown.addItems(self.speakers + [""])
        speaker_dropdown.setCurrentText(speaker)
        speaker_dropdown.currentTextChanged.connect(self.on_speaker_changed)
        self.table.setCellWidget(row_position, 3, speaker_dropdown)
        self.table.blockSignals(False)
        
        if self.audio_segment is not None:
            self.add_word_lines(row_position, start_time, end_time)

    def remove_row(self, row_position):
        # **Remove associated lines**
        for line in self.row_to_lines.get(row_position, []):
            self.waveform_plot.removeItem(line)
            self.word_lines = [wl for wl in self.word_lines if wl['line'] != line]
        self.row_to_lines.pop(row_position, None)

        self.table.blockSignals(True)
        self.table.removeRow(row_position)
        self.table.blockSignals(False)

    def set_cell(self, row, column, value):
        self.table.blockSignals(True)
        if column in [0, 1]:
            try:
                float_val = float(value)
                value = f"{float_val:.2f}"
            except ValueError:
                pass
        item = self.table.item(row, column)
        if item:
            item.setText(value)
        self.table.blockSignals(False)
        if column in [0, 1]:
            try:
                self.update_line_position(row, column, float(value))
            except ValueError:
                pass

    def set_speaker(self, row, speaker):
        self.table.blockSignals(True)
        speaker_dropdown = self.table.cellWidget(row, 3)
        if speaker_dropdown:
            speaker_dropdown.setCurrentText(speaker)
        self.table.blockSignals(False)
        if speaker and speaker not in self.speakers:
            self.speakers.append(speaker)
            self.update_speaker_dropdowns()


    def update_line_position(self, row, column, new_value):
        """Update the position of the line corresponding to the cell."""
        for item in self.word_lines:
            if item['row'] == row and item['column'] == column:
                item['line'].blockSignals(True)
                item['line'].setValue(new_value)
                item['line'].blockSignals(False)
                item['line']._old_value = new_value  # Update the old value
                break

    def on_waveform_clicked(self, event):
        if self.audio_segment is None:
            return
        pos = event.scenePos()
        if not self.waveform_plot.sceneBoundingRect().contains(pos):
            return
        mouse_point = self.waveform_plot.getPlotItem().vb.mapSceneToView(pos)
        clicked_time = mouse_point.x()
        clicked_time = max(0.0, min(clicked_time, self.duration))
        self.current_time = clicked_time
        self.playtime_line.setValue(round(self.current_time, 2))
        self.adjust_view_range()
        self.waveform_scrollbar.blockSignals(True)
        self.waveform_scrollbar.setValue(int(self.current_time * 1000))
        self.waveform_scrollbar.blockSignals(False)
        self.select_row_by_time(self.current_time)

    def select_row_by_time(self, time):
        for row in range(self.table.rowCount()):
            start_item = self.table.item(row, 0)
            end_item = self.table.item(row, 1)
            if start_item and end_item:
                try:
                    start_time = float(start_item.text())
                    end_time = float(end_item.text())
                    if start_time <= time < end_time:
                        self.table.selectRow(row)
                        break
                except ValueError:
                    continue

    def on_table_clicked(self, row, column):
        start_item = self.table.item(row, 0)
        if start_item:
            try:
                start_time = float(start_item.text())
                self.current_time = start_time
                self.playtime_line.setValue(round(self.current_time, 2))
                self.adjust_view_range()
                self.waveform_scrollbar.blockSignals(True)
                self.waveform_scrollbar.setValue(int(self.current_time * 1000))
                self.waveform_scrollbar.blockSignals(False)
            except ValueError:
                pass

    def on_current_cell_changed(self, current_row, current_column, previous_row, previous_column):
        if current_row >= 0:
            start_item = self.table.item(current_row, 0)
            if start_item:
                try:
                    start_time = float(start_item.text())
                    self.current_time = start_time
                    self.playtime_line.setValue(round(self.current_time, 2))
                    self.adjust_view_range()
                    self.waveform_scrollbar.blockSignals(True)
                    self.waveform_scrollbar.setValue(int(self.current_time * 1000))
                    self.waveform_scrollbar.blockSignals(False)
                except ValueError:
                    pass

    def on_selection_changed(self, selected, deselected):
        for item in getattr(self, 'word_lines', []):
            self.waveform_plot.removeItem(item['line'])
        self.word_lines = []
        selected_rows = self.get_selected_rows()

        for row in selected_rows:
            start_item = self.table.item(row, 0)
            end_item = self.table.item(row, 1)
            if start_item and end_item:
                try:
                    start_time = float(start_item.text())
                    end_time = float(end_item.text())
                    start_line = DraggableLine(pos=start_time, angle=90, pen=pg.mkPen('g', width=1))
                    end_line = DraggableLine(pos=end_time, angle=90, pen=pg.mkPen('g', width=1))
                    self.waveform_plot.addItem(start_line)
                    self.waveform_plot.addItem(end_line)
                    self.word_lines.extend([
                        {'line': start_line, 'row': row, 'column': 0},
                        {'line': end_line, 'row': row, 'column': 1}
                    ])
                    start_line.positionChangedFinished.connect(self.on_line_moved_finished)
                    end_line.positionChangedFinished.connect(self.on_line_moved_finished)
                except ValueError:
                    continue

    def on_line_moved_finished(self, line):
        """Handle the line movement and update the table."""
        for item in self.word_lines:
            if item['line'] == line:
                row = item['row']
                column = item['column']
                old_value = self.table.item(row, column).text()
                new_value = line.value()
                new_value = max(0.0, min(new_value, self.duration))
                new_value_str = f"{new_value:.2f}"

                if old_value != new_value_str:
                    # Update the table item
                    self.table.blockSignals(True)
                    self.table.item(row, column).setText(new_value_str)
                    self.table.blockSignals(False)
                    # Create an undo command
                    command = EditCellCommand(self, row, column, old_value, new_value_str)
                    self.undo_stack.push(command)
                line._old_value = new_value
                break

    def highlight_current_row(self):
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
            self.table.scrollToItem(self.table.item(current_row, 0), QAbstractItemView.PositionAtCenter)

    def get_current_row(self):
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

    def adjust_view_range(self):
        window_size = 5.0
        half_window = window_size / 2.0
        start = max(0.0, self.current_time - half_window)
        end = min(self.duration, self.current_time + half_window)
        self.waveform_plot.setXRange(start, end, padding=0)

    def on_scrollbar_moved(self, value):
        self.current_time = value / 1000.0
        self.playtime_line.setValue(round(self.current_time, 2))
        self.adjust_view_range()
        self.highlight_current_row()

    def return_to_selection(self):
        selected_rows = self.get_selected_rows()
        if selected_rows:
            first_row = selected_rows[0]
            start_item = self.table.item(first_row, 0)
            end_item = self.table.item(selected_rows[-1], 1)
            if start_item and end_item:
                try:
                    start_time = float(start_item.text())
                    self.current_time = start_time
                    self.playtime_line.setValue(round(self.current_time, 2))
                    self.adjust_view_range()
                    self.waveform_scrollbar.blockSignals(True)
                    self.waveform_scrollbar.setValue(int(self.current_time * 1000))
                    self.waveform_scrollbar.blockSignals(False)
                except ValueError:
                    pass

    def show_context_menu(self, position):
        menu = QMenu()
        add_below_action = QAction("Add Below", self)
        add_below_action.triggered.connect(self.add_below)
        delete_selected_action = QAction("Delete Selected", self)
        delete_selected_action.triggered.connect(self.delete_selected)
        bulk_edit_action = QAction("Bulk Edit Speaker", self)
        bulk_edit_action.triggered.connect(self.bulk_edit_speaker)
        menu.addAction(add_below_action)
        menu.addAction(delete_selected_action)
        menu.addAction(bulk_edit_action)
        menu.exec_(self.table.viewport().mapToGlobal(position))

    def validate_annotations(self):
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

    def get_selected_rows(self):
        return sorted(set(index.row() for index in self.table.selectionModel().selectedRows()))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AnnotationTool()
    window.show()
    sys.exit(app.exec_())