import sys
import os
import json
import tempfile
import numpy as np
import librosa
import time

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QFileDialog, QMessageBox,
    QInputDialog, QMenu, QAction, QAbstractItemView, QSplitter, QUndoStack, QUndoCommand, QScrollBar
)
from PyQt5.QtCore import Qt, QTimer, QObject, pyqtSignal, QThread
from PyQt5.QtGui import QColor

import pyqtgraph as pg
from pydub import AudioSegment
from pydub.playback import _play_with_simpleaudio as play_audio

# --- Undo/Redo Command Classes ---

class AddRowCommand(QUndoCommand):
    def __init__(self, main_window, row_position, row_data, description="Add Row"):
        super().__init__(description)
        self.main_window = main_window
        self.row_position = row_position
        self.row_data = row_data

    def redo(self):
        self.main_window.insert_row(self.row_position, self.row_data)

    def undo(self):
        self.main_window.remove_row(self.row_position)


class DeleteRowsCommand(QUndoCommand):
    def __init__(self, main_window, rows_data, row_positions, description="Delete Rows"):
        super().__init__(description)
        self.main_window = main_window
        self.rows_data = rows_data
        self.row_positions = row_positions

    def redo(self):
        for row in sorted(self.row_positions, reverse=True):
            self.main_window.remove_row(row)

    def undo(self):
        for row, data in sorted(zip(self.row_positions, self.rows_data)):
            self.main_window.insert_row(row, data)


class EditCellCommand(QUndoCommand):
    def __init__(self, main_window, row, column, old_value, new_value, description="Edit Cell"):
        super().__init__(description)
        self.main_window = main_window
        self.row = row
        self.column = column
        self.old_value = old_value
        self.new_value = new_value

    def redo(self):
        self.main_window.set_cell(self.row, self.column, self.new_value)

    def undo(self):
        self.main_window.set_cell(self.row, self.column, self.old_value)


class BulkEditSpeakerCommand(QUndoCommand):
    def __init__(self, main_window, row_positions, old_speakers, new_speaker, description="Bulk Edit Speaker"):
        super().__init__(description)
        self.main_window = main_window
        self.row_positions = row_positions
        self.old_speakers = old_speakers
        self.new_speaker = new_speaker

    def redo(self):
        for row in self.row_positions:
            self.main_window.set_speaker(row, self.new_speaker)

    def undo(self):
        for row, speaker in zip(self.row_positions, self.old_speakers):
            self.main_window.set_speaker(row, speaker)


# --- Audio Loader for Asynchronous Loading ---

class AudioLoader(QObject):
    finished = pyqtSignal(np.ndarray, int)
    error = pyqtSignal(str)

    def __init__(self, file_path, downsample_factor=100):
        super().__init__()
        self.file_path = file_path
        self.downsample_factor = downsample_factor

    def run(self):
        try:
            # Load audio using librosa for consistent sampling rate
            y, sr = librosa.load(self.file_path, sr=None, mono=True)
            duration = librosa.get_duration(y=y, sr=sr)
            samples = y

            # Normalize samples
            max_abs_sample = np.max(np.abs(samples))
            samples = samples / max_abs_sample if max_abs_sample != 0 else samples

            # Downsample if necessary
            if len(samples) > 1_000_000:
                samples = self.downsample_waveform(samples, self.downsample_factor)

            self.finished.emit(samples, sr)
        except Exception as e:
            self.error.emit(str(e))

    def downsample_waveform(self, samples, factor):
        num_blocks = len(samples) // factor
        return np.array([samples[i * factor:(i + 1) * factor].mean() for i in range(num_blocks)])


# --- Draggable Line Class for pyqtgraph ---

class DraggableLine(pg.InfiniteLine):
    def __init__(self, pos, color, idx, boundary_type, pen=None, movable=True):
        pen = pen or pg.mkPen(color=color, width=2)
        super().__init__(pos=pos, angle=90, pen=pen, movable=movable)
        self.idx = idx
        self.boundary_type = boundary_type
        self.setHoverPen(pen.color().lighter())
        self.setCursor(Qt.SizeHorCursor)


# --- WaveformCanvas Class Using pyqtgraph ---

class WaveformCanvas(QWidget):
    boundary_changed = pyqtSignal(int, str, float)  # idx, 'start'/'end', new position
    waveform_clicked = pyqtSignal(float)
    audio_loaded = pyqtSignal()
    loading_error = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setYRange(-1, 1)
        self.plot_widget.showGrid(x=True, y=False)
        self.plot_widget.setLabel('bottom', 'Time', 's')
        self.layout.addWidget(self.plot_widget)

        self.scrollbar = QScrollBar(Qt.Horizontal)
        self.layout.addWidget(self.scrollbar)
        self.scrollbar.valueChanged.connect(self.on_scrollbar_value_changed)

        self.plot_widget.plotItem.vb.sigXRangeChanged.connect(self.on_x_range_changed)

        self.words = []
        self.lines = []
        self.connecting_lines = []

        self.dragging_line = None
        
        self.utterances = []
        self.utterance_items = []
        self.utterance_regions = []

        self.plot_widget.scene().sigMouseClicked.connect(self.on_waveform_click)

        self.audio_data = None
        self.sr = None
        self.duration = None
        self.window_size = 5.0

    def load_audio(self, file_path):
        self.thread = QThread()
        self.loader = AudioLoader(file_path)
        self.loader.moveToThread(self.thread)
        self.thread.started.connect(self.loader.run)
        self.loader.finished.connect(self.on_audio_loaded)
        self.loader.finished.connect(self.thread.quit)
        self.loader.finished.connect(self.loader.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.loader.error.connect(self.on_loading_error)
        self.thread.start()

    def on_audio_loaded(self, samples, sr):
        self.audio_data = samples
        self.sr = sr
        effective_sr = sr /  self.loader.downsample_factor
        self.duration = len(samples) / effective_sr
        t = np.linspace(0, self.duration, num=len(samples))

        self.plot_widget.clear()
        self.plot_widget.plot(t, samples, pen='b')
        self.playtime_line = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('y', width=4))
        self.playtime_line.setZValue(1000)
        self.plot_widget.addItem(self.playtime_line)
        self.plot_widget.setLimits(xMin=0, xMax=self.duration)
        self.plot_widget.setXRange(0, min(self.window_size, self.duration))
        self.plot_widget.setLabel('bottom', 'Time', 's')
        self.draw_lines()
        self.boundary_changed.emit(-1, '', 0.0)  # Reset
        print("Audio loaded successfully.")
        print(f"Duration: {self.duration}, Window Size: {self.window_size}")
        self.scrollbar.setMinimum(0)
        self.scrollbar.setMaximum(int(self.duration * 1000))
        self.scrollbar.setSingleStep(100)
        self.scrollbar.setPageStep(int(self.window_size * 1000))
        self.scrollbar.setValue(0)

        self.audio_loaded.emit()
        
    def load_utterances(self, utterances):
        self.utterances = utterances
        self.draw_utterances()
        
    def draw_utterances(self):
        self.clear_utterance_items()
        self.utterance_items = []
        self.utterance_regions = []

        speaker_colors = {
            "SPEAKER_00": QColor(255, 200, 200, 100),  # Light red
            "SPEAKER_01": QColor(200, 255, 200, 100),  # Light green
            "SPEAKER_02": QColor(200, 200, 255, 100),  # Light blue
            "UNKNOWN": QColor(200, 200, 200, 100),
            "": QColor(200, 200, 200, 100),  # Light gray
        }

        for idx, utterance in enumerate(self.utterances):
            start = float(utterance['start_time'])
            end = float(utterance['end_time'])
            speaker = utterance.get('speaker', '')
            confidence = utterance.get('confidence', '')
            color = speaker_colors.get(speaker, QColor(200, 200, 200, 100))

            # Add background region for utterance
            region = pg.LinearRegionItem(values=[start+0.01, end-0.01], brush=color)
            region.setMovable(False)
            self.plot_widget.addItem(region)
            self.utterance_regions.append(region)

            # Break utterance text into words
            text = utterance.get('text', '')
            words = text.strip().split()

            num_words = len(words)
            if num_words == 0:
                continue

            # Add a label for utterance metadata (e.g., speaker, index, duration)
            label_text = f"Utterance: {idx + 1}, Speaker: {speaker}, Speaker Confidence: {confidence}, Duration: {round(end - start, 2)}s"
            meta_label = pg.TextItem(label_text, anchor=(0.5, 0), color='yellow')
            meta_label.setPos((start + end) / 2, -0.25)  # Centered above the utterance
            self.plot_widget.addItem(meta_label)
            self.utterance_items.append(meta_label)

            word_times = np.linspace(start, end, num_words+2)

            for word, word_time in zip(words, word_times[1:-1]):
                # Plot the word as a text label at y=-0.5
                label = pg.TextItem(word, anchor=(0.5, 0), color='white')
                label.setPos(word_time, -0.5)
                self.plot_widget.addItem(label)
                self.utterance_items.append(label)
            
    def clear_utterance_items(self):
        if hasattr(self, 'utterance_items'):
            for item in self.utterance_items:
                self.plot_widget.removeItem(item)
            self.utterance_items = []
        if hasattr(self, 'utterance_regions'):
            for region in self.utterance_regions:
                self.plot_widget.removeItem(region)
            self.utterance_regions = []

    def on_loading_error(self, error_message):
        self.loading_error.emit(error_message)

    def load_words(self, words):
        self.words = words
        self.draw_lines()

    def clear_lines(self):
        for line in self.lines:
            self.plot_widget.removeItem(line['line'])
        self.lines = []
        for cline in self.connecting_lines:
            self.plot_widget.removeItem(cline)
        self.connecting_lines = []

    def draw_lines(self):
        self.clear_lines()
        for idx, word in enumerate(self.words):
            
            if idx % 2 == 0:
                y_pos_line = 0.55
            else:
                y_pos_line = 0.45
            
            ### I adjust the line positions slightly because otherwise start and endlines of consecutive words ###
            start_line = DraggableLine(pos=(word['start']+0.005), color='green', idx=idx, boundary_type='start')
            end_line = DraggableLine(pos=(word['end']-0.005), color='red', idx=idx, boundary_type='end')
            self.plot_widget.addItem(start_line)
            self.plot_widget.addItem(end_line)
            self.lines.append({'line': start_line, 'idx': idx, 'type': 'start'})
            self.lines.append({'line': end_line, 'idx': idx, 'type': 'end'})

            # Connecting line at y=0.5
            connecting_line = pg.PlotCurveItem(
                [word['start'] +0.005, word['end']-0.005],
                [y_pos_line, y_pos_line],  # Position the line at y=0.5
                pen=pg.mkPen('blue', width=2),
            )
            self.plot_widget.addItem(connecting_line)

            # Create arrowheads
            start_arrow = self.create_arrow(word['start'] + 0.005, y_pos_line, 0)
            end_arrow = self.create_arrow(word['end'] - 0.005, y_pos_line, 180)

            # Create label
            label = pg.TextItem(word['word'], anchor=(0.5, 0), color='white')
            self.plot_widget.addItem(label)

            # Store all items in the connecting_lines list
            self.connecting_lines.append({
                "line": connecting_line,
                "start_arrow": start_arrow,
                "end_arrow": end_arrow,
                "label": label,
            })

            # Position label initially
            self.update_connecting_line(idx)

            # Connect signals to update arrows and labels
            start_line.sigPositionChangeFinished.connect(lambda _, line=start_line: self.on_line_moved(line))
            end_line.sigPositionChangeFinished.connect(lambda _, line=end_line: self.on_line_moved(line))

        self.plot_widget.update()
    
    def create_arrow(self, x, y, angle):
        
        arrow = pg.ArrowItem(
            pos=(x, y),
            angle=angle,  # Direction of the arrow in degrees
            tipAngle=30,
            baseAngle=20,
            headLen=15,
            brush='blue',
        )
        self.plot_widget.addItem(arrow)
        return arrow

    def update_connecting_line(self, idx):
        word = self.words[idx]
        start = word['start']
        end = word['end']
        
        if idx % 2 == 0:
            y_pos_line = 0.55
        else:
            y_pos_line = 0.45

        # Update the connecting line's x-coordinates and keep y fixed at 0.5
        self.connecting_lines[idx]['line'].setData([start+ 0.005, end- 0.005], [y_pos_line, y_pos_line])

        # Update arrowhead positions
        self.connecting_lines[idx]['start_arrow'].setPos(start+ 0.005, y_pos_line)
        self.connecting_lines[idx]['end_arrow'].setPos(end- 0.005, y_pos_line)

        # Update label position (middle of the line, slightly above)
        mid_x = (start + end) / 2
        
        if word["speaker"] in ["", "UNKOWN"]:
            mid_y = 0.7  # Slightly above y=0.5
        elif word["speaker"] == "SPEAKER_00":
            mid_y = 0.4
        elif word["speaker"] == "SPEAKER_01":
            mid_y = 0.6
        else:
            mid_y = 0.0
        
        self.connecting_lines[idx]['label'].setPos(mid_x, mid_y)

    def on_line_moved(self, line):
        idx = line.idx
        boundary_type = line.boundary_type
        new_pos = line.value()
        new_pos = max(0.0, min(new_pos, self.duration))
        self.words[idx][boundary_type] = new_pos
        self.boundary_changed.emit(idx, boundary_type, new_pos)
        self.update_connecting_line(idx)

    def on_waveform_click(self, event):
        pos = event.scenePos()
        if not self.plot_widget.sceneBoundingRect().contains(pos):
            return
        mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
        clicked_time = mouse_point.x()
        clicked_time = max(0.0, min(clicked_time, self.duration))
        self.waveform_clicked.emit(clicked_time)

    def update_playtime_line(self, current_time):
        self.playtime_line.setPos(current_time)
        # Adjust view range
        self.adjust_view_range(current_time)

    def add_word_lines(self, row, start_time, end_time):
        """Add draggable lines for a new word."""
        start_line = DraggableLine(pos=start_time, color='green', idx=row, boundary_type='start')
        end_line = DraggableLine(pos=end_time, color='red', idx=row, boundary_type='end')
        self.plot_widget.addItem(start_line)
        self.plot_widget.addItem(end_line)
        self.lines.append({'line': start_line, 'idx': row, 'type': 'start'})
        self.lines.append({'line': end_line, 'idx': row, 'type': 'end'})

        # Connecting line with arrows
        connecting_line = pg.ArrowItem(
            pos=((start_time + end_time) / 2, 0),
            angle=0,
            tipAngle=30,
            baseAngle=20,
            headLen=15,
            tailLen=0,
            tailWidth=0,
            brush='blue'
        )
        connecting_line.setParentItem(self.plot_widget.plotItem)
        self.connecting_lines.append(connecting_line)

        start_line.sigPositionChangeFinished.connect(lambda _, line=start_line: self.on_line_moved(line))
        end_line.sigPositionChangeFinished.connect(lambda _, line=end_line: self.on_line_moved(line))

    def adjust_view_range(self, current_time, window_size=None):
        if window_size is None:
            window_size = self.window_size
        half_window = window_size / 2.0
        start = max(0.0, current_time - half_window)
        end = min(self.duration, current_time + half_window)
        self.plot_widget.setXRange(start, end, padding=0)
        self.scrollbar.blockSignals(True)
        self.scrollbar.setValue(int(start * 1000))
        self.scrollbar.blockSignals(False)

    def update_line_position(self, idx, boundary_type, new_pos):
        # Find the line and update its position
        for line_info in self.lines:
            if line_info['idx'] == idx and line_info['type'] == boundary_type:
                line_info['line'].setValue(new_pos)
                break
        self.words[idx][boundary_type] = new_pos
        self.update_connecting_line(idx)

    def on_scrollbar_value_changed(self, value):
        start = min(value / 1000.0, self.duration - self.window_size)
        end = min(start + self.window_size, self.duration)
        self.plot_widget.setXRange(start, end, padding=0)

    def on_x_range_changed(self, view_box, range):
        start, end = max(0, range[0]), min(self.duration, range[1])
        self.scrollbar.blockSignals(True)
        self.scrollbar.setValue(int(start * 1000))
        self.scrollbar.blockSignals(False)


# --- MainWindow Class ---

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
        self.old_values = {}
        self.temp_file_path = tempfile.NamedTemporaryFile(delete=False, suffix='.json').name
        self.previous_current_row = None  # Initialize previous_current_row
        # Setup UI components
        self.setup_ui()
        self.setup_signals()
        
        # Start the autosave timer
        self.autosave_timer = QTimer(self)
        self.autosave_timer.timeout.connect(self.autosave)
        self.autosave_timer.start(5000)  # Trigger autosave every 5 seconds

        # Load autosave if exists
        self.load_autosave()

    def setup_ui(self):
        splitter = QSplitter(Qt.Horizontal)

        # Left panel: Waveform
        self.waveform_widget = QWidget()
        waveform_layout = QVBoxLayout(self.waveform_widget)
        self.canvas = WaveformCanvas(parent=self.waveform_widget)
        waveform_layout.addWidget(self.canvas)

        # Playback Controls
        playback_layout = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_playback)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_playback)
        self.return_button = QPushButton("Return to Selection (X)")
        self.return_button.clicked.connect(self.return_to_selection)
        playback_layout.addWidget(self.play_button)
        playback_layout.addWidget(self.stop_button)
        playback_layout.addWidget(self.return_button)
        waveform_layout.addLayout(playback_layout)

        # Load and Save Buttons
        buttons_layout = QHBoxLayout()
        load_audio_button = QPushButton("Load Audio")
        load_audio_button.clicked.connect(self.load_audio)
        load_transcript_button = QPushButton("Load Transcript")
        load_transcript_button.clicked.connect(self.load_transcript)
        save_button = QPushButton("Save Annotations")
        save_button.clicked.connect(self.save_annotations)
        buttons_layout.addWidget(load_audio_button)
        buttons_layout.addWidget(load_transcript_button)
        buttons_layout.addWidget(save_button)
        waveform_layout.addLayout(buttons_layout)

        # Undo/Redo Buttons
        undo_redo_layout = QHBoxLayout()
        undo_button = QPushButton("Undo")
        undo_button.clicked.connect(self.undo_stack.undo)
        redo_button = QPushButton("Redo")
        redo_button.clicked.connect(self.undo_stack.redo)
        undo_redo_layout.addWidget(undo_button)
        undo_redo_layout.addWidget(redo_button)
        waveform_layout.addLayout(undo_redo_layout)

        splitter.addWidget(self.waveform_widget)

        # Right panel: Transcript Table
        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(4)
        self.table_widget.setHorizontalHeaderLabels(["Word", "Start Time", "End Time", "Speaker"])
        self.table_widget.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table_widget.customContextMenuRequested.connect(self.show_context_menu)
        splitter.addWidget(self.table_widget)

        self.setCentralWidget(splitter)

    def setup_signals(self):
        self.canvas.boundary_changed.connect(self.on_boundary_changed)
        self.canvas.waveform_clicked.connect(self.on_waveform_clicked)
        self.canvas.audio_loaded.connect(self.on_audio_loaded)
        self.canvas.loading_error.connect(self.on_audio_load_error)
        self.table_widget.itemChanged.connect(self.on_item_changed)
        self.table_widget.cellDoubleClicked.connect(self.on_cell_double_clicked)
        self.table_widget.currentCellChanged.connect(self.on_current_cell_changed)
        self.table_widget.selectionModel().selectionChanged.connect(self.on_selection_changed)

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
                with open(file_path, 'r', encoding='utf-8') as f:
                    transcript = json.load(f)
                    words = transcript.get("combined_data", [])
                    speaker_segments = transcript.get("speaker_segments", [])
                    utterances = transcript.get("utterance_data", [])
                    self.canvas.load_utterances(utterances)
                # Validate and process words
                for word in words:
                    if 'word' not in word or 'start_time' not in word or 'end_time' not in word:
                        raise ValueError("Invalid transcript format. Each entry must have 'word', 'start_time', and 'end_time'.")
                    word['start'] = float(word['start_time'])
                    word['end'] = float(word['end_time'])
                    if 'speaker' not in word:
                        word['speaker'] = ''
                    else:
                        speaker = word['speaker']
                        if speaker and speaker not in self.speakers:
                            self.speakers.append(speaker)
                self.canvas.load_words(words)
                self.update_table()
                print(f"Loaded transcript file: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error Loading Transcript", f"Failed to load transcript:\n{str(e)}")

    def update_table(self):
        self.table_widget.blockSignals(True)
        words = self.canvas.words
        self.table_widget.setRowCount(len(words))
        for i, word in enumerate(words):
            word_item = QTableWidgetItem(word['word'])
            start_item = QTableWidgetItem(f"{word['start']:.2f}")
            end_item = QTableWidgetItem(f"{word['end']:.2f}")
            speaker_dropdown = QComboBox()
            speaker_dropdown.addItems(self.speakers + [""])
            speaker_dropdown.setCurrentText(word.get('speaker', ''))
            speaker_dropdown.currentTextChanged.connect(self.on_speaker_changed)
            self.table_widget.setItem(i, 0, word_item)
            self.table_widget.setItem(i, 1, start_item)
            self.table_widget.setItem(i, 2, end_item)
            self.table_widget.setCellWidget(i, 3, speaker_dropdown)
            # Set default cell colors
            for j in range(3):
                item = self.table_widget.item(i, j)
                if item:
                    item.setBackground(QColor("black"))
                    item.setForeground(QColor("white"))
        self.table_widget.blockSignals(False)
        self.statusBar().showMessage("Transcript loaded successfully.", 5000)

    def on_boundary_changed(self, idx, boundary_type, new_pos):
        if idx == -1:
            return
        self.table_widget.blockSignals(True)
        if boundary_type == 'start':
            item = self.table_widget.item(idx, 1)
            if item is not None:
                item.setText(f"{new_pos:.2f}")
        elif boundary_type == 'end':
            item = self.table_widget.item(idx, 2)
            if item is not None:
                item.setText(f"{new_pos:.2f}")
        self.table_widget.blockSignals(False)
        self.autosave()

    def on_waveform_clicked(self, time):
        self.current_time = time
        self.canvas.update_playtime_line(self.current_time)
        self.highlight_current_row()

    def on_item_changed(self, item):
        if self.table_widget.signalsBlocked():
            return
        row = item.row()
        column = item.column()
        key = (row, column)
        old_value = self.old_values.get(key, "")
        new_value = item.text()

        if old_value != new_value:
            if column in [1, 2]:
                try:
                    new_time = float(new_value)
                    boundary_type = 'start' if column == 1 else 'end'
                    self.canvas.update_line_position(row, boundary_type, new_time)
                except ValueError:
                    QMessageBox.warning(self, "Invalid Input", "Start and End times must be numeric.")
                    self.table_widget.blockSignals(True)
                    item.setText(old_value)
                    self.table_widget.blockSignals(False)
                    return

            command = EditCellCommand(self, row, column, old_value, new_value)
            self.undo_stack.push(command)
            self.autosave()

        if key in self.old_values:
            del self.old_values[key]

    def on_cell_double_clicked(self, row, column):
        item = self.table_widget.item(row, column)
        if item:
            self.old_values[(row, column)] = item.text()

    def on_speaker_changed(self, new_speaker):
        sender = self.sender()
        if new_speaker and new_speaker not in self.speakers:
            self.speakers.append(new_speaker)
            self.update_speaker_dropdowns()

    def update_speaker_dropdowns(self):
        for row in range(self.table_widget.rowCount()):
            speaker_dropdown = self.table_widget.cellWidget(row, 3)
            if speaker_dropdown:
                current_speaker = speaker_dropdown.currentText()
                speaker_dropdown.blockSignals(True)
                speaker_dropdown.clear()
                speaker_dropdown.addItems(self.speakers + [""])
                speaker_dropdown.setCurrentText(current_speaker)
                speaker_dropdown.blockSignals(False)

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
        self.highlight_current_row()
        if hasattr(self, 'playback_timer') and self.playback_timer.isActive():
            self.playback_timer.stop()

    def update_current_time(self):
        elapsed_time = time.time() - self.playback_start_time
        self.current_time = self.playback_start_position + elapsed_time
        if self.current_time >= self.canvas.duration:
            self.stop_playback()
            return
        self.canvas.update_playtime_line(self.current_time)
        self.highlight_current_row()

    def highlight_current_row(self):
        current_row = self.get_current_row()
        selected_rows = self.get_selected_rows()
        
        if current_row == self.previous_current_row:
            return  # No change, so no need to update
        
        # Reset previous current row background
        if self.previous_current_row is not None:
            for column in range(3):
                item = self.table_widget.item(self.previous_current_row, column)
                if item:
                    if self.previous_current_row in selected_rows:
                        item.setBackground(QColor("blue"))
                        item.setForeground(QColor("white"))
                    else:
                        item.setBackground(QColor("black"))
                        item.setForeground(QColor("white"))
        
        # Set new current row background
        if current_row != -1:
            for column in range(3):
                item = self.table_widget.item(current_row, column)
                if item:
                    item.setBackground(QColor("yellow"))
                    item.setForeground(QColor("black"))
            self.table_widget.scrollToItem(self.table_widget.item(current_row, 0), QAbstractItemView.PositionAtCenter)
        
        self.previous_current_row = current_row
    
    
    def get_current_row(self):
        for row in range(self.table_widget.rowCount()):
            try:
                start_time = float(self.table_widget.item(row, 1).text())
                end_time = float(self.table_widget.item(row, 2).text())
                if start_time <= self.current_time < end_time:
                    return row
            except (ValueError, AttributeError):
                continue
        return -1

    def return_to_selection(self):
        selected_rows = self.get_selected_rows()
        if selected_rows:
            first_row = selected_rows[0]
            try:
                start_time = float(self.table_widget.item(first_row, 1).text())
                self.current_time = start_time
                self.canvas.update_playtime_line(self.current_time)
                self.highlight_current_row()
            except (ValueError, AttributeError):
                pass

    def get_selected_rows(self):
        return sorted(set(index.row() for index in self.table_widget.selectionModel().selectedRows()))

    def show_context_menu(self, position):
        menu = QMenu()
        add_row_action = QAction("Add Row", self)
        add_row_action.triggered.connect(self.add_row)
        delete_row_action = QAction("Delete Selected Rows", self)
        delete_row_action.triggered.connect(self.delete_selected_rows)
        bulk_edit_action = QAction("Bulk Edit Speaker", self)
        bulk_edit_action.triggered.connect(self.bulk_edit_speaker)
        menu.addAction(add_row_action)
        menu.addAction(delete_row_action)
        menu.addAction(bulk_edit_action)
        menu.exec_(self.table_widget.viewport().mapToGlobal(position))

    def add_row(self):
        row_count = self.table_widget.rowCount()
        row_data = {'word': '', 'start_time': 0.0, 'end_time': 0.0, 'speaker': ''}
        command = AddRowCommand(self, row_count, row_data)
        self.undo_stack.push(command)
        self.autosave()

    def delete_selected_rows(self):
        selected_rows = self.get_selected_rows()
        if not selected_rows:
            QMessageBox.information(self, "No Selection", "Please select at least one row to delete.")
            return

        # Gather data
        rows_data = []
        for row in selected_rows:
            row_data = {
                'word': self.table_widget.item(row, 0).text(),
                'start_time': self.table_widget.item(row, 1).text(),
                'end_time': self.table_widget.item(row, 2).text(),
                'speaker': self.table_widget.cellWidget(row, 3).currentText()
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
            self.autosave()

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
            old_speakers = [self.table_widget.cellWidget(row, 3).currentText() for row in selected_rows]
            command = BulkEditSpeakerCommand(self, selected_rows, old_speakers, speaker)
            self.undo_stack.push(command)
            if speaker and speaker not in self.speakers:
                self.speakers.append(speaker)
                self.update_speaker_dropdowns()
            self.autosave()

    def autosave(self):
        words = self.canvas.words
        data_to_save = []
        for word in words:
            data_to_save.append({
                'word': word['word'],
                'start_time': word['start'],
                'end_time': word['end'],
                'speaker': word.get('speaker', '')
            })
        try:
            with open(self.temp_file_path, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=4)
            print(f"Autosaved annotations to {self.temp_file_path}")
        except Exception as e:
            print(f"Autosave failed: {e}")

    def load_autosave(self):
        if os.path.exists(self.temp_file_path):
            try:
                with open(self.temp_file_path, "r", encoding='utf-8') as file:
                    annotations = json.load(file)
                self.canvas.load_words(annotations)
                self.update_table()
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

        annotations = []
        for row_idx in range(self.table_widget.rowCount()):
            try:
                start = float(self.table_widget.item(row_idx, 1).text())
                end = float(self.table_widget.item(row_idx, 2).text())
                start = round(start, 2)
                end = round(end, 2)
            except (ValueError, AttributeError):
                QMessageBox.warning(self, "Invalid Input", f"Invalid start or end time at row {row_idx + 1}.")
                return

            word = self.table_widget.item(row_idx, 0).text()
            speaker = self.table_widget.cellWidget(row_idx, 3).currentText()
            annotations.append({
                "word": word,
                "start_time": start,
                "end_time": end,
                "speaker": speaker if speaker else None
            })

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
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(annotations, f, indent=4)
                QMessageBox.information(self, "Save Successful", f"Annotations saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save annotations:\n{str(e)}")

    def on_current_cell_changed(self, current_row, current_column, previous_row, previous_column):
        if current_row >= 0:
            try:
                start_time = float(self.table_widget.item(current_row, 1).text())
                self.current_time = start_time
                self.canvas.update_playtime_line(self.current_time)
                self.highlight_current_row()
            except (ValueError, AttributeError):
                pass

    def on_selection_changed(self, selected, deselected):
        self.previous_current_row = None
        self.highlight_current_row()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_X:
            self.return_to_selection()
        else:
            super().keyPressEvent(event)

    def insert_row(self, row_position, row_data):
        self.table_widget.blockSignals(True)
        self.table_widget.insertRow(row_position)
        word_item = QTableWidgetItem(row_data['word'])
        start_item = QTableWidgetItem(f"{float(row_data['start_time']):.2f}")
        end_item = QTableWidgetItem(f"{float(row_data['end_time']):.2f}")
        speaker_dropdown = QComboBox()
        speaker_dropdown.addItems(self.speakers + [""])
        speaker_dropdown.setCurrentText(row_data.get('speaker', ''))
        speaker_dropdown.currentTextChanged.connect(self.on_speaker_changed)
        self.table_widget.setItem(row_position, 0, word_item)
        self.table_widget.setItem(row_position, 1, start_item)
        self.table_widget.setItem(row_position, 2, end_item)
        self.table_widget.setCellWidget(row_position, 3, speaker_dropdown)
        # Set default cell colors
        for j in range(3):
            item = self.table_widget.item(row_position, j)
            if item:
                item.setBackground(QColor("black"))
                item.setForeground(QColor("white"))
        self.table_widget.blockSignals(False)
        # Update waveform
        self.canvas.words.insert(row_position, {
            'word': row_data['word'],
            'start': float(row_data['start_time']),
            'end': float(row_data['end_time']),
            'speaker': row_data.get('speaker', '')
        })
        self.canvas.draw_lines()

    def remove_row(self, row_position):
        self.table_widget.blockSignals(True)
        self.table_widget.removeRow(row_position)
        self.table_widget.blockSignals(False)
        # Update waveform
        del self.canvas.words[row_position]
        self.canvas.draw_lines()

    def set_cell(self, row, column, value):
        self.table_widget.blockSignals(True)
        item = self.table_widget.item(row, column)
        if item:
            item.setText(value)
        self.table_widget.blockSignals(False)
        if column in [1, 2]:
            try:
                new_time = float(value)
                boundary_type = 'start' if column == 1 else 'end'
                self.canvas.update_line_position(row, boundary_type, new_time)
                self.autosave()
            except ValueError:
                pass

    def set_speaker(self, row, speaker):
        self.table_widget.blockSignals(True)
        speaker_dropdown = self.table_widget.cellWidget(row, 3)
        if speaker_dropdown:
            speaker_dropdown.setCurrentText(speaker)
        self.table_widget.blockSignals(False)
        if speaker and speaker not in self.speakers:
            self.speakers.append(speaker)
            self.update_speaker_dropdowns()
            self.autosave()

    def validate_annotations(self):
        sorted_rows = sorted(range(self.table_widget.rowCount()), key=lambda r: float(self.table_widget.item(r, 1).text()) if self.table_widget.item(r, 1).text() else 0.0)
        for i in range(len(sorted_rows)):
            row = sorted_rows[i]
            start_item = self.table_widget.item(row, 1)
            end_item = self.table_widget.item(row, 2)
            if not start_item or not end_item:
                QMessageBox.warning(self, "Invalid Annotation", f"Missing start or end time at row {row + 1}.")
                return False
            try:
                start_time = float(start_item.text())
                end_time = float(end_item.text())
                if start_time > end_time:
                    QMessageBox.warning(self, "Invalid Annotation", f"Start time must be less than end time at row {row + 1}.")
                    return False
                if i < len(sorted_rows) - 1:
                    next_row = sorted_rows[i + 1]
                    next_start = float(self.table_widget.item(next_row, 1).text())
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

# --- Main Execution ---

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()