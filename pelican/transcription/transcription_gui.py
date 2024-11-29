import sys
import os
import json
import tempfile
import numpy as np
import librosa
import time
import re

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QFileDialog, QMessageBox,
    QInputDialog, QMenu, QAction, QUndoStack, QUndoCommand, QScrollBar, QLineEdit
)
from PyQt5.QtCore import Qt, QTimer, QObject, pyqtSignal, QThread
from PyQt5.QtGui import QColor, QCursor

import pyqtgraph as pg
from pydub import AudioSegment
from pydub.playback import _play_with_simpleaudio as play_audio

# Assume Transcript class is imported from another file
from transcription import Transcript  # Replace 'transcription' with the actual module name


# --- Undo/Redo Command Classes ---

class EditWordCommand(QUndoCommand):
    def __init__(self, main_window, idx, old_word, new_word, description="Edit Word"):
        super().__init__(description)
        self.main_window = main_window
        self.idx = idx
        self.old_word = old_word
        self.new_word = new_word

    def redo(self):
        self.main_window.transcript.combined_data[self.idx]['word'] = self.new_word
        self.main_window.canvas.words[self.idx]['word'] = self.new_word
        self.main_window.canvas.update_connecting_line(self.idx)

    def undo(self):
        self.main_window.transcript.combined_data[self.idx]['word'] = self.old_word
        self.main_window.canvas.words[self.idx]['word'] = self.old_word
        self.main_window.canvas.update_connecting_line(self.idx)


class EditSpeakerCommand(QUndoCommand):
    def __init__(self, main_window, idx, old_speaker, new_speaker, description="Edit Speaker"):
        super().__init__(description)
        self.main_window = main_window
        self.idx = idx
        self.old_speaker = old_speaker
        self.new_speaker = new_speaker

    def redo(self):
        self.main_window.transcript.combined_data[self.idx]['speaker'] = self.new_speaker
        self.main_window.canvas.words[self.idx]['speaker'] = self.new_speaker
        self.main_window.canvas.update_connecting_line(self.idx)

    def undo(self):
        self.main_window.transcript.combined_data[self.idx]['speaker'] = self.old_speaker
        self.main_window.canvas.words[self.idx]['speaker'] = self.old_speaker
        self.main_window.canvas.update_connecting_line(self.idx)


class MoveBoundaryCommand(QUndoCommand):
    def __init__(self, main_window, idx, boundary_type, old_pos, new_pos, description="Move Boundary"):
        super().__init__(description)
        self.main_window = main_window
        self.idx = idx
        self.boundary_type = boundary_type
        self.old_pos = old_pos
        self.new_pos = new_pos

    def redo(self):
        self.main_window.transcript.combined_data[self.idx][f'{self.boundary_type}_time'] = self.new_pos
        self.main_window.canvas.update_line_position(self.idx, self.boundary_type, self.new_pos)

    def undo(self):
        self.main_window.transcript.combined_data[self.idx][f'{self.boundary_type}_time'] = self.old_pos
        self.main_window.canvas.update_line_position(self.idx, self.boundary_type, self.old_pos)


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
    def __init__(self, pos, color, idx, boundary_type, span = (0,1), pen=None, movable=True):
        pen = pen or pg.mkPen(color=color, width=2)
        super().__init__(pos=pos, angle=90, pen=pen, movable=movable)
        self.idx = idx
        self.boundary_type = boundary_type
        self.setSpan(span[0], span[1])
        self.setHoverPen(pen.color().lighter())
        self.setCursor(Qt.SizeHorCursor)
        self.old_pos = pos

class WaveformCanvas(QWidget):
    """
    A widget for displaying and interacting with waveform plots using pyqtgraph.
    """
    boundary_changed = pyqtSignal(int, str, float, float)  # idx, 'start'/'end', new position, old position
    waveform_clicked = pyqtSignal(float)
    word_double_clicked = pyqtSignal(float)
    word_right_clicked = pyqtSignal(float)
    audio_loaded = pyqtSignal()
    loading_error = pyqtSignal(str)

    class CustomViewBox(pg.ViewBox):
        """
        Custom ViewBox for handling mouse events and fixing Y-axis behavior.
        """

        def __init__(self, canvas, y_limits=(-1, 1), *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.canvas = canvas
            self.y_limits = y_limits
            if self.canvas is None:
                raise ValueError("CustomViewBox requires a valid 'canvas' reference.")

        def mouseClickEvent(self, event):
            """
            Handles mouse click events. Triggers the context menu for right-clicks.
            """
            if event.button() == Qt.RightButton:
                self.raiseContextMenu(event)
                event.accept()
            else:
                super().mouseClickEvent(event)

        def mouseDoubleClickEvent(self, event):
            """
            Handles mouse double-click events. Notifies the canvas of the double-click.
            """
            mouse_point = self.mapSceneToView(event.pos())
            clicked_time = mouse_point.x()
            if hasattr(self.canvas, "on_waveform_double_clicked"):
                self.canvas.on_waveform_double_clicked(clicked_time)
            event.accept()

        def raiseContextMenu(self, event):
            """
            Displays the context menu for right-click events.
            """
            mouse_point = self.mapSceneToView(event.scenePos())
            clicked_time = mouse_point.x()
            if hasattr(self.canvas, "on_waveform_right_clicked"):
                self.canvas.on_waveform_right_clicked(clicked_time)
            event.accept()

        def scaleBy(self, s=None, center=None):
            """
            Restricts scaling to the X-axis only.
            """
            if s is not None:
                s = [s[0], 1]  # Only scale X-axis
            super().scaleBy(s, center)

        def mouseDragEvent(self, ev, axis=None):
            """
            Disables vertical dragging by restricting movements on the Y-axis.
            """
            if axis is None or axis == 1:  # Y-axis
                ev.ignore()
            else:
                super().mouseDragEvent(ev, axis=axis)

        def updateLimits(self):
            """
            Ensures the view remains within the fixed Y-axis range.
            """
            self.setRange(yRange=self.y_limits, padding=0, update=False)
            
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent)
        self.main_window = main_window  # Store the reference to MainWindow
        self.layout = QVBoxLayout(self)

        # Initialize PlotWidget with CustomViewBox
        self.plot_widget = pg.PlotWidget(viewBox=self.CustomViewBox(canvas=self, y_limits=(-1, 1)))
        self.plot_widget.setYRange(-1, 1)
        self.plot_widget.showGrid(x=True, y=False)
        self.plot_widget.setLabel('bottom', 'Time', 's')
        self.layout.addWidget(self.plot_widget)

        # self.plot_widget.plotItem.vb.setLimits(yMin=-1.05, yMax=1.05)
        self.editing_line = None
        
        
        # Add horizontal scrollbar
        self.scrollbar = QScrollBar(Qt.Horizontal)
        self.layout.addWidget(self.scrollbar)
        self.scrollbar.valueChanged.connect(self.on_scrollbar_value_changed)

        # Connect plot's X range change to update scrollbar
        self.plot_widget.plotItem.vb.sigXRangeChanged.connect(self.on_x_range_changed)

        self.words = []
        self.lines = []
        self.connecting_lines = []
        self.word_segments = []

        self.dragging_line = None

        self.utterances = []
        self.utterance_items = []
        self.utterance_regions = []

        self.audio_data = None
        self.sr = None
        self.duration = None
        self.window_size = 5.0  # Default window size of 5 seconds
        
        self.speaker_colors = {
            "SPEAKER_00": QColor(255, 200, 200, 100),  # Light red
            "SPEAKER_01": QColor(200, 255, 200, 100),  # Light green
            "SPEAKER_02": QColor(200, 200, 255, 100),  # Light blue
            "UNKNOWN": QColor(200, 200, 200, 100),
            "": QColor(200, 200, 200, 100),  # Light gray
        }

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
        effective_sr = sr / self.loader.downsample_factor
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
        self.boundary_changed.emit(-1, '', 0.0, 0.0)  # Reset

        # Configure scrollbar
        self.scrollbar.setMinimum(0)
        self.scrollbar.setMaximum(int(self.duration * 1000))
        self.scrollbar.setSingleStep(100)  # 100 ms steps
        self.scrollbar.setPageStep(int(self.window_size * 1000))  # 5-second page steps
        self.scrollbar.setValue(0)

        self.audio_loaded.emit()

    def load_utterances(self, utterances):
        self.utterances = utterances
        self.draw_utterances()

    def draw_utterances(self):
        self.clear_utterance_items()
        self.utterance_items = []
        self.utterance_regions = []

        for idx, utterance in enumerate(self.utterances):
            start = float(utterance['start_time'])
            end = float(utterance['end_time'])
            speaker = utterance.get('speaker', '')
            confidence = utterance.get('confidence', '')
            color = self.speaker_colors.get(speaker, QColor(200, 200, 200, 100))

            # Add background region for utterance
            region = pg.LinearRegionItem(values=[start + 0.005, end - 0.005], brush=color, span = (0.1, 0.4))
            region.setMovable(False)
            self.plot_widget.addItem(region)
            self.utterance_regions.append(region)

            # Break utterance text into words
            text = utterance.get('text', '')

            # Add a label for utterance metadata (e.g., speaker, index, duration)
            label_text = f"Utterance: {idx + 1}, Speaker: {speaker}, Confidence: {confidence}, Duration: {round(end - start, 2)}s"
            meta_label = pg.TextItem(label_text, anchor=(0.5, 0), color='yellow')
            meta_label.setPos((start + end) / 2, -0.95)  # Centered above the utterance
            self.plot_widget.addItem(meta_label)
            self.utterance_items.append(meta_label)

            label = pg.TextItem(text, anchor=(0.5, 0), color='white')
            label.setPos((start + end) / 2, -0.5)
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
            self.plot_widget.removeItem(cline['line'])
            self.plot_widget.removeItem(cline['start_arrow'])
            self.plot_widget.removeItem(cline['end_arrow'])
            self.plot_widget.removeItem(cline['label'])        
        self.connecting_lines = []
        for segment in self.word_segments:
            self.plot_widget.removeItem(segment['segment'])
        self.word_segments = []

    def draw_lines(self):
        self.clear_lines()
        for idx, word in enumerate(self.words):

            if idx % 2 == 0:
                y_pos_line = 0.55
            else:
                y_pos_line = 0.45

            start = (float(word['start_time']) + 0.005)
            end = (float(word['end_time']) - 0.005)
            # Adjust the line positions slightly
            start_line = DraggableLine(pos=start, color='green', idx=idx, boundary_type='start', span = (0.6,0.9))
            end_line = DraggableLine(pos=end, color='red', idx=idx, boundary_type='end', span = (0.6,0.9))
            self.plot_widget.addItem(start_line)
            self.plot_widget.addItem(end_line)
            self.lines.append({'line': start_line, 'idx': idx, 'type': 'start'})
            self.lines.append({'line': end_line, 'idx': idx, 'type': 'end'})

            # Connecting line at y=0.5
            connecting_line = pg.PlotCurveItem(
                [start, end],
                [y_pos_line, y_pos_line],  # Position the line at y=0.5
                pen=pg.mkPen('blue', width=2),
            )
            self.plot_widget.addItem(connecting_line)

            # Create arrowheads
            start_arrow = self.create_arrow(start, y_pos_line, 0)
            end_arrow = self.create_arrow(end, y_pos_line, 180)

            # Create label
            label = pg.TextItem(word['word'], anchor=(0.5, 0), color='white')
            label.setPos((float(word['start_time']) + float(word['end_time'])) / 2, 0.6)  # Adjust label position
            label.mouseClickEvent = lambda ev, idx=idx: self.main_window.on_word_clicked(idx)  # Use self.main_window
            self.plot_widget.addItem(label)

            # Store all items in the connecting_lines list
            self.connecting_lines.append({
                "line": connecting_line,
                "start_arrow": start_arrow,
                "end_arrow": end_arrow,
                "label": label,
            })
            
            color = self.speaker_colors.get(word["speaker"], QColor(200, 200, 200, 100))

            # Add background region for word segment
            word_segment = pg.LinearRegionItem(values=[start, end], brush=color, span = (0.6, 0.9))
            word_segment.setMovable(False)
            self.plot_widget.addItem(word_segment)
            self.word_segments.append(word_segment)

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
        start = float(word['start_time'])+ 0.005
        end = float(word['end_time']) - 0.005

        if idx % 2 == 0:
            y_pos_line = 0.55
        else:
            y_pos_line = 0.45

        # Update the connecting line's x-coordinates and keep y fixed at 0.5
        self.connecting_lines[idx]['line'].setData([start , end ], [y_pos_line, y_pos_line])

        # Update arrowhead positions
        self.connecting_lines[idx]['start_arrow'].setPos(start, y_pos_line)
        self.connecting_lines[idx]['end_arrow'].setPos(end, y_pos_line)

        # Update label position (middle of the line, slightly above)
        mid_x = (start + end) / 2

        self.connecting_lines[idx]['label'].setPos(mid_x, 0.5)
        self.connecting_lines[idx]['label'].setText(word['word'])

    def update_word_segment(self, idx):
        
        word = self.words[idx]
        start_time = float(word['start_time'])
        end_time = float(word['end_time'])
        speaker = word.get('speaker', '')

        # Update the LinearRegionItem for the word segment
        segment = self.word_segments[idx]
        segment.setRegion([start_time, end_time])

        # Update the color based on the speaker
        color = self.speaker_colors.get(speaker, QColor(200, 200, 200, 100))
        segment.setBrush(color)
        

    def on_line_moved(self, line):
        idx = line.idx
        boundary_type = line.boundary_type
        new_pos = line.value()
        new_pos = max(0.0, min(new_pos, self.duration))
        old_pos = line.old_pos
        line.old_pos = new_pos  # Update old_pos for next time
        # Update word data
        if boundary_type == 'start':
            self.words[idx]['start_time'] = new_pos
        elif boundary_type == 'end':
            self.words[idx]['end_time'] = new_pos
        self.boundary_changed.emit(idx, boundary_type, new_pos, old_pos)
        self.update_connecting_line(idx)
        self.update_word_segment(idx)

    def on_waveform_double_clicked(self, clicked_time):
        self.word_double_clicked.emit(clicked_time)

    def on_waveform_right_clicked(self, clicked_time):
        self.word_right_clicked.emit(clicked_time)

    def on_scrollbar_value_changed(self, value):
        start = min(value / 1000.0, self.duration - self.window_size)
        end = min(start + self.window_size, self.duration)
        self.plot_widget.setXRange(start, end, padding=0)

    def on_x_range_changed(self, view_box, range):
        start, end = max(0, range[0]), min(self.duration, range[1])
        self.scrollbar.blockSignals(True)
        self.scrollbar.setValue(int(start * 1000))
        self.scrollbar.blockSignals(False)

    def update_playtime_line(self, current_time):
        self.playtime_line.setPos(current_time)
        # Adjust view range and scrollbar
        self.adjust_view_range(current_time)

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
                line_info['line'].old_pos = new_pos
                break
        # Update word data
        if boundary_type == 'start':
            self.words[idx]['start_time'] = new_pos
        elif boundary_type == 'end':
            self.words[idx]['end_time'] = new_pos
        self.update_connecting_line(idx)


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
        self.transcript = None  # Initialize the Transcript object

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
        waveform_layout = QVBoxLayout(self.waveform_widget)
        self.canvas = WaveformCanvas(parent=self.waveform_widget, main_window=self)  # Pass self here
        waveform_layout.addWidget(self.canvas)
        self.setCentralWidget(self.waveform_widget)

        # Playback Controls
        playback_layout = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_playback)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_playback)
        playback_layout.addWidget(self.play_button)
        playback_layout.addWidget(self.stop_button)
        waveform_layout.addLayout(playback_layout)

        # Load and Save Buttons
        buttons_layout = QHBoxLayout()
        load_audio_button = QPushButton("Load Audio")
        load_audio_button.clicked.connect(self.load_audio)
        load_transcript_button = QPushButton("Load Transcript")
        load_transcript_button.clicked.connect(self.load_transcript)
        save_button = QPushButton("Save Annotations")
        save_button.clicked.connect(self.save_annotations)
        recalc_utterances_button = QPushButton("Recalculate Utterances")
        recalc_utterances_button.clicked.connect(self.recalculate_utterances)
        buttons_layout.addWidget(load_audio_button)
        buttons_layout.addWidget(load_transcript_button)
        buttons_layout.addWidget(save_button)
        buttons_layout.addWidget(recalc_utterances_button)
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
                # Extract unique speakers
                self.speakers = list(set(
                    word.get('speaker', '') for word in self.transcript.combined_data if word.get('speaker', '')
                ))
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
            new_word, ok = QInputDialog.getText(self, "Edit Word", "New word:", text=word['word'])
            if ok and new_word != word['word']:
                old_value = word['word']
                command = EditWordCommand(self, idx, old_value, new_word)
                self.undo_stack.push(command)
                self.autosave()
        else:
            QMessageBox.information(self, "No Word", "No word found at this position.")

    def on_word_right_clicked(self, time):
        idx, word = self.find_word_at_time(time)
        if word is not None:
            menu = QMenu(self)
            speakers = self.speakers + [""]  # Add empty speaker option
            for speaker in speakers:
                display_text = speaker if speaker else "(No Speaker)"
                action = QAction(display_text, self)
                action.triggered.connect(lambda checked, s=speaker: self.set_word_speaker(idx, s))
                menu.addAction(action)
            menu.exec_(QCursor.pos())
        else:
            QMessageBox.information(self, "No Word", "No word found at this position.")

    def set_word_speaker(self, idx, speaker):
        word = self.transcript.combined_data[idx]
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
                # Extract unique speakers
                self.speakers = list(set(
                    word.get('speaker', '') for word in self.transcript.combined_data if word.get('speaker', '')
                ))
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


# --- Main Execution ---

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()