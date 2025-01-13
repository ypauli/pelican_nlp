import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QScrollBar, QMenu, QInputDialog
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QColor, QCursor

from .draggable_line import DraggableLine
from .audio_loader import AudioLoader

class WaveformCanvas(QWidget):
    """A widget for displaying and interacting with waveform plots using pyqtgraph."""
    
    boundary_changed = pyqtSignal(int, str, float, float)  # idx, 'start'/'end', new position, old position
    waveform_clicked = pyqtSignal(float)
    word_double_clicked = pyqtSignal(float)
    word_right_clicked = pyqtSignal(float)
    audio_loaded = pyqtSignal()
    loading_error = pyqtSignal(str)

    class CustomViewBox(pg.ViewBox):
        """Custom ViewBox for handling mouse events and fixing Y-axis behavior."""

        def __init__(self, canvas, y_limits=(-1, 1), *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.canvas = canvas
            self.y_limits = y_limits
            if self.canvas is None:
                raise ValueError("CustomViewBox requires a valid 'canvas' reference.")

        def mouseClickEvent(self, event):
            if event.button() == Qt.RightButton:
                self.raiseContextMenu(event)
                event.accept()
            else:
                super().mouseClickEvent(event)

        def mouseDoubleClickEvent(self, event):
            mouse_point = self.mapSceneToView(event.pos())
            clicked_time = mouse_point.x()
            if hasattr(self.canvas, "on_waveform_double_clicked"):
                self.canvas.on_waveform_double_clicked(clicked_time)
            event.accept()

        def raiseContextMenu(self, event):
            mouse_point = self.mapSceneToView(event.scenePos())
            clicked_time = mouse_point.x()
            if hasattr(self.canvas, "on_waveform_right_clicked"):
                self.canvas.on_waveform_right_clicked(clicked_time)
            event.accept()

        def scaleBy(self, s=None, center=None):
            if s is not None:
                s = [s[0], 1]  # Only scale X-axis
            super().scaleBy(s, center)

        def mouseDragEvent(self, ev, axis=None):
            if axis is None or axis == 1:  # Y-axis
                ev.ignore()
            else:
                super().mouseDragEvent(ev, axis=axis)

        def updateLimits(self):
            self.setRange(yRange=self.y_limits, padding=0, update=False)

    def __init__(self, parent=None, main_window=None):
        super().__init__(parent)
        self.main_window = main_window
        self.layout = QVBoxLayout(self)

        # Initialize PlotWidget with CustomViewBox
        self.plot_widget = pg.PlotWidget(viewBox=self.CustomViewBox(canvas=self, y_limits=(-1, 1)))
        self.plot_widget.setYRange(-1, 1)
        self.plot_widget.showGrid(x=True, y=False)
        self.plot_widget.setLabel('bottom', 'Time', 's')
        
        # Connect mouse events - only single clicks through scene
        self.plot_widget.scene().sigMouseClicked.connect(self.handle_mouse_click)
        
        self.layout.addWidget(self.plot_widget)

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

    def handle_mouse_click(self, event):
        if event.button() == Qt.LeftButton:
            pos = self.plot_widget.plotItem.vb.mapSceneToView(event.scenePos())
            self.main_window.on_waveform_clicked(pos.x())

    def on_waveform_double_clicked(self, time):
        """Called by CustomViewBox when double-clicked"""
        self.main_window.on_word_double_clicked(time)

    def on_waveform_right_clicked(self, time):
        """Called by CustomViewBox when right-clicked"""
        self.main_window.on_word_right_clicked(time)

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
            region = pg.LinearRegionItem(values=[start + 0.005, end - 0.005], brush=color, span=(0.1, 0.4))
            region.setMovable(False)
            self.plot_widget.addItem(region)
            self.utterance_regions.append(region)

            # Add labels
            label_text = f"Utterance: {idx + 1}, Speaker: {speaker}, Confidence: {confidence}, Duration: {round(end - start, 2)}s"
            meta_label = pg.TextItem(label_text, anchor=(0.5, 0), color='yellow')
            meta_label.setPos((start + end) / 2, -0.95)
            self.plot_widget.addItem(meta_label)
            self.utterance_items.append(meta_label)

            text = utterance.get('text', '')
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
            self.plot_widget.removeItem(segment)
        self.word_segments = []

    def draw_lines(self):
        self.clear_lines()

        for idx, word in enumerate(self.words):
            start = float(word['start_time']) + 0.005
            end = float(word['end_time']) - 0.005

            # Create draggable boundary lines
            start_line = DraggableLine(pos=start, color='green', idx=idx, boundary_type='start', span=(0.6, 0.9))
            end_line = DraggableLine(pos=end, color='red', idx=idx, boundary_type='end', span=(0.6, 0.9))
            self.plot_widget.addItem(start_line)
            self.plot_widget.addItem(end_line)
            self.lines.append({'line': start_line, 'idx': idx, 'type': 'start'})
            self.lines.append({'line': end_line, 'idx': idx, 'type': 'end'})

            # Create connecting line and arrows
            y_pos_line = 0.55 if (idx % 2 == 0) else 0.45
            connecting_line = pg.PlotCurveItem(
                [start, end],
                [y_pos_line, y_pos_line],
                pen=pg.mkPen('blue', width=2),
            )
            self.plot_widget.addItem(connecting_line)
            start_arrow = self.create_arrow(start, y_pos_line, 0)
            end_arrow = self.create_arrow(end, y_pos_line, 180)

            # Create word label
            word_label = pg.TextItem(word['word'], anchor=(0.5, 0), color='white')
            word_label.setPos((start + end) / 2, 0.6)
            word_label.mouseClickEvent = lambda ev, idx=idx: self.main_window.on_word_clicked(idx)
            self.plot_widget.addItem(word_label)

            self.connecting_lines.append({
                "line": connecting_line,
                "start_arrow": start_arrow,
                "end_arrow": end_arrow,
                "label": word_label,
            })

            # Create speaker background segment
            speaker = word.get('speaker', '')
            color = self.speaker_colors.get(speaker, QColor(200, 200, 200, 100))
            word_segment = pg.LinearRegionItem(values=[start, end], brush=color, span=(0.6, 0.9))
            word_segment.setMovable(False)
            self.plot_widget.addItem(word_segment)
            self.word_segments.append(word_segment)

            # Connect drag signals
            start_line.sigPositionChangeFinished.connect(lambda _, line=start_line: self.on_line_moved(line))
            end_line.sigPositionChangeFinished.connect(lambda _, line=end_line: self.on_line_moved(line))

        self.plot_widget.update()

    def create_arrow(self, x, y, angle):
        arrow = pg.ArrowItem(
            pos=(x, y),
            angle=angle,
            tipAngle=30,
            baseAngle=20,
            headLen=15,
            brush='blue',
        )
        self.plot_widget.addItem(arrow)
        return arrow

    def update_connecting_line(self, idx):
        word = self.words[idx]
        start = float(word['start_time']) + 0.005
        end = float(word['end_time']) - 0.005
        y_pos_line = 0.55 if (idx % 2 == 0) else 0.45

        self.connecting_lines[idx]['line'].setData([start, end], [y_pos_line, y_pos_line])
        self.connecting_lines[idx]['start_arrow'].setPos(start, y_pos_line)
        self.connecting_lines[idx]['end_arrow'].setPos(end, y_pos_line)

        mid_x = (start + end) / 2
        self.connecting_lines[idx]['label'].setPos(mid_x, 0.6)
        self.connecting_lines[idx]['label'].setText(word['word'])

    def update_word_segment(self, idx):
        word = self.words[idx]
        start_time = float(word['start_time'])
        end_time = float(word['end_time'])
        speaker = word.get('speaker', '')

        segment = self.word_segments[idx]
        segment.setRegion([start_time, end_time])
        color = self.speaker_colors.get(speaker, QColor(200, 200, 200, 100))
        segment.setBrush(color)

    def on_line_moved(self, line):
        idx = line.idx
        boundary_type = line.boundary_type
        new_pos = max(0.0, min(line.value(), self.duration))
        old_pos = line.old_pos
        line.old_pos = new_pos

        if boundary_type == 'start':
            self.words[idx]['start_time'] = new_pos
        else:
            self.words[idx]['end_time'] = new_pos

        self.boundary_changed.emit(idx, boundary_type, new_pos, old_pos)
        self.update_connecting_line(idx)
        self.update_word_segment(idx)

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
        for line_info in self.lines:
            if line_info['idx'] == idx and line_info['type'] == boundary_type:
                line_info['line'].setValue(new_pos)
                line_info['line'].old_pos = new_pos
                break
        if boundary_type == 'start':
            self.words[idx]['start_time'] = new_pos
        elif boundary_type == 'end':
            self.words[idx]['end_time'] = new_pos
        self.update_connecting_line(idx) 