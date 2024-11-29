import sys
import numpy as np

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QSplitter,
    QWidget,
    QVBoxLayout,
    QTableWidget,
    QTableWidgetItem,
)
from PyQt5.QtCore import Qt, pyqtSignal

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class WaveformCanvas(FigureCanvas):
    boundary_changed = pyqtSignal(int, str, float)  # index of word, 'start'/'end', new position

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 4))
        super(WaveformCanvas, self).__init__(self.fig)
        self.setParent(parent)

        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Waveform with Word Boundaries")

        # Generate a simple sine wave as the waveform
        t = np.linspace(0, 10, 1000)
        self.y = np.sin(2 * np.pi * 1 * t)
        self.ax.plot(t, self.y)

        # Example words with start and end times
        self.words = [
            {'word': 'Word1', 'start': 1.0, 'end': 2.0},
            {'word': 'Word2', 'start': 2.5, 'end': 4.0},
            {'word': 'Word3', 'start': 4.5, 'end': 6.0},
            {'word': 'Word4', 'start': 6.5, 'end': 8.0},
        ]

        # Draw lines for start and end times
        self.lines = []  # List of dicts: {'line': line_object, 'word_idx': idx, 'type': 'start'/'end'}

        for idx, word in enumerate(self.words):
            # Start time line
            start_line = self.ax.axvline(word['start'], color='green', linestyle='--', picker=5)
            self.lines.append({'line': start_line, 'word_idx': idx, 'type': 'start'})

            # End time line
            end_line = self.ax.axvline(word['end'], color='red', linestyle='--', picker=5)
            self.lines.append({'line': end_line, 'word_idx': idx, 'type': 'end'})

        self.dragging_line = None
        self.prev_x = None

        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        for line_dict in self.lines:
            line = line_dict['line']
            contains, _ = line.contains(event)
            if contains:
                self.dragging_line = line_dict
                self.prev_x = event.xdata
                break

    def on_motion(self, event):
        if self.dragging_line is None or event.inaxes != self.ax:
            return
        dx = event.xdata - self.prev_x
        x = self.dragging_line['line'].get_xdata()[0] + dx
        self.dragging_line['line'].set_xdata([x, x])
        self.prev_x = event.xdata
        self.fig.canvas.draw_idle()

    def on_release(self, event):
        if self.dragging_line is not None:
            # Update the boundary positions
            idx = self.dragging_line['word_idx']
            boundary_type = self.dragging_line['type']
            new_pos = self.dragging_line['line'].get_xdata()[0]
            self.words[idx][boundary_type] = new_pos
            self.dragging_line = None
            self.prev_x = None
            # Emit signal to update table
            self.boundary_changed.emit(idx, boundary_type, new_pos)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Waveform Editor")
        self.resize(800, 600)

        splitter = QSplitter(Qt.Horizontal)

        # Left panel: Waveform
        self.waveform_widget = QWidget()
        waveform_layout = QVBoxLayout(self.waveform_widget)
        self.canvas = WaveformCanvas(parent=self.waveform_widget)
        waveform_layout.addWidget(self.canvas)
        splitter.addWidget(self.waveform_widget)

        # Right panel: Table
        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(3)
        self.table_widget.setHorizontalHeaderLabels(["Word", "Start Time", "End Time"])
        splitter.addWidget(self.table_widget)

        self.setCentralWidget(splitter)

        # Connect signals
        self.canvas.boundary_changed.connect(self.on_boundary_changed)
        self.table_widget.itemChanged.connect(self.on_table_item_changed)

        self.update_table()

    def update_table(self):
        self.table_widget.blockSignals(True)  # Prevent signals while updating
        words = self.canvas.words
        self.table_widget.setRowCount(len(words))
        for i, word in enumerate(words):
            self.table_widget.setItem(i, 0, QTableWidgetItem(word['word']))
            self.table_widget.setItem(i, 1, QTableWidgetItem(f"{word['start']:.2f}"))
            self.table_widget.setItem(i, 2, QTableWidgetItem(f"{word['end']:.2f}"))
        self.table_widget.blockSignals(False)

    def on_boundary_changed(self, idx, boundary_type, new_pos):
        # Update the table
        self.table_widget.blockSignals(True)  # Prevent recursive updates
        if boundary_type == 'start':
            item = self.table_widget.item(idx, 1)
            if item is not None:
                item.setText(f"{new_pos:.2f}")
        elif boundary_type == 'end':
            item = self.table_widget.item(idx, 2)
            if item is not None:
                item.setText(f"{new_pos:.2f}")
        self.table_widget.blockSignals(False)

    def on_table_item_changed(self, item):
        row = item.row()
        col = item.column()
        words = self.canvas.words

        if col == 1:
            # Start time changed
            try:
                new_start = float(item.text())
                words[row]['start'] = new_start
                # Update the line position in the waveform
                for line_dict in self.canvas.lines:
                    if line_dict['word_idx'] == row and line_dict['type'] == 'start':
                        line = line_dict['line']
                        line.set_xdata([new_start, new_start])
                        self.canvas.draw()
                        break
            except ValueError:
                pass  # Invalid input, ignore
        elif col == 2:
            # End time changed
            try:
                new_end = float(item.text())
                words[row]['end'] = new_end
                # Update the line position in the waveform
                for line_dict in self.canvas.lines:
                    if line_dict['word_idx'] == row and line_dict['type'] == 'end':
                        line = line_dict['line']
                        line.set_xdata([new_end, new_end])
                        self.canvas.draw()
                        break
            except ValueError:
                pass  # Invalid input, ignore


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()