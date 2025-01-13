from PyQt5.QtCore import Qt
import pyqtgraph as pg

class DraggableLine(pg.InfiniteLine):
    def __init__(self, pos, color, idx, boundary_type, span=(0,1), pen=None, movable=True):
        pen = pen or pg.mkPen(color=color, width=2)
        super().__init__(pos=pos, angle=90, pen=pen, movable=movable)
        self.idx = idx
        self.boundary_type = boundary_type
        self.setSpan(span[0], span[1])
        self.setHoverPen(pen.color().lighter())
        self.setCursor(Qt.SizeHorCursor)
        self.old_pos = pos 