import numpy as np
import librosa
from PyQt5.QtCore import QObject, pyqtSignal

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