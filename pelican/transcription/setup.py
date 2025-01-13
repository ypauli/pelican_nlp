from setuptools import setup, find_packages

setup(
    name="pelican-transcription",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "librosa",
        "numpy",
        "PyQt5",
        "pyqtgraph",
        "pydub",
        "simpleaudio",
        "torch",
        "torchaudio",
        "transformers",
        "pyannote.audio",
        "uroman",
        "pandas",
        "soundfile",
    ],
    entry_points={
        'console_scripts': [
            'pelican-transcription=gui.transcription_gui:main',
        ],
    },
) 