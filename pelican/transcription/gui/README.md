# GUI Module

A PyQt5-based graphical user interface for the PELICAn Transcription Tool, providing interactive visualization and editing of audio transcriptions.

## Architecture Overview

### Main Components

#### `MainWindow` (`transcription_gui.py`)
- Primary application window
- Features:
  - Waveform visualization with interactive timeline
  - Word-level transcript editing
  - Speaker management
  - Real-time audio playback
  - Undo/redo functionality
  - Autosave capability

#### `TranscriptionDialog` (`transcription_dialog.py`)
- Dialog for initiating and monitoring transcription process
- Progress tracking for:
  - Audio preprocessing
  - Transcription
  - Alignment
  - Diarization

#### `SimpleGUI` (`simple_gui.py`)
- Lightweight alternative interface
- Basic transcription functionality without advanced editing features

### Components (`components/`)

#### `WaveformCanvas`
- Audio visualization component
- Features:
  - Zoomable waveform display
  - Time markers
  - Word boundary indicators
  - Speaker colorization

#### `AudioLoader`
- Asynchronous audio file loading
- Progress reporting
- Error handling

#### `DraggableLine`
- Interactive timeline markers
- Word boundary adjustment
- Time point selection

### Dialogs (`dialogs/`)

#### `SpeakerManagementDialog`
- Speaker label management
- Merge/split functionality
- Color assignment

#### `BulkEditDialog`
- Batch operations on transcripts
- Multi-word editing
- Speaker assignment

### Commands (`commands/`)

#### Undo/Redo System
- `EditWordCommand`: Word text modification
- `EditSpeakerCommand`: Speaker assignment
- `MoveBoundaryCommand`: Timing adjustment
- `SplitWordCommand`: Word splitting
- `AddWordCommand`: New word insertion
- `DeleteWordCommand`: Word removal
- `BulkEditCommand`: Batch modifications
- `BulkDeleteCommand`: Multiple word deletion

## Features

### Audio Processing
- Support for various audio formats
- Real-time playback
- Waveform visualization
- Time-synchronized display

### Transcript Editing
- Word-level editing
- Speaker assignment
- Timing adjustment
- Bulk operations
- Undo/redo support

### User Interface
- Intuitive timeline navigation
- Keyboard shortcuts
- Context menus
- Status updates
- Progress tracking

### Data Management
- Autosave functionality
- JSON export/import
- Validation checks
- Error handling

## Usage

```python
from gui.transcription_gui import main

if __name__ == '__main__':
    main()
```

## Dependencies

- PyQt5
- pyqtgraph
- numpy
- librosa
- pydub
- simpleaudio

## Keyboard Shortcuts

- `Space`: Play/Pause audio
- `Ctrl+Z`: Undo
- `Ctrl+Y`: Redo
- `Ctrl+S`: Save
- `Delete`: Delete selected word
- `Enter`: Edit selected word
- `Tab`: Navigate between words

## File Formats

### Input
- Audio: WAV, MP3, FLAC, M4A
- Transcripts: JSON

### Output
- Transcripts: JSON with word-level timing and speaker information 