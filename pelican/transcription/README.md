# PELICAN Transcription Tool

A comprehensive audio transcription and editing tool that combines state-of-the-art speech recognition, speaker diarization, and an intuitive GUI for managing transcriptions.

## Architecture

The package consists of two main modules that work together:

### Core Module (`core/`)
The backend processing engine that handles:
- Audio file processing and normalization
- Speech-to-text transcription using Whisper
- Speaker diarization via pyannote.audio
- Forced alignment for precise word timing
- Transcript data management

Key components:
- `AudioFile`: Audio processing and chunking
- `AudioTranscriber`: Speech recognition
- `SpeakerDiarizer`: Speaker identification
- `ForcedAligner`: Word-level timing
- `Transcript`: Data structure for results
- `TranscriptController`: High-level manipulation API

### GUI Module (`gui/`)
The frontend interface built with PyQt5, providing:
- Interactive waveform visualization
- Real-time audio playback
- Word-level transcript editing
- Speaker management
- Undo/redo functionality

Key components:
- `MainWindow`: Primary interface
- `WaveformCanvas`: Audio visualization
- `TranscriptionDialog`: Process monitoring
- Command system for edit operations

## Component Interactions

1. **Audio Processing Pipeline**:
   ```
   AudioFile → AudioTranscriber → ForcedAligner → SpeakerDiarizer → Transcript
   ```

2. **GUI-Core Integration**:
   ```
   MainWindow ←→ TranscriptController ←→ Transcript
                ↓
           WaveformCanvas
                ↓
            AudioLoader
   ```

3. **Edit Operations Flow**:
   ```
   User Input → Command System → TranscriptController → Transcript → GUI Update
   ```

## Installation

1. Create and activate a conda environment:
```bash
conda create -n transcription python=3.10
conda activate transcription
```

2. Install the package:
```bash
pip install -e .
```

## Usage

1. Launch the application:
```bash
pelican-transcription
```

2. Basic Workflow:
   - Load an audio file
   - Initiate transcription process
   - Edit transcription:
     - Adjust word boundaries
     - Correct text
     - Manage speakers
   - Save results

## Features

### Audio Processing
- Multiple audio format support (WAV, MP3, FLAC, M4A)
- Automatic silence detection and segmentation
- Real-time playback and visualization

### Transcription
- State-of-the-art speech recognition
- Automatic speaker diarization
- Precise word-level timing
- Multiple language support

### Editing Interface
- Interactive waveform display
- Word-level editing
- Speaker management
- Bulk operations
- Undo/redo support

### Data Management
- JSON-based storage
- Autosave functionality
- Export/import capabilities

## Dependencies

### Core Dependencies
- torch
- whisper
- pyannote.audio
- uroman
- pydub
- numpy

### GUI Dependencies
- PyQt5
- pyqtgraph
- librosa
- simpleaudio

## Development

The project follows a modular architecture where:
- Core module (`core/`) handles all processing logic
- GUI module (`gui/`) manages user interaction
- Components communicate through well-defined interfaces
- Changes are tracked through the command system

For detailed documentation of each module, see:
- [Core Module Documentation](core/README.md)
- [GUI Module Documentation](gui/README.md) 