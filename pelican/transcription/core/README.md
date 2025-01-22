# Core Transcription Module

This module provides a comprehensive solution for audio transcription, speaker diarization, and forced alignment. It features a modular architecture with configurable components and a controller layer for transcript manipulation.

## Architecture Overview

### Core Components

#### `AudioFile` (`audio.py`)
- Handles audio file operations and preprocessing
- Features:
  - Audio loading and normalization
  - Intelligent chunking based on silence detection
  - Configurable silence parameters for optimal segmentation
  - Metadata tracking and model registration

#### `AudioTranscriber` (`transcription.py`)
- Manages speech-to-text transcription using Whisper
- Supports multiple Whisper model variants
- Configurable language settings
- Optimized for GPU acceleration when available

#### `ForcedAligner` (`alignment.py`)
- Performs precise forced alignment between text and audio
- Uses uroman for text normalization
- Generates word-level timing information
- Configurable alignment parameters

#### `SpeakerDiarizer` (`diarization.py`)
- Implements speaker diarization using pyannote.audio
- Features:
  - Automatic speaker count detection
  - Configurable clustering parameters
  - Adjustable segmentation settings
  - Speaker overlap handling

#### `Transcript` (`transcript.py`)
- Central data structure for managing transcription results
- Features:
  - JSON-based serialization
  - Word-level timing information
  - Speaker labels
  - Utterance aggregation

#### `TranscriptController` (`transcription_controller.py`)
- High-level interface for transcript manipulation
- Features:
  - Speaker merging and splitting
  - Word boundary adjustments
  - Speaker management
  - Data access methods

## Processing Pipeline

1. **Initialization and Setup**
   - Load and validate audio file
   - Initialize models and components
   - Configure processing parameters

2. **Audio Preprocessing**
   - Normalize audio levels
   - Split into chunks based on configurable silence parameters
   - Prepare for parallel processing

3. **Core Processing**
   - Transcribe audio using Whisper
   - Generate word-level alignments
   - Perform speaker diarization
   - Combine results into unified transcript

4. **Post-processing**
   - Aggregate words into utterances
   - Apply speaker labels
   - Generate final transcript
   - Export results to JSON

## Configuration Options

### Audio Processing
- `pause_threshold`: Minimum silence duration for segmentation
- `max_utterance_duration`: Maximum length of utterance chunks
- `silence_params`: Detailed control over audio splitting
  - `min_silence_len`: Minimum silence duration (ms)
  - `silence_thresh`: Silence threshold (dBFS)
  - `min_length`: Minimum chunk length (ms)
  - `max_length`: Maximum chunk length (ms)

### Diarization
- `num_speakers`: Optional speaker count override
- `diarizer_params`: Fine-tuning for speaker detection
  - `segmentation`: Controls for speech segment detection
  - `clustering`: Parameters for speaker clustering

### Transcription
- `model`: Choice of Whisper model variant
- `language`: Target language for transcription
- `device`: Processing device selection (CPU/GPU)
- `alignment_source`: Source for word timing information

## Usage Example

```python
from core.main import process_audio

result = process_audio(
    file_path="audio.wav",
    hf_token="your-huggingface-token",
    output_dir="output",
    model="openai/whisper-large",
    language="de",
    num_speakers=2
)
```

## Dependencies

- torch
- whisper
- pyannote.audio
- uroman
- pydub
- numpy 