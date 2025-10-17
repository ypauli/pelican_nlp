# Audio Transcription Guide

This guide explains how to use the audio transcription functionality in PELICAN-nlp.

## Overview

The transcription functionality allows you to transcribe audio files using the Whisper model, perform speaker diarization, and save the results in a structured format. The transcribed audio files are saved in the `derivatives/transcription/` subdirectory.

## Features

- **Audio Transcription**: Uses OpenAI's Whisper model for accurate speech-to-text conversion
- **Speaker Diarization**: Identifies and separates different speakers in the audio
- **Forced Alignment**: Provides word-level timestamps for precise alignment
- **Chunking**: Automatically splits long audio files into manageable chunks
- **JSON Output**: Saves all results in a structured JSON format

## Configuration

### Basic Configuration

Add the following section to your configuration file:

```yaml
# Transcription Configuration
transcription:
  hf_token: "your_hugging_face_token_here"  # Required for speaker diarization
  num_speakers: 2  # Expected number of speakers
  min_silence_len: 1000  # Minimum silence length for splitting (ms)
  silence_thresh: -30  # Silence threshold in dBFS
  min_length: 90000  # Minimum chunk length (ms)
  max_length: 150000  # Maximum chunk length (ms)
  timestamp_source: "whisper_alignments"  # Options: 'whisper_alignments' or 'forced_alignments'
```

### Required Settings

- `hf_token`: Your Hugging Face token (required for speaker diarization models)
- `num_speakers`: Expected number of speakers in the audio

### Optional Settings

- `min_silence_len`: Minimum length of silence to split audio (default: 1000ms)
- `silence_thresh`: Silence threshold in dBFS (default: -30)
- `min_length`: Minimum chunk length (default: 90000ms)
- `max_length`: Maximum chunk length (default: 150000ms)
- `timestamp_source`: Which alignment to use (default: "whisper_alignments")

## Usage

### Basic Usage

```python
from pelican_nlp.core.corpus import Corpus
from pelican_nlp.core.audio_document import AudioFile
from pelican_nlp.config import load_config

# Load configuration
config = load_config("config_transcription.yml")

# Create audio document
audio_doc = AudioFile(
    file_path="/path/to/audio",
    name="audio_file.wav",
    participant_ID="part-01",
    task="interview",
    num_speakers=2
)

# Create corpus
corpus = Corpus(
    corpus_name="example-transcription",
    documents=[audio_doc],
    configuration_settings=config,
    project_folder=project_folder
)

# Transcribe audio
corpus.transcribe_audio()
```

### Loading Transcription Results

```python
# Load transcription results
if audio_doc.transcription_file:
    audio_doc.load_transcription()
    print(f"Transcription: {audio_doc.transcript_text}")
    print(f"Speaker segments: {len(audio_doc.speaker_segments)}")
    print(f"Word alignments: {len(audio_doc.whisper_alignments)}")
```

## Output Structure

The transcription results are saved as JSON files in `derivatives/transcription/` with the following structure:

```json
{
  "audio_file_path": "path/to/audio.wav",
  "metadata": {
    "file_path": "path/to/audio.wav",
    "length_seconds": 120.5,
    "sample_rate": 16000,
    "models_used": {
      "Chunking": {...},
      "Transcription": {...},
      "Forced Alignment": {...},
      "Speaker Diarization": {...}
    }
  },
  "transcript_text": "Full transcript text...",
  "whisper_alignments": [
    {
      "word": "Hello",
      "start_time": 0.0,
      "end_time": 0.5
    }
  ],
  "forced_alignments": [...],
  "combined_data": [
    {
      "word": "Hello",
      "start_time": 0.0,
      "end_time": 0.5,
      "speaker": "SPEAKER_00"
    }
  ],
  "utterance_data": [
    {
      "text": "Hello world",
      "start_time": 0.0,
      "end_time": 1.0,
      "speaker": "SPEAKER_00",
      "confidence": 0.95
    }
  ],
  "speaker_segments": [
    {
      "start": 0.0,
      "end": 60.0,
      "speaker": "SPEAKER_00"
    }
  ]
}
```

## Dependencies

The transcription functionality requires the following additional dependencies:

- `torch` and `torchaudio`
- `transformers`
- `pyannote.audio`
- `librosa`
- `soundfile`
- `pydub`
- `uroman`

Install them with:

```bash
pip install torch torchaudio transformers pyannote.audio librosa soundfile pydub uroman
```

## Troubleshooting

### Common Issues

1. **Hugging Face Token**: Make sure you have a valid Hugging Face token for speaker diarization
2. **Audio Format**: Ensure your audio files are in a supported format (WAV, MP3, etc.)
3. **Memory**: Large audio files may require significant memory for processing
4. **CUDA**: For faster processing, ensure CUDA is available for GPU acceleration

### Error Messages

- `No transcription file found`: The transcription process failed or the file wasn't created
- `Error during transcription`: Check the audio file format and your Hugging Face token
- `Chunk length validation failed`: The audio chunking process encountered an issue

## Example Configuration File

See `pelican_nlp/sample_configuration_files/config_transcription.yml` for a complete example configuration file.
