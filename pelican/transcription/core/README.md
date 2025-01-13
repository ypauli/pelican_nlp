# Core Transcription Module

This module handles the core functionality of audio transcription, speaker diarization, and forced alignment. Here's a breakdown of the main components:

## Main Classes

### `Chunk`
- Represents a segment of audio with its transcription and alignments
- Stores audio segment, start time, transcript, and alignments

### `AudioFile`
- Handles audio file operations
- Features:
  - Audio loading and normalization
  - Splitting audio into chunks based on silence
  - Metadata tracking
  - Model registration

### `Transcript`
- Manages transcription data and operations
- Features:
  - Loading/saving transcripts from/to JSON
  - Aggregating words into utterances
  - Combining alignment and diarization data

### `AudioTranscriber`
- Handles speech-to-text transcription using Whisper
- Processes audio chunks and generates word-level alignments

### `ForcedAligner`
- Performs forced alignment between text and audio
- Uses uroman for text normalization

### `SpeakerDiarizer`
- Handles speaker diarization using pyannote.audio
- Identifies speaker segments in the audio

## Processing Pipeline

1. Audio Loading & Preprocessing:
   - Load audio file
   - Normalize audio levels
   - Split into manageable chunks based on silence

2. Transcription & Alignment:
   - Transcribe each chunk using Whisper
   - Perform forced alignment
   - Generate word-level timing information

3. Speaker Diarization:
   - Identify speaker segments
   - Combine with word alignments

4. Post-processing:
   - Aggregate words into utterances
   - Combine all data into final transcript
   - Save results to JSON

## Suggested Refactoring

The current monolithic structure should be split into:

1. `audio.py`:
   - `Chunk` class
   - `AudioFile` class
   - Audio processing utilities

2. `transcription.py`:
   - `AudioTranscriber` class
   - Core transcription logic

3. `alignment.py`:
   - `ForcedAligner` class
   - Alignment utilities

4. `diarization.py`:
   - `SpeakerDiarizer` class
   - Speaker identification logic

5. `transcript.py`:
   - `Transcript` class
   - Transcript data management

6. `utils.py`:
   - Common utilities
   - Helper functions 