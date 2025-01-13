"""
Audio processing module for handling audio files and chunks.
"""
import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from pydub.silence import detect_silence
from typing import List, Dict, Tuple, Any
from pathlib import Path


class Chunk:
    """Represents a chunk of audio with its transcription and alignments."""
    
    def __init__(self, audio_segment: AudioSegment, start_time: float):
        """
        Initialize a chunk of audio.
        
        Args:
            audio_segment: The audio segment
            start_time: Start time in the original audio (seconds)
        """
        self.audio_segment = audio_segment
        self.start_time = start_time  # Start time in seconds
        self.transcript = ""
        self.whisper_alignments = []
        self.forced_alignments = []


class AudioFile:
    """Handles all operations related to an audio file."""
    
    def __init__(self, file_path: str, target_rms_db: float = -20):
        """
        Initialize audio file handler.
        
        Args:
            file_path: Path to the audio file
            target_rms_db: Target RMS in dB for normalization
        """
        self.file_path = file_path
        self.target_rms_db = target_rms_db
        self.normalized_path = None
        self.audio = None
        self.sample_rate = None
        self.chunks: List[Chunk] = []
        self.speaker_segments = []
        
        self.metadata = {
            "file_path": file_path,
            "length_seconds": None,
            "sample_rate": None,
            "target_rms_db": target_rms_db,
            "models_used": {}
        }
        
        self.load_audio()

    def load_audio(self):
        """Load the audio file using librosa."""
        self.audio, self.sample_rate = librosa.load(self.file_path, sr=None)
        self.metadata["sample_rate"] = self.sample_rate
        print(f"Loaded audio file: {self.file_path}")
        
    def register_model(self, model_name: str, parameters: dict):
        """
        Register a model and its parameters in the metadata.
        
        Args:
            model_name: Name of the model
            parameters: Parameters used for the model
        """
        self.metadata["models_used"][model_name] = parameters

    def rms_normalization(self):
        """Normalize the audio to the target RMS level and save it."""
        target_rms = 10 ** (self.target_rms_db / 20)
        rms = np.sqrt(np.mean(self.audio ** 2))
        gain = target_rms / rms
        normalized_audio = self.audio * gain
        self.normalized_path = str(Path(self.file_path).with_suffix('')) + "_normalized.wav"
        sf.write(self.normalized_path, normalized_audio, self.sample_rate)
        print(f"Normalized audio saved as: {self.normalized_path}")

    def split_on_silence(self, min_silence_len=1000, silence_thresh=-30,
                         min_length=30000, max_length=180000):
        """
        Split the audio into chunks based on silence.
        
        Args:
            min_silence_len: Minimum length of silence to be used for a split (ms)
            silence_thresh: Silence threshold in dBFS
            min_length: Minimum length of a chunk (ms)
            max_length: Maximum length of a chunk (ms)
        """
        audio_segment = AudioSegment.from_file(self.normalized_path)
        audio_length_ms = len(audio_segment)
        self.metadata["length_seconds"] = audio_length_ms / 1000
        silence_ranges = self._detect_silence_intervals(audio_segment, min_silence_len, silence_thresh)
        splitting_points = self._get_splitting_points(silence_ranges, audio_length_ms)
        initial_intervals = self._create_initial_chunks(splitting_points)
        adjusted_intervals = self._adjust_intervals_by_length(initial_intervals, min_length, max_length)
        chunks_with_timestamps = self._split_audio_by_intervals(audio_segment, adjusted_intervals)

        self.chunks = [Chunk(chunk_audio, start_i / 1000.0) 
                      for chunk_audio, start_i, end_i in chunks_with_timestamps]
        print(f"Total chunks after splitting: {len(self.chunks)}")
    
        # Validate the combined length of chunks
        self.validate_chunk_lengths(audio_length_ms)
        
        self.register_model("Chunking", {
            "min_silence_len": min_silence_len,
            "silence_thresh": silence_thresh,
            "min_length": min_length,
            "max_length": max_length,
            "num_chunks": len(self.chunks)
        })

    def _detect_silence_intervals(self, audio_segment: AudioSegment, 
                                min_silence_len: int, silence_thresh: int) -> List[List[int]]:
        """
        Detect silent intervals in the audio segment.
        
        Args:
            audio_segment: The audio segment
            min_silence_len: Minimum length of silence to be used for a split (ms)
            silence_thresh: Silence threshold in dBFS
            
        Returns:
            List of [start_ms, end_ms] pairs representing silence periods
        """
        return detect_silence(audio_segment, min_silence_len=min_silence_len, 
                            silence_thresh=silence_thresh)

    def _get_splitting_points(self, silence_ranges: List[List[int]], 
                            audio_length_ms: int) -> List[int]:
        """
        Compute splitting points based on silence ranges.
        
        Args:
            silence_ranges: List of silence intervals
            audio_length_ms: Total length of the audio in ms
            
        Returns:
            Sorted list of splitting points in ms
        """
        splitting_points = [0] + [(start + end) // 2 for start, end in silence_ranges] + [audio_length_ms]
        return splitting_points

    def _create_initial_chunks(self, splitting_points: List[int]) -> List[Tuple[int, int]]:
        """
        Create initial chunks based on splitting points.
        
        Args:
            splitting_points: List of splitting points in ms
            
        Returns:
            List of (start_ms, end_ms) tuples
        """
        return list(zip(splitting_points[:-1], splitting_points[1:]))

    def _adjust_intervals_by_length(self, intervals: List[Tuple[int, int]], 
                                  min_length: int, max_length: int) -> List[Tuple[int, int]]:
        """
        Adjust intervals based on minimum and maximum length constraints.

        Args:
            intervals: List of (start_ms, end_ms) tuples
            min_length: Minimum length of a chunk (ms)
            max_length: Maximum length of a chunk (ms)
            
        Returns:
            Adjusted list of intervals
        """
        adjusted_intervals = []
        buffer_start, buffer_end = intervals[0]

        for start, end in intervals[1:]:
            buffer_end = end
            buffer_length = buffer_end - buffer_start

            if buffer_length < min_length:
                continue
            else:
                if buffer_length > max_length:
                    num_splits = int(np.ceil(buffer_length / max_length))
                    split_size = int(np.ceil(buffer_length / num_splits))
                    for i in range(num_splits):
                        split_start = buffer_start + i * split_size
                        split_end = min(buffer_start + (i + 1) * split_size, buffer_end)
                        adjusted_intervals.append((split_start, split_end))
                else:
                    adjusted_intervals.append((buffer_start, buffer_end))
                buffer_start = buffer_end

        # Handle any remaining buffer (final chunk)
        buffer_length = buffer_end - buffer_start
        if buffer_length > 0:
            if buffer_length >= min_length:
                adjusted_intervals.append((buffer_start, buffer_end))
            else:
                print(f"Final chunk is shorter than min_length ({buffer_length} ms), including it anyway.")
                adjusted_intervals.append((buffer_start, buffer_end))

        return adjusted_intervals
    
    def validate_chunk_lengths(self, audio_length_ms: int, tolerance: float = 1.0):
        """
        Validate that the combined length of all chunks matches the original audio length.

        Args:
            audio_length_ms: Length of the original audio in milliseconds
            tolerance: Allowed tolerance in milliseconds
        """
        combined_length = sum(len(chunk.audio_segment) for chunk in self.chunks)
        difference = abs(combined_length - audio_length_ms)
        
        if difference > tolerance:
            raise AssertionError(
                f"Chunk lengths validation failed! Combined chunk length ({combined_length} ms) "
                f"differs from original audio length ({audio_length_ms} ms) by {difference} ms, "
                f"which exceeds the allowed tolerance of {tolerance} ms."
            )
        print(f"Chunk length validation passed: Total chunks = {combined_length} ms, Original = {audio_length_ms} ms.")

    def _split_audio_by_intervals(self, audio_segment: AudioSegment, 
                                intervals: List[Tuple[int, int]]) -> List[Tuple[AudioSegment, int, int]]:
        """
        Split the audio segment into chunks based on the provided intervals.
        
        Args:
            audio_segment: The audio segment
            intervals: List of (start_ms, end_ms) tuples
            
        Returns:
            List of (chunk_audio, start_ms, end_ms) tuples
        """
        return [(audio_segment[start_ms:end_ms], start_ms, end_ms) 
                for start_ms, end_ms in intervals] 

    @property
    def transcript_text(self) -> str:
        """
        Get the complete transcript text from all chunks.
        
        Returns:
            Complete transcript text
        """
        return " ".join(chunk.transcript for chunk in self.chunks if chunk.transcript.strip())

    @property
    def whisper_alignments(self) -> List[Dict[str, Any]]:
        """
        Get all Whisper alignments from all chunks.
        
        Returns:
            List of word alignments from all chunks
        """
        alignments = []
        for chunk in self.chunks:
            alignments.extend(chunk.whisper_alignments)
        return alignments
        
    @property
    def forced_alignments(self) -> List[Dict[str, Any]]:
        """
        Get all forced alignments from all chunks.
        
        Returns:
            List of word alignments from all chunks
        """
        alignments = []
        for chunk in self.chunks:
            alignments.extend(chunk.forced_alignments)
        return alignments

    def split_on_silence(self, min_silence_len=1000, silence_thresh=-30,
                         min_length=30000, max_length=180000):
        """
        Split the audio into chunks based on silence.
        
        Args:
            min_silence_len: Minimum length of silence to be used for a split (ms)
            silence_thresh: Silence threshold in dBFS
            min_length: Minimum length of a chunk (ms)
            max_length: Maximum length of a chunk (ms)
        """
        audio_segment = AudioSegment.from_file(self.normalized_path)
        audio_length_ms = len(audio_segment)
        self.metadata["length_seconds"] = audio_length_ms / 1000
        silence_ranges = self._detect_silence_intervals(audio_segment, min_silence_len, silence_thresh)
        splitting_points = self._get_splitting_points(silence_ranges, audio_length_ms)
        initial_intervals = self._create_initial_chunks(splitting_points)
        adjusted_intervals = self._adjust_intervals_by_length(initial_intervals, min_length, max_length)
        chunks_with_timestamps = self._split_audio_by_intervals(audio_segment, adjusted_intervals)

        self.chunks = [Chunk(chunk_audio, start_i / 1000.0) 
                      for chunk_audio, start_i, end_i in chunks_with_timestamps]
        print(f"Total chunks after splitting: {len(self.chunks)}")
    
        # Validate the combined length of chunks
        self.validate_chunk_lengths(audio_length_ms)
        
        self.register_model("Chunking", {
            "min_silence_len": min_silence_len,
            "silence_thresh": silence_thresh,
            "min_length": min_length,
            "max_length": max_length,
            "num_chunks": len(self.chunks)
        }) 