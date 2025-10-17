"""
Audio document handling for PELICAN-nlp.

This module provides the AudioFile class for handling audio files and their processing,
including transcription, speaker diarization, and alignment functionality.
"""

import os
import re
import json
from typing import List

# Third-party Library Imports
import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from pydub.silence import detect_silence


class Chunk:
    """Represents a chunk of audio with transcription data."""
    
    def __init__(self, audio_segment: AudioSegment, start_time: float):
        """
        Initialize a chunk of audio.
        
        :param audio_segment: The audio segment.
        :param start_time: Start time in the original audio (seconds).
        """
        self.audio_segment = audio_segment
        self.start_time = start_time
        self.transcript = ""
        self.whisper_alignments = []
        self.forced_alignments = []


class AudioFile:
    """Handles all operations related to an audio file."""
    
    def __init__(self, file_path, name, target_rms_db: float = -20, **kwargs):
        """
        Initialize an AudioFile instance.
        
        :param file_path: Path to the audio file directory.
        :param name: Name of the audio file.
        :param target_rms_db: Target RMS in dB for normalization.
        :param kwargs: Additional attributes (participant_ID, task, num_speakers, etc.).
        """
        self.file_path = file_path
        self.name = name
        self.file = os.path.join(file_path, name)
        self.target_rms_db = target_rms_db

        # Audio processing attributes
        self.normalized_path = None
        self.audio = None
        self.sample_rate = None
        self.chunks: List[Chunk] = []
        self.speaker_segments = []

        # Metadata
        self.metadata = {
            "file_path": self.file,
            "length_seconds": None,
            "sample_rate": None,
            "target_rms_db": target_rms_db,
            "models_used": {}
        }

        # Initialize optional attributes
        self.participant_ID = kwargs.get('participant_ID')
        self.task = kwargs.get('task')
        self.num_speakers = kwargs.get('num_speakers')
        self.corpus_name = None
        self.recording_length = None

        # Analysis results
        self.opensmile_results = None
        self.prosogram_features = None
        
        # Transcription attributes
        self.transcription_file = None
        self.transcript_text = None
        self.whisper_alignments = []
        self.forced_alignments = []
        self.combined_data = []
        self.combined_utterances = []

    def load_audio(self):
        """Load the audio file using librosa."""
        self.audio, self.sample_rate = librosa.load(self.file, sr=None)
        self.metadata["sample_rate"] = self.sample_rate
        print(f"Loaded audio file: {self.file}")

    def register_model(self, model_name: str, parameters: dict):
        """
        Register a model and its parameters in the metadata.
        
        :param model_name: Name of the model.
        :param parameters: Parameters used for the model.
        """
        self.metadata["models_used"][model_name] = parameters

    def rms_normalization(self):
        """Normalize the audio to the target RMS level and save it."""
        target_rms = 10 ** (self.target_rms_db / 20)
        rms = np.sqrt(np.mean(self.audio ** 2))
        gain = target_rms / rms
        normalized_audio = self.audio * gain
        self.normalized_path = self.file.replace(".wav", "_normalized.wav")
        sf.write(self.normalized_path, normalized_audio, self.sample_rate)
        print(f"Normalized audio saved as: {self.normalized_path}")

    def split_on_silence(self, min_silence_len=1000, silence_thresh=-30,
                         min_length=30000, max_length=180000):
        """
        Split the audio into chunks based on silence.
        
        :param min_silence_len: Minimum length of silence to be used for a split (ms).
        :param silence_thresh: Silence threshold in dBFS.
        :param min_length: Minimum length of a chunk (ms).
        :param max_length: Maximum length of a chunk (ms).
        """
        audio_segment = AudioSegment.from_file(self.normalized_path)
        audio_length_ms = len(audio_segment)
        self.metadata["length_seconds"] = audio_length_ms / 1000
        
        silence_ranges = self._detect_silence_intervals(audio_segment, min_silence_len, silence_thresh)
        splitting_points = self._get_splitting_points(silence_ranges, audio_length_ms)
        initial_intervals = self._create_initial_chunks(splitting_points)
        adjusted_intervals = self._adjust_intervals_by_length(initial_intervals, min_length, max_length)
        chunks_with_timestamps = self._split_audio_by_intervals(audio_segment, adjusted_intervals)

        self.chunks = [Chunk(chunk_audio, start_i / 1000.0) for chunk_audio, start_i, end_i in chunks_with_timestamps]
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

    def _detect_silence_intervals(self, audio_segment: AudioSegment, min_silence_len: int, silence_thresh: int) -> List[List[int]]:
        """Detect silent intervals in the audio segment."""
        return detect_silence(audio_segment, min_silence_len=min_silence_len, silence_thresh=silence_thresh)

    def _get_splitting_points(self, silence_ranges: List[List[int]], audio_length_ms: int) -> List[int]:
        """Compute splitting points based on silence ranges."""
        splitting_points = [0] + [(start + end) // 2 for start, end in silence_ranges] + [audio_length_ms]
        return splitting_points

    def _create_initial_chunks(self, splitting_points: List[int]) -> List[tuple]:
        """Create initial chunks based on splitting points."""
        return list(zip(splitting_points[:-1], splitting_points[1:]))

    def _adjust_intervals_by_length(self, intervals: List[tuple], min_length: int, max_length: int) -> List[tuple]:
        """Adjust intervals based on minimum and maximum length constraints."""
        adjusted_intervals = []
        buffer_start, buffer_end = intervals[0]

        for start, end in intervals[1:]:
            buffer_end = end
            buffer_length = buffer_end - buffer_start

            if buffer_length < min_length:
                # Merge with the next interval by extending the buffer
                continue
            else:
                if buffer_length > max_length:
                    # Split the buffer into multiple chunks of `max_length`
                    num_splits = int(np.ceil(buffer_length / max_length))
                    split_size = int(np.ceil(buffer_length / num_splits))
                    for i in range(num_splits):
                        split_start = buffer_start + i * split_size
                        split_end = min(buffer_start + (i + 1) * split_size, buffer_end)
                        adjusted_intervals.append((split_start, split_end))
                else:
                    # Add the buffer as a valid interval
                    adjusted_intervals.append((buffer_start, buffer_end))
                buffer_start = buffer_end  # Reset buffer_start to the end of the current buffer

        # Handle any remaining buffer (final chunk)
        buffer_length = buffer_end - buffer_start
        if buffer_length > 0:
            if buffer_length >= min_length:
                # Include the final chunk if it's greater than `min_length`
                adjusted_intervals.append((buffer_start, buffer_end))
            else:
                # Optionally include shorter chunks
                print(f"Final chunk is shorter than min_length ({buffer_length} ms), including it anyway.")
                adjusted_intervals.append((buffer_start, buffer_end))

        return adjusted_intervals
    
    def validate_chunk_lengths(self, audio_length_ms: int, tolerance: float = 1.0):
        """Validate that the combined length of all chunks matches the original audio length."""
        # Sum up the duration of all chunks
        combined_length = sum(len(chunk.audio_segment) for chunk in self.chunks)

        # Calculate the difference
        difference = abs(combined_length - audio_length_ms)
        if difference > tolerance:
            raise AssertionError(
                f"Chunk lengths validation failed! Combined chunk length ({combined_length} ms) "
                f"differs from original audio length ({audio_length_ms} ms) by {difference} ms, "
                f"which exceeds the allowed tolerance of {tolerance} ms."
            )
        print(f"Chunk length validation passed: Total chunks = {combined_length} ms, Original = {audio_length_ms} ms.")

    def _split_audio_by_intervals(self, audio_segment: AudioSegment, intervals: List[tuple]) -> List[tuple]:
        """Split the audio segment into chunks based on the provided intervals."""
        return [(audio_segment[start_ms:end_ms], start_ms, end_ms) for start_ms, end_ms in intervals]
    
    def combine_chunks(self):
        """Combine transcripts and alignments from all chunks."""
        self.transcript_text = " ".join([chunk.transcript for chunk in self.chunks])
        self.whisper_alignments = []
        self.forced_alignments = []
        for chunk in self.chunks:
            self.whisper_alignments.extend(chunk.whisper_alignments)
            self.forced_alignments.extend(chunk.forced_alignments)
        print("Combined transcripts and alignments from all chunks.")

    def combine_alignment_and_diarization(self, alignment_source: str):
        """
        Combine alignment and diarization data by assigning speaker labels to each word.
        
        :param alignment_source: The alignment data to use ('whisper_alignments' or 'forced_alignments').
        """
        if alignment_source not in ['whisper_alignments', 'forced_alignments']:
            raise ValueError("Invalid alignment_source. Choose 'whisper_alignments' or 'forced_alignments'.")

        alignment = getattr(self, alignment_source, None)
        if alignment is None:
            raise ValueError(f"The alignment source '{alignment_source}' does not exist in the AudioFile object.")

        if not self.speaker_segments:
            print("No speaker segments available for diarization. All words will be labeled as 'UNKNOWN'.")
            self.combined_data = [{**word, 'speaker': 'UNKNOWN'} for word in alignment]
            return

        combined = []
        seg_idx = 0
        num_segments = len(self.speaker_segments)

        for word in alignment:
            word_start = word['start_time']
            word_end = word['end_time']
            word_duration = max(1e-6, word_end - word_start)  # Avoid zero-duration

            speaker_overlap = {}

            # Advance segments that have ended before the word starts
            while seg_idx < num_segments and self.speaker_segments[seg_idx]['end'] < word_start:
                seg_idx += 1

            temp_idx = seg_idx
            while temp_idx < num_segments and self.speaker_segments[temp_idx]['start'] < word_end:
                seg = self.speaker_segments[temp_idx]
                seg_start = seg['start']
                seg_end = seg['end']
                speaker = seg['speaker']

                if seg_start <= word_start < seg_end:
                    overlap = word_duration  # Full overlap
                else:
                    overlap_start = max(word_start, seg_start)
                    overlap_end = min(word_end, seg_end)
                    overlap = max(0.0, overlap_end - overlap_start)

                if overlap > 0:
                    speaker_overlap[speaker] = speaker_overlap.get(speaker, 0.0) + overlap

                temp_idx += 1

            assigned_speaker = max(speaker_overlap, key=speaker_overlap.get) if speaker_overlap else 'UNKNOWN'
            word_with_speaker = word.copy()
            word_with_speaker['speaker'] = assigned_speaker
            combined.append(word_with_speaker)

        self.combined_data = combined
        self.metadata["alignment_source"] = alignment_source
        print(f"Combined alignment and diarization data with {len(self.combined_data)} entries.")

    def aggregate_to_utterances(self):
        """Aggregate word-level data into utterances based on sentence endings."""
        if not self.combined_data:
            print("No combined data available to aggregate.")
            return

        utterances = []
        current_utterance = {
            "text": "",
            "start_time": None,
            "end_time": None,
            "speakers": {}
        }

        sentence_endings = re.compile(r'[.?!]$')
        print("Aggregating words into utterances...")
        for word_data in self.combined_data:
            word = word_data["word"]
            start_time = word_data["start_time"]
            end_time = word_data["end_time"]
            speaker = word_data["speaker"]

            if current_utterance["start_time"] is None:
                current_utterance["start_time"] = start_time

            current_utterance["text"] += ("" if current_utterance["text"] == "" else " ") + word
            current_utterance["end_time"] = end_time

            if speaker not in current_utterance["speakers"]:
                current_utterance["speakers"][speaker] = 0
            current_utterance["speakers"][speaker] += 1

            if sentence_endings.search(word):
                majority_speaker, majority_count = max(
                    current_utterance["speakers"].items(), key=lambda item: item[1]
                )
                total_words = sum(current_utterance["speakers"].values())
                confidence = round(majority_count / total_words, 2)

                utterances.append({
                    "text": current_utterance["text"],
                    "start_time": current_utterance["start_time"],
                    "end_time": current_utterance["end_time"],
                    "speaker": majority_speaker,
                    "confidence": confidence,
                })

                current_utterance = {
                    "text": "",
                    "start_time": None,
                    "end_time": None,
                    "speakers": {}
                }

        # Handle any remaining words as the last utterance
        if current_utterance["text"]:
            majority_speaker, majority_count = max(
                current_utterance["speakers"].items(), key=lambda item: item[1]
            )
            total_words = sum(current_utterance["speakers"].values())
            confidence = round(majority_count / total_words, 2)

            utterances.append({
                "text": current_utterance["text"],
                "start_time": current_utterance["start_time"],
                "end_time": current_utterance["end_time"],
                "speaker": majority_speaker,
                "confidence": confidence,
            })

        self.combined_utterances = utterances
        print("Aggregated utterances from combined data.")

    def save_as_json(self, output_file="all_transcript_data.json"):
        """
        Save all transcript data to a JSON file.
        
        :param output_file: Path to the output JSON file.
        """
        if not self.combined_data:
            print("No combined data available to save. Ensure 'combine_alignment_and_diarization' is run first.")
            return

        data = {
            "audio_file_path": self.file,
            "metadata": self.metadata,
            "transcript_text": self.transcript_text,
            "whisper_alignments": self.whisper_alignments,
            "forced_alignments": self.forced_alignments,
            "combined_data": self.combined_data,
            "utterance_data": self.combined_utterances,
            "speaker_segments": self.speaker_segments   
        }

        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            print(f"All transcript data successfully saved to '{output_file}'.")
        except Exception as e:
            print(f"Error saving JSON file: {e}")

    def load_transcription(self, transcription_file_path=None):
        """
        Load transcription results from a JSON file.
        
        :param transcription_file_path: Path to the transcription JSON file. 
                                      If None, uses self.transcription_file
        """
        if transcription_file_path is None:
            transcription_file_path = self.transcription_file
            
        if not transcription_file_path or not os.path.exists(transcription_file_path):
            print(f"No transcription file found at {transcription_file_path}")
            return
            
        try:
            with open(transcription_file_path, 'r', encoding='utf-8') as f:
                transcription_data = json.load(f)
            
            # Load transcription data into the document
            self.transcript_text = transcription_data.get('transcript_text', '')
            self.whisper_alignments = transcription_data.get('whisper_alignments', [])
            self.forced_alignments = transcription_data.get('forced_alignments', [])
            self.speaker_segments = transcription_data.get('speaker_segments', [])
            self.combined_data = transcription_data.get('combined_data', [])
            self.combined_utterances = transcription_data.get('utterance_data', [])
            self.metadata = transcription_data.get('metadata', self.metadata)
            
            print(f"Loaded transcription data from {transcription_file_path}")
            
        except Exception as e:
            print(f"Error loading transcription file {transcription_file_path}: {e}")

    def __repr__(self):
        return f"AudioFile(file_name={self.name})"