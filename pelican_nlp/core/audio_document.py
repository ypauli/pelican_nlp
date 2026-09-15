"""
Audio document handling for PELICAN-nlp.

This module provides the AudioFile class for handling audio files and their processing,
including transcription, speaker diarization, and alignment functionality.
"""

from __future__ import annotations

import os
import re
import json
from typing import TYPE_CHECKING, List

import numpy as np

from pelican_nlp.config import debug_print

if TYPE_CHECKING:
    from pydub import AudioSegment

# Sentence-final punctuation, including the ellipsis and the CJK forms Whisper
# emits for Chinese/Japanese audio (ASCII-only matching used to leave such
# transcripts as one utterance per recording).
SENTENCE_ENDINGS = re.compile(r'[.?!\u2026\u3002\uff1f\uff01]["\'\u201d\u2019)\]]*$')


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
        self.source_folder = kwargs.get('source_folder')
        self.unit_kind = kwargs.get('unit_kind')
        self.task = kwargs.get('task')
        self.num_speakers = kwargs.get('num_speakers')
        self.corpus_name = None
        self.recording_length = None
        self.transcription_file = None
        self.transcription_text_file = None  # Path to plain text transcription file
        self.transcript_text = None
        self.whisper_alignments = []
        self.forced_alignments = []
        self.combined_data = []
        self.combined_utterances = []

    def load_audio(self):
        """Load the audio file using librosa."""
        import librosa

        self.audio, self.sample_rate = librosa.load(self.file, sr=None)
        self.metadata["sample_rate"] = self.sample_rate
        debug_print(f"Loaded audio file: {self.file}")

    def register_model(self, model_name: str, parameters: dict):
        """
        Register a model and its parameters in the metadata.
        
        :param model_name: Name of the model.
        :param parameters: Parameters used for the model.
        """
        self.metadata["models_used"][model_name] = parameters

    def rms_normalization(self, output_dir=None, peak_ceiling: float = 0.99):
        """
        Normalize the audio to the target RMS level and save it.

        The gain is capped so that peaks stay below ``peak_ceiling``: WAV output is
        16-bit PCM, so an uncapped RMS gain clips loud passages and feeds distorted
        audio to the ASR model.

        :param output_dir: Directory to save normalized audio. If None, saves in same directory as original file.
        :param peak_ceiling: Maximum absolute sample value allowed after normalization.
        """
        import soundfile as sf

        target_rms = 10 ** (self.target_rms_db / 20)
        rms = float(np.sqrt(np.mean(np.square(self.audio, dtype=np.float64))))
        if not np.isfinite(rms) or rms <= 0:
            debug_print("Audio has no measurable signal (RMS is zero); skipping RMS gain.")
            gain = 1.0
        else:
            gain = target_rms / rms

        peak = float(np.max(np.abs(self.audio))) if self.audio.size else 0.0
        peak_limited = False
        if peak > 0 and peak * gain > peak_ceiling:
            gain = peak_ceiling / peak
            peak_limited = True

        normalized_audio = self.audio * gain

        if output_dir:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            # Use original filename with _normalized suffix
            base_name = os.path.splitext(self.name)[0]
            normalized_filename = f"{base_name}_normalized.wav"
            self.normalized_path = os.path.join(output_dir, normalized_filename)
        else:
            # Same directory as the original, but never the original path itself:
            # a plain ".wav" replacement is a no-op for .mp3/.m4a inputs.
            self.normalized_path = f"{os.path.splitext(self.file)[0]}_normalized.wav"

        sf.write(self.normalized_path, normalized_audio, self.sample_rate)
        self.metadata["normalization"] = {
            "target_rms_db": self.target_rms_db,
            "applied_gain_db": round(float(20 * np.log10(gain)), 3) if gain > 0 else None,
            "peak_limited": peak_limited,
        }
        if peak_limited:
            debug_print(
                f"Reduced normalization gain to keep peaks below {peak_ceiling}; "
                f"target RMS of {self.target_rms_db} dB was not reached."
            )
        debug_print(f"Normalized audio saved as: {self.normalized_path}")

    def split_on_silence(self, min_silence_len=1000, silence_thresh=-30,
                         min_length=30000, max_length=180000):
        """
        Split the audio into chunks based on silence.
        
        :param min_silence_len: Minimum length of silence to be used for a split (ms).
        :param silence_thresh: Silence threshold in dBFS.
        :param min_length: Minimum length of a chunk (ms).
        :param max_length: Maximum length of a chunk (ms).
        """
        from pydub import AudioSegment

        audio_segment = AudioSegment.from_file(self.normalized_path)
        audio_length_ms = len(audio_segment)
        self.metadata["length_seconds"] = audio_length_ms / 1000
        
        silence_ranges = self._detect_silence_intervals(audio_segment, min_silence_len, silence_thresh)
        splitting_points = self._get_splitting_points(silence_ranges, audio_length_ms)
        initial_intervals = self._create_initial_chunks(splitting_points)
        adjusted_intervals = self._adjust_intervals_by_length(
            initial_intervals, min_length, max_length, candidate_points=splitting_points
        )
        chunks_with_timestamps = self._split_audio_by_intervals(audio_segment, adjusted_intervals)

        self.chunks = [Chunk(chunk_audio, start_i / 1000.0) for chunk_audio, start_i, end_i in chunks_with_timestamps]
        debug_print(f"Total chunks after splitting: {len(self.chunks)}")
    
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
        from pydub.silence import detect_silence

        return detect_silence(audio_segment, min_silence_len=min_silence_len, silence_thresh=silence_thresh)

    def _get_splitting_points(self, silence_ranges: List[List[int]], audio_length_ms: int) -> List[int]:
        """Compute splitting points based on silence ranges."""
        splitting_points = [0] + [(start + end) // 2 for start, end in silence_ranges] + [audio_length_ms]
        return splitting_points

    def _create_initial_chunks(self, splitting_points: List[int]) -> List[tuple]:
        """Create initial chunks based on splitting points."""
        return list(zip(splitting_points[:-1], splitting_points[1:]))

    def _adjust_intervals_by_length(self, intervals: List[tuple], min_length: int, max_length: int,
                                    candidate_points: List[int] = None) -> List[tuple]:
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
                    adjusted_intervals.extend(
                        self._split_oversized_buffer(
                            buffer_start, buffer_end, max_length, candidate_points
                        )
                    )
                else:
                    # Add the buffer as a valid interval
                    adjusted_intervals.append((buffer_start, buffer_end))
                buffer_start = buffer_end  # Reset buffer_start to the end of the current buffer

        # Handle any remaining buffer (final chunk)
        buffer_length = buffer_end - buffer_start
        if buffer_length > 0:
            if buffer_length < min_length:
                debug_print(f"Final chunk is shorter than min_length ({buffer_length} ms), including it anyway.")
            if buffer_length > max_length:
                adjusted_intervals.extend(
                    self._split_oversized_buffer(
                        buffer_start, buffer_end, max_length, candidate_points
                    )
                )
            else:
                adjusted_intervals.append((buffer_start, buffer_end))

        return adjusted_intervals

    @staticmethod
    def _split_oversized_buffer(buffer_start: int, buffer_end: int, max_length: int,
                                candidate_points: List[int] = None) -> List[tuple]:
        """Split a too-long span, snapping cuts to silence instead of cutting mid-word.

        Each cut aims at an evenly spaced target and may move to a detected silence
        midpoint within a quarter of ``max_length`` of that target. Snapping can add
        one chunk compared to a blind division, which is cheap; cutting through a
        word is not, because both sides then start or end mid-utterance.

        Pieces stay contiguous, gapless and within ``max_length`` so that
        :meth:`validate_chunk_lengths` still passes.
        """
        if buffer_end - buffer_start <= max_length:
            return [(buffer_start, buffer_end)]

        candidates = sorted(
            point for point in (candidate_points or []) if buffer_start < point < buffer_end
        )
        tolerance = max(1, max_length // 4)

        pieces = []
        cut_start = buffer_start
        while buffer_end - cut_start > max_length:
            remaining = buffer_end - cut_start
            splits = int(np.ceil(remaining / max_length))
            target = cut_start + int(np.ceil(remaining / splits))
            earliest = max(cut_start + tolerance, target - tolerance)
            latest = min(cut_start + max_length, target + tolerance)

            best = None
            for point in candidates:
                if point < earliest:
                    continue
                if point > latest:
                    break  # candidates are sorted
                if best is None or abs(point - target) < abs(best - target):
                    best = point

            cut = best if best is not None else min(target, cut_start + max_length)
            if cut <= cut_start:  # pathological input; fall back to a hard cut
                cut = min(cut_start + max_length, buffer_end)
            pieces.append((cut_start, cut))
            cut_start = cut

        pieces.append((cut_start, buffer_end))
        return pieces

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
        debug_print(f"Chunk length validation passed: Total chunks = {combined_length} ms, Original = {audio_length_ms} ms.")

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
        debug_print("Combined transcripts and alignments from all chunks.")

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
            # If only one speaker is specified, assign a default speaker label
            # Otherwise, label as 'UNKNOWN' (diarization may have failed)
            if self.num_speakers and self.num_speakers == 1:
                debug_print("No speaker segments available (single speaker mode). All words will be labeled as 'SPEAKER_0'.")
                self.combined_data = [{**word, 'speaker': 'SPEAKER_0'} for word in alignment]
            else:
                debug_print("No speaker segments available for diarization. All words will be labeled as 'UNKNOWN'.")
                self.combined_data = [{**word, 'speaker': 'UNKNOWN'} for word in alignment]
            self.metadata["alignment_source"] = alignment_source
            return

        segments = sorted(self.speaker_segments, key=lambda seg: (seg['start'], seg['end']))
        num_segments = len(segments)

        # Whisper word timestamps are not strictly monotonic (chunk boundaries and
        # collapsed timestamps reorder them), so walk the words in time order and
        # write results back to their original position.
        order = sorted(
            range(len(alignment)),
            key=lambda i: (alignment[i]['start_time'], alignment[i]['end_time']),
        )

        combined = [None] * len(alignment)
        seg_idx = 0
        words_without_speaker = 0

        for position in order:
            word = alignment[position]
            word_start = word['start_time']
            word_end = max(word['end_time'], word_start)

            speaker_overlap = {}

            # Advance segments that have ended before the word starts
            while seg_idx < num_segments and segments[seg_idx]['end'] < word_start:
                seg_idx += 1

            temp_idx = seg_idx
            while temp_idx < num_segments and segments[temp_idx]['start'] <= word_end:
                seg = segments[temp_idx]
                seg_start = seg['start']
                seg_end = seg['end']

                overlap = max(0.0, min(word_end, seg_end) - max(word_start, seg_start))
                if overlap <= 0 and seg_start <= word_start < seg_end:
                    # Zero-duration word (Whisper emits many) that falls inside a
                    # segment: credit it to that segment instead of dropping it.
                    overlap = 1e-6

                if overlap > 0:
                    speaker = seg['speaker']
                    speaker_overlap[speaker] = speaker_overlap.get(speaker, 0.0) + overlap

                temp_idx += 1

            assigned_speaker = max(speaker_overlap, key=speaker_overlap.get) if speaker_overlap else 'UNKNOWN'
            if assigned_speaker == 'UNKNOWN':
                words_without_speaker += 1

            word_with_speaker = word.copy()
            word_with_speaker['speaker'] = assigned_speaker
            combined[position] = word_with_speaker

        self._fill_unknown_speakers(combined)
        self._smooth_word_speakers(combined)

        self.combined_data = combined
        self.metadata["alignment_source"] = alignment_source
        unique_speakers = sorted({word['speaker'] for word in combined})
        debug_print(
            f"Combined {len(combined)} words with {num_segments} speaker segments; "
            f"speakers={unique_speakers}; unassigned={words_without_speaker}."
        )
        if words_without_speaker:
            debug_print(
                f"Warning: {words_without_speaker} words fell outside every speaker "
                "segment and were labeled 'UNKNOWN'."
            )

    @staticmethod
    def _fill_unknown_speakers(words: List[dict]) -> int:
        """Assign words in a diarization gap to the speaker surrounding the gap.

        A stretch of words that overlaps no speaker segment but is flanked by the
        same speaker on both sides belongs to that speaker's turn. Without this the
        gap becomes a spurious 'UNKNOWN' utterance in the middle of a turn.

        :return: Number of words relabeled.
        """
        relabeled = 0
        index = 0
        total = len(words)
        while index < total:
            if words[index]['speaker'] != 'UNKNOWN':
                index += 1
                continue
            end = index
            while end < total and words[end]['speaker'] == 'UNKNOWN':
                end += 1
            if index > 0 and end < total:
                previous = words[index - 1]['speaker']
                if previous != 'UNKNOWN' and previous == words[end]['speaker']:
                    for word in words[index:end]:
                        word['speaker'] = previous
                    relabeled += end - index
            index = end
        if relabeled:
            debug_print(f"Filled {relabeled} words in diarization gaps from surrounding speaker.")
        return relabeled

    @staticmethod
    def _smooth_word_speakers(words: List[dict], min_run: int = 2) -> int:
        """Absorb single-word speaker flips into the surrounding speaker.

        Diarization plus noisy word timestamps produces isolated one-word speaker
        switches inside an otherwise uniform turn. Left in place they fragment
        utterance aggregation, so a run shorter than ``min_run`` that is flanked by
        the same speaker on both sides is reassigned to that speaker.

        :return: Number of words relabeled.
        """
        relabeled = 0
        index = 0
        total = len(words)
        while index < total:
            end = index + 1
            while end < total and words[end]['speaker'] == words[index]['speaker']:
                end += 1
            run_length = end - index
            has_neighbours = index > 0 and end < total
            if run_length < min_run and has_neighbours:
                previous = words[index - 1]['speaker']
                following = words[end]['speaker']
                if previous == following and previous != words[index]['speaker']:
                    for word in words[index:end]:
                        word['speaker'] = previous
                    relabeled += run_length
            index = end
        if relabeled:
            debug_print(f"Smoothed {relabeled} isolated speaker flips at word level.")
        return relabeled

    def aggregate_to_utterances(self, max_gap: float = 1.0):
        """Aggregate word-level data into utterances.

        An utterance ends at sentence-final punctuation, at a speaker change, or
        after a silent gap longer than ``max_gap`` seconds. Punctuation alone is not
        enough: Whisper does not guarantee it (a transcript without any ``.?!`` used
        to collapse into a single utterance spanning the whole recording) and a
        sentence can run straight through a speaker change.

        :param max_gap: Silence in seconds that forces an utterance break. ``None``
            disables gap-based splitting.
        """
        if not self.combined_data:
            debug_print("No combined data available to aggregate.")
            return

        utterances = []
        current = None
        breaks = {"punctuation": 0, "speaker": 0, "gap": 0}

        debug_print("Aggregating words into utterances...")

        def flush():
            nonlocal current
            if current is None:
                return
            text = current["text"].strip()
            if text:
                majority_speaker, majority_count = max(
                    current["speakers"].items(), key=lambda item: item[1]
                )
                total_words = sum(current["speakers"].values())
                utterances.append({
                    "text": text,
                    "start_time": current["start_time"],
                    "end_time": current["end_time"],
                    "speaker": majority_speaker,
                    "confidence": round(majority_count / total_words, 2),
                })
            current = None

        for word_data in self.combined_data:
            word = word_data["word"]
            start_time = word_data["start_time"]
            end_time = word_data["end_time"]
            speaker = word_data.get("speaker", "UNKNOWN")

            if current is not None:
                if speaker != current["speaker"]:
                    breaks["speaker"] += 1
                    flush()
                elif max_gap is not None and start_time - current["end_time"] > max_gap:
                    breaks["gap"] += 1
                    flush()

            if current is None:
                current = {
                    "text": "",
                    "start_time": start_time,
                    "end_time": end_time,
                    "speaker": speaker,
                    "speakers": {},
                }

            current["text"] += ("" if current["text"] == "" else " ") + word
            current["end_time"] = max(current["end_time"], end_time)
            current["speakers"][speaker] = current["speakers"].get(speaker, 0) + 1

            if SENTENCE_ENDINGS.search(word):
                breaks["punctuation"] += 1
                flush()

        flush()

        self.combined_utterances = utterances
        debug_print(
            f"Aggregated {len(utterances)} utterances "
            f"(breaks: {breaks['punctuation']} punctuation, {breaks['speaker']} speaker, "
            f"{breaks['gap']} gap>{max_gap}s)."
        )

    def save_as_json(self, output_file="all_transcript_data.json"):
        """
        Save all transcript data to a JSON file.
        
        :param output_file: Path to the output JSON file.
        """
        if not self.combined_data:
            debug_print("No combined data available to save. Ensure 'combine_alignment_and_diarization' is run first.")
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
            debug_print(f"All transcript data successfully saved to '{output_file}'.")
        except Exception as e:
            print(f"Error saving JSON file: {e}")

    def save_as_text(self, output_file=None, include_speakers=True):
        """
        Save transcript text to a plain text file.
        
        The text is built from ``combined_utterances`` whenever those exist, so the
        ``.txt`` and the ``utterance_data`` in the ``.json`` describe the same
        transcript. Only when aggregation produced nothing does this fall back to the
        raw concatenation of chunk transcripts.

        :param output_file: Path to the output text file. If None, generates from JSON file path.
        :param include_speakers: If True and multiple speakers detected, include speaker labels in output.
        """
        utterances = [
            utterance for utterance in (self.combined_utterances or [])
            if (utterance.get('text') or '').strip()
        ]
        if not utterances and not self.transcript_text:
            debug_print("No transcript text available to save. Ensure transcription is complete.")
            return
        
        # Generate text file path from JSON file path if not provided
        if output_file is None:
            if self.transcription_file:
                # Replace .json extension with .txt, or add .txt if no extension
                output_file = os.path.splitext(self.transcription_file)[0] + ".txt"
            else:
                # Fallback: use audio file name
                base_name = os.path.splitext(self.name)[0]
                output_file = os.path.join(self.file_path, f"{base_name}_transcript.txt")
        
        try:
            # Label speakers when diarization actually distinguished more than one.
            distinct_speakers = {utterance.get('speaker', 'UNKNOWN') for utterance in utterances}
            label_speakers = bool(
                include_speakers
                and utterances
                and (len(distinct_speakers) > 1 or (self.num_speakers or 1) > 1)
            )

            with open(output_file, "w", encoding="utf-8") as f:
                if not utterances:
                    f.write(self.transcript_text)
                    debug_print(
                        f"No utterances available; wrote raw transcript text to '{output_file}'."
                    )
                elif label_speakers:
                    # Format with speaker labels: "SPEAKER_00: text here"
                    for utterance in utterances:
                        speaker = utterance.get('speaker', 'UNKNOWN')
                        f.write(f"{speaker}: {utterance['text'].strip()}\n")
                    debug_print(f"Transcript text with speaker labels saved to '{output_file}'.")
                else:
                    f.write(" ".join(utterance['text'].strip() for utterance in utterances))
                    debug_print(f"Transcript text successfully saved to '{output_file}'.")
            
            self.transcription_text_file = output_file
        except Exception as e:
            print(f"Error saving text file: {e}")

    def clear_audio_data(self):
        """
        Clear large audio data structures from memory to prevent memory accumulation.
        This is useful when processing many files in sequence.
        """
        self.audio = None
        self.sample_rate = None
        # Clear audio segments from chunks but keep metadata
        if self.chunks:
            for chunk in self.chunks:
                if hasattr(chunk, 'audio_segment'):
                    chunk.audio_segment = None
        # Note: We keep chunks list and other metadata for reference
        # but clear the actual audio data

    def __repr__(self):
        return f"AudioFile(file_name={self.name})"