"""
Transcript management module for handling transcription data.

This module provides functionality for managing and manipulating transcription data, including:
- Combining word-level alignments with speaker diarization
- Aggregating words into meaningful utterances
- Managing different types of alignments (Whisper and forced)
- Serializing and deserializing transcript data to/from JSON
- Handling transcript metadata and processing history

The module works closely with the audio processing module to maintain synchronization
between audio segments and their corresponding transcriptions.
"""
import json
import re
from typing import Dict, List, Optional, Any
from pathlib import Path
from .audio import AudioFile


class Transcript:
    """
    Manages transcription data and operations.
    
    This class serves as the central point for managing transcription data,
    combining different sources of information (transcription text, word alignments,
    and speaker diarization) into a coherent structure.

    Key features:
    - Loads transcription data from either AudioFile objects or JSON files
    - Combines word-level alignments with speaker information
    - Aggregates words into meaningful utterances based on pauses and duration
    - Provides methods for data manipulation and export
    
    Attributes:
        audio_file (Optional[AudioFile]): Reference to associated audio file
        audio_file_path (str): Path to the source audio file
        transcript_text (str): Raw transcription text
        whisper_alignments (List): Word-level alignments from Whisper
        forced_alignments (List): Word-level alignments from forced alignment
        speaker_segments (List): Speaker diarization segments
        combined_data (List): Processed data combining alignments and speakers
        combined_utterances (List): Utterance-level aggregated data
        metadata (Dict): Processing metadata and parameters
    """
    
    def __init__(self, audio_file: Optional[AudioFile] = None, json_data: Optional[Dict] = None):
        """
        Initialize the Transcript class.
        
        Args:
            audio_file: AudioFile object to initialize from
            json_data: Dictionary loaded from a JSON file
        """
        if audio_file:
            self.audio_file = audio_file  # Store reference to audio_file
            self.audio_file_path = audio_file.file_path
            self.transcript_text = audio_file.transcript_text
            self.whisper_alignments = audio_file.whisper_alignments
            self.forced_alignments = audio_file.forced_alignments
            self.speaker_segments = audio_file.speaker_segments
            self.combined_data = []
            self.combined_utterances = []
            self.metadata = audio_file.metadata
        elif json_data:
            self.audio_file = None  # No audio_file for JSON data
            self.audio_file_path = json_data["audio_file_path"]
            self.metadata = json_data["metadata"]
            self.transcript_text = json_data.get("transcript_text", "")
            self.whisper_alignments = json_data.get("whisper_alignments", [])
            self.forced_alignments = json_data.get("forced_alignments", [])
            self.speaker_segments = json_data.get("speaker_segments", [])
            self.combined_data = json_data.get("combined_data", [])
            self.combined_utterances = json_data.get("utterance_data", [])
        else:
            raise ValueError("Either an AudioFile object or JSON data must be provided.")

    @classmethod
    def from_json_file(cls, json_file: str) -> 'Transcript':
        """
        Create a Transcript instance from a JSON file.
        
        Args:
            json_file: Path to the JSON file
            
        Returns:
            Transcript instance
        """
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                json_data = json.load(f)
            print(f"Loaded transcript data from '{json_file}'.")
            return cls(json_data=json_data)
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            raise

    def get_word_data(self) -> List[Dict[str, Any]]:
        """
        Get the combined word-level data sorted by start time.
        
        Returns:
            List of word data dictionaries sorted by start time
        """
        if not self.combined_data:
            print("No combined data available. Run combine_alignment_and_diarization first.")
            return []
            
        return sorted(self.combined_data, key=lambda x: x["start_time"])

    def aggregate_to_utterances(self, pause_threshold: float = 2.0, max_duration: float = 30.0) -> None:
        """
        Aggregate word-level data into utterances based on sentence endings, pauses, and maximum duration.
        
        Args:
            pause_threshold: Minimum pause duration (in seconds) to split utterances
            max_duration: Maximum duration (in seconds) for a single utterance
        """
        print("\nAggregating words into utterances...")
        
        # Get word data sorted by start time
        word_data = self.get_word_data()
        if not word_data:
            print("  ✗ No word data available")
            return
            
        # Initialize variables
        current_utterance = []
        self.combined_utterances = []  # Store in combined_utterances instead of utterances
        total_words = len(word_data)
        current_word = 0
        
        for i, word in enumerate(word_data):
            current_word += 1
            current_utterance.append(word)
            
            # Check if we should finalize the current utterance
            should_split = False
            split_reason = None
            
            # Check for sentence ending
            if word["word"][-1] in ".!?":
                should_split = True
                split_reason = "sentence"
                
            # Check for long pause (if not the last word)
            elif i < len(word_data) - 1:
                next_word = word_data[i + 1]
                pause_duration = next_word["start_time"] - word["end_time"]
                if pause_duration >= pause_threshold:
                    should_split = True
                    split_reason = f"pause ({pause_duration:.1f}s)"
                    
            # Check for maximum duration
            if current_utterance:
                utterance_duration = current_utterance[-1]["end_time"] - current_utterance[0]["start_time"]
                if utterance_duration >= max_duration:
                    should_split = True
                    split_reason = f"duration ({utterance_duration:.1f}s)"
            
            # Split if needed or if this is the last word
            if should_split or i == len(word_data) - 1:
                self._finalize_utterance(current_utterance)
                if split_reason:
                    print(f"  → Split at word {current_word}/{total_words} ({split_reason})")
                current_utterance = []
        
        # Print summary
        print(f"\n✓ Created {len(self.combined_utterances)} utterances from {total_words} words")
        print(f"  - Average words per utterance: {total_words / len(self.combined_utterances):.1f}")
        
        # Print speaker statistics
        print("\nSpeaker Distribution:")
        speaker_counts = {}
        for utt in self.combined_utterances:
            speaker = utt["speaker"]
            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
        
        for speaker, count in speaker_counts.items():
            percentage = (count / len(self.combined_utterances)) * 100
            print(f"  {speaker}: {count} ({percentage:.1f}%)")

    def _create_empty_utterance(self) -> Dict[str, Any]:
        """Create an empty utterance dictionary."""
        return {
            "text": "",
            "start_time": None,
            "end_time": None,
            "speakers": {}
        }

    def _finalize_utterance(self, word_list: List[Dict[str, Any]]) -> None:
        """
        Finalize an utterance and add it to the utterances list.
        
        Args:
            word_list: List of words in the utterance
        """
        if not word_list:
            return

        # Count speakers
        speaker_counts = {}
        for word in word_list:
            speaker = word["speaker"]
            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1

        # Get majority speaker
        majority_speaker = max(speaker_counts.items(), key=lambda x: x[1])[0]
        total_words = len(word_list)
        confidence = round(max(speaker_counts.values()) / total_words, 2)

        # Create utterance text
        text = " ".join(word["word"] for word in word_list)

        # Add to combined_utterances instead of utterances
        self.combined_utterances.append({
            "text": text,
            "start_time": word_list[0]["start_time"],
            "end_time": word_list[-1]["end_time"],
            "speaker": majority_speaker,
            "confidence": confidence,
        })

    def combine_alignment_and_diarization(self, alignment_source: str = "forced_alignments") -> None:
        """
        Combine word alignments with speaker diarization data.
        
        Args:
            alignment_source: Which alignments to use ('whisper_alignments' or 'forced_alignments')
            
        Raises:
            ValueError: If the requested alignment source is invalid or contains no alignments
        """
        if alignment_source not in ["whisper_alignments", "forced_alignments"]:
            raise ValueError("alignment_source must be either 'whisper_alignments' or 'forced_alignments'")
            
        # Get the selected alignments
        alignments = []
        
        # Collect alignments from all chunks
        for chunk in self.audio_file.chunks:
            chunk_alignments = (getattr(chunk, alignment_source) or []).copy()
            alignments.extend(chunk_alignments)
            
        if not alignments:
            raise ValueError(
                f"No alignments found for source '{alignment_source}'. "
                f"Make sure the requested alignment type is available before processing."
            )
            
        print(f"Using {alignment_source} for {len(alignments)} words")
        
        # Sort alignments by start time
        alignments.sort(key=lambda x: float(x["start_time"]))
        
        # Find speaker for each word based on timing
        self.combined_data = []
        for alignment in alignments:
            word_start = float(alignment["start_time"])
            word_end = float(alignment["end_time"])
            
            # Find overlapping speaker segments
            word_speakers = {}
            for segment in self.speaker_segments:
                segment_start = float(segment["start"])
                segment_end = float(segment["end"])
                
                # Check for overlap
                if word_end > segment_start and word_start < segment_end:
                    overlap = min(word_end, segment_end) - max(word_start, segment_start)
                    speaker = segment["speaker"]
                    word_speakers[speaker] = word_speakers.get(speaker, 0) + overlap
            
            # Get speaker with maximum overlap
            if word_speakers:
                speaker = max(word_speakers.items(), key=lambda x: x[1])[0]
            else:
                speaker = "UNKNOWN"
            
            # Add combined word data
            self.combined_data.append({
                "word": alignment["word"],
                "start_time": word_start,
                "end_time": word_end,
                "speaker": speaker
            })
        
        print(f"Combined {len(self.combined_data)} words with speaker information")

    def _calculate_overlap(self, word_start: float, word_end: float, 
                         seg_start: float, seg_end: float) -> float:
        """
        Calculate the overlap duration between a word and a speaker segment.
        
        Args:
            word_start: Word start time
            word_end: Word end time
            seg_start: Segment start time
            seg_end: Segment end time
            
        Returns:
            Overlap duration in seconds
        """
        overlap_start = max(word_start, seg_start)
        overlap_end = min(word_end, seg_end)
        return max(0.0, overlap_end - overlap_start)

    def save_as_json(self, output_file: str = "all_transcript_data.json"):
        """
        Save all transcript data to a JSON file.
        
        Args:
            output_file: Path to the output JSON file
        """
        if not self.combined_data:
            print("No combined data available to save. Run combine_alignment_and_diarization first.")
            return

        data = {
            "audio_file_path": self.audio_file_path,
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