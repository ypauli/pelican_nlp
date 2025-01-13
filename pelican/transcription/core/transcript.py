"""
Transcript management module for handling transcription data.
"""
import json
import re
from typing import Dict, List, Optional, Any
from pathlib import Path
from .audio import AudioFile


class Transcript:
    """Manages transcription data and operations."""
    
    def __init__(self, audio_file: Optional[AudioFile] = None, json_data: Optional[Dict] = None):
        """
        Initialize the Transcript class.
        
        Args:
            audio_file: AudioFile object to initialize from
            json_data: Dictionary loaded from a JSON file
        """
        if audio_file:
            self.audio_file_path = audio_file.file_path
            self.transcript_text = audio_file.transcript_text
            self.whisper_alignments = audio_file.whisper_alignments
            self.forced_alignments = audio_file.forced_alignments
            self.speaker_segments = audio_file.speaker_segments
            self.combined_data = []
            self.combined_utterances = []
            self.metadata = audio_file.metadata
        elif json_data:
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
                self._finalize_utterance(current_utterance, utterances)
                current_utterance = self._create_empty_utterance()

        # Handle any remaining words as the last utterance
        if current_utterance["text"]:
            self._finalize_utterance(current_utterance, utterances)

        self.combined_utterances = utterances
        print("Aggregated utterances from combined data.")

    def _create_empty_utterance(self) -> Dict[str, Any]:
        """Create an empty utterance dictionary."""
        return {
            "text": "",
            "start_time": None,
            "end_time": None,
            "speakers": {}
        }

    def _finalize_utterance(self, utterance: Dict[str, Any], utterances: List[Dict[str, Any]]):
        """
        Finalize an utterance and add it to the list.
        
        Args:
            utterance: Current utterance dictionary
            utterances: List of completed utterances
        """
        majority_speaker, majority_count = max(
            utterance["speakers"].items(), key=lambda item: item[1]
        )
        total_words = sum(utterance["speakers"].values())
        confidence = round(majority_count / total_words, 2)

        utterances.append({
            "text": utterance["text"],
            "start_time": utterance["start_time"],
            "end_time": utterance["end_time"],
            "speaker": majority_speaker,
            "confidence": confidence,
        })

    def combine_alignment_and_diarization(self, alignment_source: str):
        """
        Combine alignment and diarization data by assigning speaker labels to each word.
        
        Args:
            alignment_source: The alignment data to use ('whisper_alignments' or 'forced_alignments')
        """
        if alignment_source not in ['whisper_alignments', 'forced_alignments']:
            raise ValueError("Invalid alignment_source. Choose 'whisper_alignments' or 'forced_alignments'.")

        alignment = getattr(self, alignment_source, None)
        if alignment is None:
            raise ValueError(f"The alignment source '{alignment_source}' does not exist.")

        if not self.speaker_segments:
            print("No speaker segments available. All words will be labeled as 'UNKNOWN'.")
            self.combined_data = [{**word, 'speaker': 'UNKNOWN'} for word in alignment]
            return

        combined = []
        seg_idx = 0
        num_segments = len(self.speaker_segments)

        for word in alignment:
            word_start = word['start_time']
            word_end = word['end_time']
            word_duration = max(1e-6, word_end - word_start)

            speaker_overlap = {}

            # Advance segments that have ended before the word starts
            while seg_idx < num_segments and self.speaker_segments[seg_idx]['end'] < word_start:
                seg_idx += 1

            temp_idx = seg_idx
            while temp_idx < num_segments and self.speaker_segments[temp_idx]['start'] < word_end:
                seg = self.speaker_segments[temp_idx]
                overlap = self._calculate_overlap(word_start, word_end, seg['start'], seg['end'])
                
                if overlap > 0:
                    speaker_overlap[seg['speaker']] = speaker_overlap.get(seg['speaker'], 0.0) + overlap

                temp_idx += 1

            assigned_speaker = max(speaker_overlap, key=speaker_overlap.get) if speaker_overlap else 'UNKNOWN'
            word_with_speaker = word.copy()
            word_with_speaker['speaker'] = assigned_speaker
            combined.append(word_with_speaker)

        self.combined_data = combined
        self.metadata["alignment_source"] = alignment_source
        print(f"Combined alignment and diarization data with {len(self.combined_data)} entries.")

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