from typing import Optional, List, Dict, Any
from .transcript import Transcript

class TranscriptController:
    """Controller class to handle transcript data operations."""
    
    def __init__(self, transcript: Optional[Transcript] = None):
        self.transcript = transcript

    def set_transcript(self, transcript: Transcript) -> None:
        """Set the transcript instance to work with."""
        self.transcript = transcript

    def update_word_boundaries(self, idx: int, start_time: float = None, end_time: float = None) -> None:
        """Update word boundaries at the given index."""
        if not self.transcript or idx >= len(self.transcript.combined_data):
            return
        
        word_data = self.transcript.combined_data[idx]
        if start_time is not None:
            word_data['start_time'] = start_time
        if end_time is not None:
            word_data['end_time'] = end_time

    def merge_speakers(self, from_speaker: str, to_speaker: str) -> None:
        """Merge all words from one speaker to another."""
        if not self.transcript:
            return
        
        for word_data in self.transcript.combined_data:
            if word_data.get('speaker') == from_speaker:
                word_data['speaker'] = to_speaker

    def split_speaker(self, speaker: str, word_indices: List[int], new_speaker: str) -> None:
        """Split specified words from one speaker to a new speaker label."""
        if not self.transcript:
            return
        
        for idx in word_indices:
            if 0 <= idx < len(self.transcript.combined_data):
                word_data = self.transcript.combined_data[idx]
                if word_data.get('speaker') == speaker:
                    word_data['speaker'] = new_speaker

    def get_speakers(self) -> List[str]:
        """Get list of unique speaker labels."""
        if not self.transcript:
            return []
        
        return list(set(
            word.get('speaker', '') 
            for word in self.transcript.combined_data 
            if word.get('speaker')
        ))

    def get_word_data(self, idx: int) -> Optional[Dict[str, Any]]:
        """Get word data at the given index."""
        if not self.transcript or idx >= len(self.transcript.combined_data):
            return None
        return self.transcript.combined_data[idx] 