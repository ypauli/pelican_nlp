"""
Audio transcription module using Whisper for speech-to-text.
"""
import io
import torch
import numpy as np
from transformers import pipeline
from typing import List, Dict, Any, Optional
from .audio import Chunk
from .utils import get_device


class AudioTranscriber:
    """Handles speech-to-text transcription using Whisper."""
    
    def __init__(self, model_name: str = "openai/whisper-base.en", device: Optional[torch.device] = None):
        """
        Initialize the transcriber.
        
        Args:
            model_name: Name of the Whisper model to use
            device: Device to use for inference (default: best available)
        """
        self.model_name = model_name
        self.device = device if device is not None else get_device()
        print(f"Initializing AudioTranscriber on device: {self.device}")
        
        self.transcriber = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            chunk_length_s=30,
            return_timestamps=True,
            device=self.device
        )

    def transcribe_chunk(self, chunk: Chunk) -> Dict[str, Any]:
        """
        Transcribe an audio chunk and get word-level alignments.
        
        Args:
            chunk: Audio chunk to transcribe
            
        Returns:
            Dictionary containing transcription and alignments
        """
        # Get audio data as numpy array
        audio_data = np.array(chunk.audio_segment.get_array_of_samples())
        
        # Get transcription with timestamps
        result = self.transcriber(
            audio_data,
            return_timestamps="word"
        )
        
        # Process results
        chunk.transcript = result["text"]
        chunk.whisper_alignments = self._process_alignments(result["chunks"], chunk.start_time)
        
        return {
            "transcript": chunk.transcript,
            "alignments": chunk.whisper_alignments
        }

    def _process_alignments(self, chunks: List[Dict[str, Any]], 
                          base_time: float) -> List[Dict[str, Any]]:
        """
        Process word-level alignments from Whisper output.
        
        Args:
            chunks: List of word chunks from Whisper
            base_time: Base time offset for the chunk
            
        Returns:
            List of processed word alignments
        """
        alignments = []
        for chunk in chunks:
            if isinstance(chunk, dict) and "text" in chunk:
                alignment = {
                    "word": chunk["text"].strip(),
                    "start_time": base_time + chunk.get("start", 0),
                    "end_time": base_time + chunk.get("end", 0)
                }
                alignments.append(alignment)
        return alignments