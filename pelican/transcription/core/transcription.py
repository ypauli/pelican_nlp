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
import librosa


class AudioTranscriber:
    """Handles speech-to-text transcription using Whisper."""
    
    def __init__(self, model_name: str = "openai/whisper-large", 
                 language: str = "de", device: Optional[torch.device] = None):
        """
        Initialize the transcriber.
        
        Args:
            model_name: Name of the Whisper model to use (default: whisper-large for best accuracy)
            language: Language code (e.g., 'de' for German, 'en' for English)
            device: Device to use for inference (default: best available)
        """
        self.model_name = model_name
        self.language = language
        self.device = device if device is not None else get_device()
        print(f"Initializing AudioTranscriber on device: {self.device}")
        print(f"Using language: {language}")
        
        self.transcriber = pipeline(
            task="automatic-speech-recognition",
            model=model_name,
            chunk_length_s=30,
            return_timestamps=True,
            device=self.device,
            generate_kwargs={"language": language, "task": "transcribe"}
        )

    def transcribe_chunk(self, chunk: Chunk) -> Dict[str, Any]:
        """
        Transcribe an audio chunk and get word-level alignments.
        
        Args:
            chunk: Audio chunk to transcribe
            
        Returns:
            Dictionary containing transcription and alignments
        """
        # Export chunk to WAV format in memory
        with io.BytesIO() as wav_io:
            chunk.audio_segment.export(wav_io, format='wav')
            wav_io.seek(0)
            # Load as numpy array with librosa to handle resampling
            audio_data, sample_rate = librosa.load(wav_io, sr=16000)  # Whisper expects 16kHz
            
        # Print debug info
        print(f"  Audio chunk info:")
        print(f"    - Duration: {len(audio_data)/16000:.2f}s")
        print(f"    - Sample rate: {sample_rate}Hz")
        print(f"    - Shape: {audio_data.shape}")
        print(f"    - Range: [{audio_data.min():.2f}, {audio_data.max():.2f}]")
            
        # Skip if chunk is too short or silent
        if len(audio_data) < 1000:  # Arbitrary minimum length
            print("    - Skipping: too short")
            chunk.transcript = ""
            chunk.whisper_alignments = []
            return {
                "transcript": "",
                "alignments": []
            }
            
        # Get transcription with timestamps
        result = self.transcriber(
            audio_data,
            return_timestamps="word"
        )
        
        # Process results
        chunk.transcript = result["text"].strip()
        chunk.whisper_alignments = self._process_alignments(result.get("chunks", []), chunk.start_time)
        
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