"""
Audio transcription module using Whisper for speech-to-text.

This module provides speech-to-text functionality using OpenAI's Whisper model,
offering high-quality transcription across multiple languages. Key features include:
- Support for multiple Whisper model variants (tiny to large)
- Multi-language transcription support
- Word-level timing information
- Batch processing of audio chunks
- GPU acceleration for faster processing
- Confidence scores for transcriptions

The module uses the Hugging Face Transformers implementation of Whisper,
providing both transcription and timing information for each word.
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
    """
    Handles speech-to-text transcription using OpenAI's Whisper model.
    
    This class provides functionality for transcribing audio into text using
    Whisper, a state-of-the-art speech recognition model. It handles:
    - Audio format conversion and preprocessing
    - Model loading and optimization
    - Transcription with word-level timing
    - Language-specific processing
    - Batch processing of audio chunks
    
    The transcriber supports multiple Whisper model variants, from tiny models
    suitable for quick transcription to large models for maximum accuracy.
    It automatically handles device placement and optimization.

    Attributes:
        model_name (str): Name/path of the Whisper model
        language (str): Language code for transcription
        device (torch.device): Compute device for model inference
        pipe (pipeline): Loaded Whisper pipeline
        chunk_length (int): Maximum chunk length in seconds
        stride_length (int): Overlap between chunks in seconds
    """
    
    def __init__(self, model_name: str = "openai/whisper-large", 
                 language: str = "de", device: Optional[torch.device] = None):
        """
        Initialize the transcriber with specified model and language.
        
        Args:
            model_name (str): Name of the Whisper model to use. Options include:
                - openai/whisper-tiny: Fastest but least accurate
                - openai/whisper-base: Good balance for shorter content
                - openai/whisper-small: Better accuracy, still reasonably fast
                - openai/whisper-medium: High accuracy, slower
                - openai/whisper-large: Best accuracy, slowest
            language (str): Language code for transcription (e.g., 'de', 'en')
            device (Optional[torch.device]): Device to use for inference
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
        
        # Skip if chunk is too short or silent
        if len(audio_data) < 10:  # Arbitrary minimum length
            print("  ✗ Skipping: too short")
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
        
        # Print concise output
        duration = len(audio_data)/16000
        print(f"\n  ✓ {len(chunk.whisper_alignments)} words ({duration:.1f}s)")
        print(f"  → {chunk.transcript}")
        
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