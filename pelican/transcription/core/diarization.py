"""
Speaker diarization module for identifying speakers in audio.
"""
import torch
from typing import Dict, Optional
from pyannote.audio import Pipeline as DiarizationPipeline
from .audio import AudioFile
from .utils import get_device


class SpeakerDiarizer:
    """Handles speaker diarization of audio files."""
    
    def __init__(self, hf_token: str, parameters: Dict, 
                 model: str = "pyannote/speaker-diarization-3.1",
                 device: Optional[torch.device] = None):
        """
        Initialize the speaker diarizer.
        
        Args:
            hf_token: Hugging Face token for accessing diarization models
            parameters: Parameters for the diarization pipeline
            model: Model name/path for diarization
            device: Device to use for inference (default: best available)
        """
        self.device = device if device is not None else get_device()
        print(f"Initializing SpeakerDiarizer on device: {self.device}")

        self.diarization_pipeline = DiarizationPipeline.from_pretrained(
            model,
            use_auth_token=hf_token
        )
        self.model = model
        print("Initializing SpeakerDiarizer with parameters...")
        self.parameters = parameters
        self.diarization_pipeline.instantiate(parameters)
        self.diarization_pipeline.to(self.device)

    def diarize(self, audio_file: AudioFile, num_speakers: Optional[int] = None):
        """
        Perform speaker diarization on the given audio file.
        
        Args:
            audio_file: AudioFile instance containing audio data
            num_speakers: Expected number of speakers (optional)
        """
        print("Starting speaker diarization...")
        try:
            if num_speakers is not None:
                diarization_result = self.diarization_pipeline(
                    audio_file.normalized_path,
                    num_speakers=num_speakers
                )
                print(f"Diarization completed with {num_speakers} speakers.")
            else:
                diarization_result = self.diarization_pipeline(
                    audio_file.normalized_path
                )
                print("Diarization completed without specifying number of speakers.")

            # Extract speaker segments
            audio_file.speaker_segments = []
            for segment, _, speaker in diarization_result.itertracks(yield_label=True):
                audio_file.speaker_segments.append({
                    "start": segment.start,
                    "end": segment.end,
                    "speaker": speaker
                })
            print(f"Detected {len(audio_file.speaker_segments)} speaker segments.")
        except Exception as e:
            print(f"An error occurred during diarization: {e}")
            
        audio_file.register_model("Speaker Diarization", {
            "model": self.model,
            "device": str(self.device),
            "parameters": self.parameters,
            "speakers": num_speakers if num_speakers else "not specified"
        }) 