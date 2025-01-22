"""
Speaker diarization module for identifying speakers in audio.

This module provides speaker diarization functionality using pyannote.audio,
enabling the identification and segmentation of different speakers in audio files.
Key features include:
- Automatic speaker detection and segmentation
- Support for known number of speakers or automatic detection
- Integration with pyannote's state-of-the-art diarization models
- Configurable clustering and segmentation parameters
- GPU acceleration support for faster processing

The module uses pyannote.audio's pipeline for robust speaker diarization,
supporting both fixed and variable numbers of speakers.
"""
import torch
from typing import Dict, Optional
from pyannote.audio import Pipeline as DiarizationPipeline
from .audio import AudioFile
from .utils import get_device
import tempfile


class SpeakerDiarizer:
    """
    Handles speaker diarization of audio files using pyannote.audio.
    
    This class provides functionality to identify and segment different speakers
    in audio recordings. It uses pyannote.audio's speaker diarization pipeline
    to perform:
    - Speaker change detection
    - Speech activity detection
    - Speaker embedding extraction
    - Speaker clustering
    
    The diarizer supports both scenarios where the number of speakers is known
    in advance and where it needs to be automatically determined. It can be
    configured through various parameters to optimize for different use cases.

    Attributes:
        device (torch.device): Compute device for model inference
        pipeline (DiarizationPipeline): Loaded diarization pipeline
        parameters (Dict): Configuration parameters for diarization
        model (str): Name/path of the diarization model
    """
    def __init__(self, hf_token: str, parameters: Dict, device = None, model = "pyannote/speaker-diarization-3.1"):
        """
        Initializes the SpeakerDiarizer.
        
        Args:
            hf_token (str): Hugging Face token for accessing diarization models
            parameters (Dict): Parameters for the diarization pipeline including:
                - segmentation: Parameters for speech activity detection
                - clustering: Parameters for speaker clustering
            device (Optional[torch.device]): Device to use for processing
            model (str): Model identifier for the diarization pipeline
        """
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.diarization_pipeline = DiarizationPipeline.from_pretrained(
            model,
            use_auth_token=hf_token
        )
        self.model = model
        print("Initializing SpeakerDiarizer with parameters...")
        self.parameters = parameters
        self.diarization_pipeline.instantiate(parameters)
        self.diarization_pipeline.to(self.device)
        print("Initialized SpeakerDiarizer successfully.")

    def diarize(self, audio_file: AudioFile, num_speakers: int = None):
        """
        Performs speaker diarization on the given audio file.
        
        :param audio_file: AudioFile instance containing audio data.
        :param num_speakers: Expected number of speakers.
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
            raise
            
        audio_file.register_model("Speaker Diarization", {
            "model": self.model,
            "device": str(self.device),
            "parameters": self.parameters,
            "speakers": num_speakers if num_speakers else "not specified"
        }) 