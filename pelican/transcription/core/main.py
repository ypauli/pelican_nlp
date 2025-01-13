"""
Main processing script for audio transcription pipeline.
"""
import os
from pathlib import Path
from typing import List, Dict
import torch

from .audio import AudioFile
from .transcription import AudioTranscriber
from .alignment import ForcedAligner
from .diarization import SpeakerDiarizer
from .transcript import Transcript
from .utils import get_device


def process_audio(
    file_path: str,
    hf_token: str,
    output_dir: str = "output",
    num_speakers: int = 2,
    device: torch.device = None,
    diarizer_params: Dict = {
        "segmentation": {
            "min_duration_off": 0.0,
        },
        "clustering": {
            "method": "centroid",
            "min_cluster_size": 12,
            "threshold": 0.8,
        }
    },
    silence_params: Dict = {
        "min_silence_len": 1000,  # ms
        "silence_thresh": -30,    # dBFS
        "min_length": 30000,      # ms
        "max_length": 180000      # ms
    }
) -> str:
    """
    Process a single audio file through the complete pipeline.
    
    Args:
        file_path: Path to the audio file
        hf_token: HuggingFace token for diarization model
        output_dir: Directory to save output files
        num_speakers: Expected number of speakers
        device: Torch device to use (will use best available if None)
        diarizer_params: Parameters for speaker diarization
        silence_params: Parameters for silence-based audio splitting
        
    Returns:
        Path to the output JSON file
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get device if not provided
    if device is None:
        device = get_device()
    print(f"\nUsing device: {device}")
    
    # Initialize components
    print("\n1. Initializing components...")
    transcriber = AudioTranscriber(device=device)
    aligner = ForcedAligner(device=device)
    diarizer = SpeakerDiarizer(hf_token, diarizer_params, device=device)
    
    # Load and preprocess audio
    print("\n2. Loading and preprocessing audio...")
    audio_file = AudioFile(file_path)
    audio_file.rms_normalization()
    
    # Split audio into chunks
    print("\n3. Splitting audio into chunks...")
    audio_file.split_on_silence(**silence_params)
    
    # Transcribe chunks
    print("\n4. Transcribing audio chunks...")
    for idx, chunk in enumerate(audio_file.chunks, 1):
        result = transcriber.transcribe_chunk(chunk)
        print(f"Chunk {idx} transcript: {result['transcript']}")
    
    # Perform forced alignment
    print("\n5. Performing forced alignment...")
    aligner.align(audio_file)
    
    # Perform speaker diarization
    print("\n6. Performing speaker diarization...")
    diarizer.diarize(audio_file, num_speakers)
    
    # Create and process transcript
    print("\n7. Processing transcript...")
    transcript = Transcript(audio_file)
    transcript.combine_alignment_and_diarization("whisper_alignments")
    transcript.aggregate_to_utterances()
    
    # Save results
    output_file = os.path.join(output_dir, f"{Path(file_path).stem}_transcript.json")
    transcript.save_as_json(output_file)
    
    return output_file


if __name__ == "__main__":
    # Example usage
    AUDIO_FILE = "example_audio/holmes_control_nova.wav"
    HF_TOKEN = "YOUR_HUGGINGFACE_TOKEN"  # Replace with your token
    
    # Get the best available device
    DEVICE = get_device()
    print(f"Using device: {DEVICE}")
    
    output_path = process_audio(
        file_path=AUDIO_FILE,
        hf_token=HF_TOKEN,
        num_speakers=2,
        device=DEVICE
    )
    print(f"\nProcessing complete! Output saved to: {output_path}") 