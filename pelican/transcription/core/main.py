"""
Main processing script for audio transcription pipeline.

This module orchestrates the entire audio processing pipeline, including:
- Audio file loading and normalization
- Speech recognition using Whisper
- Forced alignment for precise word timing
- Speaker diarization
- Transcript generation and processing

The pipeline is configurable through various parameters and supports
progress tracking through callbacks. It handles all the necessary setup
and cleanup of models and resources.

Key components:
- AudioFile: Handles audio processing and chunking
- AudioTranscriber: Manages speech recognition
- ForcedAligner: Provides precise word-level timing
- SpeakerDiarizer: Identifies and labels different speakers
- Transcript: Manages and combines all transcription data
"""
import os
import warnings
import logging
from pathlib import Path
from typing import List, Dict, Optional, Callable
import torch
import argparse

# Suppress all warnings
warnings.filterwarnings('ignore')

# Set logging level to ERROR only
logging.getLogger().setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Suppress huggingface/tokenizers warning

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
    num_speakers: Optional[int] = None,
    model: str = "openai/whisper-large",
    language: str = "de",
    device: Optional[torch.device] = None,
    pause_threshold: float = 2.0,
    max_utterance_duration: float = 30.0,
    alignment_source: str = "forced_alignments",
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
    },
    progress_callback: Optional[Callable[[str, float, str], None]] = None
) -> str:
    """
    Process an audio file through the complete transcription pipeline.
    
    This function orchestrates the entire processing pipeline, including:
    1. Audio loading and normalization
    2. Chunking based on silence detection
    3. Speech recognition with Whisper
    4. Forced alignment for precise word timing
    5. Speaker diarization
    6. Combining all data into a structured transcript
    
    Args:
        file_path (str): Path to the input audio file
        hf_token (str): HuggingFace token for accessing models
        output_dir (str): Directory to save output files
        num_speakers (Optional[int]): Expected number of speakers (None for auto-detection)
        model (str): Whisper model to use for transcription
        language (str): Language code (e.g., 'de' for German)
        device (Optional[torch.device]): Compute device to use
        pause_threshold (float): Minimum pause duration for utterance splitting
        max_utterance_duration (float): Maximum duration of a single utterance
        alignment_source (str): Which alignment to use ('whisper_alignments' or 'forced_alignments')
        diarizer_params (Dict): Parameters for speaker diarization
        silence_params (Dict): Parameters for silence-based audio chunking
        progress_callback (Optional[Callable]): Function to call for progress updates
    
    Returns:
        str: Path to the output JSON file containing all transcript data
    
    Raises:
        ValueError: If input parameters are invalid
        RuntimeError: If processing fails at any stage
    """
    def update_progress(step: str, progress: float = 0, message: str = ""):
        if progress_callback:
            progress_callback(step, progress, message)
        else:
            if message:
                print(message)

    device = device if device is not None else get_device()
    update_progress("init", 0, f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize components
    update_progress("init", 0.1, "1. Initializing components...")
    transcriber = AudioTranscriber(
        model_name=model,
        language=language,
        device=device
    )
    
    # Aligner must skip MPS
    aligner = ForcedAligner(device=get_device(skip_mps=True))
    
    # Diarizer can use MPS
    diarizer = SpeakerDiarizer(hf_token, diarizer_params, device=get_device(skip_mps=False))
    
    # Load and preprocess audio
    update_progress("preprocessing", 0.2, "2. Loading and preprocessing audio...")
    audio_file = AudioFile(file_path)
    audio_file.rms_normalization()
    
    # Split audio into chunks
    update_progress("chunking", 0.3, "3. Splitting audio into chunks...")
    audio_file.split_on_silence(**silence_params)
    
    # Transcribe chunks
    update_progress("transcribing", 0.4, "4. Transcribing audio chunks...")
    total_chunks = len(audio_file.chunks)
    for idx, chunk in enumerate(audio_file.chunks, 1):
        update_progress("transcribing", 0.4 + (0.2 * idx/total_chunks), f"Chunk {idx}/{total_chunks}")
        result = transcriber.transcribe_chunk(chunk)
    
    # Perform forced alignment
    update_progress("aligning", 0.6, "5. Performing forced alignment...")
    aligner.align(audio_file)
    
    # Perform speaker diarization
    update_progress("diarizing", 0.7, "6. Performing speaker diarization...")
    diarizer.diarize(audio_file, num_speakers)
    
    # Create and process transcript
    update_progress("processing", 0.8, "7. Processing transcript...")
    transcript = Transcript(audio_file)
    update_progress("processing", 0.85, f"Using {alignment_source} for word timing")
    transcript.combine_alignment_and_diarization(alignment_source)
    transcript.aggregate_to_utterances(pause_threshold=pause_threshold, 
                                     max_duration=max_utterance_duration)
    
    # Save results
    update_progress("saving", 0.9, "Saving results...")
    output_file = os.path.join(output_dir, f"{Path(file_path).stem}_transcript.json")
    transcript.save_as_json(output_file)
    
    update_progress("complete", 1.0, f"Processing complete! Output saved to: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Process an audio file through the transcription pipeline.")
    parser.add_argument("audio_file", help="Path to the audio file to process")
    parser.add_argument("--hf-token", required=True, help="HuggingFace token for diarization model")
    parser.add_argument("--output-dir", default="output", help="Directory to save output files")
    parser.add_argument("--num-speakers", type=int, default=None, 
                       help="Number of speakers to constrain diarization (optional, model will estimate if not provided)")
    parser.add_argument("--model", default="openai/whisper-large", help="Whisper model to use")
    parser.add_argument("--language", default="de", help="Language code (e.g., 'de' for German)")
    parser.add_argument("--pause-threshold", type=float, default=2.0, 
                       help="Minimum pause duration (in seconds) to split utterances")
    parser.add_argument("--max-utterance-duration", type=float, default=30.0,
                       help="Maximum duration (in seconds) for a single utterance")
    
    args = parser.parse_args()
    
    # Process the audio file
    output_path = process_audio(
        file_path=args.audio_file,
        hf_token=args.hf_token,
        output_dir=args.output_dir,
        num_speakers=args.num_speakers,
        model=args.model,
        language=args.language,
        pause_threshold=args.pause_threshold,
        max_utterance_duration=args.max_utterance_duration
    )
    
    print(f"\nProcessing complete! Output saved to: {output_path}")


if __name__ == "__main__":
    main() 