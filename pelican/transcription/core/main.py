"""
Main processing script for audio transcription pipeline.
"""
import os
from pathlib import Path
from typing import List, Dict, Optional
import torch
import argparse

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
    }
) -> str:
    """
    Process an audio file through the transcription pipeline.
    
    Args:
        file_path: Path to the audio file
        hf_token: HuggingFace token for diarization model
        output_dir: Directory to save output files
        num_speakers: Optional number of speakers to constrain diarization (default: None, let model decide)
        model: Whisper model to use for transcription
        language: Language code (e.g., 'de' for German)
        device: Device to use for processing (default: best available)
        pause_threshold: Minimum pause duration (in seconds) to split utterances
        max_utterance_duration: Maximum duration (in seconds) for a single utterance
        alignment_source: Which alignments to use ('whisper_alignments' or 'forced_alignments')
        diarizer_params: Parameters for speaker diarization
        silence_params: Parameters for silence-based audio splitting
    
    Returns:
        Path to the output JSON file
    """
    device = device if device is not None else get_device()
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize components
    print("\n1. Initializing components...")
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
    print("\n2. Loading and preprocessing audio...")
    audio_file = AudioFile(file_path)
    audio_file.rms_normalization()
    
    # Split audio into chunks
    print("\n3. Splitting audio into chunks...")
    audio_file.split_on_silence(**silence_params)
    
    # Transcribe chunks
    print("\n4. Transcribing audio chunks...")
    for idx, chunk in enumerate(audio_file.chunks, 1):
        print(f"\n Chunk {idx}/{len(audio_file.chunks)}:")
        result = transcriber.transcribe_chunk(chunk)
    
    # Perform forced alignment
    print("\n5. Performing forced alignment...")
    aligner.align(audio_file)
    
    # Perform speaker diarization
    print("\n6. Performing speaker diarization...")
    diarizer.diarize(audio_file, num_speakers)
    
    # Create and process transcript
    print("\n7. Processing transcript...")
    transcript = Transcript(audio_file)
    print(f"Using {alignment_source} for word timing")
    transcript.combine_alignment_and_diarization(alignment_source)
    transcript.aggregate_to_utterances(pause_threshold=pause_threshold, 
                                     max_duration=max_utterance_duration)
    
    # Save results
    output_file = os.path.join(output_dir, f"{Path(file_path).stem}_transcript.json")
    transcript.save_as_json(output_file)
    
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