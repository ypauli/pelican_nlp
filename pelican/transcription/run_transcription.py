#!/usr/bin/env python3
"""
Script to run the transcription pipeline on an audio file.
"""
import argparse
from core.main import process_audio
from core.utils import get_device

def main():
    parser = argparse.ArgumentParser(description="Process an audio file through the transcription pipeline.")
    parser.add_argument("audio_file", help="Path to the audio file to process")
    parser.add_argument("--hf-token", required=True, help="HuggingFace token for diarization model")
    parser.add_argument("--output-dir", default="output", help="Directory to save output files")
    parser.add_argument("--num-speakers", type=int, default=2, help="Expected number of speakers")
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], help="Device to use (default: best available)")
    
    args = parser.parse_args()
    
    # Get device
    device = get_device() if args.device is None else args.device
    print(f"Using device: {device}")
    
    output_path = process_audio(
        file_path=args.audio_file,
        hf_token=args.hf_token,
        output_dir=args.output_dir,
        num_speakers=args.num_speakers,
        device=device
    )
    
    print(f"\nProcessing complete! Output saved to: {output_path}")

if __name__ == "__main__":
    main() 