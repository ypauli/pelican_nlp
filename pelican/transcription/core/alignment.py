"""
Forced alignment module for aligning transcripts with audio.
"""
import io
import torch
import torchaudio
import torchaudio.transforms as T
import uroman as ur
from typing import List, Dict, Optional
from .audio import AudioFile
from .utils import normalize_text, get_device


class ForcedAligner:
    """Handles forced alignment of transcripts with audio."""
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize the forced aligner.
        
        Args:
            device: Device to use for inference (default: best available)
        """
        self.device = device if device is not None else get_device()
        print(f"Initializing ForcedAligner on device: {self.device}")

        # Initialize forced aligner components
        self.bundle = torchaudio.pipelines.MMS_FA
        self.model = self.bundle.get_model().to(self.device)
        self.tokenizer = self.bundle.get_tokenizer()
        self.aligner = self.bundle.get_aligner()
        self.uroman = ur.Uroman()
        self.sample_rate = self.bundle.sample_rate

    def normalize_uroman(self, text: str) -> str:
        """
        Normalize text using Uroman.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        text = text.encode('utf-8').decode('utf-8')
        text = text.lower()
        text = text.replace("'", "'")
        text = normalize_text(text)
        return text.strip()

    def align(self, audio_file: AudioFile):
        """
        Perform forced alignment and populate the audio file chunks.
        
        Args:
            audio_file: AudioFile instance containing audio chunks
        """
        print("Starting forced alignment of transcripts...")
        for idx, chunk in enumerate(audio_file.chunks, start=1):
            try:
                with io.BytesIO() as wav_io:
                    chunk.audio_segment.export(wav_io, format="wav")
                    wav_io.seek(0)
                    waveform, sample_rate = torchaudio.load(wav_io)
                    waveform = waveform.to(self.device)
                
                # Resample if necessary
                if sample_rate != self.sample_rate:
                    resampler = T.Resample(
                        orig_freq=sample_rate,
                        new_freq=self.sample_rate
                    ).to(self.device)
                    waveform = resampler(waveform)
                    sample_rate = self.sample_rate

                # Normalize and tokenize the transcript
                text_roman = self.uroman.romanize_string(chunk.transcript)
                text_normalized = self.normalize_uroman(text_roman)
                transcript_list = text_normalized.split()
                tokens = self.tokenizer(transcript_list)

                # Perform forced alignment
                with torch.inference_mode():
                    emission, _ = self.model(waveform)
                    token_spans = self.aligner(emission[0], tokens)

                # Extract timestamps
                num_frames = emission.size(1)
                ratio = waveform.size(1) / num_frames
                for spans, word in zip(token_spans, transcript_list):
                    start_sec = (spans[0].start * ratio / sample_rate) + chunk.start_time
                    end_sec = (spans[-1].end * ratio / sample_rate) + chunk.start_time
                    chunk.forced_alignments.append({
                        "word": word,
                        "start_time": start_sec,
                        "end_time": end_sec
                    })
                print(f"Aligned chunk {idx} successfully.")
            except Exception as e:
                print(f"Error during alignment of chunk {idx}: {e}")
                
        audio_file.register_model("Forced Alignment", {
            "model": "torchaudio.pipelines.MMS_FA",
            "device": str(self.device)
        }) 