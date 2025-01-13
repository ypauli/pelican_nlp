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
            device: Device to use for inference (default: best available excluding MPS)
        """
        # Force skip MPS as it's not supported for alignment
        self.device = device if device is not None else get_device(skip_mps=True)
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
        # Basic cleanup
        text = text.encode('utf-8').decode('utf-8')
        text = text.lower()
        text = text.replace("'", "'")
        
        # Handle numbers and special characters
        text = text.replace("1984", "neunzehnhundertvierundachtzig")  # Convert year to words
        text = text.replace("10.", "zehnten")  # Convert ordinal number
        text = text.replace("mrcs", "m r c s")  # Split acronym
        text = text.replace("-", " ")  # Replace hyphens with spaces
        
        # General text normalization
        text = normalize_text(text)
        
        # Remove any remaining non-letter characters except spaces
        text = ''.join(c for c in text if c.isalpha() or c.isspace())
        
        return ' '.join(text.split())  # Normalize whitespace

    def align(self, audio_file: AudioFile):
        """
        Perform forced alignment and populate the audio file chunks.
        
        Args:
            audio_file: AudioFile instance containing audio chunks
        """
        print("Starting forced alignment of transcripts...")
        total_aligned = 0
        total_chunks = len(audio_file.chunks)
        
        for idx, chunk in enumerate(audio_file.chunks, start=1):
            # Skip chunks with empty transcripts
            if not chunk.transcript.strip():
                print(f"Skipping chunk {idx}/{total_chunks} - empty transcript")
                continue
                
            try:
                # Export audio to WAV and load
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

                # Clean and normalize the transcript
                try:
                    text_roman = self.uroman.romanize_string(chunk.transcript)
                    text_normalized = self.normalize_uroman(text_roman)
                    transcript_list = text_normalized.split()
                    
                    print(f"  Chunk {idx}/{total_chunks}:")
                    print(f"    Original: {chunk.transcript}")
                    print(f"    Normalized: {text_normalized}")
                except Exception as e:
                    print(f"    Error during text normalization: {str(e)}")
                    continue

                # Skip if no words after normalization
                if not transcript_list:
                    print(f"    Skipping - no words after normalization")
                    continue
                    
                # Tokenize and align
                try:
                    print(f"    Attempting tokenization of {len(transcript_list)} words...")
                    tokens = self.tokenizer(transcript_list)
                    print(f"    Tokenization successful: {len(tokens)} tokens")
                    
                    print("    Starting alignment...")
                    with torch.inference_mode():
                        emission, _ = self.model(waveform)
                        print(f"    Generated emission matrix: {emission.shape}")
                        try:
                            token_spans = self.aligner(emission[0], tokens)
                            print(f"    Alignment successful: {len(token_spans)} spans")
                        except Exception as align_err:
                            print(f"    Alignment failed with error: {str(align_err)}")
                            print(f"    Emission shape: {emission.shape}")
                            print(f"    Number of tokens: {len(tokens)}")
                            print(f"    First few tokens: {tokens[:10]}")
                            raise align_err
                except Exception as e:
                    print(f"    Error during alignment step: {str(e)}")
                    print(f"    Transcript list: {transcript_list[:10]} ...")
                    continue

                # Extract timestamps
                num_frames = emission.size(1)
                ratio = waveform.size(1) / num_frames
                
                # Clear previous alignments
                chunk.forced_alignments = []
                
                for spans, word in zip(token_spans, transcript_list):
                    start_sec = (spans[0].start * ratio / sample_rate) + chunk.start_time
                    end_sec = (spans[-1].end * ratio / sample_rate) + chunk.start_time
                    chunk.forced_alignments.append({
                        "word": word,
                        "start_time": start_sec,
                        "end_time": end_sec
                    })
                
                total_aligned += 1
                print(f"    Successfully aligned {len(chunk.forced_alignments)} words")
                
            except Exception as e:
                print(f"    Error processing chunk: {str(e)}")
                continue
        
        print(f"\nAlignment complete: {total_aligned}/{total_chunks} chunks aligned successfully")
        
        if total_aligned == 0:
            print("WARNING: No chunks were successfully aligned!")
            print("Falling back to Whisper alignments...")
            for chunk in audio_file.chunks:
                chunk.forced_alignments = chunk.whisper_alignments
            
        audio_file.register_model("Forced Alignment", {
            "model": "torchaudio.pipelines.MMS_FA",
            "device": str(self.device),
            "chunks_aligned": total_aligned,
            "total_chunks": total_chunks
        }) 