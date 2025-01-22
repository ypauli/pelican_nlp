"""
Forced alignment module for aligning transcripts with audio.

This module provides functionality for precise word-level alignment between
transcribed text and audio using the MMS Forced Aligner from torchaudio.
Key features include:
- Automatic text normalization and romanization
- Word-level timing extraction
- Support for multiple languages
- Batch processing of audio chunks
- Robust handling of different audio formats and sampling rates

The module uses torchaudio's MMS-FA pipeline for high-accuracy alignment
and uroman for text normalization.
"""
import io
import torch
import torchaudio
import torchaudio.transforms as T
import uroman as ur
from typing import List, Dict, Optional, Any
from .audio import AudioFile
from .utils import normalize_text, get_device
import librosa


class ForcedAligner:
    """
    Handles forced alignment of transcripts with audio using MMS-FA.
    
    This class provides precise word-level alignment between transcribed text
    and audio using torchaudio's MMS Forced Aligner. It handles:
    - Text normalization and romanization for multiple languages
    - Audio resampling and format conversion
    - Batch processing of audio chunks
    - Generation of word-level timing information
    
    The aligner uses a pre-trained model that supports multiple languages
    and can handle various audio conditions. It provides timing information
    at both the word and phoneme level.

    Attributes:
        device (torch.device): Compute device for model inference
        bundle (torchaudio.pipelines): MMS-FA model bundle
        model (torch.nn.Module): The loaded alignment model
        processor (object): Text and audio processor for the model
        uroman_dir (str): Directory containing uroman normalization rules
    """
    
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

    def normalize_uroman(self, text: str) -> Dict[str, Any]:
        """
        Normalize text using Uroman, preserving original words.
        Handles German characters by converting them to ASCII equivalents.
        
        Args:
            text: Text to normalize
            
        Returns:
            Dictionary containing normalized text and mapping to original words
        """
        # Split original text into words and remove empty strings
        original_words = [w for w in text.strip().split() if w]
        normalized_words = []
        word_mapping = []  # Maps normalized word indices to original word indices
        
        for i, word in enumerate(original_words):
            # Remove all non-letter characters first
            clean_word = ''.join(c for c in word if c.isalpha() or c in 'äöüß-')
            if not clean_word:
                continue
                
            # Basic cleanup and German character normalization
            norm_word = clean_word.lower()
            norm_word = norm_word.replace('ß', 'ss')
            norm_word = norm_word.replace('ä', 'ae')
            norm_word = norm_word.replace('ö', 'oe')
            norm_word = norm_word.replace('ü', 'ue')
            
            # Handle hyphenated words
            if "-" in norm_word:
                sub_words = [w for w in norm_word.split("-") if w]
                normalized_words.extend(sub_words)
                word_mapping.extend([i] * len(sub_words))
                continue
            
            if norm_word:
                normalized_words.append(norm_word)
                word_mapping.append(i)
        
        # Join normalized words with spaces
        normalized_text = " ".join(normalized_words)
        
        return {
            "normalized_text": normalized_text,
            "original_words": original_words,
            "word_mapping": word_mapping
        }

    def align(self, audio_file: AudioFile) -> None:
        """
        Perform forced alignment on an audio file.
        
        Args:
            audio_file: AudioFile object containing chunks to align
        """
        total_aligned = 0
        total_words = 0
        total_chunks = len(audio_file.chunks)
        
        print("\nPerforming forced alignment...")
        for idx, chunk in enumerate(audio_file.chunks, 1):
            if not chunk.transcript:
                continue
                
            print(f"\nChunk {idx}/{total_chunks}:")
            
            try:
                # Process audio data
                with io.BytesIO() as wav_io:
                    chunk.audio_segment.export(wav_io, format='wav')
                    wav_io.seek(0)
                    waveform, sample_rate = torchaudio.load(wav_io)
                    waveform = waveform.to(self.device)
                
                # Resample if necessary
                if sample_rate != self.sample_rate:
                    resampler = T.Resample(orig_freq=sample_rate, new_freq=self.sample_rate).to(self.device)
                    waveform = resampler(waveform)
                
                # Normalize text for alignment
                norm_result = self.normalize_uroman(chunk.transcript)
                
                # Get alignments for normalized text
                alignments = self._get_alignments(waveform, norm_result['normalized_text'])
                
                if alignments:
                    # Map alignments back to original words
                    original_alignments = []
                    current_word_idx = None
                    current_alignment = None
                    
                    for i, alignment in enumerate(alignments):
                        orig_idx = norm_result['word_mapping'][i]
                        orig_word = norm_result['original_words'][orig_idx]
                        
                        if orig_idx != current_word_idx:
                            # New original word
                            if current_alignment:
                                original_alignments.append(current_alignment)
                            current_word_idx = orig_idx
                            current_alignment = {
                                "word": orig_word,
                                "start_time": chunk.start_time + alignment["start_time"],
                                "end_time": chunk.start_time + alignment["end_time"]
                            }
                        else:
                            # Same original word (e.g., "m r c s"), update end time
                            current_alignment["end_time"] = chunk.start_time + alignment["end_time"]
                    
                    # Add last alignment if exists
                    if current_alignment:
                        original_alignments.append(current_alignment)
                    
                    chunk.forced_alignments = original_alignments
                    total_aligned += 1
                    total_words += len(original_alignments)
                    print(f"  ✓ Aligned {len(original_alignments)} words")
                else:
                    print(f"  ✗ Failed to align")
                    
            except Exception as e:
                print(f"  ✗ Error: {str(e)}")
                continue
        
        # Print summary
        success_rate = (total_aligned / total_chunks) * 100 if total_chunks > 0 else 0
        print(f"\nAlignment Summary:")
        print(f"  - Chunks: {total_aligned}/{total_chunks} ({success_rate:.1f}% success)")
        print(f"  - Total words aligned: {total_words}")
        
        if total_aligned == 0:
            raise ValueError("No chunks were successfully aligned! Consider using Whisper alignments instead.")
            
        audio_file.register_model("Forced Alignment", {
            "model": "torchaudio.pipelines.MMS_FA",
            "device": str(self.device),
            "chunks_aligned": total_aligned,
            "total_chunks": total_chunks,
            "total_words": total_words
        })

    def _get_alignments(self, waveform: torch.Tensor, text: str) -> List[Dict[str, Any]]:
        """
        Get word-level alignments using the forced aligner.
        
        Args:
            waveform: Audio waveform tensor
            text: Normalized text to align
            
        Returns:
            List of word alignments with timestamps
        """
        try:
            # Split text into words and tokenize
            words = text.split()
            if not words:
                print("  ✗ No words to align")
                return []
                
            # Tokenize and align
            tokens = self.tokenizer(words)
            
            # Generate emission matrix
            with torch.inference_mode():
                emission, _ = self.model(waveform)
                token_spans = self.aligner(emission[0], tokens)
            
            # Convert spans to timestamps
            num_frames = emission.size(1)
            ratio = waveform.size(1) / num_frames
            
            alignments = []
            for spans, word in zip(token_spans, words):
                start_sec = spans[0].start * ratio / self.sample_rate
                end_sec = spans[-1].end * ratio / self.sample_rate
                alignments.append({
                    "word": word,
                    "start_time": start_sec,
                    "end_time": end_sec
                })
            
            return alignments
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            return [] 