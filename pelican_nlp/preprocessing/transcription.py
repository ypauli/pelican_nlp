"""
Audio transcription utilities for PELICAN-nlp.

This module provides utility classes for audio transcription, forced alignment,
and speaker diarization using various machine learning models.
"""

import io
import os
import re
import unicodedata
import warnings
from typing import Dict

# Third-party Library Imports
import torch
import torchaudio
import torchaudio.transforms as T
from transformers import pipeline
from pyannote.audio import Pipeline as DiarizationPipeline
import uroman as ur

# Suppress FutureWarning from transformers about 'inputs' vs 'input_features'
# This is a deprecation warning from the transformers library that will be fixed in a future version
warnings.filterwarnings("ignore", category=FutureWarning, message=".*input name `inputs` is deprecated.*")


class AudioTranscriber:
    """Handles transcription of audio chunks using Whisper."""
    
    def __init__(self, model="openai/whisper-medium"):
        """
        Initialize the AudioTranscriber.
        
        :param model: Whisper model to use for transcription.
        """
        # Determine device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        self.model = model
        # Initialize the Whisper pipeline
        self.transcriber = pipeline(
            "automatic-speech-recognition",
            model=model,
            device=self.device,
            return_timestamps="word"
        )
        print(f"Initialized AudioTranscriber on device: {self.device}")

    def transcribe(self, audio_file):
        """
        Transcribe each audio chunk and populate the AudioFile instance.
        
        :param audio_file: AudioFile instance containing audio chunks.
        """
        print("Starting transcription of audio chunks...")
        for idx, chunk in enumerate(audio_file.chunks, start=1):
            try:
                with io.BytesIO() as wav_io:
                    chunk.audio_segment.export(wav_io, format="wav")
                    wav_io.seek(0)
                    transcription_result = self.transcriber(wav_io.read())

                # Assign transcript to the chunk
                chunk.transcript = transcription_result.get('text', "").strip()

                # Extract word alignments
                raw_chunks = transcription_result.get('chunks', [])
                clean_chunks = []
                for word_info in raw_chunks:
                    if 'timestamp' in word_info and len(word_info['timestamp']) == 2:
                        start_time = float(word_info['timestamp'][0]) + chunk.start_time
                        end_time = float(word_info['timestamp'][1]) + chunk.start_time
                        word_text = word_info.get('text', "").strip()
                        if word_text:
                            clean_chunks.append({
                                "word": word_text,
                                "start_time": start_time,
                                "end_time": end_time
                            })
                chunk.whisper_alignments = clean_chunks
                print(f"Transcribed chunk {idx} with {len(clean_chunks)} words.")
            except Exception as e:
                print(f"Error during transcription of chunk {idx}: {e}")
                chunk.transcript = ""
                chunk.whisper_alignments = []
                
        audio_file.register_model("Transcription", {
            "model": self.model,
            "device": str(self.device)
        })


class ForcedAligner:
    """Handles forced alignment of transcripts with audio."""
    
    def __init__(self, device: str = None):
        """
        Initialize the ForcedAligner.
        
        :param device: Device to use for alignment (auto-detected if None).
        """
        # Determine device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Initialize forced aligner components
        self.bundle = torchaudio.pipelines.MMS_FA
        self.model = self.bundle.get_model().to(self.device)
        self.tokenizer = self.bundle.get_tokenizer()
        self.aligner = self.bundle.get_aligner()
        self.uroman = ur.Uroman()
        self.sample_rate = self.bundle.sample_rate
        print(f"Initialized ForcedAligner on device: {self.device}")

    def normalize_uroman(self, text: str) -> str:
        """
        Normalize text using Uroman.
        
        :param text: Text to normalize.
        :return: Normalized text.
        """
        text = text.encode('utf-8').decode('utf-8')
        text = text.lower()
        text = text.replace("'", "'")
        text = unicodedata.normalize('NFC', text)
        text = re.sub("([^a-z' ])", " ", text)
        text = re.sub(' +', ' ', text)
        return text.strip()

    def align(self, audio_file):
        """
        Perform forced alignment and populate the AudioFile instance.
        
        :param audio_file: AudioFile instance containing audio chunks.
        """
        print("Starting forced alignment of transcripts...")
        for idx, chunk in enumerate(audio_file.chunks, start=1):
            try:
                with io.BytesIO() as wav_io:
                    chunk.audio_segment.export(wav_io, format="wav")
                    wav_io.seek(0)
                    waveform, sample_rate = torchaudio.load(wav_io)

                # Resample if necessary
                if sample_rate != self.sample_rate:
                    resampler = T.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
                    waveform = resampler(waveform)
                    sample_rate = self.sample_rate

                # Normalize and tokenize the transcript
                text_roman = self.uroman.romanize_string(chunk.transcript)
                text_normalized = self.normalize_uroman(text_roman)
                transcript_list = text_normalized.split()
                tokens = self.tokenizer(transcript_list)

                # Perform forced alignment
                with torch.inference_mode():
                    emission, _ = self.model(waveform.to(self.device))
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


class SpeakerDiarizer:
    """Handles speaker diarization of audio files."""
    
    def __init__(self, hf_token: str, parameters: Dict, model="pyannote/speaker-diarization-3.1"):
        """
        Initialize the SpeakerDiarizer.
        
        :param hf_token: Hugging Face token for accessing diarization models.
        :param parameters: Parameters for the diarization pipeline.
        :param model: Diarization model to use.
        """
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model = model
        self.parameters = parameters
        
        if not hf_token:
            print("=" * 60)
            print("WARNING: No Hugging Face token provided!")
            print("Speaker diarization will be skipped.")
            print("")
            print("To enable speaker diarization, add your Hugging Face token to the config:")
            print("  transcription:")
            print("    hf_token: 'your_hugging_face_token_here'")
            print("")
            print("You can get a token from: https://huggingface.co/settings/tokens")
            print("=" * 60)
            self.diarization_pipeline = None
            return
            
        try:
            # Set Hugging Face token as environment variable for authentication
            import os
            os.environ['HF_TOKEN'] = hf_token
            os.environ['HUGGING_FACE_HUB_TOKEN'] = hf_token
            
            # Try different ways to pass the token based on pyannote.audio version
            try:
                # Method 1: Try use_auth_token (older pyannote.audio versions)
                self.diarization_pipeline = DiarizationPipeline.from_pretrained(
                    model,
                    use_auth_token=hf_token
                )
            except (TypeError, ValueError) as e1:
                try:
                    # Method 2: Try without explicit token (uses environment variable)
                    self.diarization_pipeline = DiarizationPipeline.from_pretrained(model)
                except Exception as e2:
                    # Method 3: Try with token parameter (newer versions)
                    try:
                        self.diarization_pipeline = DiarizationPipeline.from_pretrained(
                            model,
                            token=hf_token
                        )
                    except Exception as e3:
                        raise Exception(f"Failed to initialize pipeline. Tried use_auth_token (error: {e1}), "
                                      f"environment variable (error: {e2}), and token (error: {e3})")
            
            print("Initializing SpeakerDiarizer with parameters...")
            self.diarization_pipeline.instantiate(parameters)
            self.diarization_pipeline.to(self.device)
            print("Initialized SpeakerDiarizer successfully.")
        except Exception as e:
            print("=" * 60)
            print(f"ERROR: Failed to initialize SpeakerDiarizer!")
            print(f"Error: {e}")
            print("")
            print("Common causes:")
            print("  1. Invalid Hugging Face token")
            print("  2. Missing model access permissions")
            print("  3. Network connection issues")
            print("  4. Missing dependencies")
            print("")
            print("Speaker diarization will be skipped.")
            print("=" * 60)
            import traceback
            traceback.print_exc()
            self.diarization_pipeline = None

    def diarize(self, audio_file, num_speakers: int = None):
        """
        Perform speaker diarization on the given audio file.
        
        :param audio_file: AudioFile instance containing audio data.
        :param num_speakers: Expected number of speakers.
        """
        if self.diarization_pipeline is None:
            print("Speaker diarization skipped - no pipeline available.")
            audio_file.speaker_segments = []
            audio_file.register_model("Speaker Diarization", {
                "model": "none",
                "device": "none",
                "parameters": {},
                "speakers": "skipped"
            })
            return
            
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
            
            # DEBUG: Print first few speaker segments to verify they're populated
            if audio_file.speaker_segments:
                print(f"DEBUG: First 3 speaker segments:")
                for i, seg in enumerate(audio_file.speaker_segments[:3]):
                    print(f"  Segment {i}: {seg}")
                print(f"DEBUG: Speaker segment time range: {audio_file.speaker_segments[0]['start']:.2f}s - {audio_file.speaker_segments[-1]['end']:.2f}s")
            else:
                print("DEBUG: WARNING - speaker_segments is empty after diarization!")
        except Exception as e:
            print(f"An error occurred during diarization: {e}")
            
        audio_file.register_model("Speaker Diarization", {
            "model": self.model,
            "device": str(self.device),
            "parameters": self.parameters,
            "speakers": num_speakers if num_speakers else "not specified"
        })


def process_single_audio_file(audio_file,
                              hf_token: str,
                              diarizer_params: Dict = None,
                              num_speakers: int = 2,
                              min_silence_len: int = 1000,
                              silence_thresh: int = -30,
                              min_length: int = 90000,
                              max_length: int = 150000,
                              timestamp_source: str = "whisper_alignments"):
    """
    Process a single audio file through the entire transcription pipeline.
    
    :param audio_file: AudioFile instance to process.
    :param hf_token: Hugging Face token for accessing diarization models.
    :param diarizer_params: Parameters for the SpeakerDiarizer model.
    :param num_speakers: Expected number of speakers.
    :param min_silence_len: Minimum silence length for splitting (ms).
    :param silence_thresh: Silence threshold in dBFS for splitting.
    :param min_length: Minimum chunk length in ms.
    :param max_length: Maximum chunk length in ms.
    :param timestamp_source: Alignment source to use ('whisper_alignments' or 'forced_alignments').
    :return: The processed AudioFile instance.
    """
    # Set default diarizer parameters if not provided
    if diarizer_params is None:
        diarizer_params = {
            "segmentation": {
                "min_duration_off": 0.0,
            },
            "clustering": {
                "method": "centroid",
                "min_cluster_size": 12,
                "threshold": 0.8,
            }
        }
    
    print(f"Processing audio file: {audio_file.file}")
    print(f"Audio file exists: {os.path.exists(audio_file.file)}")

    # Initialize processing classes
    print("Initializing processing classes...")
    transcriber = AudioTranscriber()
    aligner = ForcedAligner()
    diarizer = SpeakerDiarizer(hf_token, parameters=diarizer_params)
    print("Processing classes initialized successfully.")

    # Step 1: Load audio
    print("Step 1/7: Loading audio...")
    audio_file.load_audio()

    # Step 2: Normalize audio
    print("Step 2/7: Normalizing audio...")
    # Use normalized audio directory if set, otherwise use default (same directory as original)
    normalized_audio_dir = getattr(audio_file, '_normalized_audio_dir', None)
    audio_file.rms_normalization(output_dir=normalized_audio_dir)

    # Step 3: Split audio into chunks based on silence
    print("Step 3/7: Splitting audio on silence...")
    audio_file.split_on_silence(
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        min_length=min_length,
        max_length=max_length
    )

    # Step 4: Transcribe audio chunks
    print("Step 4/7: Transcribing audio chunks...")
    transcriber.transcribe(audio_file)
    for idx, chunk in enumerate(audio_file.chunks, start=1):
        print(f"Chunk {idx} Transcript: {chunk.transcript}\n")

    # Step 5: Perform forced alignment
    print("Step 5/7: Performing forced alignment...")
    aligner.align(audio_file)        
    audio_file.combine_chunks()

    # Step 6: Perform speaker diarization (only if more than one speaker)
    # Ensure num_speakers is set on audio_file for later use
    if not hasattr(audio_file, 'num_speakers') or audio_file.num_speakers is None:
        audio_file.num_speakers = num_speakers
    
    if num_speakers and num_speakers > 1:
        print("Step 6/7: Performing speaker diarization...")
        diarizer.diarize(audio_file, num_speakers)
    else:
        print("Step 6/7: Skipping speaker diarization (only one speaker specified)...")
        audio_file.speaker_segments = []
        audio_file.register_model("Speaker Diarization", {
            "model": "none",
            "device": "none",
            "parameters": {},
            "speakers": "skipped (single speaker)"
        })

    # Step 7: Combine alignment and diarization data
    print("Step 7/7: Combining alignment and diarization data...")
    audio_file.combine_alignment_and_diarization(timestamp_source)
    audio_file.aggregate_to_utterances()

    print(f"Finished processing: {audio_file.file}")
    
    # Clean up models and free GPU memory
    # Note: We avoid moving complex pipeline objects to CPU as this can cause segfaults
    # Instead, we just delete references and clear the cache
    
    # Delete transcriber (Whisper pipeline)
    try:
        if hasattr(transcriber, 'transcriber'):
            del transcriber.transcriber
    except Exception:
        pass
    try:
        del transcriber
    except Exception:
        pass
    
    # Delete aligner model
    try:
        if hasattr(aligner, 'model'):
            del aligner.model
    except Exception:
        pass
    try:
        del aligner
    except Exception:
        pass
    
    # Delete diarizer pipeline
    try:
        if hasattr(diarizer, 'diarization_pipeline'):
            del diarizer.diarization_pipeline
    except Exception:
        pass
    try:
        del diarizer
    except Exception:
        pass
    
    # Clear GPU cache after transcription
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Ensure cache clearing is complete
        print("GPU memory cleared after transcription")
    
    # Force Python garbage collection to help release memory
    import gc
    gc.collect()
    
    return audio_file