"""
Audio transcription utilities for PELICAN-nlp.

This module provides utility classes for audio transcription, forced alignment,
and speaker diarization using various machine learning models.
"""

import io
import os
import re
import unicodedata
from typing import Dict

# Third-party Library Imports
import torch
import torchaudio
import torchaudio.transforms as T
from transformers import pipeline
from pyannote.audio import Pipeline as DiarizationPipeline
import uroman as ur


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
            print("Warning: No Hugging Face token provided. Speaker diarization will be skipped.")
            self.diarization_pipeline = None
            return
            
        try:
            self.diarization_pipeline = DiarizationPipeline.from_pretrained(
                model,
                token=hf_token
            )
            print("Initializing SpeakerDiarizer with parameters...")
            self.diarization_pipeline.instantiate(parameters)
            self.diarization_pipeline.to(self.device)
            print("Initialized SpeakerDiarizer successfully.")
        except Exception as e:
            print(f"Error initializing SpeakerDiarizer: {e}")
            print("Speaker diarization will be skipped.")
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
    audio_file.rms_normalization()

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

    # Step 6: Perform speaker diarization
    print("Step 6/7: Performing speaker diarization...")
    diarizer.diarize(audio_file, num_speakers)

    # Step 7: Combine alignment and diarization data
    print("Step 7/7: Combining alignment and diarization data...")
    audio_file.combine_alignment_and_diarization(timestamp_source)
    audio_file.aggregate_to_utterances()

    print(f"Finished processing: {audio_file.file}")
    
    # Clean up
    del transcriber
    del aligner
    del diarizer
    
    return audio_file