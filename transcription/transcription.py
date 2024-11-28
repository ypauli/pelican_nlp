# Standard Library Imports
import io
import re
import unicodedata
from typing import List, Dict
from pathlib import Path
import json

# Third-party Library Imports
import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T
from pydub import AudioSegment
from pydub.silence import detect_silence
from transformers import pipeline
from pyannote.audio import Pipeline as DiarizationPipeline
import uroman as ur
import pandas as pd


class Chunk:
    def __init__(self, audio_segment: AudioSegment, start_time: float):
        """
        Represents a chunk of audio.
        
        :param audio_segment: The audio segment.
        :param start_time: Start time in the original audio (seconds).
        """
        self.audio_segment = audio_segment
        self.start_time = start_time  # Start time in seconds
        self.transcript = ""
        self.whisper_alignments = []
        self.forced_alignments = []


class AudioFile:
    def __init__(self, file_path: str, target_rms_db: float = -20):
        """
        Handles all operations related to an audio file.
        
        :param file_path: Path to the audio file.
        :param target_rms_db: Target RMS in dB for normalization.
        """
        self.file_path = file_path
        self.target_rms_db = target_rms_db
        self.normalized_path = None
        self.audio = None
        self.sample_rate = None
        self.chunks: List[Chunk] = []
        self.speaker_segments = []
        
        self.metadata = {
            "file_path": file_path,
            "length_seconds": None,
            "sample_rate": None,
            "target_rms_db": target_rms_db,
            "models_used": {}
        }
        
        self.load_audio()

    def load_audio(self):
        """
        Loads the audio file using librosa.
        """
        self.audio, self.sample_rate = librosa.load(self.file_path, sr=None)
        self.metadata["sample_rate"] = self.sample_rate
        print(f"Loaded audio file: {self.file_path}")
        
    def register_model(self, model_name: str, parameters: dict):
        """
        Registers a model and its parameters in the metadata.
        
        :param model_name: Name of the model.
        :param parameters: Parameters used for the model.
        """
        self.metadata["models_used"][model_name] = parameters

    def rms_normalization(self):
        """
        Normalizes the audio to the target RMS level and saves it.
        """
        target_rms = 10 ** (self.target_rms_db / 20)
        rms = np.sqrt(np.mean(self.audio ** 2))
        gain = target_rms / rms
        normalized_audio = self.audio * gain
        self.normalized_path = self.file_path.replace(".wav", "_normalized.wav")
        sf.write(self.normalized_path, normalized_audio, self.sample_rate)
        print(f"Normalized audio saved as: {self.normalized_path}")

    def split_on_silence(self, min_silence_len=1000, silence_thresh=-30,
                         min_length=30000, max_length=180000):
        """
        Splits the audio into chunks based on silence.
        
        :param min_silence_len: Minimum length of silence to be used for a split (ms).
        :param silence_thresh: Silence threshold in dBFS.
        :param min_length: Minimum length of a chunk (ms).
        :param max_length: Maximum length of a chunk (ms).
        """
        audio_segment = AudioSegment.from_file(self.normalized_path)
        audio_length_ms = len(audio_segment)
        self.metadata["length_seconds"] = audio_length_ms / 1000
        silence_ranges = self._detect_silence_intervals(audio_segment, min_silence_len, silence_thresh)
        splitting_points = self._get_splitting_points(silence_ranges, audio_length_ms)
        initial_intervals = self._create_initial_chunks(splitting_points)
        adjusted_intervals = self._adjust_intervals_by_length(initial_intervals, min_length, max_length)
        chunks_with_timestamps = self._split_audio_by_intervals(audio_segment, adjusted_intervals)

        self.chunks = [Chunk(chunk_audio, start_i / 1000.0) for chunk_audio, start_i, end_i in chunks_with_timestamps]
        print(f"Total chunks after splitting: {len(self.chunks)}")
    
        # Validate the combined length of chunks
        self.validate_chunk_lengths(audio_length_ms)
        
        self.register_model("Chunking", {
            "min_silence_len": min_silence_len,
            "silence_thresh": silence_thresh,
            "min_length": min_length,
            "max_length": max_length,
            "num_chunks": len(self.chunks)
        })
        

    def _detect_silence_intervals(self, audio_segment: AudioSegment, min_silence_len: int, silence_thresh: int) -> List[List[int]]:
        """
        Detects silent intervals in the audio segment.
        
        :param audio_segment: The audio segment.
        :param min_silence_len: Minimum length of silence to be used for a split (ms).
        :param silence_thresh: Silence threshold in dBFS.
        :return: List of [start_ms, end_ms] pairs representing silence periods.
        """
        return detect_silence(audio_segment, min_silence_len=min_silence_len, silence_thresh=silence_thresh)

    def _get_splitting_points(self, silence_ranges: List[List[int]], audio_length_ms: int) -> List[int]:
        """
        Computes splitting points based on silence ranges.
        
        :param silence_ranges: List of silence intervals.
        :param audio_length_ms: Total length of the audio in ms.
        :return: Sorted list of splitting points in ms.
        """
        splitting_points = [0] + [(start + end) // 2 for start, end in silence_ranges] + [audio_length_ms]
        return splitting_points

    def _create_initial_chunks(self, splitting_points: List[int]) -> List[tuple]:
        """
        Creates initial chunks based on splitting points.
        
        :param splitting_points: List of splitting points in ms.
        :return: List of (start_ms, end_ms) tuples.
        """
        return list(zip(splitting_points[:-1], splitting_points[1:]))

    def _adjust_intervals_by_length(self, intervals: List[tuple], min_length: int, max_length: int) -> List[tuple]:
        """
        Adjusts intervals based on minimum and maximum length constraints.

        :param intervals: List of (start_ms, end_ms) tuples.
        :param min_length: Minimum length of a chunk (ms).
        :param max_length: Maximum length of a chunk (ms).
        :return: Adjusted list of intervals.
        """
        adjusted_intervals = []
        buffer_start, buffer_end = intervals[0]

        for start, end in intervals[1:]:
            buffer_end = end
            buffer_length = buffer_end - buffer_start

            if buffer_length < min_length:
                # Merge with the next interval by extending the buffer
                continue
            else:
                if buffer_length > max_length:
                    # Split the buffer into multiple chunks of `max_length`
                    num_splits = int(np.ceil(buffer_length / max_length))
                    split_size = int(np.ceil(buffer_length / num_splits))
                    for i in range(num_splits):
                        split_start = buffer_start + i * split_size
                        split_end = min(buffer_start + (i + 1) * split_size, buffer_end)
                        adjusted_intervals.append((split_start, split_end))
                else:
                    # Add the buffer as a valid interval
                    adjusted_intervals.append((buffer_start, buffer_end))
                buffer_start = buffer_end  # Reset buffer_start to the end of the current buffer

        # Handle any remaining buffer (final chunk)
        buffer_length = buffer_end - buffer_start
        if buffer_length > 0:
            if buffer_length >= min_length:
                # Include the final chunk if it's greater than `min_length`
                adjusted_intervals.append((buffer_start, buffer_end))
            else:
                # Optionally include shorter chunks
                print(f"Final chunk is shorter than min_length ({buffer_length} ms), including it anyway.")
                adjusted_intervals.append((buffer_start, buffer_end))

        return adjusted_intervals
    
    def validate_chunk_lengths(self, audio_length_ms: int, tolerance: float = 1.0):
        """
        Validates that the combined length of all chunks matches the original audio length.

        :param audio_length_ms: Length of the original audio in milliseconds.
        :param tolerance: Allowed tolerance in milliseconds.
        """
        # Sum up the duration of all chunks
        combined_length = sum(len(chunk.audio_segment) for chunk in self.chunks)

        # Calculate the difference
        difference = abs(combined_length - audio_length_ms)
        if difference > tolerance:
            raise AssertionError(
                f"Chunk lengths validation failed! Combined chunk length ({combined_length} ms) "
                f"differs from original audio length ({audio_length_ms} ms) by {difference} ms, "
                f"which exceeds the allowed tolerance of {tolerance} ms."
            )
        print(f"Chunk length validation passed: Total chunks = {combined_length} ms, Original = {audio_length_ms} ms.")

    def _split_audio_by_intervals(self, audio_segment: AudioSegment, intervals: List[tuple]) -> List[tuple]:
        """
        Splits the audio segment into chunks based on the provided intervals.
        
        :param audio_segment: The audio segment.
        :param intervals: List of (start_ms, end_ms) tuples.
        :return: List of (chunk_audio, start_ms, end_ms) tuples.
        """
        return [(audio_segment[start_ms:end_ms], start_ms, end_ms) for start_ms, end_ms in intervals]
    
    def combine_chunks(self):
        """
        Combines transcripts and alignments from all chunks.
        
        :param chunks: List of Chunk instances.
        """
        self.transcript_text = " ".join([chunk.transcript for chunk in self.chunks])
        self.whisper_alignments = []
        self.forced_alignments = []
        for chunk in self.chunks:
            self.whisper_alignments.extend(chunk.whisper_alignments)
            self.forced_alignments.extend(chunk.forced_alignments)
        print("Combined transcripts and alignments from all chunks into Transcript.")

class Transcript:
    def __init__(self, audio_file: AudioFile = None, json_data: dict = None):
        """
        Initializes the Transcript class.
        
        :param audio_file: AudioFile object to initialize from.
        :param json_data: Dictionary loaded from a JSON file.
        """
        if audio_file:
            self.audio_file_path = audio_file.file_path
            self.transcript_text = audio_file.transcript_text
            self.whisper_alignments = audio_file.whisper_alignments
            self.forced_alignments = audio_file.forced_alignments
            self.speaker_segments = audio_file.speaker_segments
            self.combined_data = []
            self.combined_utterances = []
            self.metadata = audio_file.metadata
        elif json_data:
            self.audio_file_path = json_data["audio_file_path"]
            self.metadata = json_data["metadata"]
            self.transcript_text = json_data.get("transcript_text", "")
            self.whisper_alignments = json_data.get("whisper_alignments", [])
            self.forced_alignments = json_data.get("forced_alignments", [])
            self.speaker_segments = json_data.get("speaker_segments", [])
            self.combined_data = json_data.get("combined_data", [])
            self.combined_utterances = json_data.get("utterance_data", [])
        else:
            raise ValueError("Either an AudioFile object or JSON data must be provided.")

    @classmethod
    def from_json_file(cls, json_file: str):
        """
        Creates a Transcript instance from a JSON file.
        
        :param json_file: Path to the JSON file.
        :return: Transcript instance.
        """
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                json_data = json.load(f)
            print(f"Loaded transcript data from '{json_file}'.")
            return cls(json_data=json_data)
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            raise

    def aggregate_to_utterances(self):
        """
        Aggregates word-level data into utterances based on sentence endings.
        """
        if not self.combined_data:
            print("No combined data available to aggregate.")
            return

        utterances = []
        current_utterance = {
            "text": "",
            "start_time": None,
            "end_time": None,
            "speakers": {}
        }

        sentence_endings = re.compile(r'[.?!]$')
        print("Aggregating words into utterances...")
        for word_data in self.combined_data:
            word = word_data["word"]
            start_time = word_data["start_time"]
            end_time = word_data["end_time"]
            speaker = word_data["speaker"]

            if current_utterance["start_time"] is None:
                current_utterance["start_time"] = start_time

            current_utterance["text"] += ("" if current_utterance["text"] == "" else " ") + word
            current_utterance["end_time"] = end_time

            if speaker not in current_utterance["speakers"]:
                current_utterance["speakers"][speaker] = 0
            current_utterance["speakers"][speaker] += 1

            if sentence_endings.search(word):
                majority_speaker, majority_count = max(
                    current_utterance["speakers"].items(), key=lambda item: item[1]
                )
                total_words = sum(current_utterance["speakers"].values())
                confidence = round(majority_count / total_words, 2)

                utterances.append({
                    "text": current_utterance["text"],
                    "start_time": current_utterance["start_time"],
                    "end_time": current_utterance["end_time"],
                    "speaker": majority_speaker,
                    "confidence": confidence,
                })

                current_utterance = {
                    "text": "",
                    "start_time": None,
                    "end_time": None,
                    "speakers": {}
                }

        # Handle any remaining words as the last utterance
        if current_utterance["text"]:
            majority_speaker, majority_count = max(
                current_utterance["speakers"].items(), key=lambda item: item[1]
            )
            total_words = sum(current_utterance["speakers"].values())
            confidence = round(majority_count / total_words, 2)

            utterances.append({
                "text": current_utterance["text"],
                "start_time": current_utterance["start_time"],
                "end_time": current_utterance["end_time"],
                "speaker": majority_speaker,
                "confidence": confidence,
            })

        self.combined_utterances = utterances
        print("Aggregated utterances from combined data.")

    def combine_alignment_and_diarization(self, alignment_source: str):
        """
        Combines alignment and diarization data by assigning speaker labels to each word.
        
        :param speaker_segments: List of speaker segments with 'start', 'end', and 'speaker'.
        :param alignment_source: The alignment data to use ('whisper_alignments' or 'forced_alignments').
        """
        if alignment_source not in ['whisper_alignments', 'forced_alignments']:
            raise ValueError("Invalid alignment_source. Choose 'whisper_alignments' or 'forced_alignments'.")

        alignment = getattr(self, alignment_source, None)
        if alignment is None:
            raise ValueError(f"The alignment source '{alignment_source}' does not exist in the Transcript object.")

        if not self.speaker_segments:
            print("No speaker segments available for diarization. All words will be labeled as 'UNKNOWN'.")
            self.combined_data = [{**word, 'speaker': 'UNKNOWN'} for word in alignment]
            return

        combined = []
        seg_idx = 0
        num_segments = len(self.speaker_segments)

        for word in alignment:
            word_start = word['start_time']
            word_end = word['end_time']
            word_duration = max(1e-6, word_end - word_start)  # Avoid zero-duration

            speaker_overlap = {}

            # Advance segments that have ended before the word starts
            while seg_idx < num_segments and self.speaker_segments[seg_idx]['end'] < word_start:
                seg_idx += 1

            temp_idx = seg_idx
            while temp_idx < num_segments and self.speaker_segments[temp_idx]['start'] < word_end:
                seg = self.speaker_segments[temp_idx]
                seg_start = seg['start']
                seg_end = seg['end']
                speaker = seg['speaker']

                if seg_start <= word_start < seg_end:
                    overlap = word_duration  # Full overlap
                else:
                    overlap_start = max(word_start, seg_start)
                    overlap_end = min(word_end, seg_end)
                    overlap = max(0.0, overlap_end - overlap_start)

                if overlap > 0:
                    speaker_overlap[speaker] = speaker_overlap.get(speaker, 0.0) + overlap

                temp_idx += 1

            assigned_speaker = max(speaker_overlap, key=speaker_overlap.get) if speaker_overlap else 'UNKNOWN'
            word_with_speaker = word.copy()
            word_with_speaker['speaker'] = assigned_speaker
            combined.append(word_with_speaker)

        self.combined_data = combined
        self.metadata["alignment_source"] = alignment_source
        print(f"Combined alignment and diarization data with {len(self.combined_data)} entries.")

    def save_as_json(self, output_file="all_transcript_data.json"):
        """
        Saves all transcript data to a JSON file.
        
        :param output_file: Path to the output JSON file.
        """
        if not self.combined_data:
            print("No combined data available to save. Ensure 'combine_alignment_and_diarization' is run first.")
            return

        data = {
            "audio_file_path": self.audio_file_path,
            "metadata": self.metadata,
            "transcript_text": self.transcript_text,
            "whisper_alignments": self.whisper_alignments,
            "forced_alignments": self.forced_alignments,
            "combined_data": self.combined_data,
            "utterance_data": self.combined_utterances,
            "speaker_segments": self.speaker_segments   
        }

        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            print(f"All transcript data successfully saved to '{output_file}'.")
        except Exception as e:
            print(f"Error saving JSON file: {e}")


class AudioTranscriber:
    """
    Handles transcription of audio chunks using Whisper.
    """
    def __init__(self, model = "openai/whisper-medium"):
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
            return_timestamps="word"  # Ensure word-level timestamps are returned
        )
        print(f"Initialized AudioTranscriber on device: {self.device}")

    def transcribe(self, audio_file: AudioFile):
        """
        Transcribes each audio chunk and populates the Transcript instance.
        
        :param transcript: Transcript instance to populate.
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
    """
    Handles forced alignment of transcripts with audio.
    """
    def __init__(self, device: str = None):
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
        Normalizes text using Uroman.
        
        :param text: Text to normalize.
        :return: Normalized text.
        """
        text = text.encode('utf-8').decode('utf-8')
        text = text.lower()
        text = text.replace("’", "'")
        text = unicodedata.normalize('NFC', text)
        text = re.sub("([^a-z' ])", " ", text)
        text = re.sub(' +', ' ', text)
        return text.strip()

    def align(self, audio_file: AudioFile):
        """
        Performs forced alignment and populates the Transcript instance.
        
        :param transcript: Transcript instance to populate.
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
    """
    Handles speaker diarization of audio files.
    """
    def __init__(self, hf_token: str, parameters: Dict, model = "pyannote/speaker-diarization-3.1"):
        """
        Initializes the SpeakerDiarizer.
        
        :param hf_token: Hugging Face token for accessing diarization models.
        :param parameters: Parameters for the diarization pipeline.
        """
        if torch.cuda.is_available():
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
            
        audio_file.register_model("Speaker Diarization", {
            "model": self.model,
            "device": str(self.device),
            "parameters": self.parameters,
            "speakers": num_speakers if num_speakers else "not specified"
        })


def process_audio_files(files: List[str],
                        hf_token: str,
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
                        num_speakers: int = 2,
                        output_folder: str = "output",
                        min_silence_len: int = 1000,
                        silence_thresh: int = -30,
                        min_length: int = 90000,
                        max_length: int = 150000,
                        timestamp_source: str = "whisper_alignments"):
    """
    Processes one or more audio files through the entire pipeline.
    
    :param files: List of file paths to process.
    :param hf_token: Hugging Face token for accessing diarization models.
    :param diarizer_params: Parameters for the SpeakerDiarizer model.
    :param num_speakers: Expected number of speakers.
    :param output_folder: Folder to save the output JSON files.
    :param min_silence_len: Minimum silence length for splitting (ms).
    :param silence_thresh: Silence threshold in dBFS for splitting.
    :param min_length: Minimum chunk length in ms.
    :param max_length: Maximum chunk length in ms.
    :param timestamp_source: Alignment source to use ('whisper_alignments' or 'forced_alignments').
    """
    Path(output_folder).mkdir(exist_ok=True)  # Create output folder if it doesn't exist
    print("Starting processing of audio files...")

    # Initialize processing classes
    transcriber = AudioTranscriber()
    aligner = ForcedAligner()
    diarizer = SpeakerDiarizer(hf_token, parameters=diarizer_params)

    for file_path in files:
        print(f"\nProcessing file: {file_path}")
        audio_file = AudioFile(file_path)

        # Step 1: Normalize audio
        print("Step 1/6: Normalizing audio...")
        audio_file.rms_normalization()

        # Step 2: Split audio into chunks based on silence
        print("Step 2/6: Splitting audio on silence...")
        audio_file.split_on_silence(
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            min_length=min_length,
            max_length=max_length
        )

        # Step 3: Transcribe audio chunks
        print("Step 3/6: Transcribing audio chunks...")
        transcriber.transcribe(audio_file)
        for idx, chunk in enumerate(audio_file.chunks, start=1):
            print(f"Chunk {idx} Transcript: {chunk.transcript}\n")

        # Step 4: Perform forced alignment
        print("Step 4/6: Performing forced alignment...")
        aligner.align(audio_file)        
        audio_file.combine_chunks()

        # Step 5: Perform speaker diarization
        print("Step 5/6: Performing speaker diarization...")
        diarizer.diarize(audio_file, num_speakers)

        # Step 6: Combine alignment and diarization data
        print("Step 6/6: Combining alignment and diarization data...")
        transcript = Transcript(audio_file)
        transcript.combine_alignment_and_diarization(timestamp_source)
        transcript.aggregate_to_utterances()

        # Save all data as JSON
        all_output_file = Path(output_folder) / f"{Path(file_path).stem}_all_outputs.json"
        print(f"Saving results to: {all_output_file}")
        transcript.save_as_json(all_output_file)
        print(f"Finished processing: {file_path}\n{'-' * 60}")

    del transcriber
    del aligner
    del diarizer

    print("All files have been processed.")


# Example Usage
if __name__ == "__main__":
    import os

    # Define input and output paths
    audio_file_path = "audio.wav"  # Replace with your actual audio file path
    output_directory = "output"

    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # List of files to process
    files_to_process = [audio_file_path]

    # Hugging Face token (replace with your actual token)
    hugging_face_token = "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"

    # Process the audio files
    process_audio_files(
        files=files_to_process,
        hf_token=hugging_face_token,
        output_folder=output_directory,
        num_speakers=2
    )