# Standard Library Imports
import io
import re
import unicodedata
from typing import List, Dict
from pathlib import Path
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
import textgrids
import json


def detect_silence_intervals(audio_segment, min_silence_len=1000, silence_thresh=-40):
    """
    Detects silent intervals in the audio segment.
    Returns a list of [start_ms, end_ms] pairs representing silence periods.
    """
    return detect_silence(
        audio_segment,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh
    )


def get_splitting_points(silence_ranges, audio_length_ms):
    """
    Computes splitting points based on silence ranges.
    Returns a sorted list of splitting points in milliseconds.
    """
    # Use list comprehension for faster execution
    splitting_points = [0] + [(start + end) // 2 for start, end in silence_ranges] + [audio_length_ms]
    return splitting_points


def create_initial_chunks(splitting_points):
    """
    Creates initial chunks based on splitting points.
    Returns a list of (start_ms, end_ms) tuples.
    """
    # Utilize zip for efficient pairing
    return list(zip(splitting_points[:-1], splitting_points[1:]))


def adjust_intervals_by_length(intervals, min_length=30000, max_length=180000):
    adjusted_intervals = []
    buffer_start, buffer_end = intervals[0]

    for start, end in intervals[1:]:
        buffer_end = end
        buffer_length = buffer_end - buffer_start

        if buffer_length < min_length:
            # Merge with the next interval by extending the buffer
            continue
        else:
            # Split the buffer if it exceeds max_length
            if buffer_length > max_length:
                num_splits = int(np.ceil(buffer_length / max_length))
                split_size = int(np.ceil(buffer_length / num_splits))
                for i in range(num_splits):
                    split_start = buffer_start + i * split_size
                    split_end = min(buffer_start + (i + 1) * split_size, buffer_end)
                    adjusted_intervals.append((split_start, split_end))
            else:
                adjusted_intervals.append((buffer_start, buffer_end))
            # Correctly update buffer_start to prevent overlap
            buffer_start = buffer_end  # Start from the end of the last buffer

    # Handle any remaining buffer
    if buffer_end > buffer_start:
        buffer_length = buffer_end - buffer_start
        if buffer_length >= min_length:
            if buffer_length > max_length:
                num_splits = int(np.ceil(buffer_length / max_length))
                split_size = int(np.ceil(buffer_length / num_splits))
                for i in range(num_splits):
                    split_start = buffer_start + i * split_size
                    split_end = min(buffer_start + (i + 1) * split_size, buffer_end)
                    adjusted_intervals.append((split_start, split_end))
            else:
                adjusted_intervals.append((buffer_start, buffer_end))
        else:
            # Decide how to handle intervals shorter than min_length
            pass

    return adjusted_intervals


def split_audio_by_intervals(audio_segment, intervals):
    """
    Splits the audio segment into chunks based on the provided intervals.
    Returns a list of (chunk_audio, start_time_ms, end_time_ms) tuples.
    """
    return [(audio_segment[start_ms:end_ms], start_ms, end_ms) for start_ms, end_ms in intervals]


class Chunk:
    def __init__(self, audio_segment, start_time):
        self.audio_segment = audio_segment
        self.start_time = start_time  # Start time in the original audio (seconds)
        self.transcript = ""
        self.whisper_alignments = []
        self.forced_alignments = []


# Define the AudioFile class
class AudioFile:
    def __init__(self, file_path: str, target_rms_db: float = -20):
        self.file_path = file_path
        self.target_rms_db = target_rms_db
        self.normalized_path = None
        self.audio = None
        self.sample_rate = None
        self.chunks: List[Chunk] = []
        self.transcript_text = ""
        self.whisper_alignments = []
        self.forced_alignments = []
        self.speaker_segments = []
        self.combined_data = []
        self.combined_utterances = []
        self.load_audio()

    def load_audio(self):
        # Load the audio file
        self.audio, self.sample_rate = librosa.load(self.file_path, sr=None)
        print(f"Loaded audio file {self.file_path}")

    def rms_normalization(self):
        # Convert target RMS dB to linear scale
        target_rms = 10 ** (self.target_rms_db / 20)

        # Calculate current RMS of the audio
        rms = np.sqrt(np.mean(self.audio ** 2))

        # Calculate gain required to reach target RMS
        gain = target_rms / rms
        normalized_audio = self.audio * gain

        # Save the normalized audio to a temporary file
        self.normalized_path = self.file_path.replace(".wav", "_normalized.wav")
        sf.write(self.normalized_path, normalized_audio, self.sample_rate)
        print(f"Normalized audio saved as {self.normalized_path}")

    def split_on_silence(self, min_silence_len=1000, silence_thresh=-30,
                        min_length=30000, max_length=180000):
        # Load the normalized audio using pydub
        audio_segment = AudioSegment.from_file(self.normalized_path)
        audio_length_ms = len(audio_segment)

        # Step 1: Detect silence intervals
        silence_ranges = detect_silence_intervals(
            audio_segment,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh
        )

        # Step 2: Identify splitting points
        splitting_points = get_splitting_points(silence_ranges, audio_length_ms)

        # Step 3: Create initial chunks covering the entire audio
        initial_intervals = create_initial_chunks(splitting_points)

        # Step 4: Adjust intervals based on length constraints
        adjusted_intervals = adjust_intervals_by_length(
            initial_intervals,
            min_length=min_length,
            max_length=max_length
        )

        # Step 5: Extract chunks
        chunks_with_timestamps = split_audio_by_intervals(audio_segment, adjusted_intervals)

        # Create Chunk instances with accurate start times
        self.chunks = []
        for chunk_audio, start_i, end_i in chunks_with_timestamps:
            self.chunks.append(Chunk(chunk_audio, start_i / 1000.0))  # Convert ms to seconds
        print(f"Total chunks after splitting: {len(self.chunks)}")

    def combine_chunks(self):
        # Combine transcripts
        self.transcript_text = " ".join([chunk.transcript for chunk in self.chunks])
        # Combine word alignments
        for chunk in self.chunks:
            self.whisper_alignments.extend(chunk.whisper_alignments)
            self.forced_alignments.extend(chunk.forced_alignments)
            
    def aggregate_to_utterances(self):
        if not self.combined_data:
            print("No combined data available to adjust.")
            return

        utterances = []
        current_utterance = {
            "text": "",
            "start_time": None,
            "end_time": None,
            "speakers": {}
        }
        
        sentence_endings = re.compile(r'[.?!]$')
        print(self.combined_data)
        for word_data in self.combined_data:
            word = word_data["word"]
            start_time = word_data["start_time"]
            end_time = word_data["end_time"]
            speaker = word_data["speaker"]
            
            # Initialize the current utterance start time if not already set
            if current_utterance["start_time"] is None:
                current_utterance["start_time"] = start_time
            
            # Append the word to the current utterance text
            current_utterance["text"] += ("" if current_utterance["text"] == "" else " ") + word
            
            # Update the end time of the current utterance
            current_utterance["end_time"] = end_time
            
            # Update the speaker count for this utterance
            if speaker not in current_utterance["speakers"]:
                current_utterance["speakers"][speaker] = 0
            current_utterance["speakers"][speaker] += 1
            
            # Check if this word ends the sentence
            if sentence_endings.search(word):
                # Determine the majority speaker
                majority_speaker, majority_count = max(
                    current_utterance["speakers"].items(), key=lambda item: item[1]
                )
                total_words = sum(current_utterance["speakers"].values())
                confidence = majority_count / total_words
                
                # Append the completed utterance
                utterances.append({
                    "text": current_utterance["text"],
                    "start_time": current_utterance["start_time"],
                    "end_time": current_utterance["end_time"],
                    "speaker": majority_speaker,
                    "confidence": round(confidence, 2),
                })
                
                # Reset the current utterance
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
            confidence = majority_count / total_words
            
            utterances.append({
                "text": current_utterance["text"],
                "start_time": current_utterance["start_time"],
                "end_time": current_utterance["end_time"],
                "speaker": majority_speaker,
                "confidence": round(confidence, 2),
            })
        
        self.combined_utterances = utterances
            
    def combine_alignment_and_diarization(self, alignment_source: str):
        """
        Combines alignment and diarization data by assigning speaker labels to each word.

        Parameters:
        - alignment_source (str): The alignment data to use. Options:
            - 'whisper_alignments'
            - 'forced_alignments'

        Updates:
        - self.combined_data: List of words with assigned speakers.
        """
        # Validate the alignment_source
        if alignment_source not in ['whisper_alignments', 'forced_alignments']:
            raise ValueError("Invalid alignment_source. Choose 'whisper_alignments' or 'forced_alignments'.")

        # Select the appropriate alignment list
        alignment = getattr(self, alignment_source, None)
        if alignment is None:
            raise ValueError(f"The alignment source '{alignment_source}' does not exist in the AudioFile object.")

        if not self.speaker_segments:
            print("No speaker segments available for diarization. All words will be labeled as 'UNKNOWN'.")
            # Assign 'UNKNOWN' to all words
            self.combined_data = [
                {**word, 'speaker': 'UNKNOWN'} for word in alignment
            ]
            return

        combined = []
        seg_idx = 0
        num_segments = len(self.speaker_segments)
        speaker_segments = self.speaker_segments

        for word in alignment:
            word_start = word['start_time']
            word_end = word['end_time']
            word_duration = max(1e-6, word_end - word_start)  # Avoid zero-duration

            speaker_overlap = {}

            while seg_idx < num_segments and speaker_segments[seg_idx]['end'] < word_start:
                seg_idx += 1

            temp_idx = seg_idx
            while temp_idx < num_segments and speaker_segments[temp_idx]['start'] < word_end:
                seg = speaker_segments[temp_idx]
                seg_start = seg['start']
                seg_end = seg['end']
                speaker = seg['speaker']

                if seg_start <= word_start < seg_end:  # Handle zero-duration case
                    overlap = word_duration  # Treat as full overlap for zero-duration
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
        print(self.combined_data)
        print(f"Combined alignment and diarization data with {len(self.combined_data)} entries.")
        # self.adjust_overlapping_intervals()
        self.aggregate_to_utterances()


    def save_all_data_as_json(self, output_file="all_audio_data.json"):
        """
        Saves multiple attributes of the AudioFile instance into a single JSON file.

        Parameters:
        - output_file: str, optional
            The path to save the JSON file. Defaults to 'all_audio_data.json'.
        """
        # Check if combined data exists
        if not self.combined_data:
            print("No combined data available to save. Ensure 'combine_alignment_and_diarization' is run first.")
            return

        # Prepare the data dictionary
        data = {
            "transcript_text": self.transcript_text,
            "whisper_alignments": self.whisper_alignments,
            "forced_alignments": self.forced_alignments,
            "speaker_segments": self.speaker_segments,
            "combined_data": self.combined_data,
            "utterance_data": self.combined_utterances
        }

        # Convert the data to JSON format
        try:
            json_output = json.dumps(data, indent=4)
        except TypeError as e:
            print(f"Error serializing data to JSON: {e}")
            return

        # Optional: Warn if output_file already exists
        if Path(output_file).exists():
            print(f"Warning: '{output_file}' already exists and will be overwritten.")

        # Save JSON to the specified file
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(json_output)
            print(f"All audio data successfully saved to '{output_file}'.")
        except Exception as e:
            print(f"Error saving JSON file: {e}")
        
        print(self.combined_data)

# Define the AudioTranscriber class
class AudioTranscriber:
    """
    Handles transcription of audio chunks using Whisper.
    """
    def __init__(self):
        # Use 'cuda' if available, else 'mps' for Apple devices, else 'cpu'
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        self.transcriber = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-medium",
            device=self.device,
            return_timestamps="word"  # Ensure word-level timestamps are returned
        )
        print(f"Initialized AudioTranscriber on device: {self.device}")

    def transcribe(self, audio_file: AudioFile):
        """
        Transcribes each speech chunk and extracts word-level timestamps.
        """
        print("Transcribing audio chunks...")
        for idx, chunk in enumerate(audio_file.chunks, start=1):
            try:
                with io.BytesIO() as wav_io:
                    chunk.audio_segment.export(wav_io, format="wav")
                    wav_io.seek(0)
                    # Pass the audio data to the transcriber
                    transcription_result = self.transcriber(wav_io.read())
                
                chunk.transcript = transcription_result.get('text', "").strip()
                
                # Extract word alignments from 'chunks' instead of 'words'
                raw_chunks = transcription_result.get('chunks', [])
                clean_chunks = []
                for word_info in raw_chunks:
                    # Ensure 'timestamp' exists and has two elements
                    if 'timestamp' in word_info and len(word_info['timestamp']) == 2:
                        # Convert timestamps to float before addition
                        start_time = float(word_info['timestamp'][0]) + chunk.start_time
                        end_time = float(word_info['timestamp'][1]) + chunk.start_time
                        word_text = word_info.get('text', "").strip()
                        if word_text:  # Ensure word_text is not empty
                            clean_chunks.append({
                                "word": word_text,
                                "start_time": start_time,
                                "end_time": end_time
                            })
                chunk.whisper_alignments = clean_chunks
                print(f"Transcribed chunk {idx} successfully with {len(clean_chunks)} words.")
            except Exception as e:
                print(f"Error during transcription of chunk {idx}: {e}")
                chunk.transcript = ""
                chunk.whisper_alignments = []


# Define the ForcedAligner class
class ForcedAligner:
    def __init__(self, device: str = None):
        # Forcing CPU device due to potential compatibility issues
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        self.bundle = torchaudio.pipelines.MMS_FA
        self.model = self.bundle.get_model().to(self.device)
        self.tokenizer = self.bundle.get_tokenizer()
        self.aligner = self.bundle.get_aligner()
        self.uroman = ur.Uroman()
        self.sample_rate = self.bundle.sample_rate
        print(f"Initialized ForcedAligner on device: {self.device}")

    def normalize_uroman(self, text: str) -> str:
        text = text.encode('utf-8').decode('utf-8')
        text = text.lower()
        text = text.replace("’", "'")
        text = unicodedata.normalize('NFC', text)
        text = re.sub("([^a-z' ])", " ", text)
        text = re.sub(' +', ' ', text)
        return text.strip()

    def align(self, audio_file: AudioFile):
        for idx, chunk in enumerate(audio_file.chunks):
            print(f"Aligning chunk {idx + 1}/{len(audio_file.chunks)}")
            # Export chunk to a WAV file in memory
            with io.BytesIO() as wav_io:
                chunk.audio_segment.export(wav_io, format="wav")
                wav_io.seek(0)
                waveform, sample_rate = torchaudio.load(wav_io)
            # Resample if needed
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


# Define the SpeakerDiarizer class
class SpeakerDiarizer:
    def __init__(self, hf_token: str, parameters):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        self.diarization_pipeline = DiarizationPipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        print("Initializing SpeakerDiarizer with parameters...")
        self.diarization_pipeline.instantiate(parameters)
        self.diarization_pipeline.to(self.device)
        print("Initialized SpeakerDiarizer successfully.")

    def diarize(self, audio_file: AudioFile, num_speakers: int = None):
        """
        Performs speaker diarization on the given audio file.
        
        Parameters:
        - audio_file (AudioFile): The audio file to diarize.
        - num_speakers (int, optional): The expected number of speakers.
        """
        print("Performing diarization on the entire audio file")
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
            
            # Prepare speaker segments
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


def process_audio_files(files, hf_token="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", diarizer_params={
                        "segmentation": {
                            "min_duration_off": 0.0,
                        },
                        "clustering": {
                            "method": "centroid",
                            "min_cluster_size": 12,
                            "threshold": 0.8,
                        }}, 
                        
                        num_speakers = 2,
                        
                        output_folder="output", 
                        min_silence_len=1000, 
                        silence_thresh=-30, 
                        min_length=90000, 
                        max_length=150000,
                        
                        timestamp_source = "whisper_alignments"
                        ):
    """
    Processes one or more audio files through all steps of the pipeline.

    Parameters:
    - files: str or List[str]
        A single file path or a list of file paths to process.
    - hf_token: str, optional
        Hugging Face token for accessing diarization models.
    - diarizer_params: dict, optional
        Parameters for the SpeakerDiarizer model.
    - output_folder: str, optional
        Folder to save the output JSON files. Defaults to 'output'.
    - min_silence_len: int, optional
        Minimum silence length for splitting. Defaults to 1000ms.
    - silence_thresh: int, optional
        Silence threshold in dB for splitting. Defaults to -40dB.
    - min_length: int, optional
        Minimum chunk length in milliseconds. Defaults to 30000ms.
    - max_length: int, optional
        Maximum chunk length in milliseconds. Defaults to 180000ms.
    """
    
    Path(output_folder).mkdir(exist_ok=True)  # Create output folder if it doesn't exist

    # Ensure `files` is a list
    if isinstance(files, str):
        files = [files]

    # Initialize necessary classes
    transcriber = AudioTranscriber()
    aligner = ForcedAligner()
    diarizer = SpeakerDiarizer(hf_token, parameters=diarizer_params)

    # Process each file
    for file_path in files:
        print(f"Processing file: {file_path}")

        # Step 1: Load and normalize audio
        audio_file = AudioFile(file_path)
        print("Step 1/6: Normalizing audio...")
        audio_file.rms_normalization()

        # Step 2: Split audio into chunks
        print("Step 2/6: Splitting audio on silence...")
        audio_file.split_on_silence(
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            min_length=min_length,
            max_length=max_length
        )

        # Step 3: Transcribe chunks
        print("Step 3/6: Transcribing audio chunks...")
        transcriber.transcribe(audio_file)
        
        for chunk in audio_file.chunks:
            print(chunk.transcript)
            print("\n")

        # Step 4: Align transcription with audio
        print("Step 4/6: Performing forced alignment...")
        aligner.align(audio_file)

        # Step 5: Perform speaker diarization
        print("Step 5/6: Performing speaker diarization...")
        diarizer.diarize(audio_file, num_speakers)

        # Step 6: Combine alignment and diarization
        print("Step 6/6: Combining alignment and diarization data...")
        audio_file.combine_chunks()
        audio_file.combine_alignment_and_diarization(timestamp_source)

        # Save output
        all_output_file = Path(output_folder) / f"{Path(file_path).stem}_all_outputs.json"
        print(f"Saving results to: {all_output_file}")

        # audio_file.create_textgrid(f"{Path(file_path).stem}_output.TextGrid")
        audio_file.save_all_data_as_json(all_output_file)
        print(f"Finished processing: {file_path}\n{'-' * 40}")


# Example Usage
if __name__ == "__main__":
    import os

    # Define input and output paths
    audio_file_path = "audio.wav"  # Replace with your actual audio file path
    output_directory = "output"

    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # List of files to process
    files = [audio_file_path]

    # Process the audio files
    process_audio_files(
        files=files,
        hf_token="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",  # Replace with your actual Hugging Face token
        output_folder=output_directory,
    )