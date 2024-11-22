import os
import json
import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T
import uroman as ur
import re
import unicodedata
from typing import List, Dict
from transformers import pipeline
from pyannote.audio import Pipeline as DiarizationPipeline
from sklearn.cluster import AgglomerativeClustering


class AudioFile:
    def __init__(self, file_path: str, target_rms_db: float = -20):
        self.file_path = file_path
        self.target_rms_db = target_rms_db
        self.normalized_path = None
        self.audio = None
        self.sample_rate = None
        self.transcript_text = ""
        self.word_alignments = []
        self.speaker_segments = []
        self.combined_data = []
        self.whisper_word_alignments = []
        self.forced_alignment_output = []
        self.whisper_output = []
        self.diarization_output = []
        self.aggregated_output = []
        self.load_audio()

    def load_audio(self):
        self.audio, self.sample_rate = librosa.load(self.file_path, sr=None)
        print(f"Loaded audio file {self.file_path}")

    def rms_normalization(self):
        target_rms = 10 ** (self.target_rms_db / 20)
        rms = np.sqrt(np.mean(self.audio ** 2))
        gain = target_rms / rms
        normalized_audio = self.audio * gain
        self.normalized_path = self.file_path.replace(".wav", "_normalized.wav")
        sf.write(self.normalized_path, normalized_audio, self.sample_rate)
        print(f"Normalized audio saved as {self.normalized_path}")

    def aggregate_transcripts(self, whisper_transcriptions: List[Dict], forced_alignments: List[Dict]):
        print("Aggregating transcripts from Whisper and Forced Alignment")
        self.whisper_output = whisper_transcriptions
        self.forced_alignment_output = forced_alignments

        merged_words = {}
        for source in [whisper_transcriptions, forced_alignments]:
            for word_info in source:
                word = word_info['word'].lower()
                if word not in merged_words:
                    merged_words[word] = {'count': 0, 'start_time': 0.0, 'end_time': 0.0}
                merged_words[word]['count'] += 1
                merged_words[word]['start_time'] += word_info['start_time']
                merged_words[word]['end_time'] += word_info['end_time']

        aggregated_words = []
        for word, data in merged_words.items():
            avg_start = data['start_time'] / data['count']
            avg_end = data['end_time'] / data['count']
            aggregated_words.append({
                'word': word,
                'start_time': avg_start,
                'end_time': avg_end
            })

        aggregated_words.sort(key=lambda x: x['start_time'])
        self.aggregated_output = aggregated_words.copy()

        print("Assigning speakers to words based on diarization segments")
        for word in aggregated_words:
            word_start = word['start_time']
            word_end = word['end_time']
            assigned_speaker = self.assign_speaker(word_start, word_end)
            aggregated_entry = {
                'word': word['word'],
                'start_time': word_start,
                'end_time': word_end,
                'speaker': assigned_speaker
            }
            self.combined_data.append(aggregated_entry)
            self.aggregated_output.append(aggregated_entry)

    def assign_speaker(self, word_start: float, word_end: float) -> str:
        max_overlap = 0
        assigned_speaker = "Unknown"
        for segment in self.speaker_segments:
            overlap_start = max(word_start, segment['start'])
            overlap_end = min(word_end, segment['end'])
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap > max_overlap:
                max_overlap = overlap
                assigned_speaker = segment['speaker']
        return assigned_speaker

    def save_individual_outputs(self, output_dir: str):
        try:
            os.makedirs(output_dir, exist_ok=True)
            base_filename = os.path.splitext(os.path.basename(self.file_path))[0]

            whisper_path = os.path.join(output_dir, f"{base_filename}_whisper_timestamps.json")
            with open(whisper_path, 'w', encoding='utf-8') as f:
                json.dump(self.whisper_output, f, ensure_ascii=False, indent=4)
            print(f"Saved Whisper timestamps to {whisper_path}")

            forced_align_path = os.path.join(output_dir, f"{base_filename}_forced_alignment_timestamps.json")
            with open(forced_align_path, 'w', encoding='utf-8') as f:
                json.dump(self.forced_alignment_output, f, ensure_ascii=False, indent=4)
            print(f"Saved Forced Alignment timestamps to {forced_align_path}")

            diarization_path = os.path.join(output_dir, f"{base_filename}_speaker_segments.json")
            with open(diarization_path, 'w', encoding='utf-8') as f:
                json.dump(self.speaker_segments, f, ensure_ascii=False, indent=4)
            print(f"Saved Speaker Diarization segments to {diarization_path}")

        except Exception as e:
            print(f"Error saving individual outputs: {e}")

    def save_aggregated_output(self, output_dir: str):
        try:
            os.makedirs(output_dir, exist_ok=True)
            base_filename = os.path.splitext(os.path.basename(self.file_path))[0]
            aggregated_path = os.path.join(output_dir, f"{base_filename}_aggregated_transcript.json")
            with open(aggregated_path, 'w', encoding='utf-8') as f:
                json.dump(self.aggregated_output, f, ensure_ascii=False, indent=4)
            print(f"Saved Aggregated Transcript to {aggregated_path}")
        except Exception as e:
            print(f"Error saving aggregated output: {e}")


class AudioTranscriber:
    def __init__(self, device: str = None):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.transcriber = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-medium",
            device=self.device
        )
        print(f"Initialized AudioTranscriber on device: {self.device}")

    def transcribe(self, audio_file: AudioFile):
        print("Transcribing the entire audio file")
        try:
            with open(audio_file.normalized_path, 'rb') as f:
                audio_data = f.read()
            transcription_result = self.transcriber(
                audio_data,
                return_timestamps="word",

            )
            audio_file.transcript_text = transcription_result['text'].strip()
            audio_file.whisper_word_alignments = transcription_result.get('words', [])
            print("Transcription completed")
        except Exception as e:
            print(f"Error during transcription: {e}")


class ForcedAligner:
    def __init__(self, device: str = None):
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
        print("Performing forced alignment on the entire audio")
        try:
            waveform, sample_rate = torchaudio.load(audio_file.normalized_path)
            if sample_rate != self.sample_rate:
                resampler = T.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
                waveform = resampler(waveform)
                sample_rate = self.sample_rate

            text_roman = self.uroman.romanize_string(audio_file.transcript_text)
            text_normalized = self.normalize_uroman(text_roman)
            transcript_list = text_normalized.split()

            tokens = self.tokenizer(transcript_list)

            with torch.inference_mode():
                emission, _ = self.model(waveform.to(self.device))
                token_spans = self.aligner(emission[0], tokens)

            num_frames = emission.size(1)
            ratio = waveform.size(1) / num_frames

            word_alignments = []
            for spans, word in zip(token_spans, transcript_list):
                start_sec = spans[0].start * ratio / sample_rate
                end_sec = spans[-1].end * ratio / sample_rate
                word_alignments.append({
                    "word": word,
                    "start_time": start_sec,
                    "end_time": end_sec
                })

            audio_file.word_alignments = word_alignments
            print("Forced alignment completed")
        except Exception as e:
            print(f"Error during forced alignment: {e}")


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
        self.diarization_pipeline.instantiate(parameters)
        self.diarization_pipeline.to(self.device)
        print("Initialized SpeakerDiarizer")

    def diarize(self, audio_file: AudioFile):
        print("Performing diarization on the entire audio file")
        try:
            diarization_result = self.diarization_pipeline(audio_file.normalized_path)
            audio_file.speaker_segments = []
            for segment, _, speaker in diarization_result.itertracks(yield_label=True):
                audio_file.speaker_segments.append({
                    "start": segment.start,
                    "end": segment.end,
                    "speaker": speaker
                })
            print("Diarization completed")
        except Exception as e:
            print(f"Error during diarization: {e}")


def process_audio(file_path: str, hf_token: str, output_dir: str,
                  diarizer_params={
                    "segmentation": {
                        "min_duration_off": 0.0,
                    },
                    "clustering": {
                        "method": "centroid",
                        "min_cluster_size": 12,
                        "threshold": 0.7,
                    }}, ):
    
    audio_file = AudioFile(file_path)
    audio_file.rms_normalization()

    transcriber = AudioTranscriber()
    transcriber.transcribe(audio_file)

    aligner = ForcedAligner()
    aligner.align(audio_file)

    diarizer = SpeakerDiarizer(hf_token, diarizer_params)
    diarizer.diarize(audio_file)

    audio_file.aggregate_transcripts(
        whisper_transcriptions=audio_file.whisper_word_alignments,
        forced_alignments=audio_file.word_alignments
    )

    audio_file.save_individual_outputs(output_dir=output_dir)

    audio_file.save_aggregated_output(output_dir=output_dir)

    try:
        combined_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(file_path))[0]}_combined_data.json")
        with open(combined_path, 'w', encoding='utf-8') as f:
            json.dump(audio_file.combined_data, f, ensure_ascii=False, indent=4)
        print(f"Saved Combined Data to {combined_path}")
    except Exception as e:
        print(f"Error saving combined data: {e}")

    print("\nFinal Aggregated and Diarized Transcript:")
    for entry in audio_file.combined_data:
        print(f"[{entry['start_time']:.2f}-{entry['end_time']:.2f}] {entry['speaker']}: {entry['word']}")


if __name__ == "__main__":
    file_path = "audio.wav"
    output_dir = "output"
    hf_token = ""
    process_audio(file_path, hf_token, output_dir)


