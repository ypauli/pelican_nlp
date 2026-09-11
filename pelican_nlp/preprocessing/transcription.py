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

from pelican_nlp.utils.model_cache import huggingface_from_pretrained_kwargs
from pelican_nlp.config import debug_print
from pelican_nlp.utils.progress import active_reporter

# Suppress FutureWarning from transformers about 'inputs' vs 'input_features'
# This is a deprecation warning from the transformers library that will be fixed in a future version
warnings.filterwarnings("ignore", category=FutureWarning, message=".*input name `inputs` is deprecated.*")
# pyannote: TF32 is turned off on purpose; empty/short windows also warn on std().
warnings.filterwarnings("ignore", message=".*TensorFloat-32 \\(TF32\\) has been disabled.*")
warnings.filterwarnings("ignore", message=".*std\\(\\): degrees of freedom is <= 0.*")


def _is_word_timestamp_merge_error(exc: BaseException) -> bool:
    msg = str(exc)
    return isinstance(exc, TypeError) and "NoneType" in msg and "<=" in msg


def _clear_cuda():
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()


def _asr_torch_dtype(model_name, device):
    """Use float16 only when CUDA cannot hold fp32 weights plus ASR working memory."""
    if getattr(device, "type", None) != "cuda":
        return None
    from pelican_nlp.utils.gpu_budget import estimate_pretrained_weight_bytes, gpu_can_hold

    fp32_bytes = estimate_pretrained_weight_bytes(model_name, bytes_per_param=4)
    if fp32_bytes is None:
        return None
    # Word-level timestamps need working memory on the order of the weights.
    if gpu_can_hold(fp32_bytes * 2):
        return None
    return torch.float16


class AudioTranscriber:
    """Handles transcription of audio chunks using Whisper."""
    
    def __init__(self, model="openai/whisper-medium"):
        """
        Initialize the AudioTranscriber.
        
        :param model: Whisper model to use for transcription.
        """
        from pelican_nlp.utils.gpu_budget import runtime_torch_device

        self.device = runtime_torch_device(min_free_gb=2.0, allow_mps=True)
        self.model = model
        self._torch_dtype = _asr_torch_dtype(model, self.device)
        if self._torch_dtype == torch.float16:
            active_reporter().status(
                f"{model} fp32 weights do not fit the GPU budget; loading float16."
            )
        self.transcriber = self._build_pipeline()
        dtype_note = " (float16)" if self._torch_dtype == torch.float16 else ""
        debug_print(f"Initialized AudioTranscriber on device: {self.device}{dtype_note}")

    def _build_pipeline(self):
        pipeline_kwargs = {
            "model": self.model,
            "device": self.device,
            "return_timestamps": "word",
            "model_kwargs": huggingface_from_pretrained_kwargs(),
        }
        if self._torch_dtype is not None:
            pipeline_kwargs["torch_dtype"] = self._torch_dtype
        return pipeline("automatic-speech-recognition", **pipeline_kwargs)

    def _release_pipeline(self):
        """Move the current ASR pipeline off GPU without dropping the attribute."""
        pipe = getattr(self, "transcriber", None)
        self.transcriber = None
        if pipe is None:
            _clear_cuda()
            return
        model = getattr(pipe, "model", None)
        if model is not None:
            try:
                model.to("cpu")
            except Exception:
                pass
        del model, pipe
        _clear_cuda()

    def _ensure_float16(self):
        """Switch an fp32 ASR pipeline to float16. Keep ``self.transcriber`` assigned."""
        if self._torch_dtype == torch.float16:
            return False
        if getattr(self.device, "type", None) != "cuda":
            return False

        active_reporter().status(
            f"{self.model} ran out of GPU memory in fp32; switching to float16."
        )
        pipe = getattr(self, "transcriber", None)
        model = getattr(pipe, "model", None) if pipe is not None else None
        if model is not None and hasattr(model, "to"):
            try:
                if hasattr(model, "half"):
                    converted = model.half()
                else:
                    converted = model.to(dtype=torch.float16)
                pipe.model = converted
                if hasattr(pipe, "torch_dtype"):
                    pipe.torch_dtype = torch.float16
                self._torch_dtype = torch.float16
                _clear_cuda()
                return True
            except Exception:
                pass

        self._torch_dtype = torch.float16
        self._release_pipeline()
        try:
            self.transcriber = self._build_pipeline()
        except Exception:
            self.transcriber = None
            raise
        return True

    def park_on_cpu(self):
        """Free GPU VRAM so MMS/pyannote can load after ASR."""
        pipe = getattr(self, "transcriber", None)
        model = getattr(pipe, "model", None) if pipe is not None else None
        if model is None:
            return
        try:
            model.to("cpu")
        except Exception:
            pass
        _clear_cuda()

    def restore_to_device(self):
        """Move Whisper back onto the ASR device for the next file."""
        pipe = getattr(self, "transcriber", None)
        model = getattr(pipe, "model", None) if pipe is not None else None
        if model is None:
            return
        kwargs = {}
        if self._torch_dtype == torch.float16:
            kwargs["dtype"] = torch.float16
        model.to(device=self.device, **kwargs)

    def _invoke_asr(self, wav_bytes, asr_kwargs):
        try:
            return self.transcriber(wav_bytes, **asr_kwargs), True
        except TypeError as exc:
            if not _is_word_timestamp_merge_error(exc):
                raise
            fallback = dict(asr_kwargs)
            fallback["return_timestamps"] = True
            return self.transcriber(wav_bytes, **fallback), False

    def _transcribe_audio(self, wav_bytes, chunk_duration):
        asr_kwargs = {}
        if chunk_duration > 30:
            asr_kwargs["chunk_length_s"] = 30
        try:
            return self._invoke_asr(wav_bytes, asr_kwargs)
        except torch.cuda.OutOfMemoryError:
            _clear_cuda()
            asr_kwargs["chunk_length_s"] = 30
            try:
                return self._invoke_asr(wav_bytes, asr_kwargs)
            except torch.cuda.OutOfMemoryError:
                _clear_cuda()
                if not self._ensure_float16():
                    raise
                if getattr(self, "transcriber", None) is None:
                    raise
                asr_kwargs["chunk_length_s"] = 15
                return self._invoke_asr(wav_bytes, asr_kwargs)

    @staticmethod
    def _infer_uniform_word_timings(text: str, chunk_start: float, chunk_duration: float):
        """Create fallback word timings when timestamped chunks are unavailable."""
        words = text.split()
        if not words:
            return []

        # Keep a minimal span for very short chunks.
        duration = max(chunk_duration, 0.2)
        per_word = duration / len(words)
        alignments = []
        for idx, word in enumerate(words):
            start = chunk_start + (idx * per_word)
            end = chunk_start + ((idx + 1) * per_word)
            alignments.append({
                "word": word,
                "start_time": start,
                "end_time": end
            })
        return alignments

    def transcribe(self, audio_file):
        """
        Transcribe each audio chunk and populate the AudioFile instance.
        
        :param audio_file: AudioFile instance containing audio chunks.
        """
        debug_print("Starting transcription of audio chunks...")
        self.restore_to_device()
        n_chunks = len(audio_file.chunks)
        for idx, chunk in enumerate(audio_file.chunks, start=1):
            active_reporter().set_postfix(f"transcribe {idx}/{n_chunks}")
            try:
                if getattr(self, "transcriber", None) is None:
                    self.transcriber = self._build_pipeline()
                with io.BytesIO() as wav_io:
                    chunk.audio_segment.export(wav_io, format="wav")
                    wav_io.seek(0)
                    wav_bytes = wav_io.read()
                chunk_duration = len(chunk.audio_segment) / 1000.0
                transcription_result, word_timestamps = self._transcribe_audio(
                    wav_bytes, chunk_duration
                )

                # Assign transcript to the chunk
                chunk.transcript = transcription_result.get('text', "").strip()

                # Extract word alignments
                raw_chunks = transcription_result.get('chunks', []) if word_timestamps else []
                clean_chunks = []
                prev_end_rel = 0.0
                for word_info in raw_chunks:
                    word_text = word_info.get('text', "").strip()
                    timestamp = word_info.get('timestamp')
                    if not word_text or not isinstance(timestamp, (list, tuple)) or len(timestamp) != 2:
                        continue

                    raw_start, raw_end = timestamp
                    if raw_start is None and raw_end is None:
                        continue

                    # Whisper can occasionally return one-sided timestamps; recover instead of failing chunk.
                    if raw_start is None:
                        raw_start = prev_end_rel
                    if raw_end is None:
                        raw_end = min(raw_start + 0.25, chunk_duration)

                    try:
                        start_rel = max(0.0, float(raw_start))
                        end_rel = max(start_rel + 1e-3, float(raw_end))
                    except (TypeError, ValueError):
                        continue

                    start_rel = min(start_rel, chunk_duration)
                    end_rel = min(end_rel, chunk_duration)
                    if end_rel <= start_rel:
                        end_rel = min(start_rel + 1e-3, chunk_duration)
                        if end_rel <= start_rel:
                            continue

                    start_time = chunk.start_time + start_rel
                    end_time = chunk.start_time + end_rel
                    clean_chunks.append({
                        "word": word_text,
                        "start_time": start_time,
                        "end_time": end_time
                    })
                    prev_end_rel = end_rel

                # Fallback: reconstruct transcript from word chunks when top-level text is empty.
                if not chunk.transcript and raw_chunks:
                    reconstructed_words = [
                        word_info.get('text', "").strip()
                        for word_info in raw_chunks
                        if word_info.get('text', "").strip()
                    ]
                    if reconstructed_words:
                        chunk.transcript = " ".join(reconstructed_words).strip()

                if not clean_chunks and chunk.transcript:
                    clean_chunks = self._infer_uniform_word_timings(
                        text=chunk.transcript,
                        chunk_start=chunk.start_time,
                        chunk_duration=chunk_duration
                    )
                chunk.whisper_alignments = clean_chunks
                if not chunk.transcript:
                    debug_print(f"Warning: Transcription result for chunk {idx} was empty.")
                debug_print(f"Transcribed chunk {idx} with {len(clean_chunks)} words.")
            except Exception as e:
                debug_print(f"Error during transcription of chunk {idx}: {e}")
                active_reporter().warn(f"Error during transcription of chunk {idx}: {e}")
                chunk.transcript = ""
                chunk.whisper_alignments = []
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        audio_file.register_model("Transcription", {
            "model": self.model,
            "device": str(self.device),
            "dtype": "float16" if self._torch_dtype == torch.float16 else "float32",
        })


class ForcedAligner:
    """Handles forced alignment of transcripts with audio."""
    
    def __init__(self, device: str = None):
        """
        Initialize the ForcedAligner.
        
        :param device: Device to use for alignment (auto-detected if None).
        """
        from pelican_nlp.utils.gpu_budget import runtime_torch_device

        self.device = runtime_torch_device(min_free_gb=2.0, allow_mps=False)

        # Initialize forced aligner components
        self.bundle = torchaudio.pipelines.MMS_FA
        self.model = self.bundle.get_model().to(self.device)
        self.tokenizer = self.bundle.get_tokenizer()
        self.aligner = self.bundle.get_aligner()
        self.uroman = ur.Uroman()
        self.sample_rate = self.bundle.sample_rate
        debug_print(f"Initialized ForcedAligner on device: {self.device}")

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
        debug_print("Starting forced alignment of transcripts...")
        n_chunks = len(audio_file.chunks)
        for idx, chunk in enumerate(audio_file.chunks, start=1):
            active_reporter().set_postfix(f"align {idx}/{n_chunks}")
            try:
                if not chunk.transcript or not chunk.transcript.strip():
                    debug_print(f"Skipping alignment for chunk {idx}: empty transcript.")
                    continue

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
                if not transcript_list:
                    debug_print(f"Skipping alignment for chunk {idx}: transcript has no alignable words.")
                    continue
                tokens = self.tokenizer(transcript_list)
                if not tokens:
                    debug_print(f"Skipping alignment for chunk {idx}: tokenizer returned no tokens.")
                    continue

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
                debug_print(f"Aligned chunk {idx} successfully.")
            except Exception as e:
                debug_print(f"Error during alignment of chunk {idx}: {e}")
                
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
        from pelican_nlp.utils.gpu_budget import runtime_torch_device

        self.device = runtime_torch_device(min_free_gb=2.0, allow_mps=True)

        self.model = model
        self.parameters = parameters
        
        if not hf_token:
            active_reporter().warn(
                "No Hugging Face token provided; speaker diarization will be skipped."
            )
            self.diarization_pipeline = None
            return
            
        try:
            # Set Hugging Face token as environment variable for authentication
            import os
            os.environ['HF_TOKEN'] = hf_token
            os.environ['HUGGING_FACE_HUB_TOKEN'] = hf_token
            
            # Try different ways to pass the token based on pyannote.audio version
            hub_kwargs = huggingface_from_pretrained_kwargs()
            try:
                # Method 1: Try use_auth_token (older pyannote.audio versions)
                self.diarization_pipeline = DiarizationPipeline.from_pretrained(
                    model,
                    use_auth_token=hf_token,
                    **hub_kwargs,
                )
            except (TypeError, ValueError) as e1:
                try:
                    # Method 2: Try without explicit token (uses environment variable)
                    self.diarization_pipeline = DiarizationPipeline.from_pretrained(
                        model, **hub_kwargs
                    )
                except Exception as e2:
                    # Method 3: Try with token parameter (newer versions)
                    try:
                        self.diarization_pipeline = DiarizationPipeline.from_pretrained(
                            model,
                            token=hf_token,
                            **hub_kwargs,
                        )
                    except Exception as e3:
                        raise Exception(f"Failed to initialize pipeline. Tried use_auth_token (error: {e1}), "
                                      f"environment variable (error: {e2}), and token (error: {e3})")
            
            debug_print("Initializing SpeakerDiarizer with parameters...")
            self.diarization_pipeline.instantiate(parameters)
            self.diarization_pipeline.to(self.device)
            debug_print("Initialized SpeakerDiarizer successfully.")
        except Exception as e:
            active_reporter().warn(
                f"Failed to initialize SpeakerDiarizer ({e}). Speaker diarization will be skipped."
            )
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
            debug_print("Speaker diarization skipped - no pipeline available.")
            audio_file.speaker_segments = []
            audio_file.register_model("Speaker Diarization", {
                "model": "none",
                "device": "none",
                "parameters": {},
                "speakers": "skipped"
            })
            return
            
        debug_print("Starting speaker diarization...")
        try:
            if num_speakers is not None:
                diarization_result = self.diarization_pipeline(
                    audio_file.normalized_path,
                    num_speakers=num_speakers
                )
                debug_print(f"Diarization completed with {num_speakers} speakers.")
            else:
                diarization_result = self.diarization_pipeline(
                    audio_file.normalized_path
                )
                debug_print("Diarization completed without specifying number of speakers.")

            # Extract speaker segments
            audio_file.speaker_segments = []
            for segment, _, speaker in diarization_result.itertracks(yield_label=True):
                audio_file.speaker_segments.append({
                    "start": segment.start,
                    "end": segment.end,
                    "speaker": speaker
                })
            debug_print(f"Detected {len(audio_file.speaker_segments)} speaker segments.")
            
            # DEBUG: Print first few speaker segments to verify they're populated
            if audio_file.speaker_segments:
                debug_print(f"DEBUG: First 3 speaker segments:")
                for i, seg in enumerate(audio_file.speaker_segments[:3]):
                    debug_print(f"  Segment {i}: {seg}")
                debug_print(f"DEBUG: Speaker segment time range: {audio_file.speaker_segments[0]['start']:.2f}s - {audio_file.speaker_segments[-1]['end']:.2f}s")
            else:
                debug_print("DEBUG: WARNING - speaker_segments is empty after diarization!")
        except Exception as e:
            debug_print(f"An error occurred during diarization: {e}")
            
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
                              timestamp_source: str = "whisper_alignments",
                              transcription_model: str = None,
                              transcriber=None,
                              aligner=None,
                              diarizer=None,
                              release_models: bool = True):

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
    
    reporter = active_reporter()
    debug_print(f"Processing audio file: {audio_file.file}")
    debug_print(f"Audio file exists: {os.path.exists(audio_file.file)}")

    created_transcriber = transcriber is None
    created_aligner = aligner is None
    created_diarizer = diarizer is None
    if created_transcriber:
        debug_print("Initializing processing classes...")
        if transcription_model:
            debug_print(f"Using custom transcription model: {transcription_model}")
            transcriber = AudioTranscriber(model=transcription_model)
        else:
            transcriber = AudioTranscriber()
        debug_print("Processing classes initialized successfully.")

    reporter.set_postfix("load audio")
    audio_file.load_audio()

    reporter.set_postfix("normalize")
    # Use normalized audio directory if set, otherwise use default (same directory as original)
    normalized_audio_dir = getattr(audio_file, '_normalized_audio_dir', None)
    audio_file.rms_normalization(output_dir=normalized_audio_dir)

    reporter.set_postfix("split silence")
    audio_file.split_on_silence(
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        min_length=min_length,
        max_length=max_length
    )

    reporter.set_postfix("transcribe")
    transcriber.transcribe(audio_file)
    for idx, chunk in enumerate(audio_file.chunks, start=1):
        debug_print(f"Chunk {idx} Transcript: {chunk.transcript}\n")

    non_empty_chunks = sum(1 for chunk in audio_file.chunks if chunk.transcript and chunk.transcript.strip())
    if non_empty_chunks == 0:
        raise RuntimeError(
            "No chunks produced any transcript text. Check model availability, audio content, and language compatibility."
        )

    if hasattr(transcriber, "park_on_cpu"):
        reporter.set_postfix("park whisper")
        transcriber.park_on_cpu()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

    reporter.set_postfix("align")
    if created_aligner:
        aligner = ForcedAligner()
    aligner.align(audio_file)
    audio_file.combine_chunks()

    # Step 6: Perform speaker diarization (only if more than one speaker)
    # Ensure num_speakers is set on audio_file for later use
    if not hasattr(audio_file, 'num_speakers') or audio_file.num_speakers is None:
        audio_file.num_speakers = num_speakers
    
    if num_speakers and num_speakers > 1:
        reporter.set_postfix("diarize")
        if created_diarizer:
            diarizer = SpeakerDiarizer(hf_token, parameters=diarizer_params)
        diarizer.diarize(audio_file, num_speakers)
    else:
        reporter.set_postfix("skip diarize")
        audio_file.speaker_segments = []
        audio_file.register_model("Speaker Diarization", {
            "model": "none",
            "device": "none",
            "parameters": {},
            "speakers": "skipped (single speaker)"
        })

    reporter.set_postfix("combine")
    if timestamp_source == "forced_alignments" and not audio_file.forced_alignments and audio_file.whisper_alignments:
        debug_print("Forced alignments are empty; falling back to whisper_alignments for combination.")
        timestamp_source = "whisper_alignments"
    audio_file.combine_alignment_and_diarization(timestamp_source)
    audio_file.aggregate_to_utterances()

    debug_print(f"Finished processing: {audio_file.file}")

    if release_models:
        release_transcription_models(
            transcriber if created_transcriber else None,
            aligner if created_aligner else None,
            diarizer if created_diarizer else None,
        )
    else:
        # Keep Whisper for the next file; free MMS/pyannote so ASR is not sharing 16 GiB VRAM.
        release_transcription_models(None, aligner, diarizer)
        if hasattr(transcriber, "restore_to_device"):
            transcriber.restore_to_device()

    return audio_file


def release_transcription_models(transcriber=None, aligner=None, diarizer=None):
    """Drop transcription model references and clear GPU cache."""
    try:
        if transcriber is not None and hasattr(transcriber, "transcriber"):
            del transcriber.transcriber
    except Exception:
        pass
    try:
        del transcriber
    except Exception:
        pass
    try:
        if aligner is not None and hasattr(aligner, "model"):
            del aligner.model
    except Exception:
        pass
    try:
        del aligner
    except Exception:
        pass
    try:
        if diarizer is not None and hasattr(diarizer, "diarization_pipeline"):
            del diarizer.diarization_pipeline
    except Exception:
        pass
    try:
        del diarizer
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        debug_print("GPU memory cleared after transcription")
    import gc
    gc.collect()
