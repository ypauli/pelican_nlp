"""
Audio transcription utilities for PELICAN-nlp.

This module provides utility classes for audio transcription, forced alignment,
and speaker diarization using various machine learning models.
"""

import io
import json
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
from pelican_nlp.config_defaults import DEFAULT_TRANSCRIPTION_MODEL
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


def normalize_language(language) -> str | None:
    """Return a Whisper-usable language name/code, or ``None`` for auto-detection."""
    if not isinstance(language, str):
        return None
    cleaned = language.strip().lower()
    if not cleaned or cleaned in {"auto", "none", "null", "multilingual"}:
        return None
    return cleaned


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


def _park_model(obj):
    """Move a cached model off the GPU so the next stage gets the VRAM."""
    if obj is not None and hasattr(obj, "park_on_cpu"):
        obj.park_on_cpu()


def _restore_model(obj):
    """Move a parked model back onto its compute device."""
    if obj is not None and hasattr(obj, "restore_to_device"):
        obj.restore_to_device()


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


def _cached_hub_file(model_id, filename, hf_kwargs=None):
    """Resolve a Hub (or local) sidecar without requiring it to exist."""
    from transformers.utils import cached_file

    hf_kwargs = hf_kwargs or {}
    allowed = {
        key: hf_kwargs[key]
        for key in ("cache_dir", "revision", "token", "local_files_only", "subfolder")
        if key in hf_kwargs
    }
    return cached_file(
        model_id,
        filename,
        _raise_exceptions_for_missing_entries=False,
        _raise_exceptions_for_connection_errors=False,
        **allowed,
    )


def asr_tokenizer_from_pretrained_kwargs(model_id, hf_kwargs=None):
    """Overrides so transformers 4.x can load tokenizers saved by transformers 5.

    Hugging Face 5 writes ``extra_special_tokens`` as a list. 4.49 then calls
    ``.keys()`` on that value during ``from_pretrained``.
    """
    path = _cached_hub_file(model_id, "tokenizer_config.json", hf_kwargs)
    if not path:
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    extra = config.get("extra_special_tokens") if isinstance(config, dict) else None
    if isinstance(extra, list):
        return {"extra_special_tokens": {}}
    return {}


def load_asr_tokenizer(model_id, hf_kwargs=None):
    from transformers import AutoTokenizer

    hf_kwargs = dict(hf_kwargs or {})
    overrides = asr_tokenizer_from_pretrained_kwargs(model_id, hf_kwargs)
    return AutoTokenizer.from_pretrained(model_id, **hf_kwargs, **overrides)


def load_asr_feature_extractor(model_id, hf_kwargs=None):
    """Load a Whisper feature extractor, including TF5 ``processor_config.json`` repos."""
    from transformers import WhisperFeatureExtractor

    hf_kwargs = dict(hf_kwargs or {})
    try:
        return WhisperFeatureExtractor.from_pretrained(model_id, **hf_kwargs)
    except (OSError, EnvironmentError, ValueError) as exc:
        extractor_error = exc

    path = _cached_hub_file(model_id, "processor_config.json", hf_kwargs)
    if not path:
        raise extractor_error
    try:
        with open(path, encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise extractor_error from exc
    nested = config.get("feature_extractor") if isinstance(config, dict) else None
    if not isinstance(nested, dict):
        raise extractor_error
    kwargs = {key: value for key, value in nested.items() if key != "feature_extractor_type"}
    return WhisperFeatureExtractor(**kwargs)


def asr_pipeline_preprocessor_kwargs(model_id) -> dict:
    """Tokenizer and feature extractor for ``pipeline("automatic-speech-recognition")``.

    Hub cards for Whisper fine-tunes (e.g. ``Flix-AI/flix-swissgerman-full``) load
    ``WhisperProcessor.from_pretrained``. Transformers 4.49 cannot do that for
    checkpoints saved with transformers 5 (list ``extra_special_tokens``, nested
    ``processor_config.json`` and no ``preprocessor_config.json``). Passing the
    shims here keeps Pelican's word-timestamp pipeline on those weights.
    """
    hf_kwargs = huggingface_from_pretrained_kwargs()
    return {
        "tokenizer": load_asr_tokenizer(model_id, hf_kwargs),
        "feature_extractor": load_asr_feature_extractor(model_id, hf_kwargs),
    }


def _sanitize_whisper_generation_config(pipe):
    """Drop TF5 ``forced_decoder_ids`` slots whose token id is JSON ``null``."""
    gen = getattr(pipe, "generation_config", None)
    if gen is None:
        return
    forced = getattr(gen, "forced_decoder_ids", None)
    if not isinstance(forced, list):
        return
    cleaned = []
    for pair in forced:
        if isinstance(pair, (list, tuple)) and len(pair) >= 2 and pair[1] is None:
            continue
        cleaned.append(pair)
    gen.forced_decoder_ids = cleaned or None


def _normalize_asr_pipeline_output(result):
    """Coerce Hugging Face ASR output to ``{"text", "chunks"}``."""
    if isinstance(result, dict):
        return result
    if isinstance(result, list):
        texts = []
        chunks = []
        for item in result:
            if not isinstance(item, dict):
                continue
            text = (item.get("text") or "").strip()
            if text:
                texts.append(text)
            chunks.extend(item.get("chunks") or [])
        return {"text": " ".join(texts), "chunks": chunks}
    raise TypeError(
        f"ASR pipeline returned {type(result).__name__}, expected a dict or list of dicts."
    )


class AudioTranscriber:
    """Handles transcription of audio chunks using Whisper."""
    
    def __init__(self, model=None, language=None, generate_kwargs=None):
        """
        Initialize the AudioTranscriber.
        
        :param model: Whisper model to use for transcription.
        :param language: Spoken language (e.g. ``"german"`` or ``"de"``). ``None``
            leaves Whisper's per-chunk language detection enabled, which lets long
            recordings drift into the wrong language.
        :param generate_kwargs: Extra ``generate`` kwargs forwarded to Whisper
            (for example temperature fallback thresholds).
        """
        from pelican_nlp.utils.gpu_budget import runtime_torch_device

        self.device = runtime_torch_device(min_free_gb=2.0, allow_mps=True)
        self.model = model or DEFAULT_TRANSCRIPTION_MODEL
        self.language = normalize_language(language)
        self.extra_generate_kwargs = dict(generate_kwargs or {})
        self._generate_kwargs_disabled = False
        self._torch_dtype = _asr_torch_dtype(model, self.device)
        if self._torch_dtype == torch.float16:
            active_reporter().status(
                f"{model} fp32 weights do not fit the GPU budget; loading float16."
            )
        self.transcriber = self._build_pipeline()
        dtype_note = " (float16)" if self._torch_dtype == torch.float16 else ""
        lang_note = f", language={self.language}" if self.language else ", language=auto-detect"
        debug_print(
            f"Initialized AudioTranscriber on device: {self.device}{dtype_note}{lang_note}"
        )

    def _build_pipeline(self):
        pipeline_kwargs = {
            "model": self.model,
            "device": self.device,
            "return_timestamps": "word",
            "model_kwargs": huggingface_from_pretrained_kwargs(),
        }
        if self._torch_dtype is not None:
            pipeline_kwargs["torch_dtype"] = self._torch_dtype
        pipeline_kwargs.update(asr_pipeline_preprocessor_kwargs(self.model))
        pipe = pipeline("automatic-speech-recognition", **pipeline_kwargs)
        _sanitize_whisper_generation_config(pipe)
        return pipe

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

    def build_generate_kwargs(self):
        """Whisper ``generate`` kwargs: pin the language so long files cannot drift."""
        if self._generate_kwargs_disabled:
            return {}
        generate_kwargs = dict(self.extra_generate_kwargs)
        if self.language:
            generate_kwargs.setdefault("language", self.language)
            generate_kwargs.setdefault("task", "transcribe")
        return generate_kwargs

    def _disable_generate_kwargs(self, exc):
        """Drop unsupported generate kwargs once (e.g. language on an English-only model)."""
        if self._generate_kwargs_disabled or not self.build_generate_kwargs():
            return False
        self._generate_kwargs_disabled = True
        active_reporter().warn(
            f"{self.model} rejected the configured transcription options ({exc}); "
            "continuing with Whisper defaults (language auto-detection)."
        )
        return True

    def _invoke_asr(self, wav_bytes, asr_kwargs):
        call_kwargs = dict(asr_kwargs)
        generate_kwargs = self.build_generate_kwargs()
        if generate_kwargs:
            call_kwargs["generate_kwargs"] = generate_kwargs
        try:
            return self.transcriber(wav_bytes, **call_kwargs), True
        except TypeError as exc:
            if not _is_word_timestamp_merge_error(exc):
                raise
            fallback = dict(call_kwargs)
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
        except ValueError as exc:
            if not self._disable_generate_kwargs(exc):
                raise
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
                transcription_result = _normalize_asr_pipeline_output(transcription_result)

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
            "language": self.language or "auto-detect",
            "generate_kwargs": self.build_generate_kwargs(),
        })


def _merge_token_spans(token_spans, owners, surface_words, ratio, sample_rate, chunk_start):
    """Turn per-token MMS spans into one entry per surface word.

    A single word can produce several MMS tokens (``"z.B."`` normalizes to ``z b``),
    so consecutive spans owned by the same word are merged into one span. The mean
    MMS score is kept: it drops sharply for text that is not actually in the audio,
    which makes it a usable per-word confidence signal.
    """
    entries = []
    for spans, owner in zip(token_spans, owners):
        if not spans:
            continue
        start_sec = (spans[0].start * ratio / sample_rate) + chunk_start
        end_sec = (spans[-1].end * ratio / sample_rate) + chunk_start
        scores = [float(getattr(span, "score", 0.0)) for span in spans]
        score = sum(scores) / len(scores) if scores else None

        if entries and entries[-1]["_owner"] == owner:
            previous = entries[-1]
            previous["start_time"] = min(previous["start_time"], start_sec)
            previous["end_time"] = max(previous["end_time"], end_sec)
            previous["_scores"].append(score)
            continue

        entries.append({
            "_owner": owner,
            "word": surface_words[owner] if owner < len(surface_words) else "",
            "start_time": start_sec,
            "end_time": end_sec,
            "_scores": [score],
        })

    merged = []
    for entry in entries:
        scores = [s for s in entry.pop("_scores") if s is not None]
        entry.pop("_owner")
        entry["score"] = round(sum(scores) / len(scores), 4) if scores else None
        merged.append(entry)
    return merged


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

    def park_on_cpu(self):
        """Free VRAM between files without dropping the loaded MMS weights."""
        model = getattr(self, "model", None)
        if model is None:
            return
        try:
            model.to("cpu")
        except Exception:
            pass
        _clear_cuda()

    def restore_to_device(self):
        """Move MMS back onto the alignment device."""
        model = getattr(self, "model", None)
        if model is None:
            return
        try:
            model.to(self.device)
        except Exception:
            pass

    def normalize_uroman(self, text: str) -> str:
        """
        Normalize text using Uroman.
        
        :param text: Text to normalize.
        :return: Normalized text.
        """
        text = text.encode('utf-8').decode('utf-8')
        text = text.lower()
        # Fold typographic apostrophes onto the ASCII one kept by the MMS tokenizer.
        text = text.replace("\u2019", "'").replace("\u02bc", "'").replace("\u00b4", "'")
        text = unicodedata.normalize('NFC', text)
        text = re.sub("([^a-z' ])", " ", text)
        text = re.sub(' +', ' ', text)
        return text.strip()

    def _alignment_tokens(self, transcript: str):
        """Map MMS-alignable tokens back to the original surface words.

        ``normalize_uroman`` strips casing and punctuation, so the normalized
        tokens must not be written back into the transcript. Returning the owning
        surface word for each token keeps timestamps aligned to readable text.

        :return: ``(surface_words, normalized_tokens, owners)`` where ``owners[i]``
            is the index in ``surface_words`` that produced ``normalized_tokens[i]``.
        """
        surface_words = transcript.split()
        if not surface_words:
            return [], [], []

        romanized = self.uroman.romanize_string(transcript).split()
        if len(romanized) != len(surface_words):
            # Romanization changed the token count (script expansion, dropped
            # symbols); redo it per word so the mapping stays exact.
            romanized = [self.uroman.romanize_string(word) for word in surface_words]

        normalized_tokens = []
        owners = []
        for index, roman in enumerate(romanized):
            for token in self.normalize_uroman(roman).split():
                normalized_tokens.append(token)
                owners.append(index)
        return surface_words, normalized_tokens, owners

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

                # Normalize for MMS, but remember which surface word each token came from
                surface_words, transcript_list, owners = self._alignment_tokens(chunk.transcript)
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

                # Extract timestamps, merging tokens that belong to the same word
                num_frames = emission.size(1)
                ratio = waveform.size(1) / num_frames
                for entry in _merge_token_spans(
                    token_spans, owners, surface_words, ratio, sample_rate, chunk.start_time
                ):
                    chunk.forced_alignments.append(entry)
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

    def park_on_cpu(self):
        """Free VRAM between files without re-downloading/re-instantiating pyannote."""
        if self.diarization_pipeline is None:
            return
        try:
            self.diarization_pipeline.to(torch.device("cpu"))
        except Exception:
            pass
        _clear_cuda()

    def restore_to_device(self):
        """Move pyannote back onto the diarization device."""
        if self.diarization_pipeline is None:
            return
        try:
            self.diarization_pipeline.to(self.device)
        except Exception:
            pass

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
                              language: str = None,
                              generate_kwargs: Dict = None,
                              utterance_max_gap: float = 1.0,
                              transcriber=None,
                              aligner=None,
                              diarizer=None,
                              shared_models: Dict = None,
                              release_models: bool = True,
                              restore_transcriber: bool = True):

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

    # ``shared_models`` lets the caller reuse Whisper/MMS/pyannote across files.
    pool = shared_models if shared_models is not None else {}
    transcriber = transcriber or pool.get("transcriber")
    aligner = aligner or pool.get("aligner")
    diarizer = diarizer or pool.get("diarizer")

    created_transcriber = transcriber is None
    created_aligner = aligner is None
    created_diarizer = diarizer is None
    if created_transcriber:
        debug_print("Initializing processing classes...")
        if transcription_model:
            debug_print(f"Using custom transcription model: {transcription_model}")
            transcriber = AudioTranscriber(
                model=transcription_model, language=language, generate_kwargs=generate_kwargs
            )
        else:
            transcriber = AudioTranscriber(language=language, generate_kwargs=generate_kwargs)
        debug_print("Processing classes initialized successfully.")
    pool["transcriber"] = transcriber

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
    pool["aligner"] = aligner
    _restore_model(aligner)
    aligner.align(audio_file)
    _park_model(aligner)
    audio_file.combine_chunks()

    # Step 6: Perform speaker diarization (only if more than one speaker)
    # Ensure num_speakers is set on audio_file for later use
    if not hasattr(audio_file, 'num_speakers') or audio_file.num_speakers is None:
        audio_file.num_speakers = num_speakers
    
    if num_speakers and num_speakers > 1:
        reporter.set_postfix("diarize")
        if created_diarizer:
            diarizer = SpeakerDiarizer(hf_token, parameters=diarizer_params)
        pool["diarizer"] = diarizer
        _restore_model(diarizer)
        diarizer.diarize(audio_file, num_speakers)
        _park_model(diarizer)
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
    audio_file.aggregate_to_utterances(max_gap=utterance_max_gap)

    debug_print(f"Finished processing: {audio_file.file}")

    if release_models:
        release_transcription_models(
            transcriber if created_transcriber else None,
            aligner if created_aligner else None,
            diarizer if created_diarizer else None,
        )
        # Only drop what this call built; models handed in by the caller stay theirs.
        for key, was_created in (
            ("transcriber", created_transcriber),
            ("aligner", created_aligner),
            ("diarizer", created_diarizer),
        ):
            if was_created:
                pool.pop(key, None)
    elif restore_transcriber:
        # Keep every model for the next file. MMS/pyannote are already parked on
        # CPU above, so ASR does not have to share VRAM with them.
        if hasattr(transcriber, "restore_to_device"):
            transcriber.restore_to_device()
    else:
        _park_model(transcriber)

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


def unload_transcription_runtime(shared_models: Dict = None):
    """Drop every cached ASR/MMS/pyannote handle and clear GPU memory.

    Used to separate two independent transcription passes: the primary result
    must already be stored, then this runs, then the default model is loaded.
    """
    pool = shared_models if shared_models is not None else {}
    transcriber = pool.pop("transcriber", None)
    aligner = pool.pop("aligner", None)
    diarizer = pool.pop("diarizer", None)
    _park_model(transcriber)
    _park_model(aligner)
    _park_model(diarizer)
    release_transcription_models(transcriber, aligner, diarizer)
    pool.clear()
    _clear_cuda()


def clone_audio_file_for_independent_run(audio_file):
    """New AudioFile pointing at the same wav, sharing no in-memory audio/chunks."""
    from pelican_nlp.core.audio_document import AudioFile

    clone = AudioFile(
        file_path=audio_file.file_path,
        name=audio_file.name,
        target_rms_db=getattr(audio_file, "target_rms_db", -20),
        participant_ID=getattr(audio_file, "participant_ID", None),
        source_folder=getattr(audio_file, "source_folder", None),
        unit_kind=getattr(audio_file, "unit_kind", None),
        task=getattr(audio_file, "task", None),
        num_speakers=getattr(audio_file, "num_speakers", None),
    )
    clone._normalized_audio_dir = getattr(audio_file, "_normalized_audio_dir", None)
    return clone


def transcribe_with_independent_patch(
    audio_file,
    *,
    process_kwargs: Dict,
    primary_model: str,
    patch_model: str,
    store_primary,
):
    """Run primary ASR to completion, persist it, clear GPU, then run the patch model.

    ``store_primary`` is called with the finished primary ``AudioFile`` *before*
    models are unloaded. The patch pass uses a fresh AudioFile and a new model
    pool so it cannot reuse GPU weights or in-memory chunks from the first pass.
    """
    primary_pool: Dict = {}
    primary = process_single_audio_file(
        audio_file,
        transcription_model=primary_model,
        shared_models=primary_pool,
        release_models=False,
        restore_transcriber=False,
        **process_kwargs,
    )
    store_primary(primary)
    unload_transcription_runtime(primary_pool)

    patch_file = clone_audio_file_for_independent_run(primary)
    patch_pool: Dict = {}
    patch = process_single_audio_file(
        patch_file,
        transcription_model=patch_model,
        shared_models=patch_pool,
        release_models=False,
        restore_transcriber=False,
        **process_kwargs,
    )
    unload_transcription_runtime(patch_pool)
    return primary, patch
