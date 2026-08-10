# Audio file processing pipeline — specification

This document describes every processing step applied to audio files in PELICAN-nlp: execution order, configuration choices, parameters, defaults, inputs, outputs, and branching logic.

**Related docs:** [transcription_guide.md](transcription_guide.md) (transcription usage and JSON schema).

---

## Publication reporting — what you can claim from PELICAN

For peer-reviewed work, reviewers expect enough detail that another lab could reproduce preprocessing and see which analytical properties the pipeline introduced. The table below maps **typical publication requirements** to **what PELICAN implements, logs, or leaves to the study team**.

**How to use this section when writing a paper**

1. **Methods — study design (you write):** microphone, room, distance, file format at acquisition (PELICAN does not record these).
2. **Methods — pipeline (cite config + this doc):** copy the [suggested pipeline table](#suggested-methods-table-for-publications) after filling values from your YAML and from `*_allOutputs.json` → `metadata.models_used`.
3. **Supplementary:** attach the exact config YAML, `pyproject.toml` / environment lockfile, and representative `*_allOutputs.json` snippets.
4. **Repository:** LPDS layout + config under version control (already aligned with reproducibility best practice).

### At-a-glance: ten publication areas

| # | Publication area | Reportable from PELICAN? | Where it lives / what to do |
|---|------------------|--------------------------|-----------------------------|
| 1 | **Recording & acquisition** | **External only** | Describe in Methods; optionally add columns to `participants.tsv` / `dataset_description.json`. PELICAN only stores **observed** `sample_rate` in transcription JSON after load. |
| 2 | **Audio preprocessing** | **Partial** | Document RMS normalization, silence chunking, and normalized WAV path (see [Phase A](#phase-a--transcription-pipeline)). **Not automated:** resample-to-16 kHz, denoising, neural VAD, discarding short segments. |
| 3 | **ASR** | **Partial** | Model id and device in `metadata.models_used`; configurable via `transcription.transcription_model`. **You must add manually:** language conditioning, decoding (beam/temperature), WER, human correction, hardware label. Config key `language` is **not** passed to Whisper. |
| 4 | **Segmentation** | **Partial** | Chunking: `min_silence_len`, `silence_thresh`, `min_length`, `max_length`. Utterances: **punctuation-based** (`.?!`), not pause-duration (e.g. 500 ms). State this explicitly in Methods. |
| 5 | **Text normalization** | **Yes (if configured)** | Fully determined by `pipeline_options`, `cleaning_options`, `normalization_options` in YAML — **critical for clinical NLP** (fillers/disfluencies depend on your cleaning flags). |
| 6 | **Linguistic annotation** | **Minimal** | Lemmatization/stemming via spaCy `de_core_news_sm` when enabled. **No** POS, dependency, or NER in pipeline. |
| 7 | **Embeddings** | **Partial** | Model and tokenization from `options_embeddings` (and related blocks). **Not explicit in config:** pooling strategy, layer selection — document what your chosen `Model` implementation does. |
| 8 | **Temporal alignment** | **Partial** | Word-level Whisper timestamps and/or MMS_FA (`timestamp_source`); stored in JSON. **Not MFA**; no phoneme lexicon field; fMRI sync is external. |
| 9 | **Quality control** | **No** | No ASR confidence filter, exclusion log, or inter-rater workflow. Report exclusions and manual review **outside** PELICAN. |
| 10 | **Reproducibility / provenance** | **Partial** | YAML config + git repo = best current practice. Per-file `metadata.models_used` + chunking params in JSON. **Missing auto-run manifest:** Python/PyTorch versions, seeds (except distance-from-randomness), CUDA, full config hash in `derivatives/`. |

### High-impact choices — state these explicitly in Methods

These parameters often change results substantially; PELICAN touches some but does not always export them as a single “methods table”:

| Analytically consequential choice | PELICAN today | Recommended wording discipline |
|-----------------------------------|---------------|------------------------------|
| VAD / silence handling | Silence-based **chunking**, not speech VAD | Do not write “neural VAD” unless you added one upstream. Report `min_silence_len`, `silence_thresh`, `min_length`, `max_length`. |
| ASR decoding | Transformers pipeline defaults | Report exact `transcription_model`; do not claim beam width unless you set it in code. |
| Text cleaning / fillers | Config-dependent | State whether fillers/disfluencies were **retained or removed** (`cleaning_options`, fluency mode). |
| Utterance definition | Punctuation + optional speaker lines | Do not imply pause-based segmentation unless you define pauses post hoc from JSON timestamps. |
| Embeddings | Model-specific pooling | Name `model_name`, tokenization, and semantic-similarity `window_sizes`; describe pooling from model docs if not in YAML. |
| Clinical speech | Not a dedicated mode | If clinically relevant, disable aggressive cleaning and fluency duplicate removal; say so explicitly. |
| fMRI / neuro timing | Word times in JSON | Cite `timestamp_source` (`whisper_alignments` vs `forced_alignments`) and external sync procedure. |

### Suggested methods table for publications

Fill one row per stage you enabled in your config. Defaults shown; **replace with your YAML values**.

| Stage | Tool / implementation | Key parameters (PELICAN defaults) | Provenance in LPDS |
|-------|------------------------|-----------------------------------|----------------------|
| Input layout | LPDS | `participants/part-*/…/<task>/` | Folder structure |
| Load audio | librosa | Native sample rate (`sr=None`) | `metadata.sample_rate` in transcription JSON |
| Loudness normalization | PELICAN `AudioFile` | RMS target **−20 dB** (not in YAML) | `metadata.target_rms_db`; file in `derivatives/normalized-audio/` |
| Segmentation (ASR chunks) | pydub silence detect | `min_silence_len=1000` ms, `silence_thresh=−30` dBFS, `min_length=90000` ms, `max_length=150000` ms | `metadata.models_used` → Chunking |
| ASR | Hugging Face Whisper | Default **`openai/whisper-medium`**; override `transcription_model` | `metadata.models_used` → Transcription |
| Forced alignment | torchaudio MMS_FA | Word times; uroman + lowercase | `metadata.models_used` → Forced Alignment |
| Speaker diarization | pyannote 3.1 | If `num_speakers>1` and `hf_token` set; `diarizer_params` | `metadata.models_used` → Speaker Diarization |
| Utterance build | PELICAN | Sentence endings `.?!`; speaker from overlap with diarization | `utterance_data` in JSON |
| Acoustic features | openSMILE | eGeMAPSv02 **Functionals** (hardcoded); `duration`/`offset` from YAML | `derivatives/opensmile-features/*.csv` |
| Prosody | Prosogram (Praat) | Via `prosogram_extraction` | `derivatives/prosogram-features/` |
| Text clean / normalize | PELICAN + spaCy | From `pipeline_options`, `cleaning_options`, `normalization_options` | Config YAML only |
| Embeddings / metrics | fastText, transformers, etc. | From `metrics_to_extract`, `options_*` | `derivatives/<metric>/` CSVs |

**Example Methods sentence (adapt):**

> Speech files were processed with PELICAN-nlp (configuration archived at [repository URL]) using LPDS layout. Audio was RMS-normalized (−20 dB target), split on silence for transcription with OpenAI Whisper (`openai/whisper-medium` unless otherwise noted), force-aligned with torchaudio MMS_FA, and speaker-turn labels were obtained with pyannote speaker-diarization-3.1 when multiple speakers were configured. Word- and utterance-level timestamps and model identifiers were stored in `derivatives/transcription/*_allOutputs.json`. Recording hardware and room acoustics were [describe study setup]. Text preprocessing and embedding extraction followed the parameters in `config_<study>.yml`.

### LPDS alignment with publication expectations

PELICAN already supports practices strong computational papers use:

| Best practice | PELICAN support |
|---------------|-----------------|
| Config files in repository | YAML next to project (`pelican-run` / `-m pelican_nlp.main`) |
| Parameter tables in supplement | Export from this doc + your YAML + JSON `metadata` |
| Processing graph | [High-level pipeline](#high-level-pipeline-when-input_file-audio) and phase A–D below |
| Model lineage (partial) | `metadata.models_used` per transcription |
| Segmentation hierarchy | Chunks → words → utterances in JSON |
| Temporal alignment provenance | `timestamp_source`, `alignment_source`, both alignment arrays in JSON |

**Gaps vs ideal LPDS provenance** (worth stating as limitations in Discussion or supplement): automatic archiving of software versions and full config hash into `derivatives/`, ASR confidence and QC exclusion logs, acquisition metadata schema, and a single run-level provenance record. See [Checklist coverage analysis](#checklist-coverage-analysis) for the full gap list.

---

**Coverage legend** (used in [Checklist coverage analysis](#checklist-coverage-analysis)):

| Status | Meaning |
|--------|---------|
| **Addressed** | Implemented and configurable (YAML and/or code) |
| **Partial** | Implemented with limited choices, implicit defaults, or incomplete provenance |
| **Not addressed** | Not in PELICAN-nlp today (may be external study design or future work) |
| **N/A** | Outside pipeline scope (acquisition hardware, manual QC protocols, etc.) |

---

## Entry points

| Entry | Location | Audio-related behavior |
|-------|----------|------------------------|
| `pelican-run` | `pelican_nlp/cli.py` | Finds exactly one `.yml`/`.yaml` in the **current working directory**, runs `Pelican(config).run()`. No extra CLI flags. |
| `python -m pelican_nlp.main` | `pelican_nlp/main.py` | Positional `config_path`; `--text-from-transcriptions` runs only the second-phase text pipeline on existing transcripts. |
| Programmatic | `Pelican(config_path, text_from_transcriptions=False)` | Same orchestration as CLI. |

**CLI flags (audio):**

| Flag | Default | Effect |
|------|---------|--------|
| `config_path` | Dev path in `main.py` when omitted | Path to YAML configuration |
| `--text-from-transcriptions` | `false` | Skip audio steps; load `derivatives/transcription/*_transcript.txt` and run text preprocessing + metrics |

**Runtime flags (not in YAML):**

| Flag | Default | Effect |
|------|---------|--------|
| `Pelican.skip_existing` | `true` | If both `*_allOutputs.json` and `*_transcript.txt` exist, skip re-transcription |
| `Pelican.dev_mode` | `true` in `__main__` | May remove `derivatives/` when `skip_existing` is false and user confirms |

---

## Prerequisites: LPDS layout and discovery

### Step 0 — LPDS validation and derivatives root

| Item | Detail |
|------|--------|
| **Module** | `pelican_nlp/preprocessing/LPDS.py` |
| **Functions** | `LPDS_checker()`, `derivative_dir_creator()` |
| **Config** | `multiple_sessions` (`true` \| `false`) |
| **Output** | `<project>/derivatives/` |

### Step 1 — Participant and file discovery

| Item | Detail |
|------|--------|
| **Module** | `pelican_nlp/utils/setup_functions.py` |
| **Functions** | `participant_instantiator()`, `_instantiate_document()` |
| **Filter** | Files under `participants/part-*/` (and optionally `ses-*/`) whose LPDS filename contains `task-<task_name>` matching `config['task_name']` |
| **Document type** | `AudioFile` when `input_file: "audio"` |

**Input audio location:**

```
<project>/participants/part-<ID>/[ses-<ID>/]<task_name>/<LPDS_filename>.<ext>
```

**Supported formats:** Not restricted in code; **librosa**, **pydub**, and **audiofile** accept common formats (e.g. WAV, MP3). README examples often use `*_audio.wav`.

**LPDS filename entities** (parsed by `parse_lpds_filename`): `part`, `task`, optional `ses`, `acq`, `group`, `cat`, `run`, `rec`, `ch`, etc. Corpus grouping uses `corpus_key` + `corpus_values` (e.g. `group-control`).

---

## High-level pipeline (when `input_file: "audio"`)

```
Pelican.run()
  → LPDS check, create derivatives/
  → participant_instantiator()
  → For each corpus_value in corpus_values:
       → Build Corpus from matching AudioFile documents
       → _process_audio_corpus()
            → [optional] transcribe_audio()
            → [optional] extract_opensmile_features()
            → [optional] extract_prosogram()
            → [optional] subprocess: second phase (--text-from-transcriptions)
  → Second phase (if triggered):
       → Load *_transcript.txt from derivatives/transcription/
       → Text preprocessing + metrics_to_extract
```

---

## Phase A — Transcription pipeline

**Enabled when:** `transcription` is a **non-empty dict** (truthy). Use `transcription: false` to disable. Do **not** use `transcription: true` (boolean); code calls `.get()` on the dict.

**Orchestrator:** `Corpus.transcribe_audio(skip_existing)` → `process_single_audio_file()` per file.

**Per-file steps (7 steps):**

### A1 — Load audio

| Parameter | Default | Source |
|-----------|---------|--------|
| Input path | Original file under `participants/` | `AudioFile.file` |
| Sample rate | Native (`sr=None`) | `librosa.load` |

**Module:** `AudioFile.load_audio()` (`pelican_nlp/core/audio_document.py`)

---

### A2 — RMS normalization

| Parameter | Default | Configurable in YAML? |
|-----------|---------|----------------------|
| `target_rms_db` | **-20** dB | No (constructor only) |
| Output directory | `derivatives/normalized-audio/` | Set by corpus |

**Output:** `derivatives/normalized-audio/<stem>_normalized.wav`

**Module:** `AudioFile.rms_normalization()`

---

### A3 — Silence-based chunking

| Parameter | Default (transcription path) | `AudioFile` signature default | Unit |
|-----------|------------------------------|-------------------------------|------|
| `min_silence_len` | **1000** | 1000 | ms |
| `silence_thresh` | **-30** | -30 | dBFS |
| `min_length` | **90000** | 30000 | ms |
| `max_length` | **150000** | 180000 | ms |

**Config keys** (under `transcription:`): `min_silence_len`, `silence_thresh`, `min_length`, `max_length`

**Behavior:** Detect silence with pydub; split at silence midpoints; merge/split intervals to respect min/max chunk length; validate total duration matches original (tolerance 1 ms).

**Module:** `AudioFile.split_on_silence()`

---

### A4 — Speech-to-text (Whisper)

| Parameter | Default | Config key |
|-----------|---------|------------|
| Model | **`openai/whisper-medium`** | `transcription_model` (optional Hugging Face model id) |
| Timestamps | Word-level | Hardcoded `return_timestamps="word"` |
| Device | CUDA → MPS → CPU | Auto |

**Per chunk:** Export chunk WAV → `transformers` ASR pipeline → `chunk.transcript`, `chunk.whisper_alignments`.

**Fallback:** If Whisper returns text but no word timestamps, uniform word timings are inferred across the chunk duration.

**Failure:** If no chunk produces transcript text → `RuntimeError`.

**Module:** `AudioTranscriber.transcribe()` (`pelican_nlp/preprocessing/transcription.py`)

---

### A5 — Forced alignment + combine chunks

| Component | Detail |
|-----------|--------|
| Model | **torchaudio.pipelines.MMS_FA** |
| Romanization | **uroman** + lowercase NFC normalization |
| Device | CUDA if available, else CPU |

**Per chunk:** Align transcript to chunk audio → `chunk.forced_alignments`.

Then: `AudioFile.combine_chunks()` → file-level `transcript_text`, `whisper_alignments`, `forced_alignments`.

**Module:** `ForcedAligner.align()`, `AudioFile.combine_chunks()`

---

### A6 — Speaker diarization (conditional)

| Condition | Behavior |
|-----------|----------|
| `num_speakers > 1` | Run pyannote on **full normalized WAV** |
| `num_speakers <= 1` | Skip; `speaker_segments = []` |

| Parameter | Default | Config key |
|-----------|---------|------------|
| Model | **`pyannote/speaker-diarization-3.1`** | Not exposed in YAML |
| HF token | `''` | `hf_token` (required for diarization) |
| Expected speakers | `number_of_speakers` or **2** | `num_speakers` |

**If `hf_token` empty or pipeline init fails:** Diarization skipped; words may be labeled `UNKNOWN` (multi-speaker) or `SPEAKER_0` (single-speaker).

**Default `diarizer_params`** (pyannote `instantiate()`):

```yaml
segmentation:
  min_duration_off: 0.0
clustering:
  method: centroid
  min_cluster_size: 12
  threshold: 0.8
```

**Config key:** `diarizer_params` (optional override)

**Module:** `SpeakerDiarizer.diarize()`

---

### A7 — Merge alignments with speakers + utterances

| Parameter | Default | Choices |
|-----------|---------|---------|
| `timestamp_source` | **`whisper_alignments`** | `whisper_alignments` \| `forced_alignments` |

**Fallback:** If `forced_alignments` is chosen but empty, automatically falls back to `whisper_alignments`.

**Functions:**

1. `combine_alignment_and_diarization(timestamp_source)` — assign speaker label per word (overlap-based)
2. `aggregate_to_utterances()` — group words into utterances at sentence endings (`.`, `?`, `!`)

**Module:** `AudioFile` (`pelican_nlp/core/audio_document.py`)

---

### A8 — Save transcription artifacts

| Output | Path |
|--------|------|
| Full JSON | `derivatives/transcription/<audio_stem>_allOutputs.json` |
| Plain text | `derivatives/transcription/<audio_stem>_transcript.txt` |

**Text format:**

| Condition | Format |
|-----------|--------|
| `num_speakers > 1` and utterances exist | Lines: `SPEAKER_XX: utterance text` |
| Otherwise | Plain `transcript_text` (no speaker prefixes) |

**Skip:** When `skip_existing=True` and both JSON and TXT already exist.

**JSON fields** (see `AudioFile.save_as_json()`): `metadata`, `transcript_text`, `whisper_alignments`, `forced_alignments`, `combined_data`, `utterance_data`, `speaker_segments`.

---

## Phase B — openSMILE acoustic features

**Enabled when:** `opensmile_feature_extraction: true` (**required key** for audio configs — missing key causes `KeyError`).

| Item | Detail |
|------|--------|
| **Module** | `Corpus.extract_opensmile_features()` → `AudioFeatureExtraction.opensmile_extraction()` |
| **Input** | **Original** audio path (`document.file`), not normalized WAV |
| **Feature set** | **eGeMAPSv02 Functionals** (hardcoded in code) |
| **Output** | `derivatives/opensmile-features/part-*/[ses-*/]task-*/<name>_opensmile-features.csv` |

### `opensmile_configurations`

| Key | Default | Used in code |
|-----|---------|--------------|
| `duration` | `null` | Yes — passed to `audiofile.read` (full file if null) |
| `offset` | `null` | Yes |
| `always_2d` | `true` | Yes — passed to `audiofile.read` |
| `feature_set` | Documented in YAML | **No** — code uses `eGeMAPSv02` |
| `feature_level` | Documented in YAML | **No** — code uses `Functionals` |

---

## Phase C — Prosogram prosodic features

**Enabled when:** `prosogram_extraction: true` (**required key** for audio configs).

| Item | Detail |
|------|--------|
| **Module** | `Corpus.extract_prosogram()` → `AudioFeatureExtraction.extract_prosogram_profile()` |
| **Input** | **Original** audio path |
| **Engine** | Praat via **parselmouth**, script `pelican_nlp/praat/prosomain.praat` |
| **Output dir** | `derivatives/prosogram-features/part-<ID>/` |

**Possible output files** (prefix = audio basename):

| Suffix | Content |
|--------|---------|
| `_profile_data.txt` | Main prosodic profile (TSV) |
| `_profile.txt` | Profile report |
| `_data.txt` | Syllable data |
| `_table.txt` | Long-format syllabic features |
| `_styl.txt` | Stylization targets |
| `_eval.txt` | Evaluation file |

Note: Prosogram results are **not** written via `store_features_to_csv` (unlike openSMILE).

---

## Phase D — Text metrics from transcripts (second phase)

**Triggered when:** Any of `embeddings`, `logits`, `perplexity`, `topic_modeling` appears in `metrics_to_extract` **and** `transcription` is enabled.

**Mechanism:** Subprocess: `python -m pelican_nlp.main <config> --text-from-transcriptions`

### D1 — Load transcript documents

- Scan `derivatives/transcription/*_transcript.txt`
- Filter by `corpus_key-corpus_value` via LPDS entities on the audio stem
- Build `Document` objects pointing at the transcription directory

### D2 — Text preprocessing

Controlled by `pipeline_options` (each `true`/`false`):

| Step | Config key | Handler |
|------|------------|---------|
| Quality check | `quality_check` | No-op stub |
| Clean | `clean_text` | `TextCleaner` + `cleaning_options` |
| Tokenize | `tokenize_text` | `TextTokenizer` + `tokenization_options` |
| Normalize | `normalize_text` | `TextNormalizer` + `normalization_options` |

Also: `detect_sections()` per document if sections are configured.

**Discourse (multi-speaker text):** If `discourse: true` and `participant_speakertag` is set, `TextDiarizer.parse_speaker()` filters participant speech during metric extraction (separate from pyannote audio diarization).

### D3 — Metric extraction

| Metric | Config block |
|--------|--------------|
| `embeddings` | `options_embeddings` |
| `logits` | `options_logits` |
| `perplexity` | `options_perplexity` |
| `topic_modeling` | `options_topic-modeling` |

Outputs go under `derivatives/<metric>/...` via `store_features_to_csv`.

Optional post-steps: `create_aggregation_of_results`, `output_document_information` + `document_information_output.parameters`.

---

## Master configuration reference (audio)

### Required / branching top-level keys

| Key | Values | Role |
|-----|--------|------|
| `input_file` | `"audio"` | Selects audio pipeline |
| `task_name` | string | Filters participant files |
| `corpus_key` | string | Corpus grouping entity |
| `corpus_values` | list of strings | One corpus run per value |
| `multiple_sessions` | bool | LPDS layout validation |
| `number_of_speakers` | int | Fallback for diarization; passed to `AudioFile` |
| `transcription` | `false` or `{...}` | Transcription block |
| `opensmile_feature_extraction` | bool | openSMILE on/off |
| `prosogram_extraction` | bool | Prosogram on/off |
| `metrics_to_extract` | list | May trigger phase D |
| `opensmile_configurations` | dict | Required if openSMILE enabled |
| `discourse` | bool | Text-phase speaker filtering |
| `participant_speakertag` | string \| null | e.g. `"B"` for discourse |
| `pipeline_options` | dict | Phase D preprocessing |
| `cleaning_options` | dict | Text cleaning |
| `options_*` | dict | Per-metric settings |

### Full `transcription:` block

| Key | Default | Description |
|-----|---------|-------------|
| `hf_token` | `''` | Hugging Face token for pyannote |
| `num_speakers` | `number_of_speakers` or **2** | Expected speaker count |
| `min_silence_len` | **1000** | Min silence for split (ms) |
| `silence_thresh` | **-30** | Silence threshold (dBFS) |
| `min_length` | **90000** | Min chunk length (ms) |
| `max_length` | **150000** | Max chunk length (ms) |
| `timestamp_source` | `whisper_alignments` | Word timing source |
| `transcription_model` | `null` → whisper-medium | Hugging Face ASR model id |
| `diarizer_params` | See A6 | pyannote pipeline parameters |

---

## Branching summary

| Decision | Condition | Effect |
|----------|-----------|--------|
| Audio vs text | `input_file` | Entire pipeline branch |
| Transcription | `transcription` dict truthy | Steps A1–A8 |
| Skip re-transcribe | `skip_existing` + existing JSON+TXT | Skip file |
| Diarization | `num_speakers > 1` + valid `hf_token` | pyannote vs skipped |
| Timestamp source | `timestamp_source` | Whisper vs MMS_FA alignments |
| Transcript TXT layout | multi-speaker + utterances | Speaker-prefixed lines |
| openSMILE | `opensmile_feature_extraction` | Phase B |
| Prosogram | `prosogram_extraction` | Phase C |
| Text metrics | `metrics_to_extract` ∩ text metrics + transcription | Phase D subprocess |
| Phase D only | `--text-from-transcriptions` | No audio processing |

---

## External dependencies (audio)

| Step | Library / model |
|------|-----------------|
| Load / normalize | librosa, soundfile, numpy |
| Chunking | pydub |
| Transcription | transformers + **openai/whisper-medium** (default) |
| Forced alignment | torchaudio **MMS_FA**, uroman |
| Diarization | pyannote **speaker-diarization-3.1** |
| openSMILE | opensmile (eGeMAPSv02 Functionals), audiofile |
| Prosogram | parselmouth / Praat |

---

## Sample configuration files

| File | Focus |
|------|-------|
| `pelican_nlp/sample_configuration_files/config_transcription.yml` | Transcription + embeddings (phase D) |
| `examples/example_acoustic-features/config_acousticfeatures.yml` | openSMILE + Prosogram, `transcription: false` |
| `examples/example_general/config_general.yml` | Annotated option reference for all keys |

---

## Source file map

| Path | Role |
|------|------|
| `pelican_nlp/main.py` | Orchestration, two-phase subprocess |
| `pelican_nlp/cli.py` | CWD config discovery |
| `pelican_nlp/core/audio_document.py` | `AudioFile`, `Chunk` |
| `pelican_nlp/preprocessing/transcription.py` | Whisper, MMS_FA, pyannote |
| `pelican_nlp/core/corpus.py` | `transcribe_audio`, openSMILE, Prosogram |
| `pelican_nlp/extraction/acoustic_feature_extraction.py` | openSMILE + Prosogram extraction |
| `pelican_nlp/utils/setup_functions.py` | Participant / `AudioFile` discovery |
| `pelican_nlp/preprocessing/LPDS.py` | Folder validation |

---

## Checklist coverage analysis

This section maps a methodological checklist (acquisition → storage → reproducibility) to what **PELICAN-nlp actually implements**. Status refers to the **codebase**, not only what the pipeline spec document describes.

### Summary

| Layer | Addressed in PELICAN | Main gaps |
|-------|----------------------|-----------|
| Acquisition | Partial | No enforced format/SR/bit depth; no recording-environment metadata |
| Signal preprocessing | Partial | RMS + silence chunking only; no VAD/denoise/resample policy |
| Segmentation / transcription | Partial–strong | Whisper + MMS_FA + pyannote; few ASR decoding/language controls |
| Linguistic normalization | Partial | Cleaning + lemmatization; no explicit clinical disfluency policy |
| Linguistic annotation | Not addressed | No POS, parsing, NER, coreference |
| Acoustic features | Partial | eGeMAPSv02 + Prosogram; limited configurability |
| Representation / embeddings | Partial–strong | fastText, transformers, semantic similarity; limited pooling/layer control |
| Downstream analytics | Partial | Embeddings, logits, perplexity, topic modeling, aggregations |
| Storage / LPDS | Partial | JSON/CSV + folder layout; incomplete provenance |
| Quality control | Not addressed | `quality_check` is a no-op |
| Reproducibility | Partial | Per-step model metadata in transcription JSON only |
| Clinical / privacy | Partial | Fluency + discourse paths; no pause taxonomy or de-identification |

---

### 1. Audio acquisition parameters

| Topic | Status | PELICAN behavior |
|-------|--------|------------------|
| File type (WAV, FLAC, MP3, AAC, …) | **Partial** | Any format readable by librosa/pydub/audiofile; **no validation** or preference for lossless |
| Bit depth | **Not addressed** | Not read, stored, or normalized |
| Sample rate (8–48 kHz, …) | **Partial** | Loaded at **native** rate (`librosa.load(..., sr=None)`); not standardized to 16 kHz for ASR |
| Mono vs stereo / multi-channel | **Not addressed** | No channel selection, downmixing, or beamforming |
| Recording environment metadata | **N/A** / **Not addressed** | Not captured as pipeline covariates (mic type, distance, reverb, noise, clipping) |

**Important missing aspects:** Acquisition QA (target SR, mono policy, lossless requirement), channel handling, and linking environmental metadata to derivatives for later analysis.

---

### 2. Signal preprocessing choices

| Topic | Status | PELICAN behavior |
|-------|--------|------------------|
| Resampling (target SR, filter, algorithm) | **Partial** | Only **inside MMS forced alignment** (chunk → model sample rate via `torchaudio Resample`); not applied at load or before Whisper |
| Peak / RMS / LUFS normalization | **Partial** | **RMS** to `target_rms_db=-20` (not in YAML); no peak or LUFS; **per-file** only |
| Corpus-wide normalization | **Not addressed** | — |
| Noise reduction | **Not addressed** | — |
| VAD (rule-based / neural / thresholds / padding) | **Partial** | **Silence-based chunking** (`min_silence_len`, `silence_thresh`) — not speech VAD; does not trim non-speech before ASR |
| Segmentation strategy | **Partial** | Pause-based **chunks** for ASR (`min_length`, `max_length`); utterances from punctuation + diarization merge; no fixed-window ASR |
| Overlap between segments | **Not addressed** for audio | Overlap exists for **logits** chunks only (`overlap_size` in phase D) |

**Important missing aspects:** Explicit ASR-oriented preprocessing (16 kHz mono), optional neural VAD, denoising, and documented tradeoffs for pause/speech-rate metrics.

---

### 3. Speaker processing

| Topic | Status | PELICAN behavior |
|-------|--------|------------------|
| No diarization | **Addressed** | `num_speakers <= 1` skips pyannote |
| Automatic diarization | **Addressed** | pyannote `speaker-diarization-3.1`; `num_speakers`, `diarizer_params` |
| Oracle speaker labels | **Partial** | **Text phase:** `discourse` + `participant_speakertag` filters participant lines; not oracle **audio** labels |
| Clustering threshold / min duration | **Partial** | `diarizer_params.clustering.threshold`, `min_cluster_size`; segmentation `min_duration_off` |
| Overlap handling | **Not addressed** | No overlap speech regions in diarization output handling |
| Embedding model choice | **Not addressed** | pyannote model fixed in code |
| Speaker identification / verification | **Not addressed** | Anonymous `SPEAKER_XX` only |

**Important missing aspects:** Oracle diarization input, overlap-aware diarization, and speaker-ID workflows.

---

### 4. Automatic speech recognition (ASR)

| Topic | Status | PELICAN behavior |
|-------|--------|------------------|
| ASR model family | **Partial** | **Whisper** via Hugging Face (`transcription_model`; default `openai/whisper-medium`) |
| Other ASR (wav2vec2, Conformer, Kaldi, …) | **Not addressed** | — |
| Model size / quantization | **Partial** | Any HF Whisper id; no explicit quant / precision flags |
| Decoding (beam, temperature, length penalty, suppression) | **Not addressed** | Default `transformers` pipeline behavior only |
| Language configuration | **Not addressed** | Config key `language` exists in YAML but is **not passed** to the Whisper pipeline |
| Automatic language detection | **Partial** | Implicit Whisper behavior; not configured |
| Timestamps | **Addressed** | Word-level (`return_timestamps="word"`); utterance-level via aggregation |
| Alignment confidence threshold | **Not addressed** | — |
| Forced alignment tools | **Partial** | **MMS_FA** (torchaudio), not MFA / WhisperX / Gentle |
| Phoneme lexicon / alignment filtering | **Not addressed** | Word-level romanized alignment only |

**Important missing aspects:** Language forcing per config, decoding parameter control, ASR confidence scores, and alternative ASR backends for domain-specific clinical speech.

---

### 5. Linguistic normalization (phase D)

| Topic | Status | PELICAN behavior |
|-------|--------|------------------|
| Preserve fillers / disfluencies / repetitions | **Partial** | **No dedicated flags**; default cleaning can remove punctuation/brackets; fluency mode **removes duplicates/hyphens** — opposite of clinical preservation unless configured off |
| Lowercase / punctuation | **Addressed** | `cleaning_options` |
| Spelling correction / abbreviation expansion | **Not addressed** | — |
| Tokenization method | **Addressed** | `tokenization_method`: whitespace, model, model_roberta |
| Lemmatization / stemming | **Addressed** | `normalization_options.method`: lemmatization \| stemming (spaCy `de_core_news_sm`) |
| Sentence segmentation | **Partial** | Utterances from **punctuation** in `aggregate_to_utterances`; sections via `has_multiple_sections` / `section_identification`; no pause-based sentence boundaries |

**Important missing aspects:** Explicit **clinical disfluency preservation** policy, language-aware normalizer (German hardcoded in normalizer), and prosodic/pause-based sentence boundaries.

---

### 6. Linguistic annotation

| Topic | Status | PELICAN behavior |
|-------|--------|------------------|
| POS tagging | **Not addressed** | — |
| Dependency parsing | **Not addressed** | — |
| NER (general / clinical) | **Not addressed** | — |
| Coreference | **Not addressed** | — |

**Important missing aspects:** Entire annotation layer if downstream analyses need syntax, entities, or discourse structure beyond speaker tags.

---

### 7. Acoustic feature extraction

| Topic | Status | PELICAN behavior |
|-------|--------|------------------|
| Low-level descriptors (F0, jitter, shimmer, MFCCs, …) | **Partial** | **eGeMAPSv02 Functionals** via openSMILE (bundle of descriptors); not per-descriptor YAML control |
| Frame length / hop / window | **Not addressed** | openSMILE functionals are file-level; Prosogram uses Praat defaults |
| Prosodic features (rate, pauses, intonation) | **Partial** | **Prosogram** (`_profile_data.txt`, syllable-level outputs) |
| Emotion / paralinguistic models | **Not addressed** | — |
| Pause metrics from audio | **Not addressed** | Pauses affect chunking only, not exported pause features |

**Important missing aspects:** Configurable low-level extraction, explicit pause/speech-rate from waveform, emotion/valence channels.

---

### 8. Embedding extraction (phase D)

| Topic | Status | PELICAN behavior |
|-------|--------|------------------|
| Static embeddings (fastText, word2vec, GloVe) | **Partial** | **fastText** supported via `options_embeddings` |
| Contextual (BERT, RoBERTa, Llama, …) | **Addressed** | Multiple models via `language_model.Model` |
| Speech embeddings (wav2vec, …) | **Not addressed** | Text-side only after ASR |
| Pooling strategy | **Not addressed** | Model-default; no CLS/mean/max/layer config |
| Layer selection | **Not addressed** | — |
| Windowing | **Addressed** | Semantic similarity `window_sizes`; logits `chunk_size` / `overlap_size`; topic modeling chunk options |

**Important missing aspects:** Documented pooling/layer choices and optional **acoustic** embeddings without going through text.

---

### 9. Temporal modeling

| Topic | Status | PELICAN behavior |
|-------|--------|------------------|
| Independent utterances | **Addressed** | Default unit for similarity / many metrics |
| Sequential / hierarchical modeling | **Not addressed** | No RNN/transformer over time series of embeddings |
| Context window / recurrence | **Partial** | Sliding windows only (`semantic-similarity`, logits overlap) |

**Important missing aspects:** Longitudinal or session-level temporal models if research questions require them.

---

### 10. Quality control

| Topic | Status | PELICAN behavior |
|-------|--------|------------------|
| `quality_check` pipeline step | **Not addressed** | Stub in `TextPreprocessingPipeline._quality_check` |
| ASR / alignment confidence filtering | **Not addressed** | — |
| Artifact detection (clipping, dropout, music) | **Not addressed** | — |
| Human validation workflow | **N/A** | External to software |

**Important missing aspects:** Automated QC gates before metrics (empty transcript check exists; no confidence or audio artifact checks).

---

### 11. Storage / data structure (LPDS)

| Topic | Status | PELICAN behavior |
|-------|--------|------------------|
| LPDS folder layout | **Addressed** | `participants/`, `derivatives/`, LPDS filename parsing |
| Granularity stored | **Partial** | **File**, **word** (JSON alignments), **utterance** (JSON + multi-speaker TXT); openSMILE **file-level** CSV; not standard **frame-level** store |
| Serialization | **Partial** | **JSON** (transcription), **CSV** (metrics, openSMILE), **TSV/txt** (Prosogram) |
| Parquet / HDF5 / Arrow / SQLite | **Not addressed** | — |
| Timestamps in outputs | **Addressed** | Word/utterance times in JSON |
| Provenance / model versioning | **Partial** | `metadata.models_used` + chunking params in transcription JSON; **full config not serialized** to derivatives |
| Preprocessing history / parameter logs | **Not addressed** | No run manifest or config hash per derivative |

**Important missing aspects:** Machine-readable **run provenance** (config snapshot, package versions, git hash) attached to every output artifact.

---

### 12. Computational parameters

| Topic | Status | PELICAN behavior |
|-------|--------|------------------|
| CPU vs GPU | **Partial** | Auto CUDA → MPS → CPU for Whisper/diarization; MMS_FA CUDA or CPU |
| Mixed precision | **Not addressed** | — |
| Distributed inference | **Not addressed** | — |
| Batch size (ASR) | **Not addressed** | Sequential per chunk / per file |
| Chunk size / memory | **Partial** | Audio chunking for long files; `skip_existing`; GPU cache clear after files |
| Streaming vs offline | **Not addressed** | Offline only |

---

### 13. Clinical / research-specific decisions

| Topic | Status | PELICAN behavior |
|-------|--------|------------------|
| Disfluency preservation | **Partial** | Risk of removal via cleaning; fluency pipeline **strips** duplicates/hyphens by design |
| Pause definitions (min duration, filled vs unfilled) | **Not addressed** | — |
| Narrative / event segmentation | **Partial** | Document sections; corpus grouping; no event ontology |
| Privacy / anonymization | **Not addressed** | No NER masking, voice conversion, or metadata stripping in pipeline |

**Important missing aspects:** First-class **clinical speech** mode (preserve fillers, pauses, repairs) and privacy tooling for shared corpora.

---

### 14. Reproducibility parameters

| Topic | Status | PELICAN behavior |
|-------|--------|------------------|
| Model versions in outputs | **Partial** | Transcription JSON: `metadata.models_used` (model id, device, diarizer params) |
| Random seeds | **Partial** | `random_seed` in `options_dis_from_randomness` only |
| Decoding parameters logged | **Not addressed** | — |
| Preprocessing order logged | **Not addressed** | Documented here; not emitted per run |
| Hardware / software environment | **Not addressed** | — |
| Quantization settings | **Not addressed** | — |
| Config file archived with run | **Not addressed** | User must keep YAML manually |

**Important missing aspects:** Single **reproducibility record** per run (config + environment + seeds + model revisions) stored under `derivatives/`.

---

### Organizational principle (checklist vs PELICAN layers)

| Layer | Example parameter | PELICAN status | Where configured / stored |
|-------|-------------------|----------------|---------------------------|
| Acquisition | Sample rate | **Partial** | Native load; stored in transcription `metadata.sample_rate` |
| Signal preprocessing | VAD / silence threshold | **Partial** | `transcription.min_silence_len`, `silence_thresh` |
| Segmentation | Utterance window | **Partial** | `min_length`, `max_length`; utterance aggregation in code |
| Transcription | ASR model | **Addressed** | `transcription.transcription_model` |
| Linguistic normalization | Lemmatization | **Addressed** | `normalization_options.method` (phase D) |
| Annotation | POS tagger | **Not addressed** | — |
| Representation | Embedding model | **Addressed** | `options_embeddings.model_name` |
| Analytics | Perplexity, semantic similarity | **Addressed** | `metrics_to_extract`, `options_*` |
| Storage | LPDS schema | **Partial** | Folder layout + JSON/CSV paths |
| Provenance | Parameter logs | **Partial** | `models_used` in transcription JSON only |

---

### Highest-priority gaps (recommended documentation or roadmap)

These are the most consequential **missing or weak** areas relative to a full audio–NLP methodological checklist:

1. **Acquisition contract** — No enforced or documented requirements for format, sample rate, channels, or recording metadata.
2. **ASR-oriented preprocessing** — No standard resample/downmix; `language` in config unused for Whisper.
3. **ASR control surface** — No decoding parameters, confidence scores, or alternative ASR engines.
4. **VAD vs silence chunking** — Chunking is energy/silence-based for length limits, not speech detection or trim-before-ASR.
5. **Clinical speech policy** — No explicit mode to preserve disfluencies, fillers, and repairs; fluency cleaning can remove clinically relevant variation.
6. **Pause and timing analytics** — Rich word timestamps exist, but no exported pause metrics or pause-definition parameters.
7. **Linguistic annotation stack** — No POS, dependency, NER, or coreference.
8. **Quality control** — No operational QC (confidence, artifacts, clipping); pipeline `quality_check` unused.
9. **Provenance / reproducibility** — Partial model metadata only; no run manifest, environment capture, or config hash in derivatives.
10. **Privacy** — No de-identification or anonymization steps in the pipeline.
11. **Acoustic feature flexibility** — openSMILE feature set/level fixed in code; Prosogram not integrated into unified CSV/feature store.
12. **Speaker overlap and identification** — Diarization is fixed-model, no overlap or enrollment-based ID.

Items **well covered** relative to the checklist: optional **Whisper + MMS_FA + pyannote** path, **word/utterance** storage, **discourse speaker filtering** on text, **openSMILE + Prosogram**, and **downstream text metrics** (embeddings, logits, perplexity, topic modeling, semantic similarity windows).
