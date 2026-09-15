# Transcription

Store data according to LPDS (Language Processing Data Structure) guidelines, described in the PELICAN paper: https://doi.org/10.48550/arXiv.2511.15512

Copy `examples/example_transcription` and use `config_transcription.yml` in that folder as the template. Put your `.wav` files under `participants/` with LPDS names, then from that folder:

```bash
pelican-run
```

`input_file: "audio"` plus a `transcription:` block (a mapping, not `true`) turns transcription on. Output goes to `derivatives/transcription/` (`*_transcript.txt` and `*_allOutputs.json`). Both files are built from the same aggregated utterances, so the plain text and the `utterance_data` in the JSON always agree; speaker labels (`SPEAKER_00: ...`) appear only when diarization actually distinguished more than one speaker.

Install extra: `pelican_nlp[transcription]` (and `pelican_nlp[acoustic]` if you enable openSMILE or Prosogram). The default `pip install pelican_nlp` still includes these libraries.

Keys already commented in the example YAML are not repeated here.

## `transcription:` options that are not obvious

**`language`** (or the top-level `language`)  
Set this. Whisper otherwise detects the language independently for every chunk, so a long recording can switch mid-file and come back partly transcribed (or translated) into another language. Accepts a name (`"german"`) or a code (`"de"`); the transcription block overrides the top-level value. Leave empty only if the spoken language genuinely varies within a file. If the chosen model rejects the value (for example an English-only checkpoint), Pelican warns once and continues with auto-detection.

**`hf_token`**  
Leave empty to skip speaker diarization. Set a Hugging Face token only if you need pyannote diarization (accept the model terms on the Hub first).

**`num_speakers`**  
Expected speaker count for diarization. Diarization runs only when this is greater than 1 **and** `hf_token` is set. Top-level `number_of_speakers` is used only if `num_speakers` is omitted; an explicit `null` on either key counts as "not set" and resolves to 1.

**`transcription_model`**  
`null` loads `openai/whisper-medium`. Any other value is a Hugging Face ASR model id.

**Chunking (`min_silence_len`, `silence_thresh`, `min_length`, `max_length`)**  
Long recordings are split on silence so Whisper fits in memory. Units are milliseconds (`silence_thresh` is dBFS). If a run is killed for RAM, lower `max_length` and/or `min_silence_len` (the example already uses 500 ms / 120 s). A span longer than `max_length` with no usable silence inside it is cut at the detected silence closest to an evenly spaced target, so cuts avoid landing mid-word where possible.

**`timestamp_source`**  
`whisper_alignments` uses Whisper word times. `forced_alignments` uses the MMS forced aligner; if that is empty, Pelican falls back to Whisper times. Forced alignment also records a per-word `score`: MMS scores collapse for text that is not actually present in the audio, which makes them useful for spotting hallucinated passages.

**`utterance_max_gap`**  
Silence in seconds that ends an utterance (default `1.0`; `null` disables it). Utterances also end at sentence-final punctuation and at every speaker change. Gap and speaker breaks matter because Whisper does not guarantee punctuation — without them a transcript containing no `.?!` becomes a single utterance spanning the whole recording.

**`generate_kwargs`**  
Passed straight to Whisper's `generate`. Use it for decoding options Pelican does not expose, for example `condition_on_prev_tokens: false` to reduce repetition loops. Unsupported keys make Pelican fall back to defaults with a warning rather than fail the run.

**`transcription_patching`**  
Off by default. When `true`, Pelican first transcribes with `transcription_model`, writes that result, **unloads every ASR/MMS/pyannote model and clears GPU memory**, then transcribes the same file again with `openai/whisper-medium`. Clean primary utterances are kept; high-severity spans (decoder loops, subtitle credits, script that contradicts `language`) are replaced by overlapping clean medium utterances. Spans that both models taint are dropped and listed in `*_fusion.json`, not silently rewritten. If the primary model is already whisper-medium, Pelican warns and skips the second pass.

Raw per-model dumps are stored as `*_model-<id>_allOutputs.json` / `*_transcript.txt`. The canonical `*_transcript.txt` and `*_allOutputs.json` are the fused transcript that the text-metrics phase reads.

See also [text_processing_guide.md](text_processing_guide.md) for transcripts that are already text, and [overview.md](overview.md) for what the pipeline includes.

## Citation

Pauli Y, Marsman J-B, Rabe F, et al. Standardising the NLP Workflow: A Framework for Reproducible Linguistic Analysis. arXiv preprint arXiv:2511.15512 [cs.CL] 2025. https://doi.org/10.48550/arXiv.2511.15512
