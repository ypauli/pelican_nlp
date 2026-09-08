# Transcription

Store data according to LPDS (Language Processing Data Structure) guidelines, described in the PELICAN paper: https://doi.org/10.48550/arXiv.2511.15512

Copy `examples/example_transcription` and use `config_transcription.yml` in that folder as the template. Put your `.wav` files under `participants/` with LPDS names, then from that folder:

```bash
pelican-run
```

`input_file: "audio"` plus a `transcription:` block (a mapping, not `true`) turns transcription on. Output goes to `derivatives/transcription/` (`*_transcript.txt` and `*_allOutputs.json`).

Install extra: `pelican_nlp[transcription]` (and `pelican_nlp[acoustic]` if you enable openSMILE or Prosogram). The default `pip install pelican_nlp` still includes these libraries.

Keys already commented in the example YAML are not repeated here.

## `transcription:` options that are not obvious

**`hf_token`**  
Leave empty to skip speaker diarization. Set a Hugging Face token only if you need pyannote diarization (accept the model terms on the Hub first).

**`num_speakers`**  
Expected speaker count for diarization. Diarization runs only when this is greater than 1 **and** `hf_token` is set. Top-level `number_of_speakers` is used only if `num_speakers` is omitted.

**`transcription_model`**  
`null` loads `openai/whisper-medium`. Any other value is a Hugging Face ASR model id.

**Chunking (`min_silence_len`, `silence_thresh`, `min_length`, `max_length`)**  
Long recordings are split on silence so Whisper fits in memory. Units are milliseconds (`silence_thresh` is dBFS). If a run is killed for RAM, lower `max_length` and/or `min_silence_len` (the example already uses 500 ms / 120 s).

**`timestamp_source`**  
`whisper_alignments` uses Whisper word times. `forced_alignments` uses the MMS forced aligner; if that is empty, Pelican falls back to Whisper times.

See also [text_processing_guide.md](text_processing_guide.md) for transcripts that are already text, and [overview.md](overview.md) for what the pipeline includes.

## Citation

Pauli Y, Marsman J-B, Rabe F, et al. Standardising the NLP Workflow: A Framework for Reproducible Linguistic Analysis. arXiv preprint arXiv:2511.15512 [cs.CL] 2025. https://doi.org/10.48550/arXiv.2511.15512
