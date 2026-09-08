# Text input

Store data according to LPDS (Language Processing Data Structure) guidelines, described in the PELICAN paper: https://doi.org/10.48550/arXiv.2511.15512

Copy a text example under `examples/` and use its YAML as the template. Start with `examples/example_fluency` (`config_fluency.yml`) unless you need interview/discourse, image descriptions, or perplexity (`example_discourse`, `example_image-descriptions`, `example_perplexity`).

Put transcripts under `participants/` with LPDS names (`.txt`, `.rtf`, `.docx`, or `.pdf`). From that folder:

```bash
pelican-run
```

Set `input_file: "text"`. Do not use a `transcription:` block. Metrics are written under `derivatives/`.

Install extras: `pelican_nlp[embeddings]` for embeddings, logits, or perplexity; `pelican_nlp[nlp]` for spaCy normalization; `pelican_nlp[topic]` for topic modeling. The default `pip install pelican_nlp` still includes these libraries (except BERTopic, which is the `topic` extra).

If the files already came from audio transcription (`derivatives/transcription/*_transcript.txt`), run the text phase only:

```bash
python -m pelican_nlp.main path/to/config.yml --text-from-transcriptions
```

Keys already commented in the example YAML are not repeated here.

## Options that are not obvious

**`language`**  
Selects the spaCy model for optional lemmatization/normalization (`german` or `english`). It is not passed to Whisper.

**`corpus_key` / `corpus_values`**  
Groups files that share a filename tag (for example `acq-animals`). Omit both to process each unit folder as its own group.

**`fluency_task`**  
Word-list tasks. Cleaning then uses `word_splitter` (often `;` or `,`), and may drop hyphens or duplicates. Do not enable this for continuous prose.

**`discourse` / `participant_speakertag`**  
Interview-style transcripts with speaker labels. Only turns matching `participant_speakertag` (for example `"B"`) are kept for metrics.

**`has_multiple_sections` / `section_identification`**  
Split a file on a heading prefix (for example `"Bild:"`). Leave off for a single block of text.

**`metrics_to_extract`**  
`embeddings`, `logits`, `perplexity`, and/or `topic_modeling`. Logits and perplexity need a causal LM and are slow; embeddings can use `fastText` or an encoder.

See also [transcription_guide.md](transcription_guide.md) for audio, and [overview.md](overview.md) for what the pipeline includes.

## Citation

Pauli Y, Marsman J-B, Rabe F, et al. Standardising the NLP Workflow: A Framework for Reproducible Linguistic Analysis. arXiv preprint arXiv:2511.15512 [cs.CL] 2025. https://doi.org/10.48550/arXiv.2511.15512
