# PELICAN-nlp

PELICAN-nlp builds reproducible language-processing pipelines for research. Point it at an LPDS project (one YAML config plus `participants/`), run `pelican-run`, and it writes metrics under `derivatives/`.

It can be used on any written language data: embeddings, similarity, logits, perplexity, and the other text metrics are not tied to a task type. The folders under `examples/` are templates (fluency, interviews, image descriptions, generated text, audio transcription), not a closed list of supported domains. Copy one that is close to your layout and adapt its config.

Store data according to LPDS (Language Processing Data Structure) guidelines, described in the PELICAN paper: https://doi.org/10.48550/arXiv.2511.15512

How to run: [transcription_guide.md](transcription_guide.md) (audio) and [text_processing_guide.md](text_processing_guide.md) (text).

## Pipeline

- LPDS layout check and `derivatives/` output paths
- Text import (`.txt`, `.rtf`, `.docx`, `.pdf`)
- Audio load, RMS normalization, silence-based chunking
- Transcription (Whisper), optional forced alignment and speaker diarization
- Text cleaning; optional tokenization and lemmatization/stemming
- Fluency word-list cleaning; discourse speaker-turn filtering; section splits
- Embeddings (fastText or encoder models)
- Semantic similarity (windows and sentences)
- Distance from randomness
- Token logits and perplexity (causal LMs)
- Topic modeling
- openSMILE and Prosogram acoustic features
- Optional result aggregation

## Citation

Pauli Y, Marsman J-B, Rabe F, et al. Standardising the NLP Workflow: A Framework for Reproducible Linguistic Analysis. arXiv preprint arXiv:2511.15512 [cs.CL] 2025. https://doi.org/10.48550/arXiv.2511.15512
