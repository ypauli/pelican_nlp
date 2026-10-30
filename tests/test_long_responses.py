"""Guards for the clinical interview run: no truncation, word pooling, stable ids."""

from types import SimpleNamespace

import pandas as pd

from pelican_nlp.extraction.extract_embeddings import EmbeddingsExtractor
from pelican_nlp.extraction.extract_perplexity import PerplexityExtractor
from pelican_nlp.extraction.long_responses import (
    clear_embedding_stats,
    embedding_response_summary,
    stitch_window_slices,
)
from pelican_nlp.extraction.model_registry import MODEL_KIND_ENCODER
from pelican_nlp.extraction.semantic_similarity import (
    align_embeddings_to_source_sentences,
    get_semantic_similarity_windows,
    tokens_to_text,
)
from pelican_nlp.extraction.word_pooling import pool_subword_embeddings


def test_stitch_keeps_each_token_once():
    specials = {1, 2}
    windows = [
        [1, 10, 11, 12, 13, 14, 15, 16, 17, 2],
        [1, 16, 17, 18, 19, 20, 21, 22, 23, 2],
        [1, 22, 23, 24, 2],
    ]
    slices = stitch_window_slices(windows, 2, specials)
    stitched = []
    for index, start, end in slices:
        stitched.extend(windows[index][start:end])
    assert stitched == [1, *range(10, 25), 2]


def test_subword_pieces_pool_to_words_and_keep_sentence_punctuation():
    pairs = [
        ("<bos>", [0.0, 0.0]),
        ("▁Ich", [1.0, 0.0]),
        ("▁sehe", [0.0, 2.0]),
        (".", [0.0, 2.0]),
        ("▁Er", [4.0, 0.0]),
        ("<eos>", [0.0, 0.0]),
    ]
    pooled = pool_subword_embeddings(pairs)
    assert [token for token, _ in pooled] == ["Ich", "sehe.", "Er"]
    # "sehe" and the following period share one vector: mean of [0, 2] and [0, 2].
    assert list(pooled[1][1]) == [0.0, 2.0]


def test_plain_tokens_stay_separate_words():
    pooled = pool_subword_embeddings([("Ich", [1.0]), ("sehe", [2.0])])
    assert [token for token, _ in pooled] == ["Ich", "sehe"]


def test_tokens_to_text_spaces_words_and_glues_subwords():
    assert tokens_to_text(["Ich", "sehe"]) == "Ich sehe"
    assert tokens_to_text(["▁Ich", "▁sehe", "."]) == "Ich sehe."
    assert tokens_to_text(["▁Bub", "##en"]) == "Buben"


def test_sentence_alignment_follows_source_length_not_equal_splits():
    vectors = [
        ("Hi", [1.0, 0.0]),
        ("This", [0.0, 1.0]),
        ("is", [0.0, 1.0]),
        ("longer", [0.0, 1.0]),
    ]
    groups = align_embeddings_to_source_sentences(vectors, ["Hi.", "This is longer."])
    assert [token for token, _ in groups[0]] == ["Hi"]
    assert [token for token, _ in groups[1]] == ["This", "is", "longer"]

    summary, details = get_semantic_similarity_windows(
        vectors,
        "sentence",
        return_details=True,
        source_text="Hi. This is longer.",
    )
    assert details
    assert details[0]["sentence_i_text"] == "Hi"
    assert "This" in details[0]["sentence_j_text"]
    assert summary[0] == summary[0]  # finite similarity, not an equal-split mix


def test_configured_max_length_does_not_truncate():
    clear_embedding_stats()
    extractor = EmbeddingsExtractor.__new__(EmbeddingsExtractor)
    extractor.embeddings_configurations = {"max_length": 512, "long_response_overlap": 64}
    extractor.model_instance = SimpleNamespace(
        config=SimpleNamespace(max_position_embeddings=8192)
    )
    extractor._warned_budget = False
    extractor._warned_chunk = False

    class _Tokenizer:
        def __call__(self, text, add_special_tokens=True, truncation=False, padding=False, **kwargs):
            assert truncation is False
            assert "max_length" not in kwargs
            return {"input_ids": list(range(600))}

    extractor.Tokenizer = SimpleNamespace(tokenizer=_Tokenizer())
    assert extractor._embeddings_over_context("ignored", {}, observe=True) is None
    assert extractor._last_observed_length == 600
    summary = embedding_response_summary()
    assert summary["longer_than_configured_max"] == 1
    assert summary["chunked"] == 0
    assert summary["max_tokens"] == 600
    clear_embedding_stats()


def test_perplexity_keeps_original_sentence_index(tmp_path):
    extractor = PerplexityExtractor({}, tmp_path)
    rows = [
        {"token": "Hello", "logprob_actual": -0.1},
        {"token": "world", "logprob_actual": -0.1},
        {"token": ".", "logprob_actual": -0.1},
        {"token": "Ja", "logprob_actual": -0.2},
        {"token": ".", "logprob_actual": -0.2},
        {"token": "More", "logprob_actual": -0.3},
        {"token": "words", "logprob_actual": -0.3},
        {"token": "here", "logprob_actual": -0.3},
        {"token": ".", "logprob_actual": -0.3},
    ]
    document = SimpleNamespace(
        name="part-01_task-interview_text.txt",
        source_folder="part-01",
        lpds_entities=None,
    )
    extractor.extract_perplexity_from_document(document, rows, section_index=4)
    frame = pd.read_csv(next(tmp_path.rglob("*perplexity-sentence.csv")))
    assert frame["section_index"].tolist() == [4, 4, 4]
    assert frame["sentence_index"].tolist() == [0, 1, 2]
    assert pd.isna(frame.loc[1, "perplexity"])
    assert frame.loc[0, "perplexity"] > 0
    assert frame.loc[2, "perplexity"] > 0


def test_overflow_embeddings_cover_every_token():
    clear_embedding_stats()
    extractor = EmbeddingsExtractor.__new__(EmbeddingsExtractor)
    extractor.embeddings_configurations = {"max_length": 6, "long_response_overlap": 2}
    extractor.model = SimpleNamespace(kind=MODEL_KIND_ENCODER)
    extractor.model_instance = SimpleNamespace(
        config=SimpleNamespace(max_position_embeddings=8),
        hf_device_map={"layer": "cpu"},
    )
    extractor._warned_budget = False
    extractor._warned_chunk = False

    class _Tokenizer:
        all_special_ids = [1, 2]

        def __call__(self, text, add_special_tokens=True, truncation=False, padding=False, **kwargs):
            content = list(range(10, 20))
            if kwargs.get("return_overflowing_tokens"):
                window = kwargs["max_length"]
                stride = kwargs["stride"]
                content_window = window - 2
                windows = []
                start = 0
                while start < len(content):
                    piece = content[start:start + content_window]
                    windows.append([1, *piece, 2])
                    if start + content_window >= len(content):
                        break
                    start += content_window - stride
                return {"input_ids": windows}
            ids = [1, *content, 2] if add_special_tokens else content
            return {"input_ids": ids}

        def convert_ids_to_tokens(self, ids):
            return [str(token_id) for token_id in ids]

    extractor.Tokenizer = SimpleNamespace(tokenizer=_Tokenizer())

    def _hidden(token_ids):
        import torch

        # Dim 0 stores the token id so the stitch can be checked after collection.
        rows = []
        for token_id in token_ids:
            rows.append([float(token_id), 1.0])
        return torch.tensor(rows)

    extractor._hidden_for_ids = _hidden
    embedded = extractor._embeddings_over_context("ignored", {}, observe=True)
    assert embedding_response_summary()["chunked"] == 1
    assert [token for token, _ in embedded] == [str(token_id) for token_id in [1, *range(10, 20), 2]]
    clear_embedding_stats()
