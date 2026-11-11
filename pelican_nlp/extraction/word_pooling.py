"""Pool subword embedding pieces into words.

WordPiece marks a continuation with ``##``; an unmarked token starts a word.
SentencePiece (``▁``) and byte-level BPE (``Ġ``) do the opposite: the marker
starts a word and an unmarked token continues it.
"""

from __future__ import annotations

import string

import numpy as np

_SPECIAL = {
    "<bos>",
    "<eos>",
    "<s>",
    "</s>",
    "<pad>",
    "<unk>",
    "<mask>",
    "[CLS]",
    "[SEP]",
    "[PAD]",
    "[UNK]",
    "[MASK]",
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|endoftext|>",
}


def _is_special(token) -> bool:
    text = str(token)
    if text in _SPECIAL:
        return True
    return text.startswith("<") and text.endswith(">") and len(text) > 2


def _boundary_scheme(pairs) -> str:
    """Choose word boundaries from the markers actually present.

    ``prefix`` covers SentencePiece and byte-level BPE. ``wordpiece`` covers
    ``##`` continuations. ``plain`` means there is nothing to join.
    """
    prefix = False
    wordpiece = False
    for token, _vector in pairs:
        text = str(token)
        if text.startswith("##"):
            wordpiece = True
        elif text.startswith(("▁", "Ġ")):
            prefix = True
    if prefix:
        return "prefix"
    if wordpiece:
        return "wordpiece"
    return "plain"


def _piece(token):
    """Return ``(core, boundary)`` or ``None`` for a special token.

    ``boundary`` is True at a new word, False for a continuation, and None when
    the token has no subword marker. Unmarked tokens are resolved later, because
    WordPiece and SentencePiece disagree about what an unmarked token means.
    """
    if _is_special(token):
        return None
    text = str(token)
    if text.startswith("##"):
        return text[2:], False
    if text.startswith("▁") or text.startswith("Ġ"):
        return text[1:], True
    return text, None


def _attaches_to_previous_word(core) -> bool:
    """Punctuation with no subword marker stays on the preceding word."""
    return bool(core) and all(character in string.punctuation for character in core)


def pool_subword_embeddings(pairs):
    """Mean-pool subword vectors that belong to the same word.

    Punctuation with no word-boundary marker stays on the preceding word, so a
    sentence-final period remains visible to sentence alignment. Special tokens
    are dropped.
    """
    if not pairs:
        return []
    parsed = []
    marked = False
    for token, vector in pairs:
        piece = _piece(token)
        if piece is None:
            parsed.append(None)
            continue
        core, boundary = piece
        if boundary is not None:
            marked = True
        parsed.append((core, boundary, vector))
    if not marked:
        words = []
        for item in parsed:
            if not item or not item[0]:
                continue
            core, _, vector = item
            words.append((core, vector))
        return words

    words = []
    cores = []
    vectors = []
    scheme = _boundary_scheme(pairs)

    def flush():
        if not vectors:
            return
        text = "".join(cores).strip()
        if text:
            stacked = np.stack([np.asarray(vector, dtype=float) for vector in vectors], axis=0)
            words.append((text, np.mean(stacked, axis=0)))
        cores.clear()
        vectors.clear()

    for item in parsed:
        if item is None:
            continue
        core, boundary, vector = item
        if boundary is None:
            if scheme == "prefix":
                # ▁ / Ġ already started the word. This piece belongs to it.
                boundary = False
            else:
                # WordPiece: an unmarked word starts a new row. Punctuation stays.
                boundary = not _attaches_to_previous_word(core)
        if boundary and vectors:
            flush()
        cores.append(core)
        vectors.append(vector)
    flush()
    return words
