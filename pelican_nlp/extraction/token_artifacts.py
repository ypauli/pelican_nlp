"""Shared trailing-token cleanup used by logits and perplexity extractors."""

import string
from typing import Any, Callable, Iterable, List, Sequence


DEFAULT_TRAILING_ARTIFACT_SEQUENCES = [
    ["Ġâģ", "¦", "âģ", "©"],
    ["âģ", "¦", "âģ", "©"],
]

SPECIAL_TOKENS = {"<s>", "</s>", "<pad>", "<unk>"}


def is_punctuation_token(token):
    token_str = str(token)
    if token_str in SPECIAL_TOKENS:
        return True
    token_core = token_str.replace("▁", "").strip()
    if token_core == "":
        return True
    return all(char in string.punctuation for char in token_core)


def normalize_token_for_matching(token: Any) -> str:
    token_str = str(token).strip().strip('"')
    return token_str.lstrip("▁")


def _normalized_sequences(sequences: Iterable[Sequence[str]]) -> List[List[str]]:
    normalized = [
        [normalize_token_for_matching(tok) for tok in sequence]
        for sequence in sequences
        if sequence
    ]
    return sorted(normalized, key=len, reverse=True)


def remaining_after_trailing_artifacts(
    tokens: Sequence[Any],
    sequences: Iterable[Sequence[str]] | None = None,
) -> int:
    """Return how many leading tokens remain after stripping known suffixes."""
    sequences = sequences or DEFAULT_TRAILING_ARTIFACT_SEQUENCES
    normalized_sequences = _normalized_sequences(sequences)
    if not tokens or not normalized_sequences:
        return len(tokens)

    normalized = [normalize_token_for_matching(tok) for tok in tokens]
    changed = True
    while changed and normalized:
        changed = False
        for sequence in normalized_sequences:
            if len(normalized) < len(sequence):
                continue
            if normalized[-len(sequence) :] == sequence:
                del normalized[-len(sequence) :]
                changed = True
                break
    return len(normalized)


def strip_trailing_artifact_items(
    items: List[Any],
    sequences: Iterable[Sequence[str]] | None = None,
    token_of: Callable[[Any], Any] = lambda item: item,
) -> List[Any]:
    """Drop known trailing artifact tokens from a list of items."""
    if not items:
        return items
    tokens = [token_of(item) for item in items]
    keep = remaining_after_trailing_artifacts(tokens, sequences)
    return items[:keep]
