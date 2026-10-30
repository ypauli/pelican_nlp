"""Keep every token when an embedding response is longer than one window."""

from __future__ import annotations

_stats = {
    "responses": 0,
    "max_tokens": 0,
    "configured_max_length": None,
    "context_length": None,
    "longer_than_configured_max": 0,
    "chunked": 0,
}


def clear_embedding_stats() -> None:
    _stats.update(
        responses=0,
        max_tokens=0,
        configured_max_length=None,
        context_length=None,
        longer_than_configured_max=0,
        chunked=0,
    )


def note_embedding_response(n_tokens, configured_max, context_length, chunked) -> None:
    """Record one encoded response. ``chunked`` is true only past the model context."""
    _stats["responses"] += 1
    _stats["max_tokens"] = max(int(_stats["max_tokens"]), int(n_tokens))
    if configured_max:
        _stats["configured_max_length"] = int(configured_max)
    if context_length:
        _stats["context_length"] = int(context_length)
    if configured_max and n_tokens > configured_max:
        _stats["longer_than_configured_max"] += 1
    if chunked:
        _stats["chunked"] += 1


def embedding_response_summary() -> dict:
    return dict(_stats)


def content_bounds(window, special_ids) -> tuple[int, int]:
    """Return the ``[start, end)`` span of non-special tokens in one window."""
    specials = set(special_ids or [])
    start = 0
    end = len(window)
    while start < end and window[start] in specials:
        start += 1
    while end > start and window[end - 1] in specials:
        end -= 1
    return start, end


def stitch_window_slices(windows, stride, special_ids) -> list[tuple[int, int, int]]:
    """Slices ``(window, start, end)`` that cover the response once.

    The first window keeps its leading special tokens. Later windows drop the
    overlapping content tokens (``stride`` of them). The last window keeps its
    trailing special tokens. Overlap tokens are kept from the earlier window.
    """
    if not windows:
        return []
    overlap = max(0, int(stride))
    last = len(windows) - 1
    slices = []
    for index, window in enumerate(windows):
        content_start, content_end = content_bounds(window, special_ids)
        if index == 0:
            start = 0
            end = len(window) if index == last else content_end
        else:
            start = min(content_start + overlap, content_end)
            end = len(window) if index == last else content_end
        if start < end:
            slices.append((index, start, end))
    return slices
