"""Fuse a primary Whisper transcript with a second (default) model.

The primary model is kept where it is clean. High-severity spans (decoder loops,
subtitle credits, script that contradicts the configured language) are replaced
with overlapping clean utterances from the patch model. Spans that both models
taint are dropped and recorded, not rewritten.

This module has no torch dependency so the splice can be unit-tested in isolation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

TOKEN_RE = re.compile(r"[A-Za-zÄÖÜäöüß0-9']+")
NON_LATIN_RE = re.compile(
    r"[\u0590-\u05FF\u0600-\u06FF\u0400-\u04FF\u3040-\u30FF\u4E00-\u9FFF]"
)
CJK_RE = re.compile(r"[\u3040-\u30FF\u4E00-\u9FFF]")
CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
HEBREW_RE = re.compile(r"[\u0590-\u05FF]")

BACKCHANNELS = {
    "mhm", "hm", "hmm", "äh", "ähm", "ah", "oh", "ja", "nein", "ok", "okay",
    "oké", "mh", "aha", "genau", "also", "und", "so", "go",
}

BOILERPLATE = [
    (re.compile(r"untertitel(?:ung)?(?:\s+des)?\s+zdf", re.I), "boilerplate_zdf"),
    (re.compile(r"subtitles?\s+by", re.I), "boilerplate_subtitles"),
    (re.compile(r"thanks?\s+for\s+watching", re.I), "boilerplate_thanks"),
    (re.compile(r"thank\s+you\s+for\s+watching", re.I), "boilerplate_thanks"),
    (re.compile(r"please\s+subscribe", re.I), "boilerplate_subscribe"),
    (re.compile(r"like\s+and\s+subscribe", re.I), "boilerplate_subscribe"),
    (re.compile(r"amara\.org", re.I), "boilerplate_amara"),
    (re.compile(r"i['’]m going to make a new one", re.I), "english_hallucination"),
    (re.compile(r"see you next time", re.I), "english_hallucination"),
    (re.compile(r"\bpeace crowd\b", re.I), "english_hallucination"),
    (re.compile(r"\bnettopp\b", re.I), "foreign_hallucination"),
    (re.compile(r"\bgå inn\b", re.I), "foreign_hallucination"),
]

ENGLISH_NOISE = {
    "the", "and", "you", "that", "this", "with", "for", "are", "have", "was",
    "thank", "thanks", "please", "peace", "crowd", "going", "make", "new",
    "one", "watching", "subscribe", "channel", "video",
}

SOURCE_PRIMARY = "primary"
SOURCE_PATCH = "patch"

_ENGLISH_NAMES = {"english", "en", "eng"}
_LANGUAGE_SCRIPTS: dict[str, tuple[str, ...]] = {}
for _name in (
    "german", "de", "english", "en", "eng", "french", "fr", "spanish", "es",
    "italian", "it", "dutch", "nl", "portuguese", "pt", "swedish", "sv",
    "danish", "da", "norwegian", "no", "finnish", "fi", "polish", "pl",
):
    _LANGUAGE_SCRIPTS[_name] = ("latin",)
for _name in ("chinese", "zh", "cmn", "yue", "japanese", "ja", "korean", "ko"):
    _LANGUAGE_SCRIPTS[_name] = ("cjk",)
for _name in ("russian", "ru", "ukrainian", "uk", "bulgarian", "bg"):
    _LANGUAGE_SCRIPTS[_name] = ("cyrillic",)
for _name in ("arabic", "ar", "persian", "fa", "farsi"):
    _LANGUAGE_SCRIPTS[_name] = ("arabic",)
for _name in ("hebrew", "he", "iw"):
    _LANGUAGE_SCRIPTS[_name] = ("hebrew",)


@dataclass
class Quality:
    flags: list[str] = field(default_factory=list)
    penalty: float = 0.0
    high_severity: bool = False
    n_tokens: int = 0

    def add(self, flag: str, penalty: float, high: bool = False) -> None:
        self.flags.append(flag)
        self.penalty += penalty
        if high:
            self.high_severity = True


@dataclass
class Utterance:
    text: str
    start: float
    end: float
    speaker: str
    quality: Quality


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


def ordered_span(start: float, end: float, min_dur: float = 0.02) -> tuple[float, float]:
    a, b = float(start), float(end)
    if b < a:
        a, b = b, a
    if b - a < min_dur:
        b = a + min_dur
    return a, b


def _configured_scripts(language: str | None) -> set[str]:
    if not language:
        return set()
    return set(_LANGUAGE_SCRIPTS.get(language.strip().lower(), ()))


def _unexpected_non_latin(raw: str, language: str | None) -> list[str]:
    """Non-Latin characters that do not belong to the configured language."""
    chars = NON_LATIN_RE.findall(raw)
    if not chars:
        return []
    scripts = _configured_scripts(language)
    if not scripts:
        # No language pin: do not treat other scripts as hallucinations.
        return []
    unexpected = []
    for char in chars:
        if CJK_RE.match(char) and "cjk" in scripts:
            continue
        if CYRILLIC_RE.match(char) and "cyrillic" in scripts:
            continue
        if ARABIC_RE.match(char) and "arabic" in scripts:
            continue
        if HEBREW_RE.match(char) and "hebrew" in scripts:
            continue
        if "latin" in scripts:
            unexpected.append(char)
            continue
        # Configured for a non-Latin script: only flag *other* non-Latin families.
        if CJK_RE.match(char) and "cjk" not in scripts:
            unexpected.append(char)
        elif CYRILLIC_RE.match(char) and "cyrillic" not in scripts:
            unexpected.append(char)
        elif ARABIC_RE.match(char) and "arabic" not in scripts:
            unexpected.append(char)
        elif HEBREW_RE.match(char) and "hebrew" not in scripts:
            unexpected.append(char)
    return unexpected


def analyze_text(text: str, duration: float | None = None, language: str | None = None) -> Quality:
    """Reference-free quality score. Higher penalty = worse."""
    q = Quality()
    raw = text or ""
    tokens = tokenize(raw)
    q.n_tokens = len(tokens)
    if not raw.strip():
        return q

    unexpected = _unexpected_non_latin(raw, language)
    if len(unexpected) >= 3:
        q.add(f"non_latin:{len(unexpected)}", 40 + 0.05 * len(unexpected), high=True)

    language_key = (language or "").strip().lower()
    if q.n_tokens >= 6 and language_key and language_key not in _ENGLISH_NAMES:
        en = sum(1 for t in tokens if t in ENGLISH_NOISE)
        if en / q.n_tokens >= 0.4:
            q.add("english_burst", 30, high=True)

    for cre, name in BOILERPLATE:
        if cre.search(raw):
            q.add(name, 50, high=True)

    i = 0
    n = len(tokens)
    while i < n:
        j = i + 1
        while j < n and tokens[j] == tokens[i]:
            j += 1
        run = j - i
        if run >= 8:
            q.add(f"word_run:{tokens[i]}x{run}", min(80, 4 * run), high=True)
        elif run >= 5 and tokens[i] not in BACKCHANNELS:
            q.add(f"word_run:{tokens[i]}x{run}", 2 * run, high=False)
        elif run >= 6:
            q.add(f"word_run:{tokens[i]}x{run}", run, high=False)
        i = j

    used = [False] * n
    for ngram in range(8, 1, -1):
        i = 0
        while i + ngram * 3 <= n:
            if used[i]:
                i += 1
                continue
            phrase = tuple(tokens[i : i + ngram])
            if len(set(phrase)) == 1:
                i += 1
                continue
            reps = 1
            k = i + ngram
            while k + ngram <= n and tuple(tokens[k : k + ngram]) == phrase:
                reps += 1
                k += ngram
            spanned = reps * ngram
            if reps >= 3 and spanned >= 12:
                high = reps >= 6 or spanned >= 20
                q.add(
                    f"phrase_loop:{reps}x{' '.join(phrase)[:40]}",
                    min(80, 3 * spanned),
                    high=high,
                )
                for t in range(i, k):
                    used[t] = True
                i = k
            else:
                i += 1

    if duration is not None and duration >= 2.0 and q.n_tokens >= 30:
        rate = q.n_tokens / duration
        if rate >= 8.0:
            q.add(f"rate:{rate:.1f}w/s", 12, high=False)
        elif rate >= 6.0:
            q.add(f"rate:{rate:.1f}w/s", 4, high=False)

    return q


def mark_identical_runs(utts: list[Utterance], min_repeat: int = 6) -> None:
    i = 0
    while i < len(utts):
        key = utts[i].text.strip().lower()
        j = i + 1
        while j < len(utts) and utts[j].text.strip().lower() == key and key:
            j += 1
        run = j - i
        if run >= min_repeat:
            toks = tokenize(utts[i].text)
            bc = len(toks) <= 2 and all(t in BACKCHANNELS or t == "..." for t in toks)
            if not bc:
                for u in utts[i:j]:
                    u.quality.add(f"utterance_run:x{run}", min(80, 2 * run), high=True)
        i = j if j > i else i + 1


def emitible(text: str, language: str | None = None) -> bool:
    """False for leftover hallucination stubs we should not write out."""
    if not (text or "").strip():
        return False
    q = analyze_text(text, language=language)
    if q.high_severity:
        return False
    tokens = tokenize(text)
    # Short doubled words ("Hallo hallo") are kept: they are often real speech.
    if tokens and len(set(tokens)) == 1 and tokens[0] not in BACKCHANNELS and len(tokens) >= 8:
        return False
    return True


def is_clean(u: Utterance, language: str | None = None) -> bool:
    return (not u.quality.high_severity) and emitible(u.text, language=language)


def spans_overlap(a0: float, a1: float, b0: float, b1: float, min_ov: float = 0.05) -> bool:
    return min(a1, b1) - max(a0, b0) > min_ov or abs(a0 - b0) < 0.12


def speaker_at(t: float, segs: list[dict[str, Any]]) -> str | None:
    best = None
    best_start = -1e18
    for s in segs or []:
        a, b = ordered_span(s.get("start", 0.0), s.get("end", 0.0))
        if a <= t < b and a >= best_start:
            best = str(s.get("speaker") or "UNKNOWN")
            best_start = a
    return best


def build_utterances(data: dict[str, Any] | None, language: str | None = None) -> list[Utterance]:
    out: list[Utterance] = []
    if not data:
        return out
    for u in data.get("utterance_data") or []:
        text = (u.get("text") or "").strip()
        start, end = ordered_span(u.get("start_time", 0.0), u.get("end_time", 0.0))
        q = analyze_text(text, duration=end - start, language=language)
        out.append(
            Utterance(
                text=text,
                start=start,
                end=end,
                speaker=str(u.get("speaker") or "UNKNOWN"),
                quality=q,
            )
        )
    mark_identical_runs(out)
    return out


def row_from_utterance(
    u: Utterance,
    source: str,
    reason: str,
    segs: list[dict[str, Any]],
    language: str | None = None,
) -> dict[str, Any] | None:
    text = (u.text or "").strip()
    if not emitible(text, language=language):
        return None
    mid = 0.5 * (u.start + u.end)
    speaker = speaker_at(mid, segs) or u.speaker or "UNKNOWN"
    flags = sorted(set(u.quality.flags))[:12]
    return {
        "start": round(u.start, 3),
        "end": round(u.end, 3),
        "speaker": speaker,
        "text": text,
        "chosen_source": source,
        "decision": reason,
        "penalty_primary": round(u.quality.penalty, 2) if source == SOURCE_PRIMARY else None,
        "penalty_patch": round(u.quality.penalty, 2) if source == SOURCE_PATCH else None,
        "flags_primary": flags if source == SOURCE_PRIMARY else [],
        "flags_patch": flags if source == SOURCE_PATCH else [],
    }


def dropped_row(u: Utterance, reason: str) -> dict[str, Any]:
    return {
        "start": round(u.start, 3),
        "end": round(u.end, 3),
        "speaker": u.speaker,
        "text": "",
        "chosen_source": None,
        "decision": reason,
        "penalty_primary": round(u.quality.penalty, 2),
        "penalty_patch": None,
        "flags_primary": sorted(set(u.quality.flags))[:12],
        "flags_patch": [],
    }


def model_name(data: dict[str, Any] | None) -> str | None:
    if not data:
        return None
    try:
        return data["metadata"]["models_used"]["Transcription"]["model"]
    except (KeyError, TypeError):
        return None


def fuse_transcriptions(
    primary: dict[str, Any] | None,
    patch: dict[str, Any] | None,
    *,
    language: str | None = None,
    session: str | None = None,
) -> dict[str, Any]:
    """Fuse two ``*_allOutputs.json`` payloads. ``segments`` includes dropped spans."""
    if primary is None and patch is None:
        raise ValueError("fuse_transcriptions requires at least one transcript")

    only: str | None = None
    if primary is None:
        only = SOURCE_PATCH
    elif patch is None:
        only = SOURCE_PRIMARY

    source = primary or patch
    segs = (primary or patch).get("speaker_segments") or []
    d_utts = build_utterances(primary, language=language) if primary else []
    a_utts = build_utterances(patch, language=language) if patch else []

    d_clean = [u for u in d_utts if is_clean(u, language=language)] if only != SOURCE_PATCH else []
    a_clean = [u for u in a_utts if is_clean(u, language=language)] if only != SOURCE_PRIMARY else []
    d_bad = [u for u in d_utts if not is_clean(u, language=language)] if only != SOURCE_PATCH else []

    rows: list[dict[str, Any]] = []
    kept: list[Utterance] = []
    used_a = [False] * len(a_clean)

    keep_reason = "single_source_primary" if only == SOURCE_PRIMARY else "keep_primary_clean"
    for u in d_clean:
        row = row_from_utterance(u, SOURCE_PRIMARY, keep_reason, segs, language=language)
        if row:
            rows.append(row)
            kept.append(u)

    for du in d_bad:
        for i, au in enumerate(a_clean):
            if used_a[i]:
                continue
            if any(spans_overlap(au.start, au.end, k.start, k.end) for k in kept):
                continue
            if spans_overlap(du.start, du.end, au.start, au.end):
                row = row_from_utterance(
                    au, SOURCE_PATCH, "patch_from_default_high_taint", segs, language=language
                )
                if row:
                    rows.append(row)
                    kept.append(au)
                    used_a[i] = True

    fill_reason = "single_source_patch" if only == SOURCE_PATCH else "hole_fill_from_default"
    for i, au in enumerate(a_clean):
        if used_a[i]:
            continue
        if any(spans_overlap(au.start, au.end, k.start, k.end) for k in kept):
            continue
        if only != SOURCE_PATCH and any(spans_overlap(au.start, au.end, du.start, du.end) for du in d_utts):
            # Overlaps some primary span that was not kept: still a patch, not a hole.
            if any(spans_overlap(au.start, au.end, du.start, du.end) for du in d_bad):
                row = row_from_utterance(
                    au, SOURCE_PATCH, "patch_from_default_high_taint", segs, language=language
                )
                if row:
                    rows.append(row)
                    kept.append(au)
                    used_a[i] = True
            continue
        row = row_from_utterance(au, SOURCE_PATCH, fill_reason, segs, language=language)
        if row:
            rows.append(row)
            kept.append(au)
            used_a[i] = True

    for du in d_bad:
        if not any(spans_overlap(du.start, du.end, k.start, k.end) for k in kept):
            rows.append(dropped_row(du, "dropped_both_tainted"))

    rows.sort(key=lambda r: (r["start"], r["end"], r["decision"] == "dropped_both_tainted"))

    kept_rows = [r for r in rows if r["decision"] != "dropped_both_tainted" and r.get("text")]
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["decision"]] = counts.get(r["decision"], 0) + 1
        if r.get("chosen_source"):
            key = f"source:{r['chosen_source']}"
            counts[key] = counts.get(key, 0) + 1

    metadata = (source or {}).get("metadata") or {}
    return {
        "session": session,
        "audio_file_path": (source or {}).get("audio_file_path"),
        "length_seconds": metadata.get("length_seconds"),
        "language": language,
        "models": {
            SOURCE_PRIMARY: model_name(primary),
            SOURCE_PATCH: model_name(patch),
        },
        "n_segments": len(kept_rows),
        "n_dropped": sum(1 for r in rows if r["decision"] == "dropped_both_tainted"),
        "decision_counts": counts,
        "segments": rows,
        "kept_segments": kept_rows,
    }


def fused_utterances(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Map fusion rows onto the AudioFile utterance schema."""
    utterances = []
    for row in payload.get("kept_segments") or []:
        utterances.append(
            {
                "text": row["text"],
                "start_time": row["start"],
                "end_time": row["end"],
                "speaker": row["speaker"],
                "confidence": 1.0,
                "chosen_source": row.get("chosen_source"),
                "decision": row.get("decision"),
            }
        )
    return utterances


def apply_fusion_to_audio_file(audio_file, payload: dict[str, Any]) -> None:
    """Replace canonical utterances/text on ``audio_file`` with the fused transcript."""
    utterances = fused_utterances(payload)
    audio_file.combined_utterances = utterances
    audio_file.transcript_text = " ".join(u["text"] for u in utterances)
    audio_file.register_model(
        "TranscriptionPatching",
        {
            "primary_model": (payload.get("models") or {}).get(SOURCE_PRIMARY),
            "patch_model": (payload.get("models") or {}).get(SOURCE_PATCH),
            "n_segments": payload.get("n_segments"),
            "n_dropped": payload.get("n_dropped"),
            "decision_counts": payload.get("decision_counts"),
        },
    )


def audio_file_to_source_dict(audio_file) -> dict[str, Any]:
    """JSON-shaped snapshot of an AudioFile for fusion (does not write disk)."""
    return {
        "audio_file_path": getattr(audio_file, "file", None),
        "metadata": getattr(audio_file, "metadata", None) or {},
        "utterance_data": list(getattr(audio_file, "combined_utterances", None) or []),
        "speaker_segments": list(getattr(audio_file, "speaker_segments", None) or []),
        "transcript_text": getattr(audio_file, "transcript_text", None),
    }
