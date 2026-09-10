"""Two-level pipeline progress: participants (outer) and current-unit work (inner)."""

from __future__ import annotations

import os
import sys
from collections import OrderedDict
from typing import Iterable, List, Optional, Sequence, TextIO

from tqdm import tqdm

_TRUE = {"1", "true", "yes", "on"}

_ACTIVE: Optional["PipelineReporter"] = None


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE


def apply_verbosity(verbose: Optional[bool] = None) -> bool:
    """Enable debug prints when requested; otherwise quiet third-party loaders.

    ``verbose=None`` follows ``PELICAN_DEBUG`` / ``PELICAN_VERBOSE``.
    """
    import pelican_nlp.config as cfg

    if verbose is None:
        enabled = cfg.debug_enabled()
    else:
        enabled = bool(verbose)
        cfg.DEBUG_MODE = enabled
    if not enabled:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return enabled


def bars_enabled(stream: Optional[TextIO] = None, force_plain: Optional[bool] = None) -> bool:
    if force_plain is True:
        return False
    if force_plain is False:
        return True
    if env_flag("TQDM_DISABLE"):
        return False
    stream = stream or sys.stderr
    isatty = getattr(stream, "isatty", None)
    if not callable(isatty):
        return False
    try:
        return bool(isatty())
    except Exception:
        return False


def short_model_name(name) -> Optional[str]:
    if not name:
        return None
    text = str(name).strip()
    if not text:
        return None
    if "/" in text:
        return text.rsplit("/", 1)[-1]
    return text


def pipeline_stage_labels(config, *, text_from_transcriptions: bool = False) -> List[str]:
    """Human-readable stage list for the run header."""
    config = config or {}
    metrics = list(config.get("metrics_to_extract") or [])
    input_file = config.get("input_file")
    labels: List[str] = []

    def metric_label(name: str) -> str:
        if name == "embeddings":
            model = short_model_name((config.get("options_embeddings") or {}).get("model_name"))
            return f"embeddings ({model})" if model else "embeddings"
        if name == "logits":
            model = short_model_name((config.get("options_logits") or {}).get("model_name"))
            return f"logits ({model})" if model else "logits"
        if name == "topic_modeling":
            return "topic modeling"
        return str(name)

    if text_from_transcriptions or input_file == "text":
        labels.append("preprocess")
        labels.extend(metric_label(name) for name in metrics)
        if config.get("create_aggregation_of_results"):
            labels.append("aggregation")
        if config.get("output_document_information"):
            labels.append("document information")
        return labels

    if config.get("transcription"):
        labels.append("transcription")
    if config.get("opensmile_feature_extraction"):
        labels.append("opensmile")
    if config.get("prosogram_extraction"):
        labels.append("prosogram")
    text_metrics = {"embeddings", "logits", "perplexity", "topic_modeling"}
    if any(name in text_metrics for name in metrics):
        labels.append("text-from-transcriptions")
    return labels


def documents_by_unit(documents: Iterable) -> OrderedDict:
    """Group documents by participant/collection folder, preserving first-seen order."""
    from pelican_nlp.utils.lpds_paths import unit_folder_for_document

    grouped: OrderedDict = OrderedDict()
    for document in documents or []:
        key = unit_folder_for_document(document)
        grouped.setdefault(key, []).append(document)
    return grouped


def walk_units(reporter: "PipelineReporter", documents, stage_name: str, *, label: Optional[str] = None):
    """Yield ``(unit, docs)`` and drive the two progress bars."""
    grouped = documents_by_unit(documents)
    reporter.start_stage(stage_name, list(grouped), label=label)
    for unit, docs in grouped.items():
        reporter.start_unit(unit, len(docs))
        yield unit, docs
        reporter.finish_unit()


def active_reporter() -> "PipelineReporter":
    return _ACTIVE if _ACTIVE is not None else NullReporter()


def get_reporter(obj=None) -> "PipelineReporter":
    if obj is not None:
        reporter = getattr(obj, "reporter", None)
        if reporter is not None:
            return reporter
    return active_reporter()


class PipelineReporter:
    """Outer bar = participants finished in the current stage; inner = current unit."""

    def __init__(
        self,
        stream: Optional[TextIO] = None,
        *,
        enabled: bool = True,
        force_plain: Optional[bool] = None,
    ) -> None:
        self.stream = stream or sys.stderr
        self.enabled = enabled
        self.stage_name = ""
        self.stage_label = ""
        self._force_plain = force_plain
        self._use_bars = bool(enabled) and bars_enabled(self.stream, force_plain)
        self._outer = None
        self._inner = None
        self._n_units = 0
        self._unit_done = 0
        self._unit_name = ""
        self._n_items = 0
        self._item_done = 0
        self._prev: Optional[PipelineReporter] = None

    def __enter__(self) -> "PipelineReporter":
        global _ACTIVE
        self._prev = _ACTIVE
        _ACTIVE = self
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        global _ACTIVE
        self.close()
        _ACTIVE = self._prev
        self._prev = None

    def print_summary(self, config, n_units, n_docs, stages, *, now: Optional[str] = None) -> None:
        if not self.enabled:
            return
        config = config or {}
        input_file = config.get("input_file") or "?"
        language = config.get("language") or "?"
        unit_word = "participant" if int(n_units or 0) == 1 else "participants"
        doc_word = "document" if int(n_docs or 0) == 1 else "documents"
        stage_text = " → ".join(stages) if stages else "(none)"
        self._writeln(
            f"Pelican  {input_file}  {language}  {int(n_units or 0)} {unit_word} / {int(n_docs or 0)} {doc_word}"
        )
        self._writeln(f"Stages: {stage_text}")
        if now:
            self._writeln(f"Now: {now}")

    def start_stage(self, name: str, units: Sequence, *, label: Optional[str] = None) -> None:
        self._close_inner()
        if self._outer is not None:
            self._outer.close()
            self._outer = None
        self.stage_name = str(name or "")
        self.stage_label = label or self.stage_name
        names = list(units or [])
        self._n_units = len(names)
        self._unit_done = 0
        self._unit_name = ""
        if not self.enabled:
            return
        if self.stage_label:
            self._writeln(f"Now: {self.stage_label}")
        if self._use_bars and self._n_units:
            self._outer = tqdm(
                total=self._n_units,
                desc="Participants",
                unit="part",
                file=self.stream,
                leave=True,
                dynamic_ncols=True,
                mininterval=0.2,
            )

    def start_unit(self, name: str, n_items: int, already: int = 0) -> None:
        self._close_inner()
        self._unit_name = str(name or "")
        self._n_items = max(0, int(n_items or 0))
        self._item_done = max(0, int(already or 0))
        if not self.enabled or not self._use_bars:
            return
        self._inner = tqdm(
            total=max(self._n_items, 1),
            desc="Current",
            unit="doc",
            file=self.stream,
            leave=False,
            dynamic_ncols=True,
            mininterval=0.2,
        )
        if self._item_done:
            self._inner.n = min(self._item_done, self._inner.total)
            self._inner.refresh()
        postfix = self._unit_name
        self._inner.set_postfix_str(postfix, refresh=False)
        if self._outer is not None:
            self._outer.set_postfix_str(postfix, refresh=True)

    def set_postfix(self, label: str) -> None:
        if not self.enabled or self._inner is None:
            return
        self._inner.set_postfix_str(str(label or ""), refresh=True)

    def advance_item(self, label: str = "") -> None:
        self._item_done += 1
        if not self.enabled or not self._use_bars or self._inner is None:
            return
        if label:
            self._inner.set_postfix_str(str(label), refresh=False)
        self._inner.update(1)

    def finish_unit(self) -> None:
        self._close_inner()
        self._unit_done += 1
        if not self.enabled:
            return
        if self._use_bars:
            if self._outer is not None:
                if self._unit_name:
                    self._outer.set_postfix_str(self._unit_name, refresh=False)
                self._outer.update(1)
            return
        total = self._n_units or self._unit_done
        stage = self.stage_name or self.stage_label or "stage"
        unit = self._unit_name or "?"
        self._writeln(f"{stage}  {self._unit_done}/{total}  {unit}")

    def status(self, msg: str) -> None:
        if not self.enabled:
            return
        self._writeln(msg)

    def warn(self, msg: str) -> None:
        text = str(msg)
        if not text.lower().startswith("warning") and not text.lower().startswith("error"):
            text = f"Warning: {text}"
        self._writeln(text)

    def close(self) -> None:
        self._close_inner()
        if self._outer is not None:
            self._outer.close()
            self._outer = None

    def _close_inner(self) -> None:
        if self._inner is not None:
            self._inner.close()
            self._inner = None

    def _writeln(self, msg: str) -> None:
        text = str(msg)
        if self._outer is not None or self._inner is not None:
            tqdm.write(text, file=self.stream)
            return
        print(text, file=self.stream, flush=True)


class NullReporter(PipelineReporter):
    """No bars or status lines; warnings still go to stderr."""

    def __init__(self, stream: Optional[TextIO] = None) -> None:
        super().__init__(stream=stream or sys.stderr, enabled=False, force_plain=True)

    def __enter__(self) -> "NullReporter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None


class UnitTracker:
    """Tick bars as documents finish, including when batching interleaves units."""

    def __init__(self, reporter: PipelineReporter, documents, stage_name: str, *, label: Optional[str] = None):
        self.reporter = reporter
        self.grouped = documents_by_unit(documents)
        self.totals = {unit: len(docs) for unit, docs in self.grouped.items()}
        self.remaining = dict(self.totals)
        self._active = None
        self.reporter.start_stage(stage_name, list(self.grouped), label=label)

    def start_document(self, document) -> str:
        from pelican_nlp.utils.lpds_paths import unit_folder_for_document

        unit = unit_folder_for_document(document)
        if self._active != unit:
            already = self.totals.get(unit, 0) - self.remaining.get(unit, 0)
            self.reporter.start_unit(unit, self.totals.get(unit, 1), already=already)
            self._active = unit
        return unit

    def finish_document(self, document, label: Optional[str] = None) -> None:
        from pelican_nlp.utils.lpds_paths import unit_folder_for_document

        unit = unit_folder_for_document(document)
        self.start_document(document)
        self.reporter.advance_item(label or getattr(document, "name", "") or unit)
        if unit in self.remaining:
            self.remaining[unit] -= 1
            if self.remaining[unit] <= 0:
                self.reporter.finish_unit()
                self._active = None
