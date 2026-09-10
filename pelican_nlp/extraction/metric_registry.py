"""Named metric runners used by the Pelican text pipeline.

To add a metric:

1. Implement extraction on the relevant extractor (or a new module).
2. Register a runner with ``register_metric``.
3. List the metric name in YAML ``metrics_to_extract``.

Built-in runners stay backward compatible with existing YAML names.
Nested flags such as ``options_embeddings.semantic-similarity`` still run
inside the embeddings step.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional

MetricRunner = Callable[[object], None]

_EXTRA_RUNNERS: Dict[str, MetricRunner] = {}

BUILTIN_METRIC_NAMES = (
    "logits",
    "embeddings",
    "perplexity",
    "topic_modeling",
)


def register_metric(name: str, runner: MetricRunner) -> None:
    """Register or replace a metric runner (for plugins and tests)."""
    if not name or not str(name).strip():
        raise ValueError("metric name must be a non-empty string")
    _EXTRA_RUNNERS[str(name).strip()] = runner


def unregister_metric(name: str) -> None:
    _EXTRA_RUNNERS.pop(name, None)


def available_metrics() -> List[str]:
    names = set(BUILTIN_METRIC_NAMES)
    names.update(_EXTRA_RUNNERS)
    return sorted(names)


def get_metric_runner(name: str) -> MetricRunner:
    extra = _EXTRA_RUNNERS.get(name)
    if extra is not None:
        return extra
    builtins = _builtin_runners()
    if name not in builtins:
        known = ", ".join(available_metrics())
        raise ValueError(f"Unsupported metric: {name}. Known metrics: {known}")
    return builtins[name]


_METRIC_EXTRAS = {
    "logits": "embeddings",
    "embeddings": "embeddings",
    "perplexity": "embeddings",
    "topic_modeling": "topic",
}


def run_configured_metrics(corpus, metrics: Optional[Iterable[str]] = None) -> None:
    """Run each metric listed in ``metrics`` or ``corpus.config``."""
    from pelican_nlp.extraction.resources import release_gpu
    from pelican_nlp.extras import require_extra

    if metrics is None:
        metrics = (corpus.config or {}).get("metrics_to_extract") or []
    for metric in metrics:
        extra = _METRIC_EXTRAS.get(metric)
        if extra:
            require_extra(extra)
        get_metric_runner(metric)(corpus)
        # Drop leftover FastText / Hub weights before the next metric (e.g. Llama).
        release_gpu()


def run_logits(corpus) -> None:
    from pelican_nlp.extraction.extract_logits import LogitsExtractor

    LogitsExtractor(corpus.config["options_logits"]).process_corpus(corpus)


def run_embeddings(corpus) -> None:
    from pelican_nlp.extraction.extract_embeddings import EmbeddingsExtractor

    EmbeddingsExtractor(corpus.config["options_embeddings"]).process_corpus(corpus)


def run_perplexity(corpus) -> None:
    from pelican_nlp.extraction.extract_perplexity import PerplexityExtractor

    PerplexityExtractor(
        corpus.config["options_perplexity"], corpus.project_folder
    ).process_corpus(corpus)


def run_topic_modeling(corpus) -> None:
    from pelican_nlp.extraction.extract_topic_modeling import TopicModelingExtractor

    TopicModelingExtractor(
        corpus.config["options_topic-modeling"], corpus.project_folder
    ).process_corpus(corpus)


def _builtin_runners() -> Dict[str, MetricRunner]:
    return {
        "logits": run_logits,
        "embeddings": run_embeddings,
        "perplexity": run_perplexity,
        "topic_modeling": run_topic_modeling,
    }
