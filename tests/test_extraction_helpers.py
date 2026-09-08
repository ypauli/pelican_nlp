from pelican_nlp.config_defaults import apply_config_defaults
from pelican_nlp.extraction.metric_registry import (
    available_metrics,
    get_metric_runner,
    register_metric,
    run_configured_metrics,
    unregister_metric,
)
from pelican_nlp.extraction.sectioning import iter_section_groups, split_section_text
from pelican_nlp.extraction.token_artifacts import (
    remaining_after_trailing_artifacts,
    strip_trailing_artifact_items,
)


def test_split_section_plain_text():
    parts = split_section_text("hello", {"discourse": False})
    assert parts == ["hello"]


def test_split_section_discourse_without_tag():
    parts = split_section_text("A: x\nB: y", {"discourse": True, "participant_speakertag": None})
    assert parts == ["A: x\nB: y"]


def test_split_section_discourse_with_tag():
    text = "A: investigator\nB: participant answer"
    parts = split_section_text(
        text,
        {"discourse": True, "participant_speakertag": "B"},
        keep_speakertags=False,
    )
    assert parts == ["participant answer"]


def test_iter_section_groups():
    class _Doc:
        cleaned_sections = {"s1": "hello", "s2": "world"}

    groups = list(iter_section_groups(_Doc(), {"discourse": False}))
    assert groups == [("s1", ["hello"]), ("s2", ["world"])]


def test_trailing_artifact_stripping():
    tokens = ["good", "token", "Ġâģ", "¦", "âģ", "©"]
    assert remaining_after_trailing_artifacts(tokens) == 2
    items = [{"token": t} for t in tokens]
    stripped = strip_trailing_artifact_items(items, token_of=lambda item: item["token"])
    assert [item["token"] for item in stripped] == ["good", "token"]


def test_metric_registry_rejects_unknown():
    try:
        get_metric_runner("not-a-metric")
    except ValueError as exc:
        assert "Unsupported metric" in str(exc)
        assert "embeddings" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_metric_registry_plugin_and_run():
    seen = []

    def _runner(corpus):
        seen.append(corpus.name)

    register_metric("dummy_metric", _runner)
    try:
        assert "dummy_metric" in available_metrics()

        class _Corpus:
            name = "acq-animals"
            config = {"metrics_to_extract": ["dummy_metric"]}

        run_configured_metrics(_Corpus())
        assert seen == ["acq-animals"]
    finally:
        unregister_metric("dummy_metric")


def test_config_defaults_fill_optional_keys():
    filled = apply_config_defaults({"task_name": "fluency", "input_file": "text"})
    assert filled["task_name"] == "fluency"
    assert filled["opensmile_feature_extraction"] is False
    assert filled["metrics_to_extract"] == []
    assert filled["options_embeddings"]["distance-from-randomness"] is False
    assert filled["options_embeddings"]["batch_size"] == 1
    assert filled["pipeline_options"]["clean_text"] is True


def test_config_defaults_alias_divergence_flag():
    filled = apply_config_defaults(
        {"options_embeddings": {"divergence_from_optimality": True}}
    )
    assert filled["options_embeddings"]["distance-from-randomness"] is True


def test_config_defaults_do_not_mutate_input():
    original = {"task_name": "fluency"}
    apply_config_defaults(original)
    assert "metrics_to_extract" not in original
