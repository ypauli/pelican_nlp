from pathlib import Path

from pelican_nlp.core.corpus import Corpus, similarity_aggregation_family


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_similarity_family_is_dynamic():
    assert similarity_aggregation_family("a_semantic-similarity-window-2.csv") == "window_2"
    assert similarity_aggregation_family("a_semantic-similarity-window-5.csv") == "window_5"
    assert similarity_aggregation_family("a_semantic-similarity-window-20.csv") == "window_20"
    assert similarity_aggregation_family("a_semantic-similarity-sentence.csv") == "sentence"
    assert similarity_aggregation_family("a_semantic-similarity-window-details-2.csv") is None
    assert similarity_aggregation_family("a_embeddings.csv") is None


def test_aggregation_includes_any_window_size(tmp_path):
    derivatives = tmp_path / "derivatives"
    summary = (
        "Metric,Similarity_Score\n"
        "mean_of_window_means,0.5\n"
        "mean_of_window_medians,0.4\n"
    )
    _write(
        derivatives
        / "semantic-similarity-window-5"
        / "part-01"
        / "task-fluency"
        / "part-01_task-fluency_semantic-similarity-window-5.csv",
        summary,
    )
    _write(
        derivatives
        / "semantic-similarity-window-2"
        / "part-01"
        / "task-fluency"
        / "part-01_task-fluency_semantic-similarity-window-2.csv",
        "Metric,Similarity_Score\nmean_of_window_means,0.25\n",
    )
    _write(
        derivatives
        / "semantic-similarity-sentence"
        / "part-01"
        / "task-fluency"
        / "part-01_task-fluency_semantic-similarity-sentence.csv",
        "Metric,Similarity_Score\nmean_of_window_means,0.75\n",
    )

    corpus = Corpus("acq-animals", [], {}, tmp_path)
    corpus.create_corpus_results_consolidation_csv()
    output = derivatives / "aggregations" / "acq-animals_semantic-similarity_comprehensive_aggregation.csv"
    text = output.read_text(encoding="utf-8")
    assert "window_5_avg_per_window_mean_of_window_means" in text
    assert "window_2_avg_per_window_mean_of_window_means" in text
    assert "sentence_avg_over_all_sentences_mean_of_window_means" in text
    assert ",0.5" in text
    assert ",0.25" in text
    assert ",0.75" in text
