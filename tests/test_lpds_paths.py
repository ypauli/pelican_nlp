from pathlib import Path
from types import SimpleNamespace

from pelican_nlp.utils.lpds_paths import (
    aggregation_unit_key,
    csv_output_filename,
    derivatives_output_path,
    derivatives_subdir,
    file_matches_task,
    grouped_corpus_jobs,
    is_data_file,
    is_participant_folder,
    resolve_unit_folder,
    unit_kind,
)


def test_participant_vs_collection_folder_names():
    assert is_participant_folder("part-01")
    assert unit_kind("part-01") == "participant"
    assert not is_participant_folder("stories")
    assert unit_kind("stories") == "collection"
    assert not is_participant_folder("part-")
    assert not is_participant_folder("participants")


def test_fluency_derivatives_layout_unchanged():
    filename = "part-01_task-fluency_cat-semantic_acq-animals_text.txt"
    relative = derivatives_subdir("part-01", filename)
    assert relative == Path("part-01") / "task-fluency"
    assert csv_output_filename(filename, "embeddings") == (
        "part-01_task-fluency_cat-semantic_acq-animals_embeddings.csv"
    )


def test_image_description_keeps_session_then_task():
    filename = "part-01_ses-01_task-imgdesc_acq-beach_text.txt"
    relative = derivatives_subdir("part-01", filename)
    assert relative == Path("part-01") / "ses-01" / "task-imgdesc"


def test_collection_file_has_no_extra_path_segments():
    relative = derivatives_subdir("stories", "hansel.txt")
    assert relative == Path("stories")
    assert csv_output_filename("hansel.txt", "embeddings") == "hansel_embeddings.csv"
    assert csv_output_filename("hansel_and_gretel.txt", "embeddings") == (
        "hansel_and_gretel_embeddings.csv"
    )


def test_missing_part_falls_back_without_raising(tmp_path):
    path = derivatives_output_path(tmp_path, None, "hansel.txt", "embeddings")
    assert Path(path) == tmp_path / "embeddings" / "unassigned" / "hansel_embeddings.csv"


def test_filename_part_is_used_when_source_folder_missing():
    assert resolve_unit_folder(None, {"part": "01"}) == "part-01"
    assert str(derivatives_subdir(None, "part-01_task-fluency_text.txt")) == str(
        Path("part-01") / "task-fluency"
    )


def test_task_filter_keeps_untagged_files():
    assert file_matches_task({}, "fluency")
    assert file_matches_task({"task": "fluency"}, "fluency")
    assert not file_matches_task({"task": "imgdesc"}, "fluency")
    assert file_matches_task({"task": "fluency"}, None)


def test_readme_is_not_a_data_file():
    assert is_data_file("hansel.txt")
    assert is_data_file("part-01_task-fluency_text.txt")
    assert not is_data_file("readme.txt")
    assert not is_data_file("notes.txt")
    assert not is_data_file("codes.csv")


def test_aggregation_key_comes_from_unit_folder(tmp_path):
    derivatives = tmp_path / "derivatives"
    fluency = (
        derivatives
        / "semantic-similarity-window-2"
        / "part-01"
        / "task-fluency"
        / "part-01_task-fluency_acq-animals_semantic-similarity-window-2.csv"
    )
    stories = derivatives / "embeddings" / "stories" / "hansel_and_gretel_embeddings.csv"
    fluency.parent.mkdir(parents=True)
    stories.parent.mkdir(parents=True)
    fluency.write_text("Metric,Similarity_Score\n", encoding="utf-8")
    stories.write_text("Token,Dim_0\n", encoding="utf-8")
    assert aggregation_unit_key(fluency, derivatives) == "part-01"
    assert aggregation_unit_key(stories, derivatives) == "stories"


def test_grouped_corpus_jobs_fluency_then_collection_leftovers():
    fluency = SimpleNamespace(name="part-01_task-fluency_acq-animals_text.txt")
    extra = SimpleNamespace(name="part-01_task-fluency_acq-test_text.txt")
    story = SimpleNamespace(name="hansel.txt")
    units = [
        SimpleNamespace(name="part-01", documents=[fluency, extra]),
        SimpleNamespace(name="stories", documents=[story]),
    ]
    jobs = list(grouped_corpus_jobs(units, "acq", ["animals", "clothes"]))
    names = [name for name, _ in jobs]
    assert names == ["acq-animals", "stories"]
    assert jobs[0][1] == [fluency]
    assert jobs[1][1] == [story]


def test_grouped_corpus_jobs_without_corpus_key_uses_folders():
    story = SimpleNamespace(name="hansel.txt")
    units = [SimpleNamespace(name="stories", documents=[story])]
    jobs = list(grouped_corpus_jobs(units, None, None))
    assert jobs == [("stories", [story])]
