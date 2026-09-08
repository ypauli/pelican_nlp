import json
import os
from pathlib import Path

import pytest

from pelican_nlp.testing import (
    CONFIG_ONLY_EXAMPLES,
    ExampleSettings,
    compare_derivative_trees,
    discover_example_projects,
    examples_root,
    import_root,
    select_example_projects,
)
from pelican_nlp.testing.example_projects import inspect_example_project


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_inspect_complete_example(tmp_path):
    root = tmp_path / "demo"
    _write(root / "config.yml", "task_name: demo\ninput_file: text\n")
    _write(root / "participants" / "part-01" / "a.txt", "hello")
    _write(root / "derivatives" / "embeddings" / "a_embeddings.csv", "Token,Dim_0\na,0.1\n")
    project = inspect_example_project(root)
    assert project.runnable
    assert project.skip_reason is None
    assert project.config_path.name == "config.yml"
    assert len(project.input_files) == 1
    assert len(project.golden_files) == 1


def test_inspect_skips_incomplete_examples(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert inspect_example_project(empty).skip_reason == "no YAML config"

    yaml_only = tmp_path / "yaml_only"
    _write(yaml_only / "config.yml", "task_name: x\n")
    assert inspect_example_project(yaml_only).skip_reason == "no participants/ directory"

    no_data = tmp_path / "no_data"
    _write(no_data / "config.yml", "task_name: x\n")
    (no_data / "participants").mkdir()
    assert "no input data files" in inspect_example_project(no_data).skip_reason

    no_goldens = tmp_path / "no_goldens"
    _write(no_goldens / "config.yml", "task_name: x\n")
    _write(no_goldens / "participants" / "part-01" / "a.txt", "hello")
    assert "no comparable golden" in inspect_example_project(no_goldens).skip_reason


def test_sidecar_can_disable_and_skip_globs(tmp_path):
    root = tmp_path / "demo"
    _write(root / "config.yml", "task_name: demo\n")
    _write(root / "participants" / "part-01" / "a.txt", "hello")
    _write(root / "derivatives" / "keep" / "a.csv", "Metric,Score\na,1\n")
    _write(root / "derivatives" / "skip-me" / "a.csv", "Metric,Score\na,1\n")
    _write(
        root / ".pelican-test.yml",
        "skip_globs:\n  - skip-me/**\n",
    )
    project = inspect_example_project(root)
    assert project.runnable
    assert [path.name for path in project.golden_files] == ["a.csv"]

    _write(root / ".pelican-test.yml", "enabled: false\n")
    assert inspect_example_project(root).skip_reason == "disabled in .pelican-test.yml"


def test_float_overrides_apply_to_matching_globs(tmp_path):
    golden = tmp_path / "golden"
    actual = tmp_path / "actual"
    _write(golden / "perplexity-section" / "a.csv", "perplexity\n4.0\n")
    _write(actual / "perplexity-section" / "a.csv", "perplexity\n4.1\n")
    _write(golden / "embeddings" / "a.csv", "Dim_0\n0.10\n")
    _write(actual / "embeddings" / "a.csv", "Dim_0\n0.10\n")
    settings = ExampleSettings(
        float_rtol=1e-5,
        float_atol=1e-6,
        float_overrides=(("perplexity-*/**", 0.05, 0.05),),
    )
    assert compare_derivative_trees(actual, golden, settings) == []

    _write(actual / "embeddings" / "a.csv", "Dim_0\n0.50\n")
    messages = compare_derivative_trees(actual, golden, settings)
    assert messages
    assert "embeddings" in messages[0]


def test_dir_glob_matches_nested_lpds_paths():
    skipped = ExampleSettings(skip_globs=("logits/**",))
    nested_logits = Path("logits/part-0/ses-0/task-generated/a.csv")
    nested_ppl = Path("perplexity-section/part-0/ses-0/a.csv")
    assert not skipped.is_comparable(nested_logits)
    assert skipped.is_comparable(nested_ppl)
    rtol, atol = ExampleSettings().tolerances_for(nested_logits)
    assert rtol == 0.02
    assert atol == 0.05
    ppl_rtol, ppl_atol = ExampleSettings().tolerances_for(nested_ppl)
    assert ppl_rtol == 0.02


def test_config_yaml_golden_test_skip_globs(tmp_path):
    root = tmp_path / "demo"
    _write(
        root / "config.yml",
        "task_name: demo\ngolden_test:\n  skip_globs:\n    - logits/**\n",
    )
    _write(root / "participants" / "part-01" / "a.txt", "hello")
    _write(root / "derivatives" / "logits" / "a.csv", "logprob\n-9.15\n")
    _write(root / "derivatives" / "embeddings" / "a.csv", "Dim_0\n0.1\n")
    project = inspect_example_project(root)
    assert project.runnable
    assert [path.parent.name for path in project.golden_files] == ["embeddings"]


def test_default_tolerances_allow_small_logit_drift(tmp_path):
    golden = tmp_path / "golden"
    actual = tmp_path / "actual"
    header = "token,logprob_actual,most_likely_token\n"
    _write(golden / "logits" / "a.csv", header + "se,-9.154272,1\n")
    _write(actual / "logits" / "a.csv", header + "se,-9.140625,1\n")
    assert compare_derivative_trees(actual, golden, ExampleSettings()) == []

    _write(actual / "logits" / "a.csv", header + "se,-1.0,1\n")
    assert compare_derivative_trees(actual, golden, ExampleSettings())

    _write(actual / "logits" / "a.csv", header + "se,-9.140625,other\n")
    assert compare_derivative_trees(actual, golden, ExampleSettings()) == []

    _write(actual / "logits" / "a.csv", header + "xx,-9.140625,1\n")
    messages = compare_derivative_trees(actual, golden, ExampleSettings())
    assert messages
    assert "xx" in messages[0]


def test_perplexity_example_compares_logits_goldens():
    projects = discover_example_projects(examples_root())
    project = next(item for item in projects if item.name == "example_perplexity")
    assert project.runnable
    assert any("logits" in path.as_posix() for path in project.golden_files)
    assert any("perplexity" in path.as_posix() for path in project.golden_files)
    relative = Path("logits/part-0/ses-0/task-generated/a.csv")
    rtol, atol = project.settings.tolerances_for(relative)
    assert rtol == 0.02
    assert atol == 0.05


def test_transcription_example_is_discoverable():
    projects = discover_example_projects(examples_root())
    project = next(item for item in projects if item.name == "example_transcription")
    assert project.input_files
    if project.runnable:
        assert project.skip_reason is None
        assert any(path.suffix == ".txt" for path in project.golden_files)
        return
    assert "no comparable golden" in (project.skip_reason or "")


def test_progress_line_filter():
    from pelican_nlp.testing.runner import format_duration, is_progress_line

    assert is_progress_line("Extracting Logits...\n")
    assert is_progress_line("Logits [1/4] part-0_ses-0_task-generated_group-a_timepoint-0.txt")
    assert is_progress_line("Perplexity [2/4] doc.txt")
    assert not is_progress_line("token,logprob_actual,logprob_max")
    assert not is_progress_line("UserWarning: Can't initialize NVML")
    assert format_duration(65) == "1m 05s"
    assert format_duration(5) == "5s"


def test_isolated_run_prints_progress_not_noise(tmp_path, monkeypatch, capsys):
    from pelican_nlp.testing.runner import run_example_golden

    root = tmp_path / "demo"
    _write(root / "config.yml", "task_name: demo\ninput_file: text\n")
    _write(root / "participants" / "part-01" / "a.txt", "hello")
    _write(root / "derivatives" / "a.csv", "Token\nhello\n")
    project = inspect_example_project(root)

    def _fake_process(lines, returncode):
        class FakeStdin:
            def write(self, data):
                return None

            def close(self):
                return None

        class FakeProcess:
            stdin = FakeStdin()
            stdout = iter(lines)

            def wait(self):
                return returncode

            def poll(self):
                return returncode

        return FakeProcess()

    monkeypatch.delenv("PELICAN_EXAMPLE_LOGS", raising=False)
    monkeypatch.setattr(
        "pelican_nlp.testing.runner.subprocess.Popen",
        lambda *args, **kwargs: _fake_process(
            [
                "UserWarning: Can't initialize NVML\n",
                "Extracting Logits...\n",
                "token,logprob_actual\n",
                "Pipeline ran successfully!\n",
            ],
            0,
        ),
    )
    run_example_golden(project)
    printed = capsys.readouterr().out
    assert "Extracting Logits..." in printed
    assert "Pipeline ran successfully!" in printed
    assert "passed" in printed
    assert "Can't initialize NVML" not in printed
    assert "token,logprob_actual" not in printed


def test_discover_and_select(tmp_path):
    for name in ("example_alpha", "beta"):
        folder = tmp_path / name
        _write(folder / "config.yml", "task_name: x\n")
        _write(folder / "participants" / "part-01" / "a.txt", "hello")
        _write(folder / "derivatives" / "a.csv", "Token\nhello\n")
    projects = discover_example_projects(tmp_path)
    assert [project.name for project in projects] == ["beta", "example_alpha"]

    selected = select_example_projects(projects, "alpha,beta")
    assert [project.name for project in selected] == ["example_alpha", "beta"]

    with pytest.raises(ValueError, match="Unknown example"):
        select_example_projects(projects, "missing")


def test_compare_csv_and_text(tmp_path):
    golden = tmp_path / "golden"
    actual = tmp_path / "actual"
    _write(golden / "emb.csv", "Token,Dim_0\ncat,0.1000000\n")
    _write(actual / "emb.csv", "Token,Dim_0\ncat,0.1000003\n")
    _write(golden / "note.txt", "ok\n")
    _write(actual / "note.txt", "ok\n")
    _write(golden / "ignored.wav", "binary")
    _write(actual / "ignored.wav", "different")
    settings = ExampleSettings()
    assert compare_derivative_trees(actual, golden, settings) == []

    _write(actual / "emb.csv", "Token,Dim_0\ncat,9.0\n")
    messages = compare_derivative_trees(actual, golden, settings)
    assert messages
    assert "emb.csv" in messages[0]

    _write(actual / "emb.csv", "Token,Dim_0\ncat,0.1000003\n")
    _write(golden / "nan.csv", "Metric,Score\nstd,nan\n")
    _write(actual / "nan.csv", "Metric,Score\nstd,NaN\n")
    assert compare_derivative_trees(actual, golden, settings) == []


def test_compare_reports_missing_and_extra(tmp_path):
    golden = tmp_path / "golden"
    actual = tmp_path / "actual"
    _write(golden / "keep.csv", "A\n1\n")
    _write(actual / "other.csv", "A\n1\n")
    messages = compare_derivative_trees(actual, golden, ExampleSettings())
    assert any("missing from run: keep.csv" in item for item in messages)
    assert any("unexpected in run: other.csv" in item for item in messages)


def test_public_examples_root_is_discoverable():
    root = examples_root()
    projects = discover_example_projects(root)
    names = [project.name for project in projects]
    assert names, "expected at least one folder under examples/"
    assert "example_fluency" in names
    fluency = next(project for project in projects if project.name == "example_fluency")
    assert fluency.runnable


def test_config_only_examples_are_not_golden_tested():
    root = examples_root()
    projects = discover_example_projects(root)
    selected = [
        project
        for project in select_example_projects(projects, "all")
        if project.name not in CONFIG_ONLY_EXAMPLES
    ]
    names = {project.name for project in selected}
    assert "example_acoustic-features" not in names
    assert "example_general" not in names
    assert "example_fluency" in names


def test_run_example_golden_isolates_by_default(tmp_path, monkeypatch):
    from pelican_nlp.testing.runner import GoldenRegressionError, run_example_golden

    root = tmp_path / "demo"
    _write(root / "config.yml", "task_name: demo\ninput_file: text\n")
    _write(root / "participants" / "part-01" / "a.txt", "hello")
    _write(root / "derivatives" / "a.csv", "Token\nhello\n")
    project = inspect_example_project(root)

    calls = []

    def _fake_process(returncode, on_stdin=None):
        class FakeStdin:
            def write(self, data):
                if on_stdin is not None:
                    on_stdin(data)

            def close(self):
                return None

        class FakeProcess:
            stdin = FakeStdin()
            stdout = iter(())

            def wait(self):
                return returncode

        return FakeProcess()

    def fake_popen(cmd, **kwargs):
        calls.append((list(cmd), kwargs.get("env")))
        return _fake_process(0)

    monkeypatch.setattr("pelican_nlp.testing.runner.subprocess.Popen", fake_popen)
    run_example_golden(project)
    assert calls
    command, env = calls[0]
    assert command[-1] == "pelican_nlp.testing.isolated_run"
    assert "-m" in command
    assert env is not None
    assert "PYTHONPATH" in env
    assert str(import_root()) in env["PYTHONPATH"].split(os.pathsep)
    assert env.get("PYTHONUNBUFFERED") == "1"

    def failing_popen(cmd, **kwargs):
        return _fake_process(1)

    monkeypatch.setattr("pelican_nlp.testing.runner.subprocess.Popen", failing_popen)
    with pytest.raises(GoldenRegressionError, match="isolated process"):
        run_example_golden(project)

    def failing_popen_with_detail(cmd, **kwargs):
        def on_stdin(data):
            payload = json.loads(data)
            Path(payload["error_file"]).write_text(
                "example_fluency derivatives differ from golden:\nmissing from run: foo.csv\n",
                encoding="utf-8",
            )

        return _fake_process(1, on_stdin=on_stdin)

    monkeypatch.setattr("pelican_nlp.testing.runner.subprocess.Popen", failing_popen_with_detail)
    with pytest.raises(GoldenRegressionError, match="missing from run: foo.csv"):
        run_example_golden(project)


def test_run_example_golden_can_stay_in_process(tmp_path, monkeypatch):
    from pelican_nlp.testing.runner import run_example_golden

    root = tmp_path / "demo"
    _write(root / "config.yml", "task_name: demo\ninput_file: text\n")
    _write(root / "participants" / "part-01" / "a.txt", "hello")
    _write(root / "derivatives" / "a.csv", "Token\nhello\n")
    project = inspect_example_project(root)
    work = tmp_path / "work"
    seen = {}

    def fake_run_in(project, work_dir, update_goldens):
        seen["work_dir"] = work_dir
        seen["update_goldens"] = update_goldens
        return work_dir / "derivatives"

    monkeypatch.setattr("pelican_nlp.testing.runner._run_in", fake_run_in)

    def boom(*args, **kwargs):
        raise AssertionError("subprocess should not run when isolate=False")

    monkeypatch.setattr("pelican_nlp.testing.runner.subprocess.Popen", boom)
    run_example_golden(project, isolate=False, work_dir=work)
    assert seen["work_dir"] == work
    assert seen["update_goldens"] is False
