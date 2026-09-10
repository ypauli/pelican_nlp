"""CLI dispatch for pelican-run --run-tests vs the project pipeline."""

import pytest

from pelican_nlp.cli import main
from pelican_nlp.testing.cli import main as examples_main
from pelican_nlp.testing.paths import examples_root, import_root, pytest_root
from pelican_nlp.testing.paths import tests_root as packaged_tests_root


def test_run_tests_dispatches_to_pytest_not_pipeline(monkeypatch):
    called = {}

    def fake_suite(**kwargs):
        called.update(kwargs)
        return 0

    def boom(*args, **kwargs):
        raise AssertionError("pipeline should not run")

    monkeypatch.setattr("pelican_nlp.testing.cli.run_pytest_suite", fake_suite)
    monkeypatch.setattr("pelican_nlp.main.Pelican", boom)

    with pytest.raises(SystemExit) as exited:
        main(["--run-tests"])
    assert exited.value.code == 0
    assert called["examples"] is None
    assert called["update_goldens"] is False
    assert called["pytest_args"] == []


def test_run_tests_examples_defaults_to_all(monkeypatch):
    called = {}

    def fake_suite(**kwargs):
        called.update(kwargs)
        return 0

    monkeypatch.setattr("pelican_nlp.testing.cli.run_pytest_suite", fake_suite)
    with pytest.raises(SystemExit) as exited:
        main(["--run-tests", "--examples"])
    assert exited.value.code == 0
    assert called["examples"] == "all"


def test_run_tests_examples_selection_and_extra_args(monkeypatch):
    called = {}

    def fake_suite(**kwargs):
        called.update(kwargs)
        return 17

    monkeypatch.setattr("pelican_nlp.testing.cli.run_pytest_suite", fake_suite)
    with pytest.raises(SystemExit) as exited:
        main(["--run-tests", "--examples", "fluency,discourse", "--", "-k", "test_cli", "-q"])
    assert exited.value.code == 17
    assert called["examples"] == "fluency,discourse"
    assert called["pytest_args"] == ["-k", "test_cli", "-q"]


def test_examples_flag_requires_run_tests():
    with pytest.raises(SystemExit) as exited:
        main(["--examples"])
    assert exited.value.code != 0


def test_example_root_flag_requires_run_tests():
    with pytest.raises(SystemExit) as exited:
        main(["--example-root", "/tmp/examples"])
    assert exited.value.code != 0


def test_pipeline_runs_when_cwd_has_one_yaml(monkeypatch, tmp_path):
    (tmp_path / "config.yml").write_text("input_file: text\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    seen = {}

    class FakePelican:
        def __init__(self, path, **kwargs):
            seen["path"] = path
            seen["verbose"] = kwargs.get("verbose")

        def run(self):
            seen["ran"] = True

    monkeypatch.setattr("pelican_nlp.main.Pelican", FakePelican)
    main([])
    assert seen.get("ran") is True
    assert seen["path"].endswith("config.yml")
    assert seen.get("verbose") is False


def test_pipeline_verbose_flag(monkeypatch, tmp_path):
    (tmp_path / "config.yml").write_text("input_file: text\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    seen = {}

    class FakePelican:
        def __init__(self, path, **kwargs):
            seen["path"] = path
            seen["verbose"] = kwargs.get("verbose")

        def run(self):
            seen["ran"] = True

    monkeypatch.setattr("pelican_nlp.main.Pelican", FakePelican)
    main(["--verbose"])
    assert seen.get("ran") is True
    assert seen.get("verbose") is True


def test_pipeline_skipped_when_run_tests(monkeypatch, tmp_path):
    (tmp_path / "config.yml").write_text("input_file: text\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    class FakePelican:
        def __init__(self, path, **kwargs):
            raise AssertionError("pipeline should not run")

        def run(self):
            raise AssertionError("pipeline should not run")

    monkeypatch.setattr("pelican_nlp.main.Pelican", FakePelican)
    monkeypatch.setattr("pelican_nlp.testing.cli.run_pytest_suite", lambda **kwargs: 0)
    with pytest.raises(SystemExit) as exited:
        main(["--run-tests"])
    assert exited.value.code == 0


def test_pelican_test_examples_is_alias(monkeypatch):
    seen = []

    def fake_main(argv=None):
        seen.append(list(argv))

    monkeypatch.setattr("pelican_nlp.cli.main", fake_main)
    examples_main([])
    assert seen[0][:2] == ["--run-tests", "--examples"]

    seen.clear()
    examples_main(["--examples", "fluency", "--update-goldens"])
    assert seen[0] == ["--run-tests", "--examples", "fluency", "--update-goldens"]


def test_checkout_paths_point_at_repo_tests_and_examples():
    root = pytest_root()
    assert (root / "pytest.ini").is_file()
    assert packaged_tests_root() == root / "tests"
    assert packaged_tests_root().is_dir()
    assert (examples_root() / "example_fluency").is_dir()
    assert import_root().joinpath("pelican_nlp").is_dir()


def test_run_pytest_suite_command(monkeypatch):
    seen = []

    def fake_main(command):
        seen.append(list(command))
        return 0

    monkeypatch.setattr("pytest.main", fake_main)
    from pelican_nlp.testing.cli import run_pytest_suite

    assert run_pytest_suite() == 0
    unit = seen[0]
    assert str(packaged_tests_root()) in unit
    assert "--run-examples" not in unit

    assert run_pytest_suite(examples="fluency") == 0
    goldens = seen[1]
    assert "--run-examples" in goldens
    assert "--examples=fluency" in goldens
    assert any(item.startswith("--example-root=") for item in goldens)
    assert "--example-logs" not in goldens

    assert run_pytest_suite(examples="fluency", example_logs=True) == 0
    verbose = seen[2]
    assert "--example-logs" in verbose


def test_bundled_paths_when_not_a_checkout(monkeypatch, tmp_path):
    from pelican_nlp.testing import paths

    bundled = tmp_path / "_bundled"
    (bundled / "tests").mkdir(parents=True)
    (bundled / "examples").mkdir()
    (bundled / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    monkeypatch.setattr(paths, "repo_root", lambda: None)
    monkeypatch.setattr(paths, "package_dir", lambda: tmp_path)
    assert paths.pytest_root() == bundled
    assert paths.tests_root() == bundled / "tests"
    assert paths.examples_root() == bundled / "examples"


def test_copy_bundled_assets_skips_config_only_examples(tmp_path):
    pytest.importorskip("_pelican_build")
    from _pelican_build import copy_bundled_assets

    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_dummy.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    for name in ("example_fluency", "example_general", "example_acoustic-features"):
        folder = tmp_path / "examples" / name
        folder.mkdir(parents=True)
        (folder / "config.yml").write_text("input_file: text\n", encoding="utf-8")

    dest = copy_bundled_assets(tmp_path)
    names = {path.name for path in (dest / "examples").iterdir()}
    assert names == {"example_fluency"}
    assert (dest / "pytest.ini").is_file()
    assert (dest / "tests" / "test_dummy.py").is_file()
