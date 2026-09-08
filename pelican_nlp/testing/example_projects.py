"""Discover Pelican example projects under a root directory."""

from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path

import yaml

from pelican_nlp.utils.lpds_paths import is_data_file
from pelican_nlp.utils.setup_functions import is_hidden_or_system_file

SETTINGS_FILENAME = ".pelican-test.yml"
NO_GOLDENS_SKIP = "no comparable golden files under derivatives/"
# Config-only folders: no participants/input data, so golden tests omit them.
CONFIG_ONLY_EXAMPLES = frozenset({"example_acoustic-features", "example_general"})
DEFAULT_COMPARE_SUFFIXES = (".csv", ".txt")
DEFAULT_FLOAT_RTOL = 1e-5
DEFAULT_FLOAT_ATOL = 1e-6
# Logits and perplexity floats move with GPU dtype; keep structure checks tight
# and numeric checks a little loose so a second run of the same example passes.
DEFAULT_FLOAT_OVERRIDES: tuple[tuple[str, float, float], ...] = (
    ("logits/**", 0.02, 0.05),
    ("perplexity-*/**", 0.02, 0.02),
    ("embeddings/**", 1e-3, 1e-3),
    ("semantic-similarity-*/**", 1e-3, 1e-3),
)
# Argmax token identity is unstable under fp16; numeric logit columns still compare.
DEFAULT_SKIP_COLUMNS: tuple[str, ...] = ("most_likely_token",)


@dataclass(frozen=True)
class ExampleSettings:
    """Comparison rules from ``golden_test:`` in the example YAML (or a sidecar)."""

    enabled: bool = True
    compare_suffixes: tuple[str, ...] = DEFAULT_COMPARE_SUFFIXES
    skip_globs: tuple[str, ...] = ()
    float_rtol: float = DEFAULT_FLOAT_RTOL
    float_atol: float = DEFAULT_FLOAT_ATOL
    float_overrides: tuple[tuple[str, float, float], ...] = DEFAULT_FLOAT_OVERRIDES
    skip_columns: tuple[str, ...] = DEFAULT_SKIP_COLUMNS

    def is_comparable(self, relative: Path) -> bool:
        if relative.suffix.lower() not in self.compare_suffixes:
            return False
        posix = Path(relative).as_posix()
        return not any(_glob_match(posix, pattern) for pattern in self.skip_globs)

    def tolerances_for(self, relative: Path) -> tuple[float, float]:
        posix = Path(relative).as_posix()
        for pattern, rtol, atol in self.float_overrides:
            if _glob_match(posix, pattern):
                return rtol, atol
        return self.float_rtol, self.float_atol


def _glob_match(posix: str, pattern: str) -> bool:
    """Match ``posix`` against a glob. ``dir/**`` skips that directory tree."""
    pattern = pattern.replace("\\", "/").lstrip("./")
    posix = posix.lstrip("./")
    if fnmatch(posix, pattern):
        return True
    if pattern.endswith("/**"):
        prefix = pattern[:-3]
        if posix == prefix or fnmatch(posix, prefix):
            return True
        top = posix.split("/", 1)[0]
        return bool(top) and fnmatch(top, prefix)
    return False


@dataclass(frozen=True)
class ExampleProject:
    name: str
    path: Path
    config_path: Path | None
    settings: ExampleSettings = field(default_factory=ExampleSettings)
    input_files: tuple[Path, ...] = ()
    golden_files: tuple[Path, ...] = ()
    skip_reason: str | None = None

    @property
    def runnable(self) -> bool:
        return self.skip_reason is None


def load_example_settings(example_dir: Path) -> ExampleSettings:
    """Load comparison rules from the example YAML ``golden_test`` block.

    A ``.pelican-test.yml`` sidecar still works and overrides the YAML block.
    """
    merged: dict = {}
    configs = _config_files(example_dir)
    if len(configs) == 1:
        loaded = yaml.safe_load(configs[0].read_text(encoding="utf-8")) or {}
        block = loaded.get("golden_test") if isinstance(loaded, dict) else None
        if isinstance(block, dict):
            merged.update(block)
    sidecar = example_dir / SETTINGS_FILENAME
    if sidecar.is_file():
        extra = yaml.safe_load(sidecar.read_text(encoding="utf-8")) or {}
        if isinstance(extra, dict):
            merged.update(extra)
    return _settings_from_mapping(merged)


def _settings_from_mapping(loaded: dict) -> ExampleSettings:
    suffixes = loaded.get("compare_suffixes", DEFAULT_COMPARE_SUFFIXES)
    globs = loaded.get("skip_globs", ())
    overrides = []
    for item in loaded.get("float_overrides", ()) or ():
        if not isinstance(item, dict) or "glob" not in item:
            continue
        overrides.append(
            (
                str(item["glob"]),
                float(item.get("rtol", loaded.get("float_rtol", DEFAULT_FLOAT_RTOL))),
                float(item.get("atol", loaded.get("float_atol", DEFAULT_FLOAT_ATOL))),
            )
        )
    named = {item[0] for item in overrides}
    overrides.extend(
        item for item in DEFAULT_FLOAT_OVERRIDES if item[0] not in named
    )
    columns = [str(item) for item in loaded.get("skip_columns", ()) or ()]
    skip_columns = tuple(
        dict.fromkeys((*DEFAULT_SKIP_COLUMNS, *columns))
    )
    return ExampleSettings(
        enabled=bool(loaded.get("enabled", True)),
        compare_suffixes=tuple(str(item).lower() for item in suffixes),
        skip_globs=tuple(str(item) for item in globs),
        float_rtol=float(loaded.get("float_rtol", DEFAULT_FLOAT_RTOL)),
        float_atol=float(loaded.get("float_atol", DEFAULT_FLOAT_ATOL)),
        float_overrides=tuple(overrides),
        skip_columns=skip_columns,
    )


def discover_example_projects(root: Path) -> list[ExampleProject]:
    """Return one entry per subdirectory of ``root``, sorted by name."""
    root = Path(root)
    if not root.is_dir():
        return []
    projects = []
    for entry in sorted(root.iterdir(), key=lambda path: path.name.lower()):
        if not entry.is_dir() or is_hidden_or_system_file(entry.name):
            continue
        projects.append(inspect_example_project(entry))
    return projects


def inspect_example_project(example_dir: Path) -> ExampleProject:
    example_dir = Path(example_dir)
    settings = load_example_settings(example_dir)
    configs = _config_files(example_dir)
    participants = example_dir / "participants"
    input_files = tuple(_data_files(participants)) if participants.is_dir() else ()
    config_path = configs[0] if len(configs) == 1 else None
    golden_files = tuple(_golden_files(example_dir / "derivatives", settings))

    skip_reason = None
    if not settings.enabled:
        skip_reason = f"disabled in {SETTINGS_FILENAME}"
    elif len(configs) == 0:
        skip_reason = "no YAML config"
    elif len(configs) > 1:
        skip_reason = f"multiple YAML configs: {', '.join(path.name for path in configs)}"
    elif not participants.is_dir():
        skip_reason = "no participants/ directory"
    elif not input_files:
        skip_reason = "no input data files under participants/"
    elif not golden_files:
        skip_reason = NO_GOLDENS_SKIP

    return ExampleProject(
        name=example_dir.name,
        path=example_dir,
        config_path=config_path,
        settings=settings,
        input_files=input_files,
        golden_files=golden_files,
        skip_reason=skip_reason,
    )


def select_example_projects(
    projects: list[ExampleProject],
    selection: str | None = "all",
) -> list[ExampleProject]:
    """Filter discovered projects.

    ``selection`` is ``all`` (default), or a comma-separated list of folder
    names. The ``example_`` prefix is optional.
    """
    if selection is None or str(selection).strip() == "" or str(selection).strip().lower() == "all":
        return list(projects)
    wanted = [_normalize_example_name(item) for item in str(selection).split(",") if item.strip()]
    if not wanted:
        return list(projects)
    by_name = {_normalize_example_name(project.name): project for project in projects}
    missing = [name for name in wanted if name not in by_name]
    if missing:
        known = ", ".join(project.name for project in projects) or "(none)"
        raise ValueError(f"Unknown example(s): {', '.join(missing)}. Known: {known}")
    return [by_name[name] for name in wanted]


def _normalize_example_name(name: str) -> str:
    cleaned = str(name).strip().lower().rstrip("/")
    if cleaned.startswith("example_"):
        return cleaned
    return f"example_{cleaned}"


def _config_files(example_dir: Path) -> list[Path]:
    configs = [
        path
        for path in example_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".yml", ".yaml"}
        and not is_hidden_or_system_file(path.name)
    ]
    return sorted(configs, key=lambda path: path.name.lower())


def _data_files(participants_dir: Path) -> list[Path]:
    files = []
    for path in participants_dir.rglob("*"):
        if not path.is_file() or is_hidden_or_system_file(path.name):
            continue
        if is_data_file(path.name):
            files.append(path)
    return sorted(files)


def _golden_files(derivatives_dir: Path, settings: ExampleSettings) -> list[Path]:
    if not derivatives_dir.is_dir():
        return []
    files = []
    for path in derivatives_dir.rglob("*"):
        if not path.is_file() or is_hidden_or_system_file(path.name):
            continue
        if settings.is_comparable(path.relative_to(derivatives_dir)):
            files.append(path)
    return sorted(files)
