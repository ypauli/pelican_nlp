"""Compare a run's derivatives tree to a frozen golden tree."""

from __future__ import annotations

import csv
import math
from pathlib import Path

from pelican_nlp.utils.setup_functions import is_hidden_or_system_file

from .example_projects import ExampleSettings


def compare_derivative_trees(
    actual_dir: Path,
    golden_dir: Path,
    settings: ExampleSettings | None = None,
) -> list[str]:
    """Return human-readable mismatch messages. Empty means equal."""
    settings = settings or ExampleSettings()
    actual_dir = Path(actual_dir)
    golden_dir = Path(golden_dir)
    actual = _index_comparable(actual_dir, settings)
    golden = _index_comparable(golden_dir, settings)

    messages = []
    missing = sorted(set(golden) - set(actual))
    extra = sorted(set(actual) - set(golden))
    for relative in missing:
        messages.append(f"missing from run: {relative.as_posix()}")
    for relative in extra:
        messages.append(f"unexpected in run: {relative.as_posix()}")

    for relative in sorted(set(golden) & set(actual)):
        messages.extend(
            _compare_file(
                actual[relative],
                golden[relative],
                relative,
                settings,
            )
        )
    return messages


def _index_comparable(root: Path, settings: ExampleSettings) -> dict[Path, Path]:
    indexed = {}
    if not root.is_dir():
        return indexed
    for path in root.rglob("*"):
        if not path.is_file() or is_hidden_or_system_file(path.name):
            continue
        relative = path.relative_to(root)
        if settings.is_comparable(relative):
            indexed[relative] = path
    return indexed


def _compare_file(
    actual_path: Path,
    golden_path: Path,
    relative: Path,
    settings: ExampleSettings,
) -> list[str]:
    suffix = relative.suffix.lower()
    label = relative.as_posix()
    if suffix == ".csv":
        return _compare_csv(actual_path, golden_path, label, relative, settings)
    actual_text = actual_path.read_text(encoding="utf-8")
    golden_text = golden_path.read_text(encoding="utf-8")
    if actual_text == golden_text:
        return []
    return [f"{label}: text differs from golden"]


def _compare_csv(
    actual_path: Path,
    golden_path: Path,
    label: str,
    relative: Path,
    settings: ExampleSettings,
) -> list[str]:
    actual_rows = _read_csv_rows(actual_path)
    golden_rows = _read_csv_rows(golden_path)
    if len(actual_rows) != len(golden_rows):
        return [
            f"{label}: row count {len(actual_rows)} != golden {len(golden_rows)}"
        ]
    rtol, atol = settings.tolerances_for(relative)
    skip_names = {name.lower() for name in settings.skip_columns}
    skip_indexes = set()
    if golden_rows:
        skip_indexes = {
            index
            for index, name in enumerate(golden_rows[0])
            if name.strip().lower() in skip_names
        }
    messages = []
    for index, (actual_row, golden_row) in enumerate(zip(actual_rows, golden_rows)):
        if len(actual_row) != len(golden_row):
            messages.append(
                f"{label} row {index}: {len(actual_row)} columns != golden {len(golden_row)}"
            )
            continue
        for column, (actual_cell, golden_cell) in enumerate(zip(actual_row, golden_row)):
            if column in skip_indexes:
                continue
            if _cells_match(actual_cell, golden_cell, rtol=rtol, atol=atol):
                continue
            messages.append(
                f"{label} row {index} col {column}: {actual_cell!r} != golden {golden_cell!r}"
            )
            if len(messages) >= 20:
                messages.append(f"{label}: further mismatches omitted")
                return messages
    return messages


def _read_csv_rows(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [list(row) for row in csv.reader(handle)]


def _cells_match(actual: str, golden: str, *, rtol: float, atol: float) -> bool:
    if actual == golden:
        return True
    actual_number = _as_float(actual)
    golden_number = _as_float(golden)
    if actual_number is None or golden_number is None:
        return False
    if math.isnan(actual_number) and math.isnan(golden_number):
        return True
    if math.isnan(actual_number) or math.isnan(golden_number):
        return False
    tolerance = atol + rtol * abs(golden_number)
    return abs(actual_number - golden_number) <= tolerance


def _as_float(value: str):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
