import os

import pytest


def pytest_load_initial_conftests(early_config, parser, args):
    """Keep example progress lines live.

    Default pytest capture would hide the summary until a test finishes, so a
    long Llama load looks frozen. Full pipeline logs stay off unless
    ``--example-logs`` is passed.
    """
    if "--run-examples" not in args:
        return
    capture_already_set = any(
        item == "-s" or item.startswith("--capture") for item in args
    )
    if not capture_already_set:
        args.append("-s")


def pytest_addoption(parser):
    parser.addoption(
        "--run-examples",
        action="store_true",
        default=False,
        help="Run end-to-end example golden tests (slow; needs models).",
    )
    parser.addoption(
        "--examples",
        action="store",
        default="all",
        help="Example selection: 'all' or comma-separated names.",
    )
    parser.addoption(
        "--example-root",
        action="store",
        default=None,
        help="Directory of example projects (default: packaged examples).",
    )
    parser.addoption(
        "--update-goldens",
        action="store_true",
        default=False,
        help="Overwrite example derivatives/ with the current run.",
    )
    parser.addoption(
        "--example-logs",
        action="store_true",
        default=False,
        help="Print full example pipeline logs instead of a progress summary.",
    )


def pytest_configure(config):
    if not config.getoption("--example-root"):
        from pelican_nlp.testing.paths import examples_root

        config.option.example_root = str(examples_root())
    if config.getoption("--example-logs"):
        os.environ["PELICAN_EXAMPLE_LOGS"] = "1"


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-examples"):
        items[:] = [item for item in items if "example" not in item.keywords]
        return
    example_items = [item for item in items if "example" in item.keywords]
    total = len(example_items)
    for index, item in enumerate(example_items, start=1):
        item._pelican_example_progress = (index, total)


def pytest_runtest_setup(item):
    progress = getattr(item, "_pelican_example_progress", None)
    if not progress:
        return
    index, total = progress
    label = item.callspec.id if hasattr(item, "callspec") else item.name
    print(f"\n[{index}/{total}] {label}", flush=True)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    progress = getattr(item, "_pelican_example_progress", None)
    if not progress or report.when != "call":
        return
    label = item.callspec.id if hasattr(item, "callspec") else item.name
    reason = ""
    if report.skipped:
        reason = _skip_reason(report)
        print(f"  {label}  SKIPPED ({reason})", flush=True)
        status = "skipped"
    elif report.failed:
        status = "failed"
    elif report.passed:
        status = "passed"
    else:
        return
    results = getattr(item.config, "_pelican_example_results", None)
    if results is None:
        results = []
        item.config._pelican_example_results = results
    results.append((label, status, reason))


def pytest_sessionstart(session):
    if not session.config.getoption("--run-examples"):
        return
    session.config._pelican_example_results = []
    if session.config.getoption("--example-logs"):
        return
    print(
        "Example goldens: progress summary only "
        "(pass --example-logs for full pipeline output).",
        flush=True,
    )


def pytest_sessionfinish(session, exitstatus):
    if not session.config.getoption("--run-examples"):
        return
    results = list(getattr(session.config, "_pelican_example_results", []) or [])
    if not results:
        return
    counts = {"passed": 0, "failed": 0, "skipped": 0}
    for _name, status, _reason in results:
        counts[status] = counts.get(status, 0) + 1
    print("\nExample golden summary", flush=True)
    print(
        f"  {counts['passed']} passed, {counts['failed']} failed, {counts['skipped']} skipped",
        flush=True,
    )
    for name, status, reason in results:
        extra = f" ({reason})" if reason else ""
        print(f"  {status.upper():<8} {name}{extra}", flush=True)


def _skip_reason(report) -> str:
    longrepr = getattr(report, "longrepr", None)
    if isinstance(longrepr, tuple) and len(longrepr) >= 3:
        text = str(longrepr[2])
        prefix = "Skipped: "
        if text.startswith(prefix):
            return text[len(prefix):]
        return text
    return "skipped"
