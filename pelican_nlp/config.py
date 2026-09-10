"""
Global configuration settings for the Pelican project.

This file is not the configuration.yml file created for the users adaptations.
For consistency of pipeline, DO NOT CHANGE.
"""

import os

# Debug flag. Default quiet; enable with --verbose or PELICAN_DEBUG=1.
DEBUG_MODE = False

_TRUE = {"1", "true", "yes", "on"}


def debug_enabled() -> bool:
    if DEBUG_MODE:
        return True
    for name in ("PELICAN_DEBUG", "PELICAN_VERBOSE"):
        if os.environ.get(name, "").strip().lower() in _TRUE:
            return True
    return False


def debug_print(*args, **kwargs):
    """Print only if debug mode is enabled."""
    if debug_enabled():
        print(*args, **kwargs)
