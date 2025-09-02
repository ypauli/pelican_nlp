"""
Global configuration settings for the Pelican project.

This file is not the configuration.yml file created for the users adaptations.
For consistency of pipeline, DO NOT CHANGE.
"""

# Debug flag
DEBUG_MODE = True

# Test flag - set to True to run all example tests
RUN_TESTS = False

def debug_print(*args, **kwargs):
    """Print only if debug mode is enabled."""
    DEBUG_MODE = True
    if DEBUG_MODE:
        print(*args, **kwargs)