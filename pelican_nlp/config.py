"""
Global configuration settings for the Pelican project.

This file is not the configuration.yml file created for the users adaptations.
For consistency of pipeline, DO NOT CHANGE.
"""

# Global debug flag
DEBUG_MODE = False

def set_debug_mode(mode: bool) -> None:
    """Set the global debug mode."""
    global DEBUG_MODE
    DEBUG_MODE = mode

def is_debug_mode() -> bool:
    """Get the current debug mode state."""
    return DEBUG_MODE

def debug_print(*args, **kwargs):
    """Print only if debug mode is enabled."""
    if DEBUG_MODE:
        print(*args, **kwargs) 