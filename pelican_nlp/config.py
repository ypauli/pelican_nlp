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

def run_tests():
    """Run all example tests if RUN_TESTS is enabled."""
    if RUN_TESTS:
        import unittest
        from pathlib import Path

        # Get the path to the test file
        test_file = Path(__file__).parent / "utils" / "unittests" / "test_examples.py"

        # Create a test suite and add the test file
        loader = unittest.TestLoader()
        suite = loader.discover(str(test_file.parent), pattern="test_examples.py")

        # Run the tests
        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(suite)