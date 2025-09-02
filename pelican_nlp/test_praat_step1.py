#!/usr/bin/env python3
"""
Step 1: Test if we can run Praat from command line
"""
import subprocess
import os

def test_praat_installation():
    """Test if Praat is installed and accessible"""
    try:
        result = subprocess.run(["praat", "--version"], capture_output=True, text=True)
        print(f"Praat version: {result.stdout.strip()}")
        print(f"Return code: {result.returncode}")
        return result.returncode == 0
    except FileNotFoundError:
        print("Praat not found in PATH")
        return False

if __name__ == "__main__":
    print("=== Step 1: Testing Praat Installation ===")
    success = test_praat_installation()
    print(f"Praat test: {'PASSED' if success else 'FAILED'}")

