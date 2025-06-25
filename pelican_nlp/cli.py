import os
from pathlib import Path
from pelican_nlp.main import Pelican
from pelican_nlp.config import RUN_TESTS

def main():
    # Run tests if enabled
    if RUN_TESTS:
        print("Running tests...")
        Pelican(test_mode=True).run_tests()
        return

    # Look for configuration files in the current working directory
    config_dir = Path.cwd()
    
    print(f"Looking for configuration files in: {config_dir}")
    
    config_files = [f for f in os.listdir(config_dir) if f.endswith((".yml", ".yaml"))]
    
    if not config_files:
        print("No .yml or .yaml configuration file found in the current directory.")
        print("Please ensure you have a configuration file in your current working directory.")
        return

    if len(config_files) > 1:
        print("Warning: Multiple configuration files found in current directory:")
        for i, file in enumerate(config_files, 1):
            print(f"  {i}. {file}")
        print("Please ensure only one configuration file is present in the current directory.")
        return

    config_file = str(config_dir / config_files[0])
    print(f"Using configuration file: {config_file}")

    pelican = Pelican(config_file)
    pelican.run()
