import os
from pathlib import Path
from pelican_nlp.main import Pelican
from pelican_nlp.config import RUN_TESTS
from pelican_nlp.utils.setup_functions import is_hidden_or_system_file

def main():
    # Run tests if enabled
    if RUN_TESTS:
        print("Running tests...")
        Pelican(test_mode=True).run_tests()
        return

    # Look for configuration files in the current working directory
    config_dir = Path.cwd()
    config_files = [f for f in os.listdir(config_dir) if f.endswith((".yml", ".yaml")) and not is_hidden_or_system_file(f)]
    
    if not config_files:
        print("Error: No .yml or .yaml configuration file found in the current directory.")
        return

    if len(config_files) > 1:
        print("Error: Multiple configuration files found. Please ensure only one configuration file is present.")
        return

    config_file = str(config_dir / config_files[0])
    
    try:
        pelican = Pelican(config_file)
        pelican.run()
        print("Pipeline ran successfully")
    except Exception as e:
        print(f"Error: {e}")
        return
