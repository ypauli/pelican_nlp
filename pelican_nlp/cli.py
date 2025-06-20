import os
from pathlib import Path
from pelican_nlp.main import Pelican
from pelican_nlp.config import RUN_TESTS, run_tests

def main():
    # Run tests if enabled
    if RUN_TESTS:
        print("Running tests...")
        run_tests()
        return

    # Get the package directory's sample_configuration_files folder
    package_dir = Path(__file__).parent
    config_dir = package_dir / 'sample_configuration_files'
    
    if not config_dir.exists():
        print("sample_configuration_files directory not found in package directory.")
        return
        
    config_files = [f for f in os.listdir(config_dir) if f.endswith(".yml")]
    if not config_files:
        print("No .yml configuration file found in the sample_configuration_files directory.")
        return

    if len(config_files) > 1:
        print("More than one configuration file found in sample_configuration_files directory - please specify which one to use")
        return

    config_file = str(config_dir / config_files[0])
    print(f"Using configuration file: {config_file}")

    pelican = Pelican(config_file)
    pelican.run()
