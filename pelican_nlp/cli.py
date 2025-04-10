import os
from pelican_nlp.main import Pelican

def main():
    config_files = [f for f in os.listdir(".") if f.endswith(".yml")]
    if not config_files:
        print("No .yml configuration file found in the current directory.")
        return

    if len(config_files) > 1:
        print("More than one configuration file found - remove unneeded files from project directory")
        return

    config_file = config_files[0]  # You could also add logic to choose or validate
    print(f"Using configuration file: {config_file}")

    pelican = Pelican(config_file)
    pelican.run()
