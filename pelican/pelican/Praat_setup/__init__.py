import os

# Get the directory where the Praat scripts are stored
PRAAT_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths to individual scripts
PROSOMAIN_SCRIPT = os.path.join(PRAAT_SCRIPTS_DIR, 'prosomain.praat')
PROSOGRAM_SCRIPT = os.path.join(PRAAT_SCRIPTS_DIR, 'prosogram.praat')
PROSOPLOT_SCRIPT = os.path.join(PRAAT_SCRIPTS_DIR, 'prosoplot.praat')
# ... other scripts as needed 