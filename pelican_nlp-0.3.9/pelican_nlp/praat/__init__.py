import os

# Get the directory where the Praat scripts are stored
PRAAT_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths to individual scripts
PROSOMAIN_SCRIPT = os.path.join(PRAAT_SCRIPTS_DIR, 'prosomain.praat')
PROSOGRAM_SCRIPT = os.path.join(PRAAT_SCRIPTS_DIR, 'prosogram.praat')
PROSOPLOT_SCRIPT = os.path.join(PRAAT_SCRIPTS_DIR, 'prosoplot.praat')
SEGMENT_SCRIPT = os.path.join(PRAAT_SCRIPTS_DIR, 'segment.praat')
STYLIZE_SCRIPT = os.path.join(PRAAT_SCRIPTS_DIR, 'stylize.praat')
POLYTONIA_SCRIPT = os.path.join(PRAAT_SCRIPTS_DIR, 'polytonia.praat')
UTIL_SCRIPT = os.path.join(PRAAT_SCRIPTS_DIR, 'util.praat')
EPS_CONV_SCRIPT = os.path.join(PRAAT_SCRIPTS_DIR, 'eps_conv.praat')
SETUP_SCRIPT = os.path.join(PRAAT_SCRIPTS_DIR, 'setup.praat')

# Export all script paths
__all__ = [
    'PRAAT_SCRIPTS_DIR',
    'PROSOMAIN_SCRIPT',
    'PROSOGRAM_SCRIPT',
    'PROSOPLOT_SCRIPT',
    'SEGMENT_SCRIPT',
    'STYLIZE_SCRIPT',
    'POLYTONIA_SCRIPT',
    'UTIL_SCRIPT',
    'EPS_CONV_SCRIPT',
    'SETUP_SCRIPT'
]