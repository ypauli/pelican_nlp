#configuration file, contains variable parameters
#==================================================

class Config:
    def __init__(self):
        self.PATH_FOLDER_TEXTFILES = 'sample_path' #set default to home directory
        self.language = 'german' #possibly add options for german and english
        self.task1 = 'sampleTask'  # give name of task used for creation of text file (e.g. 'fluency')
        self.task2 = None  # optional: possible second task used for each subject, default: None
        self.cleaning_options = {
            'remove_punctuation': True,
            'lowercase': True,
            'strip_whitespace': True
        }
        self.tokenization_options = {
            'method': 'whitespace' #regex, nltk, etc.
        }
        self.normalization_options = {
            'method': 'lemmatization' #or stemming
        }