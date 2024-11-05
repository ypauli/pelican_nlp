#configuration file, contains variable parameters
#==================================================

class Config:
    def __init__(self):
        self.PATH_TO_SUBJECTS_FOLDER = '/home/yvespauli/PycharmProjects/KetamineStudy/KetamineStudyData_Restructured/' #set default to home directory, e.g. '/home/usr/...'
        self.corpus_names = ['Ketamin','Placebo'] #names of individual categories that should be grouped together
        self.language = 'german' #possibly add options for german and english
        self.numberOfSubjects = None #specify number of subjects, if 'None' number of subjects automatically detected
        self.task_names = ['sampleTask']  #give name of task(s) used for creation of text file (e.g. ['fluency','interview'])
        self.numberOfGroups = 2 #Number of groups that need to be analyzed individually, e.g. two groups (ketamine/placebo)
        self.cleaning_options = {
            'general_cleaning': True, #general cleaning options used for most text preprocessing, default: True. To set individual options change individual values in "..."
            'remove_punctuation': True,
            'lowercase': True,
            'remove_brackets_and_bracketcontent': True
        }
        self.tokenization_options = {
            'method': 'whitespace' #regex, nltk, etc.
        }
        self.normalization_options = {
            'method': 'lemmatization' #or stemming
        }