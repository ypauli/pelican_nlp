#configuration file, contains variable parameters
#==================================================
import torch
import psutil

class Config:
    def __init__(self):
        self.input_file = 'text' #possibly add option of audio file
        self.PATH_TO_PROJECT_FOLDER = '/home/yvespauli/PycharmProjects/KetamineStudy/KetamineStudy_ProjectFolder/' #set default to home directory, e.g. '/home/usr/...'
        self.language = 'german' #possibly add options for german and english
        self.numberOfSubjects = None #specify number of subjects, if 'None' number of subjects automatically detected
        self.multipleSessions = False #set to True if multiple sessions per subject
        self.task_names = ['sampleTask']  #give name of task(s) used for creation of text file (e.g. ['fluency','interview'])
        self.corpus_names = ['Ketamin','Placebo'] #names of individual categories that should be grouped together
        self.tokenization = 'wordLevel' #options include 'characterLevel', 'subWordLevel'

        self.gpu_available = True if torch.cuda.is_available()==True else False
        self.gpu_version = torch.cuda.get_device_name(0) if self.gpu_available==True else None
        self.VRAM = torch.cuda.get_device_properties(0).total_memory/(1024 ** 3) if self.gpu_available==True else None #[gb]
        self.VRAM_str = str(int(self.VRAM)-1)+'GB'
        self.RAM = psutil.virtual_memory().available/(1024 ** 3) #available RAM in GB
        self.RAM_str = str(int(self.RAM)-1)+'GB'

        self.PATH_TO_SUBJECTS = self.PATH_TO_PROJECT_FOLDER + 'Subjects'

        #options for extract_logits
        self.chunk_size = None
        self.overlap_size = None

        self.cleaning_options = {
            'general_cleaning': True, #general cleaning options used for most text preprocessing, default: True. To set individual options change individual values in 'general_cleaning_options' and set value to False.
            'remove_punctuation': True,
            'lowercase': True,
            'remove_brackets_and_bracketcontent': True
        }
        self.general_cleaning_options = {
            'strip_whitespace': True,
            'merge_multiple_whitespaces': True,
            'remove_whitespace_before_punctuation': True,
            'merge_newline_characters': True,
            'remove_backslashes': True
        }
        self.tokenization_options = {
            'method': 'model', #model, regex, nltk, etc.
            'model_name': 'DiscoResearch/Llama3-German-8B-32k' # Replace with your model
        }
        # names: 'mayflowergmbh/Llama3-German-8B-32k-GPTQ' 'DiscoResearch/Llama3-German-8B-32k' 'QuantFactory/Llama3-German-8B-GGUF'
        self.normalization_options = {
            'method': 'lemmatization' #or stemming
        }