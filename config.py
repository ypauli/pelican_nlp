#file for all definable parameters
#=================================

PATH_FOLDER_TEXTFILES = 'sample_path' #set default to home directory
language = 'german'     #possibly add options for german and english
task1 = 'sampleTask'    #give name of task used for creation of text file (e.g. 'fluency')
task2 = None            #optional: possible second task used for each subject, default: None

desired_modifications = {
    'removePunctuation': False,
    'lowercase': False,
    'removeSpecialCharacters': False
    #etc...
}
