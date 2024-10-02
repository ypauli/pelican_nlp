#class document
#=================

class Document:
    def __init__(self) -> None:
        
        #document parameters
        self.task = None
        self.numberOfSpeakers = 1 #specify number of speakers in transcript, default: 1
        self.sections = None

        #etc...