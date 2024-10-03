#class document
#=================

class Document:
    def __init__(self) -> None:

        #should layout of document be dictated?
        #how much variation acceptable?

        #document parameters
        self.task = None
        self.numberOfSpeakers = 1 #specify number of speakers in transcript, default: 1
        self.sections = None
        #etc...

    def tokenization(self):
        return
    def lowercase(self):
        return