#class subject
#==============

from Document import Document

class Subject:
    def __init__(self) -> None:
        
        #subject parameters...
        self.age = None
        self.gender = None

        self.healthy = True #set to false if disease subject...

        self.document1 = Document(self.files[0])
        self.document2 = None #optional, in case subject has more than one textfile
        #etc...
