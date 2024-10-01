#class subject
#==============

from Document import Document

class Subject:
    def __init__(self) -> None:
        
        #subject parameters...
        self.age = None
        self.gender = None

        #specify all files belonging to same subject
        self.files = []
        if filelist:
            for file in filelist:
                self.files.append(file)

        self.document1 = Document(self.files[0])
        #etc...
