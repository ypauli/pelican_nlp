"""
This module provides the Subject class, each instance representing one subject.
The Subject class stores all subject specific information and a list of corresponding documents.
"""

class Subject:
    def __init__(self, subjectID, description=None):

        self.subjectID = subjectID
        self.gender = None
        self.age = None
        self.name = None
        self.description = description  # Description of the subject
        self.documents = []  # List of TextDocument instances
        self.numberOfSessions = None

    def __repr__(self):
        return f"Subject(ID={self.subjectID})"

    def add_document(self, document):
        self.documents.append(document)
        document.subject = self

    def process_subject(self, importer, cleaner, tokenizer, normalizer):
        print(f'Subject {self.subjectID} is being processed')
        for document in self.documents:
            continue

    def get_subject_info(self):
        return f"Subject: {self.name}\nDescription: {self.description}\nNumber of files: {len(self.documents)}"
