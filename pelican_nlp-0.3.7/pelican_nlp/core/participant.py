"""
This module provides the Participant class, each instance representing one participant.
The Participant class stores all participant specific information and a list of corresponding documents.
"""

class Participant:
    def __init__(self, name, description=None):

        self.name = name
        self.participantID = None
        self.gender = None
        self.age = None
        self.description = description  # Description of the participant
        self.documents = []  # List of TextDocument instances
        self.numberOfSessions = None

    def __repr__(self):
        return f"Participant(ID={self.participantID})"

    def add_document(self, document):
        self.documents.append(document)
        document.participant = self

    def process_participant(self, importer, cleaner, tokenizer, normalizer):
        print(f'Participant {self.participantID} is being processed')
        for document in self.documents:
            continue

    def get_participant_info(self):
        return f"Participant: {self.name}\nDescription: {self.description}\nNumber of files: {len(self.documents)}"
