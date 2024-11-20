class Subject:
    def __init__(self, subjectID, description=None):

        self.subjectID = subjectID
        self.gender = None
        self.age = None
        self.name = None
        self.description = description  # Description of the subject
        self.documents = []  # List of TextDocument instances

    def __repr__(self):
        return f"Subject(ID={self.subjectID}, num_documents={len(self.documents)})"

    def add_document(self, document):
        self.documents.append(document)
        document.subject = self

    def process_subject(self, importer, cleaner, tokenizer, normalizer):
        """Processes all files under this subject."""
        for document in self.documents:
            document.load_text(importer)
            document.clean_text(cleaner, is_dialog=(
                        document.num_speakers > 1))  # Automatically set as dialog if there are speakers
            document.tokenize_text(tokenizer)
            document.normalize_text(normalizer)

    def get_subject_info(self):
        """Returns a string with subject metadata."""
        return f"Subject: {self.name}\nDescription: {self.description}\nNumber of files: {len(self.documents)}"
