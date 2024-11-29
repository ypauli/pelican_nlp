import re
from preprocessing import TextImporter

class Document:
    def __init__(self, file_path, name, corpus_name, task=None, num_speakers=None, has_sections=False):

        self.audiofile = False
        self.name = name
        self.file_path = file_path
        self.file = self.file_path + '/' + self.name
        self.results_path = None
        self.subject = None #reference to subject instance
        self.corpus_name = corpus_name #reference to corpus instance
        self.task = task  # e.g., Interview, Fluency...
        self.num_speakers = num_speakers  # Number of speakers in the dialog
        self.has_sections = has_sections  # Boolean flag for sections

        self.importer = TextImporter(self.file_path)
        self.raw_text = self.importer.load_text(self.file)
        self.cleaned_text = None
        self.tokens_logits = None
        self.tokens_embeddings = None
        self.normalized_tokens = None
        self.processed_text = None
        self.logits = None
        self.embeddings = None
        self.acoustic_features = None
        self.sections = []
        self.processed_sections = []

    def __repr__(self):
        return f"TextDocument(file={self.file}, task={self.task}, speakers={self.num_speakers}, has_sections={self.has_sections}, cleaned_text={self.cleaned_text})"

    def process_document(self, pipeline):
        pipeline.process_document(self)

    def create_results_csv(self, project_path):
        import os
        relative_path = os.path.relpath(self.file_path, project_path) + '/'
        first_slash_index = relative_path.find('/')
        modified_relative_path = relative_path[first_slash_index+1:]
        output_path = os.path.join(project_path, 'Outputs', modified_relative_path)
        filename_no_extension = os.path.splitext(self.name)[0]
        self.results_path = output_path + filename_no_extension + '_results' + '.csv'
        with open(self.results_path, 'w') as file:
            pass

    def clean_text(self, cleaner, is_dialog=False):
        """Cleans the raw text. If it's a dialog, extract spoken lines."""
        if self.raw_text is None:
            raise ValueError("Text must be loaded before cleaning.")

        if is_dialog:
            self.cleaned_text = self.extract_spoken_text(self.raw_text)
        else:
            self.cleaned_text = cleaner.clean(self.raw_text)

    def tokenize_text(self, tokenizer, purpose):
        """Tokenizes the cleaned text."""
        if self.cleaned_text is None:
            raise ValueError("Text must be cleaned before tokenizing.")
        if purpose == 'logits':
            self.tokens_logits = tokenizer.tokenize(self.cleaned_text)
        elif purpose == 'embeddings':
            self.tokens_embeddings = tokenizer.tokenize(self.cleaned_text)

    def normalize_text(self, normalizer):
        """Normalizes the tokens_logits (e.g., stemming or lemmatization)."""
        if self.tokens_logits is None:
            raise ValueError("Text must be tokenized before normalization.")
        self.normalized_tokens = normalizer.normalize(self.tokens_logits)
        print(self.normalized_tokens)

    def get_processed_text(self):
        """Returns the fully processed text (normalized tokens_logits joined as string)."""
        if self.normalized_tokens is not None:
            return ' '.join(self.normalized_tokens)
        elif self.tokens_logits is not None:
            return self.tokens_logits
        else:
            print('no tokens_logits')

    def get_document_metadata(self):
        """Returns a summary of metadata for this document."""
        return {
            "file_path": self.file_path,
            "task": self.task,
            "num_speakers": self.num_speakers,
            "has_sections": self.has_sections
        }

    def detect_sections(self):
        """Optional method to detect sections within a document (if applicable)."""
        if self.raw_text:
            # Assuming sections are divided by specific markers like 'Section:', numbers, etc.
            self.has_sections = bool(re.findall(r'(Section\s\d+|Part\s\d+|Chapter\s\d+)', self.raw_text))

    def extract_spoken_text(self, raw_text):
        """
        Extracts spoken text from a dialog format.
        Assumes that the spoken text is either in quotes or follows a character name format.
        """
        # Extract text inside quotation marks (commonly used for dialog)
        spoken_text = re.findall(r'“(.*?)”|"(.*?)"', raw_text)

        # Extract lines that begin after a character name (e.g., "John: Hello world.")
        dialog_lines = re.findall(r'^[A-Za-z]+\s*:\s*(.+)', raw_text, re.MULTILINE)

        spoken_text_combined = ' '.join([dialog[0] or dialog[1] for dialog in spoken_text] + dialog_lines)
        return spoken_text_combined.strip()
