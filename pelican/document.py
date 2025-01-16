import re
from pelican.preprocessing import TextImporter

class Document:
    def __init__(self, file_path, name, corpus_name, task=None, num_speakers=None, has_sections=False, section_identifier=None, number_of_sections=None):

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
        self.section_identifier = section_identifier
        self.number_of_sections = number_of_sections

        self.importer = TextImporter(self.file_path)
        self.raw_text = self.importer.load_text(self.file)

        self.sections = []
        self.processed_sections = []
        self.cleaned_sections = {}
        self.tokens_logits = []
        self.tokens_embeddings = []
        self.normalized_tokens = None
        self.processed_text = None
        self.logits = []
        self.embeddings = []
        self.acoustic_features = None


    def __repr__(self):
        return f"TextDocument(file={self.file}, task={self.task}, speakers={self.num_speakers}, has_sections={self.has_sections}, cleaned_text={self.cleaned_text})"

    def process_document(self, pipeline):
        print(f'=====processing document {self.name}======')
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
        print('cleaning text...')
        if self.sections is None:
            raise ValueError("Text must be loaded and divided into sections before cleaning.")

        print(f'amount of sections to clean: {len(self.sections)}')
        print(f'sections to clean: {self.sections}')
        self.cleaned_sections = self.sections.copy()
        for title, content in self.sections.items():
            if is_dialog:
                self.cleaned_sections[title] = self.extract_spoken_text(content)
            else:
                self.cleaned_sections[title] = cleaner.clean(content)
        print(f'cleaned sections: {self.cleaned_sections}')

    def tokenize_text(self, tokenizer, purpose):
        print('tokenizing text...')
        if self.cleaned_sections is {}:
            raise ValueError("Text must be cleaned before tokenizing.")
        for title,content in self.cleaned_sections.items():
            if purpose == 'logits':
                self.tokens_logits.append(tokenizer.tokenize(content))
            elif purpose == 'embeddings':
                self.tokens_embeddings.append(tokenizer.tokenize(content))

    def normalize_text(self, normalizer):
        print('normalizing text...')
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

        print('detecting sections (if any)...')
        lines = self.raw_text.splitlines()

        if not self.has_sections:
            if lines:
                title = lines[0].strip()
                content = "\n".join(lines[1:]).strip()
            else:
                title = 'untitled section'
                content = ''
            self.sections = {title: content}
            return

        sections = {}
        current_section_title = None
        current_section_content = []
        section_titles = []

        for line in lines:
            if line.startswith(self.section_identifier):  # Check if the line starts with the pattern
                if current_section_title is not None:
                    # Save the previous section before starting a new one
                    sections[current_section_title] = "\n".join(current_section_content).strip()

                # Set the new section title and reset content
                current_section_title = line.strip()
                section_titles.append(current_section_title)
                current_section_content = []
            else:
                if current_section_title is not None:
                    # Add the line to the current section's content
                    current_section_content.append(line)

        # Save the last section
        if current_section_title is not None:
            sections[current_section_title] = "\n".join(current_section_content).strip()

        self.sections = sections
        print(type(self.sections))

        #Check if correct number of sections
        if self.number_of_sections is not None:
            if len(self.sections) != self.number_of_sections:
                print(f"Number of sections detected: {len(self.sections)}, expected: {self.number_of_sections}")
                print(f"Section identifier: {self.section_identifier}")
                print(f"Section titles: {section_titles}")
                raise ValueError("Incorrect number of sections detected.")

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
