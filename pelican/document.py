import re
from pelican.preprocessing import TextImporter

import os
import re
from collections import defaultdict, OrderedDict


#combined class with speaker diarization...
class Document:
    """Represents a document with multiple lines of text, metadata, and processing capabilities."""

    def __init__(
        self,
        file_path,
        name,
        fluency=None,
        subject_ID=None,
        task=None,
        num_speakers=None,
        has_sections=False,
        section_identifier=None,
        number_of_sections=None,
        lines=None,
    ):
        self.file_path = file_path
        self.name = name
        self.results_path = None
        self.results_file = None
        self.file = os.path.join(file_path, name)
        self.extension = None
        self.corpus_name = self.extract_corpus_name()
        self.subject_ID = subject_ID
        self.task = task
        self.num_speakers = num_speakers
        self.has_sections = has_sections
        self.section_identifier = section_identifier
        self.number_of_sections = number_of_sections
        self.lines = lines if lines else []
        self.has_segments = task == "discourse"
        self.segments = [] if self.has_segments else ["default"] * len(self.lines)
        self.session = None

        self.sections = {}
        self.section_metrics = {}
        self.length_in_lines = len(self.lines)
        self.length_in_words = sum(line.length_in_words for line in self.lines)

        self.importer = TextImporter(self.file_path)
        self.raw_text = self.importer.load_text(self.file)

        self.fluency = fluency
        self.number_of_duplicates = None
        self.number_of_hyphenated_words=None

        # Processing attributes
        self.cleaned_sections = {}
        self.tokens_logits = []
        self.tokens_embeddings = []
        self.normalized_tokens = None
        self.processed_text = None
        self.logits = []
        self.embeddings = []
        self.acoustic_features = None

    def __repr__(self):
        return f"file_name={self.name}"

    def extract_corpus_name(self):
        parts = self.name.split('_')
        if len(parts) >= 3:  # Ensure there are enough parts
            return parts[3].split('.')[0]  # Corpus is at index 3 (0-based), return corpus without possible filetype
        else:
            raise ValueError("Filename format is incorrect. Expected at least 5 parts.")

    def create_results_path(self, project_path):
        """Generates a results CSV for a given document."""
        relative_path = os.path.relpath(self.file_path, project_path) + '/'
        first_slash_index = relative_path.find('/')
        modified_relative_path = relative_path[first_slash_index + 1:]
        output_path = os.path.join(project_path, 'Outputs', modified_relative_path)
        filename_no_extension = os.path.splitext(self.name)[0]
        self.results_path = os.path.join(output_path, f"{filename_no_extension}_results.csv")
        print(self.results_path)

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(self.results_path), exist_ok=True)

        # Create the CSV file
        with open(self.results_path, 'w') as file:
            pass

    def add_line(self, line):
        self.lines.append(line)
        self.length_in_lines = len(self.lines)
        self.length_in_words += line.length_in_words
        if not self.has_segments:
            self.segments.append("default")

    def compile_texts_and_tags(self):
        self.words, self.word_tags, self.word_segments = [], [], []
        for line, segment in zip(self.lines, self.segments):
            line_words = line.text.split()
            tag = "i" if line.speaker.lower() == "investigator" else "s"

            self.word_segments.extend([segment] * len(line_words))
            self.words.extend(line_words)
            self.word_tags.extend([tag] * len(line_words))

    def segment_task(self, protocol, cutoff=1):
        if not self.has_segments:
            return self.segments

        patterns = {
            section: re.compile("|".join(f"(?:\\b{re.escape(term)}\\b)" for term in terms), re.IGNORECASE)
            for section, terms in protocol.items()
        }

        match_scores = defaultdict(list)
        for section, pattern in patterns.items():
            for line_index, line in enumerate(self.lines):
                if pattern.search(line.text):
                    match_scores[section].append(line_index)

        section_order = sorted(protocol.keys(), key=lambda x: int(x))
        section_starts = OrderedDict()
        last_index_used = -1

        for section in section_order:
            line_indices = match_scores[section]
            valid_starts = [idx for idx in line_indices if idx > last_index_used and len(line_indices) >= cutoff]
            if valid_starts:
                start_line = min(valid_starts)
                section_starts[section] = start_line
                last_index_used = start_line

        segment_names = ["1"] * len(self.lines)
        current_section = None
        for i in range(len(self.lines)):
            if i in section_starts.values():
                current_section = [sec for sec, start in section_starts.items() if start == i][0]
            segment_names[i] = current_section if current_section else "default"

        self.segments = segment_names
        self.sections = self._create_sections(segment_names)
        return segment_names

    def _create_sections(self, segment_names):
        sections = defaultdict(list)
        for line, segment in zip(self.lines, segment_names):
            sections[segment].append(line)
        return sections

    def detect_sections(self):

        if not self.raw_text:
            raise ValueError("Raw text must be loaded before detecting sections.")

        lines = self.raw_text.splitlines()
        if not self.has_sections:
            title, content = (lines[0].strip(), "\n".join(lines[1:]).strip()) if lines else ("untitled section", "")
            self.sections = {title: content}
            return

        sections = {}
        current_title, current_content = None, []
        section_titles = []

        for line in lines:
            if line.startswith(self.section_identifier):
                if current_title:
                    sections[current_title] = "\n".join(current_content).strip()

                current_title = line.strip()
                section_titles.append(current_title)
                current_content = []
            else:
                if current_title:
                    current_content.append(line)

        if current_title:
            sections[current_title] = "\n".join(current_content).strip()

        self.sections = sections

        if self.number_of_sections is not None and len(self.sections) != self.number_of_sections:
            raise ValueError("Incorrect number of sections detected.")

    def process_document(self, pipeline):
        print(f"Processing document: {self.name}")
        pipeline.process_document(self)

    def clean_text(self, cleaner):
        if not self.sections:
            raise ValueError("Text must be divided into sections before cleaning.")

        self.cleaned_sections = self.sections.copy()
        for title, content in self.sections.items():
            self.cleaned_sections[title] = (
                cleaner.clean(self, content)
            )

    def tokenize_text(self, tokenizer, purpose):
        if not self.cleaned_sections:
            raise ValueError("Text must be cleaned before tokenizing.")

        for _, content in self.cleaned_sections.items():
            tokens = tokenizer.tokenize(content)
            if purpose == "logits":
                self.tokens_logits.append(tokens)
            elif purpose == "embeddings":
                self.tokens_embeddings.append(tokens)

    def normalize_text(self, normalizer):
        if not self.tokens_logits:
            raise ValueError("Text must be tokenized before normalization.")

        self.normalized_tokens = normalizer.normalize(self.tokens_logits)

    def get_processed_text(self):
        return " ".join(self.normalized_tokens) if self.normalized_tokens else self.tokens_logits

    def get_document_metadata(self):
        return {
            "file_path": self.file_path,
            "task": self.task,
            "num_speakers": self.num_speakers,
            "has_sections": self.has_sections,
        }