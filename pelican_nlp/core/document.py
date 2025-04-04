"""
This module provides the Document class, each instance representing one file within a corpus.
The Document class stores all document specific information.
"""

import os
import re
from pelican_nlp.preprocessing import TextImporter
from collections import defaultdict, OrderedDict

class Document:

    def __init__(self, file_path, name, **kwargs):
        """Initialize Document object.
        
        Args:
            file_path: Path to document file
            name: Document name
            **kwargs: Optional document attributes
        """
        self.file_path = file_path
        self.name = name
        self.file = os.path.join(file_path, name)
        
        # Initialize optional attributes
        self.subject_ID = kwargs.get('subject_ID')
        self.task = kwargs.get('task')
        self.num_speakers = kwargs.get('num_speakers')
        self.has_sections = kwargs.get('has_sections', False)
        self.has_section_titles = kwargs.get('has_section_titles')
        self.section_identifier = kwargs.get('section_identifier')
        self.number_of_sections = kwargs.get('number_of_sections')
        self.lines = kwargs.get('lines', [])
        self.new_parameter = kwargs.get('new_parameter')
        self.another_metric = kwargs.get('another_metric')
        
        # Derived attributes
        self.has_segments = self.task == "discourse"
        self.segments = [] if self.has_segments else ["default"] * len(self.lines)
        self.sections = None
        
        # Initialize processing attributes
        self._init_processing_attributes()
        self._init_document_metrics()

    def _init_processing_attributes(self):
        """Initialize attributes related to text processing."""
        self.extension = None
        self.session = None
        self.corpus_name = None
        self.sections = {}
        self.section_metrics = {}
        
        # Load raw text
        self.importer = TextImporter(self.file_path)
        self.raw_text = self.importer.load_text(self.file)
        
        # Text processing state
        self.fluency_word_count = None
        self.fluency_duplicate_count = None
        self.cleaned_sections = {}
        self.tokens_logits = []
        self.tokens_embeddings = []
        self.normalized_tokens = None
        self.processed_text = None
        self.logits = []
        self.embeddings = []
        self.acoustic_features = None

    def _init_document_metrics(self):
        """Initialize document metrics."""
        self.length_in_lines = len(self.lines)
        self.length_in_words = sum(line.length_in_words for line in self.lines)
        self.fluency = None
        self.number_of_duplicates = None
        self.number_of_hyphenated_words = None

    def __repr__(self):
        return f"file_name={self.name}"

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
        print(f'detecting sections...')
        if not self.raw_text:
            raise ValueError("Raw text must be loaded before detecting sections.")

        lines = self.raw_text.splitlines()
        if not self.has_sections:
            if self.has_section_titles and lines:
                title, content = (lines[0].strip(), "\n".join(lines[1:]).strip()) if lines else ("untitled section", "")
            else:
                title, content = "untitled section", "\n".join(lines).strip()
            self.sections = {title: content}
            print(self.sections)
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
            if self.fluency:
                self.cleaned_sections[title] = (
                    cleaner.clean_fluency_transcripts(self, content)
                )
            else:
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