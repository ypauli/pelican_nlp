#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 14:21:27 2024

@author: nilsl
"""

import re
from collections import defaultdict, OrderedDict
import os
import re
import chardet
from collections import defaultdict, OrderedDict
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from striprtf.striprtf import rtf_to_text
import numpy as np
from text_cleaner import TextCleaner

protocol = {
    "1": [
        "täglichen Leben",
        "ein paar Dinge",
        "ein paar Sachen",
        "über sich erzählen",
    ],
    "2": [
        "etwas Wichtiges",
        "Ihrer Kindheit",
        "letzte Woche",
        "in Ihrem Leben",
        "beliebigen Zeit",
        "zurückdenken",
    ],
    "3": [
        "Ihre Gesundheit",
        "Gesundheit sprechen",
        "psychische Krankheit",
        "seit Beginn",
        "wie Sie sich gefühlt haben",
    ],
    "4": [
        "erste Bild",
        "Bild Nummer",
        "zweite Bild",
        "dritte Bild",
        "auf dem Bild",
        "Bild sehen",
    ],
    "5": [
        "Geschichte zeigen",
        "der Reihe nach",
        "Bilder aus einer Geschichte",
        "Bilder weg",
        "ein paar Bilder",
        "Bilderreihe",
        "Geschichte aus Bilder",
        "Geschichte aus Bildern",
        "so viel Zeit",
    ],
    "6": [
        "wiederkehrende Träume",
        "solche Träume",
        "Träume",
        "wiederkehrenden Traum",
    ],
    "7": [
        "einseitige",
        "geschriebene Geschichte",
        "Geschichte geschrieben",
        "Blatt",
        "eine Minute",
        "eigenen Worten",
        "eigene Worte",
        "laut vorzulesen",
        "laut vorlesen",
    ],
}

def load_rtf_files(directory):
    """Load and aggregate RTF files by patient and task."""
    content_by_patient_and_task = defaultdict(lambda: defaultdict(str))

    for filename in os.listdir(directory):
        if filename.endswith(".rtf") and not filename.startswith("."):
            file_path = os.path.join(directory, filename)
            patient_id, task = parse_filename(filename)
            base_task_name = (
                "_".join(task.split("_")[:-1]) if "teil" in task else task
            )
            try:
                rtf_content = read_rtf(file_path)
                content_by_patient_and_task[patient_id][
                    base_task_name
                ] += f" {rtf_content}"
            except:
                continue

    return content_by_patient_and_task


def parse_filename(filename):
    """Parse filename to extract patient ID and task."""
    filename = filename[:-4]
    parts = filename.split("_")
    patient_id = parts[0]
    task = parts[2] if len(parts) > 2 else "unknown"
    part_info = parts[3] if len(parts) > 3 else ""
    full_task = f"{task}_{part_info}" if part_info else task
    return patient_id, full_task


def read_rtf(file_path):
    """Read RTF file and convert its content to plain text."""
    with open(file_path, "rb") as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result["encoding"]

    with open(file_path, "r", encoding=encoding, errors="ignore") as file:
        rtf_content = file.read()

    return rtf_to_text(rtf_content)


def split_into_lines(text):
    """Split text into lines, filtering out empty lines and unwanted content."""
    lines = text.splitlines()
    return [
        line
        for line in lines
        if line.strip() and ".mp3" not in line and "audio" not in line
    ]


def extract_and_remove_hashtags(text):
    """Extract and remove hashtags from the text."""
    pattern = r"#(.*?)#"
    matches = re.findall(pattern, text)
    text_without_hashtags = re.sub(pattern, "", text)
    return text_without_hashtags, matches


class Line:
    """Represents a line of text with associated metadata."""

    def __init__(self, speaker, text, line_number, tokenizer=None):
        self.speaker = speaker
        self.text = text
        self.line_number = line_number
        self.length_in_words = len(self.text.split())


def process_lines(
        pat_id,
        task,
        lines,
        stopwords_list,
        remove_numbers=False,

):
    """Process lines of text to create a Document object."""
    document = Document(pat_id, task)
    for i, line_text in enumerate(lines, start=1):
        speaker = (
            "Investigator"
            if line_text.startswith(("I:", "I::", "I1:", "I2:"))
            else "Subject"
        )
        cleaned_line = TextCleaner.clean_text_diarization_all(line_text, stopwords_list, remove_numbers)
        if cleaned_line != ".":
            line = Line(speaker, cleaned_line, i)
            document.add_line(line)
    return document


def main(
        transcripts_dict,
        output_path,
        task_dir,
        protocol,
        remove_stopwords,
        remove_numbers,
):
    """Main function to process documents."""
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Folder '{output_path}' was created.")
    else:
        print(f"Folder '{output_path}' already exists.")

    if remove_stopwords:
        stop_list = list(stopwords.words("german"))
    else:
        stop_list = []

    for patient_id, tasks in transcripts_dict.items():
        for task, rtf_content in tasks.items():

            no_hashtags, hashtags = extract_and_remove_hashtags(
                rtf_content
            )

            lines = split_into_lines(no_hashtags)
            document = process_lines(
                patient_id,
                task,
                lines,
                stop_list,
                remove_numbers=remove_numbers,
            )

            if task_dir == "discourse":
                document.segment_task(protocol)

            document.compile_texts_and_tags()
            s_tok = np.array(document.word_tags) == "s"
            words = np.array(document.words)

            with open(output_path + f"{patient_id}_{task}.txt", 'w', encoding='utf-8') as file:
                file.write(" ".join(words[s_tok]))


#Document class from original diarization script
class Document:
    """Represents a document with multiple lines of text and associated metadata."""

    def __init__(self, pat_id, task, lines=None):
        self.pat_id = pat_id
        self.task = task
        self.lines = lines if lines is not None else []
        self.has_segments = True if task == "discourse" else False
        self.sections = {}
        self.section_metrics = {}
        self.length_in_lines = len(self.lines)
        self.length_in_words = sum(line.length_in_words for line in self.lines)

        # Initialize segments to a default value if no sections are applicable
        self.segments = ['default'] * len(self.lines) if not self.has_segments else []

    def add_line(self, line):
        self.lines.append(line)
        self.length_in_lines = len(self.lines)
        self.length_in_words += line.length_in_words

        if not self.has_segments:
            self.segments.append('default')

    def compile_texts_and_tags(self):
        """Compile lists of all words and tokens with corresponding speaker tags."""
        self.words, self.word_tags, = (
            [],
            [],
        )
        self.word_segments = []

        for line, segment in zip(self.lines, self.segments):
            line_words = line.text.split()
            tag = "i" if line.speaker.lower() == "investigator" else "s"

            self.word_segments.extend([segment] * len(line_words))
            self.words.extend(line_words)
            self.word_tags.extend([tag] * len(line_words))

    def segment_task(self, protocol, cutoff=1):
        """Segment the document based on the given protocol and store sections."""
        if not self.has_segments:
            return self.segments  # Return default segments if segmentation not applicable

        patterns = {
            section: re.compile(
                "|".join(f"(?:\\b{re.escape(term)}\\b)" for term in terms),
                re.IGNORECASE,
            )
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


if __name__ == "__main__":
    for task_dir in ["interview"]:
        transcripts_directory = os.path.join(
            "..", "..", "..", "data", "language", task_dir, "transcripts")

        transcripts_dict = load_rtf_files(transcripts_directory)
        print(transcripts_dict.keys())

        args = {
            "remove_stopwords": False,
            "remove_numbers": False,
        }
        output_path = f"/Users/nilsl/Documents/PUK/VELAS/data/language/{task_dir}/preprocessed_transcripts/"
        main(transcripts_dict, output_path, task_dir, protocol, **args)
