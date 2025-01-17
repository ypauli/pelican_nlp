# Main project file
# Created 1.10.24 by Yves Pauli
# =============================
import os
import sys
import shutil
import yaml
import torch.cuda

from pelican.document import Document
from pelican.preprocessing import Corpus
from pelican.setup_functions import subject_instatiator

class Pelican:
    def __init__(self, config_path='config.yml', dev_mode=True):
        self.dev_mode = dev_mode
        self.config = self._load_config(config_path)
        self.path_to_subjects = os.path.join(self.config['PATH_TO_PROJECT_FOLDER'], 'Subjects')
        self.output_directory = os.path.join(self.config['PATH_TO_PROJECT_FOLDER'], 'Outputs')

        if not os.path.isdir(self.path_to_subjects):
            sys.exit('Warning: Could not find subjects; check folder structure.')

    def run(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self.dev_mode:
            self._reset_output_directory()

        subjects = subject_instatiator(self.config)

        for corpus_name in self.config['corpus_names']:
            print(f'Processing corpus: {corpus_name}')
            documents = self._process_corpus(subjects, corpus_name)

            corpus = Corpus(corpus_name, documents, self.config)
            corpus.preprocess_all_documents()

            if self.config['extract_logits']:
                print('Extracting logits...')
                corpus.extract_logits()

            if self.config['extract_embeddings']:
                print('Extracting embeddings...')
                corpus.extract_embeddings()

            del corpus

    def _load_config(self, config_path):
        try:
            with open(config_path, 'r') as stream:
                return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            sys.exit(f"Error loading configuration: {exc}")

    def _reset_output_directory(self):
        if os.path.isdir(self.output_directory):
            shutil.rmtree(self.output_directory)
        shutil.copytree(self.path_to_subjects, self.output_directory, ignore=self.ignore_files)
        os.mkdir(os.path.join(self.output_directory, 'results_consolidation'))

    def _process_corpus(self, subjects, corpus_name):
        documents = []
        for subject in subjects:
            session_paths = self._get_subject_sessions(subject, corpus_name)
            for filepath in session_paths:
                documents.extend(self._create_documents(filepath, corpus_name))
                subject.add_document(documents[-1])  # Add only the last document
        return documents

    def _get_subject_sessions(self, subject, corpus_name):
        if not subject.numberOfSessions:
            subject_path = os.path.join(self.path_to_subjects, subject.subjectID, corpus_name)
            return [subject_path] if os.path.isdir(subject_path) else []

        session_dir = os.path.join(self.path_to_subjects, subject.subjectID)
        return [
            os.path.join(session_dir, session, corpus_name)
            for session in os.listdir(session_dir)
            if os.path.isdir(os.path.join(session_dir, session, corpus_name))
        ]

    def _create_documents(self, filepath, corpus_name):
        return [
            Document(
                filepath,
                file_name,
                corpus_name,
                has_sections=self.config['has_multiple_sections'],
                section_identifier=self.config['section_identification'],
                number_of_sections=self.config['number_of_sections']
            )
            for file_name in os.listdir(filepath)
        ]

    @staticmethod
    def ignore_files(directory, files):
        return [f for f in files if os.path.isfile(os.path.join(directory, f))]

if __name__ == '__main__':
    Pelican().run()
