#this is the main project file
#created 1.10.24
#created by Yves Pauli
#=============================
import os
import torch.cuda
import sys
import shutil
import yaml
from document import Document
from preprocessing import Subject, Corpus
from setup_functions import ignore_files

class Pelican:
    def __init__(self, config_path='config.yml', dev_mode=True):

        self.dev_mode = dev_mode
        self.config = self._load_config(config_path)
        self.path_to_subjects = os.path.join(self.config['PATH_TO_PROJECT_FOLDER'], 'Subjects')
        self.output_directory = os.path.join(self.config['PATH_TO_PROJECT_FOLDER'], 'Outputs')

        if not os.path.isdir(self.path_to_subjects):
            print('Warning: Could not find subjects; Check folder structure.')
            sys.exit()

    def _load_config(self, config_path):
        try:
            with open(config_path, 'r') as stream:
                return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(f"Error loading configuration: {exc}")
            sys.exit()

    def run(self):

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Remove Outputs folder if it already exists (only in development mode)
        if self.dev_mode and os.path.isdir(self.output_directory):
            shutil.rmtree(self.output_directory)

        # Create output directory and copy subjects
        shutil.copytree(self.path_to_subjects, self.output_directory, ignore=self.ignore_files)
        os.mkdir(os.path.join(self.output_directory, 'results_consolidation'))

        # Instantiate all subjects
        subjects = [Subject(subject) for subject in os.listdir(self.path_to_subjects)]

        # Process each corpus specified in the configuration
        for current_corpus in self.config['corpus_names']:
            print(f'{current_corpus} is being processed')

            documents = []
            # Load all files belonging to the same corpus
            for subject in subjects:
                print(subject.subjectID)
                print(current_corpus)
                filepath = os.path.join(self.path_to_subjects, subject.subjectID, current_corpus)
                if os.path.isdir(filepath):
                    file_name = os.listdir(filepath)[0]
                    documents.append(Document(filepath, file_name, current_corpus,
                                              has_sections=self.config['has_multiple_sections'],
                                              section_identifier=self.config['section_identification'],
                                              number_of_sections=self.config['number_of_sections']))
                    subject.add_document(documents[-1])

            # Initialize and process the corpus
            corpus = Corpus(current_corpus, documents, self.config)
            corpus.preprocess_all_documents()
            print(f'all files in {current_corpus} have been preprocessed')

            if self.config['extract_logits']:
                print('==================Extracting logits===================')
                corpus.extract_logits()

            if self.config['extract_embeddings']:
                print('================Extracting embeddings=================')
                corpus.extract_embeddings()

            del corpus

    @staticmethod
    def ignore_files(directory, files):
        """Ignore certain files when copying the subjects folder."""
        # Add logic for which files to ignore if necessary
        return []

if __name__ == '__main__':
    app = Pelican()
    app.run()