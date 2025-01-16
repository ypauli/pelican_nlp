#this is the main project file
#created 1.10.24
#created by Yves Pauli
#=============================
import os
import torch.cuda
import sys
import shutil
import yaml
from torchvision.datasets.utils import download_file_from_google_drive

from pelican.document import Document
from pelican.preprocessing import Subject, Corpus
from pelican.setup_functions import ignore_files, subject_instatiator


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

        #Instantiating all subjects
        subjects = subject_instatiator(self.config)

        # Process each corpus specified in the configuration
        for current_corpus in self.config['corpus_names']:
            print(f'corpus {current_corpus} is being processed')

            documents = []
            # Load all files belonging to the same corpus
            for subject in subjects:

                if subject.numberOfSessions is not None:
                    path_sessionfolder = os.path.join(self.path_to_subjects, subject.subjectID)
                    print(f'The sessions of subject {subject} are {os.listdir(path_sessionfolder)}')

                    for i in range(subject.numberOfSessions):
                        current_session = os.listdir(path_sessionfolder)[i]
                        print(f'The current session is: {current_session}')
                        filepath = os.path.join(self.path_to_subjects, subject.subjectID, current_session, current_corpus)
                        print(f'The filepath is: {filepath}')
                        if os.path.isdir(filepath):
                            file_name = os.listdir(filepath)[0]
                            print(f'The current file is: {file_name}')
                            documents.append(Document(filepath, file_name, current_corpus,
                                                    has_sections=self.config['has_multiple_sections'],
                                                    section_identifier=self.config['section_identification'],
                                                    number_of_sections=self.config['number_of_sections']))
                            subject.add_document(documents[-1])
                else:
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