# Main project file
# Created 1.10.24 by Yves Pauli
# =============================
import os
import sys
import torch.cuda
import re

from pelican.preprocessing import Corpus
from pelican.setup_functions import subject_instantiator, _load_config, _remove_previous_derivative_dir
from pelican.LPDS import LPDS

class Pelican:
    def __init__(self, config_path='config_discourse.yml', dev_mode=True):
        self.dev_mode = dev_mode
        self.config = _load_config(config_path)
        self.project_path = self.config['PATH_TO_PROJECT_FOLDER']
        self.path_to_subjects = os.path.join(self.project_path, 'subjects')
        self.output_directory = os.path.join(self.project_path, 'derivatives')
        self.task = self.config['task_name']

        if not os.path.isdir(self.path_to_subjects):
            sys.exit('Warning: Could not find subjects; check folder structure.')

    def run(self):

        self.empty_cuda_cache()

        if self.dev_mode:
            _remove_previous_derivative_dir(self.output_directory)
        else:
            if self.output_directory:
                print('Warning: An output directory already exists. Continuing might invalidate previously computed results.')
                confirm = input("Do you want to continue? Type 'yes' to proceed: ").strip().lower()
                if confirm not in ('yes', 'y'):
                    print("Operation aborted.")
                    exit(1)

        #Check LPDS and create derivative directory
        LPDS_instance = LPDS(self.project_path)
        LPDS_instance.LPDS_checker()
        LPDS_instance.derivative_dir_creator(self.config['metric_to_extract'])

        #Instantiate all subjects
        subjects = subject_instantiator(self.config)
        print(f'instantiated subjects: {subjects}')

        for corpus_name in self.config['corpus_names']:

            print(f'Processing corpus: {corpus_name}')

            #Identifying documents belonging to corpus
            corpus_documents = self._identify_corpus_files(subjects, corpus_name)
            print(f'The corpus documents are: {corpus_documents}')

            corpus = Corpus(corpus_name, corpus_documents[corpus_name], self.config)
            corpus.preprocess_all_documents()

            if self.config['extract_logits']:
                print('Extracting logits...')
                corpus.extract_logits()
                self.empty_cuda_cache()

            if self.config['extract_embeddings']:
                print('Extracting embeddings...')
                corpus.extract_embeddings()
                self.empty_cuda_cache()

            del corpus


    def _identify_corpus_files(self, subjects, corpus):
        corpus_dict = {corpus: []}
        for subject in subjects:
            for document in subject.documents:
                base, ext = os.path.splitext(document.name)
                document.extension = ext
                parts = re.split('[_]', base)
                if len(parts) >= 4 and parts[3] == corpus:
                    corpus_dict[corpus].append(document)
        return corpus_dict


    def empty_cuda_cache(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == '__main__':
    Pelican().run()
