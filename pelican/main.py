# Main project file
# Created 1.10.24 by Yves Pauli
# =============================
import os
import sys
import torch.cuda

from pelican.preprocessing import Corpus
from pelican.setup_functions import subject_instatiator, _load_config, _reset_output_directory, _create_documents

class Pelican:
    def __init__(self, config_path='config.yml', dev_mode=True):
        self.dev_mode = dev_mode
        self.config = _load_config(config_path)
        self.path_to_subjects = os.path.join(self.config['PATH_TO_PROJECT_FOLDER'], 'Subjects')
        self.output_directory = os.path.join(self.config['PATH_TO_PROJECT_FOLDER'], 'Outputs')

        if not os.path.isdir(self.path_to_subjects):
            sys.exit('Warning: Could not find subjects; check folder structure.')

    def run(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self.dev_mode:
            _reset_output_directory(self.output_directory, self.path_to_subjects)

        #Instantiate all subjects
        subjects = subject_instatiator(self.config)

        for corpus_name in self.config['corpus_names']:
            print(f'Processing corpus: {corpus_name}')
            documents = self._process_corpus(subjects, corpus_name)

            corpus = Corpus(corpus_name, documents, self.config)
            corpus.preprocess_all_documents()

            if self.config['extract_logits']:
                print('Extracting logits...')
                corpus.extract_logits()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if self.config['extract_embeddings']:
                print('Extracting embeddings...')
                corpus.extract_embeddings()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            del corpus

    def _process_corpus(self, subjects, corpus_name):
        documents = []
        for subject in subjects:
            session_paths = self._get_subject_sessions(subject, corpus_name)
            for filepath in session_paths:
                documents.extend(_create_documents(filepath, corpus_name, self.config))
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

if __name__ == '__main__':
    Pelican().run()
