#!/usr/bin/env python3
"""
Pelican Project
===============

Pelican is a tool developed to enable consistent and reproducible language processing.
Main entry point for the Pelican project handling document processing and metric extraction.

Author: Yves Pauli
Created: 2024-01-10
Version: 1.0.0

Copyright (c) 2024 Yves Pauli
License: Attribution-NonCommercial 4.0 International
All rights reserved.
"""

from pathlib import Path
from typing import Dict, List
import torch.cuda
import sys

from pelican.core.corpus import Corpus
from pelican.utils.setup_functions import subject_instantiator, load_config, remove_previous_derivative_dir
from pelican.preprocessing.LPDS import LPDS

# Constants
DEFAULT_CONFIG_PATH = 'configuration_files/config_morteza.yml'

class Pelican:

    """Main class for the Pelican project handling document processing and metric extraction."""
    
    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH, dev_mode: bool = True) -> None:
        self.dev_mode = dev_mode
        self.config = load_config(config_path)
        self.project_path = Path(self.config['PATH_TO_PROJECT_FOLDER'])
        self.path_to_subjects = self.project_path / 'subjects'
        self.output_directory = self.project_path / 'derivatives'
        self.task = self.config['task_name']

        if not self.path_to_subjects.is_dir():
            sys.exit('Error: Could not find subjects directory; check folder structure.')

    def run(self) -> None:
        """Execute the main processing pipeline."""
        self._clear_gpu_memory()
        self._handle_output_directory()

        #Check/Create LPDS
        self._LPDS()

        #Instantiate all subjects
        subjects = subject_instantiator(self.config)

        # Process each corpus
        for corpus_name in self.config['corpus_names']:
            self._process_corpus(corpus_name, subjects)

    def _process_corpus(self, corpus_name: str, subjects: List) -> None:

        """Process a single corpus including preprocessing and metric extraction."""
        print(f'Processing corpus: {corpus_name}')

        corpus_documents = self._identify_corpus_files(subjects, corpus_name)
        corpus = Corpus(corpus_name, corpus_documents[corpus_name], self.config)

        for document in corpus_documents[corpus_name]:
            document.corpus_name = corpus_name

        if self.config['input_file']=='text':
            corpus.preprocess_all_documents()
            print(f'Corpus {corpus_name} is preprocessed')

            self._extract_metrics(corpus)

            if self.config['create_aggregation_of_results']:
                corpus.create_corpus_results_consolidation_csv()

            if self.config['output_document_information']:
                corpus.create_document_information_csv()


        elif self.config['input_file']=='audio':
            if self.config['opensmile_feature_extraction']:
                corpus.extract_opensmile_features()

            if self.config['prosogram_extraction']:
                corpus.extract_prosogram()

        del corpus


    def _LPDS(self):
        # Initialize LPDS and create derivative directory
        lpds = LPDS(self.project_path, self.config['multiple_sessions'])
        lpds.LPDS_checker()
        lpds.derivative_dir_creator()

    def _extract_metrics(self, corpus: Corpus) -> None:
        """Extract specified metrics from the corpus."""
        metric = self.config['metric_to_extract']
        if metric == 'logits':
            print('Extracting logits...')
            corpus.extract_logits()
        elif metric == 'embeddings':
            print('Extracting embeddings...')
            corpus.extract_embeddings()
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        self._clear_gpu_memory()

    def _identify_corpus_files(self, subjects: List, corpus: str) -> Dict:
        """Identify and group files belonging to a specific corpus."""
        corpus_dict = {corpus: []}
        for subject in subjects:
            for document in subject.documents:
                name = Path(document.name)
                document.extension = name.suffix
                # Split by both '_' and '.' to get all parts
                parts = name.stem.replace('.', '_').split('_')
                # Check if corpus name appears in any part
                if corpus in parts:
                    corpus_dict[corpus].append(document)
        return corpus_dict

    def _handle_output_directory(self) -> None:
        """Handle the output directory based on dev mode."""
        if self.dev_mode:
            remove_previous_derivative_dir(self.output_directory)
        elif self.output_directory.exists():
            self._prompt_for_continuation()

    @staticmethod
    def _prompt_for_continuation() -> None:
        """Prompt user for continuation if output directory exists."""
        print('Warning: An output directory already exists. Continuing might invalidate previously computed results.')
        confirm = input("Do you want to continue? Type 'yes' to proceed: ").strip().lower()
        if confirm not in ('yes', 'y'):
            print("Operation aborted.")
            sys.exit(0)

    @staticmethod
    def _clear_gpu_memory() -> None:
        """Clear CUDA cache if GPU is available."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == '__main__':
    Pelican().run()
