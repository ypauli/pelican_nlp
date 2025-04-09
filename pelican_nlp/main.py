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

from pelican_nlp.core import Corpus
from pelican_nlp.utils.setup_functions import subject_instantiator, load_config, remove_previous_derivative_dir
from pelican_nlp.preprocessing import LPDS

class Pelican:

    """Main class for the Pelican project handling document processing and metric extraction."""
    
    def __init__(self, config_path: str = None, dev_mode: bool = True) -> None:
        self.dev_mode = dev_mode
        
        # If no config path is provided, use the default config from package
        if config_path is None:
            package_dir = Path(__file__).parent
            default_config = package_dir / 'configuration_files' / 'config_fluency.yml'
            if default_config.exists():
                config_path = str(default_config)
                print(f"Using default configuration file: {config_path}")
            else:
                sys.exit('Error: Default configuration file not found in package.')
        
        # Verify the provided path is a YAML file
        elif not config_path.endswith(('.yml', '.yaml')):
            sys.exit('Error: Configuration file must be a YAML file (*.yml or *.yaml)')

        self.config = load_config(config_path)
        self.project_path = Path(config_path).resolve().parent
        self.path_to_subjects = self.project_path / 'subjects'
        self.output_directory = self.project_path / 'derivatives'
        self.task = self.config['task_name']
        
        # Add test configuration, TESTS NOT YET IMPLEMENTED
        self.test_config = {
            'run_all': True,  # Run all tests by default
            'test_paths': ['tests'],  # Default test directory
            'markers': [],  # Specific test markers to run
            'skip_slow': True,  # Skip slow tests by default
        }
        
        if not self.path_to_subjects.is_dir():
            sys.exit('Error: Could not find subjects directory; check folder structure.')

    def run(self) -> None:
        """Execute the main processing pipeline."""
        self._clear_gpu_memory()

        '''
        #run unittests in dev_mode; not yet implemented
        if self.dev_mode:
            self._run_tests()
        '''

        self._handle_output_directory()
        
        # Check/Create LPDS
        self._LPDS()
        
        # Instantiate all subjects
        subjects = subject_instantiator(self.config, self.project_path)
        
        # Process each corpus
        for corpus_name in self.config['corpus_names']:
            self._process_corpus(corpus_name, subjects)

    def _process_corpus(self, corpus_name: str, subjects: List) -> None:

        """Process a single corpus including preprocessing and metric extraction."""
        print(f'Processing corpus: {corpus_name}')

        corpus_documents = self._identify_corpus_files(subjects, corpus_name)
        corpus = Corpus(corpus_name, corpus_documents[corpus_name], self.config, self.project_path)

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
        """Initialize LPDS and create derivative directory"""
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

    def _run_tests(self) -> None:
        # Run unittests to test implemented functions... not yet in use
        """Run test suite in development mode with configurable options."""
        import pytest
        print("Running tests in development mode...")

        # Build pytest arguments
        pytest_args = ["-v", "--no-header"]

        # Add test paths
        pytest_args.extend(self.test_config['test_paths'])

        # Add markers if specified
        for marker in self.test_config['markers']:
            pytest_args.extend(["-m", marker])

        # Skip slow tests if configured
        if self.test_config['skip_slow']:
            pytest_args.extend(["-m", "not slow"])

        # Run pytest with constructed arguments
        result = pytest.main(pytest_args)

        # Handle test results
        if result != 0:
            print("Tests failed. Aborting execution.")
            sys.exit(1)

        print("All tests passed. Continuing with execution.\n")

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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    Pelican().run()
