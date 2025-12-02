#!/usr/bin/env python3
"""
Pelican-nlp Project
===================

Pelican-nlp is a tool developed to enable consistent and reproducible language processing.
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
from pelican_nlp.utils.setup_functions import participant_instantiator, load_config, remove_previous_derivative_dir
from pelican_nlp.preprocessing import LPDS
from pelican_nlp.utils.filename_parser import parse_lpds_filename

from pelican_nlp.config import debug_print, RUN_TESTS

# Project path pointing to current workspace example configuration file.
# Used if pipeline is run in programming environment instead of terminal.
#project_path = '/home/yvespauli/PELICAN-nlp/examples/example_transcription/config_transcription.yml'
project_path = '/home/yvespauli/PELICAN-nlp/examples/example_Cogmap/config_cogmap.yml'
#project_path = '/home/yvespauli/PELICAN-nlp/examples/example_image-descriptions/config_image-descriptions.yml'
#project_path = '/home/yvespauli/PycharmProjects/Transcription_Finn/config_transcription.yml'

class Pelican:

    """Main class for the Pelican project handling document processing and metric extraction."""
    
    def __init__(self, config_path: str = None, dev_mode: bool = False, test_mode: bool = False) -> None:

        self.dev_mode = dev_mode
        self.test_mode = test_mode
        
        # Skip config loading and project setup for test mode
        if test_mode:
            return

        self.config = load_config(config_path)
        self.project_path = Path(config_path).resolve().parent
        self.path_to_participants = self.project_path / 'participants'
        self.output_directory = self.project_path / 'derivatives'
        self.task = self.config['task_name']

        if not self.path_to_participants.is_dir():
            sys.exit('Error: Could not find participants directory; check folder structure.')

    def run(self) -> None:
        """Execute the main processing pipeline."""
        self._clear_gpu_memory()

        self._handle_output_directory()
        
        # Check/Create LPDS
        self._LPDS()
        
        # Instantiate all participants
        print("Instantiating all participants")
        participants = participant_instantiator(self.config, self.project_path)
        
        # Process each corpus
        for corpus_value in self.config['corpus_values']:
            self._process_corpus(self.config['corpus_key'], corpus_value, participants)

        print("Pipeline ran successfully!")

    def _process_corpus(self, corpus_key: str, corpus_value: str, participants: List) -> None:
        """Process a single corpus including preprocessing and metric extraction."""

        corpus_entity = corpus_key + '-' + corpus_value
        print(f'Processing corpus: {corpus_entity}')
        debug_print(participants, corpus_entity)
        corpus_documents = self._identify_corpus_files(participants, corpus_entity)
        debug_print(len(corpus_documents))
        corpus = Corpus(corpus_entity, corpus_documents[corpus_entity], self.config, self.project_path)

        for document in corpus_documents[corpus_entity]:
            document.corpus_name = corpus_entity

        if self.config['input_file'] == 'audio':
            # Process audio files first
            self._process_audio_corpus(corpus, corpus_entity)

        elif self.config['input_file'] == 'text':
            # Process text files (skip audio processing)
            self._process_text_corpus(corpus)

        del corpus


    def _LPDS(self):
        """Initialize LPDS and create derivative directory"""
        lpds = LPDS(self.project_path, self.config['multiple_sessions'])
        lpds.LPDS_checker()
        lpds.derivative_dir_creator()

    def _process_audio_corpus(self, corpus: Corpus, corpus_entity: str) -> None:
        """Process a corpus through the audio processing pipeline."""
        if self.config['transcription']:
            corpus.transcribe_audio()

        if self.config['opensmile_feature_extraction']:
            corpus.extract_opensmile_features()

        if self.config['prosogram_extraction']:
            corpus.extract_prosogram()

        # Check if text features are also needed (embeddings, logits, perplexity)
        text_metrics_needed = any(
            metric in self.config.get('metrics_to_extract', [])
            for metric in ['embeddings', 'logits', 'perplexity']
        )
        
        if text_metrics_needed:
            # Ensure transcription was completed
            if not self.config.get('transcription', False):
                print("Warning: Text metrics requested but transcription is disabled. "
                      "Enable transcription in config to extract text features from audio.")
            else:
                # Create Documents from transcription files
                transcription_docs = corpus.create_documents_from_transcriptions()
                
                if transcription_docs:
                    # Create a new corpus with transcription documents for text processing
                    transcription_corpus = Corpus(
                        corpus_entity, 
                        transcription_docs, 
                        self.config, 
                        self.project_path
                    )
                    
                    # Process transcription documents through text pipeline
                    self._process_text_corpus(transcription_corpus)
                    
                    del transcription_corpus
                else:
                    print("Warning: No transcription files found to process for text metrics.")

    def _process_text_corpus(self, corpus: Corpus) -> None:
        """Process a corpus through the text processing pipeline."""
        corpus.preprocess_all_documents()
        self._extract_metrics(corpus)

        if self.config['create_aggregation_of_results']:
            corpus.create_corpus_results_consolidation_csv()

        if self.config['output_document_information']:
            corpus.create_document_information_csv()

    def _extract_metrics(self, corpus: Corpus) -> None:
        """Extract specified metrics from the corpus."""
        for metric in self.config['metrics_to_extract']:
            if metric == 'logits':
                corpus.extract_logits()
            elif metric == 'embeddings':
                corpus.extract_embeddings()
            elif metric == 'perplexity':
                corpus.extract_perplexity()
            elif metric == 'topic_modeling':
                corpus.extract_topic_modeling()
            else:
                raise ValueError(f"Unsupported metric: {metric}")
        
        self._clear_gpu_memory()

    def _identify_corpus_files(self, participants: List, entity: str) -> Dict:
        """Identify and group files based on specified entity-value pair."""
        debug_print(f'identifying corpus files')
        corpus_dict = {entity: []}
        debug_print(len(participants))
        
        # Check if entity is in key-value format
        if '-' in entity:
            key, value = entity.split('-', 1)
            
            for participant in participants:
                debug_print(participant.documents)
                for document in participant.documents:
                    entities = parse_lpds_filename(document.name)
                    debug_print(entities)
                    if key in entities and str(entities[key]) == value:
                        corpus_dict[entity].append(document)
        else:
            # Entity is just a value, check all keys
            for participant in participants:
                debug_print(participant.documents)
                for document in participant.documents:
                    entities = parse_lpds_filename(document.name)
                    debug_print(entities)
                    # Convert all values to strings for comparison
                    if any(str(val) == entity for val in entities.values()):
                        corpus_dict[entity].append(document)
        
        return corpus_dict

    def _handle_output_directory(self) -> None:
        """Handle the output directory based on dev mode."""
        if self.dev_mode:
            remove_previous_derivative_dir(self.output_directory)
        elif self.output_directory.exists():
            self._prompt_for_continuation()

    def run_tests(self):
        """Run all example tests from utils/unittests/examples folders."""
        from pathlib import Path
        import tempfile
        import shutil

        print("Running tests from utils/unittests/examples...")
        
        # Get the path to the examples directory
        examples_dir = Path(__file__).parent / "utils" / "unittests" / "examples"

        if not examples_dir.exists():
            print(f"Examples directory not found: {examples_dir}")
            return
        
        # Sync config files from sample_configuration_files to example directories
        self._sync_config_files()
        
        # Create a temporary directory for test outputs
        test_dir = tempfile.mkdtemp()
        
        try:
            # Find all example directories
            example_dirs = [d for d in examples_dir.iterdir() if d.is_dir() and d.name.startswith('example_')]
            
            if not example_dirs:
                print("No example directories found")
                return
            
            print(f"Found {len(example_dirs)} example directories")

            success_counter=0
            # Run each example
            for example_dir in example_dirs:
                example_name = example_dir.name.replace('example_', '')
                print(f"\nTesting {example_name} example...")
                
                # Find the config file in the example directory
                config_files = list(example_dir.glob(f"config_{example_name}.yml"))
                if not config_files:
                    print(f"No config file found for {example_name}")
                    continue
                elif len(config_files)>1:
                    print(f"More than one config file in {example_name} directory.")
                    continue

                config_file = config_files[0]
                output_dir = Path(test_dir) / example_name
                output_dir.mkdir(exist_ok=True)
                
                # Run the pipeline directly using the Pelican class
                try:
                    print(f"Running pipeline for {example_name}...")
                    
                    # Create a Pelican instance with the config file from the example directory
                    pelican = Pelican(str(config_file))
                    
                    # Run the pipeline
                    pelican.run()

                    print(f"✓ {example_name} test completed successfully")
                    success_counter += 1

                except Exception as e:
                    print(f"✗ {example_name} test failed with error: {str(e)}")
            
            print("\nAll tests completed")
            print(f"{success_counter} out of {len(example_dirs)} tests ran successfully")

        finally:
            # Clean up temporary directory
            shutil.rmtree(test_dir)

    def _sync_config_files(self):
        """Sync configuration files from sample_configuration_files to example directories."""
        from pathlib import Path
        import shutil
        
        sample_config_dir = Path(__file__).parent / "sample_configuration_files"
        examples_dir = Path(__file__).parent / "utils" / "unittests" / "examples"
        
        if not sample_config_dir.exists():
            print(f"Sample configuration directory not found: {sample_config_dir}")
            return
        
        print("Syncing configuration files from sample_configuration_files to example directories...")
        
        # Find all example directories
        example_dirs = [d for d in examples_dir.iterdir() if d.is_dir() and d.name.startswith('example_')]
        
        for example_dir in example_dirs:
            example_name = example_dir.name.replace('example_', '')
            source_config = sample_config_dir / f"config_{example_name}.yml"
            target_config = example_dir / f"config_{example_name}.yml"
            
            if source_config.exists():
                try:
                    shutil.copy2(source_config, target_config)
                    print(f"Synced config_{example_name}.yml to {example_dir.name}")
                except Exception as e:
                    print(f"Failed to sync config_{example_name}.yml: {str(e)}")
            else:
                print(f"Source config file not found: {source_config}")

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
    if RUN_TESTS:
        print("Running tests...")
        Pelican(test_mode=True).run_tests()
    else:
        # For direct execution, use default config
        Pelican(project_path, dev_mode=True).run()
