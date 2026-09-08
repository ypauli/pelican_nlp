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
from typing import List
import sys
import subprocess

from pelican_nlp.core import Corpus
from pelican_nlp.utils.setup_functions import participant_instantiator, load_config, remove_previous_derivative_dir
from pelican_nlp.preprocessing import LPDS
from pelican_nlp.utils.filename_parser import parse_lpds_filename
from pelican_nlp.utils.lpds_paths import grouped_corpus_jobs, grouped_document_jobs, resolve_unit_folder
from pelican_nlp.extraction.metric_registry import run_configured_metrics

from pelican_nlp.config import debug_print


class Pelican:

    """Main class for the Pelican project handling document processing and metric extraction."""
    
    def __init__(
        self,
        config_path: str = None,
        dev_mode: bool = False,
        test_mode: bool = False,
        text_from_transcriptions: bool = False,
    ) -> None:

        self.dev_mode = dev_mode
        self.test_mode = test_mode
        self.text_from_transcriptions = text_from_transcriptions
        self.config_path = config_path
        self.skip_existing = True  # Flag to skip already processed files
        
        # Skip config loading and project setup for test mode
        if test_mode:
            return

        if config_path is None:
            raise ValueError("config_path must be provided unless running in test_mode.")

        self.config = load_config(config_path)
        self.project_path = Path(config_path).resolve().parent
        self.path_to_participants = self.project_path / 'participants'
        self.output_directory = self.project_path / 'derivatives'
        self.task = self.config.get('task_name')

        from pelican_nlp.extras import require_config_extras
        from pelican_nlp.utils.model_cache import configure_device_caches
        from pelican_nlp.utils.gpu_budget import apply_gpu_budget

        require_config_extras(self.config)

        configure_device_caches()
        apply_gpu_budget()

        if not self.path_to_participants.is_dir():
            sys.exit('Error: Could not find participants directory; check folder structure.')

    def run(self) -> None:
        """Execute the main processing pipeline."""
        self._clear_gpu_memory()

        # Only handle/remove output directory in the first (audio/text) phase.
        if not self.text_from_transcriptions:
            self._handle_output_directory()
        else:
            print("Skipping output directory handling for text-from-transcriptions phase.")
        
        # Check/Create LPDS
        self._LPDS()
        
        # Instantiate all unit folders (participants and collections)
        print("Instantiating all participants")
        participants = participant_instantiator(self.config, self.project_path)
        
        # If this is the second phase, run only text-from-transcriptions
        if self.text_from_transcriptions:
            self._run_text_from_transcriptions(participants)
            print("Text-from-transcriptions phase completed!")
            return

        for corpus_entity, documents in grouped_corpus_jobs(
            participants,
            self.config.get('corpus_key'),
            self.config.get('corpus_values'),
        ):
            self._run_on_documents(corpus_entity, documents)

        print("Pipeline ran successfully!")

    def _run_on_documents(self, corpus_entity: str, documents: List) -> None:
        """Process a single corpus including preprocessing and metric extraction."""
        if not documents:
            print(f"No documents for corpus {corpus_entity}, skipping.")
            return

        print(f'Processing corpus: {corpus_entity}')
        debug_print(documents, corpus_entity)
        corpus = Corpus(corpus_entity, documents, self.config, self.project_path)

        for document in documents:
            document.corpus_name = corpus_entity

        if self.text_from_transcriptions or self.config.get('input_file') == 'text':
            self._process_text_corpus(corpus)
        elif self.config.get('input_file') == 'audio':
            self._process_audio_corpus(corpus, corpus_entity)

        del corpus


    def _LPDS(self):
        """Initialize LPDS and create derivative directory"""
        lpds = LPDS(self.project_path, self.config['multiple_sessions'])
        lpds.LPDS_checker()
        lpds.derivative_dir_creator()

    def _process_audio_corpus(self, corpus: Corpus, corpus_entity: str) -> None:
        """Process a corpus through the audio processing pipeline."""
        if self.config.get('transcription'):
            corpus.transcribe_audio(skip_existing=self.skip_existing)

        if self.config.get('opensmile_feature_extraction'):
            corpus.extract_opensmile_features()

        if self.config.get('prosogram_extraction'):
            corpus.extract_prosogram()

        # Check if text features are also needed (embeddings, logits, perplexity, topic modeling)
        text_metrics_needed = any(
            metric in self.config.get('metrics_to_extract', [])
            for metric in ['embeddings', 'logits', 'perplexity', 'topic_modeling']
        )
        
        if text_metrics_needed:
            # Ensure transcription was completed
            if not self.config.get('transcription', False):
                print("Warning: Text metrics requested but transcription is disabled. "
                      "Enable transcription in config to extract text features from audio.")
            else:
                # Launch a second-phase Pelican run in a fresh process for text-from-transcriptions
                if not self.config_path:
                    print("Error: Cannot start text-from-transcriptions phase without a config path.")
                    return
                
                print("Starting second-phase Pelican run for text-from-transcriptions in a fresh process...")
                cmd = [
                    sys.executable,
                    "-m",
                    "pelican_nlp.main",
                    self.config_path,
                    "--text-from-transcriptions",
                ]
                result = subprocess.run(cmd)
                if result.returncode != 0:
                    print(f"Second-phase Pelican run failed with exit code {result.returncode}")
                else:
                    print("Second-phase Pelican run completed successfully.")

    def _process_text_corpus(self, corpus: Corpus) -> None:
        """Process a corpus through the text processing pipeline."""
        corpus.preprocess_all_documents()
        self._extract_metrics(corpus)

        if self.config.get('create_aggregation_of_results'):
            corpus.create_corpus_results_consolidation_csv()

        if self.config.get('output_document_information'):
            corpus.create_document_information_csv()

    def _extract_metrics(self, corpus: Corpus) -> None:
        """Extract specified metrics from the corpus."""
        run_configured_metrics(corpus)
        self._clear_gpu_memory()

    def _handle_output_directory(self) -> None:
        """Handle the output directory based on dev mode."""
        # If skip_existing is True, never delete the directory
        if self.skip_existing:
            print("skip_existing is True - preserving existing files in output directory.")
            return
        
        if self.dev_mode:
            remove_previous_derivative_dir(self.output_directory)
        elif self.output_directory.exists():
            should_continue = self._prompt_for_continuation()
            if not should_continue:
                # User chose "no" - set flag to skip existing files
                self.skip_existing = True
                print("Will skip files that are already transcribed.")

    @staticmethod
    def _prompt_for_continuation() -> bool:
        """
        Prompt user for continuation if output directory exists.
        
        Returns:
            True if user wants to continue (overwrite), False if user wants to skip existing files.
        """
        print('Warning: An output directory already exists. Continuing might invalidate previously computed results.')
        confirm = input("Do you want to continue? Type 'yes' to proceed (will overwrite), or 'no' to skip already processed file (currently only for audio transcriptions)").strip().lower()
        if confirm in ('yes', 'y'):
            return True
        else:
            return False

    @staticmethod
    def _clear_gpu_memory() -> None:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    def _run_text_from_transcriptions(self, participants: List) -> None:
        """Second-phase run: build corpora from transcription text files on disk."""
        from pelican_nlp.core.document import Document

        transcription_dir = self.output_directory / 'transcription'
        if not transcription_dir.exists():
            print(f"Warning: Transcription directory not found at {transcription_dir}")
            return

        transcription_docs = []
        for entry in transcription_dir.iterdir():
            if not entry.is_file() or not entry.name.endswith("_transcript.txt"):
                continue

            base_name = entry.stem
            if base_name.endswith("_transcript"):
                audio_stem = base_name.rsplit("_transcript", 1)[0]
            else:
                audio_stem = base_name

            entities = parse_lpds_filename(entry.name)
            audio_entities = parse_lpds_filename(audio_stem)
            merged = {**audio_entities, **entities}

            source_folder = None
            for unit in participants or []:
                for document in unit.documents:
                    if Path(document.name).stem == audio_stem:
                        source_folder = unit.name
                        break
                if source_folder:
                    break
            if source_folder is None:
                resolved = resolve_unit_folder(None, merged)
                source_folder = None if resolved == "unassigned" else resolved

            doc = Document(
                file_path=str(transcription_dir),
                name=entry.name,
                participant_ID=merged.get("part"),
                source_folder=source_folder,
                task=merged.get("task") or self.task,
                num_speakers=self.config.get('number_of_speakers'),
                has_sections=self.config.get('has_multiple_sections', False),
                section_identifier=self.config.get('section_identification'),
                number_of_sections=self.config.get('number_of_sections'),
                has_section_titles=self.config.get('has_section_titles', False),
            )
            doc.lpds_entities = merged
            transcription_docs.append(doc)
            print(f"[Second phase] Added transcription document: {entry.name}")

        if not transcription_docs:
            print("Warning: No transcription files found for text-from-transcriptions phase.")
            return

        for corpus_entity, docs in grouped_document_jobs(
            transcription_docs,
            self.config.get('corpus_key'),
            self.config.get('corpus_values'),
        ):
            print(f"[Second phase] Processing corpus from transcriptions: {corpus_entity}")
            self._run_on_documents(corpus_entity, docs)

        self._clear_gpu_memory()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Run Pelican-nlp pipeline.")
    parser.add_argument(
        "config_path",
        nargs="?",
        default=None,
        help="Path to the configuration YAML file.",
    )
    parser.add_argument(
        "--text-from-transcriptions",
        action="store_true",
        help="Run only the text-from-transcriptions phase in a fresh process.",
    )
    args = parser.parse_args()
    if not args.config_path:
        parser.error("config_path is required (or run pelican-run from a project directory).")

    Pelican(
        args.config_path,
        dev_mode=True,
        text_from_transcriptions=args.text_from_transcriptions,
    ).run()
