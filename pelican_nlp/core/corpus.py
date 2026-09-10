"""
This module provides the Corpus class, which aggregates documents where the same processing
steps applied and results should be aggregated.
(e.g. all fluency files from task 'animals' or all image-descriptions from the same image)

This class contains the pipelines for homogenous processing and metric extraction of all grouped files.
"""

from ..preprocessing import TextPreprocessingPipeline
from ..utils.csv_functions import store_features_to_csv
from pelican_nlp.config import debug_print
from pelican_nlp.utils.setup_functions import is_hidden_or_system_file
from pelican_nlp.utils.lpds_paths import aggregation_unit_key, unit_folder_for_document
import os
import io
import re
from collections import defaultdict
import pandas as pd
import numpy as np

_SIMILARITY_WINDOW_FILE = re.compile(
    r"semantic-similarity-window-(\d+)\.csv$", re.IGNORECASE
)
_SIMILARITY_SENTENCE_MARKERS = (
    "semantic-similarity-sentence.csv",
    "semantic-similarity-window-sentence.csv",
)
_SIMILARITY_DETAIL_MARKERS = (
    "semantic-similarity-window-details-",
    "semantic-similarity-sentence-details",
)


def similarity_aggregation_family(filename):
    """Return ``window_N`` or ``sentence`` from a derivatives filename, else None."""
    name = os.path.basename(str(filename))
    lowered = name.lower()
    if any(marker in lowered for marker in _SIMILARITY_DETAIL_MARKERS):
        return None
    if any(lowered.endswith(marker) for marker in _SIMILARITY_SENTENCE_MARKERS):
        return "sentence"
    match = _SIMILARITY_WINDOW_FILE.search(lowered)
    if match:
        return f"window_{match.group(1)}"
    return None


def _corpus_key_value(corpus_name):
    name = str(corpus_name)
    if '-' in name:
        key, value = name.split('-', 1)
        return key, value
    return 'unit', name


class Corpus:
    def __init__(self, corpus_name, documents, configuration_settings, project_folder, reporter=None):
        from pelican_nlp.utils.progress import NullReporter

        self.name = corpus_name
        self.key, self.value = _corpus_key_value(corpus_name)
        self.documents = documents
        self.config = configuration_settings
        self.project_folder = project_folder
        self.derivatives_dir = project_folder / 'derivatives'
        self.pipeline = TextPreprocessingPipeline(self.config)
        self.task = configuration_settings.get('task_name')
        self.reporter = reporter if reporter is not None else NullReporter()

    def preprocess_all_documents(self):
        from pelican_nlp.utils.progress import walk_units

        reporter = self.reporter
        for _unit, docs in walk_units(reporter, self.documents, "preprocess"):
            for document in docs:
                reporter.set_postfix(document.name)
                document.detect_sections()
                document.process_document(self.pipeline)
                reporter.advance_item(document.name)

    def create_corpus_results_consolidation_csv(self) -> None:
        """Create comprehensive aggregated results CSV files for semantic similarity metrics."""
        reporter = self.reporter
        reporter.start_stage("aggregation", [self.name])
        reporter.start_unit(self.name, 1)
        reporter.set_postfix("semantic similarity")

        # Create aggregations folder
        aggregation_path = os.path.join(self.derivatives_dir, 'aggregations')
        os.makedirs(aggregation_path, exist_ok=True)
        
        semantic_similarity_data = defaultdict(lambda: defaultdict(list))
        
        # Walk through all directories in derivatives
        for root, dirs, files in os.walk(self.derivatives_dir):
            # Skip the aggregations directory itself
            if 'aggregations' in root:
                continue
                
            # Filter out hidden/system files
            filtered_files = [f for f in files if not is_hidden_or_system_file(f)]
            for file in filtered_files:
                if not file.endswith('.csv'):
                    continue
                family = similarity_aggregation_family(file)
                if family is None:
                    continue
                    
                file_path = os.path.join(root, file)
                try:
                    participant_key = aggregation_unit_key(file_path, self.derivatives_dir)
                    self._process_semantic_similarity_file(
                        file_path, semantic_similarity_data[participant_key][family]
                    )
                except Exception as e:
                    reporter.warn(f"Error processing {file_path}: {e}")
                    continue
        
        # Create comprehensive aggregation
        if semantic_similarity_data:
            self._create_semantic_similarity_aggregation(semantic_similarity_data, aggregation_path)
        else:
            debug_print("No semantic similarity results to aggregate")
        reporter.advance_item("semantic similarity")
        reporter.finish_unit()
    
    def _process_semantic_similarity_file(self, file_path, data_list):
        """Process a semantic similarity CSV file (single or multi-section) and add per-section dicts to data list."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw = f.read()
        except Exception:
            return

        # Some files contain multiple sections separated by 'New Section' with repeated headers
        if 'New Section' in raw:
            sections = [s.strip() for s in raw.split('New Section')]
            for section in sections:
                if not section:
                    continue
                # Ensure section starts with header
                if not section.startswith('Metric,'):
                    # Try to find header within the section text
                    header_pos = section.find('Metric,Similarity_Score')
                    if header_pos == -1:
                        continue
                    section = section[header_pos:]
                try:
                    df = pd.read_csv(io.StringIO(section))
                except Exception:
                    continue
                if 'Metric' not in df.columns or 'Similarity_Score' not in df.columns:
                    continue
                file_data = {}
                for _, row in df.iterrows():
                    metric = row.get('Metric')
                    if pd.isna(metric) or metric == 'Metric':
                        continue
                    score = pd.to_numeric(row.get('Similarity_Score'), errors='coerce')
                    if pd.isna(score):
                        continue
                    file_data[metric] = score
                if file_data:
                    data_list.append(file_data)
        else:
            # Standard single-section file
            try:
                df = pd.read_csv(file_path)
            except Exception:
                return
            if 'Metric' in df.columns and 'Similarity_Score' in df.columns:
                file_data = {}
                for _, row in df.iterrows():
                    metric = row.get('Metric')
                    if pd.isna(metric) or metric == 'Metric':
                        continue
                    score = pd.to_numeric(row.get('Similarity_Score'), errors='coerce')
                    if pd.isna(score):
                        continue
                    file_data[metric] = score
                if file_data:
                    data_list.append(file_data)
    
    def _create_semantic_similarity_aggregation(self, semantic_similarity_data, aggregation_path):
        """Create comprehensive semantic similarity aggregation."""
        aggregated_results = {}
        
        for participant, families in semantic_similarity_data.items():
            participant_results = {}
            for family in sorted(families):
                rows = families[family]
                if not rows:
                    continue
                if family == "sentence":
                    participant_results.update(self._aggregate_sentence_data(rows))
                else:
                    participant_results.update(self._aggregate_window_data(rows, family))
            aggregated_results[participant] = participant_results
        
        # Save aggregated results
        if aggregated_results:
            output_file = os.path.join(aggregation_path, f'{self.name}_semantic-similarity_comprehensive_aggregation.csv')
            df = pd.DataFrame(aggregated_results).T
            df.to_csv(output_file)
            debug_print(f"Comprehensive semantic similarity aggregation saved to: {output_file}")
    
    def _aggregate_window_data(self, window_data_list, window_name):
        """Aggregate window-based semantic similarity data."""
        results = {}
        
        debug_print(f"\n[_aggregate_window_data] === START: window_name={window_name} ===")
        debug_print(f"[_aggregate_window_data] Number of files to aggregate: {len(window_data_list)}")
        
        if not window_data_list:
            debug_print(f"[_aggregate_window_data] WARNING: Empty window_data_list, returning empty results")
            return results
        
        # Extract all metrics from all files for this participant
        all_metrics = {}
        for file_idx, file_data in enumerate(window_data_list):
            debug_print(f"[_aggregate_window_data] Processing file {file_idx+1}/{len(window_data_list)}: {len(file_data)} metrics")
            for metric, value in file_data.items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)
                debug_print(f"  [_aggregate_window_data] Metric '{metric}': value={value} (NaN: {pd.isna(value)})")
        
        debug_print(f"[_aggregate_window_data] Total unique metrics: {len(all_metrics)}")
        debug_print(f"[_aggregate_window_data] Metrics: {list(all_metrics.keys())}")
        
        # Calculate aggregations
        for metric, values in all_metrics.items():
            debug_print(f"\n[_aggregate_window_data] Processing metric: '{metric}'")
            debug_print(f"  [_aggregate_window_data] Total values: {len(values)}")
            debug_print(f"  [_aggregate_window_data] Raw values: {values}")
            
            # Filter out NaN values
            valid_values = [v for v in values if not pd.isna(v)]
            nan_count = len(values) - len(valid_values)
            
            debug_print(f"  [_aggregate_window_data] Valid values: {len(valid_values)}, NaN values: {nan_count}")
            debug_print(f"  [_aggregate_window_data] Valid values list: {valid_values}")
            
            if valid_values:
                # Average per window over all windows
                avg_value = np.mean(valid_values)
                debug_print(f"  [_aggregate_window_data] Calculated average: {avg_value}")
                
                results[f'{window_name}_avg_per_window_{metric}'] = avg_value
                results[f'{window_name}_avg_per_sentence_{metric}'] = avg_value
                results[f'{window_name}_avg_per_response_{metric}'] = avg_value
                
                debug_print(f"  [_aggregate_window_data] Set result key: '{window_name}_avg_per_window_{metric}' = {avg_value}")
            else:
                debug_print(f"  [_aggregate_window_data] WARNING: All values are NaN for metric '{metric}'!")
                debug_print(f"  [_aggregate_window_data] NOT setting result key (will cause <null> in output)")
        
        debug_print(f"[_aggregate_window_data] Final results keys: {list(results.keys())}")
        debug_print(f"[_aggregate_window_data] === END: window_name={window_name} ===\n")
        
        return results
    
    def _aggregate_sentence_data(self, sentence_data_list):
        """Aggregate sentence-level semantic similarity data."""
        results = {}
        
        if not sentence_data_list:
            return results
        
        # Extract all metrics from all files for this participant
        all_metrics = {}
        for file_data in sentence_data_list:
            for metric, value in file_data.items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)
        
        # Calculate aggregations
        for metric, values in all_metrics.items():
            # Filter out NaN values
            valid_values = [v for v in values if not pd.isna(v)]
            
            if valid_values:
                # Average over all sentences of a participant
                results[f'sentence_avg_over_all_sentences_{metric}'] = np.mean(valid_values)
                
                # Average per response over all sentences
                # (This is the same as average over all sentences since each file represents one response/section)
                results[f'sentence_avg_per_response_{metric}'] = np.mean(valid_values)
        
        return results

    def extract_logits(self):
        from pelican_nlp.extraction.metric_registry import run_logits

        run_logits(self)

    def extract_perplexity(self):
        from pelican_nlp.extraction.metric_registry import run_perplexity

        run_perplexity(self)

    def extract_embeddings(self):
        from pelican_nlp.extraction.metric_registry import run_embeddings

        run_embeddings(self)

    def extract_topic_modeling(self):
        from pelican_nlp.extraction.metric_registry import run_topic_modeling

        run_topic_modeling(self)

    def transcribe_audio(self, skip_existing: bool = False):
        """
        Transcribes audio files using the transcription pipeline.
        Saves transcription results to derivatives/transcription/ subdirectory.
        
        :param skip_existing: If True, skip files that already have transcription results.
        """
        from pelican_nlp.extras import require_extra
        from pelican_nlp.preprocessing.transcription import (
            AudioTranscriber,
            process_single_audio_file,
            release_transcription_models,
        )
        import os
        from pathlib import Path

        require_extra("transcription")
        from pelican_nlp.utils.progress import walk_units

        reporter = self.reporter
        transcription_dir = os.path.join(self.derivatives_dir, 'transcription')
        os.makedirs(transcription_dir, exist_ok=True)

        transcription_config = self.config.get('transcription', {})
        hf_token = transcription_config.get('hf_token', '')
        if not hf_token:
            reporter.warn(
                "No Hugging Face token provided. Speaker diarization will not work. "
                "Add hf_token to the transcription section of your config."
            )

        num_speakers = transcription_config.get('num_speakers', self.config.get('number_of_speakers', 2))
        min_silence_len = transcription_config.get('min_silence_len', 1000)
        silence_thresh = transcription_config.get('silence_thresh', -30)
        min_length = transcription_config.get('min_length', 90000)
        max_length = transcription_config.get('max_length', 150000)
        timestamp_source = transcription_config.get('timestamp_source', 'whisper_alignments')
        transcription_model = transcription_config.get('transcription_model', None)
        if isinstance(transcription_model, str):
            transcription_model = transcription_model.strip() or None

        diarizer_params = transcription_config.get('diarizer_params', {
            "segmentation": {
                "min_duration_off": 0.0,
            },
            "clustering": {
                "method": "centroid",
                "min_cluster_size": 12,
                "threshold": 0.8,
            }
        })
        debug_print(
            "Transcription settings: speakers=%s silence=%sms/%sdBFS chunks=%s-%sms source=%s model=%s"
            % (
                num_speakers,
                min_silence_len,
                silence_thresh,
                min_length,
                max_length,
                timestamp_source,
                transcription_model or "default",
            )
        )

        normalized_audio_dir = os.path.join(self.derivatives_dir, 'normalized-audio')
        os.makedirs(normalized_audio_dir, exist_ok=True)

        import gc
        import torch

        skipped_count = 0
        transcriber = aligner = diarizer = None
        for _unit, docs in walk_units(reporter, self.documents, "transcription"):
            for document in docs:
                reporter.set_postfix(getattr(document, "name", "") or str(document.file))
                if not (hasattr(document, 'file') and document.file):
                    reporter.warn("No audio file found for document")
                    reporter.advance_item(getattr(document, "name", "?"))
                    continue

                if not os.path.exists(document.file):
                    reporter.warn(f"Error: Audio file not found at {document.file}")
                    reporter.advance_item(document.name)
                    continue

                transcription_file = os.path.join(
                    transcription_dir,
                    f"{Path(document.file).stem}_allOutputs.json",
                )
                transcription_text_file = os.path.join(
                    transcription_dir,
                    f"{Path(document.file).stem}_transcript.txt",
                )

                if skip_existing and os.path.exists(transcription_file) and os.path.exists(transcription_text_file):
                    debug_print(f"Transcription already exists for {document.file}. Skipping...")
                    skipped_count += 1
                    document.transcription_file = transcription_file
                    document.transcription_text_file = transcription_text_file
                    reporter.advance_item(document.name)
                    continue

                try:
                    document._normalized_audio_dir = normalized_audio_dir

                    if transcriber is None:
                        reporter.set_postfix("loading transcription models")
                        if transcription_model:
                            transcriber = AudioTranscriber(model=transcription_model)
                        else:
                            transcriber = AudioTranscriber()

                    processed_document = process_single_audio_file(
                        audio_file=document,
                        hf_token=hf_token,
                        diarizer_params=diarizer_params,
                        num_speakers=num_speakers,
                        min_silence_len=min_silence_len,
                        silence_thresh=silence_thresh,
                        min_length=min_length,
                        max_length=max_length,
                        timestamp_source=timestamp_source,
                        transcription_model=transcription_model,
                        transcriber=transcriber,
                        aligner=None,
                        diarizer=None,
                        release_models=False,
                    )

                    processed_document.save_as_json(transcription_file)
                    processed_document.save_as_text(transcription_text_file)
                    document.transcription_file = transcription_file
                    document.transcription_text_file = transcription_text_file
                    debug_print(f"Transcription saved to: {transcription_file}")

                    if hasattr(processed_document, 'clear_audio_data'):
                        processed_document.clear_audio_data()
                    else:
                        if hasattr(processed_document, 'audio'):
                            processed_document.audio = None
                        if hasattr(processed_document, 'chunks'):
                            for chunk in processed_document.chunks:
                                if hasattr(chunk, 'audio_segment'):
                                    chunk.audio_segment = None
                    del processed_document

                except Exception as e:
                    reporter.warn(f"Error transcribing {document.file}: {e}")
                    import traceback
                    debug_print(traceback.format_exc())
                    if hasattr(document, 'clear_audio_data'):
                        document.clear_audio_data()
                    elif hasattr(document, 'audio'):
                        document.audio = None
                        if hasattr(document, 'chunks'):
                            document.chunks = []

                reporter.advance_item(document.name)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

        if transcriber is not None:
            release_transcription_models(transcriber, aligner, diarizer)
        debug_print(
            f"Audio transcription completed. Processed {len(self.documents) - skipped_count} files, "
            f"skipped {skipped_count}."
        )

    def extract_opensmile_features(self):
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp

        from pelican_nlp.extraction.acoustic_feature_extraction import (
            AudioFeatureExtraction,
            opensmile_job,
        )
        from pelican_nlp.extras import require_extra
        from pelican_nlp.utils.gpu_budget import cpu_worker_count
        from pelican_nlp.utils.progress import documents_by_unit, walk_units

        require_extra("acoustic")
        reporter = self.reporter
        grouped = documents_by_unit(self.documents)
        ordered = [document for docs in grouped.values() for document in docs]
        jobs = [
            (document.file, self.config['opensmile_configurations'])
            for document in ordered
        ]
        workers = cpu_worker_count()
        results_iter = None
        pool = None
        try:
            if workers > 1 and len(jobs) > 1:
                ctx = mp.get_context("spawn")
                pool = ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=ctx)
                results_iter = iter(pool.map(opensmile_job, jobs))
            else:
                results_iter = iter(
                    AudioFeatureExtraction.opensmile_extraction(*job) for job in jobs
                )
            for _unit, docs in walk_units(reporter, ordered, "opensmile"):
                for document in docs:
                    reporter.set_postfix(document.name)
                    results, recording_length = next(results_iter)
                    document.recording_length = recording_length
                    results['participant_ID'] = document.participant_ID
                    store_features_to_csv(
                        results,
                        self.derivatives_dir,
                        document,
                        metric='opensmile-features',
                    )
                    reporter.advance_item(document.name)
        finally:
            if pool is not None:
                pool.shutdown(wait=True)

    def extract_prosogram(self):
        from pelican_nlp.extraction.acoustic_feature_extraction import AudioFeatureExtraction
        from pelican_nlp.extras import require_extra
        from pelican_nlp.utils.progress import walk_units

        require_extra("acoustic")
        reporter = self.reporter
        for _unit, docs in walk_units(reporter, self.documents, "prosogram"):
            for document in docs:
                reporter.set_postfix(document.name)
                output_dir = os.path.join(
                    self.derivatives_dir,
                    'prosogram-features',
                    unit_folder_for_document(document),
                )
                AudioFeatureExtraction.extract_prosogram_profile(
                    document.file,
                    output_dir=output_dir,
                )
                reporter.advance_item(document.name)

    def create_document_information_csv(self):
        """Create CSV file with summarized document parameters based on config specifications."""
        reporter = self.reporter
        reporter.start_stage("document-information", [self.name], label="document information")
        reporter.start_unit(self.name, 1)

        doc_info_path = os.path.join(self.derivatives_dir, 'aggregations', 'document_information')
        os.makedirs(doc_info_path, exist_ok=True)
        output_file = os.path.join(doc_info_path, f'{self.name}_document-information.csv')
        parameters_to_include = self.config.get('document_information_output', {}).get('parameters', [])

        if not parameters_to_include:
            reporter.warn("No parameters specified in config for document information output")
            reporter.advance_item("skip")
            reporter.finish_unit()
            return

        document_info = []
        for doc in self.documents:
            attrs = vars(doc)
            info = {
                param: attrs.get(param)
                for param in parameters_to_include
                if param in attrs
            }
            document_info.append(info)

        df = pd.DataFrame(document_info)
        df.to_csv(output_file, index=False)
        debug_print(f"Document information saved to: {output_file}")
        reporter.advance_item(self.name)
        reporter.finish_unit()