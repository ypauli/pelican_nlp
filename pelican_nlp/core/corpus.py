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
import pandas as pd
import numpy as np


def _corpus_key_value(corpus_name):
    name = str(corpus_name)
    if '-' in name:
        key, value = name.split('-', 1)
        return key, value
    return 'unit', name


class Corpus:
    def __init__(self, corpus_name, documents, configuration_settings, project_folder):
        self.name = corpus_name
        self.key, self.value = _corpus_key_value(corpus_name)
        self.documents = documents
        self.config = configuration_settings
        self.project_folder = project_folder
        self.derivatives_dir = project_folder / 'derivatives'
        self.pipeline = TextPreprocessingPipeline(self.config)
        self.task = configuration_settings.get('task_name')

    def preprocess_all_documents(self):
        print("preprocessing all documents")
        for document in self.documents:
            document.detect_sections()
            document.process_document(self.pipeline)

    def create_corpus_results_consolidation_csv(self) -> None:
        """Create comprehensive aggregated results CSV files for semantic similarity metrics."""
        
        # Create aggregations folder
        aggregation_path = os.path.join(self.derivatives_dir, 'aggregations')
        os.makedirs(aggregation_path, exist_ok=True)
        
        # Initialize semantic similarity aggregation data
        semantic_similarity_data = {}
        
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
                    
                file_path = os.path.join(root, file)
                try:
                    participant_key = aggregation_unit_key(file_path, self.derivatives_dir)
                    
                    # Initialize participant dict if not exists
                    if participant_key not in semantic_similarity_data:
                        semantic_similarity_data[participant_key] = {
                            'window_2_data': [],
                            'window_8_data': [],
                            'sentence_data': []
                        }
                    
                    # Process semantic similarity files
                    if 'semantic-similarity-window-2' in file:
                        self._process_semantic_similarity_file(file_path, semantic_similarity_data[participant_key]['window_2_data'])
                    elif 'semantic-similarity-window-8' in file:
                        self._process_semantic_similarity_file(file_path, semantic_similarity_data[participant_key]['window_8_data'])
                    elif ('semantic-similarity-sentence' in file) or ('semantic-similarity-window-sentence' in file):
                        self._process_semantic_similarity_file(file_path, semantic_similarity_data[participant_key]['sentence_data'])

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    continue
        
        # Create comprehensive aggregation
        if semantic_similarity_data:
            self._create_semantic_similarity_aggregation(semantic_similarity_data, aggregation_path)
        else:
            print("No semantic similarity results to aggregate")
    
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
        
        for participant, data in semantic_similarity_data.items():
            participant_results = {}
            
            # Process window 2 data
            if data['window_2_data']:
                participant_results.update(self._aggregate_window_data(data['window_2_data'], 'window_2'))
            
            # Process window 8 data
            if data['window_8_data']:
                participant_results.update(self._aggregate_window_data(data['window_8_data'], 'window_8'))
            
            # Process sentence data
            if data['sentence_data']:
                participant_results.update(self._aggregate_sentence_data(data['sentence_data']))
            
            aggregated_results[participant] = participant_results
        
        # Save aggregated results
        if aggregated_results:
            output_file = os.path.join(aggregation_path, f'{self.name}_semantic-similarity_comprehensive_aggregation.csv')
            df = pd.DataFrame(aggregated_results).T
            df.to_csv(output_file)
            print(f"Comprehensive semantic similarity aggregation saved to: {output_file}")
    
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
            ForcedAligner,
            SpeakerDiarizer,
            process_single_audio_file,
            release_transcription_models,
        )
        import os
        from pathlib import Path

        require_extra("transcription")
        
        print("Starting audio transcription...")
        
        # Create transcription subdirectory in derivatives
        transcription_dir = os.path.join(self.derivatives_dir, 'transcription')
        os.makedirs(transcription_dir, exist_ok=True)
        
        # Get transcription parameters from config
        transcription_config = self.config.get('transcription', {})
        
        # Use configuration values with fallbacks to existing config or defaults
        hf_token = transcription_config.get('hf_token', '')
        if not hf_token:
            print("Warning: No Hugging Face token provided. Speaker diarization will not work.")
            print("Please add 'hf_token: your_token_here' to the transcription section of your config.")
        
        num_speakers = transcription_config.get('num_speakers', self.config.get('number_of_speakers', 2))
        min_silence_len = transcription_config.get('min_silence_len', 1000)
        silence_thresh = transcription_config.get('silence_thresh', -30)
        min_length = transcription_config.get('min_length', 90000)
        max_length = transcription_config.get('max_length', 150000)
        timestamp_source = transcription_config.get('timestamp_source', 'whisper_alignments')
        transcription_model = transcription_config.get('transcription_model', None)
        if isinstance(transcription_model, str):
            transcription_model = transcription_model.strip() or None
        
        # Get diarization parameters from config
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
        
        print(f"Transcription settings:")
        print(f"  - Number of speakers: {num_speakers}")
        print(f"  - Min silence length: {min_silence_len}ms")
        print(f"  - Silence threshold: {silence_thresh}dBFS")
        print(f"  - Chunk length range: {min_length}-{max_length}ms")
        print(f"  - Timestamp source: {timestamp_source}")
        if transcription_model:
            print(f"  - Transcription model: {transcription_model}")
        else:
            print("  - Transcription model: default")
        
        # Create normalized audio subdirectory in derivatives
        normalized_audio_dir = os.path.join(self.derivatives_dir, 'normalized-audio')
        os.makedirs(normalized_audio_dir, exist_ok=True)
        
        # Import garbage collection and memory monitoring
        import gc
        import torch
        
        # Process each audio document
        skipped_count = 0
        transcriber = aligner = diarizer = None
        for i, document in enumerate(self.documents):
            if hasattr(document, 'file') and document.file:
                print(f"\nProcessing document {i+1}/{len(self.documents)}: {document.file}")
                
                if not os.path.exists(document.file):
                    print(f"Error: Audio file not found at {document.file}")
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
                    print(f"Transcription already exists for {document.file}. Skipping...")
                    skipped_count += 1
                    document.transcription_file = transcription_file
                    document.transcription_text_file = transcription_text_file
                    continue
                
                try:
                    document._normalized_audio_dir = normalized_audio_dir

                    if transcriber is None:
                        print("Initializing processing classes...")
                        if transcription_model:
                            print(f"Using custom transcription model: {transcription_model}")
                            transcriber = AudioTranscriber(model=transcription_model)
                        else:
                            transcriber = AudioTranscriber()
                        aligner = ForcedAligner()
                        diarizer = SpeakerDiarizer(hf_token, parameters=diarizer_params)

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
                        aligner=aligner,
                        diarizer=diarizer,
                        release_models=False,
                    )

                    processed_document.save_as_json(transcription_file)
                    processed_document.save_as_text(transcription_text_file)
                    document.transcription_file = transcription_file
                    document.transcription_text_file = transcription_text_file
                    
                    print(f"Transcription completed and saved to: {transcription_file}")
                    print(f"Transcript text saved to: {transcription_text_file}")
                    
                    # Explicitly clear large audio data from memory after saving
                    # This prevents memory accumulation across multiple files
                    if hasattr(processed_document, 'clear_audio_data'):
                        processed_document.clear_audio_data()
                    else:
                        # Fallback: manual cleanup if method doesn't exist
                        if hasattr(processed_document, 'audio'):
                            processed_document.audio = None
                        if hasattr(processed_document, 'chunks'):
                            for chunk in processed_document.chunks:
                                if hasattr(chunk, 'audio_segment'):
                                    chunk.audio_segment = None
                    
                    # Clear processed_document reference
                    del processed_document
                    
                except Exception as e:
                    print(f"Error transcribing {document.file}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Clear any partial data on error
                    if hasattr(document, 'clear_audio_data'):
                        document.clear_audio_data()
                    elif hasattr(document, 'audio'):
                        document.audio = None
                        if hasattr(document, 'chunks'):
                            document.chunks = []
                    continue
            else:
                print(f"No audio file found for document {i}")
            
            # Force garbage collection and clear GPU cache after each file
            # This prevents memory accumulation that can lead to OOM kills
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # Synchronize to ensure cache clearing is complete
                torch.cuda.synchronize()
            
            # Print memory status every 10 files for monitoring
            if (i + 1) % 10 == 0:
                try:
                    import psutil
                    process = psutil.Process()
                    mem_info = process.memory_info()
                    mem_gb = mem_info.rss / (1024 ** 3)
                    print(f"Memory usage after {i+1} files: {mem_gb:.2f} GB")
                    if torch.cuda.is_available():
                        gpu_mem_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                        gpu_mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                        print(f"GPU memory: {gpu_mem_allocated:.2f} GB allocated, {gpu_mem_reserved:.2f} GB reserved")
                except ImportError:
                    pass  # psutil not available, skip memory monitoring
        
        if transcriber is not None:
            release_transcription_models(transcriber, aligner, diarizer)

        processed_count = len(self.documents) - skipped_count
        if skip_existing and skipped_count > 0:
            print(f"\nAudio transcription completed. Processed {processed_count} new files, skipped {skipped_count} already transcribed files.")
        else:
            print(f"\nAudio transcription completed. Processed {processed_count} documents.")

    def extract_opensmile_features(self):
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp

        from pelican_nlp.extraction.acoustic_feature_extraction import (
            AudioFeatureExtraction,
            opensmile_job,
        )
        from pelican_nlp.extras import require_extra
        from pelican_nlp.utils.gpu_budget import cpu_worker_count

        require_extra("acoustic")
        print("Extracting openSMILE features...")
        jobs = [
            (self.documents[i].file, self.config['opensmile_configurations'])
            for i in range(len(self.documents))
        ]
        workers = cpu_worker_count()
        if workers > 1 and len(jobs) > 1:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=ctx) as pool:
                raw = list(pool.map(opensmile_job, jobs))
        else:
            raw = [AudioFeatureExtraction.opensmile_extraction(*job) for job in jobs]

        for i, (results, recording_length) in enumerate(raw):
            self.documents[i].recording_length = recording_length
            results['participant_ID'] = self.documents[i].participant_ID
            store_features_to_csv(
                results,
                self.derivatives_dir,
                self.documents[i],
                metric='opensmile-features',
            )

    def extract_prosogram(self):
        from pelican_nlp.extraction.acoustic_feature_extraction import AudioFeatureExtraction
        from pelican_nlp.extras import require_extra

        require_extra("acoustic")
        print("Extracting Prosogram...")

        for i in range(len(self.documents)):
            # Create the output directory for this document's prosogram files
            output_dir = os.path.join(
                self.derivatives_dir,
                'prosogram-features',
                unit_folder_for_document(self.documents[i]),
            )
            
            results = AudioFeatureExtraction.extract_prosogram_profile(
                self.documents[i].file, 
                output_dir=output_dir
            )

    def create_document_information_csv(self):
        """Create CSV file with summarized document parameters based on config specifications."""
        
        # Create document_information folder inside aggregations
        doc_info_path = os.path.join(self.derivatives_dir, 'aggregations', 'document_information')
        os.makedirs(doc_info_path, exist_ok=True)
        
        # Define output file path
        output_file = os.path.join(doc_info_path, f'{self.name}_document-information.csv')
        
        # Get parameters to include from config
        parameters_to_include = self.config.get('document_information_output', {}).get('parameters', [])
        
        if not parameters_to_include:
            print("Warning: No parameters specified in config for document information output")
            return
        
        # Get document information based on specified parameters
        document_info = []
        for doc in self.documents:
            # Get all attributes using vars()
            attrs = vars(doc)
            # Filter based on specified parameters
            info = {
                param: attrs.get(param) 
                for param in parameters_to_include 
                if param in attrs
            }
            document_info.append(info)
        
        # Convert to DataFrame and save to CSV
        df = pd.DataFrame(document_info)
        df.to_csv(output_file, index=False)
        debug_print(f"Document information saved to: {output_file}")