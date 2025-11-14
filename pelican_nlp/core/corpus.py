"""
This module provides the Corpus class, which aggregates documents where the same processing
steps applied and results should be aggregated.
(e.g. all fluency files from task 'animals' or all image-descriptions from the same image)

This class contains the pipelines for homogenous processing and metric extraction of all grouped files.
"""

import os
import io
import pandas as pd
import numpy as np
from ..preprocessing import TextPreprocessingPipeline
from ..utils.csv_functions import store_features_to_csv
from ..extraction.language_model import Model
from ..preprocessing.speaker_diarization import TextDiarizer
from ..preprocessing import text_cleaner as textcleaner
import re

from pelican_nlp.config import debug_print
from pelican_nlp.utils.setup_functions import is_hidden_or_system_file

class Corpus:
    def __init__(self, corpus_name, documents, configuration_settings, project_folder):
        self.name = corpus_name
        self.key = corpus_name.split('-')[0]
        self.value = corpus_name.split('-')[1]
        self.documents = documents
        self.config = configuration_settings
        self.project_folder = project_folder
        self.derivatives_dir = project_folder / 'derivatives'
        self.pipeline = TextPreprocessingPipeline(self.config)
        self.task = configuration_settings['task_name']
        self.results_path = None

    def preprocess_all_documents(self):
        print("preprocessing all documents")
        for document in self.documents:
            document.detect_sections()
            document.process_document(self.pipeline)

    def get_all_processed_texts(self):
        result = {}
        for participant in self.documents:
            result[participant.name] = participant.get_processed_texts()
        return result

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
                    participant_key = os.path.basename(file).split('_')[0]
                    
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
        from pelican_nlp.extraction.extract_logits import LogitsExtractor
        from pelican_nlp.preprocessing.text_tokenizer import TextTokenizer

        print("Extracting Logits...")

        logits_options = self.config['options_logits']

        model_name = logits_options['model_name']
        logitsExtractor = LogitsExtractor(logits_options,
                                          self.pipeline,
                                          self.project_folder)
        model = Model(model_name, self.project_folder)
        model.load_model()
        model_instance = model.model_instance
        tokenizer = TextTokenizer(logits_options['tokenization_method'], model_name=logits_options['model_name'])
        for i in range(len(self.documents)):

            for key, section in self.documents[i].cleaned_sections.items():

                if self.config['discourse'] == True:
                    # Handle both single speaker tag and multiple speaker tags
                    participant_speakertag = self.config['participant_speakertag']
                    section = TextDiarizer.parse_speaker(section, participant_speakertag,
                                                         logits_options['keep_speakertags'])
                    #print(f'parsed section is {section}')
                else:
                    section = [section]


                for part in section:
                    logits = logitsExtractor.extract_features(part, tokenizer, model_instance)
                    self.documents[i].logits.append(logits)

                    #'logits' list of dictionaries; keys token, logprob_actual, logprob_max, entropy, most_likely_token
                    store_features_to_csv(logits,
                                          self.derivatives_dir,
                                          self.documents[i],
                                          metric='logits')

    def extract_perplexity(self):
        """Extract perplexity metrics from logits data."""
        from pelican_nlp.extraction.extract_perplexity import PerplexityExtractor

        print("Extracting Perplexity...")

        perplexity_options = self.config['options_perplexity']
        perplexityExtractor = PerplexityExtractor(perplexity_options, self.project_folder)

        for i in range(len(self.documents)):
            # Process each logits entry for this document
            for logits_data in self.documents[i].logits:
                perplexityExtractor.extract_perplexity_from_document(self.documents[i], logits_data)

    def extract_embeddings(self):
        from pelican_nlp.extraction.extract_embeddings import EmbeddingsExtractor

        print("Extracting Embeddings...")

        embedding_options = self.config['options_embeddings']
        embeddingsExtractor = EmbeddingsExtractor(embedding_options, self.project_folder)
        debug_print(len(self.documents))
        for i in range(len(self.documents)):

            debug_print(f'cleaned sections: {self.documents[i].cleaned_sections}')
            for key, section in self.documents[i].cleaned_sections.items():
                debug_print(f'Processing section {key}')
                
                if self.config['discourse']:
                    # Handle both single speaker tag and multiple speaker tags
                    participant_speakertag = self.config['participant_speakertag']
                    section = TextDiarizer.parse_speaker(section, participant_speakertag, embedding_options['keep_speakertags'])
                else:
                    section = [section]

                embeddings, token_count = embeddingsExtractor.extract_embeddings_from_text(section, embedding_options)
                self.documents[i].embeddings.append(embeddings)

                if self.task == 'fluency':
                    self.documents[i].fluency_word_count = token_count
                
                for utterance in embeddings:

                    if self.config['options_embeddings']['semantic-similarity']:
                        from pelican_nlp.extraction.semantic_similarity import calculate_semantic_similarity, \
                            get_semantic_similarity_windows
                        consecutive_similarities, mean_similarity = calculate_semantic_similarity(utterance)
                        debug_print(f'Mean semantic similarity: {mean_similarity:.4f}')

                        for window_size in self.config['options_semantic-similarity']['window_sizes']:
                            debug_print(f'\n[extract_embeddings] Processing window_size={window_size} for document: {self.documents[i].name}')
                            window_stats = get_semantic_similarity_windows(utterance, window_size)
                            
                            if isinstance(window_stats, tuple) and len(window_stats) == 5:
                                window_data = {
                                    'mean_of_window_means': window_stats[0],
                                    'std_of_window_means': window_stats[1],
                                    'mean_of_window_stds': window_stats[2],
                                    'std_of_window_stds': window_stats[3],
                                    'mean_of_window_medians': window_stats[4]
                                }
                                debug_print(f'[extract_embeddings] Window {window_size} stats - mean: {window_stats[0]:.4f}, std: {window_stats[1]:.4f}, median: {window_stats[4]:.4f}')
                                debug_print(f'[extract_embeddings] Window {window_size} data to store: {window_data}')
                                
                                # Check for NaN values
                                nan_metrics = [k for k, v in window_data.items() if pd.isna(v)]
                                if nan_metrics:
                                    debug_print(f'[extract_embeddings] WARNING: Window {window_size} has NaN values for metrics: {nan_metrics}')
                            else:
                                window_data = {
                                    'mean': window_stats[0] if isinstance(window_stats, tuple) else window_stats,
                                    'std': window_stats[1] if isinstance(window_stats, tuple) and len(window_stats) > 1 else None
                                }
                                debug_print(f'[extract_embeddings] Window {window_size} data (non-standard format): {window_data}')
                            
                            debug_print(f'[extract_embeddings] Storing window {window_size} data to CSV...')
                            store_features_to_csv(window_data,
                                                  self.derivatives_dir,
                                                  self.documents[i],
                                                  metric=f'semantic-similarity-window-{window_size}')
                            debug_print(f'[extract_embeddings] Stored window {window_size} data to CSV')
                        
                        # Calculate and store sentence-level semantic similarity
                        sentence_stats = get_semantic_similarity_windows(utterance, 'sentence')
                        if isinstance(sentence_stats, tuple) and len(sentence_stats) == 5:
                            sentence_data = {
                                'mean_of_window_means': sentence_stats[0],
                                'std_of_window_means': sentence_stats[1],
                                'mean_of_window_stds': sentence_stats[2],
                                'std_of_window_stds': sentence_stats[3],
                                'mean_of_window_medians': sentence_stats[4]
                            }
                            debug_print(f'Sentence similarity stats - mean: {sentence_stats[0]:.4f}, std: {sentence_stats[1]:.4f}, median: {sentence_stats[4]:.4f}')
                            
                            store_features_to_csv(sentence_data,
                                                  self.derivatives_dir,
                                                  self.documents[i],
                                                  metric='semantic-similarity-sentence')

                    if self.config['options_embeddings']['distance-from-randomness']:
                        from pelican_nlp.extraction.distance_from_randomness import get_distance_from_randomness
                        divergence = get_distance_from_randomness(utterance, self.config["options_dis_from_randomness"])
                        debug_print(f'Divergence from optimality metrics: {divergence}')
                        store_features_to_csv(divergence,
                                              self.derivatives_dir,
                                              self.documents[i],
                                              metric='distance-from-randomness')

                    # Process tokens
                    if embedding_options['clean_embedding_tokens']:
                        cleaned_embeddings = []
                        if isinstance(utterance, dict):
                            # Handle dictionary case (PyTorch models)
                            for token, embedding in utterance.items():
                                if 'xlm-roberta-base' in self.config['options_embeddings']['model_name'].lower():
                                    cleaned_token = textcleaner.clean_subword_token_RoBERTa(token)
                                else:
                                    cleaned_token = textcleaner.clean_token_generic(token)
                                if cleaned_token is not None:
                                    cleaned_embeddings.append((cleaned_token, embedding))
                        else:
                            # Handle list of tuples case (fastText)
                            for token, embedding in utterance:
                                cleaned_token = textcleaner.clean_token_generic(token)
                                if cleaned_token is not None:
                                    cleaned_embeddings.append((cleaned_token, embedding))
                    else:
                        cleaned_embeddings = utterance if isinstance(utterance, list) else [(k, v) for k, v in utterance.items()]

                    # Only store embeddings if they are not empty
                    if cleaned_embeddings:
                        store_features_to_csv(cleaned_embeddings,
                                              self.derivatives_dir,
                                              self.documents[i],
                                              metric='embeddings')
        return

    def transcribe_audio(self):
        """
        Transcribes audio files using the transcription pipeline.
        Saves transcription results to derivatives/transcription/ subdirectory.
        """
        from pelican_nlp.preprocessing.transcription import process_single_audio_file
        import os
        from pathlib import Path
        
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
        
        # Process each audio document
        for i, document in enumerate(self.documents):
            if hasattr(document, 'file') and document.file:
                print(f"\nProcessing document {i+1}/{len(self.documents)}: {document.file}")
                
                # Check if audio file exists
                if not os.path.exists(document.file):
                    print(f"Error: Audio file not found at {document.file}")
                    continue
                
                try:
                    # Process the single audio file
                    processed_document = process_single_audio_file(
                        audio_file=document,
                        hf_token=hf_token,
                        diarizer_params=diarizer_params,
                        num_speakers=num_speakers,
                        min_silence_len=min_silence_len,
                        silence_thresh=silence_thresh,
                        min_length=min_length,
                        max_length=max_length,
                        timestamp_source=timestamp_source
                    )
                    
                    # Save transcription results to JSON file
                    transcription_file = os.path.join(
                        transcription_dir, 
                        f"{Path(document.file).stem}_all_outputs.json"
                    )
                    processed_document.save_as_json(transcription_file)
                    
                    # Store transcription file path in the document
                    document.transcription_file = transcription_file
                    
                    print(f"Transcription completed and saved to: {transcription_file}")
                    
                except Exception as e:
                    print(f"Error transcribing {document.file}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            else:
                print(f"No audio file found for document {i}")
        
        print(f"\nAudio transcription completed. Processed {len(self.documents)} documents.")

    def extract_opensmile_features(self):
        from pelican_nlp.extraction.acoustic_feature_extraction import AudioFeatureExtraction

        print("Extracting openSMILE features...")

        for i in range(len(self.documents)):
            results, recording_length = AudioFeatureExtraction.opensmile_extraction(self.documents[i].file, self.config['opensmile_configurations'])
            self.documents[i].recording_length = recording_length  # Store the recording length
            results['participant_ID'] = self.documents[i].participant_ID  # Set the participant ID
            store_features_to_csv(results,
                                  self.derivatives_dir,
                                  self.documents[i],
                                  metric='opensmile-features')

    def extract_prosogram(self):
        from pelican_nlp.extraction.acoustic_feature_extraction import AudioFeatureExtraction
        from pelican_nlp.utils.csv_functions import store_features_to_csv

        print("Extracting Prosogram...")

        for i in range(len(self.documents)):
            # Create the output directory for this document's prosogram files
            output_dir = os.path.join(self.derivatives_dir, 'prosogram-features', 
                                    f"part-{self.documents[i].participant_ID}")
            
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