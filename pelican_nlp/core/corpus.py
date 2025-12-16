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

    def create_documents_from_transcriptions(self):
        """
        Create Document instances from transcription text files for audio documents that have been transcribed.
        
        Returns:
            List of Document instances created from transcription files
        """
        from pelican_nlp.core.document import Document
        
        transcription_documents = []
        
        for audio_doc in self.documents:
            # Check if this is an AudioFile with a transcription text file
            if hasattr(audio_doc, 'transcription_text_file') and audio_doc.transcription_text_file:
                import os
                if os.path.exists(audio_doc.transcription_text_file):
                    try:
                        # Create Document from transcription file
                        transcription_doc = Document.from_transcription_file(
                            transcription_text_file=audio_doc.transcription_text_file,
                            origin_audio_file=audio_doc,
                            config=self.config
                        )
                        transcription_doc.corpus_name = self.name
                        transcription_documents.append(transcription_doc)
                        print(f"Created Document from transcription: {audio_doc.transcription_text_file}")
                    except Exception as e:
                        print(f"Error creating Document from transcription {audio_doc.transcription_text_file}: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"Warning: Transcription text file not found: {audio_doc.transcription_text_file}")
        
        return transcription_documents

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
        import torch

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
                    participant_speakertag = self.config.get('participant_speakertag')
                    if participant_speakertag is not None:
                        section = TextDiarizer.parse_speaker(section, participant_speakertag,
                                                             logits_options['keep_speakertags'])
                    else:
                        # If discourse is enabled but no speaker tag specified, treat as single section
                        section = [section]
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
        
        # Clean up model and free GPU memory
        # Note: Avoid moving models to CPU if they're dispatched across devices (can cause segfaults)
        # Just delete references and clear cache
        try:
            del model_instance
        except Exception:
            pass
        try:
            if hasattr(model, 'model_instance'):
                del model.model_instance
        except Exception:
            pass
        try:
            del model
        except Exception:
            pass
        try:
            del logitsExtractor
        except Exception:
            pass
        try:
            del tokenizer
        except Exception:
            pass
        
        # Clear GPU cache after logits extraction
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU memory cleared after logits extraction")

    def extract_perplexity(self):
        """Extract perplexity metrics from logits data."""
        from pelican_nlp.extraction.extract_perplexity import PerplexityExtractor

        print("Extracting Perplexity...")

        perplexity_options = self.config['options_perplexity']
        perplexityExtractor = PerplexityExtractor(perplexity_options, self.project_folder)

        for i in range(len(self.documents)):
            # Process each logits entry for this document
            # Each logits_data corresponds to one section (already separated at document level)
            section_idx = 0
            for logits_data in self.documents[i].logits:
                perplexityExtractor.extract_perplexity_from_document(
                    self.documents[i], logits_data, section_index=section_idx
                )
                section_idx += 1

    def extract_embeddings(self):
        from pelican_nlp.extraction.extract_embeddings import EmbeddingsExtractor
        import torch

        print("Extracting Embeddings...")

        embedding_options = self.config['options_embeddings']
        semantic_similarity_options = self.config.get('options_semantic-similarity', {})
        store_window_details = semantic_similarity_options.get('store_window_details', False)
        store_sentence_details = semantic_similarity_options.get('store_sentence_details', store_window_details)
        embeddingsExtractor = EmbeddingsExtractor(embedding_options, self.project_folder)
        debug_print(len(self.documents))
        for i in range(len(self.documents)):

            debug_print(f'cleaned sections: {self.documents[i].cleaned_sections}')
            for key, section in self.documents[i].cleaned_sections.items():
                debug_print(f'Processing section {key}')
                
                if self.config['discourse']:
                    # Handle both single speaker tag and multiple speaker tags
                    participant_speakertag = self.config.get('participant_speakertag')
                    if participant_speakertag is not None:
                        section = TextDiarizer.parse_speaker(section, participant_speakertag, embedding_options['keep_speakertags'])
                    else:
                        # If discourse is enabled but no speaker tag specified, treat as single section
                        section = [section]
                else:
                    section = [section]

                embeddings, token_count = embeddingsExtractor.extract_embeddings_from_text(section, embedding_options)
                self.documents[i].embeddings.append(embeddings)

                if self.task == 'fluency':
                    self.documents[i].fluency_word_count = token_count
                
                for utterance in embeddings:

                    if self.config['options_embeddings']['semantic-similarity']:
                        from pelican_nlp.extraction.semantic_similarity import calculate_semantic_similarity, \
                            get_semantic_similarity_windows, filter_punctuation_tokens
                        exclude_punctuation_tokens = semantic_similarity_options.get('exclude_punctuation_tokens', False)
                        similarity_utterance = filter_punctuation_tokens(utterance) if exclude_punctuation_tokens else utterance

                        consecutive_similarities, mean_similarity = calculate_semantic_similarity(similarity_utterance)
                        debug_print(f'Mean semantic similarity: {mean_similarity:.4f}')

                        for window_size in self.config['options_semantic-similarity']['window_sizes']:
                            # Skip 'sentence' here - it's handled separately below
                            if window_size == 'sentence':
                                continue
                            debug_print(f'\n[extract_embeddings] Processing window_size={window_size} for document: {self.documents[i].name}')
                            collect_details = store_window_details and window_size != 'sentence'
                            window_input = similarity_utterance if window_size != 'sentence' else utterance
                            window_result = get_semantic_similarity_windows(
                                window_input,
                                window_size,
                                return_details=collect_details,
                                exclude_punctuation=exclude_punctuation_tokens
                            )
                            
                            if collect_details:
                                window_stats, window_detail_rows = window_result
                            else:
                                window_stats = window_result
                                window_detail_rows = []
                            
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
                            
                            if collect_details and window_detail_rows:
                                debug_print(f'[extract_embeddings] Storing window {window_size} detail data ({len(window_detail_rows)} rows)')
                                store_features_to_csv(window_detail_rows,
                                                      self.derivatives_dir,
                                                      self.documents[i],
                                                      metric=f'semantic-similarity-window-details-{window_size}')
                                debug_print(f'[extract_embeddings] Stored window {window_size} detail data')
                        
                        # Calculate and store sentence-level semantic similarity
                        # Only if 'sentence' is in window_sizes (it was skipped in the loop above)
                        if 'sentence' in self.config['options_semantic-similarity']['window_sizes']:
                            debug_print(f'[extract_embeddings] Calculating sentence-level semantic similarity for document: {self.documents[i].name}')
                            sentence_result = get_semantic_similarity_windows(
                                utterance,
                                'sentence',
                                return_details=store_sentence_details,
                                exclude_punctuation=exclude_punctuation_tokens
                            )
                            if store_sentence_details:
                                sentence_stats, sentence_detail_rows = sentence_result
                            else:
                                sentence_stats = sentence_result
                                sentence_detail_rows = []
                            if isinstance(sentence_stats, tuple) and len(sentence_stats) == 5:
                                sentence_data = {
                                    'mean_of_window_means': sentence_stats[0],
                                    'std_of_window_means': sentence_stats[1],
                                    'mean_of_window_stds': sentence_stats[2],
                                    'std_of_window_stds': sentence_stats[3],
                                    'mean_of_window_medians': sentence_stats[4]
                                }
                                # Format debug output safely handling NaN values
                                mean_str = f'{sentence_stats[0]:.4f}' if not pd.isna(sentence_stats[0]) else 'NaN'
                                std_str = f'{sentence_stats[1]:.4f}' if not pd.isna(sentence_stats[1]) else 'NaN'
                                median_str = f'{sentence_stats[4]:.4f}' if not pd.isna(sentence_stats[4]) else 'NaN'
                                debug_print(f'Sentence similarity stats - mean: {mean_str}, std: {std_str}, median: {median_str}')
                                
                                store_features_to_csv(sentence_data,
                                                      self.derivatives_dir,
                                                      self.documents[i],
                                                      metric='semantic-similarity-sentence')
                            
                            if store_sentence_details and sentence_detail_rows:
                                debug_print(f'[extract_embeddings] Storing sentence similarity detail data ({len(sentence_detail_rows)} rows)')
                                store_features_to_csv(sentence_detail_rows,
                                                      self.derivatives_dir,
                                                      self.documents[i],
                                                      metric='semantic-similarity-sentence-details')

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
        
        # Clean up embeddings extractor and free GPU memory
        # Note: Avoid moving models to CPU if they're dispatched across devices (can cause segfaults)
        # Just delete references and clear cache
        try:
            if hasattr(embeddingsExtractor, 'model_instance'):
                del embeddingsExtractor.model_instance
        except Exception:
            pass
        try:
            if hasattr(embeddingsExtractor, 'model'):
                del embeddingsExtractor.model
        except Exception:
            pass
        try:
            del embeddingsExtractor
        except Exception:
            pass
        
        # Clear GPU cache after embeddings extraction
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU memory cleared after embeddings extraction")
        
        return

    def _extract_documents_and_embeddings_from_document(self, doc_idx):
        """
        Helper method to extract documents (sections/utterances) and embeddings from a single document.
        
        Returns:
            Tuple of (documents_list, embeddings_list) for the document
        """
        documents_list = []
        embeddings_list = []
        
        # Check if embeddings exist for this document
        if not hasattr(self.documents[doc_idx], 'embeddings') or not self.documents[doc_idx].embeddings:
            return documents_list, embeddings_list
        
        # Get embedding options for discourse parsing
        embedding_options = self.config.get('options_embeddings', {})
        
        # Iterate through sections
        section_idx = 0
        for key, section in self.documents[doc_idx].cleaned_sections.items():
            debug_print(f'Processing section {key}')
            
            # Handle discourse parsing similar to extract_embeddings()
            if self.config.get('discourse', False):
                participant_speakertag = self.config.get('participant_speakertag')
                if participant_speakertag is not None:
                    section_texts = TextDiarizer.parse_speaker(
                        section, participant_speakertag, 
                        embedding_options.get('keep_speakertags', False)
                    )
                else:
                    # If discourse is enabled but no speaker tag specified, treat as single section
                    section_texts = [section]
            else:
                section_texts = [section]
            
            # Get corresponding embeddings for this section
            if section_idx < len(self.documents[doc_idx].embeddings):
                section_embeddings = self.documents[doc_idx].embeddings[section_idx]
                
                # section_embeddings is a list of utterances (if discourse) or a single list
                # Match embeddings to section texts
                if isinstance(section_embeddings, list) and len(section_embeddings) > 0:
                    # Check if first element is a list (list of utterances) or tuple (single utterance)
                    if isinstance(section_embeddings[0], list):
                        # List of utterances
                        for utterance_idx, utterance_embeddings in enumerate(section_embeddings):
                            if utterance_idx < len(section_texts):
                                documents_list.append(section_texts[utterance_idx])
                                embeddings_list.append(utterance_embeddings)
                    elif isinstance(section_embeddings[0], (tuple, dict)):
                        # Single utterance (list of (token, embedding) tuples or dict)
                        if len(section_texts) == 1:
                            documents_list.append(section_texts[0])
                            embeddings_list.append(section_embeddings)
                        else:
                            # Multiple section texts but single embedding - use first text
                            debug_print(f'Warning: Mismatch between section texts ({len(section_texts)}) '
                                      f'and embeddings (1) for section {key}')
                            documents_list.append(section_texts[0])
                            embeddings_list.append(section_embeddings)
                    else:
                        # Unknown format, try to use as-is
                        if len(section_texts) == 1:
                            documents_list.append(section_texts[0])
                            embeddings_list.append(section_embeddings)
                else:
                    # Empty embeddings for this section
                    debug_print(f'Warning: Empty embeddings for section {key}')
            
            section_idx += 1
        
        return documents_list, embeddings_list

    def extract_topic_modeling(self):
        """
        Extract topics from documents using BERTopic, reusing existing embeddings.
        
        This method supports two modes:
        1. Per-document: Identify topics within each document (works with single document)
        2. Corpus-level: Cluster all documents to find representative topics, then calculate
           how each document relates to these topics (deviation analysis)
        
        Mode is controlled by 'analysis_level' in options_topic-modeling config:
        - 'per_document': Topic modeling within each document
        - 'corpus_level': Topic modeling across all documents with deviation analysis
        - 'both': Run both analyses (default)
        """
        from pelican_nlp.extraction.extract_topic_modeling import TopicModelingExtractor

        print("Extracting Topics...")

        topic_modeling_options = self.config['options_topic-modeling']
        analysis_level = topic_modeling_options.get('analysis_level', 'both')
        min_topic_size = topic_modeling_options.get('min_topic_size', 10)
        extractor = TopicModelingExtractor(topic_modeling_options, self.project_folder)
        
        debug_print(f'Processing {len(self.documents)} documents for topic modeling')
        debug_print(f'Analysis level: {analysis_level}')
        
        # PER-DOCUMENT ANALYSIS
        if analysis_level in ['per_document', 'both']:
            print("Performing per-document topic modeling...")
            for i in range(len(self.documents)):
                debug_print(f'Processing document {i}: {self.documents[i].name}')
                
                documents_list, embeddings_list = self._extract_documents_and_embeddings_from_document(i)
                
                if not documents_list or not embeddings_list:
                    debug_print(f'Warning: No documents or embeddings found for document {self.documents[i].name}. '
                              f'Skipping per-document topic modeling.')
                    continue
                
                if len(documents_list) != len(embeddings_list):
                    debug_print(f'Warning: Mismatch between documents ({len(documents_list)}) '
                              f'and embeddings ({len(embeddings_list)}) for document {self.documents[i].name}')
                    min_len = min(len(documents_list), len(embeddings_list))
                    documents_list = documents_list[:min_len]
                    embeddings_list = embeddings_list[:min_len]
                
                # For single text unit, enable chunking if configured
                chunk_text_units = topic_modeling_options.get('chunk_text_units', False)
                if len(documents_list) < 2:
                    if chunk_text_units:
                        debug_print(f'Only {len(documents_list)} text unit(s) found. '
                                  f'Chunking enabled - will chunk text unit for topic modeling.')
                        # Chunking will be handled in extract_topics_from_text
                    else:
                        debug_print(f'Skipping per-document topic modeling for {self.documents[i].name}: '
                                  f'only {len(documents_list)} text unit(s). '
                                  f'Enable chunk_text_units in config to analyze single text units.')
                        continue
                
                # For per-document, use a lower min_topic_size or allow single document
                # Create a temporary extractor with adjusted min_topic_size
                # Note: If chunking is enabled, min_topic_size will be adjusted after chunking
                per_doc_options = topic_modeling_options.copy()
                # Set a reasonable default that will work even after chunking
                # If chunking happens, it will be re-adjusted in extract_topics_from_text
                if chunk_text_units:
                    # For chunked analysis, use a default that works for typical chunk counts
                    per_doc_options['min_topic_size'] = max(2, min_topic_size // 2)
                else:
                    per_doc_options['min_topic_size'] = max(1, min(len(documents_list) // 2, min_topic_size // 2))
                # Pass embedding options for predefined topic comparison
                per_doc_options['embedding_options'] = self.config.get('options_embeddings', {})
                per_doc_extractor = TopicModelingExtractor(per_doc_options, self.project_folder)
                
                debug_print(f'Extracting topics for {len(documents_list)} text units in document {self.documents[i].name}')
                
                try:
                    topic_data = per_doc_extractor.extract_topics_from_text(
                        documents_list, embeddings_list, per_doc_options
                    )
                    
                    # Prepare data for saving
                    prepared_data = per_doc_extractor.prepare_topic_data_for_saving(topic_data)
                    
                    # Save per-document results
                    assignments = prepared_data.get('assignments', [])
                    if assignments:
                        for assignment in assignments:
                            assignment['document_name'] = self.documents[i].name
                            assignment['document_index'] = i
                        
                        store_features_to_csv(
                            assignments,
                            self.derivatives_dir,
                            self.documents[i],
                            metric='topic-modeling-per-document-assignments'
                        )
                        print(f'Saved per-document topic assignments for {self.documents[i].name} '
                              f'({len(assignments)} assignments)')
                    else:
                        debug_print(f'Warning: No assignments found in prepared_data for {self.documents[i].name}')
                        debug_print(f'Topic data keys: {list(topic_data.keys())}')
                        if 'topic_assignments' in topic_data:
                            debug_print(f'topic_assignments type: {type(topic_data["topic_assignments"])}, '
                                      f'length: {len(topic_data["topic_assignments"]) if hasattr(topic_data["topic_assignments"], "__len__") else "N/A"}')
                        if 'documents' in topic_data:
                            debug_print(f'documents length: {len(topic_data["documents"])}')
                    
                    # Save per-document topic keywords
                    if prepared_data.get('keywords'):
                        keywords_with_doc = []
                        for keyword_entry in prepared_data['keywords']:
                            keyword_entry['document_name'] = self.documents[i].name
                            keywords_with_doc.append(keyword_entry)
                        
                        if keywords_with_doc:
                            store_features_to_csv(
                                keywords_with_doc,
                                self.derivatives_dir,
                                self.documents[i],
                                metric='topic-modeling-per-document-keywords'
                            )
                            debug_print(f'Saved per-document topic keywords for {self.documents[i].name}')
                    
                    # Save per-document topic info
                    if prepared_data.get('topic_info'):
                        import os
                        import pandas as pd
                        topic_info_list = prepared_data['topic_info']
                        if topic_info_list:
                            topic_info_dir = os.path.join(self.derivatives_dir, 'topic-modeling', 'per-document')
                            os.makedirs(topic_info_dir, exist_ok=True)
                            topic_info_path = os.path.join(topic_info_dir, f'{self.documents[i].name}_topic-info.csv')
                            topic_info_df = pd.DataFrame(topic_info_list)
                            topic_info_df.to_csv(topic_info_path, index=False)
                            debug_print(f'Saved per-document topic info to: {topic_info_path}')
                    
                    # Save topic comparisons to predefined topics
                    if prepared_data.get('topic_comparisons'):
                        comparisons_with_doc = []
                        for comparison in prepared_data['topic_comparisons']:
                            comparison['document_name'] = self.documents[i].name
                            comparison['document_index'] = i
                            comparisons_with_doc.append(comparison)
                        
                        if comparisons_with_doc:
                            store_features_to_csv(
                                comparisons_with_doc,
                                self.derivatives_dir,
                                self.documents[i],
                                metric='topic-modeling-predefined-comparisons'
                            )
                            print(f'Saved predefined topic comparisons for {self.documents[i].name} '
                                  f'({len(comparisons_with_doc)} comparisons)')
                
                except Exception as e:
                    import traceback
                    error_trace = traceback.format_exc()
                    debug_print(f'Error in per-document topic modeling for {self.documents[i].name}: {e}')
                    debug_print(f'Full traceback: {error_trace}')
                    print(f'Warning: Could not perform per-document topic modeling for {self.documents[i].name}. '
                          f'This may be due to insufficient text units. Error: {e}')
        
        # CORPUS-LEVEL ANALYSIS
        if analysis_level in ['corpus_level', 'both']:
            print("Performing corpus-level topic modeling...")
            
            # Collect all documents (sections/utterances) and their embeddings from ALL documents
            # Also track which document each item came from for later assignment
            all_documents_list = []
            all_embeddings_list = []
            document_mapping = []  # Maps each item to its source document index
            
            # Iterate through all documents
            for i in range(len(self.documents)):
                debug_print(f'Collecting data from document {i}: {self.documents[i].name}')
                
                documents_list, embeddings_list = self._extract_documents_and_embeddings_from_document(i)
                
                # Add to corpus-level lists
                for doc_text, emb in zip(documents_list, embeddings_list):
                    all_documents_list.append(doc_text)
                    all_embeddings_list.append(emb)
                    document_mapping.append(i)
            
            # Check if we have enough documents for corpus-level topic modeling
            # Adjust min_topic_size for small datasets
            if len(all_documents_list) < min_topic_size:
                adjusted_min = max(1, len(all_documents_list) // 2)
                print(f"Warning: Only {len(all_documents_list)} documents/utterances found (minimum recommended: {min_topic_size}). "
                      f"Adjusting min_topic_size to {adjusted_min} for corpus-level analysis.")
                # Temporarily adjust the extractor's min_topic_size
                corpus_options = topic_modeling_options.copy()
                corpus_options['min_topic_size'] = adjusted_min
                extractor = TopicModelingExtractor(corpus_options, self.project_folder)
            
            if len(all_documents_list) < 2:
                print(f"Warning: Insufficient documents for corpus-level topic modeling. "
                      f"Found {len(all_documents_list)} documents/utterances. Need at least 2. "
                      f"Skipping corpus-level topic modeling.")
            else:
                # Verify documents and embeddings match
                if len(all_documents_list) != len(all_embeddings_list):
                    debug_print(f'Warning: Mismatch between documents ({len(all_documents_list)}) '
                              f'and embeddings ({len(all_embeddings_list)})')
                    min_len = min(len(all_documents_list), len(all_embeddings_list))
                    all_documents_list = all_documents_list[:min_len]
                    all_embeddings_list = all_embeddings_list[:min_len]
                    document_mapping = document_mapping[:min_len]
                
                debug_print(f'Extracting corpus-level topics for {len(all_documents_list)} documents/utterances')
                
                # Extract topics on entire corpus
                # Pass embedding options for predefined topic comparison
                corpus_topic_options = topic_modeling_options.copy()
                corpus_topic_options['embedding_options'] = self.config.get('options_embeddings', {})
                
                topic_data = extractor.extract_topics_from_text(
                    all_documents_list, all_embeddings_list, corpus_topic_options
                )
                
                # Prepare data for saving
                prepared_data = extractor.prepare_topic_data_for_saving(topic_data)
                
                # Calculate document-level topic distributions (for deviation analysis)
                document_topic_distributions = self._calculate_document_topic_distributions(
                    prepared_data, document_mapping, len(self.documents)
                )
                
                # Map topic assignments back to individual documents and save
                if prepared_data['assignments']:
                    # Group assignments by document
                    assignments_by_doc = {}
                    for idx, assignment in enumerate(prepared_data['assignments']):
                        doc_idx = document_mapping[idx] if idx < len(document_mapping) else 0
                        if doc_idx not in assignments_by_doc:
                            assignments_by_doc[doc_idx] = []
                        assignment['document_name'] = self.documents[doc_idx].name
                        assignment['document_index'] = doc_idx
                        assignment['text_unit_index'] = idx
                        assignments_by_doc[doc_idx].append(assignment)
                    
                    # Save assignments per document
                    for doc_idx, assignments in assignments_by_doc.items():
                        store_features_to_csv(
                            assignments,
                            self.derivatives_dir,
                            self.documents[doc_idx],
                            metric='topic-modeling-corpus-assignments'
                        )
                        debug_print(f'Saved corpus-level topic assignments for document {self.documents[doc_idx].name}')
                
                # Save document-level topic distributions (deviation analysis)
                if document_topic_distributions:
                    import os
                    import pandas as pd
                    topic_info_dir = os.path.join(self.derivatives_dir, 'topic-modeling')
                    os.makedirs(topic_info_dir, exist_ok=True)
                    distributions_path = os.path.join(topic_info_dir, f'{self.name}_document-topic-distributions.csv')
                    distributions_df = pd.DataFrame(document_topic_distributions)
                    distributions_df.to_csv(distributions_path, index=False)
                    print(f'Saved document-topic distributions to: {distributions_path}')
                
                # Save topic keywords (corpus-level, save once)
                if prepared_data['keywords']:
                    import os
                    import pandas as pd
                    topic_info_dir = os.path.join(self.derivatives_dir, 'topic-modeling')
                    os.makedirs(topic_info_dir, exist_ok=True)
                    keywords_path = os.path.join(topic_info_dir, f'{self.name}_topic-keywords.csv')
                    keywords_df = pd.DataFrame(prepared_data['keywords'])
                    keywords_df.to_csv(keywords_path, index=False)
                    print(f'Saved corpus-level topic keywords to: {keywords_path}')
                
                # Save topic info summary (corpus-level)
                if prepared_data['topic_info']:
                    import os
                    import pandas as pd
                    topic_info_dir = os.path.join(self.derivatives_dir, 'topic-modeling')
                    os.makedirs(topic_info_dir, exist_ok=True)
                    topic_info_path = os.path.join(topic_info_dir, f'{self.name}_topic-info.csv')
                    topic_info_df = pd.DataFrame(prepared_data['topic_info'])
                    topic_info_df.to_csv(topic_info_path, index=False)
                    print(f'Saved corpus-level topic info to: {topic_info_path}')
                
                # Save corpus-level topic comparisons to predefined topics
                if prepared_data.get('topic_comparisons'):
                    import os
                    import pandas as pd
                    topic_info_dir = os.path.join(self.derivatives_dir, 'topic-modeling')
                    os.makedirs(topic_info_dir, exist_ok=True)
                    comparisons_path = os.path.join(topic_info_dir, f'{self.name}_predefined-topic-comparisons.csv')
                    comparisons_df = pd.DataFrame(prepared_data['topic_comparisons'])
                    comparisons_df.to_csv(comparisons_path, index=False)
                    print(f'Saved corpus-level predefined topic comparisons to: {comparisons_path}')
        
        return

    def _calculate_document_topic_distributions(self, prepared_data, document_mapping, num_documents):
        """
        Calculate how strongly each document relates to each corpus-level topic.
        
        This provides deviation analysis: how each participant's speech relates to predefined topics.
        
        Returns:
            List of dictionaries with document-level topic distributions
        """
        import numpy as np
        
        document_distributions = []
        assignments = prepared_data.get('assignments', [])
        
        # Initialize distributions for each document
        doc_topic_counts = {}  # {doc_idx: {topic_id: count}}
        doc_total_counts = {}  # {doc_idx: total_count}
        
        # Count topic assignments per document
        for idx, assignment in enumerate(assignments):
            doc_idx = document_mapping[idx] if idx < len(document_mapping) else 0
            topic_id = assignment.get('topic_id', -1)
            
            if doc_idx not in doc_topic_counts:
                doc_topic_counts[doc_idx] = {}
                doc_total_counts[doc_idx] = 0
            
            if topic_id not in doc_topic_counts[doc_idx]:
                doc_topic_counts[doc_idx][topic_id] = 0
            
            doc_topic_counts[doc_idx][topic_id] += 1
            doc_total_counts[doc_idx] += 1
        
        # Calculate distributions and probabilities
        for doc_idx in range(num_documents):
            if doc_idx not in doc_topic_counts:
                continue
            
            doc_dist = {
                'document_index': doc_idx,
                'document_name': self.documents[doc_idx].name if doc_idx < len(self.documents) else f'doc_{doc_idx}',
                'total_text_units': doc_total_counts[doc_idx]
            }
            
            # Calculate proportion for each topic
            for topic_id, count in doc_topic_counts[doc_idx].items():
                proportion = count / doc_total_counts[doc_idx] if doc_total_counts[doc_idx] > 0 else 0
                doc_dist[f'topic_{topic_id}_count'] = count
                doc_dist[f'topic_{topic_id}_proportion'] = proportion
            
            # Add probabilities if available
            if prepared_data.get('assignments'):
                # Calculate average probability for each topic in this document
                doc_probs = {}
                for idx, assignment in enumerate(prepared_data['assignments']):
                    if idx < len(document_mapping) and document_mapping[idx] == doc_idx:
                        topic_id = assignment.get('topic_id', -1)
                        if 'topic_1_probability' in assignment:
                            # Get the probability for the assigned topic
                            prob = assignment.get(f'topic_1_probability', 0.0)
                            if topic_id not in doc_probs:
                                doc_probs[topic_id] = []
                            doc_probs[topic_id].append(prob)
                
                for topic_id, probs in doc_probs.items():
                    avg_prob = np.mean(probs) if probs else 0.0
                    doc_dist[f'topic_{topic_id}_avg_probability'] = avg_prob
            
            document_distributions.append(doc_dist)
        
        return document_distributions

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
        
        # Create normalized audio subdirectory in derivatives
        normalized_audio_dir = os.path.join(self.derivatives_dir, 'normalized-audio')
        os.makedirs(normalized_audio_dir, exist_ok=True)
        
        # Import garbage collection and memory monitoring
        import gc
        import torch
        
        # Process each audio document
        for i, document in enumerate(self.documents):
            if hasattr(document, 'file') and document.file:
                print(f"\nProcessing document {i+1}/{len(self.documents)}: {document.file}")
                
                # Check if audio file exists
                if not os.path.exists(document.file):
                    print(f"Error: Audio file not found at {document.file}")
                    continue
                
                try:
                    # Store normalized audio directory in document for use during processing
                    document._normalized_audio_dir = normalized_audio_dir
                    
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
                        f"{Path(document.file).stem}_allOutputs.json"
                    )
                    processed_document.save_as_json(transcription_file)
                    
                    # Save transcription as plain text file
                    transcription_text_file = os.path.join(
                        transcription_dir,
                        f"{Path(document.file).stem}_transcript.txt"
                    )
                    processed_document.save_as_text(transcription_text_file)
                    
                    # Store transcription file paths in the document
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