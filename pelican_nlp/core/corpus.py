"""
This module provides the Corpus class, which aggregates documents where the same processing
steps applied and results should be aggregated.
(e.g. all fluency files from task 'animals' or all image-descriptions from the same image)

This class contains the pipelines for homogenous processing and metric extraction of all grouped files.
"""

import os
import pandas as pd
from ..preprocessing import TextPreprocessingPipeline
from ..utils.csv_functions import store_features_to_csv
from ..extraction.language_model import Model
from ..preprocessing.speaker_diarization import TextDiarizer
from ..preprocessing import text_cleaner as textcleaner
import re

from pelican_nlp.config import debug_print

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
        """Preprocess all documents"""
        print(f'Preprocessing all documents of corpus {self.name}...')
        for document in self.documents:
            document.detect_sections()
            document.process_document(self.pipeline)

    def get_all_processed_texts(self):
        result = {}
        for participant in self.documents:
            result[participant.name] = participant.get_processed_texts()
        return result

    def create_corpus_results_consolidation_csv(self):
        """Create separate aggregated results CSV files for each metric."""
        print("Creating aggregated results files per metric...")
        
        # Create aggregations folder
        aggregation_path = os.path.join(self.derivatives_dir, 'aggregations')
        os.makedirs(aggregation_path, exist_ok=True)
        
        # Initialize results dictionary with metrics as keys
        results_by_metric = {}
        
        # Walk through all directories in derivatives
        for root, dirs, files in os.walk(self.derivatives_dir):
            # Skip the aggregations directory itself
            if 'aggregations' in root:
                continue
                
            for file in files:
                if not file.endswith('.csv'):
                    continue
                    
                file_path = os.path.join(root, file)
                try:
                    df = pd.read_csv(file_path)
                    participant_key = os.path.basename(file).split('_')[0]
                    
                    # Determine metric type from file path
                    if 'semantic-similarity-window' in file:
                        metric = 'semantic-similarity'
                    elif 'distance-from-randomness' in file:
                        metric = 'distance-from-randomness'
                    else:
                        continue
                    
                    # Initialize metric dict if not exists
                    if metric not in results_by_metric:
                        results_by_metric[metric] = {}
                    
                    # Initialize participant dict if not exists
                    if participant_key not in results_by_metric[metric]:
                        results_by_metric[metric][participant_key] = {}
                    
                    # Process based on metric type
                    if metric == 'semantic-similarity':
                        window_size = re.search(r'window-(\d+)', file).group(1)
                        for _, row in df.iterrows():
                            if 'Metric' in df.columns and 'Similarity_Score' in df.columns:
                                metric_name = f"window_{window_size}_{row['Metric']}"
                                results_by_metric[metric][participant_key][metric_name] = row['Similarity_Score']

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    continue
        
        # Save separate aggregated results for each metric
        for metric, metric_results in results_by_metric.items():
            if metric_results:
                output_file = os.path.join(aggregation_path, f'{self.name}_{metric}_aggregated_results.csv')
                pd.DataFrame(metric_results).T.to_csv(output_file)
                print(f"Aggregated results for {metric} saved to: {output_file}")
            
        if not results_by_metric:
            print("No results to aggregate")

    def extract_logits(self):
        from pelican_nlp.extraction.extract_logits import LogitsExtractor
        from pelican_nlp.preprocessing.text_tokenizer import TextTokenizer
        logits_options = self.config['options_logits']

        print('logits extraction in progress')

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
                    section = TextDiarizer.parse_speaker(section, self.config['participant_speakertag'],
                                                         logits_options['keep_speakertags'])
                    #print(f'parsed section is {section}')
                else:
                    section = [section]

                print(f'Extracting Logits for section {key}')

                for part in section:
                    print(part)
                    logits = logitsExtractor.extract_features(part, tokenizer, model_instance)
                    print(logits)
                    self.documents[i].logits.append(logits)

                    #'logits' list of dictionaries; keys token, logprob_actual, logprob_max, entropy, most_likely_token
                    store_features_to_csv(logits,
                                          self.derivatives_dir,
                                          self.documents[i],
                                          metric='logits')

    def extract_embeddings(self):
        from pelican_nlp.extraction.extract_embeddings import EmbeddingsExtractor

        embedding_options = self.config['options_embeddings']
        print('Embeddings extraction in progress...')
        embeddingsExtractor = EmbeddingsExtractor(embedding_options, self.project_folder)
        debug_print(len(self.documents))
        for i in range(len(self.documents)):

            debug_print(f'cleaned sections: {self.documents[i].cleaned_sections}')
            for key, section in self.documents[i].cleaned_sections.items():
                debug_print(f'Processing section {key}')
                
                if self.config['discourse']:
                    section = TextDiarizer.parse_speaker(section, self.config['participant_speakertag'], embedding_options['keep_speakertags'])
                else:
                    section = [section]

                embeddings, token_count = embeddingsExtractor.extract_embeddings_from_text(section)
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
                            window_stats = get_semantic_similarity_windows(utterance, window_size)
                            if isinstance(window_stats, tuple) and len(window_stats) == 5:
                                window_data = {
                                    'mean_of_window_means': window_stats[0],
                                    'std_of_window_means': window_stats[1],
                                    'mean_of_window_stds': window_stats[2],
                                    'std_of_window_stds': window_stats[3],
                                    'mean_of_window_medians': window_stats[4]
                                }
                                debug_print(f'Window {window_size} stats - mean: {window_stats[0]:.4f}, std: {window_stats[1]:.4f}, median: {window_stats[4]:.4f}')
                            else:
                                window_data = {
                                    'mean': window_stats[0] if isinstance(window_stats, tuple) else window_stats,
                                    'std': window_stats[1] if isinstance(window_stats, tuple) and len(window_stats) > 1 else None
                                }
                            
                            store_features_to_csv(window_data,
                                                  self.derivatives_dir,
                                                  self.documents[i],
                                                  metric=f'semantic-similarity-window-{window_size}')

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

                    store_features_to_csv(cleaned_embeddings,
                                          self.derivatives_dir,
                                          self.documents[i],
                                          metric='embeddings')
        return

    def extract_opensmile_features(self):
        from pelican_nlp.extraction.acoustic_feature_extraction import AudioFeatureExtraction
        for i in range(len(self.documents)):
            results, recording_length = AudioFeatureExtraction.opensmile_extraction(self.documents[i].file, self.config['opensmile_configurations'])
            self.documents[i].recording_length = recording_length  # Store the recording length
            results['participant_ID'] = self.documents[i].participant_ID  # Set the participant ID
            print('opensmile results obtained')
            store_features_to_csv(results,
                                  self.derivatives_dir,
                                  self.documents[i],
                                  metric='opensmile-features')

    def extract_prosogram(self):
        from pelican_nlp.extraction.acoustic_feature_extraction import AudioFeatureExtraction
        from pelican_nlp.utils.csv_functions import store_features_to_csv
        for i in range(len(self.documents)):
            # Create the output directory for this document's prosogram files
            output_dir = os.path.join(self.derivatives_dir, 'prosogram-features', 
                                    f"part-{self.documents[i].participant_ID}")
            
            results = AudioFeatureExtraction.extract_prosogram_profile(
                self.documents[i].file, 
                output_dir=output_dir
            )
            print('prosogram obtained')


    def create_document_information_csv(self):
        """Create CSV file with summarized document parameters based on config specifications."""
        print("Creating document information summary...")
        
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