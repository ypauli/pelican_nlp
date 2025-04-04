"""
This module provides the Corpus class, which aggregates documents where the same processing
steps applied and results should be aggregated.
(e.g. all fluency files from task 'animals' or all image-descriptions from the same image)

This class contains the pipelines for homogenous processing and metric extraction of all grouped files.
"""


from pelican.preprocessing import TextPreprocessingPipeline
from pelican.utils.csv_functions import store_features_to_csv
from pelican.extraction.language_model import Model
from pelican.preprocessing.speaker_diarization import TextDiarizer
import pelican.preprocessing.text_cleaner as textcleaner
import os
import pandas as pd
import re

class Corpus:
    def __init__(self, corpus_name, documents, configuration_settings):
        self.name = corpus_name
        self.documents = documents
        self.config = configuration_settings
        self.derivative_dir = self.config['PATH_TO_PROJECT_FOLDER']+'/derivatives'
        self.pipeline = TextPreprocessingPipeline(self.config)
        self.task = configuration_settings['task_name']
        self.results_path = None

    def preprocess_all_documents(self):
        """Preprocess all documents"""
        print('Preprocessing all documents...')
        for document in self.documents:
            document.detect_sections()
            document.process_document(self.pipeline)

    def get_all_processed_texts(self):
        result = {}
        for subject in self.documents:
            result[subject.name] = subject.get_processed_texts()
        return result

    def create_corpus_results_consolidation_csv(self):
        """Create separate aggregated results CSV files for each metric."""
        print("Creating aggregated results files per metric...")
        
        try:
            derivatives_path = os.path.dirname(os.path.dirname(self.documents[0].results_path))
        except (AttributeError, IndexError):
            print("Error: No valid results path found in documents")
            return
        
        # Create aggregations folder
        aggregation_path = os.path.join(derivatives_path, 'aggregations')
        os.makedirs(aggregation_path, exist_ok=True)
        
        # Initialize results dictionary with metrics as keys
        results_by_metric = {}
        
        # Walk through all directories in derivatives
        for root, dirs, files in os.walk(derivatives_path):
            # Skip the aggregations directory itself
            if 'aggregations' in root:
                continue
                
            for file in files:
                if not file.endswith('.csv'):
                    continue
                    
                file_path = os.path.join(root, file)
                try:
                    df = pd.read_csv(file_path)
                    subject_key = os.path.basename(file).split('_')[0]
                    
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
                    
                    # Initialize subject dict if not exists
                    if subject_key not in results_by_metric[metric]:
                        results_by_metric[metric][subject_key] = {}
                    
                    # Process based on metric type
                    if metric == 'semantic-similarity':
                        window_size = re.search(r'window-(\d+)', file).group(1)
                        for _, row in df.iterrows():
                            if 'Metric' in df.columns and 'Similarity_Score' in df.columns:
                                metric_name = f"window_{window_size}_{row['Metric']}"
                                results_by_metric[metric][subject_key][metric_name] = row['Similarity_Score']

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
        from pelican.extraction.extract_logits import LogitsExtractor
        from pelican.preprocessing.text_tokenizer import TextTokenizer
        logits_options = self.config['options_logits']
        project_path = self.config['PATH_TO_PROJECT_FOLDER']

        print('logits extraction in progress')
        model_name = logits_options['model_name']
        logitsExtractor = LogitsExtractor(logits_options,
                                          self.pipeline,
                                          project_path)
        model = Model(model_name, project_path)
        model.load_model()
        model_instance = model.model_instance
        tokenizer = TextTokenizer(logits_options['tokenization_method'], model_name=logits_options['model_name'])
        for i in range(len(self.documents)):

            for key, section in self.documents[i].cleaned_sections.items():

                if self.config['discourse'] == True:
                    section = TextDiarizer.parse_speaker(section, self.config['subject_speakertag'],
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
                                          self.derivative_dir,
                                          self.documents[i],
                                          metric='logits')

    def extract_embeddings(self):
        from pelican.extraction.extract_embeddings import EmbeddingsExtractor

        embedding_options = self.config['options_embeddings']
        print('Embeddings extraction in progress...')
        embeddingsExtractor = EmbeddingsExtractor(embedding_options, self.config['PATH_TO_PROJECT_FOLDER'])
        for i in range(len(self.documents)):
            for key, section in self.documents[i].cleaned_sections.items():
                print(f'Processing section {key}')
                
                if self.config['discourse']:
                    section = TextDiarizer.parse_speaker(section, self.config['subject_speakertag'], embedding_options['keep_speakertags'])
                else:
                    section = [section]

                embeddings, token_count = embeddingsExtractor.extract_embeddings_from_text(section)
                self.documents[i].embeddings.append(embeddings)

                if self.task == 'fluency':
                    self.documents[i].fluency_word_count = token_count
                
                for utterance in embeddings:

                    if self.config['options_embeddings']['semantic-similarity']:
                        from pelican.extraction.semantic_similarity import calculate_semantic_similarity, \
                            get_semantic_similarity_windows
                        consecutive_similarities, mean_similarity = calculate_semantic_similarity(utterance)
                        print(f'Mean semantic similarity: {mean_similarity:.4f}')

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
                                print(f'Window {window_size} stats - mean: {window_stats[0]:.4f}, std: {window_stats[1]:.4f}, median: {window_stats[4]:.4f}')
                            else:
                                window_data = {
                                    'mean': window_stats[0] if isinstance(window_stats, tuple) else window_stats,
                                    'std': window_stats[1] if isinstance(window_stats, tuple) and len(window_stats) > 1 else None
                                }
                            
                            store_features_to_csv(window_data,
                                                  self.derivative_dir,
                                                  self.documents[i],
                                                  metric=f'semantic-similarity-window-{window_size}')

                    if self.config['options_embeddings']['distance-from-randomness']:
                        from pelican.extraction.distance_from_randomness import get_distance_from_randomness
                        divergence = get_distance_from_randomness(utterance, self.config["options_dis_from_randomness"])
                        print(f'Divergence from optimality metrics: {divergence}')
                        store_features_to_csv(divergence,
                                              self.derivative_dir,
                                              self.documents[i],
                                              metric='distance-from-randomness')

                    # Process tokens without printing intermediate results
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
                                          self.derivative_dir,
                                          self.documents[i],
                                          metric='embeddings')
        return

    def extract_opensmile_features(self):
        from pelican.extraction.acoustic_feature_extraction import AudioFeatureExtraction
        for i in range(len(self.documents)):
            results, recording_length = AudioFeatureExtraction.opensmile_extraction(self.documents[i].file, self.config['opensmile_configurations'])
            self.documents[i].recording_length = recording_length  # Store the recording length
            results['subject_ID'] = self.documents[i].subject_ID  # Set the subject ID
            print('results obtained')
            store_features_to_csv(results,
                                self.derivative_dir,
                                self.documents[i],
                                metric='opensmile-features')

    def extract_prosogram(self):
        from pelican.extraction.acoustic_feature_extraction import AudioFeatureExtraction
        for i in range(len(self.documents)):
            results = AudioFeatureExtraction.extract_prosogram_profile(self.documents[i].file)
            print('prosogram obtained')

    def create_document_information_csv(self):
        """Create CSV file with summarized document parameters based on config specifications."""
        print("Creating document information summary...")
        
        try:
            derivatives_path = os.path.dirname(os.path.dirname(self.documents[0].results_path))
        except (AttributeError, IndexError):
            print("Error: No valid results path found in documents")
            return
        
        # Create document_information folder inside aggregations
        doc_info_path = os.path.join(derivatives_path, 'aggregations', 'document_information')
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
        print(f"Document information saved to: {output_file}")