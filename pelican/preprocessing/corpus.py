from pelican.extraction.extract_logits import LogitsExtractor
from pelican.extraction.extract_embeddings import EmbeddingsExtractor
from pelican.preprocessing import TextPreprocessingPipeline
from pelican.csv_functions import store_features_to_csv
from pelican.extraction.language_model import Model
from pelican.preprocessing.speaker_diarization import TextDiarizer
import pelican.preprocessing.text_cleaner as textcleaner
from pelican.extraction.semantic_similarity import calculate_semantic_similarity, get_cosine_similarity_matrix, get_semantic_similarity_windows
from pelican.extraction.distance_from_randomness import get_distance_from_randomness
import os
import pandas as pd
import re

class Corpus:
    def __init__(self, corpus_name, documents, configuration_settings, task=None):
        """Initialize Corpus object.
        
        Args:
            corpus_name: Name of the corpus
            documents: List of Document objects
            configuration_settings: Dictionary of configuration options
            task: Optional task identifier
        """
        self.name = corpus_name
        self.documents = documents
        self.config = configuration_settings
        self.derivative_dir = self.config['PATH_TO_PROJECT_FOLDER']+'/derivatives'
        self.pipeline = TextPreprocessingPipeline(self.config)
        self.task = task
        self.results_path = None

    def preprocess_all_documents(self):
        """Process all documents and create aggregated results."""
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
            derivatives_path = os.path.dirname(os.path.dirname(os.path.dirname(self.documents[0].results_path)))
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

                embeddings = embeddingsExtractor.extract_embeddings_from_text(section)
                self.documents[i].embeddings.append(embeddings)
                
                for utterance in embeddings:
                    print(f'Processing utterance (length: {len(utterance)} tokens)')
                    
                    if self.config['options_embeddings']['semantic-similarity']:

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
                        divergence = get_distance_from_randomness(utterance, self.config["options_dis_from_randomness"])
                        print(f'Divergence from optimality metrics: {divergence}')
                        store_features_to_csv(divergence,
                                              self.derivative_dir,
                                              self.documents[i],
                                              metric='distance-from-randomness')

                    # Process tokens without printing intermediate results
                    if embedding_options['clean_tokens']:
                        cleaned_dict = {token: embeddings for token, embeddings in utterance.items()
                                     if (cleaned_token := textcleaner.clean_subword_token_RoBERTa(token)) is not None}
                    else:
                        cleaned_dict = utterance

                    store_features_to_csv(cleaned_dict,
                                          self.derivative_dir,
                                          self.documents[i],
                                          metric='embeddings')
        return

    def get_corpus_info(self):
        info = []
        for subject in self.documents:
            info.append(subject.get_subject_info())
        return '\n'.join(info)