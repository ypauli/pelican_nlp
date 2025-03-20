from pelican.extraction.extract_logits import LogitsExtractor
from pelican.extraction.extract_embeddings import EmbeddingsExtractor
from pelican.preprocessing import TextPreprocessingPipeline
from pelican.csv_functions import store_features_to_csv
from pelican.extraction.language_model import Model
from pelican.preprocessing.speaker_diarization import TextDiarizer
import pelican.preprocessing.text_cleaner as textcleaner
from pelican.extraction.semantic_similarity import calculate_semantic_similarity, get_cosine_similarity_matrix, get_semantic_similarity_windows
from pelican.extraction.distance_from_randomness import get_divergence_from_optimality
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
        self.pipeline = TextPreprocessingPipeline(self.config)
        self.task = task
        self.results_path = None

    def preprocess_all_documents(self):
        """Process all documents and create aggregated results."""
        print(f'preprocessing all documents (corpus.py)')
        for document in self.documents:
            document.detect_sections()
            document.process_document(self.pipeline)
        
        # After all documents are processed, create aggregation
        if self.config.get('create_aggregation_of_results', True):  # Make it configurable
            self.create_corpus_results_consolidation_csv()

    def get_all_processed_texts(self):
        result = {}
        for subject in self.documents:
            result[subject.name] = subject.get_processed_texts()
        return result

    def create_corpus_results_consolidation_csv(self):
        """Create aggregated results CSV file for all metrics."""
        print("Creating aggregated results file...")
        
        # Define the aggregation folder path
        try:
            derivatives_path = os.path.dirname(os.path.dirname(os.path.dirname(self.documents[0].results_path)))
        except (AttributeError, IndexError):
            print("Error: No valid results path found in documents")
            return
        
        aggregation_path = os.path.join(derivatives_path, 'aggregations')
        os.makedirs(aggregation_path, exist_ok=True)
        
        # Initialize results dictionary
        aggregated_results = {}
        
        for document in self.documents:
            subject_id = document.subject_ID or "unknown"
            session = document.session or "session1"  # Default to session1 if None
            task = document.task or "unknown"
            
            if subject_id not in aggregated_results:
                aggregated_results[subject_id] = {
                    'subject_id': subject_id,
                    'task': task,
                    'corpus': document.corpus_name
                }
                
            # Get embeddings results
            embeddings_path = os.path.join(derivatives_path, 'embeddings', 
                                         str(subject_id), session, task)
            if os.path.exists(embeddings_path):
                for file in os.listdir(embeddings_path):
                    if file.endswith('.csv'):
                        try:
                            embeddings_data = pd.read_csv(os.path.join(embeddings_path, file))
                            # Add relevant embeddings metrics
                            aggregated_results[subject_id]['embedding_dimensions'] = len(embeddings_data.columns) - 1
                            aggregated_results[subject_id]['token_count'] = len(embeddings_data)
                        except Exception as e:
                            print(f"Error processing embeddings file {file}: {e}")

            # Get semantic similarity results
            semantic_path = os.path.join(derivatives_path, 'semantic-similarity',
                                       str(subject_id), session, task)
            if os.path.exists(semantic_path):
                for file in os.listdir(semantic_path):
                    try:
                        if 'consecutive' in file:
                            consec_data = pd.read_csv(os.path.join(semantic_path, file))
                            aggregated_results[subject_id]['mean_consecutive_similarity'] = consec_data['Consecutive_Similarity'].mean()
                            aggregated_results[subject_id]['overall_mean_similarity'] = consec_data['Mean_Similarity'].iloc[0]
                        elif 'window' in file:
                            window_match = re.search(r'window-(\d+)', file)
                            if window_match:
                                window_size = window_match.group(1)
                                window_data = pd.read_csv(os.path.join(semantic_path, file))
                                aggregated_results[subject_id][f'window_{window_size}_mean'] = window_data.iloc[0, 0]
                                aggregated_results[subject_id][f'window_{window_size}_std'] = window_data.iloc[0, 1]
                    except Exception as e:
                        print(f"Error processing semantic similarity file {file}: {e}")

            # Add document-specific metrics
            try:
                aggregated_results[subject_id].update({
                    'number_of_duplicates': getattr(document, 'number_of_duplicates', None),
                    'number_of_hyphenated_words': getattr(document, 'number_of_hyphenated_words', None),
                    'length_in_words': getattr(document, 'length_in_words', None),
                    'length_in_lines': getattr(document, 'length_in_lines', None)
                })
            except Exception as e:
                print(f"Error adding document metrics for subject {subject_id}: {e}")

        # Convert to DataFrame and save
        if aggregated_results:
            try:
                df = pd.DataFrame(list(aggregated_results.values()))
                output_file = os.path.join(aggregation_path, f'{self.name}_aggregated_results.csv')
                df.to_csv(output_file, index=False)
                print(f"Aggregated results saved to: {output_file}")
            except Exception as e:
                print(f"Error saving aggregated results: {e}")
        else:
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
            print('self documents cleaned_sections: ', self.documents[i].cleaned_sections)
            for key, section in self.documents[i].cleaned_sections.items():

                print(f'current section is {section}')

                if self.config['discourse'] == True:
                    section = TextDiarizer.parse_speaker(section, self.config['subject_speakertag'],
                                                         logits_options['keep_speakertags'])
                    print(f'parsed section is {section}')
                else:
                    section = [section]

                print(f'Extracting Logits for section {key}')

                for part in section:
                    print(part)
                    logits = logitsExtractor.extract_features(part, tokenizer, model_instance)
                    print(logits)
                    self.documents[i].logits.append(logits)

                    #'logits' list of dictionaries; keys token, logprob_actual, logprob_max, entropy, most_likely_token
                    store_features_to_csv(logits, self.documents[i].results_path, self.documents[i].corpus_name,
                                          metric='logits')

    def extract_embeddings(self):
        embedding_options = self.config['options_embeddings']
        print('embeddings extraction in progress')
        embeddingsExtractor = EmbeddingsExtractor(embedding_options, self.config['PATH_TO_PROJECT_FOLDER'])
        for i in range(len(self.documents)):
            print('self documents cleaned_sections: ', self.documents[i].cleaned_sections)
            for key, section in self.documents[i].cleaned_sections.items():

                print(f'current section is {section}')

                if self.config['discourse']==True:
                    section = TextDiarizer.parse_speaker(section, self.config['subject_speakertag'], embedding_options['keep_speakertags'])
                    print(f'parsed section is {section}')
                else:
                    section = [section]

                print(f'Extracting Embeddings for section {key}')
                embeddings = embeddingsExtractor.extract_embeddings_from_text(section)
                self.documents[i].embeddings.append(embeddings)
                #embeddings is a list of dictionaries
                for utterance in embeddings:

                    print(f'current utterance keys: {utterance.keys()}')
                    #utterance of type dict, keys tokens, entries embeddings
                    if self.config['options_embeddings']['semantic-similarity']:
                        consecutive_similarities, mean_similarity = calculate_semantic_similarity(utterance)
                        print(f'mean similarity for utterance is: {mean_similarity}')
                        cosine_similarity_matrix = get_cosine_similarity_matrix(utterance)

                        # Create dictionary for consecutive similarities
                        consecutive_sim_data = {
                            'consecutive_similarities': consecutive_similarities,
                            'mean_similarity': mean_similarity
                        }

                        # Store consecutive similarities
                        store_features_to_csv(consecutive_sim_data, self.documents[i].results_path,
                                              self.documents[i].corpus_name, metric='semantic-similarity-consecutive')

                        # Store cosine similarity matrix
                        store_features_to_csv(cosine_similarity_matrix, self.documents[i].results_path,
                                              self.documents[i].corpus_name, metric='semantic-similarity-matrix')

                        for window_size in self.config['options_semantic-similarity']['window_sizes']:
                            window_stats = get_semantic_similarity_windows(utterance, window_size)
                            if isinstance(window_stats, tuple) and len(window_stats) == 4:
                                window_data = {
                                    'mean_of_window_means': window_stats[0],
                                    'std_of_window_means': window_stats[1],
                                    'mean_of_window_stds': window_stats[2],
                                    'std_of_window_stds': window_stats[3]
                                }
                            else:
                                # For the case when window_size is 'all' or other special cases
                                window_data = {
                                    'mean': window_stats[0] if isinstance(window_stats, tuple) else window_stats,
                                    'std': window_stats[1] if isinstance(window_stats, tuple) and len(window_stats) > 1 else None
                                }
                            
                            store_features_to_csv(window_data, self.documents[i].results_path,
                                                  self.documents[i].corpus_name,
                                                  metric=f'semantic-similarity-window-{window_size}')

                    if self.config['options_embeddings']['divergence_from_optimality']:
                        print(f'calculating distance from optimality...')
                        print(f'div from optimality: {get_divergence_from_optimality(utterance, self.config["options_div_from_optimality"])}')

                    if embedding_options['clean_tokens']:
                        #utterance is a dictionary
                        cleaned_dict = {}

                        # Clean each token in the dictionary
                        for token, embeddings in utterance.items():
                            cleaned_token = textcleaner.clean_subword_token_RoBERTa(token)

                            if cleaned_token is not None:
                                cleaned_dict[cleaned_token] = embeddings
                    else:
                        cleaned_dict=utterance

                    store_features_to_csv(cleaned_dict, self.documents[i].results_path, self.documents[i].corpus_name, metric='embeddings')
        return

    def get_corpus_info(self):
        info = []
        for subject in self.documents:
            info.append(subject.get_subject_info())
        return '\n'.join(info)

    def process_embeddings(self, embedding_options, utterance, document_index):
        """Process embeddings for a document.
        
        Args:
            embedding_options: Dictionary of embedding options
            utterance: Dictionary of utterance data
            document_index: Index of current document
        """
        if embedding_options.get('divergence_from_optimality'):
            divergence = get_divergence_from_optimality(
                utterance, 
                self.config["options_div_from_optimality"]
            )
            print(f'Distance from optimality: {divergence}')

        if embedding_options.get('clean_tokens'):
            cleaned_dict = {
                token: embeddings
                for token, embeddings in utterance.items()
                if (cleaned_token := textcleaner.clean_subword_token_RoBERTa(token)) is not None
            }
        else:
            cleaned_dict = utterance

        store_features_to_csv(
            cleaned_dict,
            self.documents[document_index].results_path,
            self.documents[document_index].corpus_name,
            metric='embeddings'
        )
