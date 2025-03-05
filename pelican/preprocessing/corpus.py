from pelican.extraction.extract_logits import LogitsExtractor
from pelican.extraction.extract_embeddings import EmbeddingsExtractor
from pelican.preprocessing import TextPreprocessingPipeline
from pelican.csv_functions import store_features_to_csv
from pelican.extraction.language_model import Model
from pelican.preprocessing.speaker_diarization import TextDiarizer
import pelican.preprocessing.text_cleaner as textcleaner
from pelican.extraction.semantic_similarity import calculate_semantic_similarity, get_cosine_similarity_matrix, get_semantic_similarity_windows

class Corpus:
    def __init__(self, corpus_name, documents, configuration_settings, task=None):
        """Takes a list of file instances and configures the pipeline."""
        self.name = corpus_name #e.g. placebo_group
        self.documents = documents
        self.config = configuration_settings
        self.pipeline = TextPreprocessingPipeline(self.config)
        self.task = task
        self.results_path = None

    def preprocess_all_documents(self):
        print(f'preprocessing all documents (corpus.py)')
        for document in self.documents:
            document.detect_sections()
            document.process_document(self.pipeline)

    def get_all_processed_texts(self):
        result = {}
        for subject in self.documents:
            result[subject.name] = subject.get_processed_texts()
        return result

    def create_corpus_results_consolidation_csv(self):
        #creating output file to store evaluated data
        return

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

                    #utterance of type dict, keys tokens, entries embeddings
                    if self.config['options_embeddings']['semantic_similarity']:
                        mean_similarity = calculate_semantic_similarity(utterance)
                        print(f'mean similarity for utterance is: {mean_similarity}')
                        cosine_similarity_matrix = get_cosine_similarity_matrix(utterance)
                        print(f'cosine similarity matrix: {cosine_similarity_matrix}')
                        window = get_semantic_similarity_windows(utterance, self.config['options_embeddings']['window_size'])
                        print(window)

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
