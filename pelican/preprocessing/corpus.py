from pelican.extraction.extract_logits import LogitsExtractor
from pelican.extraction.extract_embeddings import EmbeddingsExtractor
from pelican.preprocessing import TextPreprocessingPipeline
from pelican.csv_functions import store_features_to_csv
from pelican.extraction.LanguageModel import Model

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
        print('logits extraction in progress')
        logitsExtractor = LogitsExtractor(self.config['tokenization_options_logits'].get('model_name'),
                                          self.pipeline,
                                          self.config['PATH_TO_PROJECT_FOLDER'])
        model = Model(self.config['tokenization_options_logits'].get('model_name'), self.config['PATH_TO_PROJECT_FOLDER'])
        model.load_model()
        tokenizer = TextTokenizer(self.config['tokenization_options_logits'])
        for i in range(len(self.documents)):
            self.documents[i].tokenize_text(tokenizer, 'logits')
            for j in range(len(self.documents[i].sections)):
                self.documents[i].logits = logitsExtractor.extract_features(self.documents[i].tokens_logits[j], model)
                store_features_to_csv(self.documents[i].logits, self.documents[i].results_path, self.name)
            print(self.documents[i].logits)
            self.documents[i].logits = []


    def extract_embeddings(self):
        print('embeddings extraction in progress')
        embeddingsExtractor = EmbeddingsExtractor('fastText')
        for i in range(len(self.documents)):
            print('self documents cleaned_sections: ', self.documents[i].cleaned_sections)
            embeddingsExtractor.process_text(self.documents[i],
                                            self.config['tokenization_options_embeddings'],
                                            self.config['window_sizes'],
                                            self.config['aggregation_functions'],
                                            speakertag=self.config['subject_speakertag'])
        return

    def get_corpus_info(self):
        info = []
        for subject in self.documents:
            info.append(subject.get_subject_info())
        return '\n'.join(info)
