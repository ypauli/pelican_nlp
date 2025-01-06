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

    def preprocess_all_documents(self):
        for document in self.documents:
            document.create_results_csv(self.config['PATH_TO_PROJECT_FOLDER'])
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
        print('logits extraction in progress')
        logitsExtractor = LogitsExtractor(self.config['tokenization_options_logits'].get('model_name'),
                                          self.pipeline,
                                          self.config['PATH_TO_PROJECT_FOLDER'])
        model = Model(self.config['tokenization_options_logits'].get('model_name'), self.config['PATH_TO_PROJECT_FOLDER'])
        model.load_model()
        for i in range(len(self.documents)):
            for j in range(len(self.documents[i].sections)):
                self.documents[i].logits = logitsExtractor.extract_features(self.documents[i].tokens_logits[j], model)
                store_features_to_csv(self.documents[i].logits, self.documents[i].results_path, append=(j>0))
            print(self.documents[i].logits)
            self.documents[i].logits = []


    def extract_embeddings(self):
        embeddingsExtractor = EmbeddingsExtractor('fastText')
        for i in range(len(self.documents)):
            for j in range(len(self.documents[i].sections)):
                self.documents[i].embeddings = embeddingsExtractor.process_tokens(self.documents[i].tokens_embeddings[j],
                                                                                  self.config['window_sizes'],
                                                                                  self.config['aggregation_functions'])
            print(self.documents[i].embeddings)
            store_features_to_csv(self.documents[i].embeddings, self.documents[i].results_path)
        return

    def get_corpus_info(self):
        info = []
        for subject in self.documents:
            info.append(subject.get_subject_info())
        return '\n'.join(info)
