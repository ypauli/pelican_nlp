from extraction.extract_logits import LogitsExtractor
from preprocessing import TextPreprocessingPipeline
from transformers import AutoModelForCausalLM, AutoModel, LlamaForCausalLM

import torch
from accelerate import Accelerator, init_empty_weights

class Corpus:
    def __init__(self, corpus_name, documents, config, task=None):
        """Takes a list of file instances and configures the pipeline."""
        self.name = corpus_name #e.g. placebo_group
        self.documents = documents
        self.config = config
        self.pipeline = TextPreprocessingPipeline(self.config)
        self.task = task

    def preprocess_all_documents(self):
        """Processes all files in each subject."""
        for document in self.documents:
            document.process_document(self.pipeline)

    def get_all_processed_texts(self):
        """Returns processed texts for all subjects."""
        result = {}
        for subject in self.documents:
            result[subject.name] = subject.get_processed_texts()
        return result

    def extract_logits(self):

        logitsExtractor = LogitsExtractor(self.config.tokenization_options.get('model_name'), self.pipeline, self.config)
        for i in range(len(self.documents)):
            self.documents[i].logits = logitsExtractor.extract_features(self.documents[i].tokens)
            print(self.documents[i].logits)

    def get_corpus_info(self):
        """Returns metadata for the entire corpus."""
        info = []
        for subject in self.documents:
            info.append(subject.get_subject_info())
        return '\n'.join(info)
