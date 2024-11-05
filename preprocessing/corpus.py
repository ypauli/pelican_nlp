from preprocessing import TextPreprocessingPipeline
from preprocessing.subject import Subject

class Corpus:
    def __init__(self, corpus_name, documents, config, task=None):
        """Takes a list of file instances and configures the pipeline."""
        self.name = corpus_name #e.g. placebo_group
        self.documents = documents
        self.config = config
        self.pipeline = TextPreprocessingPipeline(self.config)
        self.task = task

    def process_all_documents(self):
        """Processes all files in each subject."""
        for document in self.documents:
            document.process_document(self.pipeline)

    def get_all_processed_texts(self):
        """Returns processed texts for all subjects."""
        result = {}
        for subject in self.documents:
            result[subject.name] = subject.get_processed_texts()
        return result

    def get_corpus_info(self):
        """Returns metadata for the entire corpus."""
        info = []
        for subject in self.documents:
            info.append(subject.get_subject_info())
        return '\n'.join(info)
