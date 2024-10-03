#this is the main project file
#created 1.10.24
#=============================

from config import Config
from subject import Subject
from preprocessing.corpus import Corpus

if __name__ == '__main__':
    #main script for preprocessing and extraction of data

    # Initialize the pipeline with configuration settings
    config = Config()
    pipeline = TextPreprocessingPipeline(config)

    # Create subjects with their associated document file paths
    subject_1 = Subject()

    # Initialize the corpus with subjects
    corpus = Corpus([subject_1], config)

    # Process all subjects in the corpus, assuming these documents are dialogs
    for subject in corpus.subjects:
        for document in subject.documents:
            document.load_text(corpus.importer)
            document.clean_text(corpus.cleaner, is_dialog=True)  # Set is_dialog to True for dialog
            document.tokenize_text(corpus.tokenizer)
            document.normalize_text(corpus.normalizer)

    # Run the pipeline with the specified file
    text = pipeline.process_text(config.input_file)

    # Output results
    print(text)