#this is the main project file
#created 1.10.24
#created by Yves Pauli
#=============================
import os

from config import Config
from document import Document
from preprocessing import Subject, Corpus, TextPreprocessingPipeline

if __name__ == '__main__':

    #check for correct folder structure (subject|session|group|textfile)
    #...

    # Initialize the pipeline with configuration settings
    config = Config()

    # Instantiate all subjects
    subjects = []
    for subject in os.listdir(config.PATH_TO_SUBJECTS_FOLDER):
        subjects.append(Subject(subject))

    for current_corpus in config.corpus_names:

        documents = []
        # load all files belonging to same corpus
        for i in range(len(subjects)):

            #Check if subject has file belonging to corpus and add if available
            FILEPATH = config.PATH_TO_SUBJECTS_FOLDER + subjects[i].subjectID + '/' + current_corpus
            if os.path.isdir(FILEPATH):
                file_name = os.listdir(FILEPATH)[0]
                documents.append(Document(FILEPATH, file_name, current_corpus))
                subjects[i].add_document(documents[-1])

        # Initialize the corpus with respective files
        corpus = Corpus(current_corpus, documents, config)

        # Process all files in the corpus
        corpus.process_all_documents()


        for i in range(len(documents)):
            print(documents[i].processed_text)