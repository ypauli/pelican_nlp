#this is the main project file
#created 1.10.24
#created by Yves Pauli
#=============================
import os
import torch.cuda
import sys
from config import Config
from document import Document
from preprocessing import Subject, Corpus
from pathlib import Path

if __name__ == '__main__':

    #Initializing pipeline with configuration settings
    config = Config()
    if config.gpu_available: torch.cuda.empty_cache()

    #check/create LPDS (Language Processing Data Structure)
    Path(config.PATH_TO_PROJECT_FOLDER + 'Outputs').mkdir(parents=True, exist_ok=True)
    if not os.path.isdir(config.PATH_TO_SUBJECTS):
        print('Warning: No Subjects folder')
        sys.exit()

    # Instantiate all subjects
    subjects = []
    for subject in os.listdir(config.PATH_TO_PROJECT_FOLDER+'Subjects/'):
        subjects.append(Subject(subject))

    for current_corpus in config.corpus_names:

        print(f'{current_corpus} is being processed')

        documents = []
        # load all files belonging to same corpus
        for i in range(len(subjects)):

            #Check if subject has file belonging to corpus and add if available
            FILEPATH = config.PATH_TO_PROJECT_FOLDER + 'Subjects/' + subjects[i].subjectID + '/' + current_corpus
            if os.path.isdir(FILEPATH):
                file_name = os.listdir(FILEPATH)[0]
                documents.append(Document(FILEPATH, file_name, current_corpus))
                subjects[i].add_document(documents[-1])

        # Initialize the corpus with respective files
        corpus = Corpus(current_corpus, documents, config)

        # Process all files in the corpus
        corpus.preprocess_all_documents()

        corpus.extract_logits()