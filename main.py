#this is the main project file
#created 1.10.24
#created by Yves Pauli
#=============================
import os
import torch.cuda
import sys
import shutil
import yaml
from document import Document
from preprocessing import Subject, Corpus
from setup_functions import ignore_files

if __name__ == '__main__':

    dev_mode = True

    if torch.cuda.is_available(): torch.cuda.empty_cache()

    #import configuration settings
    with open('config.yml') as stream:
        try:
            configuration_settings = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    PATH_TO_SUBJECTS = configuration_settings['PATH_TO_PROJECT_FOLDER'] + 'Subjects/'
    if not os.path.isdir(PATH_TO_SUBJECTS):
        print('Warning: Could not find subjects; Check folder structure.')
        sys.exit()

    #create output directory
    OUTPUT_DIRECTORY = configuration_settings['PATH_TO_PROJECT_FOLDER'] + 'Outputs/'

    #==============================================
    #remove Outputs folder if it already exists
    #ONLY FOR DEVELOPMENT
    if dev_mode:
        if os.path.isdir(OUTPUT_DIRECTORY):
            shutil.rmtree(OUTPUT_DIRECTORY)
    #==============================================

    shutil.copytree(PATH_TO_SUBJECTS, OUTPUT_DIRECTORY, ignore=ignore_files)
    os.mkdir(OUTPUT_DIRECTORY + 'results_consolidation')

    # Instantiate all subjects
    subjects = []
    for subject in os.listdir(PATH_TO_SUBJECTS):
        subjects.append(Subject(subject))

    for current_corpus in configuration_settings['corpus_names']:

        print(f'{current_corpus} is being processed')

        documents = []
        # load all files belonging to same corpus
        for i in range(len(subjects)):

            #Check if subject has file belonging to corpus and add if available
            FILEPATH = PATH_TO_SUBJECTS + subjects[i].subjectID + '/' + current_corpus
            if os.path.isdir(FILEPATH):
                file_name = os.listdir(FILEPATH)[0]
                documents.append(Document(FILEPATH, file_name, current_corpus))
                subjects[i].add_document(documents[-1])

        # Initialize the corpus with respective files
        corpus = Corpus(current_corpus, documents, configuration_settings)

        # Process all files in the corpus
        corpus.preprocess_all_documents()
        print(f'all files in {current_corpus} have been preprocessed')

        if configuration_settings['extract_logits']:
            print('==================Extracting logits===================')
            corpus.extract_logits()

        if configuration_settings['extract_embeddings']:
            print('================Extracting embeddings=================')
            corpus.extract_embeddings()