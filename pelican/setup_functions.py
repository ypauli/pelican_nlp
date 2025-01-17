import os
from pelican.preprocessing import Subject

def subject_instatiator(config):
    path_to_subjects = os.path.join(config['PATH_TO_PROJECT_FOLDER'], 'Subjects')
    print('Instantiating Subjects...')
    subjects = [Subject(subject) for subject in os.listdir(path_to_subjects)]

    # Identify sessions per subject
    if config['multiple_sessions']:
        for subject in subjects:
            subject.numberOfSessions = len(os.listdir(os.path.join(path_to_subjects, str(subject.subjectID))))

    return subjects