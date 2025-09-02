import os

class AudioFile:
    def __init__(self, file_path, name, **kwargs):
        self.file_path = file_path
        self.name = name
        self.file = os.path.join(file_path, name)

        #Initialize optional attributes
        self.participant_ID = kwargs.get('participant_ID')
        self.task = kwargs.get('task')
        self.num_speakers = kwargs.get('num_speakers')
        self.corpus_name = None
        self.recording_length = None

        self.opensmile_results = None
        self.prosogram_features = None

    def __repr__(self):
        return f"file_name={self.name}"