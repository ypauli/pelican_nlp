import numpy as np

class SubjectConfig:
    def __init__(self, subject_id, inter_subject_variance, intra_subject_variance, mean_baseline=0.5, mean_variance=0.2):
        self.subject_id = subject_id
        self.intra_subject_variance = intra_subject_variance
        self.mean_variance = mean_variance
        self.baseline = max(0, min(1, np.random.normal(loc=mean_baseline, scale=np.sqrt(inter_subject_variance))))
        
    @property
    def variance(self):
        return max(0, min(1, np.random.normal(loc=self.mean_variance, scale=np.sqrt(self.intra_subject_variance))))