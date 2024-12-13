import random
import numpy as np

class OptionGenerator:
    @staticmethod
    def generate_options(subject):
        """
        Generate options based on subject's baseline and variance using a normal distribution.
        Returns a dictionary with the following fields:
            - How are you feeling today? (Float between 0 and 1)
            - How did you sleep tonight? (Float between 0 and 1)
            - How happy are you today? (Float between 0 and 1)
            - How anxious are you today? (Float between 0 and 1)
            - Did you take your antipsychotic medication yesterday? (Int between 0 and 1)
            - Did you take any recreational drugs yesterday? (Int between 0 and 1)
        """
        
        baseline = subject.baseline
        variance = subject.variance
        
        def sample_normal(baseline, variance):
            return round(max(0, min(1, np.random.normal(loc=baseline, scale=variance))), 2)
        
        return {
            "wellbeing": sample_normal(baseline, variance),
            "sleep": sample_normal(baseline, variance),
            "happiness": sample_normal(baseline, variance),
            "anxiety": sample_normal(baseline, variance),
            "medication": round(max(0, min(1, sample_normal(baseline, variance),))),
            "drugs": round(max(0, min(1, sample_normal(baseline, variance),)))
        }