import numpy as np
import math

class SimuConfig:
    def __init__(self):
        self.directory = "/home/ubuntu/PELICAN/pelican/simulation/simu_output"
        self.sessions = 1
        self.subjects_start = 0
        self.subjects_end = 1
        self.timepoints = 2
        self.global_parameter_stats = {
            "temperature": {"low": 0, "high": 10},
            "sampling": {"low": 0, "high": 1}, # Using top-p sampling
            "context_span": {"low": 1, "high": 200},
            "target_length": {"low": 1, "high": 200},
        }
        self.model_name = "jphme/em_german_leo_mistral"
        self.prompts = {
            "Seit letzter Woche habe ich",
            "In meinem letzten Traum",
            "Von hier aus bis zum nächsten Supermarkt gelangt man",
            "Ich werde so viele Tiere aufzählen wie möglich: Pelikan,"
        }
        self.groups = {
            "a": "temperature",
            "b": "sampling",
            "c": "context_span",
            "d": "target_length",
        }
        
        
        
        
        
        
        
        # For state calculation, experimental use
        
        self.parameter_weights = {
            "temperature": -0.9,
            "sampling": -1.2,
            "context_span": 0.006,
            "target_length": 0.004,
        }
        self.parameter_rules = {
            # State score normalized between 0 and 1
            "state_score": lambda parameters: round(
                1 / (1 + math.exp(-sum(parameters[param] * weight for param, weight in self.parameter_weights.items()))),
                2
            ),
            # Wellbeing is directly equal to the state_score
            "wellbeing": lambda state_score: round(state_score, 2),
            # Sleep is highest when state_score is close to 0.75
            "sleep": lambda state_score: round(1 - abs(state_score - 0.75), 2),
            # Anxiety is highest when state_score is close to 0.25
            "anxiety": lambda state_score: round(1 - abs(state_score - 0.25), 2),
            # Happiness is the square the state_score
            "happiness": lambda state_score: round(state_score**2, 2),
            # Medication: More likely to be 1 if state_score is high
            "medication": lambda state_score: int(np.random.choice([0, 1], p=[1 - state_score, state_score])),
            # Drugs: Less likely than medication, and more likely to be 1 if state_score is low
            "drugs": lambda state_score: int(
                np.random.choice([0, 1], p=[
                    max(0, min(1, state_score + 0.3)),  # Ensure between 0 and 1
                    max(0, min(1, 0.7 - state_score))  # Ensure between 0 and 1
                ])
            )
        }