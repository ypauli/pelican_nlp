from datetime import datetime
import numpy as np

    # parameters = {
    #     "temperature": round(np.clip(sampled_values[0], 0.8, 5.0), 2),
    #     "sampling": ("top_p", round(np.clip(sampled_values[1], 0.75, 2.0), 2)),
    #     "retroactive_span": int(np.clip(sampled_values[2], 30, 300)),
    #     "proactive_span": int(np.clip(sampled_values[3], 30, 300)),
    #     "target_length": int(np.clip(sampled_values[4], 150, 450)), # roughly 100 - 350 words x 1.5 to account for tokens
    #     "token_noise_rate": round(np.clip(sampled_values[5], 0.0, 0.2), 2),
    #     "lie_rate": 0,

class Config:
    def __init__(self):
        self.subjects_per_cohort = 1
        self.timepoints_per_subject = 1
        self.global_parameter_stats = {
            "temperature": {"mean": 1.2, "variance": 0.0234},
            "sampling": {"mean": 0.9, "variance": 0.0026}, # using top-p sampling
            "context_span": {"mean": 150, "variance": 2108.57},
            "target_length": {"mean": 300, "variance": 1275.56},
        }
        self.cohorts = {
            "group_a": {
                "varied_parameter": "temperature",
                "mean_values": {"temperature": 1.2},
                "variance_values": {"temperature": 0.1},
            },
            "group_b": {
                "varied_parameter": "sampling",
                "mean_values": {"sampling": 0.9},
                "variance_values": {"sampling": 0.1},
            },
            "group_c": {
                "varied_parameter": "context_span",
                "mean_values": {"context_span": 150},
                "variance_values": {"context_span": 70},
            },
            "group_d": {
                "varied_parameter": "target_length",
                "mean_values": {"target_length": 300},
                "variance_values": {"target_length": 25},
            }
        }
        self.parameter_rules = {
            "wellbeing": lambda parameters: round(
                0.4 * parameters["sampling"] + 0.4 * (1 - parameters["target_length"]) + 0.2 * parameters["context_span"], 2
            ),
            "sleep": lambda parameters: round(
                1 - (parameters["context_span"] - 30) / (200 - 30), 2
            ),
            "anxiety": lambda parameters: round(
                (200 - parameters["context_span"]) / (200 - 30), 2
            ),
            "happiness": lambda parameters: round(
                1 - (parameters["temperature"] - 0.8) / (5 - 0.8), 2
            ),
            "medication": lambda parameters: int(
                np.random.choice([0, 1], p=[0.3, 0.7])
            ),
            "drugs": lambda parameters: int(
                np.random.choice([0, 1], p=[0.8, 0.2])
            ),
        }
        self.prompts = {
            "Seit letzter Woche hat sich in meinem Leben einiges verändert",
            "Mein letzter Traum war",
            "Von hier aus bis zum nächsten Supermarkt gelangt man am besten",  # "Der kürzeste Weg von hier aus zum nächsten Supermarkt ist",
            # "Dieses Bild zeigt", # change
            "Ich werde so viele Tiere aufzählen wie möglich: Pelikan,"
            # "Ich bin grundsätzlich zufrieden, hätte aber gerne", # change
            # "Ich wiederhole jetzt die Geschichte", # change
        }
        self.model_name = "jphme/em_german_leo_mistral"