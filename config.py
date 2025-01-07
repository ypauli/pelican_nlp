from datetime import datetime
import numpy as np

class Config:
    def __init__(self):
        self.num_subjects = 10
        self.num_timepoints = 5
        self.mean_values = [0.6, 0.7, 0.5, 0.5]  # wellbeing, sleep, happiness, anxiety
        self.correlation_matrix = np.array([
            [1.0,  0.5,  0.6, -0.6],  # Wellbeing correlations
            [0.5,  1.0,  0.4, -0.5],  # Sleep correlations
            [0.6,  0.4,  1.0, -0.4],  # Happiness correlations
            [-0.6, -0.5, -0.4,  1.0]  # Anxiety correlations
        ])
        self.std_devs = [0.15, 0.1, 0.12, 0.2]
        self.binary_probs = (0.7, 0.2)  # 70% for medication, 20% for drugs
        self.parameter_rules = {
            "temperature": lambda options: round(
                0.8 + (5 - 0.8) * (1 - options["wellbeing"]), 2
            ), # Temperature: Between 0.8 and 5, higher wellbeing -> lower temperature
            "num_beams": lambda options: 5,
            "sampling": lambda options: (
                "top_p",
                round(
                    0.8 + (2 - 0.8) * (1 - (0.5 * options["sleep"] - 0.5 * options["anxiety"])),
                    
                ),
            ), # Sampling: ("top_p", val) where val is between 0.8 and 2, mean 1, lower sleep and higher anxiety -> higher sampling value
            "retroactive_span": lambda options: int(
                200 - (170 * (1 - (0.5 * options["sleep"] - 0.5 * options["anxiety"])))
            ), # Retroactive Span: Between 30 and 200, mean 60, lower sleep and higher anxiety -> lower span
            "proactive_span": lambda options: int(
                200 - (170 * (1 - (0.5 * options["sleep"] - 0.5 * options["anxiety"])))
            ), # Proactive Span: Same logic as retroactive_span
            "target_length": lambda options: 200,
            "token_noise_rate": lambda options: round(options["anxiety"] * 0.2, 2),
            "lie_rate": lambda options: 0,
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