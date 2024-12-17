from datetime import datetime
import numpy as np

class Config:
    def __init__(self):
        self.num_subjects = 10
        self.num_timepoints = 5
        self.mean_values = [0.7, 0.6, 0.75, 0.4]  # wellbeing, sleep, happiness, anxiety
        self.correlation_matrix = np.array([
            [1.0,  0.5,  0.6, -0.4],  # Wellbeing correlations
            [0.5,  1.0,  0.4, -0.5],  # Sleep correlations
            [0.6,  0.4,  1.0, -0.6],  # Happiness correlations
            [-0.4, -0.5, -0.6,  1.0]  # Anxiety correlations
        ])
        self.std_devs = [0.1, 0.15, 0.12, 0.2]
        self.binary_probs = (0.7, 0.2)  # 70% for medication, 20% for drugs
        self.parameter_rules = {
            "temperature": lambda options: 0.7 + (1 - options["sleep"]) * 0.3,
            "num_beams": lambda options: max(1, int(3 - options["happiness"] * 2)),
            "sampling": lambda options: ("top_p", 0.9), # Constant sampling
            "retroactive_span": lambda options: 64 if options["medication"] else 32,
            "proactive_span": lambda options: 50,  # Constant proactive span
            "target_length": lambda options: 50, # Constant target length
            "token_noise_rate": lambda options: 0, # lambda options: options["anxiety"] * 0.2,
            "lie_rate": lambda options: 0, # lambda options: 0.5 if options["drugs"] else 0.1,
        }
        self.prompts = {
            "Seit letzter Woche hat sich in meinem Leben einiges verändert",
            "Mein letzter Traum war",
            "Der kürzeste Weg von hier aus zum nächsten Supermarkt ist",
            "Dieses Bild zeigt", # change
            "Ich werde so viele Tiere aufzählen wie möglich" #: Capibara, ...,
            "Ich bin grundsätzlich zufrieden, hätte aber gerne", # change
            "Ich wiederhole jetzt die Geschichte", # change
        }
        self.model_name = "jphme/em_german_leo_mistral"