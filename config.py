import torch
from datetime import datetime

class Config:
    
    def __init__(self):
        self.model = "jphme/em_german_leo_mistral"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
        self.file = f"output_{timestamp}.json"
       
        self.constants = {
            "target_length": 200,
            "continuous_parameters": False
        }
        
        self.parameters = {
            "prompt": [
                "Seit letzter Woche hat sich in meinem Leben einiges verändert",
                "Mein letzter Traum war",
                "Der kürzeste Weg von hier aus zum nächsten Supermarkt ist",
                "Ich werde so viele Tiere aufzählen wie möglich",
                ],
            "temperature": [2.0],
            "num_beams": [3],
            "retroactive_span": [20],
            "proactive_span": [20],
            "sampling": {
                "top_p": [0.6],
                # "top_k": [],
                # "typical_p": []
            },
            "token_noise_rate": [0],
            "lie_rate": [0], 
            "truthfulness_penalty": [0]
        }
        
        self.continuous_parameters = {
            "prompt": [
                "Ich erzähle ihnen jetzt eine Geschichte",
                "Es war einmal",
                "Mein grösstes Hobby ist"
                ],
            "temperature": [3, 1.5, 5.5], # sample temperature[0] values between temperature[1] and temperature[2]
            "num_beams": [2, 2, 5], # sample num_beams[0] integer values between num_beams[1] and num_beams[2]
            "retroactive_span": [2, 20, 100], # sample retroactive_span[0] values between retroactive_span[1] and retroactive_span[2]
            "proactive_span": [2, 20, 50], # sample proactive_span[0] values between proactive_span[1] and proactive_span[2]
            "sampling": {
                "top_p": [2, 0.6, 1.0], # sample top_p[0] values between top_p[1] and top_p[2]
                "top_k": [2, 16, 64], # sample top_k[0] values between top_k[1] and top_k[2]
                "typical_p": [2, 0.6, 1.0] # sample typical_p[0] values between typical_p[1] and typical_p[2]
            },
            "token_noise_rate": [1, 0.1, 0.5], # sample token_noise_rate[0] values between token_noise_rate[1] and token_noise_rate[2]
        }
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            raise RuntimeError("CUDA is not available. Please ensure you have a compatible GPU and CUDA installed.")