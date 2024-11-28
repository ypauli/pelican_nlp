import torch
from datetime import datetime

class Config:
    
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            raise RuntimeError("CUDA is not available. Please ensure you have a compatible GPU and CUDA installed.")
        
        self.model = "jphme/em_german_leo_mistral"
        
        self.parameters = {
            "prompt": [
                "Ich erzähle ihnen jetzt eine Geschichte",
                "Es war einmal",
                "Mein grösstes Hobby ist"
                ],
            "temperature": [1.5, 2.5],
            "num_beams": [2],
            
            "retroactive_span": [100, 20], # choose retroactive_span = target_length for unbounded context
            "proactive_span": [20, 50],
            
            "sampling": {
                "top_p": [0.2, 0.6, 1.0],
                "top_k": [4, 16, 32, 64],
                "typical_p": [0.2, 0.6, 1.0]
            } ,
        }
       
        self.constants = {
            "target_length": 100 # including the length of the prompt
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
        self.file = f"output_{timestamp}.json"
