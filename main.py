from itertools import product
import json

from config import Config
from simu import Setup
from simu import Generator

if __name__ == "__main__":
    
    config = Config()
    setup = Setup(config)
    
    with open(config.file, 'w') as file:
        
        for parameter in setup.parameters:
            
            print(parameter)
            
            parameter["text"] = Generator(setup, config.device, parameter, config.constants).out
            # "punctuation tokens": punctuation_mask
            
            print(parameter["text"])
            
            json.dump(parameter, file, indent=4)