from itertools import product
import csv
import time
import json

from config import Config
from simu import Setup
from simu import Generator

if __name__ == "__main__":
    
    config = Config()
    setup = Setup(config)
    
    with open('output.json', 'w') as file:
        
        for parameter in setup.parameters:
        
            output, metrics = Generator(setup, config.device, parameter, config.constants).out
            
            parameter_dict = {
                "prompt": parameter[0],
                "temperature": parameter[1],
                "num_beams": parameter[2],
                "retroactive_span": parameter[3][0],
                "proactive_span": parameter[3][1],
                "sampling_method": parameter[4][0],
                "sampling_parameter": parameter[4][1],
                "text": output,
                "probability_differences_tensor_single_generation": metrics[0].tolist(),
                "entropy_tensor_single_generation": metrics[1].tolist(),
                "information_content_tensor_single_generation": metrics[2].tolist(),
                "entropy_deviations_tensor_single_generation": metrics[3].tolist(),
            }
            
            json.dump(parameter_dict, file, indent=4)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # with open(config.file, "w", newline="", encoding="utf-8") as file:
    #     writer = csv.writer(file)
    #     writer.writerow(config.parameters.keys()) # might have to change depending on implementation
        
    #     i = 1
        
    #     for parameter in setup.parameters:
    #         print("Generation: ", i)
    #         print(parameter)
            
    #         start_time = time.time()
    #         output, metrics = Generator(setup, config.device, parameter, config.constants).out
    #         end_time = time.time()

    #         print(f"Time taken for generation: {end_time - start_time:.2f} seconds")
    #         writer.writerow(parameter)
    #         writer.writerow([metrics])
    #         writer.writerow([output])  
    #         writer.writerow(" ")    
            
                     
    #         i+=1