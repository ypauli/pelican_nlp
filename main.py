from itertools import product
import csv
import time

from config import Config
from simu import Setup
from simu import Generator

if __name__ == "__main__":
    
    config = Config()
    setup = Setup(config)
    
    with open(config.file, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(config.parameters.keys())
        
        i = 1
        
        for parameter in setup.parameters:
            print("Generation: ", i)
            print(parameter)
            
            start_time = time.time()
            output, metrics = Generator(setup, config.device, parameter, config.constants).out
            end_time = time.time()

            print(f"Time taken for generation: {end_time - start_time:.2f} seconds")
            print(parameter)
            writer.writerow(parameter)
            writer.writerow([metrics])
            writer.writerow([output])  
            writer.writerow(" ")             
            i+=1