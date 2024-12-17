import json
import os

from config import Config
from simu import ParameterGenerator
from simu import PipelineSetup
from simu import OptionGenerator
from simu import TextGenerator

if __name__ == "__main__":
    config = Config()
    setup = PipelineSetup(config)
    
    for subject in range(config.num_subjects):
        os.makedirs('simu_output', exist_ok=True)
        file_name = os.path.join('simu_output', f"subject_{subject}.json")
        
        with open(file_name, 'w') as file:
            for timepoint in range(config.num_timepoints):
                options = OptionGenerator.generate_options(timepoint, config, setup.covariances)
                json.dump(options, file, indent=4)
                file.write('\n')

                parameters, generation_arguments = ParameterGenerator().generate_parameters(config, setup, options)
                json.dump(parameters, file, indent=4)
                file.write('\n')

                text = TextGenerator(setup, parameters, generation_arguments).generate_text(config.prompts)
                json.dump(text, file, indent=4)
                file.write('\n')