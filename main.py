import json
import os

from config import Config
from simu import SubjectConfig
from simu import PipelineSetup
from simu import OptionGenerator
from simu import TextGenerator

if __name__ == "__main__":
    config = Config()
    setup = PipelineSetup(config)
    subjects = [SubjectConfig(i+1, config.inter_subject_variance, config.intra_subject_variance) for i in range(config.num_subjects)]   
    
    for subject in subjects:
        os.makedirs('simu_output', exist_ok=True)
        file_name = os.path.join('simu_output', f"{config.output_format}_subject_{subject.subject_id}.json")
        
        with open(file_name, 'w') as file:
            for sample in range(config.num_samples):
                
                # json dump subject_config
                json.dump(OptionGenerator.generate_options(subject))
                json.dump(TextGenerator.generate_text(setup, subject))
                # json dump options, text
                
                # (optional: generate_speech)