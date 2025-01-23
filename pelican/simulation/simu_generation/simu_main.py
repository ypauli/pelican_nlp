import os
import json
import time
import simu_config
import generate_parameter
import setup_pipeline
import generate_text

def initialize_directories(base_dir):
    """Ensure the required directories exist."""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)
    return {
        "Subjects": os.path.join(base_dir, "Subjects"),
        "Metadata": os.path.join(base_dir, "Metadata"),
        "ProgressFile": os.path.join(base_dir, "progress.json")
    }

def load_or_initialize_progress(progress_file):
    """Load progress from a file or create a new progress structure."""
    if os.path.exists(progress_file):
        with open(progress_file, "r") as file:
            return json.load(file)
    progress = {"subjects": {}}
    save_json(progress, progress_file)
    return progress

def save_json(data, file_path):
    """Save data to a JSON file."""
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)

def load_metadata(metadata_file):
    """Load metadata for a subject if it exists, else return None."""
    if os.path.exists(metadata_file):
        with open(metadata_file, "r") as file:
            return json.load(file)
    return None

def create_metadata(subject, group_name, constants, varied_param):
    """Create initial metadata for a subject and group."""
    return {
        "subject": subject,
        "group": group_name,
        "constants": constants,
        "varied_param": varied_param,
        "timepoints": []
    }

def process_timepoint(timepoint, subject, session_id, group_name, group_dir, varied_param, constants, config, setup, progress, metadata):
    """Process a single timepoint for a subject and group."""
    if progress["subjects"][str(subject)]["groups"].get(group_name, {}).get(str(timepoint), False):
        print(f"Skipping timepoint {timepoint} for subject {subject} in group {group_name}, already completed.")
        return

    print(f"Generating data for subject {subject}, group {group_name}, timepoint {timepoint}")

    varied_param_value = generate_parameter.ParameterGenerator.timepoint_sample(config, varied_param)
    parameters = {**constants, varied_param: varied_param_value}
    parameters = generate_parameter.ParameterGenerator.clean_parameters(parameters)

    # wellbeing_factors = generate_parameter.ParameterGenerator.state_sample(parameters, config.parameter_rules)
    generation_arguments = generate_parameter.ParameterGenerator.generate_arguments(parameters, setup)
    text_generator = generate_text.TextGenerator(setup, config.prompts, parameters, generation_arguments)
    generated_text = text_generator.out

    timepoint_file_path = os.path.join(
        group_dir, f"sub-{subject}_ses-{session_id}_group-{group_name}_timepoint-{timepoint}.txt"
    )
    os.makedirs(group_dir, exist_ok=True)
    with open(timepoint_file_path, "w") as timepoint_file:
        for idx, prompt in enumerate(generated_text):
            timepoint_file.write(f"New Prompt: prompt_{idx}:\n")
            timepoint_file.write(f"{generated_text[prompt]}\n\n")

    metadata["timepoints"].append({
        "timepoint": timepoint,
        "varied_param_value": varied_param_value,
        # "wellbeing_factors": wellbeing_factors
    })
    progress["subjects"][str(subject)]["groups"].setdefault(group_name, {})[str(timepoint)] = True

def process_subject(subject, session_id, config, setup, directories, progress):
    """Process a single subject across all groups and timepoints."""
    metadata_dir = os.path.join(directories["Metadata"], f"subject_{subject}")
    subject_dir = os.path.join(directories["Subjects"], f"subject_{subject}",  f"ses-{session_id}")
    os.makedirs(metadata_dir, exist_ok=True)
    
    # Initialize subject progress if it doesn't exist, else get constants
    constants = generate_parameter.ParameterGenerator.subject_sample(config)
    subject_metadata_file = os.path.join(metadata_dir, "metadata.json")
    saved_metadata = load_metadata(subject_metadata_file)

    # If generation is continued, get subject constants
    if saved_metadata:
        constants = saved_metadata.get("constants", constants)
        print(f"Fetched subject metadata")
        
    # Ensure the subject is initialized in the progress dictionary
    if str(subject) not in progress["subjects"]:
        progress["subjects"][str(subject)] = {"groups": {}}

    for group_name, group_config in config.groups.items():
        print(f"Processing subject: {subject}, group: {group_name}")

        group_dir = os.path.join(subject_dir, group_name)
        group_metadata_dir = os.path.join(metadata_dir, group_name)
        os.makedirs(group_dir, exist_ok=True)
        os.makedirs(group_metadata_dir, exist_ok=True)
        metadata_file = os.path.join(group_metadata_dir, "metadata.json")

        varied_param = config.groups[group_name]
        metadata = create_metadata(subject, group_name, constants, varied_param)

        for timepoint in range(config.timepoints):
            process_timepoint(timepoint, subject, session_id, group_name, group_dir, varied_param, constants, config, setup, progress, metadata)

        save_json(metadata, metadata_file)
        save_json(progress, directories["ProgressFile"])

if __name__ == "__main__":
    config = simu_config.SimuConfig()
    setup = setup_pipeline.PipelineSetup(config)
    start_time = time.time()

    directories = initialize_directories(config.directory)
    progress = load_or_initialize_progress(directories["ProgressFile"])
    
    for subject in range(config.subjects_start, config.subjects_end +1):
        print(f"Processing subject: {subject}")
        for session_id in range(config.sessions):
            process_subject(subject, session_id, config, setup, directories, progress)

    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")