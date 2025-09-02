import audiofile
import pandas as pd

class AudioFeatureExtraction:

    @staticmethod
    def opensmile_extraction(file, opensmile_configurations):
        print(f'opensmile extraction in progress...')
        import opensmile

        print(f'audio file is: {file}')

        signal, sampling_rate = audiofile.read(
            file,
            always_2d=True,
            duration=opensmile_configurations['duration'],
            offset=opensmile_configurations['offset']
        )

        # extract eGeMAPSv02 feature set
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals
        )

        output = smile.process_signal(
            signal,
            sampling_rate
        )

        # Create result dictionary with only the values we want
        result = {}
        result['participant_ID'] = None  # This will be set by the calling function
        for feature in smile.feature_names:
            # Extract just the numerical value from the output
            result[feature] = float(output[feature].values[0])

        # Get recording length from the index
        recording_length = output.index[0][1].total_seconds()  # This gets the end time from the MultiIndex

        return result, recording_length

    @staticmethod
    def extract_prosogram_profile(file, output_dir=None):
        """
        Extract prosodic features using prosogram through parselmouth.

        Args:
            file (str): Path to the audio file
            output_dir (str): Directory where prosogram output files should be created
        
        Returns:
            profile (DataFrame): Prosogram analysis results
        """
        import parselmouth
        import os
        from pelican_nlp.praat import PRAAT_SCRIPTS_DIR
        
        try:
            # If no output directory specified, use the same directory as the input file
            if output_dir is None:
                output_dir = os.path.dirname(file)
            
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Copy audio file to output directory
            import shutil
            base_name = os.path.splitext(os.path.basename(file))[0]
            audio_extension = os.path.splitext(file)[1]
            copied_audio_path = os.path.join(output_dir, base_name + audio_extension)
            shutil.copy2(file, copied_audio_path)
            
            # Change to the praat scripts directory so includes work
            original_cwd = os.getcwd()
            os.chdir(PRAAT_SCRIPTS_DIR)
            
            # Create inline Praat script content
            praat_script = f"""include prosomain.praat
@prosogram: "file={os.path.abspath(copied_audio_path)} save=yes draw=no"
exit"""
            
            # Run the inline script using parselmouth's run method
            result = parselmouth.praat.run(
                praat_script,
                capture_output=True
            )
            
            # Restore original working directory
            os.chdir(original_cwd)
            
            # Clean up: delete the copied audio file
            if os.path.exists(copied_audio_path):
                os.remove(copied_audio_path)
            
            # List of all possible prosogram output files
            output_files = [
                "_profile_data.txt",  # prosodic profile data (main output)
                "_profile.txt",       # prosodic profile report
                "_data.txt",          # syllable data
                "_table.txt",         # long format syllabic features
                "_styl.txt",          # stylization targets
                "_eval.txt"           # evaluation file
            ]
            
            # Check which files were actually created and read them
            created_files = {}
            for suffix in output_files:
                file_path = os.path.join(output_dir, base_name + suffix)
                if os.path.exists(file_path):
                    try:
                        # Try to read as CSV first (for structured data)
                        if suffix in ["_profile_data.txt", "_data.txt", "_table.txt"]:
                            df = pd.read_csv(file_path, sep="\t")
                            created_files[suffix] = df
                        else:
                            # For text files, read as plain text
                            with open(file_path, 'r', encoding='utf-8') as f:
                                created_files[suffix] = f.read()
                    except Exception as e:
                        print(f"Warning: Could not read {file_path}: {e}")
                        # If CSV reading fails, read as text
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                created_files[suffix] = f.read()
                        except:
                            pass
            
            if not created_files:
                raise Exception(f"No prosogram output files found for {file}")
            
            # Return the main profile data if available, otherwise return all files
            if "_profile_data.txt" in created_files:
                return created_files["_profile_data.txt"]
            else:
                # If main profile data not available, return all files as a dictionary
                return created_files
            
        except Exception as e:
            # Make sure to restore working directory even if there's an error
            try:
                os.chdir(original_cwd)
            except:
                pass
            print(f"Error processing {file}")
            print(f"Full error message: {str(e)}")
            raise