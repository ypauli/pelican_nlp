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
    def extract_prosogram_profile(file):
        """
        Extract prosodic features using prosogram through parselmouth.

        Returns:
            profile (DataFrame): Prosogram analysis results
        """
        import parselmouth
        from pelican_nlp.praat import PROSOGRAM_SCRIPT
        try:
            sound = parselmouth.Sound(file)
            # Common Prosogram parameters
            result = parselmouth.praat.run_file(
                PROSOGRAM_SCRIPT,
                arguments=[sound, "save=yes", "draw=no"],
                capture_output=True
            )
            
            # Convert result into a DataFrame with the same format
            profile = pd.read_csv(pd.compat.StringIO(result), sep="\t")
            
            return profile

        except Exception as e:
            print(f"Error processing {file}")
            print(f"Full error message: {str(e)}")
            raise  # This will show the full stack trace
            return None