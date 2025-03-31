import audiofile
import pandas as pd

class AudioFeatureExtraction:

    @staticmethod
    def opensmile_extraction(file, opensmile_configurations):
        print(f'opensmile extraction in progress...')
        import opensmile

        print(f'audio file is: {file}')

        result = {}
        signal, sampling_rate = audiofile.read(
            file,
            always_2d=True,
            duration=opensmile_configurations['duration'],
            offset=opensmile_configurations['offset']
        )

        # extract eGeMAPSv02 feature set
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        print(smile.feature_names)

        output = smile.process_signal(
            signal,
            sampling_rate
        )
        print(output)

        result['p_nr'] = file[0:4]
        for feature in smile.feature_names:
            result[feature] = output[feature]

        print(f'opensmile restults: {result}')
        return result

    @staticmethod
    def extract_prosogram_profile(file):
        """
        Extract prosodic features using prosogram through parselmouth.
        
        Args:
            file (str): Path to the speech file.
            
        Returns:
            profile (DataFrame): Prosogram analysis results
        """
        import parselmouth
        from pelican.Praat_setup import PROSOGRAM_SCRIPT
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