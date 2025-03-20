import audiofile
import opensmile
import pandas as pd

class AudioFeatureExtraction:

    @staticmethod
    def opensmile_extraction(file, opensmile_configurations):

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

        print(result)
        return result

    @staticmethod
    def extract_prosogram_profile(file, praat_configurations):
        import parselmouth
        try:
            sound = parselmouth.Sound(file)
            result = parselmouth.praat.run_file(praat_configurations['praat_path']+'prosomain.praat', sound)

            # Convert result into a DataFrame (assuming it's tab-separated text output)
            profile = pd.read_csv(pd.compat.StringIO(result), sep="\t")

            return profile

        except Exception as e:
            print(f"Error processing {file}: {e}")
            return None
