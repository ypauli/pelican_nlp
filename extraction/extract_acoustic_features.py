import os
import numpy as np
import pandas as pd
from pydub import AudioSegment
from pyannote.audio import Model, Inference


class AudioFeatureExtractor:
    def __init__(self, model_name, token, device="cpu"):
        """
        Initializes the AudioFeatureExtractor class.

        Parameters:
        - model_name: str, name of the pretrained model from pyannote
        - token: str, the Hugging Face authentication token for downloading the model
        - device: str, device to run the model on (default is "cpu")
        """
        self.model = Model.from_pretrained(model_name, use_auth_token=token).to(device)

    def extract_audio_window(self, audio, start_time=0, duration=60000):
        """
        Extract a segment from the audio starting at `start_time` with a specified duration.
        
        Parameters:
        - audio: AudioSegment object, the input audio
        - start_time: int, starting point of the window in milliseconds (default is 0)
        - duration: int, duration of the window to extract in milliseconds (default is 60000)
        
        Returns:
        - AudioSegment object of the extracted window
        """
        end_time = start_time + duration
        return audio[start_time:end_time]

    def extract_embeddings(self, inference, file_path):
        """
        Extract embeddings from an audio file using the inference model.
        
        Parameters:
        - inference: Inference object from pyannote
        - file_path: str, path to the audio file
        
        Returns:
        - numpy array of embeddings
        """
        embeddings = inference(file_path)
        return np.asarray(embeddings)

    def process_audio(self, file_path, mode="whole", start_time=0, duration=60000, window_step=None):
        """
        Process an audio file, extracting either whole or windowed embeddings based on mode.
        
        Parameters:
        - file_path: str, path to the audio file
        - mode: str, "whole" for whole file extraction or "window" for windowed extraction (default is "whole")
        - start_time: int, start time for the audio segment in milliseconds (only for window mode, default is 0)
        - duration: int, duration for the audio segment in milliseconds (default is 60000)
        - window_step: int, step size for window extraction in milliseconds (only for "window" mode)
        
        Returns:
        - numpy array of embeddings
        """
        audio = AudioSegment.from_file(file_path)
        if mode == "whole":
            inference = Inference(self.model, window="whole")
            embeddings = self.extract_embeddings(inference, file_path)
        elif mode == "window":
            # If window mode is specified, we extract in a sliding window fashion
            embeddings = []
            inference = Inference(self.model, window="sliding", duration=duration, step=window_step)
            
            # Split audio into windows and extract embeddings for each window
            for i in range(0, len(audio), window_step):
                window_audio = self.extract_audio_window(audio, start_time=i, duration=duration)
                temp_path = f"temp_window_{i}.wav"
                window_audio.export(temp_path, format="wav")
                window_embeddings = self.extract_embeddings(inference, temp_path)
                embeddings.append(window_embeddings)
                os.remove(temp_path)
            embeddings = np.vstack(embeddings)  # Stack all window embeddings
        else:
            raise ValueError("Invalid mode. Use 'whole' or 'window'.")

        return embeddings

    def save_embeddings(self, embeddings, output_path):
        """
        Save the embeddings to a CSV file.
        
        Parameters:
        - embeddings: numpy array of embeddings
        - output_path: str, path to save the CSV file
        """
        df = pd.DataFrame(embeddings)
        df.to_csv(output_path, index=False)

# Example usage:
if __name__ == "__main__":
    # Initialize the extractor
    extractor = AudioFeatureExtractor(
        model_name="pyannote/embedding",
        token="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",
        device="mps"
    )

    # Process a whole file
    whole_embeddings = extractor.process_audio(
        file_path="path/to/audio_file.wav",
        mode="whole"
    )

    # Process a file using sliding window extraction
    window_embeddings = extractor.process_audio(
        file_path="path/to/audio_file.wav",
        mode="window",
        start_time=0,
        duration=10000,  # e.g., 10 seconds window
        window_step=5000  # e.g., 5 seconds step
    )

    # Save the embeddings
    extractor.save_embeddings(whole_embeddings, "path/to/output_whole.csv")
    extractor.save_embeddings(window_embeddings, "path/to/output_window.csv")
    
    
    
import os
import numpy as np
from pydub import AudioSegment
from pyannote.audio import Model, Inference


class AudioFeatureExtractor:
    def __init__(self, model_name_or_instance, device="cpu", use_auth_token=None):
        """
        Initializes the AudioFeatureExtractor class.

        Parameters:
        - model_name_or_instance: str or Model, the name of the pretrained model from pyannote or an instance of Model
        - device: str, device to run the model on (default is "cpu")
        - use_auth_token: str, Hugging Face authentication token if required
        """
        if isinstance(model_name_or_instance, str):
            self.model = Model.from_pretrained(
                model_name_or_instance, use_auth_token=use_auth_token
            ).to(device)
        else:
            self.model = model_name_or_instance.to(device)
        self.device = device

    def extract_audio_window(self, audio, start_time=0, duration=None):
        """
        Extract a segment from the audio starting at 'start_time' with a specified 'duration'.

        Parameters:
        - audio: AudioSegment object, the input audio
        - start_time: int, starting point of the window in milliseconds (default is 0)
        - duration: int, duration of the window to extract in milliseconds (default is None, till the end)

        Returns:
        - AudioSegment object of the extracted window
        """
        if duration is None:
            duration = len(audio) - start_time
        end_time = start_time + duration
        return audio[start_time:end_time]

    def extract_embeddings(self, inference, file_path):
        """
        Extract embeddings from the audio file using the specified inference model.

        Parameters:
        - inference: Inference object from pyannote
        - file_path: str, path to the audio file

        Returns:
        - numpy array of embeddings
        """
        embeddings = inference(file_path)
        return np.asarray(embeddings)

    def process_audio(self, file_path, mode="whole", window_duration=None, window_step=None, start_time=0, end_time=None):
        """
        Process an audio file, extracting embeddings based on the specified mode.

        Parameters:
        - file_path: str, path to the audio file
        - mode: str, "whole" for whole file extraction or "windowed" for windowed extraction (default is "whole")
        - window_duration: int, duration of the window in milliseconds (required for "windowed" mode)
        - window_step: int, step size in milliseconds between windows (required for "windowed" mode)
        - start_time: int, start time in milliseconds for processing (default is 0)
        - end_time: int, end time in milliseconds for processing (default is None, till the end)

        Returns:
        - numpy array of embeddings
        """
        # Load and optionally trim the audio file
        audio = AudioSegment.from_file(file_path)
        if end_time is None or end_time > len(audio):
            end_time = len(audio)
        audio = audio[start_time:end_time]

        # Export the (possibly trimmed) audio to a temporary file
        temp_dir = "temp_audio"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, "temp_audio.wav")
        audio.export(temp_path, format="wav")

        if mode == "whole":
            inference = Inference(self.model, window="whole")
            embeddings = self.extract_embeddings(inference, temp_path)
        elif mode == "windowed":
            if window_duration is None or window_step is None:
                raise ValueError("window_duration and window_step must be specified for 'windowed' mode.")
            # Convert milliseconds to seconds for pyannote
            window_duration_sec = window_duration / 1000.0
            window_step_sec = window_step / 1000.0
            inference = Inference(
                self.model,
                window="sliding",
                duration=window_duration_sec,
                step=window_step_sec
            )
            embeddings = self.extract_embeddings(inference, temp_path)
        else:
            os.remove(temp_path)
            raise ValueError("Invalid mode. Use 'whole' or 'windowed'.")

        # Clean up temporary file
        os.remove(temp_path)
        return embeddings

    def save_embeddings(self, embeddings, output_path):
        """
        Save the embeddings to a file.

        Parameters:
        - embeddings: numpy array, the embeddings to save
        - output_path: str, the path where embeddings will be saved
        """
        np.save(output_path, embeddings)
        # Alternatively, to save as CSV:
        # np.savetxt(output_path, embeddings, delimiter=",")


# Example usage:
if __name__ == "__main__":
    # Initialize the extractor with a model name and token if required
    extractor = AudioFeatureExtractor(
        model_name_or_instance="pyannote/embedding",
        device="cpu",
        use_auth_token="YOUR_HUGGING_FACE_TOKEN"  # Replace with your token if necessary
    )

    # Path to your audio file
    audio_file_path = "path/to/your/audio_file.wav"

    # Extract embeddings from the whole audio file
    whole_embeddings = extractor.process_audio(
        file_path=audio_file_path,
        mode="whole"
    )

    # Save the embeddings
    extractor.save_embeddings(whole_embeddings, "whole_embeddings.npy")

    # Extract embeddings using sliding windows
    windowed_embeddings = extractor.process_audio(
        file_path=audio_file_path,
        mode="windowed",
        window_duration=5000,  # Window duration in milliseconds (e.g., 5000 ms = 5 seconds)
        window_step=1000       # Window step in milliseconds (e.g., 1000 ms = 1 second)
    )

    # Save the windowed embeddings
    extractor.save_embeddings(windowed_embeddings, "windowed_embeddings.npy")