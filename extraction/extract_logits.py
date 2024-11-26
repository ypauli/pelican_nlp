import torch
import torch.nn.functional as F
import pandas as pd
import time
import os
from tqdm import tqdm
from accelerate import Accelerator, dispatch_model, infer_auto_device_map, init_empty_weights
from transformers import AutoModelForCausalLM
from torch.amp import autocast
from extraction.LanguageModel import Model
from config import Config

class LogitsExtractor:
    def __init__(self, model_name, pipeline, project_path):

        self.device = 'cuda' if torch.cuda.is_available()==True else 'cpu'
        self.model_name = model_name
        self.pipeline = pipeline
        self.PROJECT_PATH = project_path

    def extract_features(self, tokens, chunk_size=128, overlap_size=64):

        model = Model(self.model_name, self.PROJECT_PATH)
        model.load_model()

        input_ids = tokens.to(self.device)
        chunks = self._split_into_chunks(input_ids, chunk_size, overlap_size)

        per_token_data = []

        total_processed_tokens = 0  # Keep track of total tokens processed to avoid duplicates

        for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):

            with torch.no_grad():
                outputs = model.model_instance(input_ids=chunk)
                logits = outputs.logits  # Shape: [1, seq_length, vocab_size]

            tokens = self.pipeline.tokenizer.convert_IDs_to_tokens(chunk.squeeze())
            num_tokens = len(tokens)

            chunk_data = []

            # Determine the starting index for predictions
            if i == 0:
                # For the first chunk, start from index 1 (since the first token has no previous context)
                start_idx = 1
            else:
                # For subsequent chunks, skip tokens that were already processed in the overlap
                start_idx = overlap_size

            # Loop over the tokens to predict
            for j in range(start_idx, num_tokens):
                # Compute per-token metrics
                per_token_metrics = self._compute_per_token_metrics(logits, chunk, tokens, j)
                chunk_data.append(per_token_metrics)

            # Append the chunk data to the per_token_data list
            per_token_data.extend(chunk_data)
            total_processed_tokens += len(chunk_data)

        self._store_features_to_csv(per_token_data)

        return per_token_data

    def _compute_per_token_metrics(self, logits, chunk, tokens, j):

        # The model_instance predicts the token at position j using tokens up to position j-1
        # Therefore, logits at position j-1 correspond to predictions for token at position j
        token_logits = logits[:, j - 1, :]  # Shape: [1, vocab_size]
        token_probs = F.softmax(token_logits, dim=-1)
        token_logprobs = F.log_softmax(token_logits, dim=-1)

        actual_token_id = chunk[:, j]  # The actual token at position j
        logprob_actual = token_logprobs[0, actual_token_id].item()
        max_logprob, max_token_id = torch.max(token_logprobs, dim=-1)
        max_logprob = max_logprob.item()
        max_token_id = max_token_id.item()
        entropy = -(token_probs * token_logprobs).sum().item()

        most_likely_token = self.pipeline.tokenizer.convert_IDs_to_tokens([max_token_id])[0]
        token = tokens[j]  # The token at position j

        return {
            'token': token,
            'logprob_actual': logprob_actual,
            'logprob_max': max_logprob,
            'entropy': entropy,
            'most_likely_token': most_likely_token
        }

    def _split_into_chunks(self, input_ids, chunk_size, overlap_size):

        input_ids = input_ids.squeeze()
        input_length = input_ids.size(0)
        stride = chunk_size - overlap_size
        chunks = []

        for i in range(0, input_length, stride):
            end_index = min(i + chunk_size, input_length)
            chunk = input_ids[i:end_index]
            chunks.append(chunk.unsqueeze(0).to(self.device))
            if end_index == input_length:
                break

        return chunks


    def _store_features_to_csv(self, per_token_data, filename='test_features.csv'):
        #saving features in desired format
        return

def process_folder(folder_path, extractor, chunk_size, overlap_size):
    """
    Process all text files in the specified folder, extract information from filenames,
    and collect results into a DataFrame.

    Parameters:
    - folder_path: The path to the folder containing text files.
    - extractor: An instance of LogitsExtractor.
    - chunk_size: The size of chunks to split the input text for processing.
    - overlap_size: The number of tokens to overlap between chunks.

    Returns:
    - df_results: A pandas DataFrame containing the filename information and language features.
    """
    data = []
    failed_files = []

    # List all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            # Read the text content
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            print(f"Processing file: {filename}")
            # Tokenize the text to get total number of tokens
            input_ids = extractor.tokenizer.encode(text, return_tensors='pt')
            total_num_tokens = input_ids.size(1)
            print(f"Total number of tokens: {total_num_tokens}")

            try:
                tic = time.time()
                # Process the text to extract features
                per_token_data = extractor.extract_features(text, chunk_size=chunk_size, overlap_size=overlap_size)
                toc = time.time()
                print(f"Time for logits extraction: {toc - tic:.2f} seconds")
                print(f"Number of per-token data entries: {len(per_token_data)}")
                # Should have total_num_tokens - 1 == len(per_token_data)
                print(f"Tokens excluding the first: {total_num_tokens - 1}")

                # Collect the results
                data.append({
                    'filename': filename,
                    'length': total_num_tokens,
                    'features': per_token_data
                })

            except Exception as e:
                print(f"Error processing file {filename}: {e}")
                failed_files.append(filename)
                continue  # Skip to the next file

    # Create a DataFrame
    df_results = pd.DataFrame(data)
    df_results.to_csv(os.path.join(folder_path, "test_features.csv"), index=False)
    return df_results