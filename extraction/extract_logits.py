import torch
import numpy as np
from collections import defaultdict

class LogitsExtractor:
    def __init__(self, model, tokenizer, metric_functions, aggregation_functions, device='cpu'):
        """
        Initialize the LogitsExtractor.

        Parameters:
        - model: The language model to use for processing text.
        - tokenizer: The tokenizer associated with the model.
        - metric_functions: A dictionary of metric functions.
                            Each function should accept (logits, input_ids) and return a list of metric values.
        - aggregation_functions: A dictionary of aggregation functions.
                                 Each function should accept a list of metric values and return a scalar.
        - device: The device to run the computations on ('cpu' or 'cuda').
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.metric_functions = metric_functions
        self.aggregation_functions = aggregation_functions
        self.device = device

    def process_text(self, text, chunk_size=512, overlap_size=256):
        """
        Process the text, calculate per-token metrics, and aggregate them.

        Parameters:
        - text: The input text string.
        - chunk_size: The size of chunks to split the input text for processing.
        - overlap_size: The number of tokens to overlap between chunks.

        Returns:
        - aggregated_metrics: A dictionary with aggregated metrics.
        - per_token_metrics: A dictionary with per-token metrics.
        - tokens_list: A list of tokens corresponding to the metrics.
        """
        # Tokenize the input text
        input_ids = self.tokenizer.encode(text, return_tensors='pt').to(self.device)

        # Split the input_ids into chunks with overlap
        chunks = self._split_into_chunks(input_ids, chunk_size, overlap_size)

        per_token_metrics = defaultdict(list)
        tokens_list = []

        for i, chunk in enumerate(chunks):
            # Perform a forward pass
            with torch.no_grad():
                outputs = self.model(input_ids=chunk)
                logits = outputs.logits

            # Calculate per-token metrics using the provided metric functions
            for metric_name, metric_function in self.metric_functions.items():
                metric_values = metric_function(logits, chunk)
                # Adjust for overlapping tokens
                if i > 0 and overlap_size > 0:
                    metric_values = metric_values[overlap_size:]
                per_token_metrics[metric_name].extend(metric_values)

            # Keep track of tokens
            tokens = self.tokenizer.convert_ids_to_tokens(chunk.squeeze())
            # Exclude overlapping tokens except for the first chunk
            if i > 0 and overlap_size > 0:
                tokens = tokens[overlap_size:]
            tokens_list.extend(tokens)

        # Aggregate the metrics using the provided aggregation functions
        aggregated_metrics = {}
        for metric_name, values in per_token_metrics.items():
            aggregated_metrics[metric_name] = {}
            for agg_name, agg_function in self.aggregation_functions.items():
                aggregated_metrics[metric_name][agg_name] = agg_function(values)

        return aggregated_metrics, per_token_metrics, tokens_list

    def _split_into_chunks(self, input_ids, chunk_size, overlap_size):
        """
        Split the input_ids into chunks with overlap.

        Parameters:
        - input_ids: The tokenized input ids.
        - chunk_size: The size of each chunk.
        - overlap_size: The number of tokens to overlap between chunks.

        Returns:
        - chunks: A list of input_ids chunks.
        """
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
    
    
def log_probability_metric(logits, input_ids):
    """
    Calculate per-token log probabilities.

    Parameters:
    - logits: The output logits from the model.
    - input_ids: The input ids corresponding to the tokens.

    Returns:
    - log_probs: A list of per-token log probabilities.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    return log_probs.squeeze().tolist()

def entropy_metric(logits, input_ids):
    """
    Calculate per-token entropy.

    Parameters:
    - logits: The output logits from the model.
    - input_ids: The input ids corresponding to the tokens.

    Returns:
    - entropies: A list of per-token entropies.
    """
    probs = torch.nn.functional.softmax(logits[:, :-1, :], dim=-1)
    log_probs = torch.nn.functional.log_softmax(logits[:, :-1, :], dim=-1)
    entropies = -torch.sum(probs * log_probs, dim=-1)
    return entropies.squeeze().tolist()

def perplexity_metric(logits, input_ids):
    """
    Calculate per-token perplexity.

    Parameters:
    - logits: The output logits from the model.
    - input_ids: The input ids corresponding to the tokens.

    Returns:
    - perplexities: A list of per-token perplexities.
    """
    log_probs = log_probability_metric(logits, input_ids)
    perplexities = [np.exp(-lp) if lp is not None else None for lp in log_probs]
    return perplexities

def mean_aggregation(values):
    """Calculate the mean of the values."""
    values = np.array(values)
    return np.mean(values)

def std_aggregation(values):
    """Calculate the standard deviation of the values."""
    values = np.array(values)
    return np.std(values)

def median_aggregation(values):
    """Calculate the median of the values."""
    values = np.array(values)
    return np.median(values)

def min_aggregation(values):
    """Calculate the minimum of the values."""
    values = np.array(values)
    return np.min(values)

def max_aggregation(values):
    """Calculate the maximum of the values."""
    values = np.array(values)
    return np.max(values)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load a pre-trained language model and tokenizer
model_name = 'gpt2'  # Replace with your desired model
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define metric functions and aggregation functions
metric_functions = {
    'log_probability': log_probability_metric,
    'entropy': entropy_metric,
    'perplexity': perplexity_metric
}

aggregation_functions = {
    'mean': mean_aggregation,
    'std': std_aggregation,
    'median': median_aggregation,
    'min': min_aggregation,
    'max': max_aggregation
}

# Initialize the LogitsExtractor
extractor = LogitsExtractor(
    model=model,
    tokenizer=tokenizer,
    metric_functions=metric_functions,
    aggregation_functions=aggregation_functions,
    device='cpu'  # Use 'cuda' if a GPU is available
)

# Input text
text = "The quick brown fox jumps over the lazy dog."

# Process the text
aggregated_metrics, per_token_metrics, tokens = extractor.process_text(
    text,
    chunk_size=16,      # Adjust based on your requirements
    overlap_size=8      # Adjust based on your requirements
)

# Display the results
print("Aggregated Metrics:")
for metric_name, aggs in aggregated_metrics.items():
    print(f"{metric_name}:")
    for agg_name, value in aggs.items():
        print(f"  {agg_name}: {value}")

print("\nPer-Token Metrics:")
for i in range(len(tokens) - 1):  # Exclude the last token which has no prediction
    print(f"Token: {tokens[i+1]}")  # Shifted by 1 due to prediction alignment
    for metric_name in per_token_metrics.keys():
        value = per_token_metrics[metric_name][i]
        print(f"  {metric_name}: {value}")
    print()