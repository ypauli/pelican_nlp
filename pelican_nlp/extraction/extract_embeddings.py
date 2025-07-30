from pelican_nlp.extraction.language_model import Model
from pelican_nlp.preprocessing.text_tokenizer import TextTokenizer

from pelican_nlp.config import debug_print

class EmbeddingsExtractor:
    def __init__(self, embeddings_configurations, project_path):
        self.embeddings_configurations = embeddings_configurations
        self.model_name = embeddings_configurations['model_name']  # Embedding model instance (e.g., fastText, RoBERTa)
        self.model = Model(self.model_name, project_path)
        self.Tokenizer = TextTokenizer(self.embeddings_configurations['tokenization_method'], self.model_name,
                                       self.embeddings_configurations['max_length'])

        self.model.load_model()
        self.model_instance = self.model.model_instance

    def extract_embeddings_from_text(self, text_list):

        doc_entry_list = []

        for text in text_list:

            embeddings = {}

            # Tokenize the input text
            inputs = self.Tokenizer.tokenize_text(text)
            debug_print(f'inputs are: {inputs}')
            debug_print(f'inputs type: {type(inputs)}')

            if self.embeddings_configurations['pytorch_based_model']:
                #e.g. RoBERTa Model or Llama Model
                import torch
                with torch.no_grad():
                    if 'llama' in self.model_name.lower():
                        # Handle Llama models which expect input_ids directly
                        outputs = self.model_instance(input_ids=inputs['input_ids'])
                    else:
                        # Handle RoBERTa and other models that accept **inputs
                        if isinstance(inputs, dict):
                            # Ensure inputs are on the same device as the model
                            inputs = {k: v.to(self.model_instance.device) for k, v in inputs.items()}
                            debug_print(f"Model inputs: {inputs}")
                            outputs = self.model_instance(**inputs, output_hidden_states=True)
                        else:
                            debug_print(f"Input type: {type(inputs)}")
                            debug_print(f"Input content: {inputs}")
                            
                            # Handle BatchEncoding objects from transformers
                            if hasattr(inputs, 'input_ids'):
                                # This is a BatchEncoding object, extract the tensors
                                input_ids = inputs['input_ids'].to(self.model_instance.device)
                                attention_mask = inputs['attention_mask'].to(self.model_instance.device) if 'attention_mask' in inputs else torch.ones_like(input_ids)
                                debug_print(f"Extracted from BatchEncoding - input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape}")
                                outputs = self.model_instance(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                            else:
                                # If inputs is a list of strings, convert to token IDs first
                                if isinstance(inputs, list):
                                    if isinstance(inputs[0], str):
                                        # Convert tokens to IDs
                                        token_ids = self.Tokenizer.tokenizer.convert_tokens_to_ids(inputs)
                                        debug_print(f"Token IDs: {token_ids}")
                                        inputs = torch.tensor([token_ids], device=self.model_instance.device)
                                    else:
                                        # If it's already a list of numbers, convert directly
                                        inputs = torch.tensor([inputs], device=self.model_instance.device)
                                else:
                                    # If it's already a tensor, just move to device
                                    inputs = inputs.to(self.model_instance.device)
                                
                                # Only print shape if inputs is a tensor
                                if hasattr(inputs, 'shape'):
                                    debug_print(f"Final tensor shape: {inputs.shape}")
                                else:
                                    debug_print(f"Inputs is not a tensor, type: {type(inputs)}")
                                
                                # Ensure proper shape
                                if len(inputs.shape) == 1:
                                    inputs = inputs.unsqueeze(0)  # Add batch dimension
                                
                                # Create attention mask
                                attention_mask = torch.ones_like(inputs)
                                debug_print(f"Model inputs - input_ids: {inputs.shape}, attention_mask: {attention_mask.shape}")
                                outputs = self.model_instance(input_ids=inputs, attention_mask=attention_mask, output_hidden_states=True)
                                debug_print(f"Model outputs type: {type(outputs)}")
                                debug_print(f"Model outputs attributes: {dir(outputs)}")

                # Get word embeddings (last hidden state)
                if outputs is None:
                    raise ValueError("Model returned None output")
                
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    word_embeddings = outputs.hidden_states[-1]
                    debug_print(f"Using hidden_states, shape: {word_embeddings.shape}")
                elif hasattr(outputs, 'last_hidden_state'):
                    word_embeddings = outputs.last_hidden_state
                    debug_print(f"Using last_hidden_state, shape: {word_embeddings.shape}")
                else:
                    raise ValueError(f"Model output has neither hidden_states nor last_hidden_state. Available attributes: {dir(outputs)}")

                # Extract input_ids and convert them back to tokens
                if isinstance(inputs, dict):
                    input_ids = inputs['input_ids'][0].tolist()
                elif hasattr(inputs, 'input_ids'):
                    # Handle BatchEncoding objects
                    input_ids = inputs['input_ids'][0].tolist()
                else:
                    input_ids = inputs[0].tolist()
                tokens = self.Tokenizer.tokenizer.convert_ids_to_tokens(input_ids)

                # Now align the tokens and embeddings
                for token, embedding in zip(tokens, word_embeddings[0]):
                    embeddings[token]=embedding.tolist()

            else:
                if self.model_name == 'fastText':
                    embeddings = []
                    for token in inputs:
                        embeddings.append((token, self.model_instance.get_word_vector(token)))

            doc_entry_list.append(embeddings)

        # Calculate token count properly
        if isinstance(inputs, dict):
            token_count = len(inputs['input_ids'][0]) if 'input_ids' in inputs else 0
        elif hasattr(inputs, 'input_ids'):
            # Handle BatchEncoding objects
            token_count = len(inputs['input_ids'][0]) if hasattr(inputs['input_ids'], '__len__') else 0
        elif hasattr(inputs, '__len__'):
            token_count = len(inputs)
        else:
            token_count = 0

        return doc_entry_list, token_count