"""
Topic modeling extraction using BERTopic.

This module provides topic modeling functionality that reuses existing
embeddings and preprocessing from the pipeline.

Supports:
- Topic modeling within single text units (by chunking)
- Comparison of identified topics to predefined topics
"""

import numpy as np
import re
from pelican_nlp.config import debug_print
from pelican_nlp.utils.csv_functions import store_features_to_csv


class TopicModelingExtractor:
    """Extracts topics from documents using BERTopic, reusing existing embeddings."""
    
    def __init__(self, topic_modeling_options, project_folder):
        """
        Initialize TopicModelingExtractor.
        
        Args:
            topic_modeling_options: Configuration dictionary for topic modeling
            project_folder: Path to project folder
        """
        self.options = topic_modeling_options
        self.project_folder = project_folder
        self.model = None
        
        # Initialize BERTopic model
        self._initialize_bertopic_model()
        
    def _initialize_bertopic_model(self):
        """Initialize BERTopic model based on configuration options."""
        try:
            from bertopic import BERTopic
        except ImportError:
            raise ImportError(
                "BERTopic is not installed. Install it with: pip install bertopic"
            )
        
        # Get BERTopic configuration options
        min_topic_size = self.options.get('min_topic_size', 10)
        nr_topics = self.options.get('nr_topics', 'auto')
        calculate_probabilities = self.options.get('calculate_probabilities', False)
        verbose = self.options.get('verbose', True)
        
        # For small datasets, disable automatic topic reduction to avoid errors
        # When nr_topics="auto", BERTopic tries to reduce topics which can fail with small datasets
        # Set to None to disable reduction, or keep "auto" for larger datasets
        if nr_topics == 'auto':
            # For per-document analysis with chunking, disable auto-reduction
            # This will be adjusted in _fit_bertopic_model if needed
            pass
        
        # Initialize BERTopic with custom embeddings (we'll provide embeddings later)
        # Using empty_documents=True since we'll use custom embeddings
        self.model = BERTopic(
            min_topic_size=min_topic_size,
            nr_topics=nr_topics,
            calculate_probabilities=calculate_probabilities,
            verbose=verbose
        )
        
        debug_print("Initialized BERTopic model")
        
    def extract_topics_from_text(self, documents_list, embeddings_list, topic_modeling_options):
        """
        Extract topics from a list of documents using pre-computed embeddings.
        
        This method follows the pattern of extract_embeddings_from_text().
        It reuses embeddings that were already computed by the embeddings extraction step.
        
        Args:
            documents_list: List of document texts (strings)
            embeddings_list: List of embeddings corresponding to documents_list
                           (from existing embeddings extraction)
                           Format: List of lists, where each inner list contains (token, embedding) tuples
            topic_modeling_options: Options for topic modeling configuration
            
        Returns:
            Dictionary containing topic assignments, topic info, and model
        """
        print("Extracting topics using BERTopic...")
        
        # Check if we need to chunk text units for single-unit topic modeling
        chunk_text_units = topic_modeling_options.get('chunk_text_units', False)
        chunk_size = topic_modeling_options.get('chunk_size', 'sentence')  # 'sentence' or integer
        was_chunked = False
        
        if chunk_text_units and len(documents_list) == 1:
            # Chunk the single text unit into smaller pieces
            debug_print("Chunking single text unit for topic modeling...")
            chunked_documents, chunked_embeddings = self._chunk_text_unit(
                documents_list[0], embeddings_list[0], chunk_size
            )
            if len(chunked_documents) > 1:
                documents_list = chunked_documents
                embeddings_list = chunked_embeddings
                was_chunked = True
                debug_print(f"Chunked into {len(chunked_documents)} chunks")
                
                # Reinitialize model with appropriate min_topic_size for chunked data
                # min_topic_size must be at least 2 for HDBSCAN, and should be reasonable for the chunk count
                original_min_topic_size = self.options.get('min_topic_size', 10)
                # Use a reasonable fraction of chunks, but at least 2
                adjusted_min = max(2, min(len(chunked_documents) // 3, original_min_topic_size))
                debug_print(f"Reinitializing model with min_topic_size={adjusted_min} for {len(chunked_documents)} chunks")
                
                # For small datasets (chunked text), disable automatic topic reduction
                # to avoid errors when all documents are outliers
                nr_topics = self.options.get('nr_topics', 'auto')
                if nr_topics == 'auto' and len(chunked_documents) < 20:
                    # Disable auto-reduction for small datasets
                    nr_topics = None
                    debug_print(f"Disabling automatic topic reduction for small dataset ({len(chunked_documents)} chunks)")
                
                from bertopic import BERTopic
                self.model = BERTopic(
                    min_topic_size=adjusted_min,
                    nr_topics=nr_topics,
                    calculate_probabilities=self.options.get('calculate_probabilities', False),
                    verbose=self.options.get('verbose', True)
                )
        
        # Convert existing embeddings to document-level embeddings
        document_embeddings = self._prepare_embeddings_for_bertopic(embeddings_list)
        
        debug_print(f"Prepared {len(document_embeddings)} document embeddings")
        debug_print(f"Embedding dimension: {document_embeddings.shape[1] if len(document_embeddings) > 0 else 0}")
        
        # Fit BERTopic model with custom embeddings
        self._fit_bertopic_model(documents_list, document_embeddings)
        
        # Get topic assignments
        topic_assignments, topic_probabilities = self._get_topic_assignments(
            self.model, documents_list, document_embeddings
        )
        
        # Get topic information
        topic_info = self._get_topic_info(self.model)
        
        # Compare to predefined topics if specified
        predefined_topics = topic_modeling_options.get('predefined_topics', None)
        topic_comparisons = None
        if predefined_topics:
            topic_comparisons = self._compare_to_predefined_topics(
                topic_info, predefined_topics, document_embeddings, topic_assignments
            )
        
        return {
            'model': self.model,
            'topic_assignments': topic_assignments,
            'topic_probabilities': topic_probabilities,
            'topic_info': topic_info,
            'documents': documents_list,
            'topic_comparisons': topic_comparisons
        }
    
    def _prepare_embeddings_for_bertopic(self, embeddings_list):
        """
        Convert existing embeddings format to format expected by BERTopic.
        
        Converts token-level embeddings to document-level embeddings by averaging.
        
        Args:
            embeddings_list: List of embeddings from existing pipeline
                           Format: List of lists, where each inner list contains (token, embedding) tuples
                           or dict of {token: embedding}
            
        Returns:
            Numpy array of document-level embeddings for BERTopic
            Shape: (n_documents, embedding_dim)
        """
        document_embeddings = []
        
        for doc_embeddings in embeddings_list:
            # Handle different embedding formats
            if isinstance(doc_embeddings, list):
                # List of (token, embedding) tuples
                if len(doc_embeddings) == 0:
                    # Empty document, create zero vector (will be handled by BERTopic)
                    # We need to know the embedding dimension, use first non-empty doc
                    continue
                
                # Extract embeddings (second element of each tuple)
                token_embeddings = []
                for item in doc_embeddings:
                    if isinstance(item, tuple) and len(item) == 2:
                        _, embedding = item
                        token_embeddings.append(embedding)
                    elif isinstance(item, dict):
                        # Handle dict format if needed
                        embedding = list(item.values())[0] if item else None
                        if embedding is not None:
                            token_embeddings.append(embedding)
                
            elif isinstance(doc_embeddings, dict):
                # Dictionary format {token: embedding}
                token_embeddings = list(doc_embeddings.values())
            else:
                # Unknown format, skip
                debug_print(f"Warning: Unknown embedding format: {type(doc_embeddings)}")
                continue
            
            if len(token_embeddings) == 0:
                # Empty document, skip for now (will handle later)
                continue
            
            # Convert to numpy arrays
            token_embeddings_np = []
            for emb in token_embeddings:
                if isinstance(emb, (list, tuple)):
                    emb_np = np.array(emb)
                elif hasattr(emb, 'numpy'):
                    emb_np = emb.numpy()
                elif hasattr(emb, 'tolist'):
                    emb_np = np.array(emb.tolist())
                else:
                    emb_np = np.array(emb)
                token_embeddings_np.append(emb_np)
            
            # Aggregate token embeddings to document embedding (mean pooling)
            if token_embeddings_np:
                token_embeddings_array = np.array(token_embeddings_np)
                # Mean pooling: average all token embeddings
                doc_embedding = np.mean(token_embeddings_array, axis=0)
                document_embeddings.append(doc_embedding)
        
        # Handle empty documents - create zero vectors with same dimension
        if len(document_embeddings) > 0:
            embedding_dim = document_embeddings[0].shape[0]
            # Fill in any missing documents with zero vectors
            # (This shouldn't happen if embeddings_list is properly structured)
            while len(document_embeddings) < len(embeddings_list):
                document_embeddings.append(np.zeros(embedding_dim))
        else:
            # No valid embeddings found
            raise ValueError("No valid embeddings found in embeddings_list. Ensure embeddings were extracted first.")
        
        return np.array(document_embeddings)
    
    def _fit_bertopic_model(self, documents, embeddings):
        """
        Fit BERTopic model on documents using pre-computed embeddings.
        
        Args:
            documents: List of document texts
            embeddings: Pre-computed document embeddings (numpy array)
        """
        print(f"Fitting BERTopic model on {len(documents)} documents...")
        
        # Handle edge case: single document
        if len(documents) == 1:
            debug_print("Warning: Only 1 document provided. BERTopic requires multiple documents for topic modeling.")
            debug_print("For single documents, consider using per-document analysis mode.")
            # Assign topic 0 to the single document
            self.model.topics_ = [0]
            self.model.probabilities_ = [[1.0]] if self.options.get('calculate_probabilities', False) else None
            return
        
        # Handle edge case: very few documents (less than min_topic_size)
        # Adjust min_topic_size dynamically for small datasets
        if len(documents) < self.options.get('min_topic_size', 10):
            original_min_topic_size = self.options.get('min_topic_size', 10)
            # Temporarily adjust to allow fitting with fewer documents
            # This will result in fewer topics or all documents in one topic
            adjusted_min = max(1, len(documents) // 2)
            debug_print(f"Adjusting min_topic_size from {original_min_topic_size} to {adjusted_min} "
                       f"for small dataset ({len(documents)} documents)")
            # Reinitialize model with adjusted min_topic_size
            from bertopic import BERTopic
            self.model = BERTopic(
                min_topic_size=adjusted_min,
                nr_topics=self.options.get('nr_topics', 'auto'),
                calculate_probabilities=self.options.get('calculate_probabilities', False),
                verbose=self.options.get('verbose', True)
            )
        
        # Fit model with custom embeddings
        # BERTopic can use custom embeddings by passing them to fit_transform
        try:
            topics, probs = self.model.fit_transform(documents, embeddings=embeddings)
            debug_print(f"BERTopic model fitted. Found {len(set(topics)) - (1 if -1 in topics else 0)} topics")
            if -1 in topics:
                outlier_count = list(topics).count(-1)
                debug_print(f"Outliers (topic -1): {outlier_count} documents")
        except ValueError as e:
            error_msg = str(e)
            # Check if it's the "0 samples" error from topic reduction
            if "0 sample" in error_msg or "minimum of 1 is required" in error_msg:
                debug_print(f"Error during topic reduction (likely all outliers): {e}")
                debug_print("Retrying without automatic topic reduction...")
                # Retry with nr_topics=None to disable reduction
                try:
                    from bertopic import BERTopic
                    self.model = BERTopic(
                        min_topic_size=self.options.get('min_topic_size', 10),
                        nr_topics=None,  # Disable automatic reduction
                        calculate_probabilities=self.options.get('calculate_probabilities', False),
                        verbose=self.options.get('verbose', True)
                    )
                    topics, probs = self.model.fit_transform(documents, embeddings=embeddings)
                    debug_print(f"BERTopic model fitted (without reduction). Found {len(set(topics)) - (1 if -1 in topics else 0)} topics")
                    if -1 in topics:
                        outlier_count = list(topics).count(-1)
                        debug_print(f"Outliers (topic -1): {outlier_count} documents")
                except Exception as e2:
                    debug_print(f"Error fitting BERTopic model even without reduction: {e2}")
                    # Final fallback: assign all documents to topic 0
                    self.model.topics_ = [0] * len(documents)
                    if self.options.get('calculate_probabilities', False):
                        self.model.probabilities_ = [[1.0]] * len(documents)
                    raise
            else:
                # Other ValueError, re-raise
                raise
        except Exception as e:
            debug_print(f"Error fitting BERTopic model: {e}")
            # Fallback: assign all documents to topic 0
            self.model.topics_ = [0] * len(documents)
            if self.options.get('calculate_probabilities', False):
                self.model.probabilities_ = [[1.0]] * len(documents)
            raise
    
    def _get_topic_assignments(self, model, documents, embeddings):
        """
        Get topic assignments for documents.
        
        Args:
            model: Fitted BERTopic model
            documents: List of documents
            embeddings: Document embeddings (for probability calculation)
            
        Returns:
            Tuple of (topic_assignments, topic_probabilities)
        """
        # Get topic assignments
        if hasattr(model, 'topics_') and model.topics_ is not None:
            topic_assignments = model.topics_
            # Convert to list if numpy array
            if isinstance(topic_assignments, np.ndarray):
                topic_assignments = topic_assignments.tolist()
            elif not isinstance(topic_assignments, list):
                topic_assignments = list(topic_assignments)
        else:
            # If topics_ not available, transform documents
            try:
                topic_assignments, _ = model.transform(documents, embeddings=embeddings)
                topic_assignments = topic_assignments.tolist() if hasattr(topic_assignments, 'tolist') else list(topic_assignments)
            except Exception as e:
                debug_print(f"Error transforming documents: {e}")
                # Fallback: assign all to topic 0
                topic_assignments = [0] * len(documents)
        
        # Ensure we have the right number of assignments
        if len(topic_assignments) != len(documents):
            debug_print(f"Warning: Topic assignments length ({len(topic_assignments)}) "
                      f"doesn't match documents length ({len(documents)})")
            # Pad or truncate to match
            if len(topic_assignments) < len(documents):
                topic_assignments.extend([-1] * (len(documents) - len(topic_assignments)))
            else:
                topic_assignments = topic_assignments[:len(documents)]
        
        # Get topic probabilities if enabled
        topic_probabilities = None
        if self.options.get('calculate_probabilities', False):
            if hasattr(model, 'probabilities_') and model.probabilities_ is not None:
                topic_probabilities = model.probabilities_
                # Convert to list if numpy array
                if isinstance(topic_probabilities, np.ndarray):
                    topic_probabilities = topic_probabilities.tolist()
            else:
                # Calculate probabilities if not stored
                try:
                    _, topic_probabilities = model.transform(documents, embeddings=embeddings)
                    if hasattr(topic_probabilities, 'tolist'):
                        topic_probabilities = topic_probabilities.tolist()
                except Exception as e:
                    debug_print(f"Error calculating probabilities: {e}")
                    topic_probabilities = None
        
        debug_print(f"Retrieved {len(topic_assignments)} topic assignments")
        return topic_assignments, topic_probabilities
    
    def _get_topic_info(self, model):
        """
        Extract topic information (keywords, sizes, representations).
        
        Args:
            model: Fitted BERTopic model
            
        Returns:
            Dictionary containing topic information
        """
        topic_info = {}
        
        try:
            # Check if model has topics_ attribute and it's not None
            if not hasattr(model, 'topics_') or model.topics_ is None:
                debug_print("Model has no topics_ attribute. Cannot extract topic info.")
                topic_info['keywords'] = {}
                topic_info['sizes'] = {}
                topic_info['num_topics'] = 0
                topic_info['dataframe'] = None
                return topic_info
            
            # Get topic information DataFrame
            try:
                info_df = model.get_topic_info()
                topic_info['dataframe'] = info_df
            except Exception as e:
                debug_print(f"Could not get topic info DataFrame: {e}")
                topic_info['dataframe'] = None
            
            # Get topic keywords for each topic
            topic_keywords = {}
            topic_sizes = {}
            
            # Get unique topics (exclude -1 which is outliers)
            if isinstance(model.topics_, list):
                unique_topics = [t for t in set(model.topics_) if t != -1]
            else:
                unique_topics = []
            
            for topic_id in unique_topics:
                try:
                    # Get keywords for this topic
                    keywords = model.get_topic(topic_id)
                    if keywords:
                        topic_keywords[topic_id] = keywords
                    
                    # Get topic size (number of documents in this topic)
                    if isinstance(model.topics_, list):
                        topic_size = model.topics_.count(topic_id)
                    else:
                        topic_size = 0
                    topic_sizes[topic_id] = topic_size
                except Exception as e:
                    debug_print(f"Error getting info for topic {topic_id}: {e}")
                    continue
            
            topic_info['keywords'] = topic_keywords
            topic_info['sizes'] = topic_sizes
            topic_info['num_topics'] = len(unique_topics)
            
            debug_print(f"Extracted info for {len(unique_topics)} topics")
            
        except Exception as e:
            debug_print(f"Error extracting topic info: {e}")
            import traceback
            debug_print(traceback.format_exc())
            topic_info['error'] = str(e)
            topic_info['keywords'] = {}
            topic_info['sizes'] = {}
            topic_info['num_topics'] = 0
            topic_info['dataframe'] = None
        
        return topic_info
    
    def prepare_topic_data_for_saving(self, topic_data):
        """
        Prepare topic modeling results in format suitable for CSV saving.
        
        This method prepares the data but doesn't save it.
        Saving should be done in corpus.extract_topic_modeling() using store_features_to_csv().
        
        Args:
            topic_data: Dictionary containing all topic modeling results
            
        Returns:
            Dictionary with prepared data ready for CSV saving
        """
        topic_assignments = topic_data.get('topic_assignments')
        topic_probabilities = topic_data.get('topic_probabilities')
        topic_info = topic_data.get('topic_info', {})
        documents = topic_data.get('documents', [])
        
        # Ensure topic_assignments is a list
        if topic_assignments is None:
            debug_print("Warning: topic_assignments is None")
            assignment_data = []
        else:
            # Convert to list if it's a numpy array or other type
            if isinstance(topic_assignments, np.ndarray):
                topic_assignments = topic_assignments.tolist()
            elif not isinstance(topic_assignments, list):
                topic_assignments = list(topic_assignments)
            
            # Ensure lengths match
            if len(documents) != len(topic_assignments):
                debug_print(f"Warning: Mismatch between documents ({len(documents)}) "
                          f"and topic_assignments ({len(topic_assignments)})")
                min_len = min(len(documents), len(topic_assignments))
                documents = documents[:min_len]
                topic_assignments = topic_assignments[:min_len]
            
            # Prepare assignment data per document
            assignment_data = []
            for idx, (doc_text, topic_id) in enumerate(zip(documents, topic_assignments)):
                doc_data = {
                    'document_index': idx,
                    'topic_id': int(topic_id) if topic_id is not None else -1
                }
                
                # Add probabilities if available
                if topic_probabilities is not None and idx < len(topic_probabilities):
                    probs = topic_probabilities[idx]
                    if isinstance(probs, (list, np.ndarray)):
                        # Convert to list if numpy array
                        if isinstance(probs, np.ndarray):
                            probs = probs.tolist()
                        # Add top 3 topic probabilities
                        top_probs = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:3]
                        for prob_idx, (topic, prob) in enumerate(top_probs):
                            doc_data[f'topic_{prob_idx+1}_id'] = int(topic)
                            doc_data[f'topic_{prob_idx+1}_probability'] = float(prob)
                
                assignment_data.append(doc_data)
            
            debug_print(f"Prepared {len(assignment_data)} topic assignments")
        
        # Prepare topic keywords data
        keywords_data = []
        if 'keywords' in topic_info:
            for topic_id, keywords in topic_info['keywords'].items():
                # keywords is a list of (word, score) tuples
                for rank, (word, score) in enumerate(keywords[:10]):  # Top 10 keywords
                    keywords_data.append({
                        'topic_id': topic_id,
                        'rank': rank + 1,
                        'keyword': word,
                        'score': float(score)
                    })
        
        # Prepare topic info summary
        topic_info_summary = []
        if 'dataframe' in topic_info and topic_info['dataframe'] is not None:
            # Convert DataFrame to list of dicts
            info_df = topic_info['dataframe']
            # DataFrame.to_dict('records') converts to list of dicts
            if hasattr(info_df, 'to_dict'):
                topic_info_summary = info_df.to_dict('records')
            else:
                # If it's already a list of dicts or similar
                topic_info_summary = info_df
        
        # Prepare topic comparisons data
        comparisons_data = []
        if 'topic_comparisons' in topic_data and topic_data['topic_comparisons']:
            for topic_id, comparison_info in topic_data['topic_comparisons'].items():
                for predefined_name, similarity in comparison_info['comparisons'].items():
                    comparisons_data.append({
                        'discovered_topic_id': topic_id,
                        'predefined_topic': predefined_name,
                        'similarity_score': similarity,
                        'is_top_match': predefined_name in [m[0] for m in comparison_info['top_matches']]
                    })
        
        return {
            'assignments': assignment_data,
            'keywords': keywords_data,
            'topic_info': topic_info_summary,
            'num_topics': topic_info.get('num_topics', 0),
            'topic_comparisons': comparisons_data
        }
    
    def _chunk_text_unit(self, text, embeddings, chunk_size):
        """
        Chunk a single text unit into smaller pieces for topic modeling.
        
        Args:
            text: Single text string
            embeddings: Embeddings for the text (list of (token, embedding) tuples)
            chunk_size: 'sentence' for sentence-based chunking, or integer for fixed-size chunks
        
        Returns:
            Tuple of (chunked_texts, chunked_embeddings)
        """
        chunked_texts = []
        chunked_embeddings = []
        
        if not embeddings or len(embeddings) == 0:
            debug_print("Warning: No embeddings provided for chunking")
            return chunked_texts, chunked_embeddings
        
        if chunk_size == 'sentence':
            # Split by sentences (period, exclamation, question mark followed by space or end)
            # More robust sentence splitting
            sentence_pattern = r'([.!?]+(?:\s+|$))'
            parts = re.split(sentence_pattern, text)
            sentences = []
            for i in range(0, len(parts) - 1, 2):
                if i + 1 < len(parts):
                    sentence = (parts[i] + parts[i + 1]).strip()
                    if sentence:
                        sentences.append(sentence)
            if len(parts) % 2 == 1 and parts[-1].strip():
                sentences.append(parts[-1].strip())
            
            # If no sentence boundaries found, use the whole text as one sentence
            if not sentences:
                sentences = [text]
            
            # For embeddings, split tokens proportionally by sentence length
            total_tokens = len(embeddings)
            total_chars = len(text)
            
            if total_chars == 0:
                return chunked_texts, chunked_embeddings
            
            char_offset = 0
            token_offset = 0
            
            for sentence in sentences:
                sentence_chars = len(sentence)
                # Calculate proportion of tokens for this sentence
                if total_chars > 0:
                    token_proportion = sentence_chars / total_chars
                    num_tokens = max(1, int(total_tokens * token_proportion))
                else:
                    num_tokens = total_tokens // len(sentences) if len(sentences) > 0 else total_tokens
                
                end_token_idx = min(token_offset + num_tokens, total_tokens)
                
                if end_token_idx > token_offset:
                    chunked_texts.append(sentence)
                    chunked_embeddings.append(embeddings[token_offset:end_token_idx])
                
                token_offset = end_token_idx
                char_offset += sentence_chars
                
                # If we've used all tokens, break
                if token_offset >= total_tokens:
                    break
        else:
            # Fixed-size chunking (by number of tokens)
            chunk_size_int = int(chunk_size) if isinstance(chunk_size, (str, int)) else 50
            total_tokens = len(embeddings)
            
            for i in range(0, total_tokens, chunk_size_int):
                end_idx = min(i + chunk_size_int, total_tokens)
                # Extract embeddings for this chunk
                chunk_emb = embeddings[i:end_idx]
                
                # Approximate text chunk (this is rough, but embeddings are what matter)
                # Calculate approximate character positions
                tokens_per_char = total_tokens / len(text) if len(text) > 0 else 1
                start_char = int(i / tokens_per_char) if tokens_per_char > 0 else 0
                end_char = int(end_idx / tokens_per_char) if tokens_per_char > 0 else len(text)
                text_chunk = text[start_char:end_char].strip()
                
                if text_chunk and len(chunk_emb) > 0:
                    chunked_texts.append(text_chunk)
                    chunked_embeddings.append(chunk_emb)
        
        # Filter out empty chunks
        filtered_texts = []
        filtered_embeddings = []
        for text_chunk, emb_chunk in zip(chunked_texts, chunked_embeddings):
            if text_chunk.strip() and len(emb_chunk) > 0:
                filtered_texts.append(text_chunk)
                filtered_embeddings.append(emb_chunk)
        
        debug_print(f"Chunked text unit into {len(filtered_texts)} chunks")
        return filtered_texts, filtered_embeddings
    
    def _compare_to_predefined_topics(self, topic_info, predefined_topics, document_embeddings, topic_assignments):
        """
        Compare identified topics to predefined topics (e.g., "food", "age").
        
        Uses embedding similarity to match discovered topics to predefined topics.
        
        Args:
            topic_info: Dictionary containing topic information (keywords, etc.)
            predefined_topics: Dictionary mapping predefined topic names to descriptions/keywords
                            Format: {'food': 'food eating meal', 'age': 'age old young', ...}
            document_embeddings: Document embeddings array
            topic_assignments: List of topic assignments
        
        Returns:
            Dictionary mapping discovered topic IDs to predefined topics with similarity scores
        """
        try:
            from pelican_nlp.extraction.language_model import Model
            from pelican_nlp.extraction.extract_embeddings import EmbeddingsExtractor
            
            debug_print("Comparing discovered topics to predefined topics...")
            
            # Get embedding model to compute embeddings for predefined topics
            embedding_options = self.options.get('embedding_options', {})
            if not embedding_options:
                # Try to get from config if available
                debug_print("Warning: No embedding options found for predefined topic comparison.")
                return None
            
            # Compute embeddings for predefined topic descriptions
            predefined_embeddings = {}
            for topic_name, topic_description in predefined_topics.items():
                # Use the same embedding model as the main pipeline
                # For simplicity, we'll use mean pooling of word embeddings
                # In a full implementation, you might want to use sentence embeddings
                try:
                    # Create a simple extractor to get embeddings
                    temp_extractor = EmbeddingsExtractor(embedding_options, self.project_folder)
                    # Get embeddings for the description
                    desc_embeddings, _ = temp_extractor.extract_embeddings_from_text(
                        [topic_description], embedding_options
                    )
                    if desc_embeddings and len(desc_embeddings) > 0:
                        # Get document-level embedding (mean pooling)
                        doc_emb = self._prepare_embeddings_for_bertopic(desc_embeddings)
                        if len(doc_emb) > 0:
                            predefined_embeddings[topic_name] = doc_emb[0]
                except Exception as e:
                    debug_print(f"Error computing embedding for predefined topic '{topic_name}': {e}")
                    continue
            
            # Get topic centroids (mean embedding of documents in each topic)
            topic_centroids = {}
            unique_topics = set(topic_assignments) if isinstance(topic_assignments, list) else set()
            
            for topic_id in unique_topics:
                if topic_id == -1:  # Skip outliers
                    continue
                # Get embeddings of documents assigned to this topic
                topic_doc_indices = [i for i, t in enumerate(topic_assignments) if t == topic_id]
                if topic_doc_indices:
                    topic_doc_embeddings = document_embeddings[topic_doc_indices]
                    # Compute centroid (mean)
                    topic_centroids[topic_id] = np.mean(topic_doc_embeddings, axis=0)
            
            # Compute cosine similarity between topic centroids and predefined topics
            comparisons = {}
            for topic_id, centroid in topic_centroids.items():
                topic_comparisons = {}
                for predefined_name, predefined_emb in predefined_embeddings.items():
                    # Cosine similarity
                    similarity = np.dot(centroid, predefined_emb) / (
                        np.linalg.norm(centroid) * np.linalg.norm(predefined_emb) + 1e-10
                    )
                    topic_comparisons[predefined_name] = float(similarity)
                
                # Sort by similarity
                sorted_comparisons = sorted(
                    topic_comparisons.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                comparisons[topic_id] = {
                    'comparisons': topic_comparisons,
                    'top_matches': sorted_comparisons[:3]  # Top 3 matches
                }
            
            debug_print(f"Compared {len(comparisons)} discovered topics to {len(predefined_topics)} predefined topics")
            return comparisons
            
        except Exception as e:
            debug_print(f"Error comparing to predefined topics: {e}")
            import traceback
            debug_print(traceback.format_exc())
            return None
