"""
Common utilities for the transcription system.

This module provides utility functions used throughout the transcription system,
including:
- Device management for PyTorch computations
- Text normalization and cleaning
- Time interval manipulation and merging
- Time formatting and conversion
- Unicode handling and normalization

The utilities are designed to be robust and handle edge cases gracefully,
with clear error messages and fallback behaviors where appropriate.
"""
import re
import torch
import unicodedata
from typing import List, Dict, Any


def get_device(skip_mps: bool = False) -> torch.device:
    """
    Get the best available device for computation.
    
    Determines the optimal compute device based on availability and requirements.
    The priority order is:
    1. CUDA (GPU) if available
    2. MPS (Apple Silicon) if available and not skipped
    3. CPU as fallback
    
    Args:
        skip_mps (bool): If True, skip MPS even if available. This is useful
            for operations that don't support MPS acceleration.
        
    Returns:
        torch.device: Best available device (cuda > mps > cpu)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif not skip_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalize_text(text: str) -> str:
    """
    Normalize text by removing special characters and extra whitespace.
    
    This function performs several text normalization steps:
    1. Unicode normalization (NFKC form)
    2. Removal of special characters while preserving sentence endings
    3. Whitespace normalization
    4. Handling of ellipsis and multiple punctuation
    5. Preservation of important punctuation marks
    
    Args:
        text (str): Input text to normalize
        
    Returns:
        str: Normalized text with consistent formatting
        
    Example:
        >>> normalize_text("Hello...   world!!!")
        "Hello... world!"
    """
    # Convert to NFKD form and remove diacritics
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    
    # Temporarily replace sentence-ending punctuation with special tokens
    text = text.replace('.', ' PERIOD ')
    text = text.replace('?', ' QMARK ')
    text = text.replace('!', ' EMARK ')
    
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Restore sentence-ending punctuation
    text = text.replace(' PERIOD ', '.')
    text = text.replace(' QMARK ', '?')
    text = text.replace(' EMARK ', '!')
    
    return text.lower()


def merge_intervals(intervals: List[Dict[str, Any]], tolerance: float = 0.1) -> List[Dict[str, Any]]:
    """
    Merge overlapping time intervals in a list of dictionaries.
    
    This function merges time intervals that overlap or are within a specified
    tolerance of each other. It's useful for combining nearby speech segments
    or cleaning up diarization results.
    
    Args:
        intervals (List[Dict[str, Any]]): List of interval dictionaries, each containing
            at least 'start' and 'end' keys with float values
        tolerance (float): Maximum gap between intervals to consider them adjacent
            
    Returns:
        List[Dict[str, Any]]: Merged intervals, maintaining all original dictionary
            keys with values from the first interval in each merged group
            
    Example:
        >>> intervals = [
        ...     {"start": 0.0, "end": 1.0, "text": "Hello"},
        ...     {"start": 0.9, "end": 2.0, "text": "world"}
        ... ]
        >>> merge_intervals(intervals)
        [{"start": 0.0, "end": 2.0, "text": "Hello"}]
    """
    if not intervals:
        return []
        
    # Sort intervals by start time
    sorted_intervals = sorted(intervals, key=lambda x: float(x['start_time']))
    merged = [sorted_intervals[0]]
    
    for interval in sorted_intervals[1:]:
        current = merged[-1]
        if float(interval['start_time']) - float(current['end_time']) <= tolerance:
            # Merge intervals
            current['end_time'] = max(float(current['end_time']), float(interval['end_time']))
        else:
            merged.append(interval)
            
    return merged


def format_time(seconds: float) -> str:
    """
    Format time in seconds to a human-readable string.
    
    Converts a duration in seconds to a formatted string in the format:
    "HH:MM:SS.mmm" for durations >= 1 hour
    "MM:SS.mmm" for durations < 1 hour
    
    Args:
        seconds (float): Time duration in seconds
        
    Returns:
        str: Formatted time string
        
    Example:
        >>> format_time(3661.5)
        "01:01:01.500"
        >>> format_time(61.5)
        "01:01.500"
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}" 