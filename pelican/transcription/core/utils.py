"""
Common utilities for the transcription system.
"""
import re
import torch
import unicodedata
from typing import List, Dict, Any


def get_device() -> torch.device:
    """
    Get the best available device (CUDA > MPS > CPU).
    
    Returns:
        torch.device: Best available device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalize_text(text: str) -> str:
    """
    Normalize text by removing special characters and extra whitespace.
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text
    """
    # Convert to NFKD form and remove diacritics
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text.lower()


def merge_intervals(intervals: List[Dict[str, Any]], tolerance: float = 0.1) -> List[Dict[str, Any]]:
    """
    Merge overlapping time intervals.
    
    Args:
        intervals: List of dictionaries with 'start_time' and 'end_time' keys
        tolerance: Time tolerance for merging intervals (seconds)
        
    Returns:
        List of merged intervals
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
    Format time in seconds to HH:MM:SS.mmm.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}" 