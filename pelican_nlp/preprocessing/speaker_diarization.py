import re
from collections import Counter

class TextDiarizer:
    def __init__(self, method=None):
        self.method = method


    def speaker_tag_identification(self, text, num_speakers, exclude_speakers=None):
        if exclude_speakers is None:
            exclude_tags = []

        # Regular expression to capture potential speaker tags
        potential_tags = re.findall(r'^(\w+):', text, re.MULTILINE)

        # Count occurrences of potential speaker tags
        tag_counts = Counter(potential_tags)

        # Filter likely speaker tags based on frequency and exclusion list
        speaker_tags = [tag for tag, count in tag_counts.most_common(num_speakers) if tag not in exclude_speakers]

        return speaker_tags

    @staticmethod
    def parse_speaker(text, speaker_tags, keep_speakertag=False):
        """
        Parse speaker content from text using one or multiple speaker tags.
        
        Args:
            text: The text to parse
            speaker_tags: Single speaker tag (str), list of speaker tags (list), or None
            keep_speakertag: Whether to keep the speaker tag in the output
            
        Returns:
            List of speaker utterances
        """
        # Handle None case
        if speaker_tags is None:
            # If no speaker tags provided, return text as single utterance
            return [text] if text else []
        
        # Handle both single tag and multiple tags
        if isinstance(speaker_tags, str):
            speaker_tags = [speaker_tags]
        
        all_matches = []
        
        for speaker_tag in speaker_tags:
            pattern = rf"{re.escape(speaker_tag)}:\s*(.*?)(?=\n\s*\w+:|\Z)"
            matches = re.findall(pattern, text, re.DOTALL)
            
            if keep_speakertag:
                all_matches.extend([f"{speaker_tag}: {match.strip()}" for match in matches])
            else:
                all_matches.extend([match.strip() for match in matches])
        
        return all_matches if all_matches else [text]
