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
    def parse_speaker(text, speaker_tag, keep_speakertag=False):

        pattern = rf"{re.escape(speaker_tag)}:\s*(.*?)(?=\n\s*\w+:|\Z)"
        matches = re.findall(pattern, text, re.DOTALL)

        if keep_speakertag:
            return [f"{speaker_tag}: {match.strip()}" for match in matches]
        else:
            return [match.strip() for match in matches]
