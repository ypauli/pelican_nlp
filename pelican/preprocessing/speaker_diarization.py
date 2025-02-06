import re
from collections import Counter
import numpy as np

class TextDiarizer:
    def __init__(self, method=None):
        self.method = method

    def speaker_tag_identification(self, text, num_speakers, exclude_tags=None):
        if exclude_tags is None:
            exclude_tags = []

        # Regular expression to capture potential speaker tags
        potential_tags = re.findall(r'^(\w+):', text, re.MULTILINE)

        # Count occurrences of potential speaker tags
        tag_counts = Counter(potential_tags)

        # Filter likely speaker tags based on frequency and exclusion list
        speaker_tags = [tag for tag, count in tag_counts.most_common(num_speakers) if tag not in exclude_tags]

        return speaker_tags

    def remove_speaker_tags(self, text, speaker_tags):
        pattern = re.compile(r'^(?:' + '|'.join(re.escape(tag) for tag in speaker_tags) + r'):\s*', re.MULTILINE)
        return re.sub(pattern, '', text)
