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

        # Keep only non-empty tags and match case-insensitively.
        # A new speaker block starts only when the token before ":" is one of these tags.
        speaker_tags = [
            tag.strip() for tag in speaker_tags
            if isinstance(tag, str) and tag.strip()
        ]
        if not speaker_tags:
            return [text] if text else []

        tag_pattern = "|".join(re.escape(tag) for tag in speaker_tags)
        speaker_block_pattern = re.compile(
            rf"^\s*(?P<tag>{tag_pattern})\s*:\s*(?P<content>.*?)(?=^\s*(?:{tag_pattern})\s*:|\Z)",
            re.IGNORECASE | re.MULTILINE | re.DOTALL
        )

        all_matches = []
        for match in speaker_block_pattern.finditer(text):
            content = match.group("content").strip()
            if keep_speakertag:
                # Keep the tag as it appears in the original text.
                all_matches.append(f"{match.group('tag')}: {content}")
            else:
                all_matches.append(content)

        return all_matches if all_matches else [text]
