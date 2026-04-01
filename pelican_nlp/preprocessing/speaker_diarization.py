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

        # Keep only non-empty configured speaker tags.
        # These tags are used for filtering which speaker blocks to analyze,
        # not for detecting where speaker blocks begin/end.
        speaker_tags = [
            tag.strip().lower() for tag in speaker_tags
            if isinstance(tag, str) and tag.strip()
        ]
        if not speaker_tags:
            return [text] if text else []

        # Detect all speaker blocks from any line-starting "tag:" pattern.
        # Boundaries are determined by the next line-starting "tag:" or end of text.
        all_tag_block_pattern = re.compile(
            r"^\s*(?P<tag>\w+)\s*:\s*(?P<content>.*?)(?=^\s*\w+\s*:|\Z)",
            re.MULTILINE | re.DOTALL
        )

        found_any_tagged_blocks = False
        filtered_matches = []

        for match in all_tag_block_pattern.finditer(text):
            found_any_tagged_blocks = True
            found_tag = match.group("tag").strip()
            if found_tag.lower() not in speaker_tags:
                continue

            content = match.group("content").strip()
            if keep_speakertag:
                filtered_matches.append(f"{found_tag}: {content}")
            else:
                filtered_matches.append(content)

        if filtered_matches:
            return filtered_matches

        # If text contains speaker-like tags but none match configured tags,
        # return an empty list so no unintended speaker content is analyzed.
        if found_any_tagged_blocks:
            return []

        # Fallback for plain text without any line-starting "tag:" structure.
        return [text] if text else []
