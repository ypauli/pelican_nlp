"""
This module provides the SectionIdentificator class for identifying and extracting sections from text documents.
The SectionIdentificator class handles different types of section identification based on configuration settings.
"""

from collections import defaultdict, OrderedDict
import re


class SectionIdentificator:
    """Class for identifying and extracting sections from text documents."""
    
    def __init__(self, config):
        """Initialize SectionIdentificator with configuration.
        
        Args:
            config: Dictionary containing section identification configuration
        """
        self.has_sections = config.get('has_multiple_sections', False)
        self.has_section_titles = config.get('has_section_titles', False)
        self.section_identifier = config.get('section_identification')
        self.number_of_sections = config.get('number_of_sections')
        self.task = config.get('task')
        self.has_segments = self.task == "discourse"
    
    def identify_sections(self, raw_text, lines=None):
        """Identify sections in the given text.
        
        Args:
            raw_text: Raw text content as string
            lines: Optional list of Line objects (for discourse tasks)
            
        Returns:
            dict: Dictionary mapping section titles to their content
        """
        if not raw_text:
            raise ValueError("Raw text must be provided for section identification.")
        
        # Handle discourse tasks with segments
        if self.has_segments and lines:
            return self._identify_segments(lines)
        
        # Handle regular section identification
        return self._identify_regular_sections(raw_text)
    
    def _identify_regular_sections(self, raw_text):
        """Identify sections in regular text documents.
        
        Args:
            raw_text: Raw text content as string
            
        Returns:
            dict: Dictionary mapping section titles to their content
        """
        lines = raw_text.splitlines()
        
        # Handle documents without sections
        if not self.has_sections:
            if self.has_section_titles and lines:
                title, content = (lines[0].strip(), "\n".join(lines[1:]).strip()) if lines else ("untitled section", "")
            else:
                title, content = "untitled section", "\n".join(lines).strip()
            return {title: content}
        
        # Handle documents with sections
        sections = {}
        current_title, current_content = None, []
        section_titles = []
        section_pattern = re.compile(
            rf"^\s*(?:(?:\d+\.|(?:[IVX]+\.)|(?:[a-z]\)\.?)|(?:[A-Z]\)\.?)|(?:\d+\)\.?))\s*)?{re.escape(self.section_identifier)}")
        for line in lines:
            if section_pattern.match(line):
                if current_title:
                    sections[current_title] = "\n".join(current_content).strip()
                
                current_title = line.strip()
                section_titles.append(current_title)
                current_content = []
            else:
                if current_title:
                    current_content.append(line)
        
        if current_title:
            sections[current_title] = "\n".join(current_content).strip()
        
        # Validate number of sections if specified
        if self.number_of_sections is not None and len(sections) != self.number_of_sections:
            raise ValueError(f"Incorrect number of sections detected. Expected {self.number_of_sections}, got {len(sections)}.")
        
        return sections
    
    def _identify_segments(self, lines, protocol=None, cutoff=1):
        """Identify segments in discourse tasks.
        
        Args:
            lines: List of Line objects
            protocol: Dictionary mapping section names to lists of terms
            cutoff: Minimum number of matches required for a section
            
        Returns:
            dict: Dictionary mapping segment names to lists of lines
        """
        if not self.has_segments:
            return {"default": lines}
        
        if not protocol:
            # Default protocol if none provided
            protocol = {"1": ["default"]}
        
        # Create regex patterns for each section
        patterns = {
            section: re.compile("|".join(f"(?:\\b{re.escape(term)}\\b)" for term in terms), re.IGNORECASE)
            for section, terms in protocol.items()
        }
        
        # Find matches for each section
        match_scores = defaultdict(list)
        for section, pattern in patterns.items():
            for line_index, line in enumerate(lines):
                if pattern.search(line.text):
                    match_scores[section].append(line_index)
        
        # Determine section starts
        section_order = sorted(protocol.keys(), key=lambda x: int(x))
        section_starts = OrderedDict()
        last_index_used = -1
        
        for section in section_order:
            line_indices = match_scores[section]
            valid_starts = [idx for idx in line_indices if idx > last_index_used and len(line_indices) >= cutoff]
            if valid_starts:
                start_line = min(valid_starts)
                section_starts[section] = start_line
                last_index_used = start_line
        
        # Assign segments to lines
        segment_names = ["1"] * len(lines)
        current_section = None
        for i in range(len(lines)):
            if i in section_starts.values():
                current_section = [sec for sec, start in section_starts.items() if start == i][0]
            segment_names[i] = current_section if current_section else "default"
        
        # Create sections dictionary
        sections = defaultdict(list)
        for line, segment in zip(lines, segment_names):
            sections[segment].append(line)
        
        return sections
    
    def validate_sections(self, sections):
        """Validate that the identified sections meet the configuration requirements.
        
        Args:
            sections: Dictionary of identified sections
            
        Returns:
            bool: True if sections are valid, False otherwise
        """
        if not sections:
            return False
        
        # Check number of sections if specified
        if self.number_of_sections is not None and len(sections) != self.number_of_sections:
            return False
        
        # Check that sections have content
        for title, content in sections.items():
            if isinstance(content, list):
                if not content:  # Empty list
                    return False
            elif isinstance(content, str):
                if not content.strip():  # Empty or whitespace-only string
                    return False
        
        return True
    
    def get_section_info(self, sections):
        """Get information about the identified sections.
        
        Args:
            sections: Dictionary of identified sections
            
        Returns:
            dict: Information about the sections
        """
        if not sections:
            return {"error": "No sections found"}
        
        info = {
            "number_of_sections": len(sections),
            "section_titles": list(sections.keys()),
            "has_content": True
        }
        
        # Check content for each section
        for title, content in sections.items():
            if isinstance(content, list):
                info[f"{title}_line_count"] = len(content)
                info[f"{title}_word_count"] = sum(len(line.text.split()) for line in content) if content else 0
            elif isinstance(content, str):
                info[f"{title}_word_count"] = len(content.split())
        
        return info 