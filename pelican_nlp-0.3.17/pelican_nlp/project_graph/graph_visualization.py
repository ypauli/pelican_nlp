#!/usr/bin/env python3
"""
Graph Visualization for Pelican-nlp Project
===========================================

This script creates a visual representation of the Pelican-nlp project structure
using graphviz.
"""

from graphviz import Digraph
from pelican_nlp.config import debug_print

def create_pelican_graph():
    # Create a new directed graph
    dot = Digraph(comment='Pelican-nlp Project Structure')
    dot.attr(rankdir='TB')
    
    # Set node styles
    dot.attr('node', shape='box', style='rounded,filled')
    
    # Main Components
    with dot.subgraph(name='cluster_main') as c:
        c.attr(label='Main Components')
        c.attr('node', fillcolor='lightblue')
        c.node('Pelican', 'Pelican\n(Main Controller)')
        c.node('LPDS', 'LPDS\n(Data Structure)')
        c.node('Corpus', 'Corpus\n(Document Collection)')
        c.node('Participant', 'Participant\n(Grouping Unit)')
        c.node('Document', 'Document\n(Data Container)')
        c.node('AudioDocument', 'AudioDocument\n(Audio Data)')
    
    # Core Processing
    with dot.subgraph(name='cluster_core') as c:
        c.attr(label='Core Processing')
        c.attr('node', fillcolor='lightgreen')
        c.node('Config', 'Configuration\n(config.py)')
        c.node('CLI', 'Command Line Interface\n(cli.py)')
        c.node('Main', 'Main Entry Point\n(main.py)')
    
    # Preprocessing Components
    with dot.subgraph(name='cluster_preprocessing') as c:
        c.attr(label='Preprocessing')
        c.attr('node', fillcolor='lightyellow')
        c.node('TextTokenizer', 'Text Tokenizer\n(text_tokenizer.py)')
        c.node('TextNormalizer', 'Text Normalizer\n(text_normalizer.py)')
        c.node('TextCleaner', 'Text Cleaner\n(text_cleaner.py)')
        c.node('TextImporter', 'Text Importer\n(text_importer.py)')
        c.node('SpeakerDiarization', 'Speaker Diarization\n(speaker_diarization.py)')
        c.node('Pipeline', 'Preprocessing Pipeline\n(pipeline.py)')
    
    # Extraction Components
    with dot.subgraph(name='cluster_extraction') as c:
        c.attr(label='Feature Extraction')
        c.attr('node', fillcolor='lightpink')
        c.node('LogitsExtractor', 'Logits Extractor\n(extract_logits.py)')
        c.node('EmbeddingsExtractor', 'Embeddings Extractor\n(extract_embeddings.py)')
        c.node('LanguageModel', 'Language Model\n(language_model.py)')
        c.node('AcousticFeatures', 'Acoustic Features\n(acoustic_feature_extraction.py)')
        c.node('SemanticSimilarity', 'Semantic Similarity\n(semantic_similarity.py)')
        c.node('RandomnessDistance', 'Distance from Randomness\n(distance_from_randomness.py)')
    
    # Utility Components
    with dot.subgraph(name='cluster_utils') as c:
        c.attr(label='Utilities')
        c.attr('node', fillcolor='lightgrey')
        c.node('FilenameParser', 'Filename Parser\n(filename_parser.py)')
        c.node('CSVFunctions', 'CSV Functions\n(csv_functions.py)')
        c.node('SetupFunctions', 'Setup Functions\n(setup_functions.py)')
    
    # Main Relationships
    dot.edge('Pelican', 'LPDS', 'manages')
    dot.edge('Pelican', 'Corpus', 'processes')
    dot.edge('Pelican', 'Participant', 'instantiates')
    dot.edge('Corpus', 'Document', 'contains')
    dot.edge('Participant', 'Document', 'groups')
    dot.edge('Document', 'AudioDocument', 'extends')
    
    # Core Processing Relationships
    dot.edge('CLI', 'Main', 'calls')
    dot.edge('Main', 'Pelican', 'instantiates')
    dot.edge('Pelican', 'Config', 'uses')
    
    # Preprocessing Relationships
    dot.edge('Pipeline', 'TextTokenizer', 'uses')
    dot.edge('Pipeline', 'TextNormalizer', 'uses')
    dot.edge('Pipeline', 'TextCleaner', 'uses')
    dot.edge('Pipeline', 'TextImporter', 'uses')
    dot.edge('Pipeline', 'SpeakerDiarization', 'uses')
    dot.edge('Corpus', 'Pipeline', 'executes')
    
    # Extraction Relationships
    dot.edge('Corpus', 'LogitsExtractor', 'uses')
    dot.edge('Corpus', 'EmbeddingsExtractor', 'uses')
    dot.edge('LogitsExtractor', 'LanguageModel', 'uses')
    dot.edge('EmbeddingsExtractor', 'LanguageModel', 'uses')
    dot.edge('Corpus', 'AcousticFeatures', 'uses')
    dot.edge('Corpus', 'SemanticSimilarity', 'uses')
    dot.edge('Corpus', 'RandomnessDistance', 'uses')
    
    # Utility Relationships
    dot.edge('Pelican', 'FilenameParser', 'uses')
    dot.edge('Corpus', 'CSVFunctions', 'uses')
    dot.edge('Pelican', 'SetupFunctions', 'uses')
    
    # Save the graph
    dot.render('pelican_structure_detailed', format='png', cleanup=True)
    debug_print("Detailed graph visualization has been created as 'pelican_structure_detailed.png'")

if __name__ == '__main__':
    create_pelican_graph() 