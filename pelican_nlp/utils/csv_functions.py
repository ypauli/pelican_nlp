import os
import csv
from pelican_nlp.config import debug_print
from .lpds_paths import derivatives_output_path, document_entities, unit_folder_for_document

def store_features_to_csv(input_data, derivatives_dir, doc_class, metric):
    """Store various types of features to CSV files with consistent formatting."""

    unit_folder = unit_folder_for_document(doc_class)
    entities = document_entities(doc_class)
    output_filepath = derivatives_output_path(
        derivatives_dir, unit_folder, doc_class.name, metric, entities
    )
    debug_print(output_filepath)
    file_exists = os.path.exists(output_filepath)
    
    # Write data based on metric type
    with open(output_filepath, mode='a', newline='', encoding='utf-8') as file:
        detail_metrics = metric.startswith('semantic-similarity-window-details-') or metric == 'semantic-similarity-sentence-details'
        writer = csv.writer(file, quoting=csv.QUOTE_ALL if detail_metrics else csv.QUOTE_MINIMAL)
        
        if metric == 'embeddings':
            if input_data is None:
                input_data = []
            if not isinstance(input_data, list):
                raise ValueError("Input data for embeddings must be a list (can be empty).")

            if input_data:
                # Get the dimensionality from the first embedding
                embedding_dim = len(input_data[0][1])
                header = ['Token'] + [f"Dim_{i}" for i in range(embedding_dim)]
            elif file_exists:
                header = _get_existing_embeddings_header(output_filepath)
            else:
                # Fallback header when first section is empty
                header = ['Token']
            _write_csv_header(writer, header, file_exists)
            
            for token, embedding in input_data:
                # Handle both list and tensor/array types
                if hasattr(embedding, 'tolist'):
                    embedding_list = embedding.tolist()
                elif isinstance(embedding, list):
                    embedding_list = embedding
                else:
                    raise ValueError(f"Embedding must be either a list or have tolist() method, got {type(embedding)}")
                writer.writerow([token] + embedding_list)

        elif metric == 'cosine-similarity-matrix':
            _write_csv_header(writer, ['Matrix'], file_exists)
            for row in input_data:
                writer.writerow(row)
                
        elif metric.startswith('semantic-similarity-window-details-'):
            if not input_data:
                return output_filepath
            header = ['window_index', 'start_token', 'end_token', 'token_span', 'mean', 'std', 'median']
            _write_csv_header(writer, header, file_exists)
            for entry in input_data:
                writer.writerow([
                    entry.get('window_index'),
                    entry.get('start_token'),
                    entry.get('end_token'),
                    entry.get('token_span'),
                    entry.get('mean'),
                    entry.get('std'),
                    entry.get('median')
                ])
        elif metric == 'semantic-similarity-sentence-details':
            if not input_data:
                return output_filepath
            header = ['sentence_i_index', 'sentence_j_index', 'sentence_i_text', 'sentence_j_text', 'similarity']
            _write_csv_header(writer, header, file_exists)
            for entry in input_data:
                writer.writerow([
                    entry.get('sentence_i_index'),
                    entry.get('sentence_j_index'),
                    entry.get('sentence_i_text'),
                    entry.get('sentence_j_text'),
                    entry.get('similarity')
                ])
        elif metric.startswith('semantic-similarity-window-') or metric == 'semantic-similarity-sentence':
            header = ['Metric', 'Similarity_Score']
            _write_csv_header(writer, header, file_exists)
            
            for metric_name, score in input_data.items():
                writer.writerow([metric_name, score])

        elif metric == 'distance-from-randomness':
            # Check if this is the new TSP divergence format or old sliding window format
            if input_data and 'section' in input_data and len(input_data['section']) > 0:
                first_result = input_data['section'][0]
                
                # Check if it has TSP divergence fields
                if 'global_div_z' in first_result or 'local_div_z' in first_result:
                    # New TSP divergence format
                    header = ['total_distance', 'avg_distance', 'global_divergence', 
                             'local_divergence', 'global_div_z', 'local_div_z', 'optimal_path']
                    _write_csv_header(writer, header, file_exists)
                    
                    for result in input_data['section']:
                        # Convert optimal_path list to string for CSV storage
                        opt_path_str = str(result.get('optimal_path', []))
                        writer.writerow([
                            result.get('total_distance', ''),
                            result.get('avg_distance', ''),
                            result.get('global_divergence', ''),
                            result.get('local_divergence', ''),
                            result.get('global_div_z', ''),
                            result.get('local_div_z', ''),
                            opt_path_str
                        ])
                else:
                    # Old sliding window format (backward compatible)
                    header = ['window_index', 'all_pairs_average', 'actual_dist', 'average_dist', 'std_dist']
                    _write_csv_header(writer, header, file_exists)

                    for window_result in input_data['section']:
                        writer.writerow([
                            window_result.get('window_index', ''),
                            window_result.get('all_pairs_average', ''),
                            window_result.get('actual_dist', ''),
                            window_result.get('average_dist', ''),
                            window_result.get('std_dist', '')
                        ])

        elif metric in ('logits', 'perplexity-section', 'perplexity-sentence'):
            if not input_data:
                return
            header = list(input_data[0].keys())
            _write_csv_header(writer, header, file_exists)
            for entry in input_data:
                writer.writerow(entry.values())

        elif metric == 'opensmile-features':
            if not input_data:
                return
                
            # Get all column names from the first entry
            csv_columns = list(input_data[0].keys()) if isinstance(input_data, list) else list(input_data.keys())
            
            # Only write header if file doesn't exist
            if not file_exists:
                writer.writerow(csv_columns)
            
            # Handle both list of dictionaries and single dictionary cases
            if isinstance(input_data, list):
                for entry in input_data:
                    # Create a new array for the row data
                    row_data = []
                    for column in csv_columns:
                        # Convert numerical values to float
                        value = entry[column]
                        if isinstance(value, (int, float)):
                            value = float(value)
                        row_data.append(value)
                    writer.writerow(row_data)
            else:
                # Handle single dictionary case
                row_data = []
                for column in csv_columns:
                    value = input_data[column]
                    if isinstance(value, (int, float)):
                        value = float(value)
                    row_data.append(value)
                writer.writerow(row_data)

        elif metric == 'prosogram-features':
            if input_data is None or (hasattr(input_data, 'empty') and input_data.empty):
                return
                
            # Handle pandas DataFrame (main case)
            if hasattr(input_data, 'to_csv'):
                # Only write header if file doesn't exist
                if not file_exists:
                    # Write column headers
                    writer.writerow(input_data.columns.tolist())
                
                # Write data rows
                for _, row in input_data.iterrows():
                    writer.writerow(row.tolist())
            
            # Handle dictionary of multiple files (fallback case)
            elif isinstance(input_data, dict):
                # Write each file type as a separate section
                for file_type, data in input_data.items():
                    if not file_exists:
                        writer.writerow([f"File_Type: {file_type}"])
                    
                    if hasattr(data, 'to_csv'):  # DataFrame
                        if not file_exists:
                            writer.writerow(data.columns.tolist())
                        for _, row in data.iterrows():
                            writer.writerow(row.tolist())
                    else:  # Text data
                        if not file_exists:
                            writer.writerow([f"Content: {file_type}"])
                        # Write text content line by line
                        for line in str(data).split('\n'):
                            if line.strip():
                                writer.writerow([line.strip()])
                    
                    if not file_exists:
                        writer.writerow([])  # Empty line between sections

        elif isinstance(input_data, list) and len(input_data) > 0 and isinstance(input_data[0], dict):
            # Generic handler for list of dictionaries (e.g., topic modeling assignments, keywords, comparisons)
            if not input_data:
                return output_filepath
            
            # Get header from first dictionary
            header = list(input_data[0].keys())
            _write_csv_header(writer, header, file_exists)
            
            # Write each dictionary as a row
            for entry in input_data:
                row_data = []
                for column in header:
                    value = entry.get(column, '')
                    # Convert None to empty string, handle other types
                    if value is None:
                        value = ''
                    elif isinstance(value, (int, float)):
                        value = float(value)
                    elif not isinstance(value, str):
                        value = str(value)
                    row_data.append(value)
                writer.writerow(row_data)
        
        # If no handler matches and data is empty, just return
        elif not input_data:
            return output_filepath

    return output_filepath


def _write_csv_header(writer, header, file_exists):
    """Write CSV header with section separation if file exists."""
    if not file_exists:
        writer.writerow(header)
    else:
        writer.writerow([])  # Separate sections
        writer.writerow(['New Section'])
        writer.writerow(header)


def _get_existing_embeddings_header(output_filepath):
    """Read the latest embeddings header from an existing CSV file."""
    try:
        with open(output_filepath, mode='r', newline='', encoding='utf-8') as existing_file:
            rows = list(csv.reader(existing_file))
    except Exception:
        return ['Token']

    # Search from end to start to pick the most recent section header.
    for row in reversed(rows):
        if not row:
            continue
        if len(row) == 1 and row[0] == 'New Section':
            continue
        if row[0] == 'Token':
            return row

    return ['Token']