from pathlib import Path

def parse_lpds_filename(filename):
    """Parse LPDS-style filename into entity-value pairs."""

    entities = {}
    name = Path(filename)
    
    # Handle extension
    entities['extension'] = name.suffix
    
    # Split into components
    parts = name.stem.split('_')
    
    # Parse each entity-value pair
    for part in parts:
        if '-' in part:
            key, value = part.split('-', 1)
            entities[key] = value
        else:
            entities['suffix'] = part
            
    return entities 