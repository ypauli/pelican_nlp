from pathlib import Path

def parse_bids_filename(filename):
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
            entity, value = part.split('-', 1)
            entities[entity] = value
        else:
            entities['suffix'] = part
            
    return entities 