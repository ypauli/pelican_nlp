#!/usr/bin/env python3
"""
Test script to verify perplexity integration with the Pelican pipeline.
"""

import sys
import os
from pathlib import Path

# Add the pelican_nlp module to the path
sys.path.insert(0, str(Path(__file__).parent))

from pelican_nlp.main import Pelican

def test_perplexity_integration():
    """Test the perplexity integration."""
    print("Testing perplexity integration...")
    
    # Use the example perplexity config
    config_path = '/home/yvespauli/PELICAN-nlp/examples/example_perplexity/config_perplexity.yml'
    
    try:
        # Create Pelican instance
        pelican = Pelican(config_path, dev_mode=True)
        
        # Check if perplexity is in the metrics to extract
        metrics = pelican.config.get('metrics_to_extract', [])
        print(f"Metrics to extract: {metrics}")
        
        if 'perplexity' in metrics:
            print("✓ Perplexity is configured in metrics_to_extract")
        else:
            print("✗ Perplexity is not configured in metrics_to_extract")
            return False
            
        # Check if perplexity options are configured
        perplexity_options = pelican.config.get('options_perplexity', {})
        print(f"Perplexity options: {perplexity_options}")
        
        if perplexity_options:
            print("✓ Perplexity options are configured")
        else:
            print("✗ Perplexity options are not configured")
            return False
            
        # Check if logits extraction is enabled (required for perplexity)
        if 'logits' in metrics:
            print("✓ Logits extraction is enabled (required for perplexity)")
        else:
            print("✗ Logits extraction is not enabled (required for perplexity)")
            return False
            
        print("\n✓ All configuration checks passed!")
        print("Perplexity integration appears to be working correctly.")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_perplexity_integration()
    if success:
        print("\n🎉 Perplexity integration test completed successfully!")
    else:
        print("\n❌ Perplexity integration test failed!")
        sys.exit(1)

