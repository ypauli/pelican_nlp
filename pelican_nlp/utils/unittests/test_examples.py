import unittest
import os
import yaml
from pathlib import Path
import shutil
import tempfile
import json
import subprocess
import sys
import logging
import signal
from contextlib import contextmanager

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from pelican_nlp.config import DEBUG_MODE, debug_print

class TimeoutError(Exception):
    pass

@contextmanager
def timeout(seconds):
    def signal_handler(signum, frame):
        raise TimeoutError(f"Timed out after {seconds} seconds")
    
    # Register the signal handler
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Disable the alarm
        signal.alarm(0)

class TestExamples(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        debug_print("Setting up test environment...")
        # Create a temporary directory for test outputs
        cls.test_dir = tempfile.mkdtemp()
        cls.examples_dir = Path(__file__).parent / "examples"
        
        # Load all example configurations
        cls.examples = {}
        for example_type in ["fluency", "discourse", "image-descriptions"]:
            example_dir = cls.examples_dir / f"example_{example_type}"
            config_path = example_dir / f"config_{example_type}.yml"
            
            debug_print(f"Loading configuration for {example_type}...")
            if not config_path.exists():
                debug_print(f"Warning: Config file not found: {config_path}")
                continue
                
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            cls.examples[example_type] = {
                "config_path": config_path,
                "config": config,
                "example_dir": example_dir
            }

    @classmethod
    def tearDownClass(cls):
        debug_print("Cleaning up test environment...")
        # Clean up temporary directory
        shutil.rmtree(cls.test_dir)

    def setUp(self):
        # Create a fresh output directory for each test
        self.output_dir = Path(self.test_dir) / "test_output"
        self.output_dir.mkdir(exist_ok=True)

    def run_pelican_pipeline(self, example_dir, config_path, output_dir):
        """Run the pelican pipeline with the given configuration file"""
        debug_print(f"Running pipeline with config: {config_path}")
        try:
            # Change to the example directory before running the command
            original_dir = os.getcwd()
            os.chdir(example_dir)
            
            # Print current directory and files
            debug_print(f"Current directory: {os.getcwd()}")
            debug_print("Files in current directory:")
            for f in os.listdir('.'):
                debug_print(f"  - {f}")
            
            # Run pelican-run with the configuration file and timeout
            with timeout(300):  # 5 minute timeout
                # Use run with real-time output
                process = subprocess.run(
                    ["pelican-run", "--config", str(config_path), "--output", str(output_dir)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True
                )
                
                # Print output after completion
                if process.stdout:
                    print("Pipeline output:")
                    print(process.stdout)
                if process.stderr:
                    print("Pipeline errors:")
                    print(process.stderr)
            
            # Change back to original directory
            os.chdir(original_dir)
            
            debug_print("Pipeline completed successfully")
            return True, "Pipeline completed successfully"
        except TimeoutError as e:
            os.chdir(original_dir)
            debug_print(f"Pipeline timed out: {str(e)}")
            return False, f"Error: Pipeline timed out after 5 minutes"
        except subprocess.CalledProcessError as e:
            # Change back to original directory even if there's an error
            os.chdir(original_dir)
            debug_print(f"Pipeline failed with exit code {e.returncode}")
            if e.stdout:
                print("Pipeline output:")
                print(e.stdout)
            if e.stderr:
                print("Pipeline errors:")
                print(e.stderr)
            return False, f"Error: Pipeline failed with exit code {e.returncode}"
        except Exception as e:
            os.chdir(original_dir)
            debug_print(f"Unexpected error: {str(e)}")
            return False, f"Error: {str(e)}"

    def test_discourse_example(self):
        """Test running the discourse example through the pipeline"""
        debug_print("Testing discourse example...")
        if "discourse" not in self.examples:
            self.skipTest("Discourse example configuration not found")
            
        example = self.examples["discourse"]
        output_dir = self.output_dir / "discourse"
        output_dir.mkdir(exist_ok=True)
        
        success, output = self.run_pelican_pipeline(
            example["example_dir"],
            example["config_path"],
            output_dir
        )
        self.assertTrue(success, f"Pipeline failed: {output}")
        
        # Verify output files were created
        self.assertTrue(output_dir.exists())
        self.assertTrue(len(list(output_dir.glob("*"))) > 0)
        debug_print("Discourse example test completed")

    def test_fluency_example(self):
        """Test running the fluency example through the pipeline"""
        debug_print("Testing fluency example...")
        if "fluency" not in self.examples:
            self.skipTest("Fluency example configuration not found")
            
        example = self.examples["fluency"]
        output_dir = self.output_dir / "fluency"
        output_dir.mkdir(exist_ok=True)
        
        success, output = self.run_pelican_pipeline(
            example["example_dir"],
            example["config_path"],
            output_dir
        )
        self.assertTrue(success, f"Pipeline failed: {output}")
        
        # Verify output files were created
        self.assertTrue(output_dir.exists())
        self.assertTrue(len(list(output_dir.glob("*"))) > 0)
        debug_print("Fluency example test completed")

    def test_image_descriptions_example(self):
        """Test running the image descriptions example through the pipeline"""
        debug_print("Testing image descriptions example...")
        if "image-descriptions" not in self.examples:
            self.skipTest("Image descriptions example configuration not found")
            
        example = self.examples["image-descriptions"]
        output_dir = self.output_dir / "image-descriptions"
        output_dir.mkdir(exist_ok=True)
        
        success, output = self.run_pelican_pipeline(
            example["example_dir"],
            example["config_path"],
            output_dir
        )
        self.assertTrue(success, f"Pipeline failed: {output}")
        
        # Verify output files were created
        self.assertTrue(output_dir.exists())
        self.assertTrue(len(list(output_dir.glob("*"))) > 0)
        debug_print("Image descriptions example test completed")

def suite():
    """Create a test suite with all test cases"""
    suite = unittest.TestSuite()
    suite.addTest(TestExamples('test_discourse_example'))
    suite.addTest(TestExamples('test_fluency_example'))
    suite.addTest(TestExamples('test_image_descriptions_example'))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite()) 