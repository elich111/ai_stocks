
import unittest
import sys
import os

# Add the project root to the Python path to allow for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == "__main__":
    # Discover and run all tests in the 'tests' directory
    loader = unittest.TestLoader()
    suite = loader.discover('tests')
    runner = unittest.TextTestRunner()
    result = runner.run(suite)

    # Exit with a non-zero status code if any tests failed
    if not result.wasSuccessful():
        sys.exit(1)
