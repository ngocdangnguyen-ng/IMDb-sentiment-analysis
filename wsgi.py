"""
WSGI entry point for the IMDb Sentiment Analysis web application.
This file is used by Gunicorn to run the application.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wsgi")

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Fix NLTK data path and download resources if needed
try:
    # Make sure NLTK_DATA environment variable is respected
    nltk_data_env = os.environ.get('NLTK_DATA')
    if nltk_data_env:
        logger.info(f"NLTK_DATA is set to: {nltk_data_env}")
        import nltk
        if nltk_data_env not in nltk.data.path:
            nltk.data.path.append(nltk_data_env)
            logger.info(f"Added NLTK_DATA to path: {nltk.data.path}")
except Exception as env_error:
    logger.error(f"Error setting up NLTK data path: {str(env_error)}")

# Run NLTK resource check before importing app
try:
    logger.info("Verifying NLTK resources before starting application...")
    from utils.nltk_check import check_nltk_resources, test_nltk_functionality
    
    # Check NLTK resources and functionality
    resources_available = check_nltk_resources()
    functionality_working = test_nltk_functionality()
    
    if not (resources_available and functionality_working):
        logger.warning("NLTK resources or functionality check failed. The app may not work correctly.")
        # Force download all as a last resort
        try:
            import nltk
            logger.info("Attempting emergency download of all NLTK resources...")
            nltk.download('all')
            logger.info("Emergency download completed")
        except Exception as download_error:
            logger.error(f"Emergency download failed: {str(download_error)}")
    else:
        logger.info("NLTK resources and functionality verified. Proceeding with app startup.")
except Exception as e:
    logger.error(f"Failed to verify NLTK resources: {str(e)}")
    logger.warning("The app will attempt to start, but may encounter NLTK-related errors.")

# Import the Flask app
from app.app import app

if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)