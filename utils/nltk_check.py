"""
NLTK Resource Check Utility
Ensures all necessary NLTK resources are available for the IMDb Sentiment Analysis app
"""

import os
import sys
import logging
import nltk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nltk_check")

def check_nltk_resources():
    """
    Check if all required NLTK resources are available and download them if needed
    """
    logger.info("Checking NLTK resource availability...")
    
    required_resources = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt'),  # punkt_tab is actually part of punkt
        ('corpora/stopwords', 'stopwords'),
        ('corpora/wordnet', 'wordnet'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger')
    ]
    
    # Print NLTK data paths
    logger.info(f"NLTK data path: {nltk.data.path}")
    
    # Check if NLTK_DATA environment variable is set
    nltk_data_env = os.environ.get('NLTK_DATA')
    if nltk_data_env:
        logger.info(f"NLTK_DATA environment variable is set to: {nltk_data_env}")
        # Add to path if not already there
        if nltk_data_env not in nltk.data.path:
            nltk.data.path.append(nltk_data_env)
            logger.info(f"Added NLTK_DATA to path: {nltk.data.path}")
    
    # Try to create NLTK data directory if it doesn't exist
    for path in nltk.data.path:
        try:
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
                logger.info(f"Created NLTK data directory: {path}")
        except Exception as e:
            logger.warning(f"Could not create NLTK data directory {path}: {str(e)}")
    
    missing_resources = []
    for resource_path, resource_name in required_resources:
        try:
            # Check if the resource exists
            nltk.data.find(resource_path)
            logger.info(f"✓ {resource_name} is available")
        except LookupError:
            logger.warning(f"✗ {resource_name} is missing")
            missing_resources.append(resource_name)
    
    # Download missing resources
    if missing_resources:
        logger.info(f"Downloading missing resources: {missing_resources}")
        
        # First try to download 'all' if we're missing multiple resources
        if len(missing_resources) > 1:
            try:
                logger.info("Attempting to download all NLTK resources...")
                nltk.download('all', quiet=False)
                logger.info("Successfully downloaded all NLTK resources")
            except Exception as e:
                logger.error(f"Failed to download all resources: {str(e)}")
                # Continue with individual downloads below
        
        # Try individual downloads as backup
        for resource in missing_resources:
            try:
                nltk.download(resource, quiet=False)
                logger.info(f"Successfully downloaded {resource}")
            except Exception as e:
                logger.error(f"Failed to download {resource}: {str(e)}")
                
                # Special handling for punkt_tab
                if resource == 'punkt' or 'punkt_tab' in str(e):
                    try:
                        # Create punkt_tab directory structure if needed
                        for path in nltk.data.path:
                            punkt_tab_dir = os.path.join(path, 'tokenizers', 'punkt_tab', 'english')
                            os.makedirs(punkt_tab_dir, exist_ok=True)
                            logger.info(f"Created directory for punkt_tab: {punkt_tab_dir}")
                    except Exception as mkdir_err:
                        logger.error(f"Failed to create punkt_tab directory: {str(mkdir_err)}")
    
    # Verify downloads were successful
    verification_failures = []
    for resource_path, resource_name in required_resources:
        try:
            # Try to use the resource
            nltk.data.find(resource_path)
        except LookupError:
            verification_failures.append(resource_name)
    
    if verification_failures:
        logger.error(f"The following resources could not be verified: {verification_failures}")
        return False
    else:
        logger.info("All NLTK resources are available")
        return True

def test_nltk_functionality():
    """
    Test NLTK functionality with the available resources
    """
    logger.info("Testing NLTK functionality...")
    
    try:
        # Test tokenization
        from nltk.tokenize import word_tokenize
        test_sentence = "This is a test sentence for NLTK tokenization."
        tokens = word_tokenize(test_sentence)
        logger.info(f"Tokenization test: {tokens[:3]}...")
        
        # Test stopwords
        from nltk.corpus import stopwords
        stop_words = stopwords.words('english')
        logger.info(f"Stopwords test: {len(stop_words)} words available")
        
        logger.info("NLTK functionality tests passed")
        return True
    except Exception as e:
        logger.error(f"NLTK functionality test failed: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Running NLTK resource check script")
    resources_available = check_nltk_resources()
    functionality_working = test_nltk_functionality()
    
    if resources_available and functionality_working:
        logger.info("All NLTK checks passed. The application should work correctly.")
        sys.exit(0)
    else:
        logger.error("NLTK resource check failed. Please address the issues above.")
        sys.exit(1)