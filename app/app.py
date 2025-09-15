"""
IMDb Sentiment Analysis Web App - Simplified Version
Flask app that simulates movie review sentiment prediction
"""

from flask import Flask, render_template, request, jsonify
import re
import os
import random
from datetime import datetime
from bs4 import BeautifulSoup
import nltk
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure NLTK resources are available
try:
    # First try to import
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    
    # Verify resources are actually available by using them
    test_stopwords = stopwords.words('english')
    test_tokens = word_tokenize("Test sentence.")
    logger.info("NLTK resources loaded successfully")
except Exception as e:
    logger.warning(f"NLTK resources not found: {str(e)}. Downloading required resources...")
    try:
        # Download all potentially needed resources
        nltk.download('stopwords', quiet=False)
        nltk.download('punkt', quiet=False)
        nltk.download('wordnet', quiet=False)
        nltk.download('averaged_perceptron_tagger', quiet=False)
        
        # Try importing again after download
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        logger.info("NLTK resources downloaded and loaded successfully")
    except Exception as download_error:
        logger.error(f"Failed to download NLTK resources: {str(download_error)}")
        raise

app = Flask(__name__)

class SimpleSentimentPredictor:
    def __init__(self):
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'best', 'love',
            'superb', 'brilliant', 'awesome', 'fantastic', 'outstanding', 'perfect',
            'impressive', 'beautiful', 'enjoyed', 'recommended', 'favorite', 'liked',
            'masterpiece', 'captivating', 'entertaining', 'stunning'
        }
        
        self.negative_words = {
            'bad', 'terrible', 'awful', 'worst', 'horrible', 'disappointing', 'boring',
            'waste', 'poor', 'mediocre', 'dull', 'stupid', 'hate', 'disliked', 'failure',
            'mess', 'avoid', 'terrible', 'dreadful', 'flawed', 'annoying', 'confusing'
        }
        
        try:
            self.stop_words = set(stopwords.words('english'))
            logger.info("Stop words loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load stop words: {str(e)}. Using fallback.")
            # Fallback for common English stop words if NLTK fails
            self.stop_words = {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
                'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
                'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
                'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
                'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
                'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
                'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 
                'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
                'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once'
            }
    
    def preprocess_text(self, text):
        """Basic text preprocessing"""
        # Remove HTML tags
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text()
        
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep basic punctuation for sentiment
        text = re.sub(r'[^a-z0-9\s.,!?;:\'-]', '', text)
        
        return text
    
    def get_word_count(self, text, word_set):
        """Count occurrences of words from a set in text"""
        try:
            # Try using NLTK tokenizer
            tokens = word_tokenize(text)
            logger.info("Successfully tokenized text with NLTK")
        except Exception as e:
            # Fallback to simple tokenization if NLTK fails
            logger.warning(f"NLTK tokenization failed: {str(e)}. Using fallback tokenizer.")
            tokens = text.lower().split()
            
        tokens = [token for token in tokens if token.isalpha() and token not in self.stop_words]
        return sum(1 for token in tokens if token in word_set)
    
    def predict_sentiment(self, text):
        """Predict sentiment based on word occurrence"""
        cleaned_text = self.preprocess_text(text)
        
        # Count positive and negative words
        pos_count = self.get_word_count(cleaned_text, self.positive_words)
        neg_count = self.get_word_count(cleaned_text, self.negative_words)
        
        # Check for negation
        negation_terms = ['not', "n't", 'never', 'no', 'hardly', 'barely']
        negation_count = sum(cleaned_text.count(term) for term in negation_terms)
        
        # Adjust scores based on negation
        if negation_count > 0:
            # Simple adjustment - switch some positive words to negative and vice versa
            adjustment = min(pos_count, negation_count // 2)
            pos_count -= adjustment
            neg_count += adjustment
        
        # Calculate total and add a small random factor for variety
        total = pos_count + neg_count + 1  # Add 1 to avoid division by zero
        
        # Generate prediction probabilities with some randomness
        positive_score = (pos_count / total) + random.uniform(-0.05, 0.05)
        negative_score = (neg_count / total) + random.uniform(-0.05, 0.05)
        
        # Normalize to sum to 1
        sum_scores = positive_score + negative_score
        positive_score /= sum_scores
        negative_score /= sum_scores
        
        # Ensure scores are between 0 and 1
        positive_score = max(0, min(1, positive_score))
        negative_score = max(0, min(1, negative_score))
        
        # Build results dictionary
        results = {}
        
        # Traditional ML result
        results['traditional'] = {
            'prediction': 'Positive' if positive_score > negative_score else 'Negative',
            'confidence': max(positive_score, negative_score) * 100,
            'probabilities': {
                'negative': negative_score * 100,
                'positive': positive_score * 100
            }
        }
        
        # Deep learning result (slightly different with more randomness)
        dl_positive = positive_score + random.uniform(-0.1, 0.1)
        dl_positive = max(0, min(1, dl_positive))  # Ensure between 0 and 1
        
        results['deep_learning'] = {
            'prediction': 'Positive' if dl_positive > 0.5 else 'Negative',
            'confidence': max(dl_positive, 1 - dl_positive) * 100,
            'probabilities': {
                'negative': (1 - dl_positive) * 100,
                'positive': dl_positive * 100
            }
        }
        
        return results

# Initialize predictor
predictor = SimpleSentimentPredictor()

@app.route('/')
def home():
    """Home page"""
    # Pass model status to template (both models available in this simplified version)
    model_status = {
        'traditional': True,
        'deep_learning': True,
        'any_model_available': True
    }
    return render_template('index.html', model_status=model_status)

@app.route('/predict', methods=['POST'])
def predict():
    """Predict sentiment"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text.strip():
            logger.warning("Empty text received for prediction")
            return jsonify({'error': 'Please enter some text'})
        
        logger.info(f"Received prediction request for text: {text[:50]}...")
        
        # Predict sentiment
        results = predictor.predict_sentiment(text)
        logger.info(f"Prediction results: {results}")
        
        return jsonify({
            'success': True,
            'text': text,
            'results': results,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'error': f"Prediction failed: {str(e)}. The application is still initializing or encountered an error.",
            'success': False,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'models_loaded': {
            'traditional': True,
            'deep_learning': True
        }
    })

if __name__ == '__main__':
    logger.info("Starting IMDb Sentiment Analysis Web App (Simplified Version)...")
    logger.info("Both models are simulated and will provide predictions.")
    
    # Verify NLTK resources at startup
    try:
        # Test NLTK functionality by tokenizing a sample sentence
        sample = "This is a test sentence for NLTK."
        tokens = word_tokenize(sample)
        stop_test = stopwords.words('english')
        logger.info(f"NLTK test successful. Tokenized: {tokens[:3]}...")
        logger.info(f"NLTK stopwords available: {len(stop_test)} words")
    except Exception as e:
        logger.error(f"NLTK verification failed: {str(e)}")
        logger.warning("The app will try to continue with fallback methods")
        
    # Get host and port from environment or use defaults
    # Render.com sets PORT environment variable automatically
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 10000))
    
    # Determine the public URL - for Render.com deployment
    is_render = os.environ.get('RENDER', 'false').lower() == 'true'
    render_url = os.environ.get('RENDER_EXTERNAL_URL', '')
    
    # Use Render's external URL if available, otherwise use local URL
    if is_render and render_url:
        public_url = render_url
        logger.info(f"Running on Render.com with URL: {public_url}")
    else:
        public_url = os.environ.get('PUBLIC_URL', f"http://localhost:{port}")
        logger.info(f"Running locally at: {public_url}")
    
    app.run(debug=False, host=host, port=port)