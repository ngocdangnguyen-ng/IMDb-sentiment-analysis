"""
Simple data preprocessing for IMDb sentiment analysis
For student learning project
"""

import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# TensorFlow/Keras - compatible import
try:
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
except ImportError:
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

try:
    from .config import *
except ImportError:
    from config import *

def clean_text(text):
    """Basic text cleaning function"""
    # Remove HTML tags
    text = BeautifulSoup(text, 'html.parser').get_text()
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra spaces
    text = ' '.join(text.split())
    
    return text

def preprocess_data():
    """Main preprocessing function"""
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    # Clean text
    print("Cleaning text...")
    df['cleaned_text'] = df['review'].apply(clean_text)
    
    # Convert labels to binary (positive=1, negative=0)
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    
    # Split data
    X = df['cleaned_text'].values
    y = df['label'].values
    
    # Train/test split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_temp
    )
    
    print(f"Train samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def prepare_traditional_ml_data():
    """Prepare data for traditional ML models (TF-IDF)"""
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data()
    
    print("Creating TF-IDF features...")
    # Create TF-IDF vectorizer
    tfidf = TfidfVectorizer(
        max_features=MAX_FEATURES,
        stop_words='english',
        ngram_range=(1, 2)  # Use both unigrams and bigrams
    )
    
    # Fit on training data and transform all sets
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    X_test_tfidf = tfidf.transform(X_test)
    
    # Save processed data
    np.savez(PROCESSED_DATA_PATH + 'traditional_ml_data.npz',
             X_train=X_train_tfidf.toarray(),
             X_val=X_val_tfidf.toarray(), 
             X_test=X_test_tfidf.toarray(),
             y_train=y_train,
             y_val=y_val,
             y_test=y_test)
    
    # Save vectorizer
    with open(MODELS_PATH + 'tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf, f)
    
    print("Traditional ML data saved!")
    return X_train_tfidf, X_val_tfidf, X_test_tfidf, y_train, y_val, y_test

def prepare_deep_learning_data():
    """Prepare data for deep learning models (sequences)"""
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data()
    
    print("Creating sequences for deep learning...")
    # Create tokenizer
    tokenizer = Tokenizer(num_words=MAX_FEATURES, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)
    
    # Convert texts to sequences
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_val_seq = tokenizer.texts_to_sequences(X_val)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    # Pad sequences to same length
    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN)
    X_val_pad = pad_sequences(X_val_seq, maxlen=MAX_LEN)
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN)
    
    # Save processed data
    np.savez(PROCESSED_DATA_PATH + 'deep_learning_data.npz',
             X_train=X_train_pad,
             X_val=X_val_pad,
             X_test=X_test_pad,
             y_train=y_train,
             y_val=y_val,
             y_test=y_test)
    
    # Save tokenizer and config
    with open(MODELS_PATH + 'tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    config = {
        'vocab_size': len(tokenizer.word_index) + 1,
        'max_length': MAX_LEN
    }
    with open(MODELS_PATH + 'preprocessing_config.pkl', 'wb') as f:
        pickle.dump(config, f)
    
    print("Deep learning data saved!")
    return X_train_pad, X_val_pad, X_test_pad, y_train, y_val, y_test

if __name__ == "__main__":
    print("Preparing data for traditional ML...")
    prepare_traditional_ml_data()
    
    print("\nPreparing data for deep learning...")
    prepare_deep_learning_data()
    
    print("\nData preprocessing completed!")