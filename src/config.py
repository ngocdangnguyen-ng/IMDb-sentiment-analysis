"""
Configuration file for IMDb Sentiment Analysis project
Simple configuration for student learning project
"""

import os

# Random seed for reproducibility
RANDOM_STATE = 42

# Data paths
DATA_PATH = "../data/raw/IMDB Dataset.csv"
PROCESSED_DATA_PATH = "../data/processed/"
MODELS_PATH = "../models/"

# Text preprocessing parameters
MAX_FEATURES = 10000  # Maximum number of words to keep
MAX_LEN = 200         # Maximum length of review
TEST_SIZE = 0.2       # 20% for testing
VAL_SIZE = 0.2        # 20% for validation

# Model parameters
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

# Create directories if they don't exist
os.makedirs("../data/processed", exist_ok=True)
os.makedirs("../models", exist_ok=True)
os.makedirs("../results", exist_ok=True)