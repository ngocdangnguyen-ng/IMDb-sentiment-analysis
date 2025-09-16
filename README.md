# IMDb Sentiment Analysis - Personal Machine Learning Project

## Project Overview
This is a personal learning project where I explore both traditional machine learning and deep learning techniques for natural language processing, specifically movie review sentiment analysis. As a final-year computer science student passionate about AI and data science, I built this project to practice the complete ML workflow from data exploration to model deployment. The project demonstrates key NLP techniques and compares different approaches to text classification.

**Objective:** 
To develop accurate sentiment classification models for movie reviews using real-world IMDb data, while gaining hands-on experience with the end-to-end machine learning pipeline for text analysis.

## Key Results

| Model Type | Model | Validation Accuracy | Test Accuracy | Notes |
|------------|-------|-------------------|---------------|-------|
| Traditional ML | Logistic Regression | 89.66% | 89.34% | TF-IDF features, best traditional |
| Traditional ML | Linear SVM | 88.66% | - | Strong performance |
| Traditional ML | Naive Bayes | 86.60% | - | Fast training |
| Traditional ML | Random Forest | 84.10% | - | Ensemble method |
| Deep Learning | CNN | 90.34% | 89.90% | **Best overall performance** |
| Deep Learning | LSTM | 89.28% | - | Sequential processing |
| Deep Learning | Bidirectional LSTM | 89.52% | - | Captures context better |

The **CNN model** achieved the best performance with **90.34% validation accuracy** and **89.90% test accuracy**, slightly outperforming traditional ML approaches. This demonstrates that for this dataset, both approaches are highly effective, with deep learning providing a marginal advantage.

## Features
- **Complete NLP pipeline**: Text cleaning, preprocessing, feature extraction, modeling
- **Dual approach comparison**: Traditional ML vs Deep Learning
- **Professional web application** for real-time sentiment prediction
- **Comprehensive evaluation**: Accuracy, F1-score, ROC-AUC, confusion matrices
- **Modular, reusable codebase** with clean separation of concerns
- **Interactive Jupyter notebooks** for step-by-step exploration
- **Production-ready models** saved for deployment

## Getting Started

**Installation**
```bash
git clone https://github.com/ngocdangnguyen-ng/IMDb-sentiment-analysis.git
cd IMDb-sentiment-analysis
pip install -r requirements.txt

# Download NLTK data (required for text preprocessing)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

**Quick Prediction Example**
```python
# Load the trained models and make predictions
import pickle
from keras.models import load_model

# Load traditional ML model (89.66% accuracy)
with open('models/saved_models/best_traditional_model.pkl', 'rb') as f:
    traditional_model = pickle.load(f)

# Load deep learning model (90.34% accuracy)
dl_model = load_model('models/saved_models/best_dl_model.h5')

# Example prediction
review = "This movie was absolutely fantastic! Great acting and plot."
# ... (see notebooks for complete preprocessing pipeline)
# prediction = model.predict(processed_review)
```

**Web Application**
```bash
# Start the Flask web app
cd app
python app.py
# Visit http://localhost:5000 for interactive predictions
```

## Project Structure
```python
IMDb-sentiment-analysis/
│
├── app/                           # Web application
│   ├── app.py                     # Main Flask application
│   └── templates/                 # HTML templates
│       └── index.html             # Main web interface
│
├── data/
│   ├── raw/                       # Original dataset
│   └── processed/                 # Preprocessed data files
│
├── models/
│   ├── saved_models/              # Trained model files
│   │   ├── best_dl_model.h5       # Deep learning model (Git LFS)
│   │   └── best_traditional_model.pkl # Traditional ML model
│   └── preprocessors/             # Feature transformation objects
│
├── notebooks/                     # Jupyter notebooks
│   ├── 01_data_exploration.ipynb  # Initial data analysis
│   ├── 02_data_preprocessing.ipynb# Text preparation workflows
│   ├── 03_model_training.ipynb    # Model development
│   └── 04_model_evaluation.ipynb  # Performance analysis
│
├── src/                           # Source code
│   ├── __init__.py                # Package initialization
│   ├── config.py                  # Configuration settings
│   ├── data_preprocessing.py      # Text preprocessing functions
│   ├── models.py                  # Model definitions
│   └── utils.py                   # Utility functions
│
├── utils/                         # Additional utilities
│   └── nltk_check.py              # NLTK resource verification
│
├── DETAILS.md                     # Technical deep dive
├── Dockerfile                     # Docker container configuration
├── docker-compose.yml             # Multi-container orchestration
├── .gitattributes                 # Git LFS configuration
├── .gitignore                     # Git ignore patterns
├── render.yaml                    # Render.com deployment settings
├── requirements.txt               # Python dependencies
├── wsgi.py                        # WSGI entry point for production
└── README.md                      # This file                  
```

## Process Overview

1. **Data Exploration**: Analyzed 50,000 IMDb movie reviews, examined text distributions, and identified preprocessing needs  
2. **Data Preprocessing**: Text cleaning, TF-IDF vectorization for traditional ML, and tokenization for deep learning  
3. **Model Development**: Built and compared traditional ML (Logistic Regression, SVM) and deep learning models (CNN, LSTM) with proper validation  
4. **Deployment**: Created Flask web application for real-time sentiment prediction with interactive interface

## What I Learned

- **Natural Language Processing**: Text preprocessing, feature extraction, TF-IDF vectorization, and sequence modeling
- **Model Comparison**: Systematic evaluation of traditional ML vs deep learning approaches with proper validation  
- **Web Deployment**: Flask application development with real-time model serving and user-friendly interface
- **Performance Optimization**: Memory management for large datasets and model optimization techniques
- **Technical Challenge**: Overcame TensorFlow compatibility issues and deployed production-ready models

*For detailed technical documentation, challenges overcome, and comprehensive analysis, see [DETAILS.md](DETAILS.md)*

## Contact
- **Email**: [nndnguyen2016@gmail.com](mailto:nndnguyen2016@gmail.com)
- **LinkedIn**: [https://www.linkedin.com/in/ngocnguyen-fr](https://www.linkedin.com/in/ngocnguyen-fr)
- **Portfolio:** [https://portfolio-qyyg.onrender.com](https://portfolio-qyyg.onrender.com)

---
I welcome feedback and suggestions for improvement. Thank you for visiting my project!

## License
This project is licensed under the MIT License. See the LICENSE file for details.
