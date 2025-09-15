"""
Simple model training for IMDb sentiment analysis
Traditional ML and Deep Learning models for students
"""

import numpy as np
import pickle
import time
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# TensorFlow/Keras - compatible import
import tensorflow as tf
try:
    from keras.models import Sequential
    from keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, Conv1D, MaxPooling1D, GlobalMaxPooling1D
    from keras.callbacks import EarlyStopping
except ImportError:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, Conv1D, MaxPooling1D, GlobalMaxPooling1D
    from tensorflow.keras.callbacks import EarlyStopping

try:
    from .config import *
except ImportError:
    from config import *

def evaluate_model(y_true, y_pred, model_name):
    """Simple model evaluation"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def train_traditional_models():
    """Train traditional ML models"""
    print("Loading traditional ML data...")
    data = np.load(PROCESSED_DATA_PATH + 'traditional_ml_data.npz')
    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        'Naive Bayes': MultinomialNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        'Linear SVM': LinearSVC(random_state=RANDOM_STATE, max_iter=1000)
    }
    
    results = {}
    best_model = None
    best_score = 0
    
    print("\nTraining Traditional ML Models:")
    print("="*40)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_time = time.time()
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_val_pred = model.predict(X_val)
        
        # Evaluate
        metrics = evaluate_model(y_val, y_val_pred, name)
        metrics['training_time'] = time.time() - start_time
        results[name] = metrics
        
        # Track best model
        if metrics['f1'] > best_score:
            best_score = metrics['f1']
            best_model = name
            
            # Save best model
            with open(MODELS_PATH + f'{name.lower().replace(" ", "_")}.pkl', 'wb') as f:
                pickle.dump(model, f)
    
    print(f"\nBest Traditional Model: {best_model} (F1: {best_score:.4f})")
    return results, best_model

def create_lstm_model(vocab_size, max_len):
    """Create simple LSTM model"""
    model = Sequential([
        Embedding(vocab_size, 128, input_length=max_len),
        LSTM(64, dropout=0.5),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_bilstm_model(vocab_size, max_len):
    """Create Bidirectional LSTM model"""
    model = Sequential([
        Embedding(vocab_size, 128, input_length=max_len),
        Bidirectional(LSTM(64, dropout=0.5)),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_deep_learning_models():
    """Train deep learning models"""
    print("Loading deep learning data...")
    data = np.load(PROCESSED_DATA_PATH + 'deep_learning_data.npz')
    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    
    # Load config
    with open(MODELS_PATH + 'preprocessing_config.pkl', 'rb') as f:
        config = pickle.load(f)
    
    vocab_size = config['vocab_size']
    max_len = config['max_length']
    
    models = {
        'LSTM': create_lstm_model(vocab_size, max_len),
        'BiLSTM': create_bilstm_model(vocab_size, max_len)
    }
    
    results = {}
    best_model = None
    best_score = 0
    
    print("\nTraining Deep Learning Models:")
    print("="*40)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_time = time.time()
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Evaluate on validation set
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        
        # Make predictions for other metrics
        y_val_pred_proba = model.predict(X_val, verbose=0)
        y_val_pred = (y_val_pred_proba > 0.5).astype(int).flatten()
        
        # Calculate metrics
        precision = precision_score(y_val, y_val_pred)
        recall = recall_score(y_val, y_val_pred)
        f1 = f1_score(y_val, y_val_pred)
        
        print(f"\n{name} Results:")
        print(f"Accuracy: {val_acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        metrics = {
            'accuracy': val_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'training_time': time.time() - start_time
        }
        
        results[name] = metrics
        
        # Track best model
        if val_acc > best_score:
            best_score = val_acc
            best_model = name
            
            # Save best model
            model.save(MODELS_PATH + f'{name.lower()}_best.h5')
    
    print(f"\nBest Deep Learning Model: {best_model} (Accuracy: {best_score:.4f})")
    return results, best_model

def compare_all_models():
    """Train and compare all models"""
    print("Starting model training and comparison...")
    
    # Train traditional ML models
    trad_results, best_trad = train_traditional_models()
    
    # Train deep learning models  
    dl_results, best_dl = train_deep_learning_models()
    
    # Compare best models
    print("\n" + "="*50)
    print("FINAL COMPARISON")
    print("="*50)
    
    print("\nTraditional ML Results:")
    for name, metrics in trad_results.items():
        print(f"{name}: F1={metrics['f1']:.4f}, Acc={metrics['accuracy']:.4f}")
    
    print("\nDeep Learning Results:")
    for name, metrics in dl_results.items():
        print(f"{name}: F1={metrics['f1']:.4f}, Acc={metrics['accuracy']:.4f}")
    
    # Determine overall best
    best_trad_score = trad_results[best_trad]['f1']
    best_dl_score = dl_results[best_dl]['f1']
    
    if best_trad_score > best_dl_score:
        overall_best = best_trad
        print(f"\nOverall Best Model: {overall_best} (Traditional ML)")
    else:
        overall_best = best_dl  
        print(f"\nOverall Best Model: {overall_best} (Deep Learning)")
    
    # Save results
    final_results = {
        'traditional_results': trad_results,
        'deep_learning_results': dl_results,
        'best_traditional': best_trad,
        'best_deep_learning': best_dl,
        'overall_best': overall_best
    }
    
    with open('../results/training_results.pkl', 'wb') as f:
        pickle.dump(final_results, f)
    
    print("Training completed and results saved!")
    return final_results

if __name__ == "__main__":
    compare_all_models()