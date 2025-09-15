"""
Utility functions for IMDb sentiment analysis project
Simple helper functions for students
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import pickle

def plot_training_history(history, model_name="Model"):
    """Plot training history for deep learning models"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title(f'{model_name} - Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title(f'{model_name} - Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, model_name="Model"):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def compare_models(results_dict):
    """Compare multiple models performance"""
    models = list(results_dict.keys())
    accuracies = [results_dict[model]['accuracy'] for model in models]
    f1_scores = [results_dict[model]['f1'] for model in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
    bars2 = ax.bar(x + width/2, f1_scores, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def print_model_summary(results_dict):
    """Print a nice summary of model results"""
    print("\n" + "="*60)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*60)
    
    df = pd.DataFrame(results_dict).T
    df = df.round(4)
    
    # Sort by F1 score
    if 'f1' in df.columns:
        df = df.sort_values('f1', ascending=False)
    elif 'accuracy' in df.columns:
        df = df.sort_values('accuracy', ascending=False)
    
    print(df.to_string())
    
    # Find best model
    if 'f1' in df.columns:
        best_model = df.index[0]
        best_score = df.loc[best_model, 'f1']
        print(f"\nBest Model: {best_model} (F1-Score: {best_score:.4f})")
    
    return df

def save_results(results, filename):
    """Save results to file"""
    with open(f"../results/{filename}", 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to ../results/{filename}")

def load_results(filename):
    """Load results from file"""
    with open(f"../results/{filename}", 'rb') as f:
        results = pickle.load(f)
    return results

def create_prediction_report(y_true, y_pred, model_name="Model"):
    """Create detailed classification report"""
    print(f"\n{model_name} - Classification Report:")
    print("="*50)
    print(classification_report(y_true, y_pred, 
                               target_names=['Negative', 'Positive']))

def plot_word_importance(model, vectorizer, top_n=20):
    """Plot most important words for traditional ML models"""
    try:
        if hasattr(model, 'coef_'):
            feature_names = vectorizer.get_feature_names_out()
            coef = model.coef_[0]
            
            # Get top positive and negative words
            top_positive_idx = coef.argsort()[-top_n:][::-1]
            top_negative_idx = coef.argsort()[:top_n]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Positive words
            pos_words = [feature_names[i] for i in top_positive_idx]
            pos_scores = [coef[i] for i in top_positive_idx]
            
            ax1.barh(range(len(pos_words)), pos_scores, color='green', alpha=0.7)
            ax1.set_yticks(range(len(pos_words)))
            ax1.set_yticklabels(pos_words)
            ax1.set_title('Top Positive Words')
            ax1.set_xlabel('Coefficient Value')
            
            # Negative words  
            neg_words = [feature_names[i] for i in top_negative_idx]
            neg_scores = [coef[i] for i in top_negative_idx]
            
            ax2.barh(range(len(neg_words)), neg_scores, color='red', alpha=0.7)
            ax2.set_yticks(range(len(neg_words)))
            ax2.set_yticklabels(neg_words)
            ax2.set_title('Top Negative Words')
            ax2.set_xlabel('Coefficient Value')
            
            plt.tight_layout()
            plt.show()
            
    except Exception as e:
        print(f"Could not plot word importance: {e}")

def predict_sentiment(text, model, vectorizer, model_type="traditional"):
    """Make prediction on new text"""
    if model_type == "traditional":
        # Clean and vectorize text
        text_vector = vectorizer.transform([text])
        prediction = model.predict(text_vector)[0]
        try:
            confidence = model.predict_proba(text_vector)[0].max()
        except:
            confidence = abs(model.decision_function(text_vector)[0])
    else:  # deep learning
        # This would need tokenizer and padding - simplified here
        prediction = model.predict([text])[0]
        confidence = prediction if prediction > 0.5 else 1 - prediction
        prediction = 1 if prediction > 0.5 else 0
    
    sentiment = "Positive" if prediction == 1 else "Negative"
    return sentiment, confidence

if __name__ == "__main__":
    print("Utility functions loaded successfully!")
    print("Available functions:")
    print("- plot_training_history()")
    print("- plot_confusion_matrix()")
    print("- compare_models()")
    print("- print_model_summary()")
    print("- save_results() / load_results()")
    print("- create_prediction_report()")
    print("- plot_word_importance()")
    print("- predict_sentiment()")