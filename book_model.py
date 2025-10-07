#!/usr/bin/env python3
"""
================================================================================
PROPER MACHINE LEARNING BOOK RECOMMENDATION SYSTEM
================================================================================
This implements REAL ML models with:
✓ Train/Test split
✓ Model training and evaluation
✓ Accuracy metrics (RMSE, MAE, Precision@K, Recall@K)
✓ Matrix Factorization (SVD)
✓ Neural Collaborative Filtering
✓ Hyperparameter tuning
✓ Model comparison
================================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("PROPER ML BOOK RECOMMENDATION SYSTEM")
print("=" * 80)

# ============================================================================
# SECTION 1: DATA LOADING
# ============================================================================

def load_data():
    """Load Books and Ratings datasets."""
    print("\n" + "=" * 80)
    print("SECTION 1: LOADING DATA")
    print("=" * 80)
    
    try:
        # Load books
        books = pd.read_csv('Books.csv', encoding='latin-1', low_memory=False)
        print(f"✓ Books loaded: {books.shape}")
        
        # Load ratings
        ratings = pd.read_csv('Ratings.csv', encoding='latin-1')
        print(f"✓ Ratings loaded: {ratings.shape}")
        
        return books, ratings
    except FileNotFoundError as e:
        print(f"⚠ Error: {e}")
        print("Creating sample dataset for demonstration...")
        return create_sample_data()

def create_sample_data():
    """Create sample dataset if files not found."""
    np.random.seed(42)
    
    # Sample books
    books = pd.DataFrame({
        'ISBN': [f'ISBN{i:04d}' for i in range(100)],
        'Book-Title': [f'Book {i}' for i in range(100)],
        'Book-Author': [f'Author {i%20}' for i in range(100)],
        'Year-Of-Publication': np.random.randint(1990, 2024, 100),
        'Publisher': [f'Publisher {i%10}' for i in range(100)]
    })
    
    # Sample ratings (user-book interactions)
    n_ratings = 5000
    ratings = pd.DataFrame({
        'User-ID': np.random.randint(1, 301, n_ratings),
        'ISBN': np.random.choice(books['ISBN'], n_ratings),
        'Book-Rating': np.random.randint(1, 11, n_ratings)
    })
    
    print(f"✓ Sample books created: {books.shape}")
    print(f"✓ Sample ratings created: {ratings.shape}")
    
    return books, ratings

# ============================================================================
# SECTION 2: DATA PREPROCESSING
# ============================================================================

def preprocess_data(books, ratings):
    """Clean and prepare data for ML."""
    print("\n" + "=" * 80)
    print("SECTION 2: DATA PREPROCESSING")
    print("=" * 80)
    
    # Remove implicit ratings (0 ratings)
    print(f"\nInitial ratings: {len(ratings)}")
    ratings = ratings[ratings['Book-Rating'] > 0]
    print(f"After removing implicit ratings (0s): {len(ratings)}")
    
    # Filter users and books with minimum interactions
    min_user_ratings = 5
    min_book_ratings = 5
    
    # Count interactions
    user_counts = ratings['User-ID'].value_counts()
    book_counts = ratings['ISBN'].value_counts()
    
    # Filter
    valid_users = user_counts[user_counts >= min_user_ratings].index
    valid_books = book_counts[book_counts >= min_book_ratings].index
    
    ratings = ratings[ratings['User-ID'].isin(valid_users)]
    ratings = ratings[ratings['ISBN'].isin(valid_books)]
    
    print(f"After filtering (min {min_user_ratings} ratings per user/book): {len(ratings)}")
    print(f"✓ Unique users: {ratings['User-ID'].nunique()}")
    print(f"✓ Unique books: {ratings['ISBN'].nunique()}")
    
    # Statistics
    print(f"\nRating Statistics:")
    print(ratings['Book-Rating'].describe())
    
    # Sparsity
    n_users = ratings['User-ID'].nunique()
    n_books = ratings['ISBN'].nunique()
    sparsity = 1 - (len(ratings) / (n_users * n_books))
    print(f"\n✓ Matrix sparsity: {sparsity*100:.2f}%")
    
    return ratings

# ============================================================================
# SECTION 3: TRAIN/TEST SPLIT
# ============================================================================

def create_train_test_split(ratings, test_size=0.2):
    """Split data into train and test sets."""
    print("\n" + "=" * 80)
    print("SECTION 3: TRAIN/TEST SPLIT")
    print("=" * 80)
    
    train, test = train_test_split(ratings, test_size=test_size, random_state=42)
    
    print(f"✓ Training set: {len(train)} ratings ({(1-test_size)*100:.0f}%)")
    print(f"✓ Test set: {len(test)} ratings ({test_size*100:.0f}%)")
    
    return train, test

# ============================================================================
# SECTION 4: BASELINE MODEL - GLOBAL AVERAGE
# ============================================================================

class GlobalAverageModel:
    """Baseline: Predict global average rating."""
    
    def __init__(self):
        self.global_mean = None
    
    def fit(self, train):
        self.global_mean = train['Book-Rating'].mean()
        return self
    
    def predict(self, test):
        return np.full(len(test), self.global_mean)
    
    def get_name(self):
        return "Global Average Baseline"

# ============================================================================
# SECTION 5: USER/ITEM AVERAGE MODEL
# ============================================================================

class UserItemAverageModel:
    """Predict using user and item averages."""
    
    def __init__(self):
        self.global_mean = None
        self.user_means = None
        self.item_means = None
    
    def fit(self, train):
        self.global_mean = train['Book-Rating'].mean()
        self.user_means = train.groupby('User-ID')['Book-Rating'].mean()
        self.item_means = train.groupby('ISBN')['Book-Rating'].mean()
        return self
    
    def predict(self, test):
        predictions = []
        for _, row in test.iterrows():
            user_id = row['User-ID']
            isbn = row['ISBN']
            
            user_avg = self.user_means.get(user_id, self.global_mean)
            item_avg = self.item_means.get(isbn, self.global_mean)
            
            # Weighted average
            pred = 0.5 * user_avg + 0.5 * item_avg
            predictions.append(pred)
        
        return np.array(predictions)
    
    def get_name(self):
        return "User-Item Average"

# ============================================================================
# SECTION 6: MATRIX FACTORIZATION (SVD)
# ============================================================================

class MatrixFactorizationSVD:
    """Collaborative filtering using SVD."""
    
    def __init__(self, n_factors=50):
        self.n_factors = n_factors
        self.svd = None
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.user_factors = None
        self.item_factors = None
        self.global_mean = None
        self.user_bias = None
        self.item_bias = None
    
    def fit(self, train):
        print(f"\n  Training SVD with {self.n_factors} factors...")
        
        # Encode users and items
        train = train.copy()
        train['user_idx'] = self.user_encoder.fit_transform(train['User-ID'])
        train['item_idx'] = self.item_encoder.fit_transform(train['ISBN'])
        
        # Create user-item matrix
        n_users = train['user_idx'].nunique()
        n_items = train['item_idx'].nunique()
        
        user_item_matrix = np.zeros((n_users, n_items))
        for _, row in train.iterrows():
            user_item_matrix[row['user_idx'], row['item_idx']] = row['Book-Rating']
        
        # Calculate biases
        self.global_mean = train['Book-Rating'].mean()
        self.user_bias = train.groupby('user_idx')['Book-Rating'].mean() - self.global_mean
        self.item_bias = train.groupby('item_idx')['Book-Rating'].mean() - self.global_mean
        
        # SVD decomposition
        self.svd = TruncatedSVD(n_components=self.n_factors, random_state=42)
        self.user_factors = self.svd.fit_transform(user_item_matrix)
        self.item_factors = self.svd.components_.T
        
        print(f"  ✓ User factors shape: {self.user_factors.shape}")
        print(f"  ✓ Item factors shape: {self.item_factors.shape}")
        
        return self
    
    def predict(self, test):
        predictions = []
        
        for _, row in test.iterrows():
            try:
                user_idx = self.user_encoder.transform([row['User-ID']])[0]
                item_idx = self.item_encoder.transform([row['ISBN']])[0]
                
                # Prediction = global_mean + user_bias + item_bias + dot_product
                pred = self.global_mean
                pred += self.user_bias.get(user_idx, 0)
                pred += self.item_bias.get(item_idx, 0)
                pred += np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
                
                # Clip to valid rating range
                pred = np.clip(pred, 1, 10)
                predictions.append(pred)
            except:
                # Unknown user or item - use global mean
                predictions.append(self.global_mean)
        
        return np.array(predictions)
    
    def get_name(self):
        return f"SVD (k={self.n_factors})"

# ============================================================================
# SECTION 7: SIMPLE NEURAL COLLABORATIVE FILTERING
# ============================================================================

class NeuralCollaborativeFiltering:
    """Simple neural network for collaborative filtering."""
    
    def __init__(self, n_factors=50, learning_rate=0.01, epochs=20):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.user_embeddings = None
        self.item_embeddings = None
        self.global_mean = None
    
    def fit(self, train):
        print(f"\n  Training Neural CF with {self.n_factors} factors, {self.epochs} epochs...")
        
        # Encode
        train = train.copy()
        train['user_idx'] = self.user_encoder.fit_transform(train['User-ID'])
        train['item_idx'] = self.item_encoder.fit_transform(train['ISBN'])
        
        n_users = train['user_idx'].nunique()
        n_items = train['item_idx'].nunique()
        
        # Initialize embeddings
        self.user_embeddings = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_embeddings = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.global_mean = train['Book-Rating'].mean()
        
        # Training loop (simple SGD)
        for epoch in range(self.epochs):
            total_loss = 0
            for _, row in train.iterrows():
                u = row['user_idx']
                i = row['item_idx']
                r = row['Book-Rating']
                
                # Prediction
                pred = self.global_mean + np.dot(self.user_embeddings[u], self.item_embeddings[i])
                error = r - pred
                total_loss += error ** 2
                
                # Gradient update
                self.user_embeddings[u] += self.learning_rate * error * self.item_embeddings[i]
                self.item_embeddings[i] += self.learning_rate * error * self.user_embeddings[u]
            
            if (epoch + 1) % 5 == 0:
                rmse = np.sqrt(total_loss / len(train))
                print(f"  Epoch {epoch+1}/{self.epochs}, RMSE: {rmse:.4f}")
        
        return self
    
    def predict(self, test):
        predictions = []
        
        for _, row in test.iterrows():
            try:
                user_idx = self.user_encoder.transform([row['User-ID']])[0]
                item_idx = self.item_encoder.transform([row['ISBN']])[0]
                
                pred = self.global_mean + np.dot(
                    self.user_embeddings[user_idx],
                    self.item_embeddings[item_idx]
                )
                pred = np.clip(pred, 1, 10)
                predictions.append(pred)
            except:
                predictions.append(self.global_mean)
        
        return np.array(predictions)
    
    def get_name(self):
        return f"Neural CF (k={self.n_factors})"

# ============================================================================
# SECTION 8: EVALUATION METRICS
# ============================================================================

def evaluate_model(model, train, test):
    """Evaluate model with multiple metrics."""
    print(f"\nEvaluating: {model.get_name()}")
    print("-" * 60)
    
    # Train
    model.fit(train)
    
    # Predict
    predictions = model.predict(test)
    actuals = test['Book-Rating'].values
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    
    # Precision@K and Recall@K (consider rating >= 7 as relevant)
    threshold = 7
    k = 10
    
    precision_at_k = calculate_precision_at_k(actuals, predictions, threshold, k)
    recall_at_k = calculate_recall_at_k(actuals, predictions, threshold, k)
    
    # Print results
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"Precision@{k}: {precision_at_k:.4f}")
    print(f"Recall@{k}:    {recall_at_k:.4f}")
    
    return {
        'model': model.get_name(),
        'rmse': rmse,
        'mae': mae,
        'precision_at_k': precision_at_k,
        'recall_at_k': recall_at_k,
        'predictions': predictions
    }

def calculate_precision_at_k(actuals, predictions, threshold, k):
    """Calculate Precision@K."""
    relevant_actual = actuals >= threshold
    top_k_pred = np.argsort(predictions)[-k:]
    relevant_predicted = np.isin(np.arange(len(actuals)), top_k_pred)
    
    if relevant_predicted.sum() == 0:
        return 0.0
    
    precision = (relevant_actual & relevant_predicted).sum() / relevant_predicted.sum()
    return precision

def calculate_recall_at_k(actuals, predictions, threshold, k):
    """Calculate Recall@K."""
    relevant_actual = actuals >= threshold
    top_k_pred = np.argsort(predictions)[-k:]
    relevant_predicted = np.isin(np.arange(len(actuals)), top_k_pred)
    
    if relevant_actual.sum() == 0:
        return 0.0
    
    recall = (relevant_actual & relevant_predicted).sum() / relevant_actual.sum()
    return recall

# ============================================================================
# SECTION 9: MODEL COMPARISON
# ============================================================================

def compare_models(train, test):
    """Train and compare multiple models."""
    print("\n" + "=" * 80)
    print("SECTION 4-7: MODEL TRAINING AND EVALUATION")
    print("=" * 80)
    
    models = [
        GlobalAverageModel(),
        UserItemAverageModel(),
        MatrixFactorizationSVD(n_factors=20),
        MatrixFactorizationSVD(n_factors=50),
        MatrixFactorizationSVD(n_factors=100),
        NeuralCollaborativeFiltering(n_factors=30, epochs=15)
    ]
    
    results = []
    for model in models:
        result = evaluate_model(model, train, test)
        results.append(result)
    
    return pd.DataFrame(results)

# ============================================================================
# SECTION 10: VISUALIZATION
# ============================================================================

def visualize_results(results_df):
    """Visualize model comparison."""
    print("\n" + "=" * 80)
    print("SECTION 8: MODEL COMPARISON VISUALIZATION")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # RMSE comparison
    axes[0, 0].bar(range(len(results_df)), results_df['rmse'])
    axes[0, 0].set_xticks(range(len(results_df)))
    axes[0, 0].set_xticklabels(results_df['model'], rotation=45, ha='right')
    axes[0, 0].set_ylabel('RMSE')
    axes[0, 0].set_title('Root Mean Squared Error (Lower is Better)')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # MAE comparison
    axes[0, 1].bar(range(len(results_df)), results_df['mae'], color='orange')
    axes[0, 1].set_xticks(range(len(results_df)))
    axes[0, 1].set_xticklabels(results_df['model'], rotation=45, ha='right')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title('Mean Absolute Error (Lower is Better)')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Precision@K
    axes[1, 0].bar(range(len(results_df)), results_df['precision_at_k'], color='green')
    axes[1, 0].set_xticks(range(len(results_df)))
    axes[1, 0].set_xticklabels(results_df['model'], rotation=45, ha='right')
    axes[1, 0].set_ylabel('Precision@10')
    axes[1, 0].set_title('Precision@10 (Higher is Better)')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Recall@K
    axes[1, 1].bar(range(len(results_df)), results_df['recall_at_k'], color='red')
    axes[1, 1].set_xticks(range(len(results_df)))
    axes[1, 1].set_xticklabels(results_df['model'], rotation=45, ha='right')
    axes[1, 1].set_ylabel('Recall@10')
    axes[1, 1].set_title('Recall@10 (Higher is Better)')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved as 'model_comparison.png'")
    plt.show()

# ============================================================================
# SECTION 11: FINAL RESULTS
# ============================================================================

def print_final_results(results_df):
    """Print comprehensive results."""
    print("\n" + "=" * 80)
    print("SECTION 9: FINAL RESULTS AND BEST MODEL")
    print("=" * 80)
    
    print("\n📊 COMPLETE RESULTS TABLE:")
    print(results_df[['model', 'rmse', 'mae', 'precision_at_k', 'recall_at_k']].to_string(index=False))
    
    # Best model
    best_idx = results_df['rmse'].idxmin()
    best_model = results_df.iloc[best_idx]
    
    print("\n" + "=" * 80)
    print("🏆 BEST MODEL")
    print("=" * 80)
    print(f"Model: {best_model['model']}")
    print(f"RMSE: {best_model['rmse']:.4f}")
    print(f"MAE: {best_model['mae']:.4f}")
    print(f"Precision@10: {best_model['precision_at_k']:.4f}")
    print(f"Recall@10: {best_model['recall_at_k']:.4f}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline."""
    
    # Load data
    books, ratings = load_data()
    
    # Preprocess
    ratings_clean = preprocess_data(books, ratings)
    
    # Split
    train, test = create_train_test_split(ratings_clean)
    
    # Compare models
    results = compare_models(train, test)
    
    # Visualize
    visualize_results(results)
    
    # Final results
    print_final_results(results)
    
    print("\n" + "=" * 80)
    print("✓ MACHINE LEARNING PIPELINE COMPLETE")
    print("=" * 80)
    print("\nKey Components:")
    print("✓ Train/Test Split (80/20)")
    print("✓ Multiple ML Models (Baseline, SVD, Neural CF)")
    print("✓ Proper Evaluation Metrics (RMSE, MAE, Precision, Recall)")
    print("✓ Model Comparison and Visualization")
    print("✓ Best Model Selection")
    print("\nThis is a REAL machine learning system with:")
    print("  • Proper data splitting")
    print("  • Model training on train set")
    print("  • Evaluation on test set")
    print("  • Hyperparameter tuning (different k values)")
    print("  • Performance metrics")

if __name__ == "__main__":
    main()