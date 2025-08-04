import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

class RecommendationEvaluator:
    """
    Evaluation utilities for recommendation systems.
    """
    
    def __init__(self):
        self.metrics = {}
    
    def calculate_rmse(self, y_true, y_pred):
        """
        Calculate Root Mean Square Error.
        
        Args:
            y_true: True ratings
            y_pred: Predicted ratings
            
        Returns:
            float: RMSE value
        """
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    def calculate_mae(self, y_true, y_pred):
        """
        Calculate Mean Absolute Error.
        
        Args:
            y_true: True ratings
            y_pred: Predicted ratings
            
        Returns:
            float: MAE value
        """
        return mean_absolute_error(y_true, y_pred)
    
    def calculate_precision_at_k(self, recommended_items, relevant_items, k=10):
        """
        Calculate Precision@K.
        
        Args:
            recommended_items: List of recommended items
            relevant_items: List of relevant items
            k (int): Number of top recommendations to consider
            
        Returns:
            float: Precision@K value
        """
        if len(recommended_items) == 0:
            return 0.0
        
        top_k_recommendations = recommended_items[:k]
        relevant_recommendations = set(top_k_recommendations) & set(relevant_items)
        
        return len(relevant_recommendations) / len(top_k_recommendations)
    
    def calculate_recall_at_k(self, recommended_items, relevant_items, k=10):
        """
        Calculate Recall@K.
        
        Args:
            recommended_items: List of recommended items
            relevant_items: List of relevant items
            k (int): Number of top recommendations to consider
            
        Returns:
            float: Recall@K value
        """
        if len(relevant_items) == 0:
            return 0.0
        
        top_k_recommendations = recommended_items[:k]
        relevant_recommendations = set(top_k_recommendations) & set(relevant_items)
        
        return len(relevant_recommendations) / len(relevant_items)
    
    def calculate_f1_at_k(self, recommended_items, relevant_items, k=10):
        """
        Calculate F1@K.
        
        Args:
            recommended_items: List of recommended items
            relevant_items: List of relevant items
            k (int): Number of top recommendations to consider
            
        Returns:
            float: F1@K value
        """
        precision = self.calculate_precision_at_k(recommended_items, relevant_items, k)
        recall = self.calculate_recall_at_k(recommended_items, relevant_items, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def calculate_ndcg_at_k(self, recommended_items, relevant_items, relevance_scores, k=10):
        """
        Calculate Normalized Discounted Cumulative Gain@K.
        
        Args:
            recommended_items: List of recommended items
            relevant_items: List of relevant items
            relevance_scores: Dictionary mapping items to relevance scores
            k (int): Number of top recommendations to consider
            
        Returns:
            float: NDCG@K value
        """
        if len(recommended_items) == 0:
            return 0.0
        
        top_k_recommendations = recommended_items[:k]
        
        # Calculate DCG
        dcg = 0
        for i, item in enumerate(top_k_recommendations):
            if item in relevance_scores:
                dcg += relevance_scores[item] / np.log2(i + 2)
        
        # Calculate IDCG (ideal DCG)
        ideal_relevance = sorted([relevance_scores.get(item, 0) for item in relevant_items], reverse=True)
        idcg = 0
        for i, relevance in enumerate(ideal_relevance[:k]):
            idcg += relevance / np.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def evaluate_rating_predictions(self, model, test_data, user_col='userId', item_col='movieId', rating_col='rating'):
        """
        Evaluate rating prediction performance.
        
        Args:
            model: Trained recommendation model
            test_data: Test dataset
            user_col: Column name for user IDs
            item_col: Column name for item IDs
            rating_col: Column name for ratings
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        y_true = []
        y_pred = []
        
        for _, row in test_data.iterrows():
            user_id = row[user_col]
            item_id = row[item_col]
            true_rating = row[rating_col]
            
            try:
                pred_rating = model.predict(user_id, item_id)
                y_true.append(true_rating)
                y_pred.append(pred_rating)
            except:
                continue
        
        if len(y_true) == 0:
            return {'rmse': float('inf'), 'mae': float('inf')}
        
        rmse = self.calculate_rmse(y_true, y_pred)
        mae = self.calculate_mae(y_true, y_pred)
        
        return {
            'rmse': rmse,
            'mae': mae,
            'n_predictions': len(y_true)
        }
    
    def evaluate_recommendations(self, model, test_users, test_data, k=10, 
                               user_col='userId', item_col='movieId', rating_col='rating'):
        """
        Evaluate recommendation quality.
        
        Args:
            model: Trained recommendation model
            test_users: List of test user IDs
            test_data: Test dataset
            k (int): Number of recommendations to evaluate
            user_col: Column name for user IDs
            item_col: Column name for item IDs
            rating_col: Column name for ratings
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        precisions = []
        recalls = []
        f1_scores = []
        
        for user_id in test_users:
            try:
                # Get recommendations
                recommended_items = model.recommend(user_id, k)
                
                # Get relevant items (items rated >= 4 in test set)
                user_test_data = test_data[test_data[user_col] == user_id]
                relevant_items = user_test_data[user_test_data[rating_col] >= 4][item_col].tolist()
                
                # Calculate metrics
                precision = self.calculate_precision_at_k(recommended_items, relevant_items, k)
                recall = self.calculate_recall_at_k(recommended_items, relevant_items, k)
                f1 = self.calculate_f1_at_k(recommended_items, relevant_items, k)
                
                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1)
                
            except:
                continue
        
        if len(precisions) == 0:
            return {
                'precision_at_k': 0.0,
                'recall_at_k': 0.0,
                'f1_at_k': 0.0,
                'n_users': 0
            }
        
        return {
            'precision_at_k': np.mean(precisions),
            'recall_at_k': np.mean(recalls),
            'f1_at_k': np.mean(f1_scores),
            'n_users': len(precisions)
        }
    
    def plot_evaluation_results(self, results_dict, title="Evaluation Results"):
        """
        Plot evaluation results.
        
        Args:
            results_dict: Dictionary containing evaluation metrics
            title: Plot title
        """
        metrics = list(results_dict.keys())
        values = list(results_dict.values())
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics, values, color='skyblue', alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.title(title)
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def compare_models(self, models_dict, test_data, test_users, k=10):
        """
        Compare multiple recommendation models.
        
        Args:
            models_dict: Dictionary mapping model names to model objects
            test_data: Test dataset
            test_users: List of test user IDs
            k (int): Number of recommendations to evaluate
            
        Returns:
            pd.DataFrame: Comparison results
        """
        results = []
        
        for model_name, model in models_dict.items():
            print(f"Evaluating {model_name}...")
            
            # Evaluate recommendations
            rec_metrics = self.evaluate_recommendations(model, test_users, test_data, k)
            
            # Evaluate rating predictions if possible
            try:
                rating_metrics = self.evaluate_rating_predictions(model, test_data)
                metrics = {**rec_metrics, **rating_metrics}
            except:
                metrics = rec_metrics
            
            metrics['model'] = model_name
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def create_train_test_split(self, ratings_df, test_size=0.2, random_state=42):
        """
        Create train-test split for recommendation evaluation.
        
        Args:
            ratings_df: Ratings dataframe
            test_size: Proportion of data to use for testing
            random_state: Random seed
            
        Returns:
            tuple: (train_data, test_data)
        """
        # Sort by timestamp to ensure temporal split
        ratings_df = ratings_df.sort_values('timestamp')
        
        # Calculate split point
        split_idx = int(len(ratings_df) * (1 - test_size))
        
        train_data = ratings_df.iloc[:split_idx]
        test_data = ratings_df.iloc[split_idx:]
        
        return train_data, test_data 