import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, TruncatedSVD
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')

class CollaborativeFiltering:
    """
    Collaborative Filtering recommendation system using user-based and item-based approaches.
    """
    
    def __init__(self, method='user'):
        """
        Initialize collaborative filtering model.
        
        Args:
            method (str): 'user' for user-based CF, 'item' for item-based CF
        """
        self.method = method
        self.user_movie_matrix = None
        self.similarity_matrix = None
        self.user_means = None
        self.item_means = None
    
    def fit(self, user_movie_matrix):
        """
        Fit the collaborative filtering model.
        
        Args:
            user_movie_matrix (pd.DataFrame): User-movie rating matrix
        """
        self.user_movie_matrix = user_movie_matrix
        
        if self.method == 'user':
            # User-based CF
            self.user_means = user_movie_matrix.mean(axis=1)
            # Center the ratings
            centered_matrix = user_movie_matrix.sub(self.user_means, axis=0)
            # Fill NaN with 0
            centered_matrix = centered_matrix.fillna(0)
            # Calculate user similarity
            self.similarity_matrix = cosine_similarity(centered_matrix)
            
        elif self.method == 'item':
            # Item-based CF
            self.item_means = user_movie_matrix.mean(axis=0)
            # Center the ratings
            centered_matrix = user_movie_matrix.sub(self.item_means, axis=1)
            # Fill NaN with 0
            centered_matrix = centered_matrix.fillna(0)
            # Calculate item similarity
            self.similarity_matrix = cosine_similarity(centered_matrix.T)
    
    def predict(self, user_id, movie_id, k=10):
        """
        Predict rating for a user-movie pair.
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            k (int): Number of similar users/items to consider
            
        Returns:
            float: Predicted rating
        """
        if self.method == 'user':
            return self._predict_user_based(user_id, movie_id, k)
        else:
            return self._predict_item_based(user_id, movie_id, k)
    
    def _predict_user_based(self, user_id, movie_id, k):
        """User-based prediction."""
        if user_id not in self.user_movie_matrix.index:
            return self.user_movie_matrix[movie_id].mean()
        
        user_idx = self.user_movie_matrix.index.get_loc(user_id)
        similar_users = np.argsort(self.similarity_matrix[user_idx])[::-1][1:k+1]
        
        numerator = 0
        denominator = 0
        
        for similar_user_idx in similar_users:
            similar_user_id = self.user_movie_matrix.index[similar_user_idx]
            similarity = self.similarity_matrix[user_idx, similar_user_idx]
            
            if movie_id in self.user_movie_matrix.columns:
                rating = self.user_movie_matrix.loc[similar_user_id, movie_id]
                if not pd.isna(rating):
                    numerator += similarity * (rating - self.user_means[similar_user_id])
                    denominator += abs(similarity)
        
        if denominator == 0:
            return self.user_means[user_id]
        
        predicted_rating = self.user_means[user_id] + (numerator / denominator)
        return max(0.5, min(5.0, predicted_rating))
    
    def _predict_item_based(self, user_id, movie_id, k):
        """Item-based prediction."""
        if movie_id not in self.user_movie_matrix.columns:
            return self.user_movie_matrix.loc[user_id].mean()
        
        movie_idx = self.user_movie_matrix.columns.get_loc(movie_id)
        similar_items = np.argsort(self.similarity_matrix[movie_idx])[::-1][1:k+1]
        
        numerator = 0
        denominator = 0
        
        for similar_item_idx in similar_items:
            similar_movie_id = self.user_movie_matrix.columns[similar_item_idx]
            similarity = self.similarity_matrix[movie_idx, similar_item_idx]
            
            if user_id in self.user_movie_matrix.index:
                rating = self.user_movie_matrix.loc[user_id, similar_movie_id]
                if not pd.isna(rating):
                    numerator += similarity * (rating - self.item_means[similar_movie_id])
                    denominator += abs(similarity)
        
        if denominator == 0:
            return self.item_means[movie_id]
        
        predicted_rating = self.item_means[movie_id] + (numerator / denominator)
        return max(0.5, min(5.0, predicted_rating))
    
    def recommend(self, user_id, n_recommendations=10, k=10):
        """
        Generate movie recommendations for a user.
        
        Args:
            user_id: User ID
            n_recommendations (int): Number of recommendations to generate
            k (int): Number of similar users/items to consider
            
        Returns:
            list: List of recommended movie IDs
        """
        if user_id not in self.user_movie_matrix.index:
            return []
        
        # Get movies the user hasn't rated (where rating is 0)
        user_ratings = self.user_movie_matrix.loc[user_id]
        unwatched_movies = user_ratings[user_ratings == 0].index
        
        # Predict ratings for unwatched movies
        predictions = []
        for movie_id in unwatched_movies:
            pred_rating = self.predict(user_id, movie_id, k)
            predictions.append((movie_id, pred_rating))
        
        # Sort by predicted rating and return top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [movie_id for movie_id, _ in predictions[:n_recommendations]]

class ContentBasedFiltering:
    """
    Content-based filtering using movie features like genres, title, etc.
    """
    
    def __init__(self, feature_columns=None):
        """
        Initialize content-based filtering model.
        
        Args:
            feature_columns (list): List of feature columns to use
        """
        self.feature_columns = feature_columns or ['genres']
        self.movies_df = None
        self.feature_matrix = None
        self.similarity_matrix = None
        self.tfidf_vectorizer = None
    
    def fit(self, movies_df):
        """
        Fit the content-based filtering model.
        
        Args:
            movies_df (pd.DataFrame): Movies dataframe with features
        """
        self.movies_df = movies_df.copy()
        
        # Create feature matrix
        feature_texts = []
        for _, movie in movies_df.iterrows():
            features = []
            for col in self.feature_columns:
                if col in movie and pd.notna(movie[col]):
                    features.append(str(movie[col]))
            feature_texts.append(' '.join(features))
        
        # Create TF-IDF matrix
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.feature_matrix = self.tfidf_vectorizer.fit_transform(feature_texts)
        
        # Calculate similarity matrix
        self.similarity_matrix = cosine_similarity(self.feature_matrix)
    
    def recommend(self, movie_id, n_recommendations=10):
        """
        Find similar movies based on content.
        
        Args:
            movie_id: Movie ID
            n_recommendations (int): Number of recommendations to generate
            
        Returns:
            list: List of similar movie IDs
        """
        if movie_id not in self.movies_df.index:
            return []
        
        movie_idx = self.movies_df.index.get_loc(movie_id)
        similar_scores = list(enumerate(self.similarity_matrix[movie_idx]))
        similar_scores = sorted(similar_scores, key=lambda x: x[1], reverse=True)
        similar_scores = similar_scores[1:n_recommendations+1]
        
        similar_movie_indices = [i[0] for i in similar_scores]
        return self.movies_df.iloc[similar_movie_indices].index.tolist()

class MatrixFactorization:
    """
    Matrix Factorization using SVD and NMF for recommendation systems.
    """
    
    def __init__(self, method='svd', n_components=50):
        """
        Initialize matrix factorization model.
        
        Args:
            method (str): 'svd' for SVD, 'nmf' for Non-negative Matrix Factorization
            n_components (int): Number of latent factors
        """
        self.method = method
        self.n_components = n_components
        self.model = None
        self.user_movie_matrix = None
        self.user_factors = None
        self.item_factors = None
    
    def fit(self, user_movie_matrix):
        """
        Fit the matrix factorization model.
        
        Args:
            user_movie_matrix (pd.DataFrame): User-movie rating matrix
        """
        self.user_movie_matrix = user_movie_matrix
        
        # Fill missing values with 0 for matrix factorization
        matrix = user_movie_matrix.fillna(0).values
        
        if self.method == 'svd':
            self.model = TruncatedSVD(n_components=self.n_components, random_state=42)
            self.user_factors = self.model.fit_transform(matrix)
            self.item_factors = self.model.components_.T
            
        elif self.method == 'nmf':
            self.model = NMF(n_components=self.n_components, random_state=42)
            self.user_factors = self.model.fit_transform(matrix)
            self.item_factors = self.model.components_.T
    
    def predict(self, user_id, movie_id):
        """
        Predict rating for a user-movie pair.
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            
        Returns:
            float: Predicted rating
        """
        if user_id not in self.user_movie_matrix.index or movie_id not in self.user_movie_matrix.columns:
            return 0
        
        user_idx = self.user_movie_matrix.index.get_loc(user_id)
        movie_idx = self.user_movie_matrix.columns.get_loc(movie_id)
        
        prediction = np.dot(self.user_factors[user_idx], self.item_factors[movie_idx])
        return max(0.5, min(5.0, prediction))
    
    def recommend(self, user_id, n_recommendations=10):
        """
        Generate movie recommendations for a user.
        
        Args:
            user_id: User ID
            n_recommendations (int): Number of recommendations to generate
            
        Returns:
            list: List of recommended movie IDs
        """
        if user_id not in self.user_movie_matrix.index:
            return []
        
        user_idx = self.user_movie_matrix.index.get_loc(user_id)
        user_vector = self.user_factors[user_idx]
        
        # Calculate predicted ratings for all movies
        predictions = np.dot(user_vector, self.item_factors.T)
        
        # Get movies the user hasn't rated
        user_ratings = self.user_movie_matrix.loc[user_id]
        unwatched_mask = user_ratings.isna()
        unwatched_indices = np.where(unwatched_mask)[0]
        
        if len(unwatched_indices) == 0:
            return []
        
        # Get predictions for unwatched movies
        unwatched_predictions = predictions[unwatched_indices]
        
        # Get top N recommendations
        top_indices = np.argsort(unwatched_predictions)[::-1][:n_recommendations]
        recommended_movie_indices = unwatched_indices[top_indices]
        
        return self.user_movie_matrix.columns[recommended_movie_indices].tolist()

class HybridRecommender:
    """
    Hybrid recommendation system combining multiple approaches.
    """
    
    def __init__(self, cf_weight=0.6, cb_weight=0.2, mf_weight=0.2):
        """
        Initialize hybrid recommender.
        
        Args:
            cf_weight (float): Weight for collaborative filtering
            cb_weight (float): Weight for content-based filtering
            mf_weight (float): Weight for matrix factorization
        """
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight
        self.mf_weight = mf_weight
        
        self.cf_model = CollaborativeFiltering(method='user')
        self.cb_model = ContentBasedFiltering()
        self.mf_model = MatrixFactorization()
        
        self.is_fitted = False
    
    def fit(self, user_movie_matrix, movies_df):
        """
        Fit all models.
        
        Args:
            user_movie_matrix (pd.DataFrame): User-movie rating matrix
            movies_df (pd.DataFrame): Movies dataframe
        """
        print("Fitting Collaborative Filtering model...")
        self.cf_model.fit(user_movie_matrix)
        
        print("Fitting Content-Based model...")
        self.cb_model.fit(movies_df)
        
        print("Fitting Matrix Factorization model...")
        self.mf_model.fit(user_movie_matrix)
        
        self.is_fitted = True
        print("All models fitted successfully!")
    
    def recommend(self, user_id, n_recommendations=10):
        """
        Generate hybrid recommendations.
        
        Args:
            user_id: User ID
            n_recommendations (int): Number of recommendations to generate
            
        Returns:
            list: List of recommended movie IDs
        """
        if not self.is_fitted:
            raise ValueError("Models must be fitted before making recommendations")
        
        # Get recommendations from each model
        cf_recs = self.cf_model.recommend(user_id, n_recommendations)
        mf_recs = self.mf_model.recommend(user_id, n_recommendations)
        
        # For content-based, we need a movie the user has watched
        user_ratings = self.cf_model.user_movie_matrix.loc[user_id]
        watched_movies = user_ratings[user_ratings.notna()].index
        
        if len(watched_movies) > 0:
            # Use the highest rated movie for content-based recommendations
            best_movie = user_ratings.idxmax()
            cb_recs = self.cb_model.recommend(best_movie, n_recommendations)
        else:
            cb_recs = []
        
        # Combine recommendations with weights
        all_recs = {}
        
        # Add CF recommendations
        for i, movie_id in enumerate(cf_recs):
            all_recs[movie_id] = all_recs.get(movie_id, 0) + self.cf_weight * (n_recommendations - i)
        
        # Add CB recommendations
        for i, movie_id in enumerate(cb_recs):
            all_recs[movie_id] = all_recs.get(movie_id, 0) + self.cb_weight * (n_recommendations - i)
        
        # Add MF recommendations
        for i, movie_id in enumerate(mf_recs):
            all_recs[movie_id] = all_recs.get(movie_id, 0) + self.mf_weight * (n_recommendations - i)
        
        # Sort by score and return top N
        sorted_recs = sorted(all_recs.items(), key=lambda x: x[1], reverse=True)
        return [movie_id for movie_id, _ in sorted_recs[:n_recommendations]] 