import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models.recommendation_models import (
    CollaborativeFiltering, 
    ContentBasedFiltering, 
    MatrixFactorization, 
    HybridRecommender
)

class TestCollaborativeFiltering:
    """Test cases for Collaborative Filtering models."""
    
    def setup_method(self):
        """Set up test data."""
        # Create sample user-movie matrix
        np.random.seed(42)
        self.user_movie_matrix = pd.DataFrame(
            np.random.randint(1, 6, (10, 10)),
            index=range(1, 11),
            columns=range(1, 11)
        )
        # Add some NaN values to simulate missing ratings
        mask = np.random.random(self.user_movie_matrix.shape) < 0.3
        self.user_movie_matrix[mask] = np.nan
    
    def test_user_based_cf_initialization(self):
        """Test user-based CF initialization."""
        cf = CollaborativeFiltering(method='user')
        assert cf.method == 'user'
        assert cf.user_movie_matrix is None
        assert cf.similarity_matrix is None
    
    def test_item_based_cf_initialization(self):
        """Test item-based CF initialization."""
        cf = CollaborativeFiltering(method='item')
        assert cf.method == 'item'
        assert cf.user_movie_matrix is None
        assert cf.similarity_matrix is None
    
    def test_user_based_cf_fit(self):
        """Test user-based CF fitting."""
        cf = CollaborativeFiltering(method='user')
        cf.fit(self.user_movie_matrix)
        
        assert cf.user_movie_matrix is not None
        assert cf.similarity_matrix is not None
        assert cf.user_means is not None
        assert cf.similarity_matrix.shape == (10, 10)
    
    def test_item_based_cf_fit(self):
        """Test item-based CF fitting."""
        cf = CollaborativeFiltering(method='item')
        cf.fit(self.user_movie_matrix)
        
        assert cf.user_movie_matrix is not None
        assert cf.similarity_matrix is not None
        assert cf.item_means is not None
        assert cf.similarity_matrix.shape == (10, 10)
    
    def test_user_based_prediction(self):
        """Test user-based CF prediction."""
        cf = CollaborativeFiltering(method='user')
        cf.fit(self.user_movie_matrix)
        
        prediction = cf.predict(1, 1, k=5)
        assert isinstance(prediction, float)
        assert 0.5 <= prediction <= 5.0
    
    def test_item_based_prediction(self):
        """Test item-based CF prediction."""
        cf = CollaborativeFiltering(method='item')
        cf.fit(self.user_movie_matrix)
        
        prediction = cf.predict(1, 1, k=5)
        assert isinstance(prediction, float)
        assert 0.5 <= prediction <= 5.0
    
    def test_recommendations(self):
        """Test recommendation generation."""
        cf = CollaborativeFiltering(method='user')
        cf.fit(self.user_movie_matrix)
        
        recommendations = cf.recommend(1, n_recommendations=5)
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 5

class TestContentBasedFiltering:
    """Test cases for Content-based Filtering model."""
    
    def setup_method(self):
        """Set up test data."""
        self.movies_df = pd.DataFrame({
            'movieId': [1, 2, 3, 4, 5],
            'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E'],
            'genres': ['Action|Adventure', 'Comedy|Romance', 'Action|Thriller', 
                      'Comedy|Drama', 'Adventure|Fantasy']
        })
        self.movies_df.set_index('movieId', inplace=True)
    
    def test_initialization(self):
        """Test CB initialization."""
        cb = ContentBasedFiltering()
        assert cb.feature_columns == ['genres']
        assert cb.movies_df is None
        assert cb.feature_matrix is None
    
    def test_custom_feature_columns(self):
        """Test CB with custom feature columns."""
        cb = ContentBasedFiltering(feature_columns=['title', 'genres'])
        assert cb.feature_columns == ['title', 'genres']
    
    def test_fit(self):
        """Test CB fitting."""
        cb = ContentBasedFiltering()
        cb.fit(self.movies_df)
        
        assert cb.movies_df is not None
        assert cb.feature_matrix is not None
        assert cb.similarity_matrix is not None
        assert cb.tfidf_vectorizer is not None
    
    def test_recommendations(self):
        """Test CB recommendations."""
        cb = ContentBasedFiltering()
        cb.fit(self.movies_df)
        
        recommendations = cb.recommend(1, n_recommendations=3)
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 3
        assert all(isinstance(movie_id, int) for movie_id in recommendations)

class TestMatrixFactorization:
    """Test cases for Matrix Factorization models."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.user_movie_matrix = pd.DataFrame(
            np.random.randint(1, 6, (10, 10)),
            index=range(1, 11),
            columns=range(1, 11)
        )
        # Add some NaN values
        mask = np.random.random(self.user_movie_matrix.shape) < 0.3
        self.user_movie_matrix[mask] = np.nan
    
    def test_svd_initialization(self):
        """Test SVD initialization."""
        mf = MatrixFactorization(method='svd', n_components=5)
        assert mf.method == 'svd'
        assert mf.n_components == 5
        assert mf.model is None
    
    def test_nmf_initialization(self):
        """Test NMF initialization."""
        mf = MatrixFactorization(method='nmf', n_components=5)
        assert mf.method == 'nmf'
        assert mf.n_components == 5
        assert mf.model is None
    
    def test_svd_fit(self):
        """Test SVD fitting."""
        mf = MatrixFactorization(method='svd', n_components=5)
        mf.fit(self.user_movie_matrix)
        
        assert mf.user_movie_matrix is not None
        assert mf.user_factors is not None
        assert mf.item_factors is not None
        assert mf.user_factors.shape == (10, 5)
        assert mf.item_factors.shape == (10, 5)
    
    def test_nmf_fit(self):
        """Test NMF fitting."""
        mf = MatrixFactorization(method='nmf', n_components=5)
        mf.fit(self.user_movie_matrix)
        
        assert mf.user_movie_matrix is not None
        assert mf.user_factors is not None
        assert mf.item_factors is not None
        assert mf.user_factors.shape == (10, 5)
        assert mf.item_factors.shape == (10, 5)
    
    def test_prediction(self):
        """Test MF prediction."""
        mf = MatrixFactorization(method='svd', n_components=5)
        mf.fit(self.user_movie_matrix)
        
        prediction = mf.predict(1, 1)
        assert isinstance(prediction, float)
        assert 0.5 <= prediction <= 5.0
    
    def test_recommendations(self):
        """Test MF recommendations."""
        mf = MatrixFactorization(method='svd', n_components=5)
        mf.fit(self.user_movie_matrix)
        
        recommendations = mf.recommend(1, n_recommendations=5)
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 5

class TestHybridRecommender:
    """Test cases for Hybrid Recommender."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.user_movie_matrix = pd.DataFrame(
            np.random.randint(1, 6, (10, 10)),
            index=range(1, 11),
            columns=range(1, 11)
        )
        mask = np.random.random(self.user_movie_matrix.shape) < 0.3
        self.user_movie_matrix[mask] = np.nan
        
        self.movies_df = pd.DataFrame({
            'movieId': range(1, 11),
            'title': [f'Movie {i}' for i in range(1, 11)],
            'genres': ['Action|Adventure'] * 10
        })
        self.movies_df.set_index('movieId', inplace=True)
    
    def test_initialization(self):
        """Test hybrid recommender initialization."""
        hybrid = HybridRecommender(cf_weight=0.6, cb_weight=0.2, mf_weight=0.2)
        assert hybrid.cf_weight == 0.6
        assert hybrid.cb_weight == 0.2
        assert hybrid.mf_weight == 0.2
        assert not hybrid.is_fitted
    
    def test_fit(self):
        """Test hybrid recommender fitting."""
        hybrid = HybridRecommender()
        hybrid.fit(self.user_movie_matrix, self.movies_df)
        
        assert hybrid.is_fitted
        assert hybrid.cf_model.user_movie_matrix is not None
        assert hybrid.cb_model.movies_df is not None
        assert hybrid.mf_model.user_movie_matrix is not None
    
    def test_recommendations_before_fit(self):
        """Test that recommendations fail before fitting."""
        hybrid = HybridRecommender()
        with pytest.raises(ValueError, match="Models must be fitted"):
            hybrid.recommend(1, n_recommendations=5)
    
    def test_recommendations_after_fit(self):
        """Test hybrid recommendations after fitting."""
        hybrid = HybridRecommender()
        hybrid.fit(self.user_movie_matrix, self.movies_df)
        
        recommendations = hybrid.recommend(1, n_recommendations=5)
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 5

if __name__ == "__main__":
    pytest.main([__file__]) 