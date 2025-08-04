#!/usr/bin/env python3
"""
Debug script to test the recommendation system
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.tmdb_processor import TMDBDataProcessor
from src.models.recommendation_models import (
    CollaborativeFiltering, 
    ContentBasedFiltering, 
    MatrixFactorization, 
    HybridRecommender
)

def test_recommendations():
    """Test the recommendation system."""
    
    print("🔍 Testing Recommendation System...")
    
    # Load data
    tmdb_processor = TMDBDataProcessor()
    
    # Load processed data
    movies_df = pd.read_csv("data/processed/tmdb_movies_processed.csv")
    ratings_df = pd.read_csv("data/processed/tmdb_ratings_simulated.csv")
    
    print(f"📊 Loaded {len(movies_df)} movies and {len(ratings_df)} ratings")
    
    # Create user-movie matrix
    user_movie_matrix = ratings_df.pivot_table(
        index='userId',
        columns='movieId',
        values='rating',
        fill_value=0
    )
    
    print(f"📈 User-movie matrix shape: {user_movie_matrix.shape}")
    print(f"👥 Users: {user_movie_matrix.index.min()} to {user_movie_matrix.index.max()}")
    print(f"🎬 Movies: {user_movie_matrix.columns.min()} to {user_movie_matrix.columns.max()}")
    
    # Test with user 19
    user_id = 19
    print(f"\n🧪 Testing with User {user_id}")
    
    if user_id not in user_movie_matrix.index:
        print(f"❌ User {user_id} not found in user-movie matrix")
        print(f"Available users: {list(user_movie_matrix.index[:10])}")
        return
    
    # Check user's ratings
    user_ratings = user_movie_matrix.loc[user_id]
    rated_movies = user_ratings[user_ratings > 0]
    print(f"📝 User {user_id} has rated {len(rated_movies)} movies")
    print(f"📊 Average rating: {rated_movies.mean():.2f}")
    
    # Train a simple model
    print("\n🤖 Training User-based CF model...")
    cf_user = CollaborativeFiltering(method='user')
    cf_user.fit(user_movie_matrix)
    
    # Test recommendation
    print(f"🎯 Getting recommendations for User {user_id}...")
    try:
        movie_ids = cf_user.recommend(user_id, 5)
        print(f"✅ Got {len(movie_ids)} recommendations: {movie_ids}")
        
        # Get movie details
        recommendations = []
        for movie_id in movie_ids:
            movie_info = movies_df[movies_df['id'] == movie_id]
            if not movie_info.empty:
                movie = movie_info.iloc[0]
                title = movie['title_x'] if pd.notna(movie['title_x']) else movie['title_y']
                recommendations.append({
                    'id': movie_id,
                    'title': title,
                    'vote_average': movie['vote_average']
                })
        
        print(f"🎬 Movie recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec['title']} (ID: {rec['id']}, Rating: {rec['vote_average']})")
            
    except Exception as e:
        print(f"❌ Error getting recommendations: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_recommendations() 