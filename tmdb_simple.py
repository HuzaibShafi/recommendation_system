#!/usr/bin/env python3
"""
TMDB Movie Recommendation System - Simplified Version

This script demonstrates a complete movie recommendation system using the TMDB dataset
with core functionality and basic analytics.

Author: Huzaib Shafi
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
from src.utils.evaluation import RecommendationEvaluator

def main():
    """
    Main function to run the simplified TMDB movie recommendation system.
    """
    print("🎬 TMDB Movie Recommendation System - Simplified")
    print("=" * 60)
    
    # Initialize components
    tmdb_processor = TMDBDataProcessor()
    evaluator = RecommendationEvaluator()
    
    # Check if TMDB data exists
    movies_file = tmdb_processor.raw_dir / "tmdb_5000_movies.csv"
    credits_file = tmdb_processor.raw_dir / "tmdb_5000_credits.csv"
    
    if not movies_file.exists() or not credits_file.exists():
        print("❌ TMDB data files not found!")
        print("\nTo get started, you need to:")
        print("1. Download the TMDB dataset from Kaggle:")
        print("   https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata")
        print("2. Extract the following files to data/raw/:")
        print("   - tmdb_5000_movies.csv")
        print("   - tmdb_5000_credits.csv")
        print("3. Run this script again")
        return
    
    # Load TMDB data
    print("\n📊 Loading TMDB data...")
    movies_df = tmdb_processor.load_tmdb_movies()
    credits_df = tmdb_processor.load_tmdb_credits()
    
    # Analyze dataset
    print("\n📈 Analyzing TMDB dataset...")
    analysis = tmdb_processor.analyze_dataset(movies_df, credits_df)
    
    print(f"Dataset Analysis:")
    print(f"  - Total movies: {analysis['total_movies']}")
    print(f"  - Total credits records: {analysis['total_credits']}")
    print(f"  - Average vote: {analysis['avg_vote']:.2f}")
    print(f"  - Average popularity: {analysis['avg_popularity']:.2f}")
    print(f"  - Top genres: {list(analysis['top_genres'].keys())[:5]}")
    
    # Preprocess data
    print("\n🔧 Preprocessing TMDB data...")
    movies_processed, ratings_df = tmdb_processor.preprocess_tmdb_data(movies_df, credits_df)
    
    # Save processed data
    tmdb_processor.save_processed_data(movies_processed, "tmdb_movies_processed.csv")
    tmdb_processor.save_processed_data(ratings_df, "tmdb_ratings_simulated.csv")
    
    # Create user-movie matrix
    print("\n🔢 Creating user-movie matrix...")
    user_movie_matrix = ratings_df.pivot_table(
        index='userId',
        columns='movieId',
        values='rating',
        fill_value=0
    )
    
    # Basic statistics
    print(f"\n📊 TMDB Dataset Statistics:")
    print(f"   - Number of movies: {len(movies_processed)}")
    print(f"   - Number of ratings: {len(ratings_df)}")
    print(f"   - Number of users: {ratings_df['userId'].nunique()}")
    print(f"   - Average rating: {ratings_df['rating'].mean():.2f}")
    
    # Train models
    print("\n🤖 Training recommendation models...")
    models = train_tmdb_models(user_movie_matrix, movies_processed)
    
    # Evaluate models
    print("\n📊 Evaluating models...")
    evaluation_results = evaluate_tmdb_models(models, ratings_df, evaluator)
    
    # Generate recommendations
    print("\n🎯 Generating recommendations...")
    generate_tmdb_recommendations(models, movies_processed, user_movie_matrix, tmdb_processor)
    
    # Show content-based recommendations for popular movies
    print("\n🎬 Content-based recommendations for popular movies:")
    show_content_based_recommendations(movies_processed, tmdb_processor)
    
    print("\n✅ TMDB Movie Recommendation System completed successfully!")

def train_tmdb_models(user_movie_matrix, movies_df):
    """
    Train different recommendation models for TMDB data.
    """
    models = {}
    
    # Collaborative Filtering - User-based
    print("   Training User-based Collaborative Filtering...")
    cf_user = CollaborativeFiltering(method='user')
    cf_user.fit(user_movie_matrix)
    models['User-based CF'] = cf_user
    
    # Collaborative Filtering - Item-based
    print("   Training Item-based Collaborative Filtering...")
    cf_item = CollaborativeFiltering(method='item')
    cf_item.fit(user_movie_matrix)
    models['Item-based CF'] = cf_item
    
    # Content-based Filtering (using feature_text)
    print("   Training Content-based Filtering...")
    # Create a copy of movies_df with feature_text as the main feature
    movies_for_cb = movies_df.copy()
    movies_for_cb['genres'] = movies_for_cb['feature_text']  # Use feature_text for content-based
    cb = ContentBasedFiltering()
    cb.fit(movies_for_cb)
    models['Content-based'] = cb
    
    # Matrix Factorization - SVD
    print("   Training Matrix Factorization (SVD)...")
    mf_svd = MatrixFactorization(method='svd', n_components=20)
    mf_svd.fit(user_movie_matrix)
    models['Matrix Factorization (SVD)'] = mf_svd
    
    # Hybrid Recommender
    print("   Training Hybrid Recommender...")
    hybrid = HybridRecommender()
    hybrid.fit(user_movie_matrix, movies_for_cb)
    models['Hybrid'] = hybrid
    
    return models

def evaluate_tmdb_models(models, ratings_df, evaluator):
    """
    Evaluate the trained models for TMDB data.
    """
    # Create train-test split
    train_data, test_data = evaluator.create_train_test_split(ratings_df, test_size=0.2)
    
    # Get test users
    test_users = test_data['userId'].unique()[:20]  # Limit for faster evaluation
    
    # Evaluate models
    evaluation_results = evaluator.compare_models(models, test_data, test_users, k=10)
    
    # Display results
    print("\n📊 TMDB Model Evaluation Results:")
    print(evaluation_results.to_string(index=False))
    
    return evaluation_results

def generate_tmdb_recommendations(models, movies_df, user_movie_matrix, tmdb_processor):
    """
    Generate recommendations for TMDB data.
    """
    # Get some sample users
    sample_users = user_movie_matrix.index[:3]
    
    print(f"\n🎯 Generating recommendations for {len(sample_users)} users...")
    
    for user_id in sample_users:
        print(f"\n👤 User {user_id}:")
        
        # Get user's rated movies
        user_ratings = user_movie_matrix.loc[user_id]
        rated_movies = user_ratings[user_ratings > 0].index
        
        if len(rated_movies) > 0:
            print(f"   Rated {len(rated_movies)} movies")
            
            # Get recommendations from each model
            for model_name, model in models.items():
                try:
                    if model_name == 'Content-based':
                        # For content-based, use the highest rated movie
                        best_movie = user_ratings.idxmax()
                        recommendations = model.recommend(best_movie, 3)
                    else:
                        recommendations = model.recommend(user_id, 3)
                    
                    # Get movie titles
                    recommended_titles = []
                    for movie_id in recommendations:
                        movie_info = movies_df[movies_df['id'] == movie_id]
                        if not movie_info.empty:
                            title = movie_info.iloc[0]['title_x'] if 'title_x' in movie_info.columns else movie_info.iloc[0]['title']
                            recommended_titles.append(title)
                    
                    print(f"   {model_name}: {', '.join(recommended_titles)}")
                    
                except Exception as e:
                    print(f"   {model_name}: Error - {str(e)}")
        else:
            print("   No rated movies")

def show_content_based_recommendations(movies_df, tmdb_processor):
    """
    Show content-based recommendations for popular movies.
    """
    # Get top 5 movies by vote_average
    top_movies = movies_df.nlargest(5, 'vote_average')
    
    for _, movie in top_movies.iterrows():
        title = movie['title_x'] if 'title_x' in movie.index else movie['title']
        print(f"\n📽️  '{title}' (Rating: {movie['vote_average']:.1f}):")
        
        try:
            # Get content-based recommendations
            recommendations = tmdb_processor.get_movie_recommendations(movies_df, movie['id'], 3)
            
            for _, rec in recommendations.iterrows():
                print(f"   - {rec['title']} (Similarity: {rec['similarity']:.2f})")
                
        except Exception as e:
            print(f"   Error getting recommendations: {str(e)}")

if __name__ == "__main__":
    main() 