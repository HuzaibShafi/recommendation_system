#!/usr/bin/env python3
"""
Movie Recommendation System - Main Application

This script demonstrates a complete movie recommendation system using various algorithms:
- Collaborative Filtering (User-based and Item-based)
- Content-based Filtering
- Matrix Factorization (SVD and NMF)
- Hybrid Recommender

Author: Huzaib Shafi
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.data_loader import MovieDataLoader
from src.models.recommendation_models import (
    CollaborativeFiltering, 
    ContentBasedFiltering, 
    MatrixFactorization, 
    HybridRecommender
)
from src.utils.evaluation import RecommendationEvaluator
from src.visualization.plots import RecommendationVisualizer

def main():
    """
    Main function to run the movie recommendation system.
    """
    print("🎬 Movie Recommendation System")
    print("=" * 50)
    
    # Initialize components
    data_loader = MovieDataLoader()
    evaluator = RecommendationEvaluator()
    visualizer = RecommendationVisualizer()
    
    # Check if data exists, if not provide instructions
    movies_file = data_loader.raw_dir / "movies.csv"
    ratings_file = data_loader.raw_dir / "ratings.csv"
    
    if not movies_file.exists() or not ratings_file.exists():
        print("❌ Data files not found!")
        print("\nTo get started, you need to:")
        print("1. Download the MovieLens dataset from Kaggle:")
        print("   https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset")
        print("2. Extract the following files to data/raw/:")
        print("   - movies.csv")
        print("   - ratings.csv")
        print("3. Run this script again")
        
        # Create sample data for demonstration
        print("\n📝 Creating sample data for demonstration...")
        create_sample_data(data_loader)
    
    # Load and preprocess data
    print("\n📊 Loading and preprocessing data...")
    movies_df = data_loader.load_movies_data()
    ratings_df = data_loader.load_ratings_data()
    
    # Preprocess data
    movies_processed = data_loader.preprocess_movies(movies_df)
    ratings_processed = data_loader.preprocess_ratings(ratings_df)
    
    # Create user-movie matrix
    print("🔢 Creating user-movie matrix...")
    user_movie_matrix = data_loader.create_user_movie_matrix(ratings_processed, movies_processed)
    
    # Save processed data
    data_loader.save_processed_data(movies_processed, "movies_processed.csv")
    data_loader.save_processed_data(ratings_processed, "ratings_processed.csv")
    data_loader.save_processed_data(user_movie_matrix, "user_movie_matrix.csv")
    
    # Data exploration
    print("\n📈 Exploring data...")
    explore_data(movies_processed, ratings_processed, visualizer)
    
    # Train and evaluate models
    print("\n🤖 Training recommendation models...")
    models = train_models(user_movie_matrix, movies_processed)
    
    # Evaluate models
    print("\n📊 Evaluating models...")
    evaluation_results = evaluate_models(models, ratings_processed, evaluator)
    
    # Generate recommendations
    print("\n🎯 Generating recommendations...")
    generate_recommendations(models, movies_processed, user_movie_matrix)
    
    # Create interactive dashboard
    print("\n📊 Creating interactive dashboard...")
    visualizer.create_interactive_dashboard(movies_processed, ratings_processed)
    
    print("\n✅ Movie Recommendation System completed successfully!")

def create_sample_data(data_loader):
    """
    Create sample data for demonstration purposes.
    """
    # Sample movies data
    sample_movies = pd.DataFrame({
        'movieId': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'title': [
            'Toy Story (1995)',
            'Jumanji (1995)',
            'Grumpier Old Men (1995)',
            'Waiting to Exhale (1995)',
            'Father of the Bride Part II (1995)',
            'Heat (1995)',
            'Sabrina (1995)',
            'Tom and Huck (1995)',
            'Sudden Death (1995)',
            'GoldenEye (1995)'
        ],
        'genres': [
            'Adventure|Animation|Children|Comedy|Fantasy',
            'Adventure|Children|Fantasy',
            'Comedy|Romance',
            'Comedy|Drama|Romance',
            'Comedy',
            'Action|Crime|Thriller',
            'Comedy|Romance',
            'Adventure|Children',
            'Action',
            'Action|Adventure|Thriller'
        ]
    })
    
    # Sample ratings data
    np.random.seed(42)
    n_users = 100
    n_movies = 10
    n_ratings = 500
    
    user_ids = np.random.randint(1, n_users + 1, n_ratings)
    movie_ids = np.random.randint(1, n_movies + 1, n_ratings)
    ratings = np.random.choice([1, 2, 3, 4, 5], n_ratings, p=[0.1, 0.15, 0.25, 0.3, 0.2])
    timestamps = np.random.randint(978300000, 978400000, n_ratings)
    
    sample_ratings = pd.DataFrame({
        'userId': user_ids,
        'movieId': movie_ids,
        'rating': ratings,
        'timestamp': timestamps
    })
    
    # Save sample data
    data_loader.save_processed_data(sample_movies, "movies.csv")
    data_loader.save_processed_data(sample_ratings, "ratings.csv")
    
    print("✅ Sample data created successfully!")

def explore_data(movies_df, ratings_df, visualizer):
    """
    Explore and visualize the data.
    """
    # Basic statistics
    print(f"📊 Dataset Statistics:")
    print(f"   - Number of movies: {len(movies_df)}")
    print(f"   - Number of ratings: {len(ratings_df)}")
    print(f"   - Number of users: {ratings_df['userId'].nunique()}")
    print(f"   - Average rating: {ratings_df['rating'].mean():.2f}")
    
    # Visualizations
    visualizer.plot_rating_distribution(ratings_df)
    visualizer.plot_genre_analysis(movies_df, ratings_df)
    visualizer.plot_user_activity(ratings_df)
    visualizer.plot_movie_popularity(movies_df, ratings_df)

def train_models(user_movie_matrix, movies_df):
    """
    Train different recommendation models.
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
    
    # Content-based Filtering
    print("   Training Content-based Filtering...")
    cb = ContentBasedFiltering()
    cb.fit(movies_df)
    models['Content-based'] = cb
    
    # Matrix Factorization - SVD
    print("   Training Matrix Factorization (SVD)...")
    mf_svd = MatrixFactorization(method='svd', n_components=20)
    mf_svd.fit(user_movie_matrix)
    models['Matrix Factorization (SVD)'] = mf_svd
    
    # Matrix Factorization - NMF
    print("   Training Matrix Factorization (NMF)...")
    mf_nmf = MatrixFactorization(method='nmf', n_components=20)
    mf_nmf.fit(user_movie_matrix)
    models['Matrix Factorization (NMF)'] = mf_nmf
    
    # Hybrid Recommender
    print("   Training Hybrid Recommender...")
    hybrid = HybridRecommender()
    hybrid.fit(user_movie_matrix, movies_df)
    models['Hybrid'] = hybrid
    
    return models

def evaluate_models(models, ratings_df, evaluator):
    """
    Evaluate the trained models.
    """
    # Create train-test split
    train_data, test_data = evaluator.create_train_test_split(ratings_df, test_size=0.2)
    
    # Get test users
    test_users = test_data['userId'].unique()[:50]  # Limit for faster evaluation
    
    # Evaluate models
    evaluation_results = evaluator.compare_models(models, test_data, test_users, k=10)
    
    # Display results
    print("\n📊 Model Evaluation Results:")
    print(evaluation_results.to_string(index=False))
    
    return evaluation_results

def generate_recommendations(models, movies_df, user_movie_matrix):
    """
    Generate recommendations for sample users.
    """
    # Get some sample users
    sample_users = user_movie_matrix.index[:5]
    
    print(f"\n🎯 Generating recommendations for {len(sample_users)} users...")
    
    for user_id in sample_users:
        print(f"\n👤 User {user_id}:")
        
        # Get user's rated movies
        user_ratings = user_movie_matrix.loc[user_id]
        rated_movies = user_ratings[user_ratings.notna()].index
        
        if len(rated_movies) > 0:
            print(f"   Rated {len(rated_movies)} movies")
            
            # Get recommendations from each model
            for model_name, model in models.items():
                try:
                    if model_name == 'Content-based':
                        # For content-based, use the highest rated movie
                        best_movie = user_ratings.idxmax()
                        recommendations = model.recommend(best_movie, 5)
                    else:
                        recommendations = model.recommend(user_id, 5)
                    
                    # Get movie titles
                    recommended_titles = []
                    for movie_id in recommendations:
                        movie_info = movies_df[movies_df['movieId'] == movie_id]
                        if not movie_info.empty:
                            title = movie_info.iloc[0]['title']
                            recommended_titles.append(title)
                    
                    print(f"   {model_name}: {', '.join(recommended_titles[:3])}")
                    
                except Exception as e:
                    print(f"   {model_name}: Error - {str(e)}")
        else:
            print("   No rated movies")

if __name__ == "__main__":
    main() 