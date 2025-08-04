#!/usr/bin/env python3
"""
TMDB Movie Recommendation System - Web Application

A Flask-based web interface for the movie recommendation system.
Provides user-friendly access to movie recommendations and data exploration.

Author: Huzaib Shafi
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.tmdb_processor import TMDBDataProcessor
from src.models.recommendation_models import (
    CollaborativeFiltering, 
    ContentBasedFiltering, 
    MatrixFactorization, 
    HybridRecommender
)

app = Flask(__name__)
app.secret_key = 'tmdb_recommendation_system_2024'

# Global variables for loaded data and models
tmdb_processor = None
movies_df = None
ratings_df = None
user_movie_matrix = None
models = {}
recommendation_cache = {}

def load_data_and_models():
    """Load TMDB data and train recommendation models."""
    global tmdb_processor, movies_df, ratings_df, user_movie_matrix, models
    
    try:
        # Initialize processor
        tmdb_processor = TMDBDataProcessor()
        
        # Check if processed data exists
        processed_movies_file = tmdb_processor.processed_dir / "tmdb_movies_processed.csv"
        processed_ratings_file = tmdb_processor.processed_dir / "tmdb_ratings_simulated.csv"
        
        if processed_movies_file.exists() and processed_ratings_file.exists():
            # Load processed data
            movies_df = pd.read_csv(processed_movies_file)
            ratings_df = pd.read_csv(processed_ratings_file)
        else:
            # Load and process raw data
            raw_movies_df = tmdb_processor.load_tmdb_movies()
            raw_credits_df = tmdb_processor.load_tmdb_credits()
            movies_df, ratings_df = tmdb_processor.preprocess_tmdb_data(raw_movies_df, raw_credits_df)
            
            # Save processed data
            tmdb_processor.save_processed_data(movies_df, "tmdb_movies_processed.csv")
            tmdb_processor.save_processed_data(ratings_df, "tmdb_ratings_simulated.csv")
        
        # Create user-movie matrix
        user_movie_matrix = ratings_df.pivot_table(
            index='userId',
            columns='movieId',
            values='rating',
            fill_value=0
        )
        
        # Train models
        train_models()
        
        return True
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return False

def train_models():
    """Train recommendation models."""
    global models
    
    try:
        # Collaborative Filtering - User-based
        cf_user = CollaborativeFiltering(method='user')
        cf_user.fit(user_movie_matrix)
        models['User-based CF'] = cf_user
        
        # Collaborative Filtering - Item-based
        cf_item = CollaborativeFiltering(method='item')
        cf_item.fit(user_movie_matrix)
        models['Item-based CF'] = cf_item
        
        # Content-based Filtering
        movies_for_cb = movies_df.copy()
        movies_for_cb['genres'] = movies_for_cb['feature_text']
        cb = ContentBasedFiltering()
        cb.fit(movies_for_cb)
        models['Content-based'] = cb
        
        # Matrix Factorization - SVD
        mf_svd = MatrixFactorization(method='svd', n_components=20)
        mf_svd.fit(user_movie_matrix)
        models['Matrix Factorization (SVD)'] = mf_svd
        
        # Hybrid Recommender
        hybrid = HybridRecommender()
        hybrid.fit(user_movie_matrix, movies_for_cb)
        models['Hybrid'] = hybrid
        
        print("All models trained successfully!")
        
    except Exception as e:
        print(f"Error training models: {str(e)}")

@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')

@app.route('/explore')
def explore():
    """Data exploration page."""
    if movies_df is None:
        flash('Data not loaded. Please check the data files.', 'error')
        return redirect(url_for('index'))
    
    # Get basic statistics
    stats = {
        'total_movies': len(movies_df),
        'total_ratings': len(ratings_df),
        'total_users': ratings_df['userId'].nunique(),
        'avg_rating': round(ratings_df['rating'].mean(), 2),
        'top_genres': get_top_genres()
    }
    
    # Get sample movies
    sample_movies = get_sample_movies(10)
    
    return render_template('explore.html', stats=stats, movies=sample_movies)

@app.route('/recommendations')
def recommendations():
    """Recommendations page."""
    if models == {}:
        flash('Models not trained. Please check the data files.', 'error')
        return redirect(url_for('index'))
    
    # Get available users for testing
    available_users = list(user_movie_matrix.index[:20])
    
    return render_template('recommendations.html', users=available_users)

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    """API endpoint for getting recommendations."""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        algorithm = data.get('algorithm', 'Hybrid')
        n_recommendations = data.get('n_recommendations', 10)
        
        if user_id is None:
            return jsonify({'error': 'User ID is required'}), 400
        
        # Get recommendations
        recommendations = get_user_recommendations(user_id, algorithm, n_recommendations)
        
        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'algorithm': algorithm,
            'user_id': user_id
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/movie_search')
def movie_search():
    """API endpoint for movie search."""
    try:
        query = request.args.get('q', '').lower()
        
        if not query:
            return jsonify([])
        
        # Search movies by title
        matching_movies = movies_df[
            movies_df['title_x'].str.lower().str.contains(query, na=False) |
            movies_df['title_y'].str.lower().str.contains(query, na=False)
        ].head(10)
        
        results = []
        for _, movie in matching_movies.iterrows():
            title = movie['title_x'] if pd.notna(movie['title_x']) else movie['title_y']
            results.append({
                'id': movie['id'],
                'title': title,
                'genres': movie['genres'],
                'vote_average': movie['vote_average'],
                'overview': movie['overview'][:200] + '...' if len(str(movie['overview'])) > 200 else movie['overview']
            })
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/movie_details/<int:movie_id>')
def movie_details(movie_id):
    """API endpoint for movie details."""
    try:
        movie = movies_df[movies_df['id'] == movie_id]
        
        if movie.empty:
            return jsonify({'error': 'Movie not found'}), 404
        
        movie = movie.iloc[0]
        
        # Get similar movies
        similar_movies = tmdb_processor.get_movie_recommendations(movies_df, movie_id, 5)
        
        details = {
            'id': movie['id'],
            'title': movie['title_x'] if pd.notna(movie['title_x']) else movie['title_y'],
            'genres': movie['genres'],
            'vote_average': movie['vote_average'],
            'overview': movie['overview'],
            'tagline': movie['tagline'],
            'release_date': movie['release_date'],
            'budget': movie['budget'],
            'revenue': movie['revenue'],
            'runtime': movie['runtime'],
            'similar_movies': similar_movies.to_dict('records')
        }
        
        return jsonify(details)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def get_stats():
    """API endpoint for getting statistics."""
    try:
        # Genre distribution
        genre_stats = get_genre_stats()
        
        # Rating distribution
        rating_stats = get_rating_stats()
        
        # Popular movies
        popular_movies = get_popular_movies(10)
        
        return jsonify({
            'genre_stats': genre_stats,
            'rating_stats': rating_stats,
            'popular_movies': popular_movies
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_user_recommendations(user_id, algorithm, n_recommendations=10):
    """Get recommendations for a specific user and algorithm."""
    try:
        if algorithm not in models:
            return []
        
        model = models[algorithm]
        
        if algorithm == 'Content-based':
            # For content-based, use the highest rated movie
            user_ratings = user_movie_matrix.loc[user_id]
            best_movie = user_ratings.idxmax()
            movie_ids = model.recommend(best_movie, n_recommendations)
        else:
            movie_ids = model.recommend(user_id, n_recommendations)
        
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
                    'genres': movie['genres'],
                    'vote_average': movie['vote_average'],
                    'overview': movie['overview'][:150] + '...' if len(str(movie['overview'])) > 150 else movie['overview']
                })
        
        return recommendations
        
    except Exception as e:
        print(f"Error getting recommendations: {str(e)}")
        return []

def get_top_genres():
    """Get top genres from the dataset."""
    try:
        all_genres = []
        for genres in movies_df['genres'].str.split('|'):
            if isinstance(genres, list):
                all_genres.extend(genres)
        
        genre_counts = pd.Series(all_genres).value_counts()
        return genre_counts.head(10).to_dict()
        
    except Exception as e:
        print(f"Error getting top genres: {str(e)}")
        return {}

def get_sample_movies(n=10):
    """Get sample movies for display."""
    try:
        sample = movies_df.sample(n=n)
        movies = []
        
        for _, movie in sample.iterrows():
            title = movie['title_x'] if pd.notna(movie['title_x']) else movie['title_y']
            movies.append({
                'id': movie['id'],
                'title': title,
                'genres': movie['genres'],
                'vote_average': movie['vote_average'],
                'overview': movie['overview'][:100] + '...' if len(str(movie['overview'])) > 100 else movie['overview']
            })
        
        return movies
        
    except Exception as e:
        print(f"Error getting sample movies: {str(e)}")
        return []

def get_genre_stats():
    """Get genre statistics."""
    try:
        all_genres = []
        for genres in movies_df['genres'].str.split('|'):
            if isinstance(genres, list):
                all_genres.extend(genres)
        
        genre_counts = pd.Series(all_genres).value_counts()
        return [{'genre': genre, 'count': count} for genre, count in genre_counts.head(10).items()]
        
    except Exception as e:
        print(f"Error getting genre stats: {str(e)}")
        return []

def get_rating_stats():
    """Get rating statistics."""
    try:
        rating_counts = ratings_df['rating'].value_counts().sort_index()
        return [{'rating': rating, 'count': count} for rating, count in rating_counts.items()]
        
    except Exception as e:
        print(f"Error getting rating stats: {str(e)}")
        return []

def get_popular_movies(n=10):
    """Get popular movies based on vote_average."""
    try:
        popular = movies_df.nlargest(n, 'vote_average')
        movies = []
        
        for _, movie in popular.iterrows():
            title = movie['title_x'] if pd.notna(movie['title_x']) else movie['title_y']
            movies.append({
                'id': movie['id'],
                'title': title,
                'vote_average': movie['vote_average'],
                'genres': movie['genres']
            })
        
        return movies
        
    except Exception as e:
        print(f"Error getting popular movies: {str(e)}")
        return []

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    print("🎬 Starting TMDB Movie Recommendation System Web App...")
    
    # Load data and models
    if load_data_and_models():
        print("✅ Data and models loaded successfully!")
        print("🌐 Starting web server...")
        app.run(debug=True, host='0.0.0.0', port=5001)
    else:
        print("❌ Failed to load data and models. Please check the data files.")
        print("Make sure tmdb_5000_movies.csv and tmdb_5000_credits.csv are in data/raw/") 