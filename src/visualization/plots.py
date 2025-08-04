import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class RecommendationVisualizer:
    """
    Visualization utilities for recommendation systems.
    """
    
    def __init__(self, style='seaborn'):
        """
        Initialize visualizer.
        
        Args:
            style (str): Matplotlib style to use
        """
        plt.style.use(style)
        sns.set_palette("husl")
    
    def plot_rating_distribution(self, ratings_df, figsize=(12, 8)):
        """
        Plot rating distribution.
        
        Args:
            ratings_df: Ratings dataframe
            figsize: Figure size
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Overall rating distribution
        sns.histplot(data=ratings_df, x='rating', bins=10, ax=axes[0, 0])
        axes[0, 0].set_title('Overall Rating Distribution')
        axes[0, 0].set_xlabel('Rating')
        axes[0, 0].set_ylabel('Count')
        
        # Rating distribution by user
        user_rating_counts = ratings_df.groupby('userId')['rating'].count()
        sns.histplot(user_rating_counts, bins=30, ax=axes[0, 1])
        axes[0, 1].set_title('Number of Ratings per User')
        axes[0, 1].set_xlabel('Number of Ratings')
        axes[0, 1].set_ylabel('Count')
        
        # Rating distribution by movie
        movie_rating_counts = ratings_df.groupby('movieId')['rating'].count()
        sns.histplot(movie_rating_counts, bins=30, ax=axes[1, 0])
        axes[1, 0].set_title('Number of Ratings per Movie')
        axes[1, 0].set_xlabel('Number of Ratings')
        axes[1, 0].set_ylabel('Count')
        
        # Average rating by movie
        movie_avg_ratings = ratings_df.groupby('movieId')['rating'].mean()
        sns.histplot(movie_avg_ratings, bins=20, ax=axes[1, 1])
        axes[1, 1].set_title('Average Rating per Movie')
        axes[1, 1].set_xlabel('Average Rating')
        axes[1, 1].set_ylabel('Count')
        
        plt.tight_layout()
        plt.show()
    
    def plot_genre_analysis(self, movies_df, ratings_df, figsize=(15, 10)):
        """
        Plot genre analysis.
        
        Args:
            movies_df: Movies dataframe
            ratings_df: Ratings dataframe
            figsize: Figure size
        """
        # Merge movies and ratings (handle different column names)
        movie_id_col = 'movieId' if 'movieId' in ratings_df.columns else 'id'
        merged_df = ratings_df.merge(movies_df, left_on=movie_id_col, right_on='id')
        
        # Split genres and create genre dataframe
        genre_data = []
        for _, row in merged_df.iterrows():
            genres = str(row['genres']).split('|')
            for genre in genres:
                genre_data.append({
                    'genre': genre,
                    'rating': row['rating'],
                    'movieId': row[movie_id_col]
                })
        
        genre_df = pd.DataFrame(genre_data)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Genre popularity (number of movies)
        genre_counts = movies_df['genres'].str.split('|').explode().value_counts()
        genre_counts.head(10).plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('Most Popular Genres (Number of Movies)')
        axes[0, 0].set_xlabel('Genre')
        axes[0, 0].set_ylabel('Number of Movies')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Genre average ratings
        genre_avg_ratings = genre_df.groupby('genre')['rating'].mean().sort_values(ascending=False)
        genre_avg_ratings.head(10).plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('Average Rating by Genre')
        axes[0, 1].set_xlabel('Genre')
        axes[0, 1].set_ylabel('Average Rating')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Genre rating distribution
        top_genres = genre_counts.head(5).index
        genre_rating_data = genre_df[genre_df['genre'].isin(top_genres)]
        sns.boxplot(data=genre_rating_data, x='genre', y='rating', ax=axes[1, 0])
        axes[1, 0].set_title('Rating Distribution by Top Genres')
        axes[1, 0].set_xlabel('Genre')
        axes[1, 0].set_ylabel('Rating')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Genre popularity vs average rating
        genre_stats = pd.DataFrame({
            'count': genre_counts,
            'avg_rating': genre_avg_ratings
        })
        genre_stats = genre_stats.dropna()
        
        axes[1, 1].scatter(genre_stats['count'], genre_stats['avg_rating'], alpha=0.6)
        axes[1, 1].set_title('Genre Popularity vs Average Rating')
        axes[1, 1].set_xlabel('Number of Movies')
        axes[1, 1].set_ylabel('Average Rating')
        
        # Add genre labels for some points
        for idx, row in genre_stats.head(10).iterrows():
            axes[1, 1].annotate(idx, (row['count'], row['avg_rating']), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        plt.show()
    
    def plot_user_activity(self, ratings_df, figsize=(15, 10)):
        """
        Plot user activity analysis.
        
        Args:
            ratings_df: Ratings dataframe
            figsize: Figure size
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Convert timestamp to datetime if it's not already
        if 'timestamp' in ratings_df.columns:
            if ratings_df['timestamp'].dtype == 'int64':
                ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'], unit='s')
        
        # User activity over time
        if 'timestamp' in ratings_df.columns:
            daily_ratings = ratings_df.groupby(ratings_df['timestamp'].dt.date).size()
            daily_ratings.plot(ax=axes[0, 0])
            axes[0, 0].set_title('Daily Rating Activity')
            axes[0, 0].set_xlabel('Date')
            axes[0, 0].set_ylabel('Number of Ratings')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # User rating frequency
        user_rating_counts = ratings_df.groupby('userId').size()
        sns.histplot(user_rating_counts, bins=50, ax=axes[0, 1])
        axes[0, 1].set_title('User Rating Frequency Distribution')
        axes[0, 1].set_xlabel('Number of Ratings per User')
        axes[0, 1].set_ylabel('Count')
        
        # User average rating
        user_avg_ratings = ratings_df.groupby('userId')['rating'].mean()
        sns.histplot(user_avg_ratings, bins=30, ax=axes[1, 0])
        axes[1, 0].set_title('User Average Rating Distribution')
        axes[1, 0].set_xlabel('Average Rating per User')
        axes[1, 0].set_ylabel('Count')
        
        # Rating frequency vs average rating
        user_stats = pd.DataFrame({
            'rating_count': user_rating_counts,
            'avg_rating': user_avg_ratings
        })
        
        axes[1, 1].scatter(user_stats['rating_count'], user_stats['avg_rating'], alpha=0.5)
        axes[1, 1].set_title('User Rating Frequency vs Average Rating')
        axes[1, 1].set_xlabel('Number of Ratings')
        axes[1, 1].set_ylabel('Average Rating')
        axes[1, 1].set_xscale('log')
        
        plt.tight_layout()
        plt.show()
    
    def plot_movie_popularity(self, movies_df, ratings_df, figsize=(15, 10)):
        """
        Plot movie popularity analysis.
        
        Args:
            movies_df: Movies dataframe
            ratings_df: Ratings dataframe
            figsize: Figure size
        """
        # Calculate movie statistics (handle different column names)
        movie_id_col = 'movieId' if 'movieId' in ratings_df.columns else 'id'
        movie_stats = ratings_df.groupby(movie_id_col).agg({
            'rating': ['count', 'mean', 'std']
        }).round(3)
        movie_stats.columns = ['rating_count', 'avg_rating', 'rating_std']
        movie_stats = movie_stats.reset_index()
        
        # Merge with movie information (handle TMDB column names)
        if 'title_x' in movies_df.columns:
            # TMDB processed data has title_x and title_y due to merge
            movie_cols = ['id', 'title_x', 'genres']
        elif 'id' in movies_df.columns:
            movie_cols = ['id', 'title', 'genres']
        else:
            movie_cols = ['movieId', 'title', 'genres']
        
        movie_stats = movie_stats.merge(movies_df[movie_cols], left_on=movie_id_col, right_on=movie_cols[0])
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Most rated movies
        top_rated_movies = movie_stats.nlargest(10, 'rating_count')
        title_col = 'title_x' if 'title_x' in movie_stats.columns else 'title'
        top_rated_movies['title_short'] = top_rated_movies[title_col].str[:30] + '...'
        top_rated_movies.plot(kind='barh', x='title_short', y='rating_count', ax=axes[0, 0])
        axes[0, 0].set_title('Most Rated Movies')
        axes[0, 0].set_xlabel('Number of Ratings')
        
        # Highest rated movies (with minimum ratings threshold)
        min_ratings = 50
        qualified_movies = movie_stats[movie_stats['rating_count'] >= min_ratings]
        top_rated = qualified_movies.nlargest(10, 'avg_rating')
        title_col = 'title_x' if 'title_x' in movie_stats.columns else 'title'
        top_rated['title_short'] = top_rated[title_col].str[:30] + '...'
        top_rated.plot(kind='barh', x='title_short', y='avg_rating', ax=axes[0, 1])
        axes[0, 1].set_title(f'Highest Rated Movies (≥{min_ratings} ratings)')
        axes[0, 1].set_xlabel('Average Rating')
        
        # Rating count vs average rating
        axes[1, 0].scatter(movie_stats['rating_count'], movie_stats['avg_rating'], alpha=0.5)
        axes[1, 0].set_title('Movie Popularity vs Average Rating')
        axes[1, 0].set_xlabel('Number of Ratings')
        axes[1, 0].set_ylabel('Average Rating')
        axes[1, 0].set_xscale('log')
        
        # Rating distribution for popular movies
        popular_movies = movie_stats[movie_stats['rating_count'] >= 100]
        sns.histplot(popular_movies['avg_rating'], bins=30, ax=axes[1, 1])
        axes[1, 1].set_title('Average Rating Distribution (Popular Movies)')
        axes[1, 1].set_xlabel('Average Rating')
        axes[1, 1].set_ylabel('Count')
        
        plt.tight_layout()
        plt.show()
    
    def plot_recommendation_results(self, recommendations_dict, movies_df, figsize=(15, 10)):
        """
        Plot recommendation results.
        
        Args:
            recommendations_dict: Dictionary mapping user_id to list of recommended movie_ids
            movies_df: Movies dataframe
            figsize: Figure size
        """
        # Analyze recommendations
        all_recommendations = []
        for user_id, movie_ids in recommendations_dict.items():
            for movie_id in movie_ids:
                all_recommendations.append(movie_id)
        
        recommendation_counts = pd.Series(all_recommendations).value_counts()
        
        # Get movie information for top recommendations
        top_recommendations = recommendation_counts.head(20)
        top_movies = movies_df[movies_df['movieId'].isin(top_recommendations.index)]
        top_movies = top_movies.merge(
            pd.DataFrame({'movieId': top_recommendations.index, 'rec_count': top_recommendations.values}),
            on='movieId'
        )
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Most recommended movies
        top_movies['title_short'] = top_movies['title'].str[:30] + '...'
        top_movies.plot(kind='barh', x='title_short', y='rec_count', ax=axes[0, 0])
        axes[0, 0].set_title('Most Recommended Movies')
        axes[0, 0].set_xlabel('Number of Recommendations')
        
        # Genre distribution of recommendations
        genre_recs = []
        for _, movie in top_movies.iterrows():
            genres = str(movie['genres']).split('|')
            for genre in genres:
                genre_recs.append(genre)
        
        genre_rec_counts = pd.Series(genre_recs).value_counts()
        genre_rec_counts.head(10).plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('Genre Distribution in Recommendations')
        axes[0, 1].set_xlabel('Genre')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Recommendation diversity (number of unique movies recommended)
        diversity_scores = []
        for user_id, movie_ids in recommendations_dict.items():
            diversity_scores.append(len(set(movie_ids)))
        
        sns.histplot(diversity_scores, bins=20, ax=axes[1, 0])
        axes[1, 0].set_title('Recommendation Diversity per User')
        axes[1, 0].set_xlabel('Number of Unique Movies Recommended')
        axes[1, 0].set_ylabel('Count')
        
        # Coverage analysis
        total_movies = len(movies_df)
        recommended_movies = len(set(all_recommendations))
        coverage = recommended_movies / total_movies
        
        axes[1, 1].pie([coverage, 1-coverage], labels=['Recommended', 'Not Recommended'], 
                      autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title(f'Movie Coverage: {coverage:.1%}')
        
        plt.tight_layout()
        plt.show()
    
    def create_interactive_dashboard(self, movies_df, ratings_df, recommendations_dict=None):
        """
        Create an interactive dashboard using Plotly.
        
        Args:
            movies_df: Movies dataframe
            ratings_df: Ratings dataframe
            recommendations_dict: Dictionary of recommendations (optional)
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Rating Distribution', 'Genre Analysis', 
                          'User Activity', 'Movie Popularity'),
            specs=[[{"type": "histogram"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Rating distribution
        fig.add_trace(
            go.Histogram(x=ratings_df['rating'], name='Ratings'),
            row=1, col=1
        )
        
        # Genre analysis
        genre_counts = movies_df['genres'].str.split('|').explode().value_counts()
        fig.add_trace(
            go.Bar(x=genre_counts.head(10).index, y=genre_counts.head(10).values, name='Genres'),
            row=1, col=2
        )
        
        # User activity
        user_rating_counts = ratings_df.groupby('userId').size()
        fig.add_trace(
            go.Scatter(x=user_rating_counts.index, y=user_rating_counts.values, 
                      mode='markers', name='User Activity'),
            row=2, col=1
        )
        
        # Movie popularity
        movie_stats = ratings_df.groupby('movieId').agg({
            'rating': ['count', 'mean']
        }).round(3)
        movie_stats.columns = ['rating_count', 'avg_rating']
        movie_stats = movie_stats.reset_index()
        
        fig.add_trace(
            go.Scatter(x=movie_stats['rating_count'], y=movie_stats['avg_rating'],
                      mode='markers', name='Movie Popularity'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(height=800, title_text="Movie Recommendation System Dashboard")
        fig.show()
    
    def plot_model_comparison(self, comparison_results, figsize=(15, 10)):
        """
        Plot model comparison results.
        
        Args:
            comparison_results: DataFrame with model comparison results
            figsize: Figure size
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Precision comparison
        comparison_results.plot(kind='bar', x='model', y='precision_at_k', ax=axes[0, 0])
        axes[0, 0].set_title('Precision@K Comparison')
        axes[0, 0].set_ylabel('Precision@K')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Recall comparison
        comparison_results.plot(kind='bar', x='model', y='recall_at_k', ax=axes[0, 1])
        axes[0, 1].set_title('Recall@K Comparison')
        axes[0, 1].set_ylabel('Recall@K')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # F1 comparison
        comparison_results.plot(kind='bar', x='model', y='f1_at_k', ax=axes[1, 0])
        axes[1, 0].set_title('F1@K Comparison')
        axes[1, 0].set_ylabel('F1@K')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # RMSE comparison (if available)
        if 'rmse' in comparison_results.columns:
            comparison_results.plot(kind='bar', x='model', y='rmse', ax=axes[1, 1])
            axes[1, 1].set_title('RMSE Comparison')
            axes[1, 1].set_ylabel('RMSE')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show() 