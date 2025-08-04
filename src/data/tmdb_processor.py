import pandas as pd
import numpy as np
import json
import ast
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class TMDBDataProcessor:
    """
    Data processor for TMDB movie dataset.
    Handles the specific structure of TMDB data including JSON fields.
    """
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def load_tmdb_movies(self, file_path=None):
        """
        Load TMDB movies dataset.
        
        Args:
            file_path (str): Path to the movies CSV file
            
        Returns:
            pd.DataFrame: Movies dataset
        """
        if file_path is None:
            file_path = self.raw_dir / "tmdb_5000_movies.csv"
        
        if not Path(file_path).exists():
            raise FileNotFoundError(f"TMDB movies file not found at {file_path}")
        
        movies_df = pd.read_csv(file_path)
        print(f"Loaded {len(movies_df)} TMDB movies")
        return movies_df
    
    def load_tmdb_credits(self, file_path=None):
        """
        Load TMDB credits dataset.
        
        Args:
            file_path (str): Path to the credits CSV file
            
        Returns:
            pd.DataFrame: Credits dataset
        """
        if file_path is None:
            file_path = self.raw_dir / "tmdb_5000_credits.csv"
        
        if not Path(file_path).exists():
            raise FileNotFoundError(f"TMDB credits file not found at {file_path}")
        
        credits_df = pd.read_csv(file_path)
        print(f"Loaded {len(credits_df)} TMDB credits records")
        return credits_df
    
    def parse_json_column(self, json_str):
        """
        Parse JSON string safely.
        
        Args:
            json_str: JSON string to parse
            
        Returns:
            list: Parsed JSON data or empty list if parsing fails
        """
        if pd.isna(json_str) or json_str == '':
            return []
        
        try:
            # Try to parse as JSON first
            return json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            try:
                # Try to parse as Python literal
                return ast.literal_eval(json_str)
            except (ValueError, SyntaxError):
                return []
    
    def extract_genres(self, movies_df):
        """
        Extract genres from movies dataframe.
        
        Args:
            movies_df (pd.DataFrame): Movies dataframe
            
        Returns:
            pd.DataFrame: Movies dataframe with extracted genres
        """
        df = movies_df.copy()
        
        # Parse genres JSON
        df['genres_parsed'] = df['genres'].apply(self.parse_json_column)
        
        # Extract genre names
        df['genre_names'] = df['genres_parsed'].apply(
            lambda x: [genre.get('name', '') for genre in x if isinstance(genre, dict)]
        )
        
        # Create genre string
        df['genres'] = df['genre_names'].apply(lambda x: '|'.join(x) if x else 'Unknown')
        
        return df
    
    def extract_keywords(self, movies_df):
        """
        Extract keywords from movies dataframe.
        
        Args:
            movies_df (pd.DataFrame): Movies dataframe
            
        Returns:
            pd.DataFrame: Movies dataframe with extracted keywords
        """
        df = movies_df.copy()
        
        # Parse keywords JSON
        df['keywords_parsed'] = df['keywords'].apply(self.parse_json_column)
        
        # Extract keyword names
        df['keyword_names'] = df['keywords_parsed'].apply(
            lambda x: [keyword.get('name', '') for keyword in x if isinstance(keyword, dict)]
        )
        
        # Create keyword string
        df['keywords_str'] = df['keyword_names'].apply(lambda x: ' '.join(x) if x else '')
        
        return df
    
    def extract_cast_and_crew(self, credits_df):
        """
        Extract cast and crew information from credits dataframe.
        
        Args:
            credits_df (pd.DataFrame): Credits dataframe
            
        Returns:
            pd.DataFrame: Credits dataframe with extracted cast and crew
        """
        df = credits_df.copy()
        
        # Parse cast JSON
        df['cast_parsed'] = df['cast'].apply(self.parse_json_column)
        
        # Parse crew JSON
        df['crew_parsed'] = df['crew'].apply(self.parse_json_column)
        
        # Extract top cast members (first 5)
        df['top_cast'] = df['cast_parsed'].apply(
            lambda x: [cast.get('name', '') for cast in x[:5] if isinstance(cast, dict)]
        )
        
        # Extract directors
        df['directors'] = df['crew_parsed'].apply(
            lambda x: [crew.get('name', '') for crew in x 
                      if isinstance(crew, dict) and crew.get('job', '').lower() == 'director']
        )
        
        # Create cast and crew strings
        df['cast_str'] = df['top_cast'].apply(lambda x: ' '.join(x) if x else '')
        df['directors_str'] = df['directors'].apply(lambda x: ' '.join(x) if x else '')
        
        return df
    
    def create_rating_simulation(self, movies_df, n_users=1000, n_ratings_per_user=50):
        """
        Create simulated ratings based on movie popularity and vote_average.
        
        Args:
            movies_df (pd.DataFrame): Movies dataframe
            n_users (int): Number of simulated users
            n_ratings_per_user (int): Number of ratings per user
            
        Returns:
            pd.DataFrame: Simulated ratings dataframe
        """
        np.random.seed(42)
        
        # Filter movies with valid vote_average
        valid_movies = movies_df[movies_df['vote_average'] > 0].copy()
        
        if len(valid_movies) == 0:
            raise ValueError("No movies with valid vote_average found")
        
        ratings_data = []
        
        for user_id in range(1, n_users + 1):
            # Randomly select movies for this user
            user_movies = valid_movies.sample(n=min(n_ratings_per_user, len(valid_movies)), replace=False)
            
            for _, movie in user_movies.iterrows():
                # Base rating on vote_average with some randomness
                base_rating = movie['vote_average'] / 2  # Convert from 10-point to 5-point scale
                
                # Add some randomness
                noise = np.random.normal(0, 0.5)
                rating = base_rating + noise
                
                # Clamp to 1-5 range
                rating = max(1, min(5, rating))
                
                # Round to nearest 0.5
                rating = round(rating * 2) / 2
                
                ratings_data.append({
                    'userId': user_id,
                    'movieId': movie['id'],
                    'rating': rating,
                    'timestamp': np.random.randint(978300000, 978400000)  # Random timestamp
                })
        
        ratings_df = pd.DataFrame(ratings_data)
        print(f"Created {len(ratings_df)} simulated ratings for {n_users} users")
        
        return ratings_df
    
    def preprocess_tmdb_data(self, movies_df, credits_df):
        """
        Preprocess TMDB data for recommendation system.
        
        Args:
            movies_df (pd.DataFrame): Raw movies dataframe
            credits_df (pd.DataFrame): Raw credits dataframe
            
        Returns:
            tuple: (processed_movies_df, processed_credits_df, ratings_df)
        """
        print("Processing TMDB movies data...")
        
        # Process movies
        movies_processed = movies_df.copy()
        movies_processed = self.extract_genres(movies_processed)
        movies_processed = self.extract_keywords(movies_processed)
        
        # Clean and prepare movies data
        movies_processed['title'] = movies_processed['title'].fillna('Unknown')
        movies_processed['overview'] = movies_processed['overview'].fillna('')
        movies_processed['tagline'] = movies_processed['tagline'].fillna('')
        
        # Extract year from release_date
        movies_processed['release_date'] = pd.to_datetime(movies_processed['release_date'], errors='coerce')
        movies_processed['year'] = movies_processed['release_date'].dt.year
        
        # Create feature text for content-based filtering
        movies_processed['feature_text'] = (
            movies_processed['title'] + ' ' +
            movies_processed['genres'] + ' ' +
            movies_processed['keywords_str'] + ' ' +
            movies_processed['overview'] + ' ' +
            movies_processed['tagline']
        )
        
        print("Processing TMDB credits data...")
        
        # Process credits
        credits_processed = self.extract_cast_and_crew(credits_df)
        
        # Merge movies and credits (credits uses 'movie_id', movies uses 'id')
        credits_processed = credits_processed.rename(columns={'movie_id': 'id'})
        merged_df = movies_processed.merge(credits_processed, on='id', how='left')
        
        # Add cast and crew to feature text
        merged_df['feature_text'] = (
            merged_df['feature_text'] + ' ' +
            merged_df['cast_str'] + ' ' +
            merged_df['directors_str']
        )
        
        print("Creating simulated ratings...")
        
        # Create simulated ratings
        ratings_df = self.create_rating_simulation(movies_processed)
        
        return merged_df, ratings_df
    
    def save_processed_data(self, data, filename):
        """
        Save processed data to processed directory.
        
        Args:
            data: Data to save (DataFrame, array, etc.)
            filename (str): Name of the file
        """
        file_path = self.processed_dir / filename
        
        if isinstance(data, pd.DataFrame):
            data.to_csv(file_path, index=False)
        elif isinstance(data, np.ndarray):
            np.save(file_path, data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        print(f"Saved processed data to {file_path}")
    
    def get_movie_recommendations(self, movies_df, movie_id, n_recommendations=10):
        """
        Get movie recommendations based on content similarity.
        
        Args:
            movies_df (pd.DataFrame): Processed movies dataframe
            movie_id (int): ID of the movie to get recommendations for
            n_recommendations (int): Number of recommendations to return
            
        Returns:
            pd.DataFrame: Recommended movies
        """
        if movie_id not in movies_df['id'].values:
            raise ValueError(f"Movie ID {movie_id} not found in dataset")
        
        # Get the target movie
        target_movie = movies_df[movies_df['id'] == movie_id].iloc[0]
        
        # Calculate similarity based on genres
        target_genres = set(target_movie['genre_names'])
        
        similarities = []
        for _, movie in movies_df.iterrows():
            if movie['id'] == movie_id:
                continue
            
            movie_genres = set(movie['genre_names'])
            
            # Jaccard similarity
            if target_genres and movie_genres:
                similarity = len(target_genres.intersection(movie_genres)) / len(target_genres.union(movie_genres))
            else:
                similarity = 0
            
            # Handle different title column names
            title_col = 'title_x' if 'title_x' in movie.index else 'title'
            similarities.append({
                'id': movie['id'],
                'title': movie[title_col],
                'genres': movie['genres'],
                'vote_average': movie['vote_average'],
                'similarity': similarity
            })
        
        # Sort by similarity and return top recommendations
        similarities_df = pd.DataFrame(similarities)
        recommendations = similarities_df.nlargest(n_recommendations, 'similarity')
        
        return recommendations
    
    def analyze_dataset(self, movies_df, credits_df):
        """
        Analyze the TMDB dataset and provide insights.
        
        Args:
            movies_df (pd.DataFrame): Movies dataframe
            credits_df (pd.DataFrame): Credits dataframe
            
        Returns:
            dict: Analysis results
        """
        analysis = {}
        
        # Basic statistics
        analysis['total_movies'] = len(movies_df)
        analysis['total_credits'] = len(credits_df)
        
        # Genre analysis
        all_genres = []
        for genres in movies_df['genres'].apply(self.parse_json_column):
            all_genres.extend([genre.get('name', '') for genre in genres if isinstance(genre, dict)])
        
        genre_counts = pd.Series(all_genres).value_counts()
        analysis['top_genres'] = genre_counts.head(10).to_dict()
        
        # Vote average analysis
        analysis['avg_vote'] = movies_df['vote_average'].mean()
        analysis['vote_std'] = movies_df['vote_average'].std()
        
        # Budget and revenue analysis
        analysis['avg_budget'] = movies_df['budget'].mean()
        analysis['avg_revenue'] = movies_df['revenue'].mean()
        
        # Popularity analysis
        analysis['avg_popularity'] = movies_df['popularity'].mean()
        
        return analysis 