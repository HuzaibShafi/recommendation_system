import pandas as pd
import numpy as np
from pathlib import Path
import requests
import zipfile
from tqdm import tqdm
import os

class MovieDataLoader:
    """
    A class to load and preprocess movie recommendation datasets.
    """
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def download_kaggle_dataset(self, dataset_url, filename):
        """
        Download dataset from Kaggle or other sources.
        
        Args:
            dataset_url (str): URL to download the dataset from
            filename (str): Name of the file to save
        """
        file_path = self.raw_dir / filename
        
        if not file_path.exists():
            print(f"Downloading {filename}...")
            response = requests.get(dataset_url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(file_path, 'wb') as file, tqdm(
                desc=filename,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    pbar.update(size)
            print(f"Downloaded {filename} successfully!")
        else:
            print(f"{filename} already exists!")
    
    def load_movies_data(self, file_path=None):
        """
        Load movies dataset.
        
        Args:
            file_path (str): Path to the movies CSV file
            
        Returns:
            pd.DataFrame: Movies dataset
        """
        if file_path is None:
            file_path = self.raw_dir / "movies.csv"
        
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Movies file not found at {file_path}")
        
        movies_df = pd.read_csv(file_path)
        print(f"Loaded {len(movies_df)} movies")
        return movies_df
    
    def load_ratings_data(self, file_path=None):
        """
        Load ratings dataset.
        
        Args:
            file_path (str): Path to the ratings CSV file
            
        Returns:
            pd.DataFrame: Ratings dataset
        """
        if file_path is None:
            file_path = self.raw_dir / "ratings.csv"
        
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Ratings file not found at {file_path}")
        
        ratings_df = pd.read_csv(file_path)
        print(f"Loaded {len(ratings_df)} ratings")
        return ratings_df
    
    def load_users_data(self, file_path=None):
        """
        Load users dataset.
        
        Args:
            file_path (str): Path to the users CSV file
            
        Returns:
            pd.DataFrame: Users dataset
        """
        if file_path is None:
            file_path = self.raw_dir / "users.csv"
        
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Users file not found at {file_path}")
        
        users_df = pd.read_csv(file_path)
        print(f"Loaded {len(users_df)} users")
        return users_df
    
    def preprocess_movies(self, movies_df):
        """
        Preprocess movies data.
        
        Args:
            movies_df (pd.DataFrame): Raw movies dataframe
            
        Returns:
            pd.DataFrame: Preprocessed movies dataframe
        """
        # Create a copy to avoid modifying original data
        df = movies_df.copy()
        
        # Handle missing values
        df['genres'] = df['genres'].fillna('Unknown')
        df['title'] = df['title'].fillna('Unknown')
        
        # Extract year from title
        df['year'] = df['title'].str.extract(r'\((\d{4})\)')
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        
        # Clean title (remove year)
        df['clean_title'] = df['title'].str.replace(r'\(\d{4}\)', '', regex=True).str.strip()
        
        # Split genres into list
        df['genre_list'] = df['genres'].str.split('|')
        
        # Create genre dummy variables
        all_genres = set()
        for genres in df['genre_list'].dropna():
            all_genres.update(genres)
        
        for genre in all_genres:
            df[f'genre_{genre.lower().replace(" ", "_")}'] = df['genres'].str.contains(genre, na=False).astype(int)
        
        return df
    
    def preprocess_ratings(self, ratings_df):
        """
        Preprocess ratings data.
        
        Args:
            ratings_df (pd.DataFrame): Raw ratings dataframe
            
        Returns:
            pd.DataFrame: Preprocessed ratings dataframe
        """
        df = ratings_df.copy()
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Add date features
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        return df
    
    def create_user_movie_matrix(self, ratings_df, movies_df):
        """
        Create user-movie rating matrix.
        
        Args:
            ratings_df (pd.DataFrame): Ratings dataframe
            movies_df (pd.DataFrame): Movies dataframe
            
        Returns:
            pd.DataFrame: User-movie matrix
        """
        # Create pivot table
        user_movie_matrix = ratings_df.pivot_table(
            index='userId',
            columns='movieId',
            values='rating',
            fill_value=0
        )
        
        return user_movie_matrix
    
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
    
    def load_processed_data(self, filename):
        """
        Load processed data from processed directory.
        
        Args:
            filename (str): Name of the file
            
        Returns:
            Loaded data
        """
        file_path = self.processed_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Processed file not found at {file_path}")
        
        if filename.endswith('.csv'):
            return pd.read_csv(file_path)
        elif filename.endswith('.npy'):
            return np.load(file_path)
        else:
            raise ValueError(f"Unsupported file format: {filename}") 