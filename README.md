# Movie Recommendation System

A comprehensive movie recommendation system built with Python, featuring multiple recommendation algorithms and advanced analytics.

## 🎬 Features

- **Multiple Recommendation Algorithms:**
  - Collaborative Filtering (User-based & Item-based)
  - Content-based Filtering
  - Matrix Factorization (SVD & NMF)
  - Hybrid Recommender System

- **Advanced Analytics:**
  - Data preprocessing and feature engineering
  - Model evaluation with multiple metrics
  - Interactive visualizations
  - Performance comparison

- **User-Friendly Interface:**
  - Easy-to-use data loading utilities
  - Comprehensive evaluation framework
  - Beautiful visualizations with Plotly
  - Sample data generation for testing

## 📁 Project Structure

```
recommendation_system/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_loader.py          # Data loading and preprocessing
│   ├── models/
│   │   ├── __init__.py
│   │   └── recommendation_models.py # Recommendation algorithms
│   ├── utils/
│   │   ├── __init__.py
│   │   └── evaluation.py           # Model evaluation utilities
│   └── visualization/
│       ├── __init__.py
│       └── plots.py                # Visualization utilities
├── data/
│   ├── raw/                        # Raw data files
│   └── processed/                  # Processed data files
├── tests/                          # Unit tests
├── notebooks/                      # Jupyter notebooks
├── docs/                           # Documentation
├── main.py                         # Main application
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/HuzaibShafi/recommendation_system.git
cd recommendation_system

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Setup

The system works with the MovieLens dataset. You have two options:

#### Option A: Use Sample Data (Recommended for testing)
The system will automatically create sample data if no dataset is found.

#### Option B: Use Real MovieLens Data
1. Download the MovieLens dataset from [Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)
2. Extract the following files to `data/raw/`:
   - `movies.csv`
   - `ratings.csv`

### 3. Run the System

```bash
python main.py
```

## 📊 What the System Does

1. **Data Loading & Preprocessing:**
   - Loads movie and rating data
   - Handles missing values
   - Extracts features (year, genres, etc.)
   - Creates user-movie rating matrix

2. **Model Training:**
   - Trains 6 different recommendation models
   - Implements collaborative filtering, content-based filtering, and matrix factorization
   - Creates a hybrid recommender combining multiple approaches

3. **Evaluation:**
   - Evaluates models using precision@k, recall@k, F1@k, and RMSE
   - Compares performance across different algorithms
   - Provides comprehensive evaluation metrics

4. **Visualization:**
   - Creates interactive dashboards
   - Plots rating distributions, genre analysis, and user activity
   - Visualizes model comparison results

5. **Recommendations:**
   - Generates personalized movie recommendations
   - Shows recommendations from different algorithms
   - Provides movie titles and details

## 🤖 Recommendation Algorithms

### 1. Collaborative Filtering
- **User-based CF:** Finds similar users and recommends movies they liked
- **Item-based CF:** Finds similar movies and recommends based on user's history

### 2. Content-based Filtering
- Uses movie features (genres, title) to find similar movies
- Implements TF-IDF vectorization for text features

### 3. Matrix Factorization
- **SVD (Singular Value Decomposition):** Decomposes user-movie matrix into latent factors
- **NMF (Non-negative Matrix Factorization):** Similar to SVD but with non-negative constraints

### 4. Hybrid Recommender
- Combines multiple approaches with weighted scoring
- Provides more robust and diverse recommendations

## 📈 Evaluation Metrics

- **Precision@K:** Percentage of recommended items that are relevant
- **Recall@K:** Percentage of relevant items that are recommended
- **F1@K:** Harmonic mean of precision and recall
- **RMSE:** Root Mean Square Error for rating predictions
- **MAE:** Mean Absolute Error for rating predictions

## 🎨 Visualizations

The system provides comprehensive visualizations:

- **Rating Distribution:** Overall and per-user/movie rating patterns
- **Genre Analysis:** Genre popularity and rating analysis
- **User Activity:** User behavior patterns over time
- **Movie Popularity:** Most rated and highest-rated movies
- **Model Comparison:** Performance comparison across algorithms
- **Interactive Dashboard:** Plotly-based interactive visualizations

## 🔧 Customization

### Adding New Models
1. Create a new class in `src/models/recommendation_models.py`
2. Implement `fit()` and `recommend()` methods
3. Add to the models dictionary in `main.py`

### Custom Evaluation Metrics
1. Add new metrics to `src/utils/evaluation.py`
2. Update the evaluation pipeline in `main.py`

### New Visualizations
1. Add new plotting functions to `src/visualization/plots.py`
2. Call them from the main application

## 📝 Example Usage

```python
from src.data.data_loader import MovieDataLoader
from src.models.recommendation_models import CollaborativeFiltering

# Load data
data_loader = MovieDataLoader()
movies_df = data_loader.load_movies_data()
ratings_df = data_loader.load_ratings_data()

# Create user-movie matrix
user_movie_matrix = data_loader.create_user_movie_matrix(ratings_df, movies_df)

# Train model
cf_model = CollaborativeFiltering(method='user')
cf_model.fit(user_movie_matrix)

# Get recommendations
recommendations = cf_model.recommend(user_id=1, n_recommendations=10)
print(f"Recommended movies for user 1: {recommendations}")
```

## 🧪 Testing

```bash
# Run tests
python -m pytest tests/
```

## 📚 Dependencies

- **Data Processing:** pandas, numpy
- **Machine Learning:** scikit-learn, scipy
- **Visualization:** matplotlib, seaborn, plotly
- **Utilities:** requests, tqdm, python-dotenv

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Huzaib Shafi**
- GitHub: [@HuzaibShafi](https://github.com/HuzaibShafi)
- Project: [Movie Recommendation System](https://github.com/HuzaibShafi/recommendation_system)

## 🙏 Acknowledgments

- MovieLens dataset for providing the movie rating data
- Scikit-learn for machine learning algorithms
- Plotly for interactive visualizations
- The open-source community for inspiration and tools

---

**Happy recommending! 🎬✨** 
