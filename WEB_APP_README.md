# 🎬 TMDB Movie Recommendation System - Web Application

A modern, user-friendly web interface for the TMDB movie recommendation system built with Flask and Bootstrap.

## 🌟 Features

### 🎯 **Core Functionality**
- **Interactive Recommendations**: Get personalized movie recommendations using 5 different algorithms
- **Real-time Search**: Search through 4,800+ movies with instant results
- **Data Exploration**: Interactive charts and statistics about the dataset
- **Movie Details**: Comprehensive movie information with similar movie suggestions

### 🎨 **User Interface**
- **Modern Design**: Clean, responsive design with Bootstrap 5
- **Interactive Elements**: Hover effects, animations, and smooth transitions
- **Mobile Responsive**: Works perfectly on all device sizes
- **Loading States**: Professional loading indicators and error handling

### 📊 **Data Visualization**
- **Genre Distribution**: Interactive pie chart showing movie genres
- **Rating Distribution**: Bar chart displaying user rating patterns
- **Popular Movies**: Top-rated movies with detailed information
- **Sample Movies**: Random selection from the dataset

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- TMDB dataset files in `data/raw/`:
  - `tmdb_5000_movies.csv`
  - `tmdb_5000_credits.csv`

### Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Web Application**:
   ```bash
   python app.py
   ```

3. **Access the Application**:
   Open your browser and go to: `http://localhost:5000`

## 📁 Project Structure

```
recommendation_system/
├── app.py                          # Main Flask application
├── templates/                      # HTML templates
│   ├── base.html                   # Base template with navigation
│   ├── index.html                  # Home page
│   ├── recommendations.html        # Recommendations interface
│   ├── explore.html                # Data exploration page
│   ├── 404.html                    # 404 error page
│   └── 500.html                    # 500 error page
├── static/                         # Static files
│   ├── css/
│   │   └── style.css              # Custom CSS styles
│   └── js/
│       └── main.js                # Main JavaScript file
├── src/                           # Backend modules
│   ├── data/
│   │   └── tmdb_processor.py      # TMDB data processor
│   ├── models/
│   │   └── recommendation_models.py # Recommendation algorithms
│   ├── utils/
│   │   └── evaluation.py          # Evaluation utilities
│   └── visualization/
│       └── plots.py               # Visualization utilities
└── data/                          # Data files
    ├── raw/                       # Raw TMDB data
    └── processed/                 # Processed data files
```

## 🎯 Pages Overview

### 🏠 **Home Page** (`/`)
- Hero section with call-to-action buttons
- Feature overview of recommendation algorithms
- Dataset statistics and information
- Quick start guide

### 🎬 **Recommendations Page** (`/recommendations`)
- **User Selection**: Choose from available users
- **Algorithm Selection**: 5 different recommendation algorithms
- **Results Display**: Ranked list of recommended movies
- **Movie Details**: Click to view comprehensive movie information

### 📊 **Data Exploration Page** (`/explore`)
- **Statistics Cards**: Key metrics about the dataset
- **Interactive Charts**: Genre and rating distributions
- **Popular Movies**: Top-rated movies from the dataset
- **Sample Movies**: Random selection with details

## 🔧 API Endpoints

### Recommendations
- `POST /api/recommendations` - Get movie recommendations
  ```json
  {
    "user_id": 1,
    "algorithm": "Hybrid",
    "n_recommendations": 10
  }
  ```

### Movie Search
- `GET /api/movie_search?q=query` - Search movies by title

### Movie Details
- `GET /api/movie_details/<movie_id>` - Get detailed movie information

### Statistics
- `GET /api/stats` - Get dataset statistics and charts data

## 🎨 Design Features

### **Color Scheme**
- **Primary**: Blue gradient (#007bff → #0056b3)
- **Success**: Green (#28a745)
- **Warning**: Yellow (#ffc107)
- **Info**: Light blue (#17a2b8)
- **Dark**: Dark gray (#343a40)

### **Typography**
- **Font Family**: Segoe UI, Tahoma, Geneva, Verdana, sans-serif
- **Headings**: Bold weights with proper hierarchy
- **Body Text**: Clean, readable font sizes

### **Animations**
- **Hover Effects**: Cards lift on hover
- **Fade-in Animations**: Elements animate in as they come into view
- **Loading Spinners**: Professional loading indicators
- **Smooth Transitions**: All interactive elements have smooth transitions

## 📱 Responsive Design

### **Breakpoints**
- **Mobile**: < 576px
- **Tablet**: 576px - 768px
- **Desktop**: > 768px

### **Mobile Features**
- Collapsible navigation menu
- Touch-friendly buttons and forms
- Optimized card layouts
- Responsive charts and tables

## 🔒 Error Handling

### **User-Friendly Error Pages**
- **404 Page**: Custom "Page Not Found" with helpful navigation
- **500 Page**: Server error page with retry functionality
- **Form Validation**: Client-side and server-side validation
- **API Error Handling**: Graceful error messages for API failures

## 🚀 Performance Features

### **Optimization**
- **Lazy Loading**: Images and content load as needed
- **Debounced Search**: Search queries are debounced to reduce API calls
- **Caching**: Recommendation results are cached for better performance
- **Minified Assets**: CSS and JS are optimized for production

### **Loading States**
- **Skeleton Screens**: Placeholder content while loading
- **Progress Indicators**: Spinners and progress bars
- **Error Recovery**: Automatic retry mechanisms

## 🛠️ Development

### **Running in Development Mode**
```bash
export FLASK_ENV=development
python app.py
```

### **Debug Mode**
The application runs in debug mode by default, providing:
- Detailed error messages
- Auto-reload on code changes
- Interactive debugger

### **Customization**
- **Styling**: Modify `static/css/style.css`
- **Functionality**: Update `static/js/main.js`
- **Templates**: Edit files in `templates/`
- **Backend**: Modify `app.py` and modules in `src/`

## 📊 Browser Support

### **Supported Browsers**
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

### **Required Features**
- ES6 JavaScript support
- CSS Grid and Flexbox
- Fetch API
- Intersection Observer API

## 🔧 Configuration

### **Environment Variables**
```bash
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=your-secret-key
```

### **Data Configuration**
- Place TMDB dataset files in `data/raw/`
- Processed data will be saved in `data/processed/`
- Models are trained automatically on first run

## 🎯 Usage Examples

### **Getting Recommendations**
1. Navigate to `/recommendations`
2. Select a user from the dropdown
3. Choose an algorithm (e.g., "Hybrid")
4. Set number of recommendations
5. Click "Get Recommendations"
6. View ranked list of movies

### **Exploring Data**
1. Navigate to `/explore`
2. View dataset statistics
3. Interact with charts
4. Browse popular movies
5. Click on movies for details

### **Searching Movies**
1. Use the search functionality
2. Type movie title
3. View instant results
4. Click for detailed information

## 🎉 Success Metrics

### **User Experience**
- **Load Time**: < 3 seconds for initial page load
- **Responsiveness**: Works on all device sizes
- **Accessibility**: WCAG 2.1 AA compliant
- **Error Rate**: < 1% of requests result in errors

### **Performance**
- **Recommendation Speed**: < 2 seconds for results
- **Search Response**: < 500ms for search results
- **Chart Rendering**: < 1 second for data visualization

## 🤝 Contributing

### **Code Style**
- Follow PEP 8 for Python code
- Use consistent JavaScript formatting
- Maintain responsive design principles
- Add comments for complex logic

### **Testing**
- Test on multiple browsers
- Verify mobile responsiveness
- Check error handling
- Validate form submissions

## 📄 License

This project is part of the TMDB Movie Recommendation System.
Created by Huzaib Shafi.

## 🙏 Acknowledgments

- **TMDB**: For providing the comprehensive movie dataset
- **Flask**: For the excellent web framework
- **Bootstrap**: For the responsive UI components
- **Chart.js**: For interactive data visualization
- **Font Awesome**: For beautiful icons

---

**🎬 Ready to discover your next favorite movie? Start the application and explore the world of personalized recommendations!** 