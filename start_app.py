#!/usr/bin/env python3
"""
Simple script to start the TMDB Movie Recommendation System Web App
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from app import app
    print("🎬 TMDB Movie Recommendation System Web App")
    print("✅ Flask app loaded successfully")
    print("🌐 Starting web server on port 5001...")
    print("📱 Open your browser and go to: http://127.0.0.1:5001")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5001)
    
except Exception as e:
    print(f"❌ Error starting the application: {str(e)}")
    print("Please check if all dependencies are installed:")
    print("pip install -r requirements.txt") 