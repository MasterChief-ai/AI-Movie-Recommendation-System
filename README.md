# AI-Movie-Recommendation-System
The AI Movie Recommendation System is a Netflix-inspired intelligent movie suggestion engine that leverages both content-based filtering and collaborative filtering to deliver personalized movie recommendations. It analyzes a user's liked movie, search preferences, and historical data to predict and rank the most relevant movies across genres.
#🔍 Key Features
Hybrid AI Model: Combines content-based filtering (via movie genres, keywords, directors, and actors) with collaborative filtering (via user ratings and behaviors).

Sleek UI Design: Fully responsive, Netflix-style dark-themed UI built with vanilla HTML, CSS, and JavaScript.

Live API Integration: Connects directly to a FastAPI backend powered by Python and scikit-learn, returning real-time recommendations.

Interactive Input: Accepts user ID, liked movie, and keyword filters for dynamic recommendation results.

Enhanced UX: Includes smooth transitions, animations, and polished styles for a premium experience.

#🛠️ Tech Stack
Frontend

HTML5 + CSS3 + JavaScript (No framework)

Google Fonts (Inter)

Netflix-style animations and responsiveness

Backend

Python 3.11+

FastAPI

Scikit-learn, Pandas, NumPy

IMDb and MovieLens datasets (combined for hybrid intelligence)

#📦 Datasets Used
IMDb 5000 Movie Dataset – for content-based filtering (actors, genres, director)
MovieLens 100k Ratings – for collaborative filtering (user-item matrix)

#🚀 How to Run
#Backend (FastAPI)
uvicorn main:app --reload
Runs on http://127.0.0.1:8000/docs
#Frontend
Just open index.html in any browser, or serve via Vite/dev server if needed.
