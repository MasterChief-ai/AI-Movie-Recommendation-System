from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from recommender import hybrid_recommend

app = FastAPI(
    title="Hybrid Movie Recommender API",
    description="A hybrid movie recommendation engine combining content-based, collaborative, and search-based filtering.",
    version="1.0"
)

# ----------------------------
# CORS Middleware for frontend access
# ----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Welcome endpoint
# ----------------------------
@app.get("/")
def root():
    return {"message": "🎬 Welcome to the Movie Recommender API"}

# ----------------------------
# Request schema using Pydantic
# ----------------------------
class RecommendRequest(BaseModel):
    user_id: int
    keywords: List[str] = []
    liked_movie: Optional[str] = None
    top_n: int = 10

# ----------------------------
# POST endpoint for recommendations
# ----------------------------
@app.post("/recommend")
def recommend_movies(request: RecommendRequest):
    """
    Get hybrid movie recommendations based on:
    - Collaborative filtering (via user_id)
    - Content-based filtering (via liked_movie)
    - Search-based filtering (via keywords)
    """
    try:
        results = hybrid_recommend(
            user_id=request.user_id,
            search_keywords=request.keywords,
            liked_movie_title=request.liked_movie,
            top_n=request.top_n
        )
        return {
            "user_id": request.user_id,
            "liked_movie": request.liked_movie,
            "keywords": request.keywords,
            "recommendations": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")
#For Running: python -m uvicorn main:app --reload

