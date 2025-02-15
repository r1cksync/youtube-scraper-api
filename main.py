from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware
from pydantic import BaseModel
import requests
import time
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_RECENT
import os

app = FastAPI(title="YouTube Comments Scraper API with Sentiment Analysis")

# ✅ Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, change this to your frontend URL if needed
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Hugging Face API configuration
HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "")
HF_API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment"
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

def analyze_sentiment(text, max_retries=5):
    payload = {"inputs": text}
    for attempt in range(max_retries):
        response = requests.post(HF_API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            if result and isinstance(result[0], dict):
                return result[0].get("label")
            elif result and isinstance(result[0], list) and len(result[0]) > 0:
                return result[0][0].get("label")
            else:
                raise Exception("Unexpected result format: " + str(result))
        else:
            error_data = response.json()
            if "currently loading" in error_data.get("error", ""):
                time.sleep(error_data.get("estimated_time", 20))
            else:
                raise Exception(f"API call failed: {response.text}")
    raise Exception("Model did not load after several attempts.")

# Define the request model
class VideoURL(BaseModel):
    url: str

@app.post("/comments/")
def get_comments(video: VideoURL):
    try:
        downloader = YoutubeCommentDownloader()
        comments = downloader.get_comments_from_url(video.url, sort_by=SORT_BY_RECENT)
        
        data = []
        for comment in comments:
            comment_text = comment.get('text', '')
            likes = comment.get('votes', 0)
            try:
                sentiment = analyze_sentiment(comment_text)
            except Exception as e:
                sentiment = "Error"
            
            data.append({
                "text": comment_text,
                "sentiment": sentiment,
                "votes": likes,
                "hearted": comment.get('heart', False),
                "replies": comment.get('reply_count', 0) or (1 if comment.get('reply', False) else 0),
                "time": comment.get('time', "Unknown")
            })
        
        return {"comments": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))