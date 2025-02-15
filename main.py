from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import time
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_RECENT

app = FastAPI(title="YouTube Comments Scraper API with Sentiment Analysis")

# Hugging Face Inference API configuration for sentiment analysis
import os
HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "")

HF_API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment"
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

def analyze_sentiment(text, max_retries=5):
    """
    Sends text to the Hugging Face Inference API and returns the sentiment label.
    If the model is loading, it waits for the estimated time before retrying.
    """
    payload = {"inputs": text}
    for attempt in range(max_retries):
        response = requests.post(HF_API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            # Check if the result is in the expected format
            if result and isinstance(result[0], dict):
                sentiment = result[0].get("label")
                return sentiment
            elif result and isinstance(result[0], list) and len(result[0]) > 0:
                sentiment = result[0][0].get("label")
                return sentiment
            else:
                raise Exception("Unexpected result format: " + str(result))
        else:
            error_data = response.json()
            error_message = error_data.get("error", "")
            if "currently loading" in error_message:
                estimated_time = error_data.get("estimated_time", 20)
                print(f"Model is loading, waiting for {estimated_time} seconds... (attempt {attempt+1})")
                time.sleep(estimated_time)
            else:
                raise Exception(f"API call failed: {response.text}")
    raise Exception("Model did not load after several attempts.")

# Define the request model
class VideoURL(BaseModel):
    url: str

@app.post("/comments/")
def get_comments(video: VideoURL):
    """
    Fetches YouTube comments for a given video URL and returns them along with sentiment analysis.
    """
    try:
        # Initialize the YouTube comment downloader
        downloader = YoutubeCommentDownloader()
        # Retrieve comments sorted by recent
        comments = downloader.get_comments_from_url(video.url, sort_by=SORT_BY_RECENT)
        
        data = []
        for comment in comments:
            comment_text = comment.get('text', '')
            likes = comment.get('votes', 0)
            try:
                # Analyze sentiment for each comment
                sentiment = analyze_sentiment(comment_text)
            except Exception as e:
                sentiment = "Error"
                print(f"Error analyzing comment: {e}")
            
            # Append the comment data along with sentiment
            data.append({
                "text": comment_text,
                "sentiment": sentiment,
                "votes": likes,
                "hearted": comment.get('heart', False),
                "replies": comment.get('reply_count', 0) or (1 if comment.get('reply', False) else 0),
                "time": comment.get('time', "Unknown")
            })
        
        # Return the data as JSON
        return {"comments": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
