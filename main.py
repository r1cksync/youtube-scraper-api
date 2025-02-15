from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_RECENT

# Initialize FastAPI app
app = FastAPI(title="YouTube Comments Scraper API")

# Pydantic model for the request payload
class VideoURL(BaseModel):
    url: str

@app.post("/comments/")
def get_comments(video: VideoURL):
    """
    Fetches YouTube comments for a given video URL.
    """
    try:
        # Initialize the YouTube comment downloader
        downloader = YoutubeCommentDownloader()
        
        # Retrieve comments sorted by recent
        comments = downloader.get_comments_from_url(video.url, sort_by=SORT_BY_RECENT)
        
        # Process and store comments in a list (customize fields as needed)
        data = []
        for comment in comments:
            data.append({
                "text": comment.get('text', ''),
                "votes": comment.get('votes', 0),
                "hearted": comment.get('heart', False),
                "replies": comment.get('reply_count', 0) or (1 if comment.get('reply', False) else 0),
                "time": comment.get('time', "Unknown")
            })
        
        return {"comments": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 