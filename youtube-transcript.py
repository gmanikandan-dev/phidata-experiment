from fastapi import FastAPI, Query
from youtube_transcript_api import YouTubeTranscriptApi
import uvicorn
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

app = FastAPI()

# Load your GROQ API key and endpoint from environment variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

def get_youtube_transcript(video_id: str):
    """Fetch transcript from YouTube video."""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([entry['text'] for entry in transcript])
    except Exception as e:
        return f"Error: {str(e)}"

def summarize_with_groq(text: str):
    """Summarize transcript using Groq API."""
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",  # Choose a Groq-supported model
            messages=[{"role": "user", "content": f"Summarize the following text:\n\n{text}"}]
        )
        return response.choices[0].message.content if response.choices else "No response"
    except Exception as e:
        return f"Groq API Error: {str(e)}"

@app.get("/transcribe/")
def transcribe_video(video_url: str, summarize: bool = Query(False)):
    """Extract and summarize YouTube video transcript."""
    video_id = video_url.split("v=")[-1].split("&")[0]  # Extract video ID
    transcript = get_youtube_transcript(video_id)

    if "Error" in transcript:
        return {"error": transcript}

    summary = summarize_with_groq(transcript) if summarize else None
    return {"video_id": video_id, "transcript": transcript, "summary": summary}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
