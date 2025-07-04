import os
import praw
from dotenv import load_dotenv
from transformers import pipeline  
from datetime import datetime, timezone
from sentence_transformers import SentenceTransformer, util

# Load environment variables from .env file
load_dotenv()

# Reddit API credentials
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")

# Initialize Reddit API
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT,
)

# Used to get the current time in UTC with timezone information
def utcnow():
    return datetime.now(timezone.utc)

# Ensure the terminal is cleared at the start
def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')


# Zero-shot classifier model for context categorization of posts
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
CATEGORIES = ["question", "complaint", "recommendation request", "praise", "other"]
def classify_post_context(post_text):
    """
    Uses zero-shot classification to categorize Reddit post context.
    
    Args:
        post_text (str): Combined title + content of a Reddit post.
        
    Returns:
        str: One of the predefined context categories.
    """
    try:
        result = classifier(post_text, CATEGORIES)
        return result["labels"][0]  # Return the top label
    except Exception as e:
        print(f"Error during classification: {e}")
        return "unknown"
    