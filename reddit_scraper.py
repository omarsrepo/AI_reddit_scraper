import os
import praw
import ollama
import textwrap
from dotenv import load_dotenv
from transformers import pipeline  
from datetime import datetime, timedelta, timezone
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

# Keywords we're looking for (You can modify this list to include other relevant terms for a different search purpose)
KEYWORDS = [
    "esim", "international roaming", "travel connectivity", 
    "bnesim", "airalo", "gigsky", "holafly", "ubigi", "flexiroam",
    "travel SIM recommendations", "roaming charges", "connectivity issue", 
    "sim not working", "bad mobile service abroad"
]

# Used to get the current time in UTC with timezone information
def utcnow():
    return datetime.now(timezone.utc)

# Ensure the terminal is cleared at the start
def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')


# SentenceTransformer model for semantic similarity
model = SentenceTransformer("all-MiniLM-L6-v2")
keyword_embeddings = model.encode(KEYWORDS, convert_to_tensor=True)
def is_semantically_relevant(post_text, threshold=0.6):
    """
    Determines whether a Reddit post is semantically relevant to any predefined keywords 
    using cosine similarity between sentence embeddings.

    Args:
        post_text (str): The text content of the Reddit post (title + body).
        threshold (float): The minimum cosine similarity score required for the post 
                           to be considered relevant to a keyword.

    Returns:
        bool: True if the post is semantically similar to at least one keyword 
              (similarity > threshold), False otherwise.

    Note:
        - This function assumes `model` is a preloaded sentence-transformers model.
        - `keyword_embeddings` must be a precomputed list of embeddings 
          (same model used for encoding).
    """
    post_embedding = model.encode(post_text, convert_to_tensor=True)

    # Compute cosine similarities with each keyword
    similarities = util.cos_sim(post_embedding, keyword_embeddings)[0]

    # Check if any similarity is above the threshold
    return any(score > threshold for score in similarities)


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


def search_reddit_by_keywords(keywords, days_back=7, limit=100):
    raw_posts = []
    relevant_posts = []
    cutoff = utcnow() - timedelta(days=days_back)
    seen_urls = set()

    for keyword in keywords:
        print(f"Searching for: {keyword}")
        results = reddit.subreddit("all").search(
            query=keyword,
            sort="new",
            time_filter="week",
            limit=limit
        )

        for post in results:
            url = f"https://www.reddit.com{post.permalink}"
            post_time = datetime.fromtimestamp(post.created_utc, tz=timezone.utc)
            if url in seen_urls or post_time < cutoff:
                continue
            seen_urls.add(url)

            raw_posts.append({
                "title": post.title,
                "content": post.selftext,
                "url": url,
                "subreddit": str(post.subreddit),
                "upvotes": post.score,
                "comments": post.num_comments,
                "created_utc": post_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "full_text": post.title + " " + post.selftext
            })

    print(f"\nDeduplicated to {len(raw_posts)} unique posts. Now filtering semantically\n")

    # Batch encode all post texts once
    all_texts = [post["full_text"] for post in raw_posts]
    post_embeddings = model.encode(all_texts, convert_to_tensor=True)

    # Compute cosine similarities in one go
    cosine_scores = util.cos_sim(post_embeddings, keyword_embeddings)

    # Threshold filtering
    for idx, post in enumerate(raw_posts):
        similarities = cosine_scores[idx]
        if any(score > 0.6 for score in similarities):
            post["context"] = classify_post_context(post["full_text"])
            relevant_posts.append(post)

    return relevant_posts


if __name__ == "__main__":
    clear_terminal()
    
    posts = search_reddit_by_keywords(KEYWORDS, days_back=7, limit=100)
    print(f"\nFound {len(posts)} relevant posts in total:\n")
    
    BLUE = "\033[94m"
    RESET = "\033[0m"
    
    i = 1
    for post in posts:
        print(f"Post {i} of {len(posts)}")
        print(f"Title: {post['title']}")
        print(f"Subreddit: {post['subreddit']}")
        print(f"URL: {BLUE}{post['url']}{RESET}")
        print(f"Date: {post['created_utc']}")
        print(f"Content: {post['content'][:200]}...")
        print(f"Engagement Metrics: {post['upvotes']} upvotes, {post['comments']} comments")
        print(f"Context Assessment: {post['context']}")
        response = ollama.chat(model='redditor', messages=[
                {
                    'role': 'user',
                    'content': post["full_text"]
                },
                ])
        post["response"] = textwrap.fill(response.message.content, width=150)
        print(f"Possible Response: {post["response"]}\n")
        i += 1
    