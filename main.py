from reddit_scraper import utcnow, clear_terminal, reddit, util, classify_post_context, SentenceTransformer
import textwrap, ollama
from datetime import datetime, timedelta, timezone

# Set this to the number of days you want to look back for posts (default is as receent as 7 days)
DAYS_BACK = 7 
# Set this list of keywords to search for in Reddit posts
KEYWORDS = [
    "esim", "international roaming", "travel connectivity", 
    "bnesim", "airalo", "gigsky", "holafly", "ubigi", "flexiroam",
    "travel SIM recommendations", "roaming charges", "connectivity issue", 
    "sim not working", "bad mobile service abroad"
]

model = SentenceTransformer("all-MiniLM-L6-v2")
keyword_embeddings = model.encode(KEYWORDS, convert_to_tensor=True)

def search_reddit_by_keywords(keywords, days_back=DAYS_BACK, limit=100):
    """
    Searches Reddit for posts containing specified keywords, filters them based on semantic relevance,
    and returns a list of relevant posts with additional metadata.
    Args:
        keywords (list): List of keywords to search for.
        days_back (int): Number of days to look back for posts.
        limit (int): Maximum number of posts to retrieve per keyword.
    Returns:
        list: A list of dictionaries containing relevant posts and their metadata.
    """
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

    #Filtering whether a Reddit post is semantically relevant to any predefined keywords using cosine similarity between sentence and keyword embeddings
    all_texts = [post["full_text"] for post in raw_posts]
    post_embeddings = model.encode(all_texts, convert_to_tensor=True)
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

        # Comment this section out if you don't want to generate AI responses
        response = ollama.chat(model='redditor', messages=[
                {
                    'role': 'user',
                    'content': post["full_text"]
                },
                ])
        post["response"] = textwrap.fill(response.message.content, width=150)
        print(f"Possible Response: {post['response']}\n")
        #
        i += 1

