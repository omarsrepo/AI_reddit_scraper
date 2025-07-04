# AI powered Reddit Post Discovery Tool

A powerful, AI-enhanced Reddit monitoring tool that finds **semantically relevant posts** across Reddit using **natural language similarity**. This tool uses semantic similarity (via NLP embeddings) to discover posts that are meaningfully relevant.
Designed for marketers, researchers, community managers, and developers who want to track conversations that matter.

<img width="1680" alt="Prototype Search Results" src="https://github.com/user-attachments/assets/76b0b7ef-a46c-4f33-b461-26294723e267" />


---

## Features

- **Searches Reddit** for posts from the past 7 days (or configurable window)
- Accepts **custom keyword lists** — track any product, topic, brand, or phrase
- Uses `all-MiniLM-L6-v2` Huggingface transformer to perform cosine similarity between posts content and list of keywords to detect relevance
- Fast filtering using **precomputed embeddings**
- Uses `bart-large-mnli` Huggingface transformer for zero-shot classification of context assesment of posts (is the post a complaint, question, feedback etc)
- Outputs relevant posts with metadata: title, content preview, date, subreddit, link
- Uses Ollama and llama3.2 to create a custom LLM through the Modelfile (provided in the project structure) for custom response generation to the posts 

---

## How It Works

1. **Input**: You provide a list of keywords or phrases you'd like to monitor.
2. **Embedding**: The tool uses a [`sentence-transformers`](https://www.sbert.net/) model (`all-MiniLM-L6-v2`) to compute sentence embeddings for both:
   - Your keyword list
   - Each Reddit post's title + body
3. **Semantic Filtering**: 
   - Calculates cosine similarity between post embeddings and keyword embeddings
   - Only keeps posts with a similarity **above a threshold** (default: `0.6`)
4. **Output**: Shows posts in the terminal with key info and a snippet of content.

---

## Project Structure
```
Reddit_discovery_tool/
├── .env 
├── reddit_scraper.py
├── main.py 
├── requirements.txt 
├── Modelfile (For creating custom models. More info here: https://ollama.readthedocs.io/en/modelfile/ )
└── README.md 
```

## Installation and Usage
1. Clone this repository onto your machine and install all the requirements into ur virtual environment.
```bash
git clone https://github.com/your-username/reddit-discovery-tool.git
cd reddit-discovery-tool
pip install -r requirements.txt
```
2. Create a .env file and add your Reddit credentials (can be obtained by creating an app here: https://www.reddit.com/prefs/apps):
```
REDDIT_CLIENT_ID=your_id
REDDIT_CLIENT_SECRET=your_secret
REDDIT_USER_AGENT=your_agent_string
```
3. Install Ollama and any ollama model you want (Im using llama3.2 for this example)
```bash
pip install ollama
ollama serve
ollama list # Lists the currently installed models
ollama install llama3.2
```
4. In the Modelfile, edit the "system prompt" and "from" parameter to your liking
```
# Modelfile
FROM llama3.2 (Change this to whatever model you wish to use)

# System Prompt (Edit this to suit your personal usecase)
SYSTEM """
  You're a helpful travel advisor on Reddit. Provide honest, casual, and non-salesy suggestions. Write a helpful, human-sounding Reddit comment that:
    - Directly answers the user's question or concern
    - Is friendly and useful 
    - Naturally mention the sim card provider 'BNESIM' if it fits the situation, but without sounding like an ad
    - Uses a casual Redditor tone
    - Do NOT sound robotic or overly polished
    - Write as if you’re replying in the comments thread.
    """

PARAMETER temperature 0.7
PARAMETER top_p 0.95
PARAMETER num_predict 200
```
5. Create a custom model for use
```bash
ollama create choose-a-model-name -f <location of the file e.g. ./Modelfile>'
```
6. Open up the main.py file. Modify the list of keywords for the kind of posts you are looking for and set the DAYS_BACK integer to set the range of days u want to search in (default is last 7 days).
```python
# Set this to the number of days you want to look back for posts (default is as receent as 7 days)
DAYS_BACK = 7 
# Set this list of keywords to search for in Reddit posts
KEYWORDS = [
    "esim", "international roaming", "travel connectivity", 
    "bnesim", "airalo", "gigsky", "holafly", "ubigi", "flexiroam",
    "travel SIM recommendations", "roaming charges", "connectivity issue", 
    "sim not working", "bad mobile service abroad"
]
```
7. Finally, run the main.py file and watch the results!
8. Optional: Comment out the following section in my main.py file if you do not want to generate responses for the posts and only want to search for them.
```python
response = ollama.chat(model='redditor', messages=[
         {
            'role': 'user',
            'content': post["full_text"]
         },
         ])
post["response"] = textwrap.fill(response.message.content, width=150)
print(f"Possible Response: {post['response']}\n")
#
```
 
## License
MIT
