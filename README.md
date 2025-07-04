# AI powered Reddit Post Discovery Tool

A powerful, AI-enhanced Reddit monitoring tool that finds **semantically relevant posts** across Reddit using **natural language similarity**. This tool uses semantic similarity (via NLP embeddings) to discover posts that are meaningfully relevant.
Designed for marketers, researchers, community managers, and developers who want to track conversations that matter.

<img width="1680" alt="Prototype Search Results" src="https://github.com/user-attachments/assets/76b0b7ef-a46c-4f33-b461-26294723e267" />


---

## Features

- **Searches Reddit** for posts from the past 7 days (or configurable window)
- Accepts **custom keyword lists** — track any product, topic, brand, or phrase
- Uses **all-MiniLM-L6-v2** Huggingface transformer to perform cosine similarity between posts content and list of keywords to detect relevance
- Fast filtering using **precomputed embeddings**
- Uses **bart-large-mnli** Huggingface transformer for zero-shot classification of context assesment of posts (is the post a complaint, question, feedback etc)
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
├── requirements.txt 
├── Modelfile (For creating custom models. More info here: https://ollama.readthedocs.io/en/modelfile/ )
└── README.md 
```

## Installation
```bash
git clone https://github.com/your-username/reddit-discovery-tool.git
cd reddit-discovery-tool
pip install -r requirements.txt
```
Add a .env file with your Reddit credentials:

```
REDDIT_CLIENT_ID=your_id
REDDIT_CLIENT_SECRET=your_secret
REDDIT_USER_AGENT=your_agent_string
```

## License
MIT
