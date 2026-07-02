# ResearchMind · AI Research Agent

Multi-agent pipeline (Streamlit UI) that searches the web, scrapes sources, writes a research report, and critiques it.

## 1) Setup

### Install dependencies
```bash
pip install -r requirements.txt
```

### Environment variables
Create a `.env` file in the project root (same folder as `requirements.txt`) with:

```bash
MISTRAL_API_KEY=YOUR_MISTRAL_API_KEY
TAVILY_API_KEY=YOUR_TAVILY_API_KEY

# Optional (defaults shown)
MISTRAL_MODEL=mistral-small-latest
MISTRAL_TIMEOUT_SECS=60

TAVILY_MAX_RESULTS=5
TAVILY_SNIPPET_MAX_CHARS=300

SCRAPE_TIMEOUT_SECS=10
SCRAPE_MAX_CHARS=3000
```

## 2) Start commands

### A) Run the Streamlit app (recommended)
```bash
streamlit run src/app.py
```

(Optional) If `streamlit` is not on PATH:
```bash
python -m streamlit run src/app.py
```

### B) Run the FastAPI backend (required by the Streamlit app)
```bash
uvicorn src.api:app --host 0.0.0.0 --port $PORT
```

(If you want a fixed local port instead)
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### C) Run the pipeline from the command line (console input)
```bash
python src/pipeline.py
```

## Notes
- The app uses Streamlit (`src/app.py`) and requires `MISTRAL_API_KEY` and `TAVILY_API_KEY`.
- `src/pipeline.py` is a simple interactive CLI runner (prompts for a topic).
