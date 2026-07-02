import os
import sys

from fastapi import FastAPI
from pydantic import BaseModel

# Ensure we can import sibling modules when running via `uvicorn src.api:app`
sys.path.append(os.path.dirname(__file__))

from pipeline import run_research_pipeline

app = FastAPI()

class ResearchRequest(BaseModel):
    topic: str

@app.post("/research")
def research(request: ResearchRequest):
    result = run_research_pipeline(request.topic)

    return {
        "success": True,
        "topic": request.topic,
        "search_results": result["search_results"],
        "scraped_content": result["scraped_content"],
        "report": result["writer"],
        "review": result["critic"],
    }
