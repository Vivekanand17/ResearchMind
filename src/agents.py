

from __future__ import annotations

import os
from typing import Any, Dict, Optional, TypeVar, cast
import os

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)

from langchain.agents import create_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

# Support both:
# - running from repo root with `uvicorn src.api:app` (sys.path hack adds src/)
# - importing as a package in some environments
try:
    from tools import scrape_url, web_search
except ModuleNotFoundError:  # pragma: no cover
    from src.tools import scrape_url, web_search

load_dotenv()

T = TypeVar("T")


class ResearchPipelineError(RuntimeError):
    """Raised for controlled, user-facing pipeline failures."""


def _truncate(text: str, *, max_chars: int) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[TRUNCATED]"


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ResearchPipelineError(
            f"Missing required environment variable: {name}. "
            "Set it in a .env file or system environment and restart the app."
        )
    return value


def _is_truthy_env(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


def _make_llm() -> ChatMistralAI:
  
    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets["MISTRAL_API_KEY"]
        except Exception:
            api_key = None

    if not api_key:
        raise ResearchPipelineError(
            
        )

    model = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
    timeout_secs = int(os.getenv("MISTRAL_TIMEOUT_SECS", "60"))

    return ChatMistralAI(
        model=model,
        temperature=0,
        api_key=api_key,
        timeout=timeout_secs,
    )

# -------- Rate limit + transient error handling --------
try:
    from httpx import HTTPStatusError, ConnectError, TimeoutException
except Exception:  # pragma: no cover
    HTTPStatusError = None  # type: ignore
    ConnectError = None  # type: ignore
    TimeoutException = None  # type: ignore


def _is_rate_limited(exc: BaseException) -> bool:
    """
    Detect HTTP 429 from common Mistral/httpx error surfaces.
    """
    if HTTPStatusError is not None and isinstance(exc, HTTPStatusError):
        try:
            return getattr(exc.response, "status_code", None) == 429
        except Exception:
            return False

    # Fallback: search in message
    msg = str(exc).lower()
    return "429" in msg and "rate" in msg


def _should_retry(exc: BaseException) -> bool:
    """
    Retry on rate-limit and a small set of transient network errors.
    """
    if _is_rate_limited(exc):
        return True

    # Optional retry on transient transport errors
    if ConnectError is not None and isinstance(exc, ConnectError):
        return True
    if TimeoutException is not None and isinstance(exc, TimeoutException):
        return True

    # Common string-based fallbacks
    msg = str(exc).lower()
    if "timeout" in msg or "temporarily unavailable" in msg or "connection reset" in msg:
        return True

    return False


def _get_max_retries() -> int:
    return max(1, int(os.getenv("MISTRAL_MAX_RETRIES", "6")))


def _get_retry_min_seconds() -> float:
    return float(os.getenv("MISTRAL_RETRY_MIN_SECS", "2"))


def _get_retry_max_seconds() -> float:
    return float(os.getenv("MISTRAL_RETRY_MAX_SECS", "60"))


def _get_disable_retries() -> bool:
    return _is_truthy_env("MISTRAL_DISABLE_RETRIES", "false")


@retry(
    wait=wait_exponential(
        multiplier=2,
        min=_get_retry_min_seconds(),
        max=_get_retry_max_seconds(),
    ),
    stop=stop_after_attempt(_get_max_retries()),
    reraise=True,
)
def _invoke_with_backoff(fn: Any) -> Any:
    """
    tenacity wrapper. We raise again only when _should_retry says so.
    """
    try:
        return fn()
    except Exception as exc:
        if _should_retry(exc):
            raise
        raise


def _safe_invoke(runnable: Any, inputs: Dict[str, Any]) -> Any:
    """
    Safely invoke a LangChain runnable with transient backoff retries.
    This is used for writer/critic chains and can also be used for agents.
    """
    if _get_disable_retries():
        try:
            return runnable.invoke(inputs)
        except Exception as exc:
            raise ResearchPipelineError(f"Invocation failed: {exc}") from exc

    try:
        return _invoke_with_backoff(lambda: runnable.invoke(inputs))
    except Exception as exc:
        if _is_rate_limited(exc):
            raise ResearchPipelineError("Rate limit exceeded (Mistral 429). Please try again shortly.") from exc
        raise ResearchPipelineError(f"Invocation failed: {exc}") from exc


def safe_invoke_raw(runnable: Any, inputs: Dict[str, Any]) -> Any:
    """
    Invoke a LangChain runnable and return the raw result.
    - Retries/backoff on transient errors (including Mistral 429)
    - Does NOT attempt to parse/extract output into a string
    """
    return _safe_invoke(runnable, inputs)


def safe_agent_invoke(runnable: Any, inputs: Dict[str, Any], *, step_name: str) -> str:
    """
    Invoke an agent/runnable and always return a string.
    Raises ResearchPipelineError only after retries/backoff fail.
    """
    out = _safe_invoke(runnable, inputs)
    text = str(out or "").strip()
    if not text:
        raise ResearchPipelineError(f"{step_name} returned empty output.")
    return text


# -------- Writer/Critic Chains --------
def _writer_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert research writer. Write clear, structured and insightful reports.",
            ),
            (
                "human",
                """Write a detailed research report on the topic below.

Topic:
{topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional.
""",
            ),
        ]
    )


def _critic_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a sharp and constructive research critic. Be honest and specific.",
            ),
            (
                "human",
                """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...

Areas to Improve:
- ...

One line verdict:
...
""",
            ),
        ]
    )


# Build chains lazily to avoid import-time env failures.
_writer_chain: Optional[Any] = None
_critic_chain: Optional[Any] = None


def _get_writer_chain() -> Any:
    global _writer_chain
    if _writer_chain is None:
        llm = _make_llm()
        _writer_chain = _writer_prompt() | llm | StrOutputParser()
    return _writer_chain


def _get_critic_chain() -> Any:
    global _critic_chain
    if _critic_chain is None:
        llm = _make_llm()
        _critic_chain = _critic_prompt() | llm | StrOutputParser()
    return _critic_chain


# Public symbols expected by pipeline.py / app.py (kept for compatibility).
# Initialize at import-time so callers can safely use `.invoke(...)`.
# If env vars are missing, this will raise a controlled `ResearchPipelineError`.
class LazyWriterChain:
    def invoke(self, inputs):
        return _get_writer_chain().invoke(inputs)


class LazyCriticChain:
    def invoke(self, inputs):
        return _get_critic_chain().invoke(inputs)


writer_chain = LazyWriterChain()
critic_chain = LazyCriticChain()


def generate_report(topic: str, research: str) -> str:
    """
    Generate the final report. Never throw raw HTTP errors to Streamlit.
    """
    topic = (topic or "").strip()
    if not topic:
        raise ResearchPipelineError("Topic is empty.")

    cap = int(os.getenv("PROMPT_RESEARCH_MAX_CHARS", "18000"))
    bounded = _truncate(research or "", max_chars=cap)

    chain = _get_writer_chain()
    out = _safe_invoke(chain, {"topic": topic, "research": bounded})
    text = str(out or "").strip()

    if not text:
        raise ResearchPipelineError(
            "Writer returned empty output. "
            f"Topic chars={len(topic)}, research bounded chars={len(bounded)}."
        )
    return text


def review_report(report: str) -> str:
    """
    Critic feedback. Never throw raw HTTP errors to Streamlit.
    """
    cap = int(os.getenv("PROMPT_REPORT_MAX_CHARS", "14000"))
    bounded = _truncate(report or "", max_chars=cap)

    chain = _get_critic_chain()
    out = _safe_invoke(chain, {"report": bounded})
    text = str(out or "").strip()

    if not text:
        raise ResearchPipelineError(
            "Critic returned empty output. "
            f"Report bounded chars={len(bounded)}."
        )
    return text


# -------- Agents --------
def build_search_agent() -> Any:
    """
    Search agent using Tavily via LangChain tool calling.
    """
    llm = _make_llm()
    # create_agent will use the tool based on LLM tool selection.
    return create_agent(model=llm, tools=[web_search])


def build_reader_agent() -> Any:
    """
    Reader agent that scrapes the most relevant URL(s).
    """
    llm = _make_llm()
    return create_agent(model=llm, tools=[scrape_url])

