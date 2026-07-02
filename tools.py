"""tools.py

Production-grade tools for ResearchMind.

- web_search: Tavily search with validation + safe error messages
- scrape_url: robust scraping with timeouts, status checks, and HTML cleanup

These tools are used by LangChain agents and must ALWAYS return a string
(even on failure) so the agent never crashes due to tool exceptions.

UI is not affected.
"""

from __future__ import annotations

import os
import re
from typing import List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.tools import tool
from tavily import TavilyClient

load_dotenv()


class ToolExecutionError(RuntimeError):
    """Raised internally for controlled tool failures."""


def _safe_truncate(text: str, max_chars: int) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[TRUNCATED]"


def _safe_tool_result(msg: str) -> str:
    # Tools should never throw to the agent in normal operation.
    return msg


def _get_tavily_client() -> TavilyClient:
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        raise ToolExecutionError(
            "Missing TAVILY_API_KEY. Set it in your environment or .env file."
        )
    return TavilyClient(api_key=api_key)


def _clean_whitespace(text: str) -> str:
    text = text or ""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _timeout_secs(default: int) -> int:
    try:
        return int(os.getenv("SCRAPE_TIMEOUT_SECS", str(default)))
    except Exception:
        return default


def _requests_headers() -> dict:
    return {
        "User-Agent": os.getenv("SCRAPE_USER_AGENT", "Mozilla/5.0"),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": os.getenv("SCRAPE_ACCEPT_LANGUAGE", "en-US,en;q=0.9"),
    }


def _strip_noise(soup: BeautifulSoup) -> None:
    # Remove scripts/styles and common page chrome.
    for tag in soup.find_all(
        ["script", "style", "noscript", "nav", "footer", "header", "form", "iframe", "svg", "button"]
    ):
        tag.decompose()

    # Remove elements that often contain boilerplate.
    for sel in [
        "[role='navigation']",
        "[aria-label*='menu']",
        "[class*='cookie']",
        "[id*='cookie']",
        "[class*='subscribe']",
        "[id*='subscribe']",
    ]:
        for tag in soup.select(sel):
            tag.decompose()


def _extract_main_text(soup: BeautifulSoup) -> str:
    # Prefer article/main tags if present.
    main = soup.find(["article", "main"])
    container = main if main is not None else soup.body if soup.body is not None else soup

    text = container.get_text(separator=" ", strip=True)
    return _clean_whitespace(text)


@tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable information.

    Returns:
        A string containing Title, URL, and snippet blocks.
    """
    try:
        query = (query or "").strip()
        if not query:
            return _safe_tool_result("web_search: empty query")

        max_results = int(os.getenv("TAVILY_MAX_RESULTS", "5"))
        max_results = max(1, min(max_results, 10))

        max_snippet_chars = int(os.getenv("TAVILY_SNIPPET_MAX_CHARS", "300"))
        max_snippet_chars = max(50, min(max_snippet_chars, 1000))

        tavily = _get_tavily_client()

        # Tavily typically returns {"results":[...]}.
        results = tavily.search(query=query, max_results=max_results)

        items = (results or {}).get("results") or []
        if not items:
            return _safe_tool_result("web_search: no results")

        out: List[str] = []
        seen_urls = set()
        for r in items:
            title = str(r.get("title") or "").strip()
            url = str(r.get("url") or "").strip()
            content = str(r.get("content") or "").strip()

            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            snippet = _safe_truncate(content, max_chars=max_snippet_chars)

            out.append(f"Title: {title}\nURL: {url}\nSnippet: {snippet}\n")

        if not out:
            return _safe_tool_result("web_search: results missing URLs")

        return "\n----\n".join(out)

    except ToolExecutionError as e:
        return _safe_tool_result(f"web_search error: {str(e)}")
    except Exception as e:
        return _safe_tool_result(f"web_search error: {str(e)}")


@tool
def scrape_url(url: str) -> str:
    """Scrape and return clean text content from a given URL.

    The returned text is truncated to keep prompts under control.
    """
    try:
        url = (url or "").strip()
        if not url:
            return _safe_tool_result("scrape_url: empty url")

        timeout_secs = int(os.getenv("SCRAPE_TIMEOUT_SECS", "10"))
        timeout_secs = max(3, min(timeout_secs, 30))

        max_chars = int(os.getenv("SCRAPE_MAX_CHARS", "3000"))
        max_chars = max(500, min(max_chars, 20000))

        resp = requests.get(url, timeout=timeout_secs, headers=_requests_headers())

        # HTTP validation
        if resp.status_code >= 400:
            return _safe_tool_result(f"Could not scrape URL: HTTP {resp.status_code} for {url}")

        # Respect encoding as best-effort.
        resp.encoding = resp.encoding or "utf-8"

        soup = BeautifulSoup(resp.text, "html.parser")
        _strip_noise(soup)

        text = _extract_main_text(soup)
        if not text or len(text) < 30:
            return _safe_tool_result(f"Could not scrape URL: no readable text for {url}")

        return _safe_truncate(text, max_chars=max_chars)

    except requests.Timeout:
        return _safe_tool_result("Could not scrape URL: timeout")
    except requests.RequestException as e:
        return _safe_tool_result(f"Could not scrape URL: network error: {str(e)}")
    except Exception as e:
        return _safe_tool_result(f"Could not scrape URL: {str(e)}")

