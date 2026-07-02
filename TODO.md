# TODO - Production refactor

## Step 1 (agents.py)
- Refactor `agents.py` for production quality.
- Ensure Mistral integration works with proper imports.
- Implement exponential backoff for HTTP 429 with retries.
- Add robust exception handling (missing API key, timeouts, http errors).
- Limit prompt size (truncate/summarize scraped input safely).
- Avoid Streamlit crashes by raising controlled exceptions/messages.
- Keep existing public API functions/exports used by `app.py`.
- COMPLETED


## Step 2 (tools.py)
- Refactor `tools.py`:

  - Tavily search error handling and response validation.
  - Scraper: timeout, HTTP status validation, block handling, safe HTML cleanup.
  - Add typing and safe return strings.
- COMPLETED


## Step 3 (app.py)
- Refactor `app.py` to:
  - Never call `writer_chain.invoke()` / `critic_chain.invoke()` directly.
  - Use `generate_report()` / `review_report()` helpers from `agents.py`.
  - Wrap each pipeline step with try/except.
  - Show `st.error()` / `st.warning()` and keep state consistent.
  - Handle rate limits gracefully.

## Step 4 (Run & verify)
- Restart Streamlit and verify:
  - UI still renders.
  - Clicking Run doesn’t crash on transient Mistral 429.
  - Errors show as Streamlit messages.

