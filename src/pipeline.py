from agents import (
    build_search_agent,
    build_reader_agent,
    writer_chain,
    critic_chain,
    _make_llm,
)

# LLM used for formatting the search report (separate from writer/critic chains)
llm = _make_llm()


def run_research_pipeline(topic: str) -> dict:
    state = {}

    # ==========================================================
    # STEP 1 - SEARCH AGENT
    # ==========================================================

    print("\n" + "=" * 80)
    print("🔍 STEP 1 - SEARCH AGENT IS WORKING...")
    print("=" * 80)

    search_agent = build_search_agent()

    search_result = search_agent.invoke(
        {
            "messages": [
                (
                    "user",
                    f"Find recent, reliable and detailed information about: {topic}",
                )
            ]
        }
    )

    message = search_result["messages"][-1]

    raw_text = ""

    if isinstance(message.content, list):
        for item in message.content:
            if isinstance(item, dict) and item.get("type") == "text":
                raw_text += item.get("text", "") + "\n"

    elif isinstance(message.content, str):
        raw_text = message.content

    else:
        raw_text = str(message.content)

    formatter_prompt = f"""
You are an expert research formatter.

Convert the following research into a professional report.

==================================================

🔍 SEARCH AGENT REPORT

==================================================

📌 Topic
{topic}

📝 Introduction

(Write a concise introduction)

🔑 Key Findings

1.
2.
3.
4.
5.

📊 Important Facts

•
•
•

📈 Latest Trends

•
•
•

🏢 Major Companies / Technologies

•
•
•

📉 Statistics

•
•
•

🌐 Sources

1.
2.
3.
4.
5.

==================================================

Research

{raw_text}

Return ONLY the formatted report.
"""

    formatted = llm.invoke(formatter_prompt)

    state["search_results"] = formatted.content

    print(state["search_results"])

    # ==========================================================
    # STEP 2 - READER AGENT
    # ==========================================================

    print("\n" + "=" * 80)
    print("📖 STEP 2 - READER AGENT IS SCRAPING...")
    print("=" * 80)

    reader_agent = build_reader_agent()

    reader_result = reader_agent.invoke(
        {
            "messages": [
                (
                    "user",
                    f"""
Based on the search report below about "{topic}",
identify the best source URL,
scrape it,
and return detailed information.

Search Report:

{state["search_results"]}
""",
                )
            ]
        }
    )

    reader_message = reader_result["messages"][-1]

    if isinstance(reader_message.content, list):
        scraped = ""

        for item in reader_message.content:
            if isinstance(item, dict) and item.get("type") == "text":
                scraped += item.get("text", "") + "\n"

        state["scraped_content"] = scraped

    else:
        state["scraped_content"] = str(reader_message.content)

    print(state["scraped_content"])

    # ==========================================================
    # STEP 3 - WRITER
    # ==========================================================

    print("\n" + "=" * 80)
    print("📝 STEP 3 - WRITER IS GENERATING REPORT...")
    print("=" * 80)

    research = f"""
SEARCH REPORT

{state["search_results"]}

==================================================

SCRAPED CONTENT

{state["scraped_content"]}
"""

    state["writer"] = writer_chain.invoke(
        {
            "topic": topic,
            "research": research,
        }
    )

    print(state["writer"])

    # ==========================================================
    # STEP 4 - CRITIC
    # ==========================================================

    print("\n" + "=" * 80)
    print("⭐ STEP 4 - CRITIC REVIEW")
    print("=" * 80)

    state["critic"] = critic_chain.invoke(
        {
            "report": state["writer"]
        }
    )

    print(state["critic"])

    # ==========================================================
    # FINAL RESPONSE
    # ==========================================================

    return {
        "topic": topic,
        "search_results": state["search_results"],
        "scraped_content": state["scraped_content"],
        "writer": state["writer"],
        "critic": state["critic"],
    }


if __name__ == "__main__":
    topic = input("\nEnter Research Topic: ")
    result = run_research_pipeline(topic)

    print("\nPipeline Completed Successfully.")