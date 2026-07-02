from agents import build_reader_agent , build_search_agent , writer_chain , critic_chain

def run_research_pipeline(topic : str) -> dict:

    state = {}

    #search agent working 
    
print("\n" + "=" * 80)
print("🔍 STEP 1 - SEARCH AGENT IS WORKING...")
print("=" * 80)

search_agent = build_search_agent()

search_result = search_agent.invoke({
    "messages": [
        (
            "user",
            f"Find recent, reliable and detailed information about: {topic}"
        )
    ]
})

# ----------------------------------------------------------
# Extract only text from AIMessage
# ----------------------------------------------------------

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

# ----------------------------------------------------------
# Format using the LLM
# ----------------------------------------------------------

formatted_result = llm.invoke(f"""
You are an expert research formatter.

Convert the following research into the EXACT format below.

==================================================
🔍 SEARCH AGENT REPORT
==================================================

📌 Topic
...

📝 Introduction
...

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

🏢 Major Companies
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

Research:

{raw_text}

Return ONLY the formatted report.
""")

state["search_results"] = formatted_result.content

print("\n" + "=" * 80)
print(state["search_results"])
print("=" * 80)

    #step 2 - reader agent 
    print("\n"+" ="*50)
    print("step 2 - Reader agent is scraping top resources ...")
    print("="*50)

    reader_agent = build_reader_agent()
    reader_result = reader_agent.invoke({
        "messages": [("user",
            f"Based on the following search results about '{topic}', "
            f"pick the most relevant URL and scrape it for deeper content.\n\n"
            f"Search Results:\n{state['search_results'][:800]}"
        )]
    })

    state['scraped_content'] = reader_result['messages'][-1].content

    print("\nscraped content: \n", state['scraped_content'])

    #step 3 - writer chain 

    print("\n"+" ="*50)
    print("step 3 - Writer is drafting the report ...")
    print("="*50)

    research_combined = (
        f"SEARCH RESULTS : \n {state['search_results']} \n\n"
        f"DETAILED SCRAPED CONTENT : \n {state['scraped_content']}"
    )

    state["report"] = writer_chain.invoke({
        "topic" : topic,
        "research" : research_combined
    })

    print("\n Final Report\n",state['report'])

    #critic report 

    print("\n"+" ="*50)
    print("step 4 - critic is reviewing the report ")
    print("="*50)

    state["feedback"] = critic_chain.invoke({
        "report":state['report']
    })

    print("\n critic report \n", state['feedback'])

    return state



if __name__ == "__main__":
    topic = input("\n Enter a research topic : ")
    run_research_pipeline(topic)