import requests
import json
import streamlit as st
from langchain.agents import initialize_agent, Tool
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage  # Import SystemMessage
import os
from urllib.parse import urlparse

# Get API keys from environment variables.
SERP_API_KEY = os.environ.get("SERP_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

def get_news_articles(query: str) -> str:
    params = {
        'api_key': SERP_API_KEY,
        'q': query,
        'gl': 'us',
        'hl': 'en',
        # Uncomment the following line to restrict strictly to news results:
        # 'tbm': 'nws',
        'google_domain': 'google.com'
    }
    response = requests.get('https://api.scaleserp.com/search', params=params)
    if response.status_code == 200:
        result_json = response.json()
        summaries = []
        # Try "news_results" first.
        if "news_results" in result_json and result_json["news_results"]:
            articles = result_json["news_results"][:3]
            for article in articles:
                title = article.get("title", "No Title")
                snippet = article.get("snippet", "No Summary")
                link = article.get("link", "No Link")
                summaries.append(f"Title: {title}\nSummary: {snippet}\nLink: {link}")
            return "\n\n".join(summaries)
        # If not available, try "top_stories".
        elif "top_stories" in result_json and result_json["top_stories"]:
            articles = result_json["top_stories"][:3]
            for article in articles:
                title = article.get("title", "No Title")
                snippet = article.get("snippet", f"Source: {article.get('source', 'Unknown')}, Date: {article.get('date', 'Unknown')}")
                link = article.get("link", "No Link")
                summaries.append(f"Title: {title}\nSummary: {snippet}\nLink: {link}")
            return "\n\n".join(summaries)
        else:
            return "No news results found. Raw response:\n" + json.dumps(result_json, indent=2)
    else:
        return f"Error: Unable to fetch data from SERP API, status code: {response.status_code}"

# Wrap the news retrieval function as a LangChain Tool.
serp_news_tool = Tool(
    name="SERPNewsAPI",
    func=get_news_articles,
    description=(
        "Fetches the latest news articles based on a given query. "
        "Use this tool after determining the user's mood to retrieve tailored news."
    )
)

# Define a system message with explicit instructions.
system_message = (
    "You are a friendly, mood-sensitive news assistant. "
    "Always start by asking: 'How has your day been so far?' "
    "Then, based on the user's answer, analyze their mood and map it to a news category as follows: "
    "Happy → Comedy news, Sad → Politics news, Stressed → Business news, Excited → Sports news, Neutral → Technology news. "
    "Once the mood and corresponding category are determined, construct a detailed query (for example, 'latest politics news') "
    "and use the SERPNewsAPI tool to fetch three recent news articles. Respond in a friendly, conversational tone and conclude after delivering the news."
)

# Initialize the ChatOpenAI language model.
llm = ChatOpenAI(temperature=1, model_name="gpt-4o", openai_api_key=OPENAI_API_KEY)

# Initialize conversation memory.
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Add the system message to the conversation memory.
memory.chat_memory.add_message(SystemMessage(content=system_message))

# Initialize the conversational agent.
agent = initialize_agent(
    tools=[serp_news_tool],
    llm=llm,
    agent="chat-conversational-react-description",
    memory=memory,
    verbose=True,
    handle_parsing_errors=True
)

def main():
    st.title("Mood-Sensitive News Chatbot")
    st.markdown("Type your message below to receive a summary of top news tailored to your mood.")
    
    # Use a form for user input.
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Your message:")
        submitted = st.form_submit_button(label="Send")
    
    if submitted and user_input:
        with st.spinner("Processing..."):
            response = agent.run(user_input)

    # Display the conversation history from memory.
    st.markdown("### Conversation History")
    for msg in memory.chat_memory.messages:
        if msg.type != "system":
            st.markdown(f"**{msg.type.capitalize()}**: {msg.content}")

    # if submitted and user_input:
    #     st.write(response)

if __name__ == "__main__":
    main()
