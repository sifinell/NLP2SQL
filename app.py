import streamlit as st
from pathlib import Path
from langchain_openai import AzureChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent
from langchain import hub
from dotenv import load_dotenv
import os
import json

# Load environment variables from a .env file
load_dotenv()

# Retrieve Azure OpenAI configuration from environment variables
aoai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
aoai_key=os.getenv("AZURE_OPENAI_API_KEY")
aoai_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
aoai_api_version=os.getenv("AZURE_OPENAI_API_VERSION")

# Set Streamlit page configuration
st.set_page_config(page_title="NLP2SQL", page_icon="🔍", layout="wide")
st.title("Chat with your SQL Database")

# Function to configure the Azure OpenAI language model
def configure_llm():
    return AzureChatOpenAI(
        azure_endpoint=aoai_endpoint,
        api_key=aoai_key,
        azure_deployment=aoai_deployment,
        api_version=aoai_api_version,
        temperature=0, 
        streaming=True
    )

# Function to configure the SQL database connection
def configure_db():
    db_filepath = (Path(__file__).parent / "Chinook.db").absolute()
    return SQLDatabase.from_uri(f"sqlite:///{db_filepath}")

# Function to configure the tools for the agent
def configure_tools(db, llm):
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()
    return tools

# Function to configure the agent
def configure_agent(llm, tools):
    prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
    system_message = prompt_template.format(dialect="SQLite", top_k=5)
    agent_executor = create_react_agent(llm, tools, state_modifier=system_message)
    return agent_executor

# Function to format the message based on its type
def format_message(message):
    message_type = message.get("type", "unknown")
    content = message.get("content", "")
    tool_calls = message.get("tool_calls", [])
    formatted_message = None

    if message_type == "human":
        formatted_message = f"**Type:** {message_type}\n\n"
        formatted_message += f"**User:** {content}\n"
    elif message_type == "ai":
        if tool_calls:
            formatted_message = f"**Type:** {message_type}\n\n"
            formatted_message += "**Tool Calls:**\n"
            for tool_call in tool_calls:
                formatted_message += f"- **Tool Name:** {tool_call['name']}\n"
                formatted_message += f"  **Arguments:** {json.dumps(tool_call['args'], indent=2)}\n"
    elif message_type == "tool":
        formatted_message = f"**Type:** {message_type}\n\n"
        formatted_message += f"**Tool Response ({message['name']}):** {content}\n"
    else:
        formatted_message = f"**Type:** {message_type}\n\n"
        formatted_message += f"**Message:** {content}\n"
    
    return formatted_message
    
def main():
    # Configure the database, language model, tools, and agent
    db = configure_db()
    llm = configure_llm()
    tools = configure_tools(db, llm)
    agent = configure_agent(llm, tools)

    # Initialize message history or clear it if the button is pressed
    if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    # Display the chat messages
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Get user input from the chat input box
    question = st.chat_input(placeholder="Ask me anything!")
    
    if question:
        # Append the user's question to the message history
        st.session_state.messages.append({"role": "user", "content": question})
        st.chat_message("user").write(question)

        # Display the agent's reasoning with an expandable explanation
        with st.expander("See explanation", expanded=False):
            with st.chat_message("agent", avatar="🔍"):
                for step in agent.stream(
                    {"messages": st.session_state.messages},
                    stream_mode="values",
                ):
                    message = step["messages"][-1]
                    message_dict = message.dict()
                    formatted_message = format_message(message_dict)
                    if formatted_message is not None:
                        st.write(formatted_message)

        # Display the final agent's response
        with st.chat_message("assistant"):
            st.session_state.messages.append({"role": "assistant", "content": message.content})
            st.write(message.content)

if __name__ == "__main__":
    main()