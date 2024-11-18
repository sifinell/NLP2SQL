import streamlit as st
from pathlib import Path
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.utilities import SQLDatabase
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain_community.vectorstores import FAISS
from sqlalchemy import create_engine
from dotenv import load_dotenv
import sqlite3
import ast
import re
import os


load_dotenv()

st.set_page_config(page_title="NLP2SQL", page_icon="🔍", layout="wide")
st.title("Chat with your SQL Database")

def configure_llm():
    
    os.environ["AZURE_OPENAI_ENDPOINT"] = "insert_endpoint_here"
    os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")

    return AzureChatOpenAI(
        azure_deployment="gpt-4o",
        api_version="2024-05-01-preview",
        temperature=0, 
        streaming=True
    )

def configure_db():
    
    db_filepath = (Path(__file__).parent / "Chinook.db").absolute()
    creator = lambda: sqlite3.connect(f"file:{db_filepath}?mode=ro", uri=True)
    
    return SQLDatabase(create_engine("sqlite:///", creator=creator))

def query_as_list(db, query):
    
    res = db.run(query)
    res = [el for sub in ast.literal_eval(res) for el in sub if el]
    res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
    
    return list(set(res))

def configure_tools(db):
    
    tools = []
    
    artists = query_as_list(db, "SELECT Name FROM Artist")
    albums = query_as_list(db, "SELECT Title FROM Album")
    
    vector_db = FAISS.from_texts(artists + albums, AzureOpenAIEmbeddings())
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    description = """Use to look up values to filter on. Input is an approximate spelling of the proper noun, output is \
    valid proper nouns. Use the noun most similar to the search."""
    
    retriever_tool = create_retriever_tool(
        retriever,
        name="search_proper_nouns",
        description=description,
    )

    tools.append(retriever_tool)

    return tools

def configure_agent(db, llm, tools):

    system = """You are an agent designed to interact with a SQL database.
    Given an input question, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
    Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
    You can order the results by a relevant column to return the most interesting examples in the database.
    Never query for all the columns from a specific table, only ask for the relevant columns given the question.
    You have access to tools for interacting with the database.
    Only use the given tools. Only use the information returned by the tools to construct your final answer.
    You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

    You have access to the following tables: {table_names}

    If you need to filter on a proper noun, you must ALWAYS first look up the filter value using the "search_proper_nouns" tool!
    Do not try to guess at the proper name - use this function to find similar ones.""".format(
        table_names=db.get_usable_table_names()
    )

    agent = create_sql_agent(
        llm=llm,
        db=db,
        extra_tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        prefix=system
    )

    return agent

llm = configure_llm()
db = configure_db()
tools = configure_tools(db)
agent = configure_agent(db, llm, tools)

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask me anything!")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())
        response = agent.run(user_query, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)