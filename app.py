import streamlit as st
import yaml
import os
import sys
import nest_asyncio
from typing import Dict, TypedDict
import pprint
from pathlib import Path

# SQLite fix for ChromaDB
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Langchain imports
from langchain import hub
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.schema import Document
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from langchain.prompts import PromptTemplate

# Apply nest_asyncio
nest_asyncio.apply()

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'workflow' not in st.session_state:
    st.session_state.workflow = None

def load_config():
    """Load configuration from YAML file"""
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def initialize_embeddings(config):
    """Initialize appropriate embeddings based on configuration"""
    if config["run_local"] == 'Yes':
        return GPT4AllEmbeddings()
    elif config["models"] == 'openai':
        return OpenAIEmbeddings(
            openai_api_key=config["openai_api_key"], 
            openai_api_base=config["openai_api_base"]
        )
    else:
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=config["google_api_key"]
        )

def initialize_llm(config):
    """Initialize appropriate LLM based on configuration"""
    if config["run_local"] == "Yes":
        return ChatOllama(model=config["local_llm"], temperature=0)
    elif config["models"] == "openai":
        return ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=config["openai_api_key"]
        )
    else:
        return ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=config["google_api_key"],
            convert_system_message_to_human=True,
            verbose=True
        )

def initialize_vectorstore(config):
    """Initialize ChromaDB vectorstore with documents"""
    try:
        # Load documents
        loader = WebBaseLoader(config["doc_url"])
        loader.requests_per_second = 1
        docs = loader.aload()

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=500, chunk_overlap=100
        )
        all_splits = text_splitter.split_documents(docs)

        # Initialize embeddings and vectorstore
        embeddings = initialize_embeddings(config)
        
        # Ensure persistent directory exists
        persist_dir = Path("./chroma_db")
        persist_dir.mkdir(exist_ok=True)
        
        vectorstore = Chroma.from_documents(
            documents=all_splits,
            collection_name="rag-chroma",
            embedding=embeddings,
            persist_directory=str(persist_dir)
        )
        return vectorstore
    except Exception as e:
        st.error(f"Error initializing vectorstore: {str(e)}")
        return None

def main():
    st.title("CRAG Ollama Chat")
    st.text("A possible query: How is the attention mechanism implemented in code in the article?")

    # Load configuration
    config = load_config()

    # Initialize vectorstore if not already done
    if st.session_state.vectorstore is None:
        with st.spinner("Initializing vectorstore..."):
            st.session_state.vectorstore = initialize_vectorstore(config)
            if st.session_state.vectorstore is None:
                st.error("Failed to initialize vectorstore")
                return

    # User input
    user_question = st.text_input("Please enter your question:")

    if user_question:
        try:
            # Initialize workflow if not already done
            if st.session_state.workflow is None:
                workflow = setup_workflow(config, st.session_state.vectorstore)
                st.session_state.workflow = workflow.compile()

            # Process question
            inputs = {
                "keys": {
                    "question": user_question,
                    "local": config["run_local"],
                }
            }

            # Create expandable sections for each step
            for output in st.session_state.workflow.stream(inputs):
                for key, value in output.items():
                    with st.expander(f"Node '{key}':"):
                        st.text(pprint.pformat(value["keys"], indent=2, width=80, depth=None))

            # Show final generation
            final_generation = value['keys'].get('generation', 'No final generation produced.')
            st.subheader("Final Generation:")
            st.write(final_generation)

        except Exception as e:
            st.error(f"Error processing question: {str(e)}")

if __name__ == "__main__":
    main()
