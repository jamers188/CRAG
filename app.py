# SQLite fix - must be at the very top before any other imports
import sys
sys.modules['sqlite3'] = __import__('pysqlite3')

import streamlit as st
import yaml
import os
from typing import Dict, TypedDict, List
import pprint
from pathlib import Path
import nest_asyncio

# Now import ChromaDB and other dependencies
import chromadb
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
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.vectorstore = None
    st.session_state.workflow = None
    st.session_state.config = None
    st.session_state.chroma_client = None

# Suggested documentation URLs
SUGGESTED_URLS = [
    "https://github.com/openai/whisper/blob/main/whisper/model.py",
    "https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py",
    "https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py",
    "https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/activation.py",
    "https://github.com/keras-team/keras/blob/master/keras/layers/attention.py"
]

def load_config():
    """Load configuration from YAML file and update with suggested URLs if needed"""
    if st.session_state.config is None:
        config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            # Update doc_url with suggested URLs if current ones are not accessible
            config["doc_url"] = SUGGESTED_URLS
            st.session_state.config = config
    return st.session_state.config

def load_documents(urls: List[str]) -> List[Document]:
    """Load documents from multiple URLs"""
    all_docs = []
    for url in urls:
        try:
            loader = WebBaseLoader(url)
            loader.requests_per_second = 1
            docs = loader.load()  # Using synchronous loading
            all_docs.extend(docs)
            st.info(f"Loaded documents from {url}")
        except Exception as e:
            st.warning(f"Error loading from {url}: {str(e)}")
    return all_docs

def initialize_embeddings(config):
    """Initialize appropriate embeddings based on configuration"""
    try:
        if config["run_local"] == 'Yes':
            return GPT4AllEmbeddings()
        elif config["models"] == 'openai':
            # Extract base URL from the complete URL
            base_url = "https://api.openai.com"
            return OpenAIEmbeddings(
                openai_api_key=config["openai_api_key"],
                openai_api_base=base_url
            )
        else:
            return GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=config["google_api_key"]
            )
    except Exception as e:
        st.error(f"Error initializing embeddings: {str(e)}")
        raise

def initialize_chroma_client():
    """Initialize ChromaDB client with error handling"""
    try:
        if st.session_state.chroma_client is None:
            persist_dir = Path("./chroma_db")
            persist_dir.mkdir(exist_ok=True)
            
            # Initialize with minimal settings
            settings = chromadb.Settings(
                is_persistent=True,
                persist_directory=str(persist_dir),
                anonymized_telemetry=False
            )
            
            st.session_state.chroma_client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=settings
            )
            
        return st.session_state.chroma_client
    except Exception as e:
        st.error(f"Error initializing ChromaDB client: {str(e)}")
        raise

def initialize_vectorstore(config):
    """Initialize ChromaDB vectorstore with documents"""
    try:
        # Initialize ChromaDB client first
        client = initialize_chroma_client()
        
        # Load documents
        docs = load_documents(config["doc_url"])
        if not docs:
            raise ValueError("No documents were loaded successfully")
        
        st.info(f"Loaded {len(docs)} documents successfully")

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=500, chunk_overlap=100
        )
        all_splits = text_splitter.split_documents(docs)
        
        st.info(f"Split into {len(all_splits)} chunks")

        # Initialize embeddings
        embeddings = initialize_embeddings(config)
        st.info("Embeddings initialized successfully")
        
        # Initialize vectorstore
        collection_name = "rag-chroma"
        try:
            # Check if collection exists and delete if it does
            if collection_name in [col.name for col in client.list_collections()]:
                client.delete_collection(collection_name)
                
            vectorstore = Chroma(
                client=client,
                collection_name=collection_name,
                embedding_function=embeddings,
            )
            
            # Add documents to vectorstore
            vectorstore.add_documents(documents=all_splits)
            st.info("Vectorstore initialized successfully")
            
            return vectorstore
            
        except Exception as e:
            st.error(f"Error with vectorstore operations: {str(e)}")
            raise
        
    except Exception as e:
        st.error(f"Error initializing vectorstore: {str(e)}")
        raise

# Rest of the code remains the same...
# (GraphState, retrieve, setup_workflow, and main functions)

def main():
    st.title("CRAG Ollama Chat")
    st.markdown("""
    ### Sample Queries:
    - How is the self-attention mechanism implemented in transformers?
    - Explain the implementation of multi-head attention in BERT
    - How does the attention mask work in the transformer implementation?
    - Show me the core attention computation code
    """)

    try:
        # Load configuration
        config = load_config()
        
        # Show available documentation sources
        with st.expander("Available Documentation Sources"):
            for url in config["doc_url"]:
                st.write(f"- {url}")
        
        # Initialize vectorstore if not already done
        if not st.session_state.initialized:
            with st.spinner("Initializing application..."):
                try:
                    st.session_state.vectorstore = initialize_vectorstore(config)
                    st.session_state.initialized = True
                    st.success("Application initialized successfully!")
                except Exception as e:
                    st.error(f"Failed to initialize application: {str(e)}")
                    return

        # User input
        user_question = st.text_input("Please enter your question:")

        if user_question:
            try:
                # Initialize workflow if not already done
                if st.session_state.workflow is None:
                    workflow = setup_workflow(config)
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

    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.error("Stack trace:", exc_info=True)

if __name__ == "__main__":
    main()
