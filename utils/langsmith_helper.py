import os
import streamlit as st

# What is LangSmith?
# LangSmith is an observability platform for LLM applications. 
# It allows developers to monitor, debug, and test their LLM pipelines.

# Why is tracing useful in RAG applications?
# A RAG pipeline involves multiple steps: document extraction, chunking, embedding generation, 
# database retrieval, and LLM generation. Tracing helps you see exactly:
# 1. What text chunks were retrieved.
# 2. What exact prompt was sent to the LLM (with the injected context).
# 3. How long each step took and how much it cost.
# 4. Where an error occurred if the pipeline fails.

def setup_langsmith_tracing(enable_tracing: bool = True):
    """
    Sets up environment variables required for LangSmith tracing based on a toggle.
    Uses Streamlit secrets if available, otherwise falls back to local environment variables.
    """
    if enable_tracing:
        # Try Streamlit Secrets first, then fallback
        try:
            api_key = st.secrets.get("LANGCHAIN_API_KEY")
        except Exception:
            api_key = None
            
        if not api_key:
            api_key = os.getenv("LANGCHAIN_API_KEY")
            
        if not api_key or api_key == "your_langsmith_api_key_here":
            print("Warning: Valid LANGCHAIN_API_KEY is not set. Tracing may not work.")
        elif api_key:
            # Set the env var so LangChain libraries can pick it up
            os.environ["LANGCHAIN_API_KEY"] = api_key
        
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        
        # Project Name
        try:
            project = st.secrets.get("LANGCHAIN_PROJECT")
        except Exception:
            project = None
            
        if not project:
            project = os.getenv("LANGCHAIN_PROJECT")
            
        if project:
            os.environ["LANGCHAIN_PROJECT"] = project
        else:
            os.environ["LANGCHAIN_PROJECT"] = "SimpleRAGProject"
    else:
        # Disable tracing
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
