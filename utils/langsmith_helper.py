import os

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
    """
    if enable_tracing:
        api_key = os.getenv("LANGCHAIN_API_KEY")
        if not api_key or api_key == "your_langsmith_api_key_here":
            print("Warning: Valid LANGCHAIN_API_KEY is not set. Tracing may not work.")
        
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        # LANGCHAIN_PROJECT should be set in .env or defaults to "default"
        if not os.getenv("LANGCHAIN_PROJECT"):
            os.environ["LANGCHAIN_PROJECT"] = "SimpleRAGProject"
    else:
        # Disable tracing
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
