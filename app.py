import streamlit as st
import os
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from utils.langsmith_helper import setup_langsmith_tracing
from utils.rag_pipeline import ingest_documents, query_rag_pipeline, clear_system_database
from utils.groq_helper import AVAILABLE_MODELS, DEFAULT_MODEL

# Configure the Streamlit page
st.set_page_config(page_title="Simple RAG App", page_icon="📚", layout="wide")

# Sidebar Configuration
with st.sidebar:
    st.title("⚙️ RAG Configuration")
    
    # Langsmith Tracing Toggle
    enable_tracing = st.toggle("Enable LangSmith Tracing", value=True)
    setup_langsmith_tracing(enable_tracing)
    
    # Model Selection
    selected_model = st.selectbox(
        "Select LLM Model (Groq)",
        options=AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index(DEFAULT_MODEL)
    )
    
    st.divider()
    
    # Document Upload section
    st.subheader("📄 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload PDF documents", 
        type="pdf", 
        accept_multiple_files=True
    )
    
    if st.button("Ingest Documents", type="primary"):
        if uploaded_files:
            with st.spinner("Processing documents (extracting, chunking, embedding)..."):
                success, message, chunk_count = ingest_documents(uploaded_files)
                if success:
                    st.success(f"{message} Generated {chunk_count} chunks.")
                    # Show sample questions after successful upload
                    with open("assets/sample_questions.txt", "r") as f:
                        sample_qs = f.read()
                    st.info(f"**Try asking:**\n\n{sample_qs}")
                else:
                    st.error(message)
        else:
            st.warning("Please upload at least one PDF file first.")
            
    st.divider()
    
    # Advanced Settings: Metadata Filtering
    st.subheader("🔍 Advanced Search")
    st.caption("Optional: Filter by metadata (JSON format)")
    st.caption('Example: {"filename": "policy.pdf"}')
    metadata_filter_str = st.text_input("Metadata Filter (JSON)")
    
    metadata_filter = None
    if metadata_filter_str:
        try:
            metadata_filter = json.loads(metadata_filter_str)
            st.success("Valid JSON filter applied.")
        except json.JSONDecodeError:
            st.error("Invalid JSON format. Filter ignored.")
            
    st.divider()
    
    # Database Management
    st.subheader("🗑️ Database Management")
    if st.button("Clear Database"):
        if clear_system_database():
            st.success("Database cleared successfully!")
        else:
            st.error("Failed to clear database.")

# Main Chat Interface
st.title("📚 Simple RAG Chatbot")
st.markdown("Ask questions about your uploaded documents. The AI will answer based **only** on the retrieved context.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Searching database and generating answer..."):
            answer, retrieved_chunks, distances = query_rag_pipeline(
                user_query=prompt,
                model_name=selected_model,
                metadata_filter=metadata_filter
            )
            
            st.markdown(answer)
            
            # EXPLAINABILITY FEATURES: Show what chunks were retrieved
            if retrieved_chunks:
                with st.expander(f"🔍 View {len(retrieved_chunks)} Retrieved Chunks & Metadata"):
                    for idx, (chunk, dist) in enumerate(zip(retrieved_chunks, distances)):
                        st.markdown(f"**Chunk {idx + 1}** (Distance: `{dist:.4f}` - *lower distance = higher similarity*)")
                        st.json(chunk["metadata"])
                        st.info(chunk["text"][:300] + "... [TRUNCATED]")
                        st.divider()

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})
