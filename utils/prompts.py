# This prompt template is used to guide the Groq LLM on how to answer the user's question.
# RAG relies on restricting the LLM to ONLY use the provided context to prevent "hallucinations" (making things up).

RAG_PROMPT_TEMPLATE = """You are a helpful AI assistant.

Answer ONLY using the provided context.
Do not hallucinate.
If the answer is unavailable in the context, say:
"I could not find the answer in the uploaded documents."

Context:
{retrieved_chunks}

Question:
{user_question}
"""
