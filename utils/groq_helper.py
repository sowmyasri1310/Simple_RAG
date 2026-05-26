import os
from groq import Groq
from utils.prompts import RAG_PROMPT_TEMPLATE

# Groq is an AI inference engine that provides incredibly fast access to open-source models like LLaMA.
# It uses special hardware (LPU) to generate tokens at high speed.

AVAILABLE_MODELS = [
    "llama-3.3-70b-versatile",
    "deepseek-r1-distill-llama-70b",
    "llama3-8b-8192"
]

DEFAULT_MODEL = "llama-3.3-70b-versatile"

def get_groq_client() -> Groq:
    """
    Initializes the Groq client using the API key from environment variables.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set. Please add it to your .env file.")
    return Groq(api_key=api_key)

def generate_rag_answer(
    client: Groq,
    user_question: str,
    retrieved_chunks: list[str],
    model: str = DEFAULT_MODEL
) -> str:
    """
    Sends the system prompt, retrieved context, and user question to the Groq LLM
    to generate a factual answer based ONLY on the context.
    """
    # Combine the retrieved chunks into a single string
    context_str = "\n\n---\n\n".join(retrieved_chunks)
    
    # Format the prompt with our context and question
    formatted_prompt = RAG_PROMPT_TEMPLATE.format(
        retrieved_chunks=context_str,
        user_question=user_question
    )
    
    try:
        # Create a chat completion request to Groq
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": formatted_prompt,
                }
            ],
            model=model,
            temperature=0.1, # Low temperature for more factual, less creative responses
            max_tokens=1024,
        )
        
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"An error occurred while generating the answer: {e}"
