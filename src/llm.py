"""Central factory for chat models.

Keeping model construction in one place means the whole system can swap LLM
providers by changing a single module. We use Groq (free tier, fast) with the
GPT-OSS models, which support tool/structured output for the intake node.
"""

from langchain_groq import ChatGroq


def get_chat_model(model: str, temperature: float = 0.0) -> ChatGroq:
    """Build a chat model. Reads the GROQ_API_KEY from the environment."""
    return ChatGroq(model=model, temperature=temperature)
