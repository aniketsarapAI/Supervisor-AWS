"""
LLM Factory — provider abstraction for chat models.

All LLM instantiation happens here.
Application code calls get_llm() and never imports provider-specific classes directly.

To add a new provider (e.g., Bedrock), add a branch here.
No other file needs to change.
"""

from backend.config import settings


def get_llm():
    """
    Return a chat model based on the configured provider.

    Supported providers:
    - openrouter: OpenAI-compatible API via OpenRouter
    - openai: Direct OpenAI API
    - groq: Groq API
    - bedrock: Amazon Bedrock (to be implemented)
    """
    provider = settings.LLM_PROVIDER
    model = settings.LLM_MODEL
    temperature = settings.LLM_TEMPERATURE

    if provider == "openrouter":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.OPENROUTER_API_KEY,
            default_headers={
                "HTTP-Referer": settings.APP_URL,
                "X-Title": "Supervisor Multi-Agent",
            },
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature)
    elif provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, temperature=temperature)
    elif provider == "bedrock":
        from langchain_aws import ChatBedrock
        return ChatBedrock(
            model_id=model,
            temperature=temperature,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
