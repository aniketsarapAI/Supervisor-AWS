"""
Embeddings Factory — provider abstraction for embedding models.

All embedding model instantiation happens here.
Application code calls get_embeddings() and never imports provider-specific classes directly.

To add a new provider (e.g., Bedrock), add a branch here.
No other file needs to change.
"""

from backend.config import settings


def get_embeddings():
    """
    Return an embeddings model based on the configured provider.

    Supported providers:
    - openai: OpenAI embeddings (via OpenRouter or direct)
    - bedrock: Amazon Bedrock embeddings (to be implemented)
    """
    provider = settings.EMBEDDINGS_PROVIDER
    model = settings.EMBEDDINGS_MODEL

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=model,
            openai_api_key=settings.OPENROUTER_API_KEY,
            openai_api_base="https://openrouter.ai/api/v1",
        )
    elif provider == "bedrock":
        from langchain_aws import BedrockEmbeddings
        return BedrockEmbeddings(model_id=model)
    else:
        raise ValueError(f"Unknown embeddings provider: {provider}")
