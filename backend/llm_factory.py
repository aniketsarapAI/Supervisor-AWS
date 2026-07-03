from backend.config import settings


def get_llm():
    provider = settings.llm_provider
    model = settings.llm_model
    temperature = settings.llm_temperature

    if provider == "openrouter":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.openrouter_api_key,
            default_headers={
                "HTTP-Referer": settings.app_url,
                "X-Title": "Supervisor Multi-Agent",
            },
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature, api_key=settings.openai_api_key)
    elif provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, temperature=temperature, api_key=settings.groq_api_key)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def get_embeddings():
    provider = settings.embeddings_provider
    model = settings.embeddings_model

    if provider == "openai" or provider == "openrouter":
        from langchain_openai import OpenAIEmbeddings
        api_key = settings.openai_api_key or settings.openrouter_api_key
        base_url = "https://openrouter.ai/api/v1" if provider == "openrouter" else None
        return OpenAIEmbeddings(model=model, api_key=api_key, openai_api_base=base_url)
    else:
        raise ValueError(f"Unknown embeddings provider: {provider}")