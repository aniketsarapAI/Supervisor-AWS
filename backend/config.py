import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # LLM
    llm_provider: str = os.getenv("LLM_PROVIDER", "openrouter")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))

    # Embeddings
    embeddings_provider: str = os.getenv("EMBEDDINGS_PROVIDER", "openai")
    embeddings_model: str = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")

    # API Keys
    openrouter_api_key: str | None = os.getenv("OPENROUTER_API_KEY")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    tavily_api_key: str | None = os.getenv("TAVILY_API_KEY")
    groq_api_key: str | None = os.getenv("GROQ_API_KEY")

    # Supabase
    supabase_url: str | None = os.getenv("SUPABASE_URL")
    supabase_key: str | None = os.getenv("SUPABASE_KEY")
    supabase_service_key: str | None = os.getenv("SUPABASE_SERVICE_KEY")
    supabase_jwt_secret: str | None = os.getenv("SUPABASE_JWT_SECRET")
    supabase_db_url: str | None = os.getenv("SUPABASE_DB_URL")

    # LangSmith
    langchain_tracing: bool = os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
    langchain_endpoint: str | None = os.getenv("LANGCHAIN_ENDPOINT")
    langchain_api_key: str | None = os.getenv("LANGCHAIN_API_KEY")
    langchain_project: str | None = os.getenv("LANGCHAIN_PROJECT")

    # Frontend / URLs
    app_url: str = os.getenv("APP_URL", "http://localhost:8501")
    backend_url: str = os.getenv("BACKEND_URL", "http://localhost:8000")
    cors_origins: list[str] = os.getenv("CORS_ORIGINS", "*").split(",")

    # Rate limiting
    rate_limit: str = os.getenv("RATE_LIMIT", "30/minute")

    # Paths
    faiss_index_dir: str = os.getenv("FAISS_INDEX_DIR", os.path.join(os.path.dirname(__file__), "..", "faiss_index"))


settings = Settings()