import os

from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model

# Accept either variable name for the Gemini key.
if os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

# Only enable LangSmith tracing if a key exists (assigning None to os.environ crashes).
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "Advance-RAG-system")


class LLM:
    # Single source of truth: ResponseGenerator logs exactly these parameters.
    PARAMS = {
        "model": "gemini-2.5-flash",
        "model_provider": "google_genai",
        "temperature": 0.0,       # verification must be deterministic
        "max_tokens": 16000,
        "max_retries": 3,
        "timeout": 120,
    }

    _llm = None

    @classmethod
    def has_api_key(cls) -> bool:
        return bool(os.getenv("GOOGLE_API_KEY"))

    @classmethod
    def load_model(cls):
        if cls._llm is None:
            if not cls.has_api_key():
                raise RuntimeError(
                    "GOOGLE_API_KEY (or GEMINI_API_KEY) is not set. "
                    "Add it to your .env file."
                )
            cls._llm = init_chat_model(**cls.PARAMS, streaming=True)

    @classmethod
    def get_llm(cls):
        cls.load_model()
        return cls._llm
