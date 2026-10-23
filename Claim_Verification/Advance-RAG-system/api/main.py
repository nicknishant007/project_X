import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# --------------------------------------------------------------------
# Make project root importable
# --------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --------------------------------------------------------------------
# Routers
# --------------------------------------------------------------------
from api.routes.chat import router as chat_router
from api.routes.ingest import router as ingest_router

# --------------------------------------------------------------------
# Services
# --------------------------------------------------------------------
from api.services.ingest_service import IngestService
from api.services.rag_service import RAGService


# ====================================================================
# Lifespan
# ====================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("=" * 70)
    print("🚀 Starting Advance RAG System...")
    print("=" * 70)

    try:

        print("Initializing registry...")
        IngestService.initialize_registry()

        print("Loading embedding model...")
        print("Loading reranker model...")
        RAGService.warmup()

        print("✅ System Ready")

    except Exception as e:

        print("Startup Error")
        print(e)

    yield

    print("=" * 70)
    print("Shutting down...")
    print("=" * 70)


# ====================================================================
# FastAPI
# ====================================================================

app = FastAPI(

    title="Advance RAG System",

    description="""
Hybrid RAG System

• BM25 Retrieval

• Dense Retrieval

• MMR

• Cross Encoder

• Gemini Streaming
""",

    version="1.0.0",

    lifespan=lifespan,
)

# ====================================================================
# CORS
# ====================================================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=[
        "*"
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)

# ====================================================================
# Static Files
# ====================================================================

FRONTEND = ROOT / "frontend"

print("ROOT:", ROOT)
print("FRONTEND:", FRONTEND)
print("FRONTEND EXISTS:", FRONTEND.exists())
print("STYLE EXISTS:", (FRONTEND / "css" / "style.css").exists())

app.mount(
    "/static",
    StaticFiles(directory=FRONTEND),
    name="static",
)

# ====================================================================
# Home Page
# ====================================================================

@app.get("/", include_in_schema=False)
async def home():

    return FileResponse(
        FRONTEND / "index.html"
    )

# ====================================================================
# Health
# ====================================================================

@app.get("/health")
async def health():

    return {

        "status": "ok",

        "service": "Advance RAG System",

        "version": "1.0.0"

    }

# ====================================================================
# Routers
# ====================================================================

app.include_router(

    chat_router,

    prefix="/chat",

    tags=["Chat"]

)

app.include_router(

    ingest_router,

    prefix="/ingest",

    tags=["Ingestion"]

)