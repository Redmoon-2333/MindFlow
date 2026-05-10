from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from mindflow.config import settings
from mindflow.models.database import init_db
from mindflow.api.routes import router as api_router
from mindflow.api.websocket import router as ws_router
from mindflow.logging_config import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("MindFlow starting up...")
    init_db()
    logger.info("Database initialized")
    yield
    logger.info("MindFlow shutting down")


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="MindFlow - Intelligent Focus Assistant",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
app.include_router(ws_router)
