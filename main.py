import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response

from config import settings
from core.logging import logger
from core.storage.db import init_db
from core.tracing import setup_tracing
from routers import health, workflows


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting agentic server...")
    await init_db()
    setup_tracing()
    logger.info(f"Server ready  host={settings.host}  port={settings.port}")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Agentic Workflow Server",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next) -> Response:
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        f"{request.method} {request.url.path}  status={response.status_code}  duration={duration_ms:.1f}ms"
    )
    return response


app.include_router(health.router)
app.include_router(workflows.router, prefix="/workflows")
