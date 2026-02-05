"""
AI Voice Detection API - Main Application

FastAPI application for detecting AI-generated voices across
multiple Indian languages with NVIDIA PersonaPlex detection.
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import logging

from .api.routes import router as api_router

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="AI Voice Detection API",
    description="API for detecting AI-generated voices using NVIDIA PersonaPlex signature matching and acoustic analysis.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "AI Voice Detection Team",
        "email": "contact@example.com"
    },
    license_info={
        "name": "MIT License"
    }
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    """Add response timing header"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}"
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "code": "INTERNAL_ERROR"
        }
    )


app.include_router(api_router)


@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "name": "AI Voice Detection API",
        "version": "1.0.0",
        "description": "Detect AI-generated voices in multiple languages",
        "docs": "/docs",
        "health": "/api/v1/health",
        "detect": "/api/v1/detect"
    }


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    logger.info("Starting AI Voice Detection API...")
    logger.info("Models will be loaded on first request (lazy loading)")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down AI Voice Detection API...")
