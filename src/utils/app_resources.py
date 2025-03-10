import logging
import sqlite3
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.settings import settings

logging.basicConfig(
    level=getattr(logging, settings.LOGGING_LEVEL),  # Set the logging level
    format="[%(asctime)s][%(name)s] %(levelname)s: %(message)s",  # Format for log messages
    datefmt="%Y-%m-%d %H:%M:%S",  # Date format
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def app_resources_lifespan(app: FastAPI):
    try:
        yield
    finally:
        # Cleanup resources
        logger.info("Database connection closed.")
