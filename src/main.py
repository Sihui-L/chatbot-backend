from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute, APIWebSocketRoute

from src.routes.health import router as health_router
from src.routes.v1.endpoints import router as v1_router
from src.settings import settings
from src.utils.app_resources import app_resources_lifespan, logger


def get_application() -> FastAPI:
    app = FastAPI(
        title=settings.API_NAME,
        version=settings.API_VERSION,
        description=settings.API_DESCRIPTION,
        lifespan=app_resources_lifespan,
    )
    logger.info("FastAPI application initalised successfully.")

    # Add routers
    for router in [health_router, v1_router]:
        app.include_router(router)
        # Log the routes that have been added
        for route in router.routes:
            if isinstance(route, APIRoute):
                logger.info(f"HTTP Route added: {route.path} - {route.methods}")
            elif isinstance(route, APIWebSocketRoute):
                logger.info(f"WebSocket Route added: {route.path}")
            else:
                logger.info(f"Other Route added: {route.path}")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )
    return app


application = get_application()
