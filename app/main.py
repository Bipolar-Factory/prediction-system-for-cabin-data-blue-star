from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import (
    data,
    spotlight_doc,
    health,
)


def create_app() -> FastAPI:
    """Factory function to create and configure the FastAPI application."""
    app = FastAPI(
        title="Blue Star Prediction System",
        description="Documentation of Cabin-data Blue Star Prediction System",
    )

    configure_cors(app)
    register_routers(app)

    return app


def configure_cors(app: FastAPI) -> None:
    """Configures CORS settings for the application."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins, customize as needed
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def register_routers(app: FastAPI) -> None:
    """Registers all the routers for the application."""

    app.include_router(router=health.router, prefix="/health", tags=["health"])
    app.include_router(router=data.router, prefix="/data", tags=["data"])
    app.include_router(router=spotlight_doc.router, prefix="/sldoc", tags=["sldoc"])


app = create_app()
