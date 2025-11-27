from fastapi import FastAPI
from service.api.endpoints import router
from service.core.config import settings
from prometheus_client import make_asgi_app

app = FastAPI(title=settings.PROJECT_NAME)

# Add Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
