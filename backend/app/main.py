from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes_audio import router as audio_router


app = FastAPI(title="voice-impersonation-protection")


# Allow the Vite dev server to call this API during local development.
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "service": "voice-impersonation-protection"
    }


app.include_router(audio_router)