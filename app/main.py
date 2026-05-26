from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from models import SceneRequest
from rag_service import analyze_scene


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "Unity Architectural AI Backend Running"
    }


@app.post("/analyze_scene")
async def analyze_scene_endpoint(request: SceneRequest):
    result = await analyze_scene(
        request.scene_data.model_dump()
    )

    return {
        "result": result
    }