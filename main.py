from fastapi import FastAPI
from api import inferno, text, vision, model, workflow

app = FastAPI(
    title="ÆTHERIUM AI Operating System",
    description="Unified AI Backend for Android & Desktop Apps",
    version="1.0"
)

app.include_router(inferno.router, prefix="/inferno", tags=["InfernoData"])
app.include_router(text.router, prefix="/text", tags=["TextVortex"])
app.include_router(vision.router, prefix="/vision", tags=["VisionBlaze"])
app.include_router(model.router, prefix="/model", tags=["ModelCraftX"])
app.include_router(workflow.router, prefix="/workflow", tags=["AlphaFlux"])

@app.get("/")
def root():
    return {"message": "ÆTHERIUM Backend is running"}
