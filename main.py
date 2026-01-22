from fastapi import FastAPI
from api import inferno, text, vision, workflow, model

app = FastAPI(
    title="ÆTHERIUM AI Operating System",
    version="1.0"
)

app.include_router(inferno.router)
app.include_router(text.router)
app.include_router(vision.router)
app.include_router(workflow.router)
app.include_router(model.router)

@app.get("/")
def root():
    return {"status": "ÆTHERIUM backend running"}
