from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import inferno, vision, text, workflow, model

app = FastAPI(title="ÆTHERIUM AI Backend")

# ---------------- CORS ---------------- #
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- ROUTERS ---------------- #
app.include_router(inferno.router)
app.include_router(vision.router)
app.include_router(text.router)
app.include_router(workflow.router)
app.include_router(model.router)

@app.get("/")
def root():
    return {"status": "ÆTHERIUM backend running"}


