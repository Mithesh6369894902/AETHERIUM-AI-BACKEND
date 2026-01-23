from fastapi import FastAPI
from modelcraft_router import router as modelcraft_router  # adjust import

app = FastAPI()
app.include_router(modelcraft_router)
