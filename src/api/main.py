'''FastAPI backend (STEP 7 stub).'''
# TODO: Implement
from fastapi import FastAPI
app = FastAPI()
@app.get("/health")
def health():
    return {"status": "ok"}

