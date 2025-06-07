from fastapi import FastAPI
import time

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Welcome to the counter API. Use /count to retrieve the current count."}

_start = time.time()

@app.get("/count")
def read_count():
    # Number of whole seconds since startup
    return {"count": int(time.time() - _start)}