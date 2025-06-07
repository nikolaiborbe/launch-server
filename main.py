from fastapi import FastAPI
import time

app = FastAPI()
_start = time.time()

@app.get("/count")
def read_count():
    # Number of whole seconds since startup
    return {"count": int(time.time() - _start)}