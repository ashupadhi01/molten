from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn
import asyncio

# application imports
from infer import generate

app = FastAPI()

@app.get("/generate")
async def generate_response(text: str):
    return StreamingResponse(
        generate(text),
        media_type = "text/event-stream"
    )

def main():
    uvicorn.run(
        "main:app",
        host = "127.0.0.1",
        port = 8000,
        reload = False
    )


if __name__ == "__main__":
    main()