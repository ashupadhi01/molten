from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# application imports
from infer import generate

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins = ['*'],
    allow_methods = ['*'],
    allow_headers = ['*']
)

async def _stream_response(text):
    async for event in generate(text):
        yield event

@app.get("/generate")
async def generate_response(text: str):
    return StreamingResponse(
        _stream_response(text),
        media_type = "text/event-stream"
        # media_type = "application/x-ndjson"
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