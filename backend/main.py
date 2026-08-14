from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# application imports
from infer import stream_response
from models import GenerateRequestDTO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins = ['*'],
    allow_methods = ['*'],
    allow_headers = ['*']
)

async def _stream_response(payload: GenerateRequestDTO):
    async for event in stream_response(text = payload.prompt, generation_config = payload.generation_config):
        yield event

@app.post("/generate")
async def generate_response(payload: GenerateRequestDTO):
    return StreamingResponse(
        _stream_response(payload),
        media_type = "text/event-stream"
    )

def main():
    uvicorn.run(
        "main:app",
        host = "127.0.0.1",
        port = 8000,
        reload = True
    )


if __name__ == "__main__":
    main()