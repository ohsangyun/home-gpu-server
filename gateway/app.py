import os
import asyncio
import subprocess
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434")
API_TOKEN = os.environ.get("API_TOKEN", "")
ALLOWED_MODELS = {m.strip() for m in os.environ.get("ALLOWED_MODELS", "gpt-oss:20b").split(",")}

# 단일 GPU: 동시에 한 요청만 통과시킨다
gpu_lock = asyncio.Semaphore(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not API_TOKEN:
        raise RuntimeError("API_TOKEN is not set")
    app.state.client = httpx.AsyncClient(base_url=OLLAMA_BASE_URL, timeout=None)
    yield
    await app.state.client.aclose()


app = FastAPI(title="home-gpu-server gateway", lifespan=lifespan)


def check_auth(authorization: str | None):
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="unauthorized")


@app.get("/healthz")
async def healthz():
    """인증 없이 열어두는 생존 확인용. 상태만 반환한다."""
    try:
        r = await app.state.client.get("/api/version", timeout=3.0)
        return {"gateway": "ok", "ollama": r.json()}
    except Exception:
        return JSONResponse({"gateway": "ok", "ollama": "unreachable"}, status_code=503)


@app.get("/metrics")
async def metrics(authorization: str | None = Header(default=None)):
    check_auth(authorization)
    fields = "utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"nvidia-smi failed: {e}")

    v = [x.strip() for x in out.split(",")]
    return {
        "gpu_util_pct": float(v[0]),
        "vram_used_mib": float(v[1]),
        "vram_total_mib": float(v[2]),
        "temp_c": float(v[3]),
        "power_w": float(v[4]),
        "queue_slots_free": gpu_lock._value,
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, authorization: str | None = Header(default=None)):
    check_auth(authorization)
    body = await request.json()

    model = body.get("model")
    if model not in ALLOWED_MODELS:
        raise HTTPException(status_code=400, detail=f"model not allowed: {model}")

    stream = bool(body.get("stream", False))

    if not stream:
        async with gpu_lock:
            r = await app.state.client.post("/v1/chat/completions", json=body)
        return JSONResponse(r.json(), status_code=r.status_code)

    async def relay():
        async with gpu_lock:
            req = app.state.client.build_request("POST", "/v1/chat/completions", json=body)
            r = await app.state.client.send(req, stream=True)
            try:
                async for chunk in r.aiter_bytes():
                    yield chunk
            finally:
                await r.aclose()

    return StreamingResponse(relay(), media_type="text/event-stream")
