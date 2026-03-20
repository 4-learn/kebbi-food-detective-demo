from pathlib import Path
from fastapi import Depends, FastAPI, HTTPException, Header, Request, Response
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from openai import OpenAI
import os
import secrets

app = FastAPI()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
API_KEY = os.environ.get("API_KEY", "")
LOGIN_USER = os.environ.get("LOGIN_USER", "")
LOGIN_PASS = os.environ.get("LOGIN_PASS", "")
SESSION_COOKIE_NAME = "liyu_session"
SESSION_TOKEN = os.environ.get("SESSION_TOKEN") or secrets.token_urlsafe(32)
BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "frontend" / "index.html"


def verify_access(request: Request, x_api_key: str | None = Header(default=None)):
    if API_KEY and x_api_key == API_KEY:
        return

    if request.cookies.get(SESSION_COOKIE_NAME) == SESSION_TOKEN:
        return

    raise HTTPException(status_code=401, detail="Unauthorized")


class ChatRequest(BaseModel):
    message: str


class LoginRequest(BaseModel):
    username: str
    password: str


class ChatResponse(BaseModel):
    reply: str


@app.get("/", include_in_schema=False)
def index():
    if INDEX_FILE.exists():
        return FileResponse(INDEX_FILE)
    return HTMLResponse("<h1>Frontend not found</h1>", status_code=404)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/login")
def login(req: LoginRequest, response: Response):
    if not LOGIN_USER or not LOGIN_PASS:
        raise HTTPException(status_code=500, detail="Login credentials are not configured")

    if req.username != LOGIN_USER or req.password != LOGIN_PASS:
        raise HTTPException(status_code=401, detail="帳號或密碼錯誤")

    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=SESSION_TOKEN,
        httponly=True,
        samesite="lax",
        max_age=86400,
    )
    return {"ok": True}


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_access)])
def chat(req: ChatRequest):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": req.message}],
        )
        return ChatResponse(reply=response.choices[0].message.content)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
