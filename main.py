from pathlib import Path
from fastapi import Depends, FastAPI, HTTPException, Header, Request, Response
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from openai import OpenAI
import os
import secrets
import json

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
    message: str = ""
    user_message: str = ""
    targets: list[str] = []
    image_source: str = ""
    image_data_url: str = ""


class LoginRequest(BaseModel):
    username: str
    password: str


class ChatResponse(BaseModel):
    reply: str
    detected_subject: str = "未知"
    target_match: bool = False
    matched_targets: list[str] = []
    unmatched_targets: list[str] = []
    confidence: float = 0.0
    risk_level: str = "中"
    reason: str = ""
    next_actions: list[str] = []


def extract_json_object(text: str) -> dict | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        candidate = stripped[start : end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return None
    return None


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
        raw_targets = req.targets or []
        targets = "、".join(raw_targets) if raw_targets else "未指定"
        user_message = req.user_message or req.message or "請先做初步判斷與需要補充資訊"
        is_human_joke_mode = "人類" in raw_targets
        system_text = (
            "你是食物影像與過敏原測試助理。"
            "你必須先辨識圖片主體，再與使用者 target 比對。"
            "請只輸出 JSON，不要輸出 markdown。"
            "JSON schema: "
            '{"detected_subject":"string","target_match":true,'
            '"matched_targets":["string"],"unmatched_targets":["string"],'
            '"confidence":0.0,"risk_level":"低|中|高","reason":"string","next_actions":["string"]}.'
        )
        if is_human_joke_mode:
            system_text += (
                "若 target 含有「人類」，請用輕鬆搞笑語氣，"
                "重點是主體比對，不要使用醫療警語或過度嚴肅措辭；"
                "風險等級預設為低，除非畫面完全無法判斷。"
            )
        user_text = (
            f"過敏原測試目標：{targets}\n"
            f"圖片來源：{req.image_source or '未提供'}\n"
            f"使用者訊息：{user_message}"
        )

        content: list[dict] = [{"type": "text", "text": user_text}]
        if req.image_data_url:
            content.append({"type": "image_url", "image_url": {"url": req.image_data_url}})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_text},
                {"role": "user", "content": content},
            ],
        )
        raw_reply = response.choices[0].message.content or ""
        parsed = extract_json_object(raw_reply) or {}

        detected_subject = str(parsed.get("detected_subject", "未知"))
        target_match = bool(parsed.get("target_match", False))
        matched_targets = [str(x) for x in parsed.get("matched_targets", []) if isinstance(x, (str, int, float))]
        unmatched_targets = [str(x) for x in parsed.get("unmatched_targets", []) if isinstance(x, (str, int, float))]
        try:
            confidence = float(parsed.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        risk_level = str(parsed.get("risk_level", "中"))
        reason = str(parsed.get("reason", raw_reply))
        next_actions = [str(x) for x in parsed.get("next_actions", []) if isinstance(x, (str, int, float))]

        # If model doesn't return valid matched/unmatched lists, generate a deterministic fallback.
        if raw_targets and not (matched_targets or unmatched_targets):
            if target_match:
                matched_targets = raw_targets
            else:
                unmatched_targets = raw_targets

        if not next_actions:
            next_actions = ["請改拍更清晰的正面包裝與成分表，再重新判斷。"]

        reply_text = (
            f"判定主體：{detected_subject}\n"
            f"是否符合目標：{'是' if target_match else '否'}\n"
            f"信心：{int(confidence * 100)}%\n"
            f"風險等級：{risk_level}\n"
            f"判斷依據：{reason}"
        )

        return ChatResponse(
            reply=reply_text,
            detected_subject=detected_subject,
            target_match=target_match,
            matched_targets=matched_targets,
            unmatched_targets=unmatched_targets,
            confidence=confidence,
            risk_level=risk_level,
            reason=reason,
            next_actions=next_actions,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
