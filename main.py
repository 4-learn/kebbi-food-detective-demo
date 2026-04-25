from pathlib import Path
import os
import re
import secrets

from fastapi import Depends, FastAPI, HTTPException, Header, Request, Response
from fastapi.responses import FileResponse, HTMLResponse
from openai import OpenAI
from pydantic import BaseModel

app = FastAPI()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
API_KEY = os.environ.get("API_KEY", "")
LOGIN_USER = os.environ.get("LOGIN_USER", "")
LOGIN_PASS = os.environ.get("LOGIN_PASS", "")
SESSION_COOKIE_NAME = "liyu_session"
SESSION_TOKEN = os.environ.get("SESSION_TOKEN") or secrets.token_urlsafe(32)
BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "frontend" / "index.html"
PROMPTS_DIR = BASE_DIR / "prompts"
PROMPTS_DIR.mkdir(exist_ok=True)
DEFAULT_SYSTEM_PROMPT = "你是友善的食物過敏小幫手「Dr. 凱比博士」。\n\n你的任務是根據使用者提供的照片、食物名稱、以及「過敏原目標」，用簡單、清楚、適合小學生閱讀的中文回答。\n\n你必須永遠遵守下面規則：\n\n1. 回答一定只能用三段格式，順序固定，標題固定，不可多也不可少：\n① 圖中食物\n② 可能過敏原\n③ 風險等級\n\n2. 每次回答都一定要真的輸出這三段，即使資訊不足也不能省略。\n\n2-1. 如果照片內容明顯不是食物，例如車子、玩具、文具、衣服、動物、風景、人物用品等：\n① 一定要寫「這不是食物，請拍食物照片」。\n② 要寫目前沒有食物可分析，所以無法判斷過敏原。\n③ 一定要寫「不適用（這不是食物）」；這種情況不能寫 🔴 高風險、🟡 可能有風險、或 🟢 暫時看起來較低風險。\n\n3. 不要輸出 Markdown，不要用 `**`、`#`、`-`、`[]()`、表格或 HTML。\n\n4. 每一段都要用短句、換行清楚，讓小學生看得懂。\n\n5. 如果使用者有提供「過敏原目標」，你必須逐一檢查每個目標過敏原，不能漏掉。\n\n6. 如果圖片模糊、看不清楚、資訊不足，不能亂猜，要明確說「我不確定，請重拍更清楚的照片」。\n\n6-1. 就算圖片模糊、看不清楚、資訊不足，也還是要完整輸出三段：\n① 圖中食物：說明看不清楚，可能是什麼，或無法判斷。\n② 可能過敏原：說明目前無法確認，但可列出如果判斷正確時可能相關的過敏原，並清楚標示不確定。\n③ 風險等級：只能在 🔴 高風險 / 🟡 可能有風險 / 🟢 暫時看起來較低風險 中選一個，並用一句話說明是因為照片模糊或資訊不足，所以需要重拍或請大人確認。\n\n7. 第③段一定要給出風險等級，只能用這三種：\n🔴 高風險\n🟡 可能有風險\n🟢 暫時看起來較低風險\n\n8. 風險判斷原則：\n- 只要食物明顯含有使用者選到的過敏原，或高度可能含有，就用 🔴 高風險。\n- 如果看起來可能含有，但照片不夠清楚、成分不明、或只是常見混入，就用 🟡 可能有風險。\n- 如果目前看不出明顯相關過敏原，才可用 🟢 暫時看起來較低風險。\n- 不能把「安全、可以放心吃、完全沒問題」說得太滿。\n\n9. 語氣要溫和、鼓勵、小朋友友善，可以用少量 emoji，但不要太多。\n\n10. 不要要求使用者自己做醫療判斷；若風險高或很不確定，要提醒先不要吃、再請大人幫忙確認。\n\n請嚴格依照這個回答模板輸出：\n\n① 圖中食物\n先說你看到的是什麼食物。\n如果看不清楚，要直接寫看不清楚，並說明可能是什麼。\n如果明顯不是食物，要直接寫「這不是食物，請拍食物照片」。\n\n② 可能過敏原\n列出這份食物可能包含的過敏原。\n如果有「過敏原目標」，一定要逐一點名是否看起來含有、可能含有、或目前看不出來。\n如果不是食物，要直接說目前沒有食物可分析，所以無法判斷過敏原。\n\n③ 風險等級\n如果是食物，先寫一行風險等級：🔴 高風險 / 🟡 可能有風險 / 🟢 暫時看起來較低風險\n如果不是食物，先寫一行風險等級：不適用（這不是食物）\n下一行用一句簡短原因解釋。\n最後加一句小朋友聽得懂的提醒，例如：先不要吃、請大人幫忙看、請重拍照片、或請拍真正的食物照片。\n\n不管使用者問什麼，都要維持這三段格式，不可改成別種格式。"
GROUP_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def verify_access(request: Request, x_api_key: str | None = Header(default=None)):
    if API_KEY and x_api_key == API_KEY:
        return
    if request.cookies.get(SESSION_COOKIE_NAME) == SESSION_TOKEN:
        return
    raise HTTPException(status_code=401, detail="Unauthorized")


def require_api_key(x_api_key: str | None = Header(default=None)):
    if not API_KEY or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


def load_system_prompt(group: str | None) -> str:
    if not group or not GROUP_RE.match(group):
        return DEFAULT_SYSTEM_PROMPT
    f = PROMPTS_DIR / f"{group}.txt"
    if not f.exists():
        return DEFAULT_SYSTEM_PROMPT
    try:
        return f.read_text().strip() or DEFAULT_SYSTEM_PROMPT
    except OSError:
        return DEFAULT_SYSTEM_PROMPT


class ChatRequest(BaseModel):
    message: str
    image_data_url: str | None = None
    group: str | None = None


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
        user_content: list[dict] = [{"type": "text", "text": req.message}]
        if req.image_data_url:
            user_content.append({"type": "image_url", "image_url": {"url": req.image_data_url}})

        sys_prompt = load_system_prompt(req.group)
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content},
        ]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        return ChatResponse(reply=response.choices[0].message.content)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/prompts/{group}", dependencies=[Depends(require_api_key)])
def read_prompt(group: str):
    if not GROUP_RE.match(group):
        raise HTTPException(status_code=400, detail="Invalid group identifier")
    f = PROMPTS_DIR / f"{group}.txt"
    if not f.exists():
        return {"group": group, "content": ""}
    return {"group": group, "content": f.read_text()}


@app.post("/prompts/{group}", dependencies=[Depends(require_api_key)])
async def update_prompt(group: str, request: Request):
    if not GROUP_RE.match(group):
        raise HTTPException(status_code=400, detail="Invalid group identifier")
    body = (await request.body()).decode("utf-8")
    f = PROMPTS_DIR / f"{group}.txt"
    f.write_text(body)
    return {"ok": True, "group": group, "bytes": len(body)}
