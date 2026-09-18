# review.py
"""最薄复核代理：鉴权/限流/预算/prompt组装/调用/校验/记账。"""
import json
import os
import time
import uuid
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from openai import OpenAI

APP_VER = "p1-v1"
MODEL = os.environ.get("REVIEW_MODEL", "qwen3-vl-flash")
BASE_URL = os.environ.get(
    "DASHSCOPE_BASE_URL",
    "https://ws-4yo7xzihx0z8lj5z.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
)
BUDGET = float(os.environ.get("DAILY_BUDGET_YUAN", "10.0"))
LEDGER = os.environ.get("LEDGER_PATH", "spool/review_ledger.jsonl")
# 按 token 估算成本（非真实账单）：total_tokens * 单价 / 1000
# DashScope qwen3-vl-flash 公开价约 0.0003 元/千 token（输入+输出合计口径），
# 真实账单以云控制台为准；这里只为让 _budget_check 能随用量增长而触发。
PRICE_PER_1K = float(os.environ.get("PRICE_PER_1K_TOKENS", "0.0003"))

client = OpenAI(
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url=BASE_URL,
)
app = FastAPI()
_tokens = set(json.loads(os.environ.get("TOKENS_JSON", '["test-token"]')))
_day, _spent, _count = "", 0.0, 0
_rate = {}

# 注意：JSON 示例里的大括号必须写成 {{ }}，因为下面用了 .format(meta=...)
PROMPT = """你是混凝土/沥青/砖石结构巡检领域的视觉质检助手。
图像是端侧裂缝检测模型从现场画面中裁剪出的候选区域，该模型的置信度不高，需要你复核。

请判断图像中的疑似目标是否为【真实裂缝】——即混凝土、沥青、砖石等结构表面的开裂。
真实裂缝的典型特征：
- 深色、不规则、有深度感的线状/网状缺陷
- 边缘清晰，可能有错台、骨料暴露或剥落
- 贯穿性或延伸性，不是表面均匀纹理

请排除以下假阳性：
阴影、水渍、油污、施工缝、伸缩缝、瓷砖/砖块接缝、线缆、划痕、
镜头脏污、光照边缘、涂料龟裂、墙面装饰纹理、抹灰层收缩纹、
以及任何无深度感的表面均匀纹理。

判定要求：
1. 只依据图像可见证据，不猜测画面外信息；
2. 若图像模糊或证据不足，is_crack 填 "uncertain"，不要强行二选一；
3. confidence 是你对自己判定的把握（0~1），不是裂缝的严重度。

严格输出如下 JSON（不要输出任何其他文字、不要用 markdown 代码块包裹）：
{{"is_crack": "yes"|"no"|"uncertain",
 "confidence": 0.0~1.0,
 "reason": "一句话中文理由，≤40字"}}

端侧元数据（供参考，可能与你的判断冲突，以你的视觉判断为准）：
{meta}"""


class Req(BaseModel):
    device_id: str
    client_ts: float = 0
    image_b64: str
    meta: dict = {}


def _budget_check():
    global _day, _spent, _count
    today = time.strftime("%Y%m%d")
    if today != _day:
        _day, _spent, _count = today, 0.0, 0
    if _spent > BUDGET:
        raise HTTPException(429, "daily budget exhausted", headers={"Retry-After": "3600"})


def _rate_check(dev):
    now = time.time()
    win = [t for t in _rate.get(dev, []) if now - t < 1.0]
    if len(win) >= 2:
        raise HTTPException(429, "rate limited", headers={"Retry-After": "10"})
    win.append(now)
    _rate[dev] = win


def _parse_vlm(text):
    t = text.strip()
    if t.startswith("```json"):
    	t = t[len("```json"):]
    elif t.startswith("```"):
    	t = t[len("```"):]
    if t.endswith("```"):
    	t = t[:-len("```")]
    t = t.strip()
    try:
        obj = json.loads(t)
    except Exception:
        s, e = t.find("{"), t.rfind("}")
        if s < 0 or e <= s:
            raise ValueError("no json object")
        obj = json.loads(t[s:e + 1])
    assert obj["is_crack"] in ("yes", "no", "uncertain")
    c = float(obj["confidence"])
    assert 0.0 <= c <= 1.0
    obj["confidence"] = c
    return obj


def _estimate_cost(usage):
    """按 total_tokens * PRICE_PER_1K 估算（非真实账单，仅用于预算闸门）。"""
    if not usage:
        return 0.0
    total = usage.get("total_tokens") or 0
    if not total:
        # 兼容部分 SDK 把明细放在 completion_tokens_details 里
        total = (usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0)
    return total * PRICE_PER_1K / 1000.0


def _ledger(rec):
    global _spent, _count
    os.makedirs(os.path.dirname(LEDGER) or ".", exist_ok=True)
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    _spent += rec.get("cost_yuan", 0.0)
    _count += 1


@app.post("/review")
def review(req: Req, authorization: str = Header("")):
    auth = authorization
    if auth.startswith("Bearer "):
        auth = auth[len("Bearer "):]
    if auth.strip() not in _tokens:
        raise HTTPException(401, "bad device token")
    _budget_check()
    _rate_check(req.device_id)

    # ===== 仅测试用：故障注入（不设 FAIL_MODE 则完全无影响，生产留空即可）=====
    if os.environ.get("FAIL_MODE") == "500":
        raise HTTPException(500, "simulated failure")
    if os.environ.get("FAIL_MODE") == "timeout":
        time.sleep(35)  # 超过 cloud_bridge 的 30s 超时
    # =====================================================================

    rid = uuid.uuid4().hex[:12]
    t0 = time.time()
    prompt = PROMPT.format(meta=json.dumps(req.meta, ensure_ascii=False))
    status, parsed, raw, usage = "ok", None, "", {}
    try:
        rsp = client.chat.completions.create(
            model=MODEL,
            temperature=0.1,
            messages=[{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{req.image_b64}"}},
                {"type": "text", "text": prompt}
            ]}])
        raw = rsp.choices[0].message.content
        usage = rsp.usage.model_dump() if rsp.usage else {}
        parsed = _parse_vlm(raw)
    except ValueError as e:
        status = f"parse_error:{e}"
    except Exception as e:
        status = f"api_error:{type(e).__name__}:{e}"

    decision = "keep_edge"
    if parsed:
        if parsed["is_crack"] == "no" and parsed["confidence"] >= 0.85:
            decision = "downgrade"
        elif parsed["is_crack"] == "yes":
            decision = "upgrade"
        elif parsed["is_crack"] == "uncertain":
            decision = "human_queue"

    _ledger({
        "rid": rid, "device": req.device_id, "ts": time.time(),
        "model": MODEL, "prompt_version": APP_VER, "status": status,
        "decision": decision, "verdict": parsed, "raw": raw[:2000],
        "usage": usage, "latency_ms": int((time.time() - t0) * 1000),
        "cost_yuan": _estimate_cost(usage),
    })
    return {
        "rid": rid, "review_status": status.split(":")[0],
        "decision": decision, "verdict": parsed,
        "model": MODEL, "prompt_version": APP_VER,
    }


@app.get("/healthz")
def healthz():
    return {"ok": True, "day": _day, "count": _count, "spent": _spent}
