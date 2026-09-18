# cloud_bridge.py
"""端侧上传守护进程：监视 spool 发件箱，串行上传灰区样本到云端复核。"""
import base64
import json
import os
import shutil
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

SPOOL = os.environ.get("SPOOL_DIR", "spool")
OUTBOX = os.path.join(SPOOL, "outbox")
DONE = os.path.join(SPOOL, "done")
FAILED = os.path.join(SPOOL, "failed")
CLOUD = os.environ.get("CLOUD_ENDPOINT", "http://localhost:8000/review")
TOKEN = os.environ.get("DEVICE_TOKEN", "test-token")
TIMEOUT = 30
MAX_RETRY = 5
MIN_INTERVAL = 0.5
POLL = 2.0
DISK_LIMIT_MB = 500

for d in (OUTBOX, DONE, FAILED):
    os.makedirs(d, exist_ok=True)

def spool_size_mb():
    total = 0
    for root, _, files in os.walk(SPOOL):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total / 1e6

def touch_pause(reason):
    with open(os.path.join(SPOOL, "PAUSE"), "w") as f:
        f.write(f"{time.time()} {reason}\n")

def clear_pause():
    p = os.path.join(SPOOL, "PAUSE")
    if os.path.exists(p):
        os.remove(p)

def post(url, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {TOKEN}"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return r.status, json.loads(r.read().decode())

def handle_one(job_dir):
    try:
        meta = json.load(open(os.path.join(job_dir, "meta.json"), encoding="utf-8"))
        img_b64 = base64.b64encode(
            open(os.path.join(job_dir, "crop.jpg"), "rb").read()).decode()
    except Exception as e:
        print(f"[bridge] 坏件 {job_dir}: {e}", flush=True)
        shutil.move(job_dir, os.path.join(FAILED, os.path.basename(job_dir)))
        return "failed"

    payload = {"meta": meta, "image_b64": img_b64,
               "device_id": os.environ.get("DEVICE_ID", "pc-replay-01"),
               "client_ts": time.time()}

    attempt = meta.get("_attempt", 0)
    try:
        status, result = post(CLOUD, payload)
        with open(os.path.join(job_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=1)
        shutil.move(job_dir, os.path.join(DONE, os.path.basename(job_dir)))
        return "done"
    except urllib.error.HTTPError as e:
        if e.code == 429:
            retry_after = int(e.headers.get("Retry-After", "60"))
            touch_pause(f"429 retry_after={retry_after}")
            time.sleep(min(retry_after, 60))
            clear_pause()
            return "pause"
        status = e.code
    except Exception as e:
        status = str(e)

    attempt += 1
    if attempt >= MAX_RETRY:
        meta["_attempt"] = attempt
        json.dump(meta, open(os.path.join(job_dir, "meta.json"), "w", encoding="utf-8"))
        shutil.move(job_dir, os.path.join(FAILED, os.path.basename(job_dir)))
        return "failed"
    meta["_attempt"] = attempt
    json.dump(meta, open(os.path.join(job_dir, "meta.json"), "w", encoding="utf-8"))
    backoff = min(2 ** attempt, 60)
    print(f"[bridge] 失败({status}), {backoff}s 后重试: {job_dir}", flush=True)
    time.sleep(backoff)
    return "retry"

def main():
    print(f"[bridge] 启动: outbox={OUTBOX} cloud={CLOUD}", flush=True)
    while True:
        if spool_size_mb() > DISK_LIMIT_MB:
            touch_pause("disk_watermark")
            dones = sorted(os.listdir(DONE))
            for name in dones[:max(1, len(dones) // 4)]:
                shutil.rmtree(os.path.join(DONE, name), ignore_errors=True)
        else:
            if os.path.exists(os.path.join(SPOOL, "PAUSE")):
                reason = open(os.path.join(SPOOL, "PAUSE")).read()
                if "disk" in reason:
                    clear_pause()
            jobs = sorted(os.listdir(OUTBOX))
            if jobs:
                handle_one(os.path.join(OUTBOX, jobs[0]))
                time.sleep(MIN_INTERVAL)
                continue
        time.sleep(POLL)

if __name__ == "__main__":
    main()