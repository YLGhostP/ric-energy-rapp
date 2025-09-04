# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import os, json, pandas as pd

from src.pm_adapter import fetch_pm
from src.policy_adapter import build_policy, post_policy
from src.etl import run_etl
from src.lstm_multi import predict_next_window
from src.decision import decide

from datetime import datetime, timezone
from pathlib import Path
import hashlib


MODEL_PATH = os.getenv("MODEL_PATH", "models/model.onnx")
META_PATH  = os.getenv("META_PATH",  "models/model_meta.json")

app = FastAPI(title="Energy-Saving rApp API", version="0.2.0")

class ForecastReq(BaseModel):
    minutes: int = 1440               # 沒有 PM_API_URL 時使用
    start: Optional[str] = None       # 有 PM_API_URL 時：ISO8601 UTC
    end:   Optional[str] = None

class DecisionReq(BaseModel):
    minutes: int = 1440
    t_low: float = 0.25
    t_high: float = 0.50
    dt_min: int = 10
    cooldown: int = 20
    targetRU: str = "RU_001"
    expire_minutes: int = 30
    confidence: float = 0.8
    dry_run: bool = False                  # ← 新增：乾跑，不對外推送
    idempotency_key: Optional[str] = None  # ← 新增：冪等鍵，避免重覆下發


class IngestReq(BaseModel):
    start: str
    end: str

@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "model_exists": Path(MODEL_PATH).exists(),
        "meta_exists": Path(META_PATH).exists(),
        "pm_api_mode": bool(os.getenv("PM_API_URL","").strip()),
        "a1_url_set": bool(os.getenv("A1_URL","").strip())
    }

@app.post("/ingest_pm")
def ingest_pm(req: IngestReq):
    raw_path = fetch_pm(start_iso=req.start, end_iso=req.end)
    return {"raw_path": raw_path}

@app.post("/forecast")
def forecast(req: ForecastReq):
    # 有 PM_API_URL → 從實驗室拉資料；否則 minutes 造資料
    raw_path = fetch_pm(start_iso=req.start, end_iso=req.end) if os.getenv("PM_API_URL") else \
               fetch_pm(start_iso=None, end_iso=None)  # 內部會走 make_raw
    feat_path = run_etl(raw_path)
    fcst_path = predict_next_window(
        feat_path=feat_path, onnx_path=MODEL_PATH, meta_path=META_PATH,
        out_path="data/processed/forecast.csv"
    )
    fc = pd.read_csv(fcst_path, parse_dates=["timestamp"])
    return {
        "forecast_path": fcst_path,
        "points": len(fc),
        "timestamps": fc["timestamp"].astype(str).tolist(),
        "p10": fc["p10"].round(6).tolist(),
        "p50": fc["p50"].round(6).tolist(),
        "p90": fc["p90"].round(6).tolist()
    }

@app.post("/decision")
def decision(req: DecisionReq):
    # -------- 0) 冪等檢查（同一 idempotency_key 只允許執行一次） --------
    if req.idempotency_key:
        idem_dir = Path("data/idempotency"); idem_dir.mkdir(parents=True, exist_ok=True)
        idem_flag = idem_dir / (hashlib.sha256(req.idempotency_key.encode()).hexdigest() + ".flag")
        if idem_flag.exists():
            return {"error": "duplicate idempotency_key", "idempotency_key": req.idempotency_key}
        idem_flag.write_text(datetime.now(timezone.utc).isoformat())

    # -------- 1) 產生/拉取資料 → ETL → LSTM 推論 --------
    raw_path = fetch_pm(start_iso=None, end_iso=None)   # 有/無 PM_API_URL 都會得到 raw.csv
    feat_path = run_etl(raw_path)
    fcst_path = predict_next_window(
        feat_path=feat_path, onnx_path=MODEL_PATH, meta_path=META_PATH,
        out_path="data/processed/forecast.csv"
    )

    # -------- 2) 決策 --------
    dec_path = decide(
        forecast_path=fcst_path, t_low=req.t_low, t_high=req.t_high,
        dt_min=req.dt_min, cooldown=req.cooldown
    )
    decision_dict = json.loads(Path(dec_path).read_text())

    # -------- 3) 組策略 payload（轉 A1/平台格式）--------
    payload = build_policy(
        decision=decision_dict,
        target_ru=req.targetRU,
        expire_minutes=req.expire_minutes,
        confidence=req.confidence
    )

    # -------- 4) 推送 or 乾跑 --------
    if req.dry_run:
        push_res = {"sent": False, "dry_run": True, "saved_to": "data/processed/policy_payload.json"}
        # 仍保留一份可檢視的 payload（不對外送）
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        Path("data/processed/policy_payload.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2)
        )
    else:
        # 若未設 A1_URL，post_policy 會落地寫 data/processed/policy_payload.json
        push_res = post_policy(payload)

    # -------- 5) 審計與追加日誌（同時保留覆蓋與歷史）--------
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    audit_id = f"{ts}-{req.targetRU}-a87"  # a87：你的批次/實驗代碼，可自行更改
    audit_dir = Path("data/audit"); audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / f"{audit_id}.json").write_text(json.dumps({
        "audit_id": audit_id,
        "ts": ts,
        "request": {
            "t_low": req.t_low, "t_high": req.t_high,
            "dt_min": req.dt_min, "cooldown": req.cooldown,
            "targetRU": req.targetRU, "dry_run": req.dry_run,
            "expire_minutes": req.expire_minutes, "confidence": req.confidence,
            "idempotency_key": req.idempotency_key
        },
        "decision": decision_dict,
        "policy_payload": payload,
        "push_result": push_res
    }, ensure_ascii=False, indent=2))

    # 追加一行到 logs（JSON Lines）
    logs_dir = Path("logs"); logs_dir.mkdir(parents=True, exist_ok=True)
    log_line = {
        "ts": ts,
        "targetRU": req.targetRU,
        "decision": decision_dict.get("action"),
        "reason": decision_dict.get("reason"),
        "mode_after": decision_dict.get("mode_after"),
        "push_sent": bool(push_res.get("sent")),
    }
    with open(logs_dir / "decision_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_line, ensure_ascii=False) + "\n")

    return {
        "audit_id": audit_id,
        "decision": decision_dict,
        "policy_payload": payload,
        "push_result": push_res
    }
