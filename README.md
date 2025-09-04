# ric-energy-rapp
An O-RAN rApp for energy saving, using O1 performance data for traffic prediction and decision making, and generating A1 policies for integration with Near-RT RIC.

# RIC Energy Saving rApp (Demo)

本專案是一個 rApp 範例，功能為：
- 從 O1 / PM API 拉取 RU 性能數據
- 進行特徵工程與 LSTM 預測
- 依門檻條件決策（downshift/restore/hold）
- 輸出 A1 Policy Payload，供 Near-RT RIC / xApp 使用

## 專案結構
- `app/` : FastAPI 主程式
- `src/` : ETL、模型推論、決策邏輯
- `models/` : 已訓練的 LSTM 模型 (`model.onnx`, `model_meta.json`)
- `data/` : 輸出資料夾（預設不會 commit 實際檔案）
- `scripts/` : 測試腳本（PowerShell/其他）

## 環境需求
- Python 3.10+
- Docker (可選)
- 依賴套件見 [requirements.txt](requirements.txt)

## 本地執行
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
