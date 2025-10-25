# server/app_server.py
from __future__ import annotations  # ⬅️ 이 줄을 최상단으로

import time
import os
from fastapi import FastAPI, Query
from pydantic import BaseModel
from dotenv import load_dotenv

# 로깅/에러 템플릿 (있으면 사용, 없으면 간단 대체)
try:
    from runtime.logging_utils import log  # with_timing 안 씀
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("app")

try:
    from runtime.error_templates import msg
except Exception:
    def msg(key: str, lang: str = "ko") -> str:
        return "요청 처리 중 오류가 발생했습니다." if lang == "ko" else "An error occurred."

load_dotenv()

# Phase 1/2/3 핵심 모듈들
import app as core
import phase3_agents_mcp as agents

app = FastAPI(title="CrewFin-AI", version="4.0")

@app.middleware("http")
async def timing_mw(request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
        return response
    finally:
        dur_ms = int((time.perf_counter() - start) * 1000)
        try:
            log.info("http_timing", path=str(request.url.path), ms=dur_ms)
        except Exception:
            pass

class ReportResponse(BaseModel):
    data: dict
    markdown: str

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/readyz")
def readyz():
    ok = bool(os.getenv("TAVILY_API_KEY"))
    return {"ready": ok}

@app.get("/report", response_model=ReportResponse)
def report(company: str = Query(...), lang: str = Query("ko"), local_fallback: bool = Query(False)):
    try:
        if local_fallback or not os.getenv("OPENAI_API_KEY"):
            rep, md = core.generate_report(company, lang)
            return {"data": rep.model_dump(), "markdown": md}
        else:
            out = agents.run_crew(company, lang)
            if "json" in out:
                return {"data": out["json"], "markdown": out["markdown"]}
            return {"data": out.get("data", {}), "markdown": out.get("markdown", "")}
    except Exception as e:
        log.error("report_error", extra={"err": str(e)})
        return {"data": {"error": msg("external_error", lang)}, "markdown": msg("external_error", lang)}
