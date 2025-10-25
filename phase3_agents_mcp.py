"""
Phase 3.5 — Agents with optional MCP (Tavily MCP) integration
- USE_MCP=true 환경변수일 때 crewai_tools.TavilySearchTool 사용 (MCP 경유)
- 그렇지 않으면 기존 REST 파이프라인(app.py) 사용

실행 예시
  USE_MCP=true python phase3_agents_mcp.py --company "엔비디아" --lang ko --pretty
  python phase3_agents_mcp.py --company "엔비디아" --lang ko --pretty --local-fallback

사전 준비
- .crewai/config.json 에 tavily-remote-mcp 등록 (이미 제공됨)
- tools/mcp_setup.sh 로 npx mcp-remote 실행 가능
- pip install crewai crewai-tools requests python-dotenv
"""
from __future__ import annotations

import os
import json
import argparse
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()

USE_MCP = os.getenv("USE_MCP", "false").lower() in {"1","true","yes"}
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# --- 기존 코어 로직 (Phase1/2) 불러오기 ---
import app as core

# --- CrewAI ---------------------------------------------------------------
from crewai import Agent, Task, Crew, Process

MODEL_NLP = getattr(core, "MODEL_NLP", "gpt-4o-mini")

# --- MCP Tool (옵션) ------------------------------------------------------
_MCP_AVAILABLE = False
try:
    if USE_MCP:
        from crewai_tools import TavilySearchTool
        _MCP_AVAILABLE = True
except Exception:
    _MCP_AVAILABLE = False

# -------------------------------------------------------------------------
# 도구 정의
# -------------------------------------------------------------------------

def tool_collect_and_analyze(company: str, lang: str) -> str:
    """기존 REST 파이프라인: 실데이터 수집 + 감성 집계."""
    tk = core.convert_company_name_to_ticker(company, lang or core.simple_lang_detect(company))
    market = tk.market or ("KR" if (lang == "ko") else "US")
    coll = core.collect_data(tk.company_name or company, market=market, language=lang)
    csa = core.aggregate_sentiment(coll.news, coll.community)
    payload = {
        "company": {
            "name": tk.company_name or company,
            "ticker": tk.primary_ticker,
            "market": tk.market,
        },
        "csa": csa.model_dump(),
        "news": [{"title": it.title, "url": it.url} for it in coll.news],
        "lang": lang,
    }
    return json.dumps(payload, ensure_ascii=False)


def tool_collect_mcp_then_analyze(company: str, lang: str) -> str:
    """MCP(TavilySearchTool)로 우선 검색 → 부족하면 REST 데이터 병합 → 감성 집계."""
    if not _MCP_AVAILABLE:
        return tool_collect_and_analyze(company, lang)

    try:
        # 1) MCP 검색
        search = TavilySearchTool()
        query = f"{company} 주식 뉴스" if (lang == "ko") else f"{company} stock news"
        mcp_raw = search.run(query)

        # 2) MCP 결과 파싱 (유연 처리)
        mcp_items: List[Dict[str, str]] = []
        try:
            obj = json.loads(mcp_raw)
            items = obj.get("results") or obj.get("data") or obj
            if isinstance(items, list):
                for it in items:
                    title = it.get("title") or it.get("name") or it.get("headline")
                    url = it.get("url") or it.get("link")
                    if title and url:
                        mcp_items.append({"title": title, "url": url})
        except Exception:
            for line in str(mcp_raw).splitlines():
                if "http" in line:
                    url = line.strip().split()[-1]
                    mcp_items.append({"title": line.strip()[:80], "url": url})

        # 3) REST 파이프라인으로 수집/정규화/커뮤니티 보강
        tk = core.convert_company_name_to_ticker(company, lang or core.simple_lang_detect(company))
        market = tk.market or ("KR" if (lang == "ko") else "US")
        coll = core.collect_data(tk.company_name or company, market=market, language=lang)

        # 4) 뉴스 목록 병합 (MCP 우선)
        rest_items = [{"title": it.title, "url": it.url} for it in coll.news]
        if mcp_items:
            ahead = mcp_items[:5]
            merged = ahead + [x for x in rest_items if x not in ahead]
        else:
            merged = rest_items

        # 5) 감성 집계
        csa = core.aggregate_sentiment(coll.news, coll.community)
        payload = {
            "company": {
                "name": tk.company_name or company,
                "ticker": tk.primary_ticker,
                "market": tk.market,
            },
            "csa": csa.model_dump(),
            "news": merged,
            "lang": lang,
        }
        return json.dumps(payload, ensure_ascii=False)
    except Exception:
        # MCP 실패 시 완전 폴백
        return tool_collect_and_analyze(company, lang)


# -------------------------------------------------------------------------
# Crew 구성/실행
# -------------------------------------------------------------------------

def build_crew(company: str, lang: Optional[str]) -> Crew:
    language = lang or core.simple_lang_detect(company)

    analyst_community = Agent(
        role="Community & News Sentiment Analyst",
        goal=(
            "Collect real-world news & community posts for the given company, "
            "run sentiment aggregation, then provide a compact JSON payload for downstream."
        ),
        backstory=(
            "You specialize in synthesizing market sentiment from both news and investor communities. "
            "Prefer calling tools and returning structured JSON over free-form writing."
        ),
        allow_delegation=True,
        tools=[tool_collect_mcp_then_analyze if _MCP_AVAILABLE else tool_collect_and_analyze],
        verbose=True,
        llm=MODEL_NLP if OPENAI_API_KEY else None,
    )

    analyst_writer = Agent(
        role="Financial Report Writer & Market Analyst",
        goal=(
            "Transform the partial JSON into a polished analyst-grade report (JSON + Markdown)."
        ),
        backstory=(
            "Experienced sell-side analyst who can turn sentiment summaries into investor-friendly insights."
        ),
        allow_delegation=True,
        tools=[tool_compose_report],
        verbose=True,
        llm=MODEL_NLP if OPENAI_API_KEY else None,
    )

    t1 = Task(
        description=(
            "Use the tool to collect & aggregate sentiment. Return STRICT JSON from the tool only.\n"
            f"Input company: {company}\nInput language: {language}\n"
        ),
        expected_output=(
            "A JSON string with keys: company{name,ticker,market}, csa, news(list of {title,url}), lang."
        ),
        tools=[tool_collect_mcp_then_analyze if _MCP_AVAILABLE else tool_collect_and_analyze],
        agent=analyst_community,
    )

    t2 = Task(
        description=(
            "Take Task 1 output (the JSON string) and pass it to compose tool. "
            "Return STRICT JSON {json: <report_obj>, markdown: <md>} from the tool only."
        ),
        expected_output=(
            "A JSON with keys: json (full report dict by PRD v3), markdown (Telegram-ready Markdown)."
        ),
        tools=[tool_compose_report],
        agent=analyst_writer,
    )

    crew = Crew(
        agents=[analyst_community, analyst_writer],
        tasks=[t1, t2],
        process=Process.sequential,
        verbose=True,
    )
    return crew


def tool_compose_report(partial_json: str) -> str:
    obj = json.loads(partial_json)
    lang = obj.get("lang") or "ko"
    csa = core.CommunitySentimentAnalysis(**obj["csa"])  # pydantic 복구
    ac = core.draft_analyst_commentary(csa, lang)
    overall = core.one_liner_summary(csa, lang)
    report = core.ReportModel(
        report_generated_at=core.now_kst_iso(),
        language=lang,
        company_info=obj.get("company", {}),
        community_sentiment_analysis=csa,
        analyst_commentary=ac,
        overall_summary=overall,
        disclaimer=(
            "본 리포트는 AI에 의해 생성된 정보로서 일반적 참고용이며, 투자 조언이 아닙니다. 투자 결정은 사용자 책임입니다." if lang=="ko" else
            "This report is AI‑generated for general information only and is not investment advice."
        ),
        sources=obj.get("news", []),
    )
    md = core.report_to_markdown(report)
    out = {"json": report.model_dump(), "markdown": md}
    return json.dumps(out, ensure_ascii=False)


def run_crew(company: str, lang: Optional[str] = None) -> Dict[str, Any]:
    language = lang or core.simple_lang_detect(company)
    crew = build_crew(company, language)
    result = crew.kickoff()
    try:
        out = json.loads(str(result))
        return out
    except Exception:
        rep, md = core.generate_report(company, language)
        return {"json": rep.model_dump(), "markdown": md}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3.5 Agents (optional MCP)")
    parser.add_argument("--company", required=True)
    parser.add_argument("--lang", default=None)
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--local-fallback", action="store_true")
    args = parser.parse_args()

    if args.local_fallback or not OPENAI_API_KEY:
        rep, md = core.generate_report(args.company, args.lang)
        out = {"json": rep.model_dump(), "markdown": md}
    else:
        out = run_crew(args.company, args.lang)

    print("\n=== JSON ===")
    if args.pretty:
        print(json.dumps(out["json"], ensure_ascii=False, indent=2))
    else:
        print(json.dumps(out["json"], ensure_ascii=False))
    print("\n=== Markdown ===")
    print(out["markdown"])

