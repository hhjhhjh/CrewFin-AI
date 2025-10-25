"""
Phase 3 — CrewAI 에이전트 구조 (delegation)

개요
- Agent 1: 커뮤니티 감성 분석 (수집 + 집계) — tools를 통해 실데이터 호출
- Agent 2: 리포트 작성(애널리스트 코멘트 + 최종 Markdown) — tools로 조립
- allow_delegation=True, 다만 OpenAI 키(쿼터) 이슈가 있으면 로컬 폴백 경로로 안전하게 동작

요구 사항
- 프로젝트 루트에 기존 Phase 1/2의 app.py가 존재한다고 가정
- .env: OPENAI_API_KEY(선택), TAVILY_API_KEY, SERPER_API_KEY(선택)
- crewai, crewai-tools, requests 설치

실행 예시
    python phase3_agents.py --company "엔비디아" --lang ko --pretty

동작 모드
- 기본: CrewAI 에이전트 두 개로 실행
- OPENAI_API_KEY가 없거나 쿼터 오류 우려 시: --local-fallback 옵션으로 app.generate_report 직접 호출
"""

from __future__ import annotations

import os
import json
import argparse
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()

# --- 기존 Phase 1/2 모듈 가져오기 ---
try:
    import app as core  # app.py (Phase 1/2) 를 동일 프로젝트 루트에 두었다고 가정
except Exception as e:
    raise RuntimeError("app.py (Phase1/2) 가 프로젝트 루트에 필요합니다.") from e

# --- CrewAI ---------------------------------------------------------------
from crewai import Agent, Task, Crew, Process

# LLM 설정: 키 없으면 CrewAI 실행 시 모델 호출을 최소화하도록 프롬프트를 도구 중심으로 구성
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NLP = getattr(core, "MODEL_NLP", "gpt-4o-mini")

# --- Tools: crewai_tools 없이, 순수 파이썬 함수 호출 ----------------------
# CrewAI는 tools 파라미터로 callables를 받는다. (crewai>=0.74 기준)
# 반환은 문자열이므로, JSON은 dumps 해서 넘긴다.

def tool_collect_and_analyze(company: str, lang: str) -> str:
    """실데이터 수집(Phase2) + 감성 집계 → 부분 JSON을 문자열로 반환.
    반환 스키마:
    {
      "company": {"name": ..., "ticker": ..., "market": ...},
      "csa": CommunitySentimentAnalysis(model_dump),
      "news": [{"title":..., "url":...}, ...],
      "lang": "ko|en"
    }
    """
    # 티커 변환
    tk = core.convert_company_name_to_ticker(company, lang or core.simple_lang_detect(company))
    # 시장 추정
    market = tk.market or ("KR" if (lang == "ko") else "US")
    # 수집 (언어 힌트 전달)
    coll = core.collect_data(tk.company_name or company, market=market, language=lang)
    # 집계
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


def tool_compose_report(partial_json: str) -> str:
    """Agent2가 호출: 부분 JSON을 받아 최종 Report JSON+Markdown을 만들어 반환.
    반환: {"json": <ReportModel.dict>, "markdown": "..."}
    """
    obj = json.loads(partial_json)
    lang = obj.get("lang") or "ko"

    # CSA 역직렬화
    csa = core.CommunitySentimentAnalysis(**obj["csa"])  # pydantic 복구

    # 애널리스트 코멘트/요약은 core 함수 사용(LLM 있으면 모델 사용, 없으면 템플릿)
    ac = core.draft_analyst_commentary(csa, lang)
    overall = core.one_liner_summary(csa, lang)

    # Report 조립
    report = core.ReportModel(
        report_generated_at=core.now_kst_iso(),
        language=lang,
        company_info=obj.get("company", {}),
        community_sentiment_analysis=csa,
        analyst_commentary=ac,
        overall_summary=overall,
        disclaimer=("본 리포트는 AI에 의해 생성된 정보로서 일반적 참고용이며, 투자 조언이 아닙니다. 투자 결정은 사용자 책임입니다." if lang=="ko" else
                    "This report is AI‑generated for general information only and is not investment advice."),
        # 간단히 상위 뉴스 링크 포함
        sources=obj.get("news", []),
    )
    md = core.report_to_markdown(report)
    out = {"json": report.model_dump(), "markdown": md}
    return json.dumps(out, ensure_ascii=False)


# --- Crew 세팅 ------------------------------------------------------------

def build_crew(company: str, lang: Optional[str]) -> Crew:
    """두 에이전트 + 두 태스크 구성. delegation 활성화.
    Agent1은 도구(tool_collect_and_analyze)만 사용하도록 유도하여 LLM 사용을 최소화.
    Agent2는 도구(tool_compose_report)로 최종 출력 생성.
    """
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
        tools=[tool_collect_and_analyze],
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
        tools=[tool_collect_and_analyze],
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


def run_crew(company: str, lang: Optional[str] = None) -> Dict[str, Any]:
    language = lang or core.simple_lang_detect(company)

    # Crew 실행 (OpenAI 키 없으면 crew 내부 LLM 사용이 최소화되지만, 도구만으로도 결과 생성됨)
    crew = build_crew(company, language)
    result = crew.kickoff()

    # result는 보통 마지막 태스크의 문자열 결과
    try:
        out = json.loads(str(result))
        return out
    except Exception:
        # 만약 에이전트가 예상 외 텍스트를 섞어 반환하면, 안전 폴백: core.generate_report 직접 호출
        rep, md = core.generate_report(company, language)
        return {"json": rep.model_dump(), "markdown": md}


# --- CLI -----------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3 CrewAI Agents runner")
    parser.add_argument("--company", required=True, help="회사명 또는 종목명")
    parser.add_argument("--lang", default=None, help="ko|en (미지정 시 자동 감지)")
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--local-fallback", action="store_true", help="에이전트 대신 Phase1/2 로컬 파이프라인을 직접 사용")
    args = parser.parse_args()

    if args.local_fallback or not OPENAI_API_KEY:
        # Crew를 쓰지 않고 바로 코어 파이프라인 실행 (429 등 이슈 회피)
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
