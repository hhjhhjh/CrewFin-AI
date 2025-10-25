"""
crew_senti_report_backend_v3.py

PRD: 다국어 커뮤니티 기반 주식 감성 분석 시스템 v3.0 — Agent 백엔드 (Telegram 연동 제외)
작성: 2025-01-20 (v3.0)

이 파일 하나로 MVP 파이프라인(티커 변환 ➜ 데이터 수집 ➜ 감성 분석 ➜ 점수/온도 산출 ➜ 리포트 JSON/Markdown 생성 ➜ 콘솔 출력)까지 동작하도록 구성했습니다.

필요 환경 변수(.env 권장):
  OPENAI_API_KEY=...
  TAVILY_API_KEY=...          # 선택(있으면 Tavily MCP 사용)
  SERPER_API_KEY=...          # 선택(Fallback)
  NAVER_CLIENT_ID=...         # 선택(Naver Search MCP)
  NAVER_CLIENT_SECRET=...     # 선택(Naver Search MCP)

필요 패키지(권장):
  pip install crewai crewai-tools openai python-dotenv cachetools pydantic python-dateutil rapidfuzz
  # MCP 원격 연결은 별도 설정 필요(실 서비스 시 반영). 이 MVP는 안전한 Fallback(더미 수집기) 포함.

실행 예시:
  python crew_senti_report_backend_v3.py --company "엔비디아" --lang ko
  python crew_senti_report_backend_v3.py --company "Samsung Electronics" --lang en

출력:
  - 표준 출력에 JSON 요약 + Markdown 리포트
  - 필요 시 함수 generate_report(company_input, language)로 모듈 임포트 사용 가능

주의:
  - 본 파일은 Telegram/Webhook 없이 Agent 백엔드만 포함합니다(요청하신 범위).
  - Tavily MCP / Naver MCP는 실제 키·연결 존재 시에만 호출하고, 없으면 Serper/더미로 폴백.
  - Reddit API는 미포함. 필요 시 TODO 주석 참고하여 추가 연결.
"""
from __future__ import annotations

import os
import json
import time
import argparse
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from datetime import datetime, timezone, timedelta
from dateutil import tz

from dotenv import load_dotenv
from cachetools import TTLCache
from pydantic import BaseModel, Field
from rapidfuzz import fuzz

# Added for Phase 2 real-data connectors
import requests
from urllib.parse import quote_plus
import random

# ============= 로깅 =============
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("crew-senti-backend")

# ============= 환경 로딩 =============
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "").strip()
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID", "").strip()
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET", "").strip()

# ============= OpenAI SDK =============
try:
    from openai import OpenAI
    _openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception as e:
    _openai_client = None
    log.warning("OpenAI SDK 불러오기 실패(실행은 가능하나 LLM 기능 제한): %s", e)

MODEL_TICKER = "gpt-4o-mini"
MODEL_NLP    = "gpt-4o-mini"

# HTTP defaults
HTTP_TIMEOUT = 15
UA = "CrewFin-AI/1.0 (+https://example.local)"

# ============= CrewAI (에이전트/태스크) =============
try:
    from crewai import Agent, Task, Crew, Process
except Exception as e:
    Agent = Task = Crew = Process = None
    log.warning("CrewAI 불러오기 실패. fallback 파이프라인으로 동작합니다: %s", e)

# ============= 간단 캐시 =============
# 회사명 ➜ 티커 변환 1시간 TTL, 수집 결과 10분 TTL
cache_ticker = TTLCache(maxsize=512, ttl=3600)
cache_collect = TTLCache(maxsize=512, ttl=600)

# ============= 유틸 =============
KST = tz.gettz("Asia/Seoul")

def now_kst_iso() -> str:
    return datetime.now(KST).isoformat()

# ---------------- 시장 온도 스케일 ----------------
TEMP_BUCKETS: List[Tuple[str, Tuple[int,int]]] = [
    ("Ice Cold", (0, 20)),
    ("Cool",     (21, 40)),
    ("Warm",     (41, 60)),
    ("Hot",      (61, 80)),
    ("Red Hot",  (81, 100)),
]

KO_TEMP_LABELS = {
    "Ice Cold": "매우 차가움",
    "Cool": "차가움",
    "Warm": "따뜻함",
    "Hot": "뜨거움",
    "Red Hot": "과열",
}

def score_to_temp_label(score: int) -> str:
    for label, (lo, hi) in TEMP_BUCKETS:
        if lo <= score <= hi:
            return label
    return "Warm"


# ---------------- 언어 감지(간단) ----------------
# 실제 서비스에선 fastText/langdetect/LLM 등으로 대체 가능
def simple_lang_detect(text: str) -> str:
    # 한글 존재 비율 기반
    korean_ratio = sum(0xAC00 <= ord(ch) <= 0xD7A3 for ch in text) / max(len(text),1)
    return "ko" if korean_ratio > 0.1 else "en"

# ============= 티커 변환 =============
class TickerResult(BaseModel):
    success: bool = True
    primary_ticker: Optional[str] = None
    market: Optional[str] = None  # "US" | "KR"
    company_name: Optional[str] = None
    alternatives: List[Dict[str, Any]] = Field(default_factory=list)
    error_message: Optional[str] = None

SYSTEM_TICKER_PROMPT = (
    "당신은 주식 티커 변환 전문가입니다. 주어진 회사명을 정확한 주식 티커로 변환하고 시장을 식별해야 합니다.\n"
    "규칙:\n"
    "1) 미국 시장: 티커만 반환(예: NVDA, AAPL)\n"
    "2) 한국 시장: .KS 또는 .KQ 접미사(예: 005930.KS, 035420.KQ)\n"
    "3) 불확실 시 상위 3개 후보와 confidence 포함\n"
    "4) 한국어 입력 시 한국 시장 우선, 영어 입력 시 미국 시장 우선\n"
    "5) 반드시 JSON으로만 응답"
)

def convert_company_name_to_ticker(company_name: str, language: str) -> TickerResult:
    key = f"{language}:{company_name.strip().lower()}"
    if key in cache_ticker:
        return cache_ticker[key]

    # LLM 사용 가능 시 프롬프트 호출
    if _openai_client:
        user_prompt = json.dumps({
            "company_name": company_name,
            "language": language,
            "priority": "KR>US" if language == "ko" else "US>KR"
        }, ensure_ascii=False)
        try:
            resp = _openai_client.chat.completions.create(
                model=MODEL_TICKER,
                messages=[
                    {"role": "system", "content": SYSTEM_TICKER_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )
            content = resp.choices[0].message.content.strip()
            data = json.loads(content)
            # 표준화
            out = TickerResult(
                success=bool(data.get("success", True)),
                primary_ticker=data.get("primary_ticker"),
                market=data.get("market"),
                company_name=data.get("company_name"),
                alternatives=data.get("alternatives", []),
                error_message=data.get("error_message"),
            )
            cache_ticker[key] = out
            return out
        except Exception as e:
            log.warning("LLM 티커 변환 실패, 휴리스틱 폴백: %s", e)

    # 폴백(간단 휴리스틱): 대표 케이스 매핑
    canonical = {
        "엔비디아": ("NVDA", "US", "NVIDIA Corporation"),
        "nvidia": ("NVDA", "US", "NVIDIA Corporation"),
        "삼성전자": ("005930.KS", "KR", "Samsung Electronics Co., Ltd."),
        "samsung electronics": ("005930.KS", "KR", "Samsung Electronics Co., Ltd."),
        "apple": ("AAPL", "US", "Apple Inc."),
        "애플": ("AAPL", "US", "Apple Inc."),
    }
    key2 = company_name.strip().lower()
    if key2 in canonical:
        t, m, n = canonical[key2]
        out = TickerResult(success=True, primary_ticker=t, market=m, company_name=n,
                           alternatives=[{"ticker": t, "market": m, "confidence": 0.8}])
    else:
        # fuzzy로 삼성전자/엔비디아/애플 근접 처리
        best = max(canonical.keys(), key=lambda k: fuzz.token_set_ratio(k, key2))
        score = fuzz.token_set_ratio(best, key2)
        if score >= 70:
            t, m, n = canonical[best]
            out = TickerResult(success=True, primary_ticker=t, market=m, company_name=n,
                               alternatives=[{"ticker": t, "market": m, "confidence": score/100}])
        else:
            out = TickerResult(success=False, error_message="회사명을 찾을 수 없습니다. 더 구체적으로 입력해주세요.")
    cache_ticker[key] = out
    return out

# ============= 데이터 수집기(뉴스/커뮤니티) =============
class CollectedItem(BaseModel):
    source: str                 # e.g., "news", "reddit", "naver_cafe"
    title: str
    url: Optional[str] = None
    published_at: Optional[str] = None  # ISO8601
    content: Optional[str] = None       # 기사 본문 또는 요약

class CollectorOutput(BaseModel):
    news: List[CollectedItem] = Field(default_factory=list)
    community: List[CollectedItem] = Field(default_factory=list)


def _dummy_news(company: str) -> List[CollectedItem]:
    now = now_kst_iso()
    return [
        CollectedItem(source="news", title=f"{company} 신제품 출시 기대", url=None, published_at=now,
                      content=f"{company} 관련 신제품 루머와 성능 향상 기대감이 커뮤니티와 뉴스에서 확산."),
        CollectedItem(source="news", title=f"{company} 경쟁 심화", url=None, published_at=now,
                      content="경쟁사 가격 인하 및 시장 진입으로 점유율 경쟁이 심화."),
    ]


def _dummy_community(company: str) -> List[CollectedItem]:
    now = now_kst_iso()
    return [
        CollectedItem(source="reddit", title=f"Long {company}?", published_at=now,
                      content="신제품 성능이 좋다는 루머, 단기 모멘텀 기대."),
        CollectedItem(source="naver_cafe", title=f"{company} 규제 우려", published_at=now,
                      content="반독점 이슈 가능성 언급, 변동성 주의 의견."),
    ]


def _serper_search_news(company_name: str, days: int, max_results: int) -> List[CollectedItem]:
    items: List[CollectedItem] = []
    try:
        from crewai_tools import SerperDevTool
        tool = SerperDevTool()  # SERPER_API_KEY는 env에서 자동 인식
        q = f"{company_name} stock news last {days} days"
        res = tool.run(q)
        if isinstance(res, dict):
            for i in res.get("news", [])[:max_results]:
                items.append(CollectedItem(
                    source="news",
                    title=i.get("title", ""),
                    url=i.get("link"),
                    published_at=i.get("date"),
                    content=i.get("snippet")
                ))
    except Exception as e:
        log.warning("Serper 검색 실패: %s", e)
    return items


def _tavily_search_news(company_name: str, days: int, max_results: int) -> List[CollectedItem]:
    """Tavily REST (Bearer auth). Docs:
    https://docs.tavily.com/documentation/api-reference/endpoint/search
    """
    items: List[CollectedItem] = []
    api_key = TAVILY_API_KEY
    if not api_key:
        return items

    url = "https://api.tavily.com/search"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",   # ✅ 핵심: Bearer 헤더
        "User-Agent": UA,
    }
    payload = {
        "query": f"{company_name} stock news",
        "topic": "finance",
        "days": days,
        "max_results": max_results,
        "include_raw_content": True,
        "search_depth": "basic",
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=HTTP_TIMEOUT)
        if r.status_code != 200:
            log.warning("Tavily HTTP %s: %s", r.status_code, r.text[:200])
            return items
        data = r.json()
        results = data.get("results", [])
        for res in results[:max_results]:
            items.append(CollectedItem(
                source="news",
                title=res.get("title") or res.get("name") or "",
                url=res.get("url"),
                published_at=res.get("published_date") or res.get("date"),
                content=(res.get("raw_content") or res.get("content") or res.get("snippet"))
            ))
    except Exception as e:
        log.warning("Tavily 예외: %s", e)
    return items



def _naver_search(endpoint: str, query: str, display: int = 7, sort: str = "date") -> List[Dict[str, Any]]:
    url = f"https://openapi.naver.com/v1/search/{endpoint}.json?query={quote_plus(query)}&display={display}&sort={sort}"
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
        "User-Agent": UA,
    }
    try:
        r = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT)
        if r.status_code == 200:
            return r.json().get("items", [])
        log.warning("Naver %s 실패(%s): %s", endpoint, r.status_code, r.text[:200])
    except Exception as e:
        log.warning("Naver %s 예외: %s", endpoint, e)
    return []


def _naver_collect(company_name: str, max_news: int = 7, cafe_n: int = 10, blog_n: int = 5) -> Tuple[List[CollectedItem], List[CollectedItem]]:
    news_items: List[CollectedItem] = []
    comm_items: List[CollectedItem] = []
    for it in _naver_search("news", company_name, display=max_news, sort="date")[:max_news]:
        news_items.append(CollectedItem(
            source="news",
            title=it.get("title",""),
            url=it.get("link"),
            published_at=None,
            content=it.get("description")
        ))
    for it in _naver_search("cafearticle", company_name, display=cafe_n, sort="date")[:cafe_n]:
        comm_items.append(CollectedItem(
            source="naver_cafe",
            title=it.get("title",""),
            url=it.get("link"),
            published_at=None,
            content=it.get("description")
        ))
    for it in _naver_search("blog", f"{company_name} 주식", display=blog_n, sort="date")[:blog_n]:
        comm_items.append(CollectedItem(
            source="naver_blog",
            title=it.get("title",""),
            url=it.get("link"),
            published_at=None,
            content=it.get("description")
        ))
    return news_items, comm_items


def _reddit_recent_posts(sub: str, query: str, limit: int = 5) -> List[CollectedItem]:
    items: List[CollectedItem] = []
    url = f"https://www.reddit.com/r/{sub}/search.json?q={quote_plus(query)}&restrict_sr=1&sort=new&t=week&limit={limit}"
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=HTTP_TIMEOUT)
        if r.status_code == 200:
            data = r.json().get("data", {}).get("children", [])
            for ch in data[:limit]:
                d = ch.get("data", {})
                items.append(CollectedItem(
                    source="reddit",
                    title=d.get("title",""),
                    url=f"https://www.reddit.com{d.get('permalink','')}",
                    published_at=datetime.fromtimestamp(d.get("created_utc", 0), tz=KST).isoformat() if d.get("created_utc") else None,
                    content=d.get("selftext") or d.get("title")
                ))
        else:
            log.warning("Reddit %s 실패(%s)", sub, r.status_code)
    except Exception as e:
        log.warning("Reddit 예외(%s): %s", sub, e)
    return items


def collect_data(company_name: str, market: str, days: int = 7, max_results_news: int = 7) -> CollectorOutput:
    key = f"collect:{company_name}:{market}:{days}:{max_results_news}"
    if key in cache_collect:
        return cache_collect[key]

    items_news: List[CollectedItem] = []
    items_comm: List[CollectedItem] = []
    used_any = False

    # 1) Tavily 우선
    if TAVILY_API_KEY:
        tv_news = _tavily_search_news(company_name, days=days, max_results=max_results_news)
        if tv_news:
            items_news.extend(tv_news)
            used_any = True

    # 2) Serper Fallback
    if not items_news and SERPER_API_KEY:
        sp_news = _serper_search_news(company_name, days=days, max_results=max_results_news)
        if sp_news:
            items_news.extend(sp_news)
            used_any = True

    # 3) Naver (KR 뉴스/카페/블로그)
    if NAVER_CLIENT_ID and NAVER_CLIENT_SECRET:
        nv_news, nv_comm = _naver_collect(company_name, max_news=max_results_news, cafe_n=10, blog_n=5)
        if nv_news:
            # 간단 중복 제거
            seen = set()
            for it in nv_news:
                k = (it.title, it.url)
                if k not in seen:
                    items_news.append(it)
                    seen.add(k)
            used_any = True or used_any
        if nv_comm:
            items_comm.extend(nv_comm)
            used_any = True or used_any

    # 4) Reddit (미국 커뮤니티)
    for s in ["wallstreetbets", "stocks", "investing"]:
        items_comm.extend(_reddit_recent_posts(s, company_name, limit=5))
    if items_comm:
        used_any = True

    # 5) 더미 폴백
    if not used_any:
        items_news = _dummy_news(company_name)
        items_comm = _dummy_community(company_name)

    out = CollectorOutput(news=items_news[:max_results_news], community=items_comm)
    cache_collect[key] = out
    return out

# ============= 감성 분석 =============
class SentimentResult(BaseModel):
    label: str   # "positive"|"negative"|"neutral"
    confidence: float


SENTIMENT_SYSTEM = (
    "당신은 금융 뉴스/커뮤니티 포스트의 감성을 분류하는 분석가입니다. 결과는 JSON만 출력합니다.\n"
    "라벨은 positive/negative/neutral 중 하나. 확률(confidence)은 0~1.")


def analyze_sentiment_batch(texts: List[str]) -> List[SentimentResult]:
    results: List[SentimentResult] = []
    if _openai_client:
        try:
            # 간단한 batched 호출: 프롬프트 내 리스트 전달
            user_payload = {"items": texts}
            resp = _openai_client.chat.completions.create(
                model=MODEL_NLP,
                messages=[
                    {"role": "system", "content": SENTIMENT_SYSTEM},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                ],
                temperature=0.0,
            )
            content = resp.choices[0].message.content.strip()
            data = json.loads(content)
            # 기대 형식: [{"label": "positive", "confidence": 0.92}, ...]
            for d in data:
                results.append(SentimentResult(label=d.get("label","neutral"), confidence=float(d.get("confidence",0.5))))
            return results
        except Exception as e:
            log.warning("LLM 감성분석 실패, 규칙 기반 폴백: %s", e)

    # 폴백: 극단적 키워드 규칙(아주 단순)
    pos_kw = ["great","good","positive","상승","호재","기대","outperform","beat"]
    neg_kw = ["bad","negative","하락","악재","우려","소송","antitrust","ban"]
    for t in texts:
        t_low = (t or "").lower()
        pos = any(k in t_low for k in pos_kw)
        neg = any(k in t_low for k in neg_kw)
        if pos and not neg:
            results.append(SentimentResult(label="positive", confidence=0.7))
        elif neg and not pos:
            results.append(SentimentResult(label="negative", confidence=0.7))
        elif pos and neg:
            results.append(SentimentResult(label="neutral", confidence=0.55))
        else:
            results.append(SentimentResult(label="neutral", confidence=0.5))
    return results

# ============= 점수/온도 & 키 테마 =============
class MarketTemperature(BaseModel):
    label: str
    score: int
    scale: str = "0(Ice Cold) ~ 100(Red Hot)"

class SentimentSummary(BaseModel):
    positive: int
    negative: int
    neutral: int

class CommunitySentimentAnalysis(BaseModel):
    market_temperature: MarketTemperature
    sentiment_summary: SentimentSummary
    data_sources: Dict[str, int]
    key_themes: List[str]


def aggregate_sentiment(news: List[CollectedItem], comm: List[CollectedItem]) -> CommunitySentimentAnalysis:
    texts = [(n.content or n.title or "") for n in news] + [(c.content or c.title or "") for c in comm]
    sents = analyze_sentiment_batch(texts)

    pos = sum(1 for s in sents if s.label == "positive")
    neg = sum(1 for s in sents if s.label == "negative")
    neu = sum(1 for s in sents if s.label == "neutral")

    # 점수: (pos - neg) / total ➜ [0..100]로 매핑 (중립 50)
    total = max(pos + neg + neu, 1)
    raw = (pos - neg) / total  # -1..+1
    score = int(round(50 + raw * 50))
    score = max(0, min(100, score))
    label = score_to_temp_label(score)

    # 키 테마 추출(아주 단순 요약)
    key_themes: List[str] = []
    if pos:
        key_themes.append("신제품/기술 모멘텀에 대한 기대감 (긍정)")
    if neg:
        key_themes.append("경쟁 심화/규제 이슈에 대한 우려 (부정)")
    if neu:
        key_themes.append("중립적 관망세 및 실적 확인 대기 (중립)")
    if not key_themes:
        key_themes = ["데이터 제한으로 유의미한 테마를 도출하기 어려움 (중립)"]

    label_en = label
    label_ko = KO_TEMP_LABELS.get(label_en, label_en)
    return CommunitySentimentAnalysis(
        market_temperature=MarketTemperature(label=f"{label_en} / {label_ko}", score=score),
        sentiment_summary=SentimentSummary(positive=pos, negative=neg, neutral=neu),
        data_sources={"news_count": len(news), "community_posts_count": len(comm)},
        key_themes=key_themes[:3],
    )

# ============= 리포트 스키마 & 포맷터 =============
class AnalystCommentary(BaseModel):
    market_reaction_interpretation: str
    investment_perspective: str
    risk_factors: List[str]
    opportunity_factors: List[str]

class ReportModel(BaseModel):
    report_generated_at: str
    language: str
    company_info: Dict[str, Any]
    community_sentiment_analysis: CommunitySentimentAnalysis
    analyst_commentary: AnalystCommentary
    overall_summary: str
    disclaimer: str


def draft_analyst_commentary(lang: str, csa: CommunitySentimentAnalysis, company_name: str) -> AnalystCommentary:
    # LLM 가능 시 더 고급 서술. 여기선 규칙 기반 템플릿.
    warmline = {
        "ko": f"커뮤니티와 뉴스에서 {company_name}에 대한 기대감과 우려가 혼재합니다.",
        "en": f"Across news and communities, sentiment on {company_name} mixes optimism with caution.",
    }
    investline = {
        "ko": "단기 모멘텀은 긍정적일 수 있으나, 중장기 리스크 모니터링이 필요합니다.",
        "en": "Short‑term momentum may be constructive, while medium‑term risks warrant monitoring.",
    }
    risks = ["경쟁 심화", "규제/반독점 이슈", "매크로 변동성 확대"]
    opps  = ["신제품/기술 주도 모멘텀", "AI 수요 확장", "생태계 강화"]
    return AnalystCommentary(
        market_reaction_interpretation=warmline.get(lang, warmline["en"]),
        investment_perspective=investline.get(lang, investline["en"]),
        risk_factors=risks,
        opportunity_factors=opps,
    )


def report_to_markdown(r: ReportModel) -> str:
    lang = r.language
    ci = r.company_info
    csa = r.community_sentiment_analysis
    ac = r.analyst_commentary

    if lang == "ko":
        md = []
        md.append(f"### 📊 커뮤니티 감성 종합 리포트 — {ci.get('name')} ({ci.get('ticker')})")
        md.append(f"- 생성시각(KST): {r.report_generated_at}")
        md.append("")
        md.append("**시장 온도(Temperature)**")
        md.append(f"- 레이블: {csa.market_temperature.label}")
        md.append(f"- 점수: {csa.market_temperature.score} / 100")
        md.append("")
        md.append("**감성 요약**")
        ss = csa.sentiment_summary
        md.append(f"- 긍정 {ss.positive} / 부정 {ss.negative} / 중립 {ss.neutral}")
        md.append(f"- 데이터 출처: 뉴스 {csa.data_sources['news_count']}개, 커뮤니티 {csa.data_sources['community_posts_count']}개")
        md.append("")
        md.append("**핵심 테마(Top 2-3)**")
        for t in csa.key_themes:
            md.append(f"- {t}")
        md.append("")
        md.append("**애널리스트 코멘트**")
        md.append(f"- 시장 반응 해석: {ac.market_reaction_interpretation}")
        md.append(f"- 투자 관점: {ac.investment_perspective}")
        md.append("- 리스크 요인:")
        for x in ac.risk_factors:
            md.append(f"  - {x}")
        md.append("- 기회 요인:")
        for x in ac.opportunity_factors:
            md.append(f"  - {x}")
        md.append("")
        md.append(f"**한줄 요약**  ")
        md.append(f"{r.overall_summary}")
        md.append("")
        md.append(
            "> 본 리포트는 AI가 생성한 정보로서 일반적 참고용이며, 투자 조언이 아닙니다. 투자 결정은 사용자 책임입니다."
        )
        return "\n".join(md)
    else:
        md = []
        md.append(f"### 📊 Community Sentiment Report — {ci.get('name')} ({ci.get('ticker')})")
        md.append(f"- Generated (KST): {r.report_generated_at}")
        md.append("")
        md.append("**Market Temperature**")
        md.append(f"- Label: {csa.market_temperature.label}")
        md.append(f"- Score: {csa.market_temperature.score} / 100")
        md.append("")
        md.append("**Sentiment Summary**")
        ss = csa.sentiment_summary
        md.append(f"- Positive {ss.positive} / Negative {ss.negative} / Neutral {ss.neutral}")
        md.append(f"- Data sources: News {csa.data_sources['news_count']}, Community {csa.data_sources['community_posts_count']}")
        md.append("")
        md.append("**Key Themes (Top 2-3)**")
        for t in csa.key_themes:
            md.append(f"- {t}")
        md.append("")
        md.append("**Analyst Commentary**")
        md.append(f"- Market Interpretation: {ac.market_reaction_interpretation}")
        md.append(f"- Investment Perspective: {ac.investment_perspective}")
        md.append("- Risk Factors:")
        for x in ac.risk_factors:
            md.append(f"  - {x}")
        md.append("- Opportunity Factors:")
        for x in ac.opportunity_factors:
            md.append(f"  - {x}")
        md.append("")
        md.append("**TL;DR**  ")
        md.append(r.overall_summary)
        md.append("")
        md.append(
            "> This report is AI‑generated and for informational purposes only. Not investment advice."
        )
        return "\n".join(md)

# ============= 메인 오케스트레이션 =============

def generate_report(company_input: str, language: Optional[str] = None) -> Tuple[ReportModel, str]:
    lang = language or simple_lang_detect(company_input)

    # 1) 티커 변환
    ticker_info = convert_company_name_to_ticker(company_input, lang)
    if not ticker_info.success or not ticker_info.primary_ticker:
        # 실패 리포트
        ci = {
            "name": company_input,
            "ticker": None,
            "market": None,
            "summary": None,
        }
        csa = CommunitySentimentAnalysis(
            market_temperature=MarketTemperature(label="Warm / 따뜻함", score=50),
            sentiment_summary=SentimentSummary(positive=0, negative=0, neutral=0),
            data_sources={"news_count": 0, "community_posts_count": 0},
            key_themes=["티커 식별 실패로 데이터 수집 불가"],
        )
        ac = draft_analyst_commentary(lang, csa, company_input)
        report = ReportModel(
            report_generated_at=now_kst_iso(),
            language=lang,
            company_info=ci,
            community_sentiment_analysis=csa,
            analyst_commentary=ac,
            overall_summary=("입력하신 회사명을 식별하지 못했습니다. 보다 구체적으로 입력해주세요." if lang=="ko"
                             else "Could not resolve the company. Please provide a more specific name."),
            disclaimer=("본 리포트는 AI에 의해 생성된 정보로서 일반적 참고용이며, 투자 조언이 아닙니다." if lang=="ko" else
                        "This report is AI‑generated and for informational purposes only. Not investment advice."),
        )
        md = report_to_markdown(report)
        return report, md

    # 2) 데이터 수집
    coll = collect_data(ticker_info.company_name or company_input, ticker_info.market or "US")

    # 3) 감성 집계/온도
    csa = aggregate_sentiment(coll.news, coll.community)

    # 4) 애널리스트 코멘트/요약
    ac = draft_analyst_commentary(lang, csa, ticker_info.company_name or company_input)

    overall = (
        f"신제품 기대와 업황 확장에 대한 긍정과, 경쟁/규제 우려가 균형을 이루는 {csa.market_temperature.label} 국면입니다."
        if lang=="ko" else
        f"A {csa.market_temperature.label} backdrop where product momentum and industry expansion are tempered by competition/regulatory risks."
    )

    report = ReportModel(
        report_generated_at=now_kst_iso(),
        language=lang,
        company_info={
            "name": ticker_info.company_name or company_input,
            "ticker": ticker_info.primary_ticker,
            "market": ticker_info.market,
            "summary": None,
        },
        community_sentiment_analysis=csa,
        analyst_commentary=ac,
        overall_summary=overall,
        disclaimer=("본 리포트는 AI에 의해 생성된 정보로서 일반적 참고용이며, 투자 조언이 아닙니다. 투자 결정은 사용자 책임입니다." if lang=="ko" else
                    "This report is AI‑generated for general information only and is not investment advice."),
    )
    md = report_to_markdown(report)
    return report, md

# ============= CLI 진입점 =============
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crew Sentiment Report Backend (Agent‑only)")
    parser.add_argument("--company", required=True, help="회사명 또는 종목명 (예: 엔비디아, 삼성전자, Apple)")
    parser.add_argument("--lang", default=None, help="응답 언어(ko|en). 미지정 시 자동 감지")
    parser.add_argument("--pretty", action="store_true", help="Pretty JSON output")
    args = parser.parse_args()

    rep, md = generate_report(args.company, args.lang)

    print("=== JSON ===")
    if args.pretty:
        print(json.dumps(rep.model_dump(), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(rep.model_dump(), ensure_ascii=False))

    print("=== Markdown ===")
    print(md)
