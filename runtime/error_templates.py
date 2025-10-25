from __future__ import annotations
from typing import Literal


Lang = Literal["ko", "en"]


TEMPLATES = {
"ticker_not_found": {
"ko": "티커를 찾지 못했습니다. 회사명을 더 구체적으로 입력해 주세요.",
"en": "Couldn't resolve a ticker. Please provide a more specific company name.",
},
"no_data": {
"ko": "최근 7일 내에 충분한 데이터가 없습니다. 다른 회사명이나 기간으로 시도해 주세요.",
"en": "Insufficient data in the last 7 days. Try another company or time range.",
},
"external_error": {
"ko": "외부 데이터 제공자 오류로 실데이터 수집에 실패했습니다. 잠시 후 다시 시도해 주세요.",
"en": "Failed to fetch live data due to external provider errors. Please try again later.",
},
}


def msg(key: str, lang: Lang = "ko") -> str:
    d = TEMPLATES.get(key) or {}
    return d.get(lang, d.get("en", key))