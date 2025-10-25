from __future__ import annotations
import os, sys, json, time
import argparse
from runtime.logging_utils import log
from runtime.cache_utils import cached
from dotenv import load_dotenv
load_dotenv()


import app as core
import phase3_agents as agents


# Simple retry decorator


def retry(times=3, backoff=0.5):
    def deco(fn):
        def _wrap(*args, **kwargs):
            last = None
            for i in range(times):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    last = e
                    delay = backoff * (2 ** i)
                    log.warning("retry", attempt=i+1, delay=delay, fn=fn.__name__)
                    time.sleep(delay)
            raise last
        return _wrap
    return deco


@cached(lambda company, lang, pretty, local_fallback: f"report:{company}:{lang}:{local_fallback}")
@retry(times=3, backoff=0.6)
def build_report(company: str, lang: str, pretty: bool = False, local_fallback: bool = False):
    if local_fallback or not os.getenv("OPENAI_API_KEY"):
        rep, md = core.generate_report(company, lang)
        return {"json": rep.model_dump(), "markdown": md}
    else:
        return agents.run_crew(company, lang)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CrewFin-AI Phase 4 runner")
    p.add_argument("--company", required=True)
    p.add_argument("--lang", default="ko")
    p.add_argument("--pretty", action="store_true")
    p.add_argument("--local-fallback", action="store_true")
    args = p.parse_args()


    out = build_report(args.company, args.lang, args.pretty, args.local_fallback)


    print("\n=== JSON ===")
    if args.pretty:
        print(json.dumps(out["json"], ensure_ascii=False, indent=2))
    else:
        print(json.dumps(out["json"], ensure_ascii=False))
    print("\n=== Markdown ===")
    print(out["markdown"])