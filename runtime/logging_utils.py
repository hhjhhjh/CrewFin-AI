from __future__ import annotations
import os, sys, time
import logging
import structlog


LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


# structlog + stdlib bridge
logging.basicConfig(
level=LOG_LEVEL,
format="%(message)s",
stream=sys.stdout,
)


structlog.configure(
processors=[
structlog.processors.TimeStamper(fmt="iso", utc=False),
structlog.processors.add_log_level,
structlog.processors.StackInfoRenderer(),
structlog.processors.format_exc_info,
structlog.processors.JSONRenderer(),
],
wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(LOG_LEVEL)),
cache_logger_on_first_use=True,
)


log = structlog.get_logger("crew-senti-runtime")




def with_timing(fn):
    def _wrap(*args, **kwargs):
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            dur = (time.perf_counter() - start) * 1000
            log.info("timing", function=fn.__name__, ms=int(dur))
        return _wrap