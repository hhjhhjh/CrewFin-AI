from __future__ import annotations
import os, time, threading
from typing import Any, Callable, Tuple
from functools import wraps


DEFAULT_TTL = int(os.getenv("CACHE_TTL_SECONDS", "600")) # 10 min


class TTLCache:
    def __init__(self, ttl: int = DEFAULT_TTL):
        self.ttl = ttl
        self._store: dict[str, Tuple[float, Any]] = {}
        self._lock = threading.RLock()


    def get(self, key: str):
        with self._lock:
            v = self._store.get(key)
            if not v:
                return None
            exp, data = v
            if exp < time.time():
                del self._store[key]
                return None
            return data


    def set(self, key: str, value: Any, ttl: int | None = None):
        with self._lock:
            t = ttl if ttl is not None else self.ttl
            self._store[key] = (time.time() + t, value)


GLOBAL_CACHE = TTLCache()




def cached(key_func: Callable[..., str] | None = None, ttl: int | None = None):
    def deco(fn: Callable):
        @wraps(fn)
        def _wrap(*args, **kwargs):
            key = key_func(*args, **kwargs) if key_func else f"{fn.__name__}:{args}:{tuple(sorted(kwargs.items()))}"
            got = GLOBAL_CACHE.get(key)
            if got is not None:
                return got
            val = fn(*args, **kwargs)
            GLOBAL_CACHE.set(key, val, ttl)
            return val
        return _wrap
    return deco