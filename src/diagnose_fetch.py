"""
One-shot diagnostic: is FBref reachable from this machine, and are the
non-Cloudflare fallback sources reachable? Tells us which data route to take.

    python3 -m src.diagnose_fetch
"""
from __future__ import annotations

import time


def _try(label, fn):
    try:
        code, note = fn()
        flag = "OK " if code == 200 else "!! "
        print(f"  {flag}{label:38s} HTTP {code}  {note}")
        return code == 200
    except Exception as exc:
        print(f"  XX {label:38s} {type(exc).__name__}: {str(exc)[:80]}")
        return False


def main() -> None:
    from curl_cffi import requests as cffi

    print("Testing data-source reachability from this machine...\n")

    s = cffi.Session(impersonate="chrome")
    s.headers.update({"Accept-Language": "en-US,en;q=0.9"})

    def fbref_home():
        r = s.get("https://fbref.com/en/", timeout=30)
        return r.status_code, "(warmup)"

    def fbref_stats_cold():
        r = cffi.get("https://fbref.com/en/comps/9/2024-2025/stats/2024-2025-Premier-League-Stats",
                     impersonate="chrome", timeout=30)
        return r.status_code, "cold, no warmup"

    def fbref_stats_warm():
        r = s.get("https://fbref.com/en/comps/9/2024-2025/stats/2024-2025-Premier-League-Stats",
                  timeout=30)
        has = "stats_standard" in r.text
        return r.status_code, f"after warmup, table_present={has}"

    def understat():
        r = cffi.get("https://understat.com/league/EPL/2024", impersonate="chrome", timeout=30)
        has = "playersData" in r.text
        return r.status_code, f"players_json_present={has}"

    def fotmob():
        r = cffi.get("https://www.fotmob.com/api/leagues?id=47", impersonate="chrome", timeout=30)
        return r.status_code, f"bytes={len(r.content)}"

    fb_home = _try("FBref homepage (warmup)", fbref_home)
    time.sleep(3)
    _try("FBref stats page (cold)", fbref_stats_cold)
    time.sleep(3)
    fb_warm = _try("FBref stats page (warmed session)", fbref_stats_warm)
    time.sleep(2)
    us = _try("Understat (fallback, top-5 EU)", understat)
    time.sleep(2)
    fm = _try("FotMob API (fallback)", fotmob)

    print("\nVerdict:")
    if fb_warm:
        print("  FBref works with the warmed session -> use: python -m src.pull_data_http")
    elif fb_home and not fb_warm:
        print("  FBref homepage OK but stat pages blocked -> likely rate/bot flag on")
        print("  deep pages. Wait ~30 min and retry, or switch source.")
    else:
        print("  FBref is blocking this machine outright (probably a temporary IP")
        print("  flag from the earlier browser attempts).")
    if us:
        print("  Understat reachable -> viable fallback for the top-5 EU leagues.")
    if fm:
        print("  FotMob reachable -> viable fallback (broader league coverage).")


if __name__ == "__main__":
    main()
