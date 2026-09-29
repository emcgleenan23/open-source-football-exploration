"""
Isolate the browser problem from soccerdata.

soccerdata swallows the real Chrome-launch error and only fails later with a
confusing 'FBref object has no attribute _driver'. This script launches the
exact same undetected Chrome that soccerdata uses and surfaces the true error,
then tries to load FBref through Cloudflare.

    python -m src.check_browser
"""
from __future__ import annotations

import sys


def main() -> int:
    try:
        import seleniumbase
        import selenium
        print(f"seleniumbase {seleniumbase.__version__}, selenium {selenium.__version__}")
    except Exception as exc:
        print(f"Import failed: {exc}")
        return 1

    from seleniumbase import Driver

    print("Launching undetected Chrome (headful) ...")
    try:
        d = Driver(uc=True, headless=False)
    except Exception as exc:
        print("\nBROWSER FAILED TO LAUNCH — this is the real error soccerdata hides:")
        print(f"  {type(exc).__name__}: {exc}")
        print("\nMost likely a stale/mismatched chromedriver. Fix with:")
        _print_fix()
        return 2

    try:
        d.get("https://fbref.com/en/comps/")
        title = d.title
        print(f"Loaded FBref. Page title: {title!r}")
        ok = "Football" in title or "comps" in d.current_url
        print("PASS — soccerdata should now work." if ok
              else "Loaded, but Cloudflare may still be challenging. Re-run the pull.")
        return 0 if ok else 3
    finally:
        d.quit()


def _print_fix() -> None:
    print("""
    pip install -U seleniumbase undetected-chromedriver selenium
    # clear stale undetected drivers so a matching one is fetched:
    rm -rf ~/.uc_driver 2>/dev/null
    # confirm your Chrome major version, then fetch a matching uc_driver:
    #   Chrome menu -> About Google Chrome
    sbase get uc_driver
    # re-run this check:
    python -m src.check_browser
    """)


if __name__ == "__main__":
    sys.exit(main())