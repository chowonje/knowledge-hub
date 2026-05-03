from __future__ import annotations

BROWSER_LIKE_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)


def default_request_headers() -> dict[str, str]:
    return {"User-Agent": BROWSER_LIKE_USER_AGENT}
