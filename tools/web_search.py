"""
Backend-controlled web search tool.

The platform's default "Web Search" child agent uses this tool for
current/external information.

Implementation notes:
- No API key required.
- Uses DuckDuckGo HTML and Lite endpoints.
- Tries multiple page shapes because DuckDuckGo can return different
  markup depending on region/request path.
- Empty/unparseable search responses are treated as failures, never as
  a successful search with zero results.
"""

from __future__ import annotations

from typing import Any, Dict, List
from urllib.parse import parse_qs, unquote, urlparse

import requests
from bs4 import BeautifulSoup

from .base import BaseTool


DUCKDUCKGO_HTML_URL = "https://html.duckduckgo.com/html/"
DUCKDUCKGO_LITE_URL = "https://lite.duckduckgo.com/lite/"

DEFAULT_MAX_RESULTS = 5
REQUEST_TIMEOUT_SECONDS = 12

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}


class WebSearchError(Exception):
    """Raised when web search cannot return usable results."""


def _unwrap_duckduckgo_url(value: str) -> str:
    """Convert DuckDuckGo redirect links into real destination URLs."""
    url = str(value or "").strip()

    if not url:
        return ""

    if url.startswith("//"):
        url = "https:" + url

    try:
        parsed = urlparse(url)
        query = parse_qs(parsed.query)

        uddg = query.get("uddg")
        if uddg:
            return unquote(uddg[0])

    except Exception:
        pass

    return url


def _append_result(
    results: List[Dict[str, str]],
    *,
    title: str,
    url: str,
    snippet: str,
    max_results: int,
) -> None:
    title = str(title or "").strip()
    url = _unwrap_duckduckgo_url(url)
    snippet = " ".join(str(snippet or "").split())

    if not title or not url:
        return

    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        host = ""

    if "duckduckgo.com" in host:
        return

    if any(item["url"] == url for item in results):
        return

    results.append(
        {
            "title": title,
            "url": url,
            "snippet": snippet,
        }
    )

    if len(results) > max_results:
        del results[max_results:]


def _parse_html_results(
    html: str,
    max_results: int,
) -> List[Dict[str, str]]:
    """Parse DuckDuckGo's HTML endpoint."""
    soup = BeautifulSoup(html, "html.parser")
    results: List[Dict[str, str]] = []

    for result_div in soup.select("div.result, div.web-result"):
        if len(results) >= max_results:
            break

        title_tag = result_div.select_one("a.result__a, a.result-link")
        if title_tag is None:
            continue

        snippet_tag = result_div.select_one(
            ".result__snippet, .result-snippet"
        )

        _append_result(
            results,
            title=title_tag.get_text(" ", strip=True),
            url=title_tag.get("href", ""),
            snippet=(
                snippet_tag.get_text(" ", strip=True)
                if snippet_tag is not None
                else ""
            ),
            max_results=max_results,
        )

    if results:
        return results[:max_results]

    # Fallback when DuckDuckGo changes the wrapper element.
    for title_tag in soup.select("a.result__a, a.result-link"):
        if len(results) >= max_results:
            break

        parent = title_tag.find_parent(["div", "tr", "td"])
        snippet = ""

        if parent is not None:
            snippet_tag = parent.select_one(
                ".result__snippet, .result-snippet"
            )
            if snippet_tag is not None:
                snippet = snippet_tag.get_text(" ", strip=True)

        _append_result(
            results,
            title=title_tag.get_text(" ", strip=True),
            url=title_tag.get("href", ""),
            snippet=snippet,
            max_results=max_results,
        )

    return results[:max_results]


def _parse_lite_results(
    html: str,
    max_results: int,
) -> List[Dict[str, str]]:
    """Parse DuckDuckGo Lite's table-oriented result page."""
    soup = BeautifulSoup(html, "html.parser")
    results: List[Dict[str, str]] = []

    for title_tag in soup.select("a.result-link, a.result__a"):
        if len(results) >= max_results:
            break

        snippet = ""
        row = title_tag.find_parent("tr")

        if row is not None:
            next_row = row.find_next_sibling("tr")
            if next_row is not None:
                snippet_tag = next_row.select_one(
                    ".result-snippet, .result__snippet"
                )
                if snippet_tag is not None:
                    snippet = snippet_tag.get_text(" ", strip=True)

        _append_result(
            results,
            title=title_tag.get_text(" ", strip=True),
            url=title_tag.get("href", ""),
            snippet=snippet,
            max_results=max_results,
        )

    return results[:max_results]


def _looks_blocked(html: str) -> bool:
    lowered = str(html or "").lower()

    markers = (
        "anomaly-modal",
        "bots use duckduckgo",
        "please complete the following challenge",
        "unusual traffic",
        "captcha",
    )

    return any(marker in lowered for marker in markers)


def _request_search_page(
    session: requests.Session,
    *,
    url: str,
    query: str,
) -> str:
    response = session.get(
        url,
        params={"q": query},
        headers=_HEADERS,
        timeout=REQUEST_TIMEOUT_SECONDS,
        allow_redirects=True,
    )
    response.raise_for_status()
    return response.text


class WebSearchTool(BaseTool):
    """Searches the public web through DuckDuckGo."""

    name = "web_search"

    description = (
        "Searches the public web via DuckDuckGo and returns real "
        "search results with title, URL, and snippet. Use this for "
        "current news, releases, rankings, prices, events, or other "
        "information that can change over time."
    )

    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query.",
            },
            "max_results": {
                "type": "integer",
                "description": (
                    "Maximum number of results to return "
                    f"(default {DEFAULT_MAX_RESULTS})."
                ),
            },
        },
        "required": ["query"],
    }

    def execute(self, **kwargs) -> Any:
        query = str(kwargs.get("query") or "").strip()

        if not query:
            raise ValueError("'query' is required.")

        try:
            max_results = int(
                kwargs.get("max_results", DEFAULT_MAX_RESULTS)
            )
        except (TypeError, ValueError):
            max_results = DEFAULT_MAX_RESULTS

        max_results = max(1, min(max_results, 10))
        errors: List[str] = []

        with requests.Session() as session:
            # 1. Standard DuckDuckGo HTML endpoint.
            try:
                html = _request_search_page(
                    session,
                    url=DUCKDUCKGO_HTML_URL,
                    query=query,
                )

                if _looks_blocked(html):
                    errors.append(
                        "DuckDuckGo HTML returned a challenge page"
                    )
                else:
                    results = _parse_html_results(
                        html,
                        max_results=max_results,
                    )

                    if results:
                        return {
                            "query": query,
                            "results": results,
                            "search_backend": "duckduckgo_html",
                        }

            except requests.RequestException as exc:
                errors.append(
                    f"DuckDuckGo HTML request failed: {exc}"
                )

            # 2. Lite fallback.
            try:
                html = _request_search_page(
                    session,
                    url=DUCKDUCKGO_LITE_URL,
                    query=query,
                )

                if _looks_blocked(html):
                    errors.append(
                        "DuckDuckGo Lite returned a challenge page"
                    )
                else:
                    results = _parse_lite_results(
                        html,
                        max_results=max_results,
                    )

                    if results:
                        return {
                            "query": query,
                            "results": results,
                            "search_backend": "duckduckgo_lite",
                        }

            except requests.RequestException as exc:
                errors.append(
                    f"DuckDuckGo Lite request failed: {exc}"
                )

        reason = (
            "; ".join(errors)
            if errors
            else (
                "DuckDuckGo returned pages but no parseable "
                "search results"
            )
        )

        raise WebSearchError(
            f"Web search returned no usable results. {reason}"
        )
