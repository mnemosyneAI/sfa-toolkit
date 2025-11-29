#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "fastmcp",
#     "httpx",
#     "beautifulsoup4",
#     "html2text",
#     "fastembed",
#     "numpy",
#     "pyyaml",
#     "youtube-transcript-api>=0.6.0",
#     "websocket-client>=1.6.0",
# ]
# ///

import argparse
import json
import re
import sys
import os
import urllib.parse
import hashlib
import datetime
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional

# Fix Windows console encoding for Unicode
if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

try:
    from fastmcp import FastMCP

    mcp = FastMCP("sfa-web")
except ImportError:
    mcp = None

import httpx
import yaml
import numpy as np
from bs4 import BeautifulSoup
import html2text

# YouTube transcript dependencies
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import (
        NoTranscriptFound,
        TranscriptsDisabled,
        VideoUnavailable,
    )

    YOUTUBE_API_AVAILABLE = True
except ImportError:
    YOUTUBE_API_AVAILABLE = False

# --- Helpers ---

# === YOUTUBE HELPERS ===


def _extract_video_id(url_or_id: str) -> str:
    """Extract video ID from various YouTube URL formats."""
    url_or_id = url_or_id.strip()
    if re.match(r"^[a-zA-Z0-9_-]{11}$", url_or_id):
        return url_or_id

    try:
        parsed = urllib.parse.urlparse(url_or_id)
        if parsed.hostname in ("youtu.be", "www.youtu.be"):
            return parsed.path.lstrip("/")
        if parsed.hostname in ("youtube.com", "www.youtube.com", "m.youtube.com"):
            if parsed.path == "/watch":
                v_param = urllib.parse.parse_qs(parsed.query).get("v")
                if v_param:
                    return v_param[0]
            elif parsed.path.startswith(("/v/", "/shorts/", "/embed/")):
                return parsed.path.split("/")[2]
    except Exception:
        pass
    raise ValueError(f"Could not extract video ID from: {url_or_id}")


def _strip_vtt_formatting(vtt_content: str) -> str:
    """Strip VTT formatting to extract clean text."""
    lines = vtt_content.splitlines()
    text_lines = []
    seen_lines = set()

    # Regex for VTT timestamps: 00:00:00.000 --> 00:00:05.000
    timestamp_pattern = re.compile(
        r"\d{2}:\d{2}:\d{2}\.\d{3}\s-->\s\d{2}:\d{2}:\d{2}\.\d{3}"
    )

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line == "WEBVTT" or line.startswith("Kind:") or line.startswith("Language:"):
            continue
        if timestamp_pattern.match(line):
            continue

        # Remove tags like <c>...</c> or <00:00:00.000>
        clean_line = re.sub(r"<[^>]+>", "", line)
        clean_line = clean_line.strip()

        if clean_line and clean_line not in seen_lines:
            text_lines.append(clean_line)
            seen_lines.add(clean_line)

    return " ".join(text_lines)


def _get_transcript_via_api(
    video_id: str,
    language: str = "en",
    with_timestamps: bool = False,
    format: str = "lines",
    http_proxy: Optional[str] = None,
    https_proxy: Optional[str] = None,
    cursor: Optional[str] = None,
    response_limit: int = 50000,
) -> Dict[str, Any]:
    """Try fetching transcript via youtube_transcript_api (>= 0.6.0).

    Args:
        video_id: YouTube video ID
        language: Language code (default: "en")
        with_timestamps: Include timestamps in output
        format: Output format - "lines" (one per subtitle), "continuous" (space-joined), "paragraphs" (grouped by gaps)
        http_proxy: HTTP proxy URL
        https_proxy: HTTPS proxy URL
        cursor: Pagination cursor (start index)
        response_limit: Max chars per response (for pagination)
    """
    if not YOUTUBE_API_AVAILABLE:
        return {"success": False, "error": "youtube_transcript_api not installed"}

    try:
        # Set up proxies if provided
        proxies = {}
        if http_proxy:
            proxies["http"] = http_proxy
        if https_proxy:
            proxies["https"] = https_proxy

        # New API: Create instance and fetch
        api = YouTubeTranscriptApi()
        if proxies:
            # Note: youtube-transcript-api uses requests internally, proxies work via env vars
            # We'll set them temporarily if provided
            old_http = os.environ.get("HTTP_PROXY")
            old_https = os.environ.get("HTTPS_PROXY")
            if http_proxy:
                os.environ["HTTP_PROXY"] = http_proxy
            if https_proxy:
                os.environ["HTTPS_PROXY"] = https_proxy

        try:
            result = api.fetch(video_id, languages=(language, "en"))
        finally:
            # Restore original proxy settings
            if proxies:
                if old_http is not None:
                    os.environ["HTTP_PROXY"] = old_http
                elif "HTTP_PROXY" in os.environ:
                    del os.environ["HTTP_PROXY"]
                if old_https is not None:
                    os.environ["HTTPS_PROXY"] = old_https
                elif "HTTPS_PROXY" in os.environ:
                    del os.environ["HTTPS_PROXY"]

        # Format output
        lines = []
        formatted_entries = []
        prev_end = 0

        # Parse cursor (start index)
        start_idx = int(cursor) if cursor else 0

        for i, snippet in enumerate(result.snippets):
            if i < start_idx:
                continue

            text = snippet.text.replace("\n", " ")
            start = snippet.start
            duration = snippet.duration

            if with_timestamps:
                minutes = int(start // 60)
                seconds = int(start % 60)
                timestamp = f"[{minutes:02d}:{seconds:02d}]"
                line = f"{timestamp} {text}"
            else:
                line = text

            # Track gap for paragraph detection (3+ second gap = new paragraph)
            gap = start - prev_end
            if format == "paragraphs" and gap > 3.0 and lines:
                lines.append("")  # Empty line for paragraph break

            lines.append(line)
            prev_end = start + duration

            formatted_entries.append(
                {"text": text, "start": start, "duration": duration}
            )

            # Check if we've hit the response limit
            current_text = (
                "\n".join(lines) if format != "continuous" else " ".join(lines)
            )
            if len(current_text) >= response_limit and i < len(result.snippets) - 1:
                # More content available, return next cursor
                next_cursor = str(i + 1)
                break
        else:
            next_cursor = None

        # Join based on format
        if format == "continuous":
            final_text = " ".join(lines)
        elif format == "paragraphs":
            final_text = "\n".join(lines)
        else:  # lines (default)
            final_text = "\n".join(lines)

        response = {
            "success": True,
            "method": "api",
            "language": result.language_code,
            "is_generated": result.is_generated,
            "text": final_text,
            "transcript": formatted_entries,
        }

        if next_cursor:
            response["next_cursor"] = next_cursor
            response["has_more"] = True

        return response

    except (TranscriptsDisabled, VideoUnavailable) as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": f"Unexpected API error: {e}"}


def _get_transcript_via_ytdlp(
    video_id: str, yt_dlp_path: str = "yt-dlp"
) -> Dict[str, Any]:
    """Fallback: Fetch transcript using yt-dlp subprocess."""
    url = f"https://www.youtube.com/watch?v={video_id}"

    with tempfile.TemporaryDirectory() as temp_dir:
        output_template = f"{temp_dir}/%(id)s"

        cmd = [
            yt_dlp_path,
            "--write-auto-sub",
            "--write-sub",
            "--sub-lang",
            "en,.*",
            "--skip-download",
            "--output",
            output_template,
            "--no-check-certificates",
            "--user-agent",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            url,
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=60)

            # Find the downloaded VTT file
            files = list(Path(temp_dir).glob("*.vtt"))
            if not files:
                return {
                    "success": False,
                    "error": "No subtitle files downloaded by yt-dlp",
                }

            # Prefer non-auto-generated if available
            vtt_file = files[0]
            for f in files:
                if not f.name.endswith(".en.vtt") and "auto" not in f.name:
                    vtt_file = f
                    break

            content = vtt_file.read_text(encoding="utf-8", errors="replace")
            clean_text = _strip_vtt_formatting(content)

            return {
                "success": True,
                "method": "yt-dlp",
                "text": clean_text,
                "transcript": [],  # VTT parsing for structured data is complex, skipping for fallback
                "metadata": {"file": vtt_file.name},
            }

        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "error": f"yt-dlp failed: {e.stderr.decode('utf-8', errors='replace')}",
            }
        except FileNotFoundError:
            return {"success": False, "error": "yt-dlp executable not found in PATH"}
        except Exception as e:
            return {"success": False, "error": f"yt-dlp error: {str(e)}"}


def _is_youtube_url(url: str) -> bool:
    """Check if URL is a YouTube URL."""
    try:
        parsed = urllib.parse.urlparse(url)
        return parsed.hostname in (
            "youtube.com",
            "www.youtube.com",
            "m.youtube.com",
            "youtu.be",
            "www.youtu.be",
        )
    except:
        return False


def _fetch_youtube_transcript(
    url: str,
    language: str = "en",
    timestamps: bool = False,
    format: str = "lines",
    http_proxy: Optional[str] = None,
    https_proxy: Optional[str] = None,
    cursor: Optional[str] = None,
    response_limit: int = 50000,
) -> Dict[str, Any]:
    """Fetch YouTube transcript with API -> yt-dlp fallback.

    Args:
        url: YouTube URL or video ID
        language: Language code
        timestamps: Include timestamps
        format: Output format (lines/continuous/paragraphs)
        http_proxy: HTTP proxy URL
        https_proxy: HTTPS proxy URL
        cursor: Pagination cursor
        response_limit: Max chars per response
    """
    try:
        video_id = _extract_video_id(url)

        # Get proxies from env if not provided
        if http_proxy is None:
            http_proxy = os.environ.get("HTTP_PROXY")
        if https_proxy is None:
            https_proxy = os.environ.get("HTTPS_PROXY")

        # 1. Try API
        result = _get_transcript_via_api(
            video_id,
            language,
            timestamps,
            format,
            http_proxy,
            https_proxy,
            cursor,
            response_limit,
        )
        if result["success"]:
            return result

        # 2. Try yt-dlp (no pagination support for fallback)
        if cursor is None:  # Only try fallback on first request
            result = _get_transcript_via_ytdlp(video_id)
            if result["success"]:
                return result

        return {
            "success": False,
            "error": "All YouTube extraction methods failed",
            "details": result,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def _get_video_info(video_id: str, yt_dlp_path: str = "yt-dlp") -> Dict[str, Any]:
    """Get YouTube video metadata using yt-dlp."""
    url = f"https://www.youtube.com/watch?v={video_id}"

    cmd = [
        yt_dlp_path,
        "--dump-json",
        "--no-download",
        "--no-warnings",
        url,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=30, text=True)
        if result.returncode != 0:
            return {"success": False, "error": f"yt-dlp failed: {result.stderr}"}

        data = json.loads(result.stdout)

        # Extract relevant fields (all but thumbnail)
        info = {
            "success": True,
            "video_id": video_id,
            "title": data.get("title"),
            "description": data.get("description"),
            "channel": data.get("channel"),
            "channel_id": data.get("channel_id"),
            "upload_date": data.get("upload_date"),  # YYYYMMDD format
            "duration": data.get("duration"),  # seconds
            "duration_string": data.get("duration_string"),
            "view_count": data.get("view_count"),
            "like_count": data.get("like_count"),
            "tags": data.get("tags", []),
            "categories": data.get("categories", []),
        }

        return info

    except subprocess.TimeoutExpired:
        return {"success": False, "error": "yt-dlp timed out"}
    except FileNotFoundError:
        return {"success": False, "error": "yt-dlp executable not found in PATH"}
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"Failed to parse yt-dlp output: {e}"}
    except Exception as e:
        return {"success": False, "error": f"Video info error: {str(e)}"}


def _download_video(
    video_id: str,
    output_dir: str = ".output/videos",
    format: str = "best",
    yt_dlp_path: str = "yt-dlp",
) -> Dict[str, Any]:
    """Download YouTube video using yt-dlp."""
    url = f"https://www.youtube.com/watch?v={video_id}"

    # Ensure output directory exists
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    output_template = str(out_path / "%(title)s.%(ext)s")

    cmd = [
        yt_dlp_path,
        "-f", format,
        "-o", output_template,
        "--no-warnings",
        "--newline",  # Progress on new lines
        url,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=600, text=True)

        if result.returncode != 0:
            return {"success": False, "error": f"yt-dlp failed: {result.stderr}"}

        # Find the downloaded file
        # yt-dlp prints "Destination: filename" or "[download] filename has already been downloaded"
        lines = result.stdout.strip().split("\n")
        filename = None
        for line in lines:
            if "Destination:" in line:
                filename = line.split("Destination:")[-1].strip()
            elif "has already been downloaded" in line:
                # Extract filename from "[download] filename has already been downloaded"
                filename = line.replace("[download]", "").replace("has already been downloaded", "").strip()

        return {
            "success": True,
            "video_id": video_id,
            "output_dir": str(out_path.resolve()),
            "filename": filename,
            "stdout": result.stdout,
        }

    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Download timed out (10 min limit)"}
    except FileNotFoundError:
        return {"success": False, "error": "yt-dlp executable not found in PATH"}
    except Exception as e:
        return {"success": False, "error": f"Download error: {str(e)}"}


# === WEB SEARCH & FETCH ===


def _duckduckgo_search(query: str, max_results: int = 10) -> List[Dict[str, str]]:
    """
    Perform a DuckDuckGo search by scraping the HTML (Zero-API approach).
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    url = "https://html.duckduckgo.com/html/"
    data = {"q": query}

    try:
        resp = httpx.post(url, data=data, headers=headers, timeout=10.0)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        results = []

        for result in soup.find_all("div", class_="result"):
            if len(results) >= max_results:
                break

            title_tag = result.find("a", class_="result__a")
            if not title_tag:
                continue

            link = title_tag["href"]
            title = title_tag.get_text(strip=True)
            snippet_tag = result.find("a", class_="result__snippet")
            snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""

            results.append({"title": title, "link": link, "snippet": snippet})

        return results
    except Exception as e:
        return [{"error": f"Search failed: {str(e)}"}]


def _brave_search(
    query: str, count: int = 10, api_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Perform a search using the Brave Search API.
    Requires BRAVE_API_KEY.
    """
    key = api_key or os.environ.get("BRAVE_API_KEY")
    if not key:
        return [
            {
                "error": "Missing Brave API Key",
                "message": "Brave Search requires an API key. Please set BRAVE_API_KEY environment variable or pass it as an argument.",
                "action": "Get a free key at https://brave.com/search/api/",
            }
        ]

    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": key,
    }
    params = {
        "q": query,
        "count": min(count, 20),  # Brave max is 20
    }

    try:
        resp = httpx.get(url, headers=headers, params=params, timeout=10.0)
        if resp.status_code == 401:
            return [
                {
                    "error": "Invalid Brave API Key",
                    "message": "The provided API key was rejected.",
                    "action": "Check your key at https://brave.com/search/api/",
                }
            ]
        resp.raise_for_status()
        data = resp.json()

        results = []
        if "web" in data and "results" in data["web"]:
            for item in data["web"]["results"]:
                results.append(
                    {
                        "title": item.get("title"),
                        "link": item.get("url"),
                        "snippet": item.get("description"),
                        "published": item.get("age", ""),
                    }
                )
        return results
    except Exception as e:
        return [{"error": f"Brave Search failed: {str(e)}"}]


def _fetch_page_content(url: str) -> Dict[str, Any]:
    """
    Fetch and convert page content to Markdown.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        resp = httpx.get(url, headers=headers, timeout=15.0, follow_redirects=True)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Cleanup
        for tag in soup(["script", "style", "nav", "footer", "aside", "iframe"]):
            tag.decompose()

        # Extract Title
        title = soup.title.string if soup.title else url

        # Convert to Markdown
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = True
        h.body_width = 0
        markdown = h.handle(str(soup))

        return {"url": url, "title": title.strip(), "content": markdown.strip()}
    except Exception as e:
        return {"error": f"Fetch failed: {str(e)}"}


def _save_as_ymj(url: str, title: str, content: str, save_dir: str) -> str:
    """Save content as YMJ file with embeddings."""
    try:
        from fastembed import TextEmbedding
    except ImportError:
        return "Error: fastembed not installed."

    # 1. Prepare Metadata
    safe_title = re.sub(r'[\\/*?:"<>|]', "", title).replace(" ", "_")[:50]
    filename = f"{safe_title}.ymj"
    path = Path(save_dir) / filename

    # 2. Create Header
    header = {
        "doc_summary": f"Web fetch of {title}",
        "kind": "web_clip",
        "version": "1.0.0",
        "subject": title,
        "maintained_by": "sfa_web",
        "ymj_spec": "1.0.0",
        "created": datetime.datetime.now().isoformat(),
        "provenance": {"source": url, "tool": "sfa_web"},
    }

    # 3. Create Body
    body = content

    # 4. Create Footer (Embeddings)
    payload_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()

    try:
        model = TextEmbedding(model_name="nomic-ai/nomic-embed-text-v1.5")
        vectors = list(model.embed([body]))
        vector = vectors[0].tolist()
    except Exception as e:
        return f"Error generating embedding: {e}"

    footer = {
        "schema": "1",
        "payload_hash": payload_hash,
        "index": {
            "embedding": vector,
            "model": "nomic-embed-text-v1.5",
            "tags": ["web_clip"],
        },
    }

    # 5. Write File
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write("---\n")
        yaml.dump(header, f, sort_keys=False)
        f.write("---\n\n")
        f.write(body)
        f.write("\n\n```json\n")
        json.dump(footer, f, indent=2)
        f.write("\n```\n")

    return str(path)


# --- MCP Tools ---

if mcp:

    @mcp.tool()
    def search(query: str, max_results: int = 10) -> str:
        """Search the web using DuckDuckGo."""
        results = _duckduckgo_search(query, max_results)
        return json.dumps(results, indent=2)

    @mcp.tool()
    def brave_search(query: str, count: int = 10) -> str:
        """
        Search the web using Brave Search.
        Requires BRAVE_API_KEY environment variable.
        """
        results = _brave_search(query, count)
        return json.dumps(results, indent=2)

    @mcp.tool()
    def fetch(url: str, save_to: Optional[str] = None) -> str:
        """
        Fetch a webpage and convert to Markdown.
        Args:
            save_to: Directory to save the content as a .ymj file (with embeddings).
        """
        result = _fetch_page_content(url)
        if "error" in result:
            return f"Error: {result['error']}"

        output = f"# {result['title']}\n\n{result['content']}"

        if save_to:
            saved_path = _save_as_ymj(url, result["title"], result["content"], save_to)
            output += f"\n\nSaved to: {saved_path}"
            output += f"\nFormat Spec: https://github.com/baldsam/ymj-spec"
            output += f"\nTip: Use 'sfa_ymj.py' to manage this file."

        return output

    @mcp.tool()
    def youtube_transcript(
        url: str,
        language: str = "en",
        timestamps: bool = False,
        format: str = "lines",
        cursor: str = None,
        response_limit: int = 50000,
    ) -> str:
        """
        Fetch YouTube transcript with hybrid extraction (API + yt-dlp fallback).
        Supports pagination for long transcripts.

        Args:
            url: YouTube URL or video ID
            language: Language code (default: en)
            timestamps: Include timestamps in output
            format: Output format - "lines" (default, one per subtitle), "continuous" (space-joined), "paragraphs" (grouped by gaps)
            cursor: Pagination cursor (from previous response's next_cursor)
            response_limit: Max characters per response (default: 50000)

        Returns:
            Transcript text. If paginated, response will include next_cursor for subsequent requests.
        """
        result = _fetch_youtube_transcript(
            url,
            language,
            timestamps,
            format,
            cursor=cursor,
            response_limit=response_limit,
        )
        if result["success"]:
            text = result["text"]
            if result.get("next_cursor"):
                text += f"\n\n[TRUNCATED - More content available. Use cursor: {result['next_cursor']}]"
            return text
        else:
            return f"Error: {result.get('error')}"

# --- CLI Dispatcher ---


def main():
    parser = argparse.ArgumentParser(description="SFA Web - Research & Retrieval")
    subparsers = parser.add_subparsers(dest="command")

    # search (Brave - default) - alias: s
    search_parser = subparsers.add_parser(
        "search", aliases=["s"], help="Search the web (Brave, default)"
    )
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--count", type=int, default=10, help="Max results")
    search_parser.add_argument("--key", help="Brave API Key (overrides env var)")

    # ddg (DuckDuckGo - alternate) - alias: sddg
    ddg_parser = subparsers.add_parser(
        "ddg", aliases=["sddg"], help="Search the web (DuckDuckGo)"
    )
    ddg_parser.add_argument("query", help="Search query")
    ddg_parser.add_argument("--count", type=int, default=10, help="Max results")

    # fetch - alias: f
    fetch_parser = subparsers.add_parser(
        "fetch", aliases=["f"], help="Fetch webpage as Markdown"
    )
    fetch_parser.add_argument("url", help="URL to fetch")
    fetch_parser.add_argument("--save-to", "-s", help="Directory to save as .ymj")

    # transcript (YouTube transcript) - alias: yt
    transcript_parser = subparsers.add_parser(
        "transcript", aliases=["yt"], help="Fetch YouTube transcript"
    )
    transcript_parser.add_argument("url", help="YouTube URL or video ID")
    transcript_parser.add_argument(
        "--lang", default="en", help="Language code (default: en)"
    )
    transcript_parser.add_argument(
        "--timestamps", action="store_true", help="Include timestamps"
    )
    transcript_parser.add_argument(
        "--format",
        choices=["lines", "continuous", "paragraphs"],
        default="lines",
        help="Output format: lines (one per subtitle), continuous (space-joined), paragraphs (grouped by gaps)",
    )
    transcript_parser.add_argument(
        "--http-proxy", help="HTTP proxy URL (or set HTTP_PROXY env var)"
    )
    transcript_parser.add_argument(
        "--https-proxy", help="HTTPS proxy URL (or set HTTPS_PROXY env var)"
    )
    transcript_parser.add_argument(
        "--cursor", help="Pagination cursor from previous response"
    )
    transcript_parser.add_argument(
        "--response-limit",
        type=int,
        default=50000,
        help="Max characters per response for pagination (default: 50000)",
    )
    transcript_parser.add_argument(
        "--json", action="store_true", help="Output full JSON result"
    )

    # video-info (YouTube metadata) - alias: yti
    info_parser = subparsers.add_parser(
        "video-info", aliases=["yti"], help="Get YouTube video metadata"
    )
    info_parser.add_argument("url", help="YouTube URL or video ID")
    info_parser.add_argument(
        "--json", action="store_true", help="Output as JSON (default: formatted text)"
    )

    # video-download (YouTube download) - alias: ytd
    dl_parser = subparsers.add_parser(
        "video-download", aliases=["ytd"], help="Download YouTube video"
    )
    dl_parser.add_argument("url", help="YouTube URL or video ID")
    dl_parser.add_argument(
        "--output", "-o", default=".output/videos", help="Output directory (default: .output/videos)"
    )
    dl_parser.add_argument(
        "--format", "-f", default="best", help="Video format (default: best)"
    )

    args = parser.parse_args()

    # Handle commands (check both canonical and alias names)
    if args.command in ("search", "s"):
        results = _brave_search(args.query, args.count, getattr(args, "key", None))
        print(json.dumps(results, indent=2))

    elif args.command in ("ddg", "sddg"):
        print(json.dumps(_duckduckgo_search(args.query, args.count), indent=2))

    elif args.command in ("fetch", "f"):
        res = _fetch_page_content(args.url)
        if "error" in res:
            print(f"Error: {res['error']}")
        else:
            print(f"# {res['title']}\n\n{res['content']}")
            if args.save_to:
                saved = _save_as_ymj(
                    args.url, res["title"], res["content"], args.save_to
                )
                print(f"\nSaved to: {saved}")
                print(f"Format Spec: https://github.com/baldsam/ymj-spec")
                print(f"Tip: Use 'sfa_ymj.py' to manage this file.")

    elif args.command in ("transcript", "yt"):
        result = _fetch_youtube_transcript(
            args.url,
            args.lang,
            args.timestamps,
            args.format,
            http_proxy=getattr(args, "http_proxy", None),
            https_proxy=getattr(args, "https_proxy", None),
            cursor=getattr(args, "cursor", None),
            response_limit=getattr(args, "response_limit", 50000),
        )
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            if result["success"]:
                print(result["text"])
                if result.get("next_cursor"):
                    print(
                        f"\n[TRUNCATED - More content available. Use --cursor {result['next_cursor']}]",
                        file=sys.stderr,
                    )
            else:
                print(f"Error: {result.get('error')}", file=sys.stderr)
                sys.exit(1)

    elif args.command in ("video-info", "yti"):
        try:
            video_id = _extract_video_id(args.url)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        result = _get_video_info(video_id)
        if not result["success"]:
            print(f"Error: {result['error']}", file=sys.stderr)
            sys.exit(1)

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            # Formatted text output
            print(f"Title: {result['title']}")
            print(f"Channel: {result['channel']}")
            print(f"Duration: {result['duration_string']}")
            print(f"Views: {result['view_count']:,}" if result['view_count'] else "Views: N/A")
            print(f"Likes: {result['like_count']:,}" if result['like_count'] else "Likes: N/A")
            if result['upload_date']:
                d = result['upload_date']
                print(f"Uploaded: {d[:4]}-{d[4:6]}-{d[6:]}")
            if result['tags']:
                print(f"Tags: {', '.join(result['tags'][:10])}")
            if result['categories']:
                print(f"Categories: {', '.join(result['categories'])}")
            print(f"\n--- Description ---\n{result['description']}")

    elif args.command in ("video-download", "ytd"):
        try:
            video_id = _extract_video_id(args.url)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        print(f"Downloading video {video_id}...", file=sys.stderr)
        result = _download_video(video_id, args.output, args.format)

        if not result["success"]:
            print(f"Error: {result['error']}", file=sys.stderr)
            sys.exit(1)

        print(f"Downloaded to: {result['output_dir']}")
        if result['filename']:
            print(f"Filename: {result['filename']}")

    else:
        if mcp:
            mcp.run()
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
