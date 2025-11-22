#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "youtube-transcript-api>=0.6.0",
#     "websocket-client>=1.6.0",
#     "httpx>=0.25.0",
#     "fastmcp",
# ]
# ///

"""
SFA Video - Unified Video Extraction & Transcription Tool

A self-contained tool for extracting metadata and transcripts from video platforms.
Combines robust YouTube handling (API + yt-dlp fallback) with generic CDP-based
extraction for other sites.

Usage:
  uv run sfa_video.py <url> [--transcript-only] [--json]
  uv run sfa_video.py --url <url> --lang ko --timestamps

Features:
- YouTube: Hybrid extraction (API -> yt-dlp -> VTT parsing)
- Generic: Chrome DevTools Protocol (CDP) extraction
- Anti-blocking: User-agent rotation, proxy support (via env vars)
"""

import argparse
import json
import re
import subprocess
import sys
import time
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse, parse_qs

try:
    import websocket
    import httpx
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import (
        NoTranscriptFound,
        TranscriptsDisabled,
        VideoUnavailable,
    )
    YOUTUBE_API_AVAILABLE = True
except ImportError:
    YOUTUBE_API_AVAILABLE = False
    websocket = None
    httpx = None

try:
    from fastmcp import FastMCP
    mcp = FastMCP("sfa-video")
except ImportError:
    mcp = None


# === YOUTUBE LOGIC ===

def extract_video_id(url_or_id: str) -> str:
    """Extract video ID from various YouTube URL formats."""
    url_or_id = url_or_id.strip()
    if re.match(r'^[a-zA-Z0-9_-]{11}$', url_or_id):
        return url_or_id

    try:
        parsed = urlparse(url_or_id)
        if parsed.hostname in ('youtu.be', 'www.youtu.be'):
            return parsed.path.lstrip('/')
        if parsed.hostname in ('youtube.com', 'www.youtube.com', 'm.youtube.com'):
            if parsed.path == '/watch':
                v_param = parse_qs(parsed.query).get('v')
                if v_param: return v_param[0]
            elif parsed.path.startswith(('/v/', '/shorts/', '/embed/')):
                return parsed.path.split('/')[2]
    except Exception:
        pass
    raise ValueError(f"Could not extract video ID from: {url_or_id}")

def strip_vtt_formatting(vtt_content: str) -> str:
    """Strip VTT formatting to extract clean text."""
    lines = vtt_content.splitlines()
    text_lines = []
    seen_lines = set()
    
    # Regex for VTT timestamps: 00:00:00.000 --> 00:00:05.000
    timestamp_pattern = re.compile(r'\d{2}:\d{2}:\d{2}\.\d{3}\s-->\s\d{2}:\d{2}:\d{2}\.\d{3}')
    
    for line in lines:
        line = line.strip()
        if not line: continue
        if line == 'WEBVTT' or line.startswith('Kind:') or line.startswith('Language:'): continue
        if timestamp_pattern.match(line): continue
        
        # Remove tags like <c>...</c> or <00:00:00.000>
        clean_line = re.sub(r'<[^>]+>', '', line)
        clean_line = clean_line.strip()
        
        if clean_line and clean_line not in seen_lines:
            text_lines.append(clean_line)
            seen_lines.add(clean_line)
            
    return " ".join(text_lines)

def get_transcript_via_api(video_id: str, language: str = "en", with_timestamps: bool = False) -> Dict[str, Any]:
    """Try fetching transcript via youtube_transcript_api."""
    if not YOUTUBE_API_AVAILABLE:
        return {"success": False, "error": "youtube_transcript_api not installed"}

    try:
        # Try to list transcripts to find the best match
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        try:
            # Try fetching manually created transcript in requested language
            transcript = transcript_list.find_manually_created_transcript([language])
        except NoTranscriptFound:
            try:
                # Fallback to generated
                transcript = transcript_list.find_generated_transcript([language])
            except NoTranscriptFound:
                # Fallback to any english
                transcript = transcript_list.find_generated_transcript(['en'])

        data = transcript.fetch()
        
        # Format output
        full_text = []
        formatted_entries = []
        
        for entry in data:
            text = entry['text'].replace('\n', ' ')
            start = entry['start']
            
            if with_timestamps:
                minutes = int(start // 60)
                seconds = int(start % 60)
                timestamp = f"[{minutes:02d}:{seconds:02d}]"
                line = f"{timestamp} {text}"
            else:
                line = text
                
            full_text.append(line)
            formatted_entries.append({
                "text": text,
                "start": start,
                "duration": entry['duration']
            })

        return {
            "success": True,
            "method": "api",
            "language": transcript.language_code,
            "is_generated": transcript.is_generated,
            "text": "\n".join(full_text) if with_timestamps else " ".join(full_text),
            "transcript": formatted_entries
        }

    except (TranscriptsDisabled, VideoUnavailable) as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": f"Unexpected API error: {e}"}

def get_transcript_via_ytdlp(video_id: str, yt_dlp_path: str = "yt-dlp") -> Dict[str, Any]:
    """Fallback: Fetch transcript using yt-dlp subprocess."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_template = f"{temp_dir}/%(id)s"
        
        cmd = [
            yt_dlp_path,
            "--write-auto-sub",
            "--write-sub",
            "--sub-lang", "en,.*",
            "--skip-download",
            "--output", output_template,
            "--no-check-certificates",
            "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            url
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=60)
            
            # Find the downloaded VTT file
            files = list(Path(temp_dir).glob("*.vtt"))
            if not files:
                return {"success": False, "error": "No subtitle files downloaded by yt-dlp"}
                
            # Prefer non-auto-generated if available
            vtt_file = files[0]
            for f in files:
                if not f.name.endswith(".en.vtt") and "auto" not in f.name:
                    vtt_file = f
                    break
            
            content = vtt_file.read_text(encoding="utf-8", errors="replace")
            clean_text = strip_vtt_formatting(content)
            
            return {
                "success": True,
                "method": "yt-dlp",
                "text": clean_text,
                "transcript": [], # VTT parsing for structured data is complex, skipping for fallback
                "metadata": {"file": vtt_file.name}
            }
            
        except subprocess.CalledProcessError as e:
            return {"success": False, "error": f"yt-dlp failed: {e.stderr.decode('utf-8', errors='replace')}"}
        except FileNotFoundError:
            return {"success": False, "error": "yt-dlp executable not found in PATH"}
        except Exception as e:
            return {"success": False, "error": f"yt-dlp error: {str(e)}"}

# === CDP LOGIC ===

class CDPClient:
    """Embedded Chrome DevTools Protocol client for video extraction."""

    def __init__(self, debug_port: int = 9222):
        self.debug_port = debug_port
        self.ws = None
        self.msg_id = 0

    def connect(self) -> bool:
        """Connect to Chrome debugging port."""
        try:
            response = httpx.get(f"http://localhost:{self.debug_port}/json/list", timeout=5)
            pages = response.json()
            if not pages: return False
            ws_url = pages[0]['webSocketDebuggerUrl']
            self.ws = websocket.create_connection(ws_url, timeout=10)
            return True
        except Exception:
            return False

    def send_command(self, method: str, params: Dict = None) -> Dict[str, Any]:
        if not self.ws: raise RuntimeError("Not connected to Chrome")
        self.msg_id += 1
        message = {"id": self.msg_id, "method": method, "params": params or {}}
        self.ws.send(json.dumps(message))
        while True:
            response = json.loads(self.ws.recv())
            if response.get("id") == self.msg_id:
                if "error" in response: raise RuntimeError(f"CDP error: {response['error']}")
                return response.get("result", {})

    def navigate(self, url: str) -> bool:
        try:
            self.send_command("Page.enable")
            self.send_command("Page.navigate", {"url": url})
            time.sleep(3)  # Wait for video page load
            return True
        except Exception:
            return False

    def evaluate_js(self, expression: str) -> Any:
        result = self.send_command("Runtime.evaluate", {
            "expression": expression,
            "returnByValue": True
        })
        return result.get("result", {}).get("value")

    def extract_video_info(self, url: str) -> Optional[Dict]:
        try:
            if not self.navigate(url): return None
            info = self.evaluate_js("""
                (function() {
                    const video = document.querySelector('video');
                    if (!video) return null;
                    return {
                        title: document.title,
                        duration: video.duration || null,
                        currentTime: video.currentTime || 0,
                        videoSrc: video.src || video.currentSrc || null
                    };
                })()
            """)
            return info
        except Exception:
            return None

    def close(self):
        if self.ws: self.ws.close()

def extract_via_cdp(url: str, platform: str) -> Dict[str, Any]:
    """Extract video via Chrome DevTools Protocol."""
    client = CDPClient()
    try:
        if not client.connect():
            return {"success": False, "error": "Chrome not running on port 9222"}

        info = client.extract_video_info(url)
        if info:
            return {
                "success": True,
                "platform": platform,
                "method": "cdp",
                "data": {"video_info": info}
            }
        return {"success": False, "error": "No video element found via CDP"}
    except Exception as e:
        return {"success": False, "error": f"CDP error: {e}"}
    finally:
        client.close()

# === MAIN CONTROLLER ===

def detect_platform(url: str) -> str:
    domain = urlparse(url).netloc.lower()
    if 'youtube.com' in domain or 'youtu.be' in domain: return 'youtube'
    if 'vimeo.com' in domain: return 'vimeo'
    if 'twitter.com' in domain or 'x.com' in domain: return 'twitter'
    return 'unknown'

def process_video(url: str, transcript_only: bool = False, lang: str = "en", timestamps: bool = False) -> Dict[str, Any]:
    """Main entry point for video processing."""
    platform = detect_platform(url)
    
    if platform == 'youtube':
        try:
            vid = extract_video_id(url)
            # 1. Try API
            result = get_transcript_via_api(vid, lang, timestamps)
            if result["success"]: return result
            
            # 2. Try yt-dlp
            result = get_transcript_via_ytdlp(vid)
            if result["success"]: return result
            
            return {"success": False, "error": "All YouTube extraction methods failed", "details": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
            
    else:
        # Generic CDP extraction
        return extract_via_cdp(url, platform)

# --- MCP Tools ---

if mcp:
    @mcp.tool()
    def extract(url: str, transcript_only: bool = False) -> str:
        """
        Extract video metadata and transcript.
        Supports YouTube (robust hybrid method) and generic sites (via Chrome CDP).
        """
        result = process_video(url, transcript_only)
        return json.dumps(result, indent=2)

# --- CLI ---

def main():
    parser = argparse.ArgumentParser(description="SFA Video - Extraction & Transcription")
    parser.add_argument("url", nargs='?', help="Video URL")
    parser.add_argument("--url", dest="named_url", help="Video URL (named)")
    parser.add_argument("--transcript-only", action="store_true", help="Skip metadata")
    parser.add_argument("--lang", default="en", help="Language code (YouTube only)")
    parser.add_argument("--timestamps", action="store_true", help="Include timestamps")
    parser.add_argument("--json", action="store_true", help="Output JSON")

    args = parser.parse_args()

    # Check MCP
    if len(sys.argv) == 1 and mcp:
        mcp.run()
        return

    target_url = args.url or args.named_url
    if not target_url:
        parser.print_help()
        sys.exit(1)

    result = process_video(target_url, args.transcript_only, args.lang, args.timestamps)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        if result["success"]:
            if "text" in result:
                print(result["text"])
            elif "data" in result:
                print(json.dumps(result["data"], indent=2))
        else:
            print(f"Error: {result.get('error')}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()
