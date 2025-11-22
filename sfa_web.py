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
from pathlib import Path
from typing import List, Dict, Any, Optional

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

# --- Helpers ---

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
            
            results.append({
                "title": title,
                "link": link,
                "snippet": snippet
            })
            
        return results
    except Exception as e:
        return [{"error": f"Search failed: {str(e)}"}]

def _brave_search(query: str, count: int = 10, api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Perform a search using the Brave Search API.
    Requires BRAVE_API_KEY.
    """
    key = api_key or os.environ.get("BRAVE_API_KEY")
    if not key:
        return [{
            "error": "Missing Brave API Key",
            "message": "Brave Search requires an API key. Please set BRAVE_API_KEY environment variable or pass it as an argument.",
            "action": "Get a free key at https://brave.com/search/api/"
        }]

    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": key
    }
    params = {
        "q": query,
        "count": min(count, 20) # Brave max is 20
    }

    try:
        resp = httpx.get(url, headers=headers, params=params, timeout=10.0)
        if resp.status_code == 401:
             return [{
                "error": "Invalid Brave API Key",
                "message": "The provided API key was rejected.",
                "action": "Check your key at https://brave.com/search/api/"
            }]
        resp.raise_for_status()
        data = resp.json()
        
        results = []
        if "web" in data and "results" in data["web"]:
            for item in data["web"]["results"]:
                results.append({
                    "title": item.get("title"),
                    "link": item.get("url"),
                    "snippet": item.get("description"),
                    "published": item.get("age", "")
                })
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
        for tag in soup(['script', 'style', 'nav', 'footer', 'aside', 'iframe']):
            tag.decompose()
            
        # Extract Title
        title = soup.title.string if soup.title else url
        
        # Convert to Markdown
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = True
        h.body_width = 0
        markdown = h.handle(str(soup))
        
        return {
            "url": url,
            "title": title.strip(),
            "content": markdown.strip()
        }
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
        "provenance": {
            "source": url,
            "tool": "sfa_web"
        }
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
            "tags": ["web_clip"]
        }
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
            saved_path = _save_as_ymj(url, result['title'], result['content'], save_to)
            output += f"\n\nSaved to: {saved_path}"
            output += f"\nFormat Spec: https://github.com/baldsam/ymj-spec"
            output += f"\nTip: Use 'sfa_ymj.py' to manage this file."
            
        return output

# --- CLI Dispatcher ---

def main():
    parser = argparse.ArgumentParser(description="SFA Web - Research & Retrieval")
    subparsers = parser.add_subparsers(dest="command")

    # search
    search_parser = subparsers.add_parser("search", help="Search the web (DuckDuckGo)")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--count", type=int, default=10, help="Max results")

    # brave-search
    brave_parser = subparsers.add_parser("brave-search", help="Search the web (Brave)")
    brave_parser.add_argument("query", help="Search query")
    brave_parser.add_argument("--count", type=int, default=10, help="Max results")
    brave_parser.add_argument("--key", help="Brave API Key (overrides env var)")

    # fetch
    fetch_parser = subparsers.add_parser("fetch", help="Fetch webpage as Markdown")
    fetch_parser.add_argument("url", help="URL to fetch")
    fetch_parser.add_argument("--save-to", "-s", help="Directory to save as .ymj")

    args = parser.parse_args()

    if args.command == "search":
        print(json.dumps(_duckduckgo_search(args.query, args.count), indent=2))
    elif args.command == "brave-search":
        print(json.dumps(_brave_search(args.query, args.count, args.key), indent=2))
    elif args.command == "fetch":
        res = _fetch_page_content(args.url)
        if "error" in res:
            print(f"Error: {res['error']}")
        else:
            print(f"# {res['title']}\n\n{res['content']}")
            if args.save_to:
                saved = _save_as_ymj(args.url, res['title'], res['content'], args.save_to)
                print(f"\nSaved to: {saved}")
                print(f"Format Spec: https://github.com/baldsam/ymj-spec")
                print(f"Tip: Use 'sfa_ymj.py' to manage this file.")
    else:
        if mcp:
            mcp.run()
        else:
            parser.print_help()

if __name__ == "__main__":
    main()
