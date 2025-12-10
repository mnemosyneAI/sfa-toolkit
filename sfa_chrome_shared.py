#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "websocket-client>=1.6.0",
#     "httpx>=0.25.0",
#     "fastmcp",
# ]
# ///

"""
SFA Chrome Shared - Shared User Profile Chrome Automation

This tool is a variant of sfa_chrome_ai.py designed to run with a specific
shared user profile ("user-ai") on a dedicated port (9223).

It automatically launches Chrome if it's not running.
"""

import argparse
import base64
import json
import sys
import time
import os
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

try:
    from fastmcp import FastMCP
    mcp = FastMCP("sfa-chrome-shared")
except ImportError:
    mcp = None

import websocket
import httpx

# --- Configuration ---
DEFAULT_PORT = 9223
CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PROFILE_DIR_NAME = "user-ai"

def get_profile_path():
    # Create a local 'chrome_profiles' directory in the toolkit folder
    base_dir = Path(__file__).parent / "chrome_profiles"
    return base_dir / PROFILE_DIR_NAME

def launch_chrome(port: int = DEFAULT_PORT):
    """Launch Chrome with the shared profile if not already running."""
    profile_path = get_profile_path()
    
    # Ensure profile directory exists
    if not profile_path.exists():
        print(f"Creating new profile directory at: {profile_path}", file=sys.stderr)
        profile_path.mkdir(parents=True, exist_ok=True)
    
    # Check if already running on this port
    try:
        httpx.get(f"http://localhost:{port}/json", timeout=1.0)
        # If successful, it's running
        return
    except Exception:
        pass # Not running, proceed to launch

    print(f"Launching Chrome on port {port} with profile 'user-ai'...", file=sys.stderr)
    
    cmd = [
        CHROME_PATH,
        f"--remote-debugging-port={port}",
        f"--user-data-dir={profile_path}",
        "--remote-allow-origins=*",
        "--no-first-run",
        "--no-default-browser-check"
    ]
    
    # Launch in background
    try:
        subprocess.Popen(cmd)
    except (FileNotFoundError, OSError) as e:
        print(f"Error: Failed to launch Chrome: {e}", file=sys.stderr)
        return
    
    # Wait for it to come up
    for i in range(10):
        try:
            httpx.get(f"http://localhost:{port}/json", timeout=1.0)
            print("Chrome launched successfully.", file=sys.stderr)
            return
        except Exception:
            time.sleep(1)
            
    print("Warning: Chrome launch initiated but port not yet responsive.", file=sys.stderr)

# --- Core CDP Client ---

class CDPClient:
    """Base client for Chrome DevTools Protocol interactions."""
    
    def __init__(self, debug_port: int = DEFAULT_PORT):
        self.debug_port = debug_port
        self.ws: Optional[websocket.WebSocket] = None
        self.msg_id = 0
        self.target_id: Optional[str] = None
        self.session_id: Optional[str] = None

    def connect(self, page_idx: int = 0) -> bool:
        """Connect to a specific page target."""
        # Auto-launch if needed
        launch_chrome(self.debug_port)
        
        try:
            # Get list of pages
            response = httpx.get(f"http://localhost:{self.debug_port}/json")
            targets = response.json()
            
            # Filter for page targets
            pages = [t for t in targets if t['type'] == 'page']
            
            if not pages:
                # If no pages (rare), try to create one? Or just fail.
                # Usually there's at least one tab.
                print("No open pages found.", file=sys.stderr)
                return False
                
            if page_idx >= len(pages):
                print(f"Page index {page_idx} out of range (0-{len(pages)-1})", file=sys.stderr)
                return False
                
            target = pages[page_idx]
            self.target_id = target['id']
            ws_url = target.get('webSocketDebuggerUrl')
            
            if not ws_url:
                print("No WebSocket URL found for target.", file=sys.stderr)
                return False

            self.ws = websocket.create_connection(ws_url)
            return True
            
        except Exception as e:
            print(f"Connection failed: {e}", file=sys.stderr)
            return False

    def close(self):
        if self.ws:
            self.ws.close()

    def send_command(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send a CDP command and wait for the result."""
        if not self.ws:
            raise RuntimeError("Not connected to Chrome")
            
        self.msg_id += 1
        message = {
            "id": self.msg_id,
            "method": method,
            "params": params or {}
        }
        
        self.ws.send(json.dumps(message))
        
        while True:
            resp = json.loads(self.ws.recv())
            if resp.get("id") == self.msg_id:
                if "error" in resp:
                    raise RuntimeError(f"CDP Error: {resp['error']['message']}")
                return resp.get("result", {})

    def enable_domains(self):
        """Enable necessary CDP domains."""
        domains = ["Page", "DOM", "Runtime", "Input", "Network"]
        for domain in domains:
            try:
                self.send_command(f"{domain}.enable")
            except Exception:
                pass 

# --- Functionality Modules ---

class Navigator(CDPClient):
    def list_pages(self) -> List[Dict[str, Any]]:
        launch_chrome(self.debug_port)
        try:
            response = httpx.get(f"http://localhost:{self.debug_port}/json")
            return [t for t in response.json() if t['type'] == 'page']
        except Exception as e:
            print(f"Failed to list pages: {e}", file=sys.stderr)
            return []

    def goto(self, url: str):
        self.send_command("Page.navigate", {"url": url})
        time.sleep(1) 

    def reload(self, ignore_cache: bool = False):
        self.send_command("Page.reload", {"ignoreCache": ignore_cache})

    def go_back(self):
        history = self.send_command("Page.getNavigationHistory")
        idx = history.get("currentIndex", 0)
        if idx > 0:
            entry_id = history["entries"][idx - 1]["id"]
            self.send_command("Page.navigateToHistoryEntry", {"entryId": entry_id})

    def go_forward(self):
        history = self.send_command("Page.getNavigationHistory")
        idx = history.get("currentIndex", 0)
        entries = history.get("entries", [])
        if idx < len(entries) - 1:
            entry_id = entries[idx + 1]["id"]
            self.send_command("Page.navigateToHistoryEntry", {"entryId": entry_id})

    def new_tab(self, url: str = "about:blank"):
        launch_chrome(self.debug_port)
        try:
            httpx.put(f"http://localhost:{self.debug_port}/json/new?{url}")
        except Exception as e:
            print(f"Failed to create tab: {e}", file=sys.stderr)

    def close_tab(self, page_idx: int):
        pages = self.list_pages()
        if 0 <= page_idx < len(pages):
            target_id = pages[page_idx]['id']
            try:
                httpx.get(f"http://localhost:{self.debug_port}/json/close/{target_id}")
            except Exception as e:
                print(f"Failed to close tab: {e}", file=sys.stderr)

class Extractor(CDPClient):
    def evaluate_js(self, expression: str) -> Any:
        result = self.send_command("Runtime.evaluate", {
            "expression": expression,
            "returnByValue": True,
            "awaitPromise": True
        })
        return result.get("result", {}).get("value")

    def take_screenshot(self, output_path: str):
        result = self.send_command("Page.captureScreenshot")
        data = base64.b64decode(result["data"])
        Path(output_path).write_bytes(data)

    def get_snapshot(self) -> Dict[str, Any]:
        root = self.send_command("DOM.getDocument", {"depth": -1})
        return root

class Interactor(CDPClient):
    def get_element_center(self, selector: str) -> Optional[Dict[str, int]]:
        doc = self.send_command("DOM.getDocument")
        root_id = doc["root"]["nodeId"]
        
        node = self.send_command("DOM.querySelector", {
            "nodeId": root_id,
            "selector": selector
        })
        
        if not node.get("nodeId"):
            return None
            
        model = self.send_command("DOM.getBoxModel", {"nodeId": node["nodeId"]})
        content = model["model"]["content"]
        x = (content[0] + content[2]) / 2
        y = (content[1] + content[5]) / 2
        
        return {"x": int(x), "y": int(y), "nodeId": node["nodeId"]}

    def click(self, selector: str):
        center = self.get_element_center(selector)
        if not center:
            raise RuntimeError(f"Element not found: {selector}")
            
        self.send_command("Input.dispatchMouseEvent", {
            "type": "mousePressed",
            "x": center["x"], "y": center["y"],
            "button": "left", "clickCount": 1
        })
        self.send_command("Input.dispatchMouseEvent", {
            "type": "mouseReleased",
            "x": center["x"], "y": center["y"],
            "button": "left", "clickCount": 1
        })

    def fill(self, selector: str, value: str):
        self.evaluate_js(f"document.querySelector('{selector}').focus()")
        self.evaluate_js(f"document.querySelector('{selector}').select()")
        
        for char in value:
            self.send_command("Input.dispatchKeyEvent", {
                "type": "char",
                "text": char
            })

    def evaluate_js(self, expression: str):
        self.send_command("Runtime.evaluate", {"expression": expression})

    def press_key(self, key: str):
        self.send_command("Input.dispatchKeyEvent", {
            "type": "keyDown",
            "text": key
        })
        self.send_command("Input.dispatchKeyEvent", {
            "type": "keyUp",
            "text": key
        })

# --- Main CLI ---

def handle_navigate(args):
    nav = Navigator(args.port)
    
    if args.action == "list":
        pages = nav.list_pages()
        print(json.dumps(pages, indent=2))
        return

    if args.action == "new":
        nav.new_tab(args.url)
        print(json.dumps({"success": True, "action": "new_tab", "url": args.url}))
        return

    if args.action == "close":
        nav.close_tab(args.page_idx)
        print(json.dumps({"success": True, "action": "close_tab", "idx": args.page_idx}))
        return

    if not nav.connect(args.page_idx):
        sys.exit(1)
    
    try:
        nav.enable_domains()
        
        if args.action == "goto":
            nav.goto(args.url)
        elif args.action == "back":
            nav.go_back()
        elif args.action == "forward":
            nav.go_forward()
        elif args.action == "reload":
            nav.reload(args.ignore_cache)
            
        print(json.dumps({"success": True, "action": args.action}))
    finally:
        nav.close()

def handle_extract(args):
    ext = Extractor(args.port)
    if not ext.connect(args.page_idx):
        sys.exit(1)
        
    try:
        ext.enable_domains()
        
        if args.action == "js":
            result = ext.evaluate_js(args.expression)
            print(json.dumps({"success": True, "result": result}))
        elif args.action == "screenshot":
            ext.take_screenshot(args.output)
            print(json.dumps({"success": True, "file": args.output}))
        elif args.action == "snapshot":
            result = ext.get_snapshot()
            print(json.dumps({"success": True, "snapshot": "DOM Tree captured (truncated)"}))
            
    finally:
        ext.close()

def handle_interact(args):
    inter = Interactor(args.port)
    if not inter.connect(args.page_idx):
        sys.exit(1)
        
    try:
        inter.enable_domains()
        
        if args.action == "click":
            inter.click(args.selector)
        elif args.action == "fill":
            inter.fill(args.selector, args.value)
        elif args.action == "key":
            inter.press_key(args.key)
            
        print(json.dumps({"success": True, "action": args.action}))
    finally:
        inter.close()

def main():
    parser = argparse.ArgumentParser(description="SFA Chrome Shared - Shared User Profile Automation")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Chrome remote debugging port")
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Navigate Command
    nav_parser = subparsers.add_parser("navigate", help="Navigation commands")
    nav_subs = nav_parser.add_subparsers(dest="action", required=True)
    
    nav_subs.add_parser("list", help="List open tabs")
    
    goto_p = nav_subs.add_parser("goto", help="Navigate to URL")
    goto_p.add_argument("url")
    goto_p.add_argument("--page-idx", type=int, default=0)
    
    back_p = nav_subs.add_parser("back", help="Go back")
    back_p.add_argument("--page-idx", type=int, default=0)
    
    fwd_p = nav_subs.add_parser("forward", help="Go forward")
    fwd_p.add_argument("--page-idx", type=int, default=0)
    
    reload_p = nav_subs.add_parser("reload", help="Reload page")
    reload_p.add_argument("--ignore-cache", action="store_true")
    reload_p.add_argument("--page-idx", type=int, default=0)
    
    new_p = nav_subs.add_parser("new", help="New tab")
    new_p.add_argument("url", nargs="?", default="about:blank")
    
    close_p = nav_subs.add_parser("close", help="Close tab")
    close_p.add_argument("page_idx", type=int)

    # Extract Command
    ext_parser = subparsers.add_parser("extract", help="Extraction commands")
    ext_subs = ext_parser.add_subparsers(dest="action", required=True)
    
    js_p = ext_subs.add_parser("js", help="Evaluate JavaScript")
    js_p.add_argument("expression")
    js_p.add_argument("--page-idx", type=int, default=0)
    
    ss_p = ext_subs.add_parser("screenshot", help="Take screenshot")
    ss_p.add_argument("--output", required=True)
    ss_p.add_argument("--page-idx", type=int, default=0)
    
    snap_p = ext_subs.add_parser("snapshot", help="Get DOM snapshot")
    snap_p.add_argument("--page-idx", type=int, default=0)

    # Interact Command
    int_parser = subparsers.add_parser("interact", help="Interaction commands")
    int_subs = int_parser.add_subparsers(dest="action", required=True)
    
    click_p = int_subs.add_parser("click", help="Click element")
    click_p.add_argument("selector")
    click_p.add_argument("--page-idx", type=int, default=0)
    
    fill_p = int_subs.add_parser("fill", help="Fill input")
    fill_p.add_argument("selector")
    fill_p.add_argument("value")
    fill_p.add_argument("--page-idx", type=int, default=0)
    
    key_p = int_subs.add_parser("key", help="Press key")
    key_p.add_argument("key")
    key_p.add_argument("--page-idx", type=int, default=0)

    args = parser.parse_args()
    
    if args.command == "navigate":
        handle_navigate(args)
    elif args.command == "extract":
        handle_extract(args)
    elif args.command == "interact":
        handle_interact(args)

if __name__ == "__main__":
    if mcp and len(sys.argv) == 1:
        mcp.run()
    else:
        main()
