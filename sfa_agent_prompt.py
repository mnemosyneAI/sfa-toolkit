#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx", "fastmcp"]
# ///
"""
SFA Agent Prompt (sfa_agent_prompt.py) v1.3.0
Unified agent interface - ephemeral prompts, forked sessions, API calls.
MCP server for tool integration.

ROUTE FORMAT: mode:interface:provider:model
  mode:      e (ephemeral) | p (persistent/fork)
  interface: ide | api
  provider:  gemini, claude, anthropic, openai, xai, google, openrouter, ollama, lms
  model:     model name or alias

COMMANDS:
  <route> "query"              - Execute prompt via route
  send <pane_id> "text"        - Send to persistent session
  read <pane_id> [--lines N]   - Read from persistent session
  list                         - List available routes
  mcp                          - Run as MCP server

EXAMPLES:
  sfa_agent_prompt.py "e:ide:gemini:gemini-2.5-pro" "What is 2+2?"
  sfa_agent_prompt.py "p:ide:claude:sonnet" "Analyze this codebase"
  sfa_agent_prompt.py "e:api:anthropic:claude-sonnet-4-20250514" "Hello"
  sfa_agent_prompt.py "e:api:openrouter:anthropic/claude-opus-4" "Hello"
  sfa_agent_prompt.py "e:api:ollama:llama3" "Hello"
  sfa_agent_prompt.py send 15 "Follow-up question"
  sfa_agent_prompt.py mcp

NOTES:
  - ide routes use CLI tools (free tier, user auth)
  - api routes need keys in --secrets file (except local: ollama, lms)
  - persistent sessions spawn to wezterm pane
  - Set SFA_SECRETS_PATH env var or use --secrets flag for API keys
"""

import subprocess
import sys
import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from fastmcp import FastMCP
    mcp = FastMCP("sfa-agent-prompt")
except ImportError:
    mcp = None


# === SECRETS ===
SECRETS: dict[str, str] = {}
DEFAULT_SECRETS_PATH = os.environ.get("SFA_SECRETS_PATH", str(Path.home() / ".secrets"))

# Local LLM defaults
OLLAMA_HOST = "http://localhost:11434"
LMSTUDIO_HOST = "http://localhost:1234"


def load_secrets(path: str) -> dict[str, str]:
    """Load secrets from KEY=value file."""
    secrets = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    secrets[key.strip()] = value.strip()
    except FileNotFoundError:
        pass
    return secrets


def get_secret(key: str) -> Optional[str]:
    """Get secret from loaded secrets or environment."""
    return SECRETS.get(key) or os.environ.get(key)


# === ROUTE PARSING ===
@dataclass
class Route:
    mode: str        # e=ephemeral, p=persistent
    interface: str   # ide, api
    provider: str    # gemini, claude, anthropic, openai, xai, google, openrouter, ollama, lms
    model: str       # model name or alias
    
    @classmethod
    def parse(cls, route_str: str) -> "Route":
        parts = route_str.split(":", maxsplit=3)
        if len(parts) != 4:
            raise ValueError(f"Invalid route format: {route_str}. Expected mode:interface:provider:model")
        return cls(mode=parts[0], interface=parts[1], provider=parts[2], model=parts[3])
    
    def __str__(self) -> str:
        return f"{self.mode}:{self.interface}:{self.provider}:{self.model}"


# === IDE HANDLERS ===

def ide_gemini_ephemeral(query: str, model: str) -> tuple[bool, str]:
    """Ephemeral prompt via gemini -p."""
    cmd = ["gemini", "-p", query]
    if model:
        cmd.extend(["--model", model])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = '\n'.join(line for line in result.stdout.split('\n')
                       if not line.startswith('[WARN]') 
                       and not line.startswith('Warning:')
                       and not line.startswith('Loaded cached'))
    return (True, output.strip()) if result.returncode == 0 else (False, result.stderr.strip())


def ide_gemini_fork(task: str, model: str, pane_id: Optional[int] = None, 
                    cwd: Optional[str] = None) -> tuple[bool, str]:
    """Fork gemini to terminal with initial task."""
    cwd = cwd or os.getcwd()
    gemini_cmd = f'gemini --model {model} -y -i "{task}"'
    
    args = ["wezterm", "cli", "spawn", "--cwd", cwd]
    if pane_id is not None:
        args.extend(["--pane-id", str(pane_id)])
    args.extend(["--", "bash", "-c", gemini_cmd])
    
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode == 0:
        new_pane = result.stdout.strip()
        return True, f"Forked to pane {new_pane}"
    return False, result.stderr.strip()


def ide_claude_ephemeral(query: str, model: str) -> tuple[bool, str]:
    """Ephemeral prompt via claude -p."""
    cmd = ["claude", "-p", "--model", model, query]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return (True, result.stdout.strip()) if result.returncode == 0 else (False, result.stderr.strip())


def ide_claude_fork(task: str, model: str, pane_id: Optional[int] = None,
                    cwd: Optional[str] = None) -> tuple[bool, str]:
    """Fork claude to terminal with initial task."""
    cwd = cwd or os.getcwd()
    claude_cmd = f'claude --model {model} "{task}"'
    
    args = ["wezterm", "cli", "spawn", "--cwd", cwd]
    if pane_id is not None:
        args.extend(["--pane-id", str(pane_id)])
    args.extend(["--", "bash", "-c", claude_cmd])
    
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode == 0:
        new_pane = result.stdout.strip()
        return True, f"Forked to pane {new_pane}"
    return False, result.stderr.strip()


# === API HANDLERS ===

def api_anthropic(query: str, model: str) -> tuple[bool, str]:
    """API call to Anthropic."""
    import httpx
    
    api_key = get_secret("ANTHROPIC_API_KEY")
    if not api_key:
        return False, "No ANTHROPIC_API_KEY. Use e:ide:claude:<model> instead (free with Claude subscription)"
    
    try:
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": query}]
            },
            timeout=120.0
        )
        response.raise_for_status()
        data = response.json()
        return True, data["content"][0]["text"]
    except Exception as e:
        return False, str(e)


def api_openai(query: str, model: str) -> tuple[bool, str]:
    """API call to OpenAI."""
    import httpx
    
    api_key = get_secret("OPENAI_API_KEY")
    if not api_key:
        return False, "OPENAI_API_KEY not found in secrets"
    
    try:
        response = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": query}]
            },
            timeout=120.0
        )
        response.raise_for_status()
        data = response.json()
        return True, data["choices"][0]["message"]["content"]
    except Exception as e:
        return False, str(e)


def api_xai(query: str, model: str) -> tuple[bool, str]:
    """API call to xAI (Grok)."""
    import httpx
    
    api_key = get_secret("XAI_API_KEY")
    if not api_key:
        return False, "XAI_API_KEY not found in secrets"
    
    try:
        response = httpx.post(
            "https://api.x.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": query}]
            },
            timeout=120.0
        )
        response.raise_for_status()
        data = response.json()
        return True, data["choices"][0]["message"]["content"]
    except Exception as e:
        return False, str(e)


def api_google(query: str, model: str) -> tuple[bool, str]:
    """API call to Google AI."""
    import httpx
    
    api_key = get_secret("GEMINI_API_KEY") or get_secret("GOOGLE_API_KEY")
    if not api_key:
        return False, "GEMINI_API_KEY or GOOGLE_API_KEY not found in secrets"
    
    try:
        response = httpx.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            headers={"Content-Type": "application/json"},
            params={"key": api_key},
            json={
                "contents": [{"parts": [{"text": query}]}]
            },
            timeout=120.0
        )
        response.raise_for_status()
        data = response.json()
        return True, data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return False, str(e)


def api_openrouter(query: str, model: str) -> tuple[bool, str]:
    """API call to OpenRouter (OpenAI-compatible, multi-provider)."""
    import httpx
    
    api_key = get_secret("OPENROUTER_API_KEY")
    if not api_key:
        return False, "OPENROUTER_API_KEY not found in secrets"
    
    try:
        response = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": query}]
            },
            timeout=120.0
        )
        response.raise_for_status()
        data = response.json()
        return True, data["choices"][0]["message"]["content"]
    except Exception as e:
        return False, str(e)


def api_ollama(query: str, model: str) -> tuple[bool, str]:
    """API call to local Ollama."""
    import httpx
    
    host = get_secret("OLLAMA_HOST") or OLLAMA_HOST
    
    try:
        response = httpx.post(
            f"{host}/api/generate",
            json={
                "model": model,
                "prompt": query,
                "stream": False
            },
            timeout=120.0
        )
        response.raise_for_status()
        data = response.json()
        return True, data.get("response", "")
    except Exception as e:
        return False, str(e)


def api_lmstudio(query: str, model: str) -> tuple[bool, str]:
    """API call to local LM Studio (OpenAI-compatible)."""
    import httpx
    
    host = get_secret("LMSTUDIO_HOST") or LMSTUDIO_HOST
    
    try:
        response = httpx.post(
            f"{host}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": query}]
            },
            timeout=120.0
        )
        response.raise_for_status()
        data = response.json()
        return True, data["choices"][0]["message"]["content"]
    except Exception as e:
        return False, str(e)


# === SESSION MANAGEMENT ===

def send_to_session(pane_id: int, text: str, provider: str = "gemini") -> tuple[bool, str]:
    """Send text to running session."""
    # Clear input first
    subprocess.run(f'printf "\\x03" | wezterm cli send-text --pane-id {pane_id} --no-paste',
                   shell=True, capture_output=True)
    
    # Send text
    cmd = ["wezterm", "cli", "send-text", "--pane-id", str(pane_id), "--no-paste", text]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return False, "Failed to send text"
    
    # Submit - different for different providers
    if provider == "claude":
        # Claude needs Ctrl+M
        subprocess.run(f'printf "\\x0d" | wezterm cli send-text --pane-id {pane_id} --no-paste',
                       shell=True, capture_output=True)
    else:
        # Gemini needs double CR
        for _ in range(2):
            subprocess.run(f'echo -e "\\r" | wezterm cli send-text --pane-id {pane_id} --no-paste',
                           shell=True, capture_output=True)
    
    return True, "Sent"


def read_from_session(pane_id: int, lines: int = 50) -> tuple[bool, str]:
    """Read output from session."""
    cmd = ["wezterm", "cli", "get-text", "--pane-id", str(pane_id), "--start-line", str(-lines)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return (True, result.stdout) if result.returncode == 0 else (False, result.stderr.strip())


# === ROUTER ===

def execute_route(route: Route, query: str, pane_id: Optional[int] = None, 
                  cwd: Optional[str] = None) -> tuple[bool, str]:
    """Execute query via route."""
    
    # IDE routes
    if route.interface == "ide":
        if route.provider == "gemini":
            if route.mode == "e":
                return ide_gemini_ephemeral(query, route.model)
            elif route.mode == "p":
                return ide_gemini_fork(query, route.model, pane_id, cwd)
        
        elif route.provider == "claude":
            if route.mode == "e":
                return ide_claude_ephemeral(query, route.model)
            elif route.mode == "p":
                return ide_claude_fork(query, route.model, pane_id, cwd)
        
        else:
            return False, f"Unknown IDE provider: {route.provider}"
    
    # API routes
    elif route.interface == "api":
        if route.mode == "p":
            return False, "Persistent mode not supported for API routes"
        
        if route.provider == "anthropic":
            return api_anthropic(query, route.model)
        elif route.provider == "openai":
            return api_openai(query, route.model)
        elif route.provider == "xai":
            return api_xai(query, route.model)
        elif route.provider == "google":
            return api_google(query, route.model)
        elif route.provider in ("openrouter", "or"):
            return api_openrouter(query, route.model)
        elif route.provider == "ollama":
            return api_ollama(query, route.model)
        elif route.provider == "lms":
            return api_lmstudio(query, route.model)
        else:
            return False, f"Unknown API provider: {route.provider}"
    
    else:
        return False, f"Unknown interface: {route.interface}"


def list_routes() -> str:
    """List available routes."""
    return """Available routes:

IDE (free tier, user auth):
  e:ide:gemini:<model>       Ephemeral gemini -p
  p:ide:gemini:<model>       Fork gemini session
  e:ide:claude:<model>       Ephemeral claude -p
  p:ide:claude:<model>       Fork claude session

API - Cloud (needs keys in --secrets):
  e:api:anthropic:<model>    Anthropic API
  e:api:openai:<model>       OpenAI API
  e:api:xai:<model>          xAI (Grok) API
  e:api:google:<model>       Google AI API
  e:api:openrouter:<model>   OpenRouter (multi-provider)
  e:api:or:<model>           OpenRouter (alias)

API - Local (no keys needed):
  e:api:ollama:<model>       Ollama (localhost:11434)
  e:api:lms:<model>          LM Studio (localhost:1234)

Model examples:
  gemini-2.5-pro, gemini-2.5-flash
  sonnet, opus, haiku, claude-sonnet-4-20250514
  gpt-4o, gpt-4o-mini
  grok-2, grok-3
  anthropic/claude-opus-4, openai/gpt-4o (openrouter)
  llama3, qwen2.5-coder, mistral"""


# === MCP TOOLS ===

if mcp:
    @mcp.tool()
    def agent_prompt(route: str, query: str, pane_id: Optional[int] = None, cwd: Optional[str] = None) -> str:
        """
        Execute a prompt via unified routing.
        
        Route format: mode:interface:provider:model
        - mode: e (ephemeral) or p (persistent/fork)
        - interface: ide or api
        - provider: gemini, claude, anthropic, openai, xai, google, openrouter, ollama, lms
        - model: model name
        
        Examples:
        - e:ide:gemini:gemini-2.5-pro (ephemeral gemini CLI)
        - p:ide:claude:sonnet (fork claude to terminal)
        - e:api:xai:grok-2 (xAI API call)
        - e:api:openrouter:anthropic/claude-opus-4 (OpenRouter)
        - e:api:ollama:llama3 (local Ollama)
        """
        global SECRETS
        SECRETS = load_secrets(DEFAULT_SECRETS_PATH)
        
        try:
            r = Route.parse(route)
            success, output = execute_route(r, query, pane_id, cwd)
            return output if success else f"Error: {output}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    @mcp.tool()
    def agent_send(pane_id: int, text: str, provider: str = "gemini") -> str:
        """
        Send text to a running agent session.
        
        Args:
            pane_id: Wezterm pane ID
            text: Text to send
            provider: gemini or claude (affects submit key)
        """
        success, output = send_to_session(pane_id, text, provider)
        return output if success else f"Error: {output}"
    
    @mcp.tool()
    def agent_read(pane_id: int, lines: int = 50) -> str:
        """
        Read output from a running agent session.
        
        Args:
            pane_id: Wezterm pane ID
            lines: Number of lines to read
        """
        success, output = read_from_session(pane_id, lines)
        return output if success else f"Error: {output}"
    
    @mcp.tool()
    def agent_list_routes() -> str:
        """List available agent routes."""
        return list_routes()


# === MAIN ===

def main():
    global SECRETS
    
    parser = argparse.ArgumentParser(description="Unified agent prompt interface")
    parser.add_argument("--secrets", default=DEFAULT_SECRETS_PATH, 
                        help="Path to secrets file (KEY=value format)")
    parser.add_argument("--pane-id", type=int, help="Target pane for fork/send")
    parser.add_argument("--cwd", help="Working directory for fork (default: current)")
    parser.add_argument("--provider", default="gemini", help="Provider for send command")
    parser.add_argument("--lines", "-n", type=int, default=50, help="Lines to read")
    parser.add_argument("args", nargs="*", help="Command and arguments")
    
    args = parser.parse_args()
    
    # Load secrets
    SECRETS = load_secrets(args.secrets)
    
    if not args.args:
        print(list_routes())
        sys.exit(0)
    
    cmd = args.args[0]
    
    # MCP server mode
    if cmd == "mcp":
        if mcp:
            mcp.run()
        else:
            print("Error: fastmcp not installed", file=sys.stderr)
            sys.exit(1)
        return
    
    # Send command
    if cmd == "send":
        if len(args.args) < 3:
            print("Usage: send <pane_id> \"text\"", file=sys.stderr)
            sys.exit(1)
        pane_id = int(args.args[1])
        text = args.args[2]
        success, output = send_to_session(pane_id, text, args.provider)
        print(output) if success else print(f"Error: {output}", file=sys.stderr)
        sys.exit(0 if success else 1)
    
    # Read command
    if cmd == "read":
        if len(args.args) < 2:
            print("Usage: read <pane_id>", file=sys.stderr)
            sys.exit(1)
        pane_id = int(args.args[1])
        success, output = read_from_session(pane_id, args.lines)
        print(output) if success else print(f"Error: {output}", file=sys.stderr)
        sys.exit(0 if success else 1)
    
    # List command
    if cmd == "list":
        print(list_routes())
        sys.exit(0)
    
    # Route execution
    if ":" in cmd:
        try:
            route = Route.parse(cmd)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        
        if len(args.args) < 2:
            print("Usage: <route> \"query\"", file=sys.stderr)
            sys.exit(1)
        
        query = args.args[1]
        success, output = execute_route(route, query, args.pane_id, args.cwd)
        print(output) if success else print(f"Error: {output}", file=sys.stderr)
        sys.exit(0 if success else 1)
    
    # Unknown command
    print(f"Unknown command: {cmd}", file=sys.stderr)
    print(list_routes())
    sys.exit(1)


if __name__ == "__main__":
    main()
