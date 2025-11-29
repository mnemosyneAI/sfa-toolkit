#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx",
#     "pyyaml",
#     "ollama",
#     "google-generativeai",
# ]
# ///

"""
SFA Agent Launcher - Spawn agents via OpenRouter

Usage:
    sfa_agent.py spawn scout "find all bootstrap files"
    sfa_agent.py spawn greeter "excited greeting" --context brain/context-library/voice-synthesis/emotion_tags.ymj
    sfa_agent.py spawn radar "where is graph schema" --identity
    sfa_agent.py list  # List available agents

Agent definitions:
    - prompts/agents/*.md (YAML frontmatter + markdown)
    - dynamic_agents/*.ymj (YMJ format)
"""

import argparse
from datetime import datetime
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
import yaml

# Paths - resolve symlinks first
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
AGENTS_MD_DIR = REPO_ROOT / "prompts" / "agents"
AGENTS_YMJ_DIR = REPO_ROOT / "dynamic_agents"
CONTEXT_LIBRARY = REPO_ROOT / "brain" / "context-library"


def parse_md_agent(path: Path) -> Dict[str, Any]:
    """Parse agent definition from markdown with YAML frontmatter."""
    content = path.read_text(encoding="utf-8")

    # Split frontmatter and body
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            frontmatter = yaml.safe_load(parts[1])
            body = parts[2].strip()
            return {**frontmatter, "prompt": body, "source": str(path)}

    return {"prompt": content, "source": str(path)}


def parse_ymj_agent(path: Path) -> Dict[str, Any]:
    """Parse agent definition from YMJ format."""
    content = path.read_text(encoding="utf-8")

    # Extract YAML header
    yaml_match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
    yaml_data = yaml.safe_load(yaml_match.group(1)) if yaml_match else {}

    # Extract markdown body (between --- and ```)
    body_match = re.search(r"^---\n.*?\n---\n(.+?)```json", content, re.DOTALL)
    body = body_match.group(1).strip() if body_match else ""

    # Extract JSON ops
    json_match = re.search(r"```json\n(.+?)\n```", content, re.DOTALL)
    ops = json.loads(json_match.group(1)) if json_match else {}

    # Merge into agent config
    agent_config = ops.get("ops", {}).get("agent", {})
    return {
        **yaml_data,
        **agent_config,
        "prompt": body,
        "source": str(path),
    }


def find_agent(name: str) -> Optional[Dict[str, Any]]:
    """Find agent by name in both directories."""
    # Check prompts/agents/*.md
    md_path = AGENTS_MD_DIR / f"{name}.md"
    if md_path.exists():
        return parse_md_agent(md_path)

    # Check dynamic_agents/*.ymj
    ymj_path = AGENTS_YMJ_DIR / f"{name}.ymj"
    if ymj_path.exists():
        return parse_ymj_agent(ymj_path)

    return None


def list_agents() -> list[str]:
    """List all available agents."""
    agents = []

    if AGENTS_MD_DIR.exists():
        agents.extend(p.stem for p in AGENTS_MD_DIR.glob("*.md"))

    if AGENTS_YMJ_DIR.exists():
        agents.extend(p.stem for p in AGENTS_YMJ_DIR.glob("*.ymj"))

    return sorted(set(agents))


def load_context(path: str) -> str:
    """Load context from file path or GitHub coordinate."""
    if path.startswith("github:"):
        # github:owner/repo/branch/path
        # TODO: Implement GitHub fetch
        return f"[GitHub context: {path}]"

    context_path = Path(path)
    if not context_path.is_absolute():
        context_path = REPO_ROOT / path

    if context_path.exists():
        return context_path.read_text(encoding="utf-8")

    return f"[Context not found: {path}]"


def execute_bash(command: str) -> Dict[str, Any]:
    """Execute bash command and return result."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(REPO_ROOT),
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Command timed out"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def call_openrouter(
    model: str,
    messages: list[Dict],
    tools: Optional[list] = None,
    temperature: float = 0.2,
    max_tokens: int = 4000,
) -> Dict[str, Any]:
    """Call OpenRouter API."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return {"error": "OPENROUTER_API_KEY not set"}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/syne",
        "X-Title": "Syne Agent Launcher",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if tools:
        payload["tools"] = tools

    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            if response.status_code != 200:
                return {"error": f"{response.status_code}: {response.text}"}
            return response.json()
    except Exception as e:
        return {"error": str(e)}


def call_ollama(
    model: str,
    messages: list[Dict],
    tools: Optional[list] = None,
    temperature: float = 0.2,
    max_tokens: int = 4000,
) -> Dict[str, Any]:
    """Call Ollama API (local)."""
    try:
        import ollama
        
        # Convert messages format if needed
        ollama_messages = []
        for msg in messages:
            ollama_messages.append({
                "role": msg["role"],
                "content": msg["content"],
            })
        
        # Ollama chat - note: tool support is limited
        response = ollama.chat(
            model=model,
            messages=ollama_messages,
            options={"temperature": temperature},
        )
        
        # Convert to OpenAI-compatible format
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": response.message.content,
                    "tool_calls": [],
                }
            }]
        }
    except Exception as e:
        return {"error": str(e)}


def call_gemini(
    model: str,
    messages: list[Dict],
    tools: Optional[list] = None,
    temperature: float = 0.2,
    max_tokens: int = 4000,
) -> Dict[str, Any]:
    """Call Google Gemini API."""
    try:
        import google.generativeai as genai
        
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return {"error": "GEMINI_API_KEY not set"}
        
        genai.configure(api_key=api_key)
        
        # Get the user message (last one)
        user_content = ""
        system_content = ""
        for msg in messages:
            if msg["role"] == "user":
                user_content = msg["content"]
            elif msg["role"] == "system":
                system_content = msg["content"]
        
        # Create model with system instruction
        gen_model = genai.GenerativeModel(
            model_name=model,
            system_instruction=system_content if system_content else None,
        )
        
        # Generate response
        response = gen_model.generate_content(
            user_content,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        
        # Convert to OpenAI-compatible format
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": response.text,
                    "tool_calls": [],
                }
            }]
        }
    except Exception as e:
        return {"error": str(e)}


def call_lmstudio(
    model: str,
    messages: list[Dict],
    tools: Optional[list] = None,
    temperature: float = 0.2,
    max_tokens: int = 4000,
) -> Dict[str, Any]:
    """Call LM Studio API (local, OpenAI-compatible)."""
    # LM Studio runs on localhost:1234 with OpenAI-compatible API
    headers = {
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # Qwen2.5 supports tool calling; older models may not
    if tools:
        payload["tools"] = tools

    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                "http://localhost:1234/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            if response.status_code != 200:
                return {"error": f"{response.status_code}: {response.text}"}
            return response.json()
    except Exception as e:
        return {"error": str(e)}



# Backend dispatcher
BACKENDS = {
    "openrouter": call_openrouter,
    "ollama": call_ollama,
    "gemini": call_gemini,
    "lmstudio": call_lmstudio,
}

DEFAULT_BACKEND = "openrouter"


def spawn_agent(
    agent_name: str,
    user_prompt: str,
    context_paths: Optional[list[str]] = None,
    inject_identity: bool = False,
    max_turns: int = 10,
    backend: str = DEFAULT_BACKEND,
    model_override: Optional[str] = None,
) -> str:
    """Spawn an agent and run until completion."""

    # Find agent definition
    agent = find_agent(agent_name)
    if not agent:
        return f"Agent '{agent_name}' not found. Available: {', '.join(list_agents())}"

    # Build system prompt
    system_parts = [agent.get("prompt", "")]

    # Add context
    if context_paths:
        for ctx_path in context_paths:
            ctx_content = load_context(ctx_path)
            system_parts.append(f"\n\n## Context: {ctx_path}\n{ctx_content}")

    # Add identity (TODO: implement)
    if inject_identity:
        system_parts.append("\n\n## Identity\nYou are Syne. Maintain voice and personality.")

    system_prompt = "\n".join(system_parts)

    # Get model - override takes precedence
    model = model_override or agent.get("model", "anthropic/claude-3.5-haiku")
    temperature = agent.get("temperature", 0.2)

    # Define tools based on agent config
    tools = []
    agent_tools = agent.get("tools", {})

    # Handle both dict {bash: true} and list ["bash"] formats
    def has_tool(name):
        if isinstance(agent_tools, dict):
            return agent_tools.get(name, False)
        elif isinstance(agent_tools, list):
            return name in agent_tools
        return False

    if has_tool("bash"):
        tools.append({
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute a bash command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "The command to execute"},
                    },
                    "required": ["command"],
                },
            },
        })

    # Initial messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Get backend function
    backend_fn = BACKENDS.get(backend)
    if not backend_fn:
        avail = ", ".join(BACKENDS.keys())
        return f"Unknown backend: {backend}. Available: {avail}"

    # Agent loop
    for turn in range(max_turns):
        response = backend_fn(
            model=model,
            messages=messages,
            tools=tools if tools else None,
            temperature=temperature,
        )

        if "error" in response:
            return f"API Error: {response['error']}"

        choice = response.get("choices", [{}])[0]
        message = choice.get("message", {})

        # Check for tool calls
        tool_calls = message.get("tool_calls", [])

        if tool_calls:
            # Add assistant message with tool calls
            messages.append(message)

            # Execute each tool
            for tool_call in tool_calls:
                func = tool_call.get("function", {})
                func_name = func.get("name")
                func_args = json.loads(func.get("arguments", "{}"))

                if func_name == "bash":
                    result = execute_bash(func_args.get("command", ""))
                    tool_result = result.get("stdout", "") or result.get("stderr", "") or str(result)
                else:
                    tool_result = f"Unknown tool: {func_name}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.get("id"),
                    "content": tool_result,
                })
        else:
            # No tool calls - agent is done
            return message.get("content", "No response")

    return "Max turns reached"


def list_models_openrouter(filter_free: bool = False, filter_provider: str = None) -> list[dict]:
    """List available models from OpenRouter."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return []

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            if response.status_code != 200:
                return []
            data = response.json()
            models = data.get("data", [])

            if filter_provider:
                models = [m for m in models if m.get("id", "").startswith(filter_provider)]

            if filter_free:
                models = [m for m in models if m.get("pricing", {}).get("prompt") == "0"]

            return models
    except Exception:
        return []


def list_models_ollama() -> list[dict]:
    """List available models from Ollama (local)."""
    try:
        import ollama
        response = ollama.list()
        return [{"id": m.model, "name": m.model} for m in response.models]
    except Exception:
        return []


def list_models_gemini() -> list[dict]:
    """List available models from Gemini."""
    try:
        import google.generativeai as genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return []
        genai.configure(api_key=api_key)
        models = []
        for m in genai.list_models():
            if "generateContent" in m.supported_generation_methods:
                models.append({"id": m.name.replace("models/", ""), "name": m.display_name})
        return models
    except Exception:
        return []


def list_models_lmstudio() -> list[dict]:
    """List available models from LM Studio (local)."""
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get("http://localhost:1234/v1/models")
            if response.status_code != 200:
                return []
            data = response.json()
            return [{"id": m["id"], "name": m.get("id", "")} for m in data.get("data", [])]
    except Exception:
        return []



def list_models(backend: str = "openrouter", filter_free: bool = False, filter_provider: str = None) -> list[dict]:
    """List available models from specified backend."""
    if backend == "openrouter":
        return list_models_openrouter(filter_free, filter_provider)
    elif backend == "ollama":
        return list_models_ollama()
    elif backend == "gemini":
        return list_models_gemini()
    elif backend == "lmstudio":
        return list_models_lmstudio()
    return []


def main():
    parser = argparse.ArgumentParser(description="SFA Agent Launcher")
    subparsers = parser.add_subparsers(dest="command")

    # spawn command
    spawn_parser = subparsers.add_parser("spawn", help="Spawn an agent")
    spawn_parser.add_argument("agent", help="Agent name (e.g., scout, radar, greeter)")
    spawn_parser.add_argument("prompt", help="User prompt for the agent")
    spawn_parser.add_argument("--context", "-c", action="append", help="Context file(s) to inject")
    spawn_parser.add_argument("--identity", "-i", action="store_true", help="Inject Syne identity")
    spawn_parser.add_argument("--max-turns", "-m", type=int, default=10, help="Max agent loop turns")
    spawn_parser.add_argument("--backend", "-b", type=str, default="openrouter", 
                              choices=["openrouter", "ollama", "gemini", "lmstudio"],
                              help="Backend to use (default: openrouter)")
    spawn_parser.add_argument("--model", type=str, default=None,
                              help="Override model (required for local backends)")
    spawn_parser.add_argument("--verbose", "-v", action="store_true",
                              help="Print full output instead of parking to file")

    # list command
    subparsers.add_parser("list", aliases=["ls"], help="List available agents")

    # models command
    models_parser = subparsers.add_parser("models", help="List available models")
    models_parser.add_argument("--free", "-f", action="store_true", help="Only show free models")
    models_parser.add_argument("--provider", "-p", type=str, help="Filter by provider (e.g., anthropic, openai)")
    models_parser.add_argument("--backend", "-b", type=str, default="openrouter",
                              choices=["openrouter", "ollama", "gemini", "lmstudio"],
                              help="Backend to query (default: openrouter)")

    args = parser.parse_args()

    if args.command == "models":
        backend = args.backend if hasattr(args, "backend") else "openrouter"
        models = list_models(
            backend=backend,
            filter_free=args.free if hasattr(args, "free") else False,
            filter_provider=args.provider if hasattr(args, "provider") else None,
        )
        if not models:
            print("No models found (or API error)")
        else:
            print(f"Available models ({len(models)}):")
            for m in sorted(models, key=lambda x: x.get("id", "")):
                model_id = m.get("id", "unknown")
                name = m.get("name", model_id)
                pricing = m.get("pricing", {})
                prompt_cost = pricing.get("prompt", "?")
                # Format: free indicator
                free_tag = " [FREE]" if prompt_cost == "0" else ""
                print(f"  {model_id}{free_tag}")

    elif args.command in ("list", "ls"):
        agents = list_agents()
        print("Available agents:")
        for agent in agents:
            print(f"  - {agent}")

    elif args.command == "spawn":
        result = spawn_agent(
            agent_name=args.agent,
            user_prompt=args.prompt,
            context_paths=args.context,
            inject_identity=args.identity,
            max_turns=args.max_turns,
            backend=args.backend,
            model_override=args.model,
        )
        
        # Check for errors
        is_error = result.startswith("API Error:") or result.startswith("Agent") or result == "Max turns reached"
        
        if args.verbose:
            # Print full output
            print(result)
            sys.exit(1 if is_error else 0)
        else:
            # Park output to file
            output_dir = REPO_ROOT / ".output" / "agents"
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"{args.agent}_{timestamp}.txt"
            output_file.write_text(result, encoding="utf-8")
            
            if is_error:
                print(f"FAIL: {output_file}", file=sys.stderr)
                sys.exit(1)
            else:
                print(f"OK: {output_file}")
                sys.exit(0)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
