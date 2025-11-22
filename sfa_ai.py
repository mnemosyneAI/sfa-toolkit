# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "fastmcp",
#     "httpx",
#     "fastembed",
#     "numpy",
# ]
# ///

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import httpx
import numpy as np

try:
    from fastmcp import FastMCP
    mcp = FastMCP("sfa-ai")
except ImportError:
    mcp = None

# --- Configuration ---

PROVIDERS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "env_key": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini"
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
        "default_model": "openai/gpt-4o-mini" # Good cheap default
    },
    "local": {
        "base_url": "http://localhost:1234/v1", # Common default (LM Studio)
        "env_key": "LOCAL_AI_KEY", # Often not needed, but good to have
        "default_model": "local-model"
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/models",
        "env_key": "GOOGLE_API_KEY",
        "default_model": "gemini-1.5-flash"
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1",
        "env_key": "ANTHROPIC_API_KEY",
        "default_model": "claude-3-5-sonnet-20241022"
    }
}

# --- RAG Logic ---

def _cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def _get_relevant_context(query: str, context_dir: str, top_k: int = 3) -> str:
    """Scan .ymj files in directory and return relevant context using embeddings."""
    try:
        from fastembed import TextEmbedding
    except ImportError:
        return "Error: fastembed not installed. Cannot perform semantic search."

    root = Path(context_dir)
    if not root.exists():
        return f"Error: Context directory {context_dir} not found."

    # 1. Embed Query
    try:
        model = TextEmbedding(model_name="nomic-ai/nomic-embed-text-v1.5")
        query_embedding = list(model.embed([query]))[0]
    except Exception as e:
        return f"Error generating query embedding: {e}"

    # 2. Scan Files
    candidates = []
    
    for fpath in root.rglob("*.ymj"):
        try:
            content = fpath.read_text(encoding="utf-8")
            
            # Extract Footer (Simplified parsing)
            footer_end = content.rfind("```")
            if footer_end == -1: continue
            
            footer_start = content.rfind("```json", 0, footer_end)
            if footer_start == -1: continue
            
            footer_json = content[footer_start+7:footer_end].strip()
            footer = json.loads(footer_json)
            
            # Check for embedding
            if "index" in footer and "embedding" in footer["index"]:
                vec = footer["index"]["embedding"]
                score = _cosine_similarity(query_embedding, vec)
                
                # Extract Body (Simplified)
                # Assuming standard format: YAML --- Body --- JSON
                # We just take everything before the footer start
                body = content[:footer_start].strip()
                # Remove header if present
                if body.startswith("---"):
                    try:
                        header_end = body.index("\n---", 3)
                        body = body[header_end+4:].strip()
                    except:
                        pass
                
                candidates.append((score, fpath.name, body))
                
        except Exception:
            continue # Skip malformed files

    # 3. Rank and Return
    candidates.sort(key=lambda x: x[0], reverse=True)
    top_results = candidates[:top_k]
    
    if not top_results:
        return "No relevant context found in .ymj files."

    context_str = f"\n--- Relevant Context (from {context_dir}) ---\n"
    for score, name, body in top_results:
        context_str += f"File: {name} (Relevance: {score:.2f})\n```\n{body[:2000]}...\n```\n" # Truncate body to avoid context overflow
        
    return context_str

# --- Core Logic ---

def _get_api_key(provider: str, key_arg: Optional[str] = None) -> str:
    """Get API key from arg or environment, or fail loudly."""
    config = PROVIDERS.get(provider)
    if not config:
        raise ValueError(f"Unknown provider: {provider}")
    
    # Local might not need a key
    if provider == "local":
        return key_arg or os.environ.get(config["env_key"], "lm-studio")

    key = key_arg or os.environ.get(config["env_key"])
    if not key:
        raise ValueError(
            f"Missing API Key for {provider}.\n"
            f"Please set {config['env_key']} environment variable or pass --key.\n"
        )
    return key

def _query_openai_compatible(
    provider: str, 
    model: str, 
    messages: List[Dict[str, str]], 
    api_key: str,
    temperature: float = 0.7
) -> str:
    """Generic handler for OpenAI, OpenRouter, and Local AI."""
    config = PROVIDERS[provider]
    # Allow override of base_url via env var for Local
    base_url = os.environ.get("LOCAL_AI_BASE_URL", config["base_url"]) if provider == "local" else config["base_url"]
    
    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # OpenRouter specific headers
    if provider == "openrouter":
        headers["HTTP-Referer"] = "https://github.com/sfa-toolkit"
        headers["X-Title"] = "SFA Toolkit"

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature
    }

    try:
        resp = httpx.post(url, json=payload, headers=headers, timeout=60.0)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"API Error ({e.response.status_code}): {e.response.text}")
    except Exception as e:
        raise RuntimeError(f"Request failed: {str(e)}")

def _query_anthropic(
    model: str, 
    messages: List[Dict[str, str]], 
    api_key: str,
    temperature: float = 0.7
) -> str:
    """Handler for Anthropic API."""
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    # Convert standard messages to Anthropic format if needed
    # (Simple pass-through for now, assuming user/assistant roles)
    system_prompt = None
    filtered_messages = []
    for m in messages:
        if m["role"] == "system":
            system_prompt = m["content"]
        else:
            filtered_messages.append(m)

    payload = {
        "model": model,
        "messages": filtered_messages,
        "max_tokens": 4096,
        "temperature": temperature
    }
    if system_prompt:
        payload["system"] = system_prompt

    try:
        resp = httpx.post(url, json=payload, headers=headers, timeout=60.0)
        resp.raise_for_status()
        data = resp.json()
        return data["content"][0]["text"]
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"API Error ({e.response.status_code}): {e.response.text}")
    except Exception as e:
        raise RuntimeError(f"Request failed: {str(e)}")

def _query_gemini(
    model: str, 
    messages: List[Dict[str, str]], 
    api_key: str,
    temperature: float = 0.7
) -> str:
    """Handler for Google Gemini API."""
    # Gemini REST API is a bit different.
    # URL: https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    
    # Convert messages to Gemini format
    contents = []
    system_instruction = None
    
    for m in messages:
        role = "user" if m["role"] == "user" else "model"
        if m["role"] == "system":
            system_instruction = {"parts": [{"text": m["content"]}]}
            continue
            
        contents.append({
            "role": role,
            "parts": [{"text": m["content"]}]
        })

    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature
        }
    }
    if system_instruction:
        payload["systemInstruction"] = system_instruction

    try:
        resp = httpx.post(url, json=payload, params={"key": api_key}, timeout=60.0)
        resp.raise_for_status()
        data = resp.json()
        
        # Handle safety blocks or empty responses
        if "candidates" not in data or not data["candidates"]:
            if "promptFeedback" in data:
                raise RuntimeError(f"Gemini blocked response: {data['promptFeedback']}")
            raise RuntimeError("Gemini returned no candidates.")
            
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"API Error ({e.response.status_code}): {e.response.text}")
    except Exception as e:
        raise RuntimeError(f"Request failed: {str(e)}")

def ask_ai(
    prompt: str, 
    provider: str = "gemini", 
    model: Optional[str] = None, 
    system: Optional[str] = None,
    context_files: Optional[List[str]] = None,
    context_dir: Optional[str] = None,
    api_key: Optional[str] = None
) -> str:
    """Unified entry point for querying AI."""
    
    # 1. Resolve Configuration
    if provider not in PROVIDERS:
        raise ValueError(f"Provider '{provider}' not supported. Options: {list(PROVIDERS.keys())}")
    
    resolved_model = model or PROVIDERS[provider]["default_model"]
    key = _get_api_key(provider, api_key)
    
    # 2. Build Context
    full_prompt = prompt
    
    # Explicit files
    if context_files:
        context_str = "\n\n--- Context Files ---\n"
        for fpath in context_files:
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    context_str += f"File: {fpath}\n```\n{f.read()}\n```\n"
            except Exception as e:
                context_str += f"File: {fpath} (Error reading: {e})\n"
        full_prompt += context_str
        
    # Semantic Search (RAG)
    if context_dir:
        rag_context = _get_relevant_context(prompt, context_dir)
        full_prompt += rag_context

    # 3. Build Messages
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": full_prompt})

    # 4. Dispatch
    if provider in ["openai", "openrouter", "local"]:
        return _query_openai_compatible(provider, resolved_model, messages, key)
    elif provider == "anthropic":
        return _query_anthropic(resolved_model, messages, key)
    elif provider == "gemini":
        return _query_gemini(resolved_model, messages, key)
    else:
        raise ValueError("Unreachable provider state")

# --- MCP Tools ---

if mcp:
    @mcp.tool()
    def ask(
        prompt: str, 
        provider: str = "gemini", 
        model: Optional[str] = None,
        context: Optional[str] = None,
        context_dir: Optional[str] = None
    ) -> str:
        """
        Ask an AI model a question.
        Providers: gemini, openai, anthropic, openrouter, local.
        Args:
            context_dir: Path to a directory containing .ymj files for semantic search (RAG).
        """
        try:
            # If context is provided as a string, append it
            final_prompt = prompt
            if context:
                final_prompt += f"\n\nContext:\n{context}"
                
            return ask_ai(final_prompt, provider, model, context_dir=context_dir)
        except Exception as e:
            return f"Error: {str(e)}"

# --- CLI ---

def main():
    parser = argparse.ArgumentParser(description="SFA AI - Unified AI Prompting")
    subparsers = parser.add_subparsers(dest="command")

    # ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a question to an AI provider")
    ask_parser.add_argument("prompt", help="The prompt to send")
    ask_parser.add_argument("--provider", "-p", default="gemini", choices=list(PROVIDERS.keys()), help="AI Provider")
    ask_parser.add_argument("--model", "-m", help="Model name (overrides default)")
    ask_parser.add_argument("--system", "-s", help="System prompt")
    ask_parser.add_argument("--file", "-f", action="append", help="Attach file to context")
    ask_parser.add_argument("--context-dir", "-d", help="Directory for semantic search (RAG) over .ymj files")
    ask_parser.add_argument("--key", "-k", help="API Key (overrides env var)")
    
    # list-models command (stub for now)
    list_parser = subparsers.add_parser("list-models", help="List default models")

    args = parser.parse_args()

    if args.command == "ask":
        # Handle Stdin (Piping) only for 'ask' command
        stdin_content = ""
        if not sys.stdin.isatty():
            stdin_content = sys.stdin.read()

        try:
            final_prompt = args.prompt
            if stdin_content:
                final_prompt += f"\n\n--- Stdin Input ---\n{stdin_content}"
            
            response = ask_ai(
                prompt=final_prompt,
                provider=args.provider,
                model=args.model,
                system=args.system,
                context_files=args.file,
                context_dir=args.context_dir,
                api_key=args.key
            )
            print(response)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
            
    elif args.command == "list-models":
        print("Default Models per Provider:")
        for p, conf in PROVIDERS.items():
            print(f"  {p.ljust(12)} : {conf['default_model']}")
            
    else:
        if mcp:
            mcp.run()
        else:
            parser.print_help()

if __name__ == "__main__":
    main()
