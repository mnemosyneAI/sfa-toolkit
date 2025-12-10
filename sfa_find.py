# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "fastmcp",
#     "fastembed-gpu",
#     "numpy",
# ]
# ///

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np

# Attempt to import fastmcp, but don't fail if running in CLI mode without it
try:
    from fastmcp import FastMCP
    mcp = FastMCP("sfa-find")
except ImportError:
    mcp = None

# --- Global Security ---
ALLOWED_PATHS: List[Path] = []

# --- Standard Runtime Helpers ---

def _normalize_path(path_str: str) -> Path:
    """
    Normalize path for cross-platform compatibility.
    Handles Windows mixed slashes and Git Bash style paths.
    """
    if not path_str:
        return Path.cwd()
    
    # Handle Git Bash style paths (e.g., /c/Users/...)
    if sys.platform == "win32" and path_str.startswith("/") and len(path_str) > 2 and path_str[2] == "/":
        # Convert /c/Users to c:/Users
        drive = path_str[1]
        rest = path_str[2:]
        path_str = f"{drive}:{rest}"
    
    path = Path(path_str).resolve()
    
    # Security Check
    if ALLOWED_PATHS:
        is_allowed = False
        for allowed in ALLOWED_PATHS:
            try:
                path.relative_to(allowed)
                is_allowed = True
                break
            except ValueError:
                continue
        
        if not is_allowed:
            raise PermissionError(f"Access denied: Path '{path}' is not in allowed paths.")
            
    return path

def _get_tool_path(tool_name: str, env_var: str) -> Optional[str]:
    """
    Find a tool in the system PATH or via environment variable override.
    """
    # 1. Check Environment Variable Override
    override = os.environ.get(env_var)
    if override and os.path.exists(override):
        return override
    
    # 2. Check System PATH
    return shutil.which(tool_name)

def _run_command(args: List[str], cwd: Optional[Path] = None) -> str:
    """Run a command and return stdout."""
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8',
            errors='replace'
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        # Return empty string or error message depending on strictness.
        # For 'find', empty string usually means no results.
        return ""

# --- Semantic Search Logic ---

def _cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_semantic(query: str, path: str = ".", top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Find .ymj files semantically related to the query.
    """
    try:
        from fastembed import TextEmbedding
    except ImportError:
        return [{"error": "fastembed not installed"}]

    root_path = _normalize_path(path)
    if not root_path.exists():
        return [{"error": f"Path {path} does not exist"}]

    # 1. Embed Query
    try:
        # Runtime CUDA detection using ONNX Runtime
        import onnxruntime as ort
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        
        model = TextEmbedding(model_name="nomic-ai/nomic-embed-text-v1.5", providers=providers)
        query_embedding = list(model.embed([query]))[0]
    except Exception as e:
        return [{"error": f"Embedding failed: {e}"}]

    # 2. Scan Files
    candidates = []
    
    # We only look for .ymj files
    # Using rglob is efficient enough for typical doc repos
    for fpath in root_path.rglob("*.ymj"):
        try:
            # Quick read of footer only? 
            # For now, read full file to be safe, but we could optimize to read last 4KB
            content = fpath.read_text(encoding="utf-8")
            
            # Extract Footer
            footer_end = content.rfind("```")
            if footer_end == -1: continue
            
            footer_start = content.rfind("```json", 0, footer_end)
            if footer_start == -1: continue
            
            footer_json = content[footer_start+7:footer_end].strip()
            footer = json.loads(footer_json)
            
            # Check for embedding
            if "index" in footer and "embedding" in footer["index"]:
                vec = footer["index"]["embedding"]
                if not vec: continue # Skip empty embeddings
                
                score = _cosine_similarity(query_embedding, vec)
                
                # Extract Summary from Header
                summary = "No summary"
                if content.startswith("---"):
                    try:
                        header_end = content.index("\n---", 3)
                        header_lines = content[3:header_end].splitlines()
                        for line in header_lines:
                            if line.strip().startswith("doc_summary:"):
                                summary = line.split(":", 1)[1].strip()
                                break
                    except Exception:
                        pass

                candidates.append({
                    "path": str(fpath),
                    "score": float(score),
                    "summary": summary
                })
                
        except Exception:
            continue

    # 3. Rank
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_k]

# --- Core Logic ---

def find_files(pattern: str, path: str = ".", use_gitignore: bool = True) -> List[str]:
    """
    Find files matching a pattern (glob or regex depending on tool).
    Prioritizes `rg` (ripgrep) -> `git` -> `find` (linux) -> `os.walk` (fallback).
    """
    root_path = _normalize_path(path)
    results = []
    
    # 1. Try Ripgrep (rg) - The Gold Standard
    rg_path = _get_tool_path("rg", "SFA_RG_PATH")
    if rg_path:
        cmd = [rg_path, "--files", "--glob", pattern]
        if not use_gitignore:
            cmd.append("--no-ignore")
        
        output = _run_command(cmd, cwd=root_path)
        if output:
            # rg returns relative paths
            results = [str(root_path / p).replace("\\", "/") for p in output.splitlines()]
            return results

    # 2. Try Git (if in a repo)
    git_path = _get_tool_path("git", "SFA_GIT_PATH")
    if git_path and (root_path / ".git").exists():
        # git ls-files with glob pattern
        cmd = [git_path, "ls-files", pattern]
        output = _run_command(cmd, cwd=root_path)
        if output:
            results = [str(root_path / p).replace("\\", "/") for p in output.splitlines()]
            return results

    # 3. Fallback: os.walk (Native Python)
    # Simple glob matching
    import fnmatch
    for root, _, files in os.walk(root_path):
        for filename in files:
            if fnmatch.fnmatch(filename, pattern):
                full_path = Path(root) / filename
                results.append(str(full_path).replace("\\", "/"))
    
    return results

def locate_tool(tool_name: str) -> Dict[str, Any]:
    """
    Check if a system tool is available and where it is.
    Useful for verifying the runtime environment.
    """
    path = shutil.which(tool_name)
    return {
        "tool": tool_name,
        "available": path is not None,
        "path": path.replace("\\", "/") if path else None
    }

# --- MCP Tools ---

if mcp:
    @mcp.tool()
    def find(pattern: str, path: str = ".", use_gitignore: bool = True) -> str:
        """
        Find files by name pattern.
        Args:
            pattern: Glob pattern (e.g. "*.py", "config.*")
            path: Root directory to search in
            use_gitignore: Whether to respect .gitignore (default: True)
        """
        files = find_files(pattern, path, use_gitignore)
        return json.dumps(files, indent=2)

    @mcp.tool()
    def semantic(query: str, path: str = ".", top_k: int = 5) -> str:
        """Find .ymj files semantically related to the query."""
        results = find_semantic(query, path, top_k)
        return json.dumps(results, indent=2)

    @mcp.tool()
    def check_runtime() -> str:
        """Check for presence of Standard Runtime tools (git, gh, rg)."""
        tools = ["git", "gh", "rg"]
        status = {t: locate_tool(t) for t in tools}
        return json.dumps(status, indent=2)

# --- CLI Dispatcher ---

def main():
    parser = argparse.ArgumentParser(description="SFA Find - Filesystem Locator")
    parser.add_argument("--allowed-paths", help="Comma-separated list of allowed paths (MCP security)")
    subparsers = parser.add_subparsers(dest="command")

    # find command
    find_parser = subparsers.add_parser("find", help="Find files by pattern")
    find_parser.add_argument("pattern", help="Glob pattern (e.g. *.py)")
    find_parser.add_argument("--path", default=".", help="Root path")
    find_parser.add_argument("--no-ignore", action="store_false", dest="use_gitignore", help="Ignore .gitignore")

    # semantic command
    sem_parser = subparsers.add_parser("semantic", help="Find .ymj files by meaning")
    sem_parser.add_argument("query", help="Natural language query")
    sem_parser.add_argument("path", nargs="?", default=".", help="Root path")
    sem_parser.add_argument("--top", "-k", type=int, default=5, help="Max results")

    # check-runtime command
    subparsers.add_parser("check-runtime", help="Check for required tools")

    args = parser.parse_args()

    if args.allowed_paths:
        global ALLOWED_PATHS
        for p in args.allowed_paths.split(","):
            if p.strip():
                ALLOWED_PATHS.append(Path(p.strip()).resolve())

    if args.command == "find":
        results = find_files(args.pattern, args.path, args.use_gitignore)
        print(json.dumps(results, indent=2))
    elif args.command == "semantic":
        results = find_semantic(args.query, args.path, args.top)
        if not results:
            print("No matches found.")
        elif "error" in results[0]:
            print(f"Error: {results[0]['error']}")
        else:
            print(f"{'SCORE':<8} {'PATH':<50} {'SUMMARY'}")
            print("-" * 80)
            for r in results:
                path_display = Path(r['path']).name
                print(f"{r['score']:.4f}   {path_display:<50} {r['summary']}")
    elif args.command == "check-runtime":
        tools = ["git", "gh", "rg"]
        status = {t: locate_tool(t) for t in tools}
        print(json.dumps(status, indent=2))
    else:
        # If no args, run MCP server
        if mcp:
            mcp.run()
        else:
            parser.print_help()

if __name__ == "__main__":
    main()
