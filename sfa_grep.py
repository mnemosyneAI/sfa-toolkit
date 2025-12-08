# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "fastmcp",
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

try:
    from fastmcp import FastMCP

    mcp = FastMCP("sfa-grep")
except ImportError:
    mcp = None

# --- Global Security ---
ALLOWED_PATHS: List[Path] = []

# --- Standard Runtime Helpers ---


def _normalize_path(path_str: str) -> Path:
    if not path_str:
        return Path.cwd()
    if (
        sys.platform == "win32"
        and path_str.startswith("/")
        and len(path_str) > 2
        and path_str[2] == "/"
    ):
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
            raise PermissionError(
                f"Access denied: Path '{path}' is not in allowed paths."
            )

    return path


def _get_tool_path(tool_name: str, env_var: str) -> Optional[str]:
    override = os.environ.get(env_var)
    if override and os.path.exists(override):
        return override
    return shutil.which(tool_name)


def _run_command(args: List[str], cwd: Optional[Path] = None) -> str:
    try:
        # Convert Path to string for Windows compatibility
        cwd_str = str(cwd) if cwd else None
        result = subprocess.run(
            args,
            cwd=cwd_str,
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
            errors="replace",
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return ""


# --- Core Logic ---


def search_text(pattern: str, path: str = ".", context_lines: int = 0) -> List[str]:
    """
    Search for text content within files.
    Uses `rg` (ripgrep) if available, falls back to `git grep`, then Python.
    """
    root_path = _normalize_path(path)
    results = []

    # Handle file vs directory
    if root_path.is_file():
        cwd_path = root_path.parent
        target_arg = root_path.name
    else:
        cwd_path = root_path
        target_arg = None

    # 1. Try Ripgrep (rg)
    rg_path = _get_tool_path("rg", "SFA_RG_PATH")
    if rg_path:
        # rg --line-number --column --no-heading [pattern] [target]
        cmd = [rg_path, "--line-number", "--column", "--no-heading", pattern]
        if context_lines > 0:
            cmd.extend(["-C", str(context_lines)])
        if target_arg:
            cmd.append(target_arg)

        output = _run_command(cmd, cwd=cwd_path)
        if output:
            return output.splitlines()

    # 2. Try Git Grep
    git_path = _get_tool_path("git", "SFA_GIT_PATH")
    if git_path and (cwd_path / ".git").exists():
        cmd = [git_path, "grep", "-n", "--column", pattern]
        if context_lines > 0:
            cmd.extend(["-C", str(context_lines)])
        if target_arg:
            cmd.extend(["--", target_arg])

        output = _run_command(cmd, cwd=cwd_path)
        if output:
            return output.splitlines()

    # 3. Fallback: Python (Slow, simple)
    # Only searches .py, .md, .txt, .json, .yml, .yaml to avoid binary mess in fallback mode
    extensions = {
        ".py",
        ".md",
        ".txt",
        ".json",
        ".yml",
        ".yaml",
        ".js",
        ".ts",
        ".html",
        ".css",
    }

    # If target is a file, just search it
    if root_path.is_file():
        files_to_search = [root_path]
        cwd_path = root_path.parent  # For relative path calc
    else:
        files_to_search = []
        for root, _, files in os.walk(root_path):
            for filename in files:
                if Path(filename).suffix in extensions:
                    files_to_search.append(Path(root) / filename)

    for file_path in files_to_search:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f, 1):
                    if pattern in line:
                        # Format: path:line:col:content
                        rel_path = str(file_path.relative_to(cwd_path)).replace(
                            "\\", "/"
                        )
                        results.append(f"{rel_path}:{i}:1:{line.strip()}")
        except Exception:
            continue

    return results


def count_matches(pattern: str, path: str = ".") -> int:
    """
    Count total occurrences of a pattern. Useful for BDA.
    """
    root_path = _normalize_path(path)

    rg_path = _get_tool_path("rg", "SFA_RG_PATH")
    if rg_path:
        cmd = [rg_path, "-c", pattern]  # -c counts lines with matches per file
        output = _run_command(cmd, cwd=root_path)
        total = 0
        for line in output.splitlines():
            # Output format: file:count
            parts = line.rsplit(":", 1)
            if len(parts) == 2 and parts[1].isdigit():
                total += int(parts[1])
        return total

    # Fallback to search_text length
    return len(search_text(pattern, path))


# --- MCP Tools ---

if mcp:

    @mcp.tool()
    def grep(pattern: str, path: str = ".", context_lines: int = 0) -> str:
        """
        Search for text in files.
        Args:
            pattern: Regex or text pattern
            path: Root directory
            context_lines: Number of context lines to show (default: 0)
        """
        lines = search_text(pattern, path, context_lines)
        return "\n".join(lines)

    @mcp.tool()
    def count(pattern: str, path: str = ".") -> str:
        """
        Count matches of a pattern. Useful for verifying edits (BDA).
        """
        count = count_matches(pattern, path)
        return json.dumps({"pattern": pattern, "count": count})

# --- CLI Dispatcher ---


def main():
    parser = argparse.ArgumentParser(description="SFA Grep - Content Search & BDA")
    parser.add_argument(
        "--allowed-paths", help="Comma-separated list of allowed paths (MCP security)"
    )
    subparsers = parser.add_subparsers(dest="command")

    # search command
    search_parser = subparsers.add_parser("search", help="Search text in files")
    search_parser.add_argument("pattern", help="Search pattern")
    search_parser.add_argument("--path", default=".", help="Root path")
    search_parser.add_argument(
        "-C", "--context", type=int, default=0, help="Context lines"
    )

    # count command
    count_parser = subparsers.add_parser("count", help="Count matches")
    count_parser.add_argument("pattern", help="Search pattern")
    count_parser.add_argument("--path", default=".", help="Root path")

    args = parser.parse_args()

    if args.allowed_paths:
        global ALLOWED_PATHS
        for p in args.allowed_paths.split(","):
            if p.strip():
                ALLOWED_PATHS.append(Path(p.strip()).resolve())

    if args.command == "search":
        results = search_text(args.pattern, args.path, args.context)
        # Use UTF-8 encoding for Windows console to handle Unicode
        output = "\n".join(results)
        sys.stdout.buffer.write(output.encode("utf-8"))
        sys.stdout.buffer.write(b"\n")
        sys.stdout.flush()
    elif args.command == "count":
        c = count_matches(args.pattern, args.path)
        print(json.dumps({"pattern": args.pattern, "count": c}, indent=2))
    else:
        if mcp:
            mcp.run()
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
