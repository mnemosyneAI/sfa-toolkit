# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "fastmcp",
# ]
# ///

import argparse
import json
import os
import sys
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any

try:
    from fastmcp import FastMCP
    mcp = FastMCP("sfa-edit")
except ImportError:
    mcp = None

# --- Global Security ---
ALLOWED_PATHS: List[Path] = []

# --- Standard Runtime Helpers ---

def _normalize_path(path_str: str) -> Path:
    if not path_str:
        return Path.cwd()
    if sys.platform == "win32" and path_str.startswith("/") and len(path_str) > 2 and path_str[2] == "/":
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

# --- Core Logic ---

def create_file(path: str, content: str) -> str:
    """Create a new file with content."""
    file_path = _normalize_path(path)
    if file_path.exists():
        return f"Error: File already exists: {path}"
    
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully created {path}"
    except Exception as e:
        return f"Error creating file: {str(e)}"

def replace_string(path: str, old: str, new: str) -> str:
    """Replace exact string occurrence."""
    file_path = _normalize_path(path)
    if not file_path.exists():
        return f"Error: File not found: {path}"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if old not in content:
            return f"Error: String not found in {path}"
        
        # Safety check: Ensure unique replacement if possible, or replace all?
        # For simplicity in SFA, we replace all occurrences but warn if count > 1?
        # Standard behavior: Replace all.
        new_content = content.replace(old, new)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
            
        return f"Successfully replaced content in {path}"
    except Exception as e:
        return f"Error replacing string: {str(e)}"

def replace_line(path: str, line_no: int, new_content: str) -> str:
    """Replace a specific line number (1-based)."""
    file_path = _normalize_path(path)
    if not file_path.exists():
        return f"Error: File not found: {path}"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        if line_no < 1 or line_no > len(lines):
            return f"Error: Line {line_no} out of range (1-{len(lines)})"
            
        # Preserve newline if present in original
        original_line = lines[line_no - 1]
        if original_line.endswith('\n') and not new_content.endswith('\n'):
            new_content += '\n'
            
        lines[line_no - 1] = new_content
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
            
        return f"Successfully replaced line {line_no} in {path}"
    except Exception as e:
        return f"Error replacing line: {str(e)}"

def append_file(path: str, content: str) -> str:
    """Append content to the end of a file."""
    file_path = _normalize_path(path)
    if not file_path.exists():
        return f"Error: File not found: {path}"
    
    try:
        with open(file_path, 'a', encoding='utf-8') as f:
            if not content.startswith('\n'):
                f.write('\n')
            f.write(content)
        return f"Successfully appended to {path}"
    except Exception as e:
        return f"Error appending to file: {str(e)}"

# --- MCP Tools ---

if mcp:
    @mcp.tool()
    def create(path: str, content: str) -> str:
        """Create a new file."""
        return create_file(path, content)

    @mcp.tool()
    def replace(path: str, old: str, new: str) -> str:
        """Replace exact string."""
        return replace_string(path, old, new)

    @mcp.tool()
    def replace_line_content(path: str, line: int, content: str) -> str:
        """Replace specific line number."""
        return replace_line(path, line, content)

    @mcp.tool()
    def append(path: str, content: str) -> str:
        """Append to file."""
        return append_file(path, content)

# --- CLI Dispatcher ---

def main():
    parser = argparse.ArgumentParser(description="SFA Edit - Surgical Modification")
    parser.add_argument("--allowed-paths", help="Comma-separated list of allowed paths (MCP security)")
    subparsers = parser.add_subparsers(dest="command")

    # create
    create_parser = subparsers.add_parser("create", help="Create file")
    create_parser.add_argument("path", help="File path")
    create_parser.add_argument("content", help="File content")

    # replace
    replace_parser = subparsers.add_parser("replace", help="Replace string")
    replace_parser.add_argument("path", help="File path")
    replace_parser.add_argument("old", help="Old string")
    replace_parser.add_argument("new", help="New string")

    # replace-line
    line_parser = subparsers.add_parser("replace-line", help="Replace line")
    line_parser.add_argument("path", help="File path")
    line_parser.add_argument("line", type=int, help="Line number")
    line_parser.add_argument("content", help="New content")

    # append
    append_parser = subparsers.add_parser("append", help="Append to file")
    append_parser.add_argument("path", help="File path")
    append_parser.add_argument("content", help="Content to append")

    args = parser.parse_args()

    if args.allowed_paths:
        global ALLOWED_PATHS
        for p in args.allowed_paths.split(","):
            if p.strip():
                ALLOWED_PATHS.append(Path(p.strip()).resolve())

    if args.command == "create":
        print(create_file(args.path, args.content))
    elif args.command == "replace":
        print(replace_string(args.path, args.old, args.new))
    elif args.command == "replace-line":
        print(replace_line(args.path, args.line, args.content))
    elif args.command == "append":
        print(append_file(args.path, args.content))
    else:
        if mcp:
            mcp.run()
        else:
            parser.print_help()

if __name__ == "__main__":
    main()
