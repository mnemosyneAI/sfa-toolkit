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
import ast
import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

try:
    from fastmcp import FastMCP
    mcp = FastMCP("sfa-read")
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

def read_file_content(path: str, start_line: int = 1, end_line: int = -1) -> str:
    """Read file content with optional line range."""
    file_path = _normalize_path(path)
    if not file_path.exists():
        return f"Error: File not found: {path}"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        total_lines = len(lines)
        if end_line == -1:
            end_line = total_lines
            
        # Adjust for 0-based indexing
        start_idx = max(0, start_line - 1)
        end_idx = min(total_lines, end_line)
        
        return "".join(lines[start_idx:end_idx])
    except Exception as e:
        return f"Error reading file: {str(e)}"

def get_tree(path: str = ".", max_depth: int = 2) -> Dict[str, Any]:
    """Generate a directory tree structure."""
    root_path = _normalize_path(path)
    
    def build_tree(dir_path: Path, current_depth: int) -> Dict[str, Any]:
        if current_depth > max_depth:
            return "..."
        
        tree = {}
        try:
            # Sort directories first, then files
            items = sorted(os.listdir(dir_path), key=lambda x: (not os.path.isdir(dir_path / x), x.lower()))
            
            for item in items:
                if item.startswith("."): continue # Skip hidden
                
                item_path = dir_path / item
                if item_path.is_dir():
                    tree[item + "/"] = build_tree(item_path, current_depth + 1)
                else:
                    tree[item] = "file"
        except PermissionError:
            return "Permission Denied"
        return tree

    return {root_path.name + "/": build_tree(root_path, 1)}

def get_code_structure(path: str) -> Dict[str, Any]:
    """
    Parse Python file and return structure (classes, functions, imports).
    Uses AST for robust parsing.
    """
    file_path = _normalize_path(path)
    if file_path.suffix != ".py":
        return {"error": "Only .py files supported for structure analysis"}
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
            
        structure = {
            "imports": [],
            "classes": [],
            "functions": []
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    structure["imports"].append(name.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for name in node.names:
                    structure["imports"].append(f"{module}.{name.name}")
            elif isinstance(node, ast.FunctionDef):
                structure["functions"].append({
                    "name": node.name,
                    "lineno": node.lineno,
                    "args": [a.arg for a in node.args.args]
                })
            elif isinstance(node, ast.ClassDef):
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                structure["classes"].append({
                    "name": node.name,
                    "lineno": node.lineno,
                    "methods": methods
                })
                
        return structure
    except Exception as e:
        return {"error": f"Failed to parse AST: {str(e)}"}

def get_docstrings(path: str) -> Dict[str, Any]:
    """
    Extract docstrings from a Python file.
    Returns module, class, and function docstrings.
    """
    file_path = _normalize_path(path)
    if file_path.suffix != ".py":
        return {"error": "Only .py files supported for docstring extraction"}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
        
        docs = {
            "module": ast.get_docstring(tree),
            "classes": {},
            "functions": {}
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                docs["functions"][node.name] = ast.get_docstring(node)
            elif isinstance(node, ast.ClassDef):
                class_doc = ast.get_docstring(node)
                method_docs = {}
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        method_docs[item.name] = ast.get_docstring(item)
                
                docs["classes"][node.name] = {
                    "doc": class_doc,
                    "methods": method_docs
                }
        return docs
    except Exception as e:
        return {"error": f"Failed to extract docstrings: {str(e)}"}

def get_stats(path: str) -> Dict[str, Any]:
    """Get file or directory statistics."""
    target_path = _normalize_path(path)
    
    if not target_path.exists():
        return {"error": f"Path not found: {path}"}
    
    try:
        stat = target_path.stat()
        return {
            "path": str(target_path),
            "type": "directory" if target_path.is_dir() else "file",
            "size_bytes": stat.st_size,
            "created": datetime.datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "permissions": oct(stat.st_mode)[-3:]
        }
    except Exception as e:
        return {"error": f"Error getting stats: {str(e)}"}

def read_directory(path: str, extensions: Optional[str] = None, recursive: bool = True) -> str:
    """
    Read all files in a directory matching extensions.
    Returns concatenated content with headers.
    """
    root_path = _normalize_path(path)
    if not root_path.exists():
        return f"Error: Path not found: {path}"

    ext_set = set(e.strip() for e in extensions.split(",")) if extensions else None
    output = []

    for root, dirs, files in os.walk(root_path):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        
        if not recursive and Path(root) != root_path:
            continue

        for file in files:
            if file.startswith("."): continue
            
            if ext_set:
                if not any(file.endswith(ext) for ext in ext_set):
                    continue
            
            file_path = Path(root) / file
            rel_path = file_path.relative_to(root_path)
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                    output.append(f"--- File: {rel_path} ---\n{content}\n")
            except Exception as e:
                output.append(f"--- File: {rel_path} ---\nError reading file: {str(e)}\n")

    if not output:
        return "No matching files found."
        
    return "\n".join(output)

# --- MCP Tools ---

if mcp:
    @mcp.tool()
    def read(path: str, start_line: int = 1, end_line: int = -1) -> str:
        """Read file content (full or range)."""
        return read_file_content(path, start_line, end_line)

    @mcp.tool()
    def tree(path: str = ".", max_depth: int = 2) -> str:
        """Get directory tree structure."""
        return json.dumps(get_tree(path, max_depth), indent=2)

    @mcp.tool()
    def structure(path: str) -> str:
        """Get code structure (AST) for Python files."""
        return json.dumps(get_code_structure(path), indent=2)

    @mcp.tool()
    def docstrings(path: str) -> str:
        """Get docstrings for Python files."""
        return json.dumps(get_docstrings(path), indent=2)

    @mcp.tool()
    def stats(path: str) -> str:
        """Get file/directory metadata."""
        return json.dumps(get_stats(path), indent=2)

    @mcp.tool()
    def read_dir(path: str, extensions: str = None, recursive: bool = True) -> str:
        """
        Read all files in directory.
        Args:
            path: Directory path
            extensions: Comma-separated extensions (e.g. ".py,.md")
            recursive: Whether to search recursively (default: True)
        """
        return read_directory(path, extensions, recursive)

# --- CLI Dispatcher ---

def main():
    parser = argparse.ArgumentParser(description="SFA Read - Content Consumer")
    parser.add_argument("--allowed-paths", help="Comma-separated list of allowed paths (MCP security)")
    subparsers = parser.add_subparsers(dest="command")

    # read command
    read_parser = subparsers.add_parser("read", help="Read file content")
    read_parser.add_argument("path", help="File path")
    read_parser.add_argument("--start", type=int, default=1, help="Start line")
    read_parser.add_argument("--end", type=int, default=-1, help="End line")

    # tree command
    tree_parser = subparsers.add_parser("tree", help="Show directory tree")
    tree_parser.add_argument("--path", default=".", help="Root path")
    tree_parser.add_argument("--depth", type=int, default=2, help="Max depth")

    # structure command
    struct_parser = subparsers.add_parser("structure", help="Show code structure")
    struct_parser.add_argument("path", help="File path")

    # docstrings command
    doc_parser = subparsers.add_parser("docstrings", help="Show docstrings")
    doc_parser.add_argument("path", help="File path")

    # stats command
    stat_parser = subparsers.add_parser("stats", help="Get metadata")
    stat_parser.add_argument("path", help="Path to inspect")

    # read-dir command
    dir_parser = subparsers.add_parser("read-dir", help="Read all files in directory")
    dir_parser.add_argument("path", help="Directory path")
    dir_parser.add_argument("--ext", help="Extensions (e.g. .py,.md)")
    dir_parser.add_argument("--no-recursive", action="store_false", dest="recursive", help="Disable recursion")

    args = parser.parse_args()

    if args.allowed_paths:
        global ALLOWED_PATHS
        for p in args.allowed_paths.split(","):
            if p.strip():
                ALLOWED_PATHS.append(Path(p.strip()).resolve())

    if args.command == "read":
        print(read_file_content(args.path, args.start, args.end))
    elif args.command == "tree":
        print(json.dumps(get_tree(args.path, args.depth), indent=2))
    elif args.command == "structure":
        print(json.dumps(get_code_structure(args.path), indent=2))
    elif args.command == "docstrings":
        print(json.dumps(get_docstrings(args.path), indent=2))
    elif args.command == "stats":
        print(json.dumps(get_stats(args.path), indent=2))
    elif args.command == "read-dir":
        print(read_directory(args.path, args.ext, args.recursive))
    else:
        if mcp:
            mcp.run()
        else:
            parser.print_help()

if __name__ == "__main__":
    main()
