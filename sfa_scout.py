#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "fastmcp"
# ]
# ///

# ==============================================================================
# DESIGN PHILOSOPHY: USER-FIRST TOOLING
# ==============================================================================
#
# This tool is built by its primary user, for its primary user.
#
# The standard is simple: it works on the first attempt, every time.
#
# If the tool produces an error or unexpected behavior, the TOOL is wrong -
# not the user. When the person who built the tool reaches for it and stumbles,
# that's a bug report, not a learning opportunity.
#
# Implications:
#   - Commands work the way you'd naturally try them
#   - Errors are clear and actionable
#   - No memorization required - intuition should suffice
#   - If you have to check --help, the interface failed
#
# This is dog-fooding at its purest. The builder is the user.
# The tool must fit the hand.
#
# ==============================================================================

"""
sfa_scout.py - Reconnaissance tool for code exploration

USAGE:
  sfa_scout <coords> [pattern]

COORDINATES:
  /path/to/dir              Search directory for pattern
  /path/to/file.py          Show file structure (functions, classes)
  /path/to/file.py:42       Show line 42 with context
  /path/to/file.py:42-58    Show line range
  /path/to/file.py:42:15    Show char position on line

EXAMPLES:
  sfa_scout /src "def.*auth"           # Find pattern in directory
  sfa_scout /src/auth.py               # Show file outline
  sfa_scout /src/auth.py:142           # Show line with context
  sfa_scout /src/auth.py:142:15        # Show char 15 on line 142

OUTPUT:
  Coordinates and context suitable for targeting.
"""

import sys
import re
from pathlib import Path
from typing import Optional, Tuple

__version__ = "1.1.0"

CONTEXT_LINES = 3

# MCP Support
try:
    from fastmcp import FastMCP
    mcp = FastMCP("sfa-scout")
except ImportError:
    mcp = None


def parse_coordinates(coords: str) -> dict:
    """Parse coordinate string into components."""
    result = {
        "path": None,
        "line_start": None,
        "line_end": None,
        "char_start": None,
        "char_end": None,
    }
    
    parts = coords.split(":")
    
    # Handle Windows paths like C:\path
    if len(parts) >= 2 and len(parts[0]) == 1 and parts[0].isalpha():
        result["path"] = f"{parts[0]}:{parts[1]}"
        parts = parts[2:]
    else:
        result["path"] = parts[0]
        parts = parts[1:]
    
    if parts:
        line_spec = parts[0]
        if "-" in line_spec:
            start, end = line_spec.split("-", 1)
            result["line_start"] = int(start) if start else 1
            result["line_end"] = int(end) if end else None
        else:
            result["line_start"] = int(line_spec)
            result["line_end"] = int(line_spec)
        parts = parts[1:]
    
    if parts:
        char_spec = parts[0]
        if "-" in char_spec:
            start, end = char_spec.split("-", 1)
            result["char_start"] = int(start) if start else 0
            result["char_end"] = int(end) if end else None
        else:
            result["char_start"] = int(char_spec)
            result["char_end"] = int(char_spec)
    
    return result


def scout_directory(path: Path, pattern: str) -> Tuple[bool, str]:
    """Search directory for pattern."""
    if not path.exists():
        return False, f"Directory not found: {path}"
    if not path.is_dir():
        return False, f"Not a directory: {path}"
    
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return False, f"Invalid regex: {e}"
    
    results = []
    files_searched = 0
    skip_suffixes = {".pyc", ".pyo", ".so", ".dll", ".exe", ".bin", ".png", ".jpg", ".gif", ".ico", ".zip", ".tar", ".gz"}
    
    for file_path in path.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.name.startswith("."):
            continue
        if file_path.suffix in skip_suffixes:
            continue
        
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            files_searched += 1
        except Exception:
            continue
        
        for line_num, line in enumerate(content.splitlines(), 1):
            if regex.search(line):
                display = line.strip()[:100]
                results.append(f"{file_path}:{line_num}: {display}")
    
    if not results:
        return True, f"No matches for '{pattern}' in {path} ({files_searched} files)"
    
    return True, f"Found {len(results)} matches:\n\n" + "\n".join(results)


def scout_file(path: Path) -> Tuple[bool, str]:
    """Show file structure."""
    if not path.exists():
        return False, f"File not found: {path}"
    if not path.is_file():
        return False, f"Not a file: {path}"
    
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return False, f"Cannot read: {e}"
    
    lines = content.splitlines()
    structures = []
    
    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        
        if stripped.startswith("class "):
            name = stripped.split("(")[0].split(":")[0].replace("class ", "")
            structures.append((line_num, indent, "class", name))
        elif stripped.startswith("def ") or stripped.startswith("async def "):
            name = stripped.split("(")[0].replace("async def ", "").replace("def ", "")
            structures.append((line_num, indent, "def", name))
    
    if not structures:
        return True, f"{path}\n  Lines: {len(lines)}\n  (No classes/functions detected)"
    
    out = [f"{path}", f"  Lines: {len(lines)}", ""]
    for line_num, indent, kind, name in structures:
        prefix = "  " * (indent // 4)
        out.append(f"  {line_num:>4}: {prefix}{kind} {name}")
    
    return True, "\n".join(out)


def scout_lines(path: Path, line_start: int, line_end: int, char_start: Optional[int] = None, char_end: Optional[int] = None) -> Tuple[bool, str]:
    """Show specific lines with context."""
    if not path.exists():
        return False, f"File not found: {path}"
    if not path.is_file():
        return False, f"Not a file: {path}"
    
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return False, f"Cannot read: {e}"
    
    lines = content.splitlines()
    total = len(lines)
    
    if line_start < 1 or line_start > total:
        return False, f"Line {line_start} out of range (1-{total})"
    
    if line_end is None:
        line_end = line_start
    line_end = min(line_end, total)
    
    ctx_start = max(1, line_start - CONTEXT_LINES)
    ctx_end = min(total, line_end + CONTEXT_LINES)
    
    out = [f"{path}:{line_start}" + (f"-{line_end}" if line_end != line_start else ""), ""]
    
    for i in range(ctx_start, ctx_end + 1):
        line = lines[i - 1]
        marker = ">>>" if line_start <= i <= line_end else "   "
        out.append(f"{marker} {i:>4}│{line}")
        
        if char_start is not None and line_start <= i <= line_end:
            c_end = char_end if char_end else char_start
            pointer = " " * (len(marker) + 5 + char_start) + "^"
            if c_end > char_start:
                pointer = " " * (len(marker) + 5 + char_start) + "^" + "~" * (c_end - char_start - 1) + "^"
            out.append(pointer)
    
    return True, "\n".join(out)


def scout_file_pattern(path: Path, pattern: str, line_start: Optional[int] = None, line_end: Optional[int] = None) -> Tuple[bool, str]:
    """Search within file for pattern."""
    if not path.exists():
        return False, f"File not found: {path}"
    if not path.is_file():
        return False, f"Not a file: {path}"
    
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return False, f"Invalid regex: {e}"
    
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return False, f"Cannot read: {e}"
    
    lines = content.splitlines()
    total = len(lines)
    
    if line_start is None:
        line_start = 1
    if line_end is None:
        line_end = total
    
    results = []
    for i in range(line_start - 1, min(line_end, total)):
        line = lines[i]
        match = regex.search(line)
        if match:
            results.append((i + 1, match.start(), match.end(), line.strip()[:100]))
    
    if not results:
        return True, f"No matches for '{pattern}' in {path}"
    
    out = [f"Found {len(results)} matches for '{pattern}':", ""]
    for line_num, cs, ce, display in results:
        out.append(f"  {path}:{line_num}:{cs}-{ce}")
        out.append(f"    {display}")
    
    return True, "\n".join(out)


# --- MCP Tools ---

if mcp:

    @mcp.tool()
    async def scout_dir(directory: str, pattern: str) -> str:
        """Search directory for regex pattern. Returns file:line matches."""
        path = Path(directory).resolve()
        success, output = scout_directory(path, pattern)
        return output if success else f"[FAIL] {output}"

    @mcp.tool()
    async def scout_structure(file_path: str) -> str:
        """Show file structure (classes, functions with line numbers)."""
        path = Path(file_path).resolve()
        success, output = scout_file(path)
        return output if success else f"[FAIL] {output}"

    @mcp.tool()
    async def scout_line(file_path: str, line: int, context: int = 3) -> str:
        """Show specific line with context. Returns numbered lines with markers."""
        global CONTEXT_LINES
        CONTEXT_LINES = context
        path = Path(file_path).resolve()
        success, output = scout_lines(path, line, line)
        return output if success else f"[FAIL] {output}"

    @mcp.tool()
    async def scout_range(file_path: str, start_line: int, end_line: int) -> str:
        """Show line range with context."""
        path = Path(file_path).resolve()
        success, output = scout_lines(path, start_line, end_line)
        return output if success else f"[FAIL] {output}"

    @mcp.tool()
    async def scout_pattern(file_path: str, pattern: str) -> str:
        """Search file for regex pattern. Returns matches with coordinates."""
        path = Path(file_path).resolve()
        success, output = scout_file_pattern(path, pattern)
        return output if success else f"[FAIL] {output}"


def main():
    args = sys.argv[1:]
    
    if not args or args[0] in ["-h", "--help"]:
        print(__doc__.strip())
        sys.exit(0)
    
    if args[0] in ["-v", "--version"]:
        print(f"sfa_scout {__version__}")
        sys.exit(0)
    
    if args[0] == "mcp" and mcp:
        mcp.run()
        return
    
    coords_str = args[0]
    pattern = args[1] if len(args) > 1 else None
    
    coords = parse_coordinates(coords_str)
    path = Path(coords["path"]).resolve()
    
    if path.is_dir():
        if pattern:
            success, output = scout_directory(path, pattern)
        else:
            items = sorted(path.iterdir())[:50]
            success, output = True, f"{path}/\n" + "\n".join(f"  {p.name}" for p in items)
    elif coords["line_start"] is not None:
        if pattern:
            success, output = scout_file_pattern(path, pattern, coords["line_start"], coords["line_end"])
        else:
            success, output = scout_lines(path, coords["line_start"], coords["line_end"], coords["char_start"], coords["char_end"])
    elif pattern:
        success, output = scout_file_pattern(path, pattern)
    else:
        success, output = scout_file(path)
    
    print(output)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
