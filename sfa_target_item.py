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
sfa_target_item.py - Pattern find/replace targeting

USAGE:
  sfa_target_item assess <coords> <old> <new>    # Preview changes
  sfa_target_item strike <coords> <old> <new>    # Execute changes

COORDINATES (scope the search):
  /path/to/dir              Find in all files in directory
  /path/to/file.py          Find in this file
  /path/to/file.py:42       Find on this line only
  /path/to/file.py:42-58    Find in this line range

CONTENT:
  "literal string"          Inline content
  @filename                 Read from file

EXAMPLES:
  sfa_target_item assess /src "old_func" "new_func"
  sfa_target_item strike /src/auth.py "broken" "fixed"
  sfa_target_item assess /src/auth.py:142 "old" "new"

OUTPUT:
  assess: Shows diff preview of all changes
  strike: Executes changes, creates backups
"""

import sys
import re
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List

__version__ = "1.1.0"

BACKUP_DIR = Path.home() / ".sfa" / "backups"

# MCP Support
try:
    from fastmcp import FastMCP
    mcp = FastMCP("sfa-target-item")
except ImportError:
    mcp = None


def parse_coordinates(coords: str) -> dict:
    """Parse coordinate string into components."""
    result = {
        "path": None,
        "line_start": None,
        "line_end": None,
    }
    
    parts = coords.split(":")
    
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
    
    return result


def read_content(content: str) -> str:
    """Read content from string or @file."""
    if content.startswith("@"):
        file_path = Path(content[1:])
        if not file_path.exists():
            raise FileNotFoundError(f"Content file not found: {file_path}")
        return file_path.read_text(encoding="utf-8")
    return content


def backup_file(path: Path) -> Path:
    """Create backup of file."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{path.name}.{timestamp}.backup"
    backup_path = BACKUP_DIR / backup_name
    shutil.copy2(path, backup_path)
    return backup_path


def find_replacements_in_file(
    file_path: Path, 
    old: str, 
    new: str,
    line_start: Optional[int] = None,
    line_end: Optional[int] = None
) -> List[dict]:
    """Find all replacements in a file, optionally limited to line range."""
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return []
    
    lines = content.splitlines(keepends=True)
    total = len(lines)
    
    # Default to whole file
    if line_start is None:
        line_start = 1
    if line_end is None:
        line_end = total
    
    replacements = []
    
    for i in range(line_start - 1, min(line_end, total)):
        line = lines[i]
        if old in line:
            new_line = line.replace(old, new)
            count = line.count(old)
            replacements.append({
                "file": file_path,
                "line_num": i + 1,
                "old_line": line.rstrip("\n"),
                "new_line": new_line.rstrip("\n"),
                "count": count,
            })
    
    return replacements


def assess(coords: dict, old: str, new: str) -> Tuple[bool, str]:
    """Preview all replacements (dry run)."""
    path = Path(coords["path"]).resolve()
    
    if not path.exists():
        return False, f"Path not found: {path}"
    
    all_replacements = []
    
    if path.is_dir():
        skip_suffixes = {".pyc", ".pyo", ".so", ".dll", ".exe", ".bin", ".png", ".jpg", ".gif", ".ico", ".zip", ".tar", ".gz"}
        for file_path in path.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.name.startswith("."):
                continue
            if file_path.suffix in skip_suffixes:
                continue
            replacements = find_replacements_in_file(file_path, old, new)
            all_replacements.extend(replacements)
    else:
        replacements = find_replacements_in_file(path, old, new, coords["line_start"], coords["line_end"])
        all_replacements.extend(replacements)
    
    if not all_replacements:
        return True, f"No matches for '{old}' in {path}"
    
    # Build diff output
    total_count = sum(r["count"] for r in all_replacements)
    out = [f"Found {total_count} replacement(s) in {len(all_replacements)} line(s):", ""]
    
    current_file = None
    for r in all_replacements:
        if r["file"] != current_file:
            current_file = r["file"]
            out.append(f"--- {r['file']}")
            out.append(f"+++ {r['file']}")
        
        out.append(f"@@ line {r['line_num']} @@")
        out.append(f"-{r['old_line']}")
        out.append(f"+{r['new_line']}")
        out.append("")
    
    out.append(f"[ASSESS] {total_count} replacement(s) ready. Run 'strike' to apply.")
    
    return True, "\n".join(out)


def strike(coords: dict, old: str, new: str) -> Tuple[bool, str]:
    """Execute all replacements."""
    path = Path(coords["path"]).resolve()
    
    if not path.exists():
        return False, f"Path not found: {path}"
    
    all_replacements = []
    files_to_modify = {}
    
    if path.is_dir():
        skip_suffixes = {".pyc", ".pyo", ".so", ".dll", ".exe", ".bin", ".png", ".jpg", ".gif", ".ico", ".zip", ".tar", ".gz"}
        for file_path in path.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.name.startswith("."):
                continue
            if file_path.suffix in skip_suffixes:
                continue
            replacements = find_replacements_in_file(file_path, old, new)
            if replacements:
                all_replacements.extend(replacements)
                files_to_modify[file_path] = True
    else:
        replacements = find_replacements_in_file(path, old, new, coords["line_start"], coords["line_end"])
        if replacements:
            all_replacements.extend(replacements)
            files_to_modify[path] = True
    
    if not all_replacements:
        return True, f"No matches for '{old}' in {path}"
    
    # Perform replacements
    backups = []
    modified_files = 0
    total_replacements = 0
    
    for file_path in files_to_modify:
        # Backup
        backup_path = backup_file(file_path)
        backups.append(backup_path)
        
        # Read content
        content = file_path.read_text(encoding="utf-8", errors="replace")
        
        # If line-scoped, only replace in those lines
        if coords["line_start"] is not None and file_path == path:
            lines = content.splitlines(keepends=True)
            for i in range(coords["line_start"] - 1, min(coords["line_end"] or len(lines), len(lines))):
                count = lines[i].count(old)
                if count:
                    lines[i] = lines[i].replace(old, new)
                    total_replacements += count
            content = "".join(lines)
        else:
            count = content.count(old)
            total_replacements += count
            content = content.replace(old, new)
        
        # Write
        file_path.write_text(content, encoding="utf-8")
        modified_files += 1
    
    out = [
        f"[STRIKE] Completed.",
        f"  Replacements: {total_replacements}",
        f"  Files modified: {modified_files}",
        f"  Backups: {BACKUP_DIR}",
    ]
    
    return True, "\n".join(out)


# --- MCP Tools ---

if mcp:

    @mcp.tool()
    async def target_item_assess(path: str, old_text: str, new_text: str, line_start: int = None, line_end: int = None) -> str:
        """Preview find/replace changes. Returns diff of proposed changes."""
        coords = {"path": path, "line_start": line_start, "line_end": line_end}
        success, output = assess(coords, old_text, new_text)
        return output if success else f"[FAIL] {output}"

    @mcp.tool()
    async def target_item_strike(path: str, old_text: str, new_text: str, line_start: int = None, line_end: int = None) -> str:
        """Execute find/replace. Creates backup and modifies files."""
        coords = {"path": path, "line_start": line_start, "line_end": line_end}
        success, output = strike(coords, old_text, new_text)
        return output if success else f"[FAIL] {output}"


def main():
    args = sys.argv[1:]
    
    if not args or args[0] in ["-h", "--help"]:
        print(__doc__.strip())
        sys.exit(0)
    
    if args[0] in ["-v", "--version"]:
        print(f"sfa_target_item {__version__}")
        sys.exit(0)
    
    if args[0] == "mcp" and mcp:
        mcp.run()
        return
    
    if len(args) < 4:
        print("Usage: sfa_target_item <assess|strike> <coords> <old> <new>", file=sys.stderr)
        print("Run with --help for details.", file=sys.stderr)
        sys.exit(1)
    
    action = args[0]
    coords_str = args[1]
    old_content = args[2]
    new_content = args[3]
    
    if action not in ["assess", "strike"]:
        print(f"Unknown action: {action}. Use 'assess' or 'strike'.", file=sys.stderr)
        sys.exit(1)
    
    # Parse coordinates
    coords = parse_coordinates(coords_str)
    
    # Read content
    try:
        old = read_content(old_content)
        new = read_content(new_content)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    
    # Execute
    if action == "assess":
        success, output = assess(coords, old, new)
    else:
        success, output = strike(coords, old, new)
    
    print(output)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
