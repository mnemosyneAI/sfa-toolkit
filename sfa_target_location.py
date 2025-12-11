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
sfa_target_location.py - Positional replacement/delete/insert/append

USAGE:
  sfa_target_location assess <coords> <new>              # Preview replace
  sfa_target_location strike <coords> <new>              # Execute replace
  sfa_target_location assess <coords> --delete           # Preview delete
  sfa_target_location strike <coords> --delete           # Execute delete
  sfa_target_location assess <coords> --insert <new>     # Preview insert (before)
  sfa_target_location strike <coords> --insert <new>     # Execute insert (before)
  sfa_target_location assess <coords> --append <new>     # Preview append (after)
  sfa_target_location strike <coords> --append <new>     # Execute append (after)

COORDINATES:
  /path/to/file.py:42         Line 42
  /path/to/file.py:42-58      Lines 42-58
  /path/to/file.py:42:15-28   Chars 15-28 on line 42

CONTENT:
  "literal string"            Inline content
  @filename                   Read from file

OPERATIONS:
  (default)                   Replace location with new content
  --delete                    Remove location entirely (clean, no orphan lines)
  --insert                    Insert before location (existing content shifts down)
  --append                    Insert after location (append to end: use last line number)

EXAMPLES:
  sfa_target_location assess /src/auth.py:142 "new line"
  sfa_target_location strike /src/auth.py:142-158 @block.txt
  sfa_target_location strike /src/auth.py:142-145 --delete
  sfa_target_location strike /src/auth.py:142 --insert "# New comment"
  sfa_target_location strike /src/auth.py:500 --append "# Added at end"
"""

import sys
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

__version__ = "1.2.0"

BACKUP_DIR = Path.home() / ".sfa" / "backups"

# MCP Support
try:
    from fastmcp import FastMCP
    mcp = FastMCP("sfa-target-location")
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


def assess(coords: dict, new_content: Optional[str], mode: str = "replace") -> Tuple[bool, str]:
    """Preview the operation (dry run)."""
    path = Path(coords["path"]).resolve()
    
    if not path.exists():
        return False, f"File not found: {path}"
    if not path.is_file():
        return False, f"Not a file: {path}"
    
    if coords["line_start"] is None:
        return False, f"Line number required. Use: {path}:LINE"
    
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return False, f"Cannot read: {e}"
    
    lines = content.splitlines()
    total = len(lines)
    
    line_start = coords["line_start"]
    line_end = coords["line_end"] if coords["line_end"] else line_start
    
    # For insert mode, allow positioning at total+1 to insert at very end
    max_line = total if mode not in ("insert",) else total + 1
    if line_start < 1 or line_start > max_line:
        return False, f"Line {line_start} out of range (1-{max_line})"
    if line_end > total:
        line_end = total
    
    # Build diff based on mode
    out = [f"--- {path}", f"+++ {path}", f"@@ lines {line_start}-{line_end} ({mode}) @@"]
    
    if mode == "delete":
        for i in range(line_start - 1, line_end):
            out.append(f"-{lines[i]}")
        out.append("")
        out.append(f"[ASSESS] {line_end - line_start + 1} line(s) will be deleted. Run 'strike' to apply.")
    
    elif mode == "insert":
        # Show context and where insertion will happen
        if line_start > 1:
            out.append(f" {lines[line_start - 2]}")
        out.append(f"+{new_content.splitlines()[0] if new_content else ''}")
        if new_content and len(new_content.splitlines()) > 1:
            for line in new_content.splitlines()[1:]:
                out.append(f"+{line}")
        if line_start <= total:
            out.append(f" {lines[line_start - 1]}")
        out.append("")
        out.append(f"[ASSESS] Content will be inserted before line {line_start}. Run 'strike' to apply.")
    
    elif mode == "append":
        # Show context and where append will happen
        out.append(f" {lines[line_start - 1]}")
        out.append(f"+{new_content.splitlines()[0] if new_content else ''}")
        if new_content and len(new_content.splitlines()) > 1:
            for line in new_content.splitlines()[1:]:
                out.append(f"+{line}")
        if line_start < total:
            out.append(f" {lines[line_start]}")
        out.append("")
        out.append(f"[ASSESS] Content will be appended after line {line_start}. Run 'strike' to apply.")
    
    else:  # replace
        # Show old lines
        for i in range(line_start - 1, line_end):
            old_line = lines[i]
            
            # Handle char-level replacement
            if coords["char_start"] is not None and i == line_start - 1:
                char_start = coords["char_start"]
                char_end = coords["char_end"] if coords["char_end"] else char_start + len(new_content or "")
                new_line = old_line[:char_start] + (new_content or "") + old_line[char_end:]
                out.append(f"-{old_line}")
                out.append(f"+{new_line}")
            else:
                out.append(f"-{old_line}")
        
        # Show new content (if not char-level)
        if coords["char_start"] is None and new_content:
            for line in new_content.splitlines():
                out.append(f"+{line}")
        
        out.append("")
        out.append(f"[ASSESS] Ready to replace. Run 'strike' to apply.")
    
    return True, "\n".join(out)


def strike(coords: dict, new_content: Optional[str], mode: str = "replace") -> Tuple[bool, str]:
    """Execute the operation."""
    path = Path(coords["path"]).resolve()
    
    if not path.exists():
        return False, f"File not found: {path}"
    if not path.is_file():
        return False, f"Not a file: {path}"
    
    if coords["line_start"] is None:
        return False, f"Line number required. Use: {path}:LINE"
    
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return False, f"Cannot read: {e}"
    
    lines = content.splitlines(keepends=True)
    
    # Ensure last line has newline for consistency
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    
    total = len(lines)
    
    line_start = coords["line_start"]
    line_end = coords["line_end"] if coords["line_end"] else line_start
    
    # For insert mode, allow positioning at total+1 to append at very end
    max_line = total if mode not in ("insert",) else total + 1
    if line_start < 1 or line_start > max_line:
        return False, f"Line {line_start} out of range (1-{max_line})"
    if line_end > total:
        line_end = total
    
    # Backup
    backup_path = backup_file(path)
    
    # Perform operation
    if mode == "delete":
        # Delete lines cleanly
        lines = lines[:line_start - 1] + lines[line_end:]
        op_desc = f"Deleted lines {line_start}-{line_end}"
    
    elif mode == "insert":
        # Insert before line_start
        insert_content = new_content if new_content else ""
        if not insert_content.endswith("\n"):
            insert_content += "\n"
        lines = lines[:line_start - 1] + [insert_content] + lines[line_start - 1:]
        op_desc = f"Inserted before line {line_start}"
    
    elif mode == "append":
        # Insert after line_start (line_start is the line AFTER which we insert)
        append_content = new_content if new_content else ""
        if not append_content.endswith("\n"):
            append_content += "\n"
        lines = lines[:line_start] + [append_content] + lines[line_start:]
        op_desc = f"Appended after line {line_start}"
    
    else:  # replace
        if coords["char_start"] is not None:
            # Char-level replacement on single line
            old_line = lines[line_start - 1]
            char_start = coords["char_start"]
            char_end = coords["char_end"] if coords["char_end"] else char_start + len(new_content or "")
            
            # Preserve line ending
            line_ending = ""
            if old_line.endswith("\n"):
                line_ending = "\n"
                old_line = old_line[:-1]
            
            new_line = old_line[:char_start] + (new_content or "") + old_line[char_end:] + line_ending
            lines[line_start - 1] = new_line
            op_desc = f"Replaced chars {char_start}-{char_end} on line {line_start}"
        else:
            # Line-level replacement
            replace_content = new_content if new_content else ""
            if replace_content and not replace_content.endswith("\n"):
                replace_content += "\n"
            
            if replace_content:
                # Split replacement into lines
                new_lines = []
                for line in replace_content.splitlines(keepends=True):
                    if not line.endswith("\n"):
                        line += "\n"
                    new_lines.append(line)
                lines = lines[:line_start - 1] + new_lines + lines[line_end:]
            else:
                # Empty replacement = delete
                lines = lines[:line_start - 1] + lines[line_end:]
            
            op_desc = f"Replaced lines {line_start}-{line_end}"
    
    # Write
    path.write_text("".join(lines), encoding="utf-8")
    
    out = [
        f"[STRIKE] Completed.",
        f"  Operation: {op_desc}",
        f"  Backup: {backup_path}",
    ]
    
    return True, "\n".join(out)


# --- MCP Tools ---

if mcp:

    @mcp.tool()
    async def target_location_replace(file_path: str, line: int, new_content: str, line_end: int = None) -> str:
        """Replace line(s) at position. Creates backup."""
        coords = {"path": file_path, "line_start": line, "line_end": line_end, "char_start": None, "char_end": None}
        success, output = strike(coords, new_content, "replace")
        return output if success else f"[FAIL] {output}"

    @mcp.tool()
    async def target_location_delete(file_path: str, line: int, line_end: int = None) -> str:
        """Delete line(s) at position. Creates backup."""
        coords = {"path": file_path, "line_start": line, "line_end": line_end, "char_start": None, "char_end": None}
        success, output = strike(coords, None, "delete")
        return output if success else f"[FAIL] {output}"

    @mcp.tool()
    async def target_location_insert(file_path: str, line: int, content: str) -> str:
        """Insert content before line. Creates backup."""
        coords = {"path": file_path, "line_start": line, "line_end": None, "char_start": None, "char_end": None}
        success, output = strike(coords, content, "insert")
        return output if success else f"[FAIL] {output}"

    @mcp.tool()
    async def target_location_append(file_path: str, line: int, content: str) -> str:
        """Append content after line. For end of file, use last line number. Creates backup."""
        coords = {"path": file_path, "line_start": line, "line_end": None, "char_start": None, "char_end": None}
        success, output = strike(coords, content, "append")
        return output if success else f"[FAIL] {output}"

    @mcp.tool()
    async def target_location_assess(file_path: str, line: int, operation: str, content: str = None, line_end: int = None) -> str:
        """Preview operation without executing. operation: replace|delete|insert|append"""
        coords = {"path": file_path, "line_start": line, "line_end": line_end, "char_start": None, "char_end": None}
        success, output = assess(coords, content, operation)
        return output if success else f"[FAIL] {output}"


def main():
    args = sys.argv[1:]
    
    if not args or args[0] in ["-h", "--help"]:
        print(__doc__.strip())
        sys.exit(0)
    
    if args[0] in ["-v", "--version"]:
        print(f"sfa_target_location {__version__}")
        sys.exit(0)
    
    if args[0] == "mcp" and mcp:
        mcp.run()
        return
    
    if len(args) < 2:
        print("Usage: sfa_target_location <assess|strike> <coords> [content|--delete|--insert content|--append content]", file=sys.stderr)
        print("Run with --help for details.", file=sys.stderr)
        sys.exit(1)
    
    action = args[0]
    coords_str = args[1]
    
    if action not in ["assess", "strike"]:
        print(f"Unknown action: {action}. Use 'assess' or 'strike'.", file=sys.stderr)
        sys.exit(1)
    
    # Parse mode and content from remaining args
    mode = "replace"
    new_content = None
    
    remaining = args[2:]
    
    if not remaining:
        print("Content required. Use 'content', '--delete', '--insert content', or '--append content'.", file=sys.stderr)
        sys.exit(1)
    
    if remaining[0] == "--delete":
        mode = "delete"
        new_content = None
    elif remaining[0] == "--insert":
        mode = "insert"
        if len(remaining) < 2:
            print("--insert requires content.", file=sys.stderr)
            sys.exit(1)
        try:
            new_content = read_content(remaining[1])
        except FileNotFoundError as e:
            print(str(e), file=sys.stderr)
            sys.exit(1)
    elif remaining[0] == "--append":
        mode = "append"
        if len(remaining) < 2:
            print("--append requires content.", file=sys.stderr)
            sys.exit(1)
        try:
            new_content = read_content(remaining[1])
        except FileNotFoundError as e:
            print(str(e), file=sys.stderr)
            sys.exit(1)
    else:
        # Replace mode
        try:
            new_content = read_content(remaining[0])
        except FileNotFoundError as e:
            print(str(e), file=sys.stderr)
            sys.exit(1)
        # Empty string means delete
        if new_content == "":
            mode = "delete"
            new_content = None
    
    # Parse coordinates
    coords = parse_coordinates(coords_str)
    
    # Execute
    if action == "assess":
        success, output = assess(coords, new_content, mode)
    else:
        success, output = strike(coords, new_content, mode)
    
    print(output)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
