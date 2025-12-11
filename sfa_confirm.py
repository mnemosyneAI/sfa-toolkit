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
sfa_confirm.py - Verification tool (BDA - Battle Damage Assessment)

USAGE:
  sfa_confirm <coords> [expected]

COORDINATES:
  /path/to/file.py            Check file exists and is readable
  /path/to/file.py:42         Show line 42
  /path/to/file.py:42-58      Show lines 42-58

EXPECTED (optional):
  "literal string"            Verify location contains this
  @filename                   Read expected from file

EXAMPLES:
  sfa_confirm /src/auth.py                     # Verify file exists
  sfa_confirm /src/auth.py:142                 # Show line 42
  sfa_confirm /src/auth.py:142 "fixed_func"   # Verify line contains text
  sfa_confirm /src/auth.py "new_api"          # Verify file contains text

OUTPUT:
  ✓ Success with details
  ✗ Failure with explanation
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

__version__ = "1.1.0"

# MCP Support
try:
    from fastmcp import FastMCP
    mcp = FastMCP("sfa-confirm")
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


def confirm(coords: dict, expected: Optional[str] = None) -> Tuple[bool, str]:
    """Verify the location and optionally check for expected content."""
    path = Path(coords["path"]).resolve()
    
    # Check existence
    if not path.exists():
        return False, f"✗ File not found: {path}"
    
    if not path.is_file():
        return False, f"✗ Not a file: {path}"
    
    # Read file
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return False, f"✗ Cannot read: {e}"
    
    lines = content.splitlines()
    total = len(lines)
    
    # If line coords specified
    if coords["line_start"] is not None:
        line_start = coords["line_start"]
        line_end = coords["line_end"] if coords["line_end"] else line_start
        
        if line_start < 1 or line_start > total:
            return False, f"✗ Line {line_start} out of range (1-{total})"
        if line_end > total:
            line_end = total
        
        # Get target lines
        target_lines = lines[line_start - 1:line_end]
        target_content = "\n".join(target_lines)
        
        # If expected content provided, verify
        if expected:
            if expected in target_content:
                out = [
                    f"✓ Confirmed: '{expected}' found at {path}:{line_start}",
                    "",
                ]
                for i, line in enumerate(target_lines, line_start):
                    marker = ">>>" if expected in line else "   "
                    out.append(f"{marker} {i:>4}│{line}")
                return True, "\n".join(out)
            else:
                out = [
                    f"✗ Not found: '{expected}' not in {path}:{line_start}-{line_end}",
                    "",
                    "Actual content:",
                ]
                for i, line in enumerate(target_lines, line_start):
                    out.append(f"    {i:>4}│{line}")
                return False, "\n".join(out)
        
        # No expected - just show the lines
        out = [f"✓ {path}:{line_start}" + (f"-{line_end}" if line_end != line_start else ""), ""]
        for i, line in enumerate(target_lines, line_start):
            out.append(f"    {i:>4}│{line}")
        return True, "\n".join(out)
    
    # File level - no line coords
    if expected:
        if expected in content:
            # Find where it appears
            for i, line in enumerate(lines, 1):
                if expected in line:
                    out = [
                        f"✓ Confirmed: '{expected}' found in {path}",
                        f"",
                        f">>> {i:>4}│{line}",
                    ]
                    return True, "\n".join(out)
        else:
            return False, f"✗ Not found: '{expected}' not in {path}"
    
    # No expected - just confirm file exists and show stats
    out = [
        f"✓ {path}",
        f"  Lines: {total}",
        f"  Size: {path.stat().st_size} bytes",
    ]
    return True, "\n".join(out)


# --- MCP Tools ---

if mcp:

    @mcp.tool()
    async def confirm_file(file_path: str) -> str:
        """Verify file exists and return stats."""
        coords = {"path": file_path, "line_start": None, "line_end": None}
        success, output = confirm(coords)
        return output

    @mcp.tool()
    async def confirm_line(file_path: str, line: int) -> str:
        """Show specific line content."""
        coords = {"path": file_path, "line_start": line, "line_end": line}
        success, output = confirm(coords)
        return output

    @mcp.tool()
    async def confirm_range(file_path: str, start_line: int, end_line: int) -> str:
        """Show line range content."""
        coords = {"path": file_path, "line_start": start_line, "line_end": end_line}
        success, output = confirm(coords)
        return output

    @mcp.tool()
    async def confirm_contains(file_path: str, expected: str, line: int = None) -> str:
        """Verify file or line contains expected text. Returns success/failure with context."""
        coords = {"path": file_path, "line_start": line, "line_end": line}
        success, output = confirm(coords, expected)
        return output


def main():
    args = sys.argv[1:]
    
    if not args or args[0] in ["-h", "--help"]:
        print(__doc__.strip())
        sys.exit(0)
    
    if args[0] in ["-v", "--version"]:
        print(f"sfa_confirm {__version__}")
        sys.exit(0)
    
    if args[0] == "mcp" and mcp:
        mcp.run()
        return
    
    if len(args) < 1:
        print("Usage: sfa_confirm <coords> [expected]", file=sys.stderr)
        print("Run with --help for details.", file=sys.stderr)
        sys.exit(1)
    
    coords_str = args[0]
    expected_content = args[1] if len(args) > 1 else None
    
    # Parse coordinates
    coords = parse_coordinates(coords_str)
    
    # Read expected content if provided
    expected = None
    if expected_content:
        try:
            expected = read_content(expected_content)
        except FileNotFoundError as e:
            print(str(e), file=sys.stderr)
            sys.exit(1)
    
    # Execute
    success, output = confirm(coords, expected)
    
    print(output)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
