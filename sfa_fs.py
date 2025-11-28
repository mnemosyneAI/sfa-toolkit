#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "fastmcp"
# ]
# ///

"""
SFA Filesystem Tool (sfa_fs.py) v2.2.0
Professional filesystem operations with Unix philosophy design.

PHILOSOPHY:
  STDOUT: Data (composable with pipes)
  STDERR: Context (controlled by -v/-vv)
  EXIT:   0=success, 1=failure

COMMANDS:
  Filesystem:
    create <path>              Create directory (--parents for recursive)
    verify <path>              Check path exists, return type
    list <path>                List contents (--mode tree|detailed)
    delete <path>              Delete with backup (-r for recursive)
    move <src> <dst>           Move/rename
    copy <src> <dst>           Copy (-r for recursive)
    tree <path>                Tree view (--depth N)
    find <path>                Find by pattern (--pattern "*.py")

  Reading (replaces Read/Grep tools):
    read <file>                Read with line numbers (--offset, --limit, --raw)
    search <path> <pattern>    Regex search (-i case, -C context, --files-only, --count)
    line-get <file> <range>    Get lines: 10, 10-20, 10-, -20

  Content Manipulation:
    line-replace <f> <n> <txt> Replace line N with text
    line-insert <f> <n> <txt>  Insert before line N
    line-delete <f> <n>        Delete line N
    line-delete-range <f> <r>  Delete range: 10-20, 10-, -20
    line-append <f> <txt>      Append to file
    line-prepend <f> <txt>     Prepend to file
    block-replace <f> <o> <n>  Replace old with new
                               --unique: fail if >1 match (like Edit tool)
                               --count:  count matches only (no modify)
                               --preview: show diff before applying
    char-replace <f> <o> <n>   Character-level replace

  Analysis:
    read-imports <file>        Extract import statements
    read-functions <file>      Extract function signatures
    read-docstrings <file>     Extract docstrings

  Processing:
    sort <file>                Sort lines
    unique <file>              Remove duplicate lines

  Not yet in CLI (functions exist, parsers pending):
    char-replace               Character-level replace
    extract-markdown           Extract markdown from .ymj

VERBOSITY:
  (default) - Silent operation, data only
  -v        - Show operation context on STDERR
  -vv       - Show full BDA (Before-During-After) on STDERR

WORKFLOW:
  Every operation: FIND → CHECK → TARGET → EXECUTE → CONFIRM → OUTPUT

Ground truth always. No filtering. No performance.
"""

# ARCHITECTURE:
#   Layer 2 (Primitives): _run_bash, _check_exists, _backup_file, _resolve_path
#   Layer 1 (Operations): read_file, search, block_replace, list_directory, ...
#   Layer 0 (Interface):  CLI parsers (argparse), MCP tools (fastmcp)

import asyncio
import sys
import os
import json
import subprocess
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from datetime import datetime

try:
    from fastmcp import FastMCP

    mcp = FastMCP("sfa-fs")
except ImportError:
    mcp = None

# --- Configuration ---
__version__ = "2.1.0"
GIT_BASH = os.getenv("SFA_GITBASH_PATH", r"C:\Program Files\Git\bin\bash.exe")
PLATFORM = sys.platform
BACKUP_DIR = Path.home() / ".sfa" / "backups"

# Verbosity levels (set by CLI)
VERBOSITY = 0  # 0=silent, 1=-v (context), 2=-vv (full BDA)

# --- Core Primitives (Layer 2) ---


def _log(message: str, level: int = 1):
    """
    Log message to STDERR at given verbosity level.
    level 1 = -v (context)
    level 2 = -vv (full BDA)
    """
    global VERBOSITY
    if VERBOSITY >= level:
        print(message, file=sys.stderr)


def _run_bash(command: str, cwd: Optional[str] = None) -> Tuple[bool, str, str, int]:
    """
    Execute command via Git Bash for ground truth.
    Returns (success, stdout, stderr, exit_code)
    """
    if not os.path.exists(GIT_BASH):
        # Fallback to native commands if Git Bash not available
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            return (
                result.returncode == 0,
                result.stdout,
                result.stderr,
                result.returncode,
            )
        except Exception as e:
            return (False, "", str(e), 1)

    try:
        creation_flags = 0
        if PLATFORM == "win32":
            creation_flags = subprocess.CREATE_NO_WINDOW

        result = subprocess.run(
            [GIT_BASH, "-c", command],
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=creation_flags,
        )

        return (result.returncode == 0, result.stdout, result.stderr, result.returncode)
    except Exception as e:
        return (False, "", str(e), 1)


def _check_exists(path: str) -> Tuple[bool, Optional[str]]:
    """
    Check if path exists and what type it is.
    Returns (exists, type) where type is 'file', 'dir', or None
    """
    path_obj = Path(path)
    if not path_obj.exists():
        return (False, None)

    if path_obj.is_file():
        return (True, "file")
    elif path_obj.is_dir():
        return (True, "dir")
    else:
        return (True, "other")


def _get_details(path: str) -> Optional[Dict[str, Any]]:
    """
    Get detailed information about a path using ls -la.
    Returns None if path doesn't exist.
    """
    if not Path(path).exists():
        return None

    # Get parent directory for ls command
    path_obj = Path(path).resolve()
    parent = path_obj.parent
    name = path_obj.name

    success, stdout, stderr, _ = _run_bash(f'ls -la "{parent}"')

    if not success:
        return None

    # Parse ls output to find our file/dir
    for line in stdout.strip().split("\n"):
        if name in line:
            return {
                "path": str(path_obj),
                "name": name,
                "parent": str(parent),
                "ls_output": line,
                "exists": True,
            }

    return None


def _list_directory(path: str, mode: str = "tree") -> str:
    """
    List directory contents with different modes.
    mode: 'tree' (ls -R), 'flat' (find), 'detailed' (ls -la)
    """
    if mode == "tree":
        success, stdout, stderr, _ = _run_bash(f'ls -R "{path}"')
        return stdout if success else f"Error: {stderr}"

    elif mode == "flat":
        success, stdout, stderr, _ = _run_bash(f'find "{path}" -type f -o -type d')
        return stdout if success else f"Error: {stderr}"

    elif mode == "detailed":
        success, stdout, stderr, _ = _run_bash(f'ls -la "{path}"')
        return stdout if success else f"Error: {stderr}"

    else:
        return f"Error: Unknown mode '{mode}'"


def _determine_create_command(path: str, parents: bool = False) -> Tuple[str, str]:
    """
    Determine correct command to create path.
    Since we use Git Bash on Windows, always use Unix commands.
    Returns (command, explanation)
    """
    path_obj = Path(path).resolve()

    # Check if parent exists
    parent_exists = path_obj.parent.exists()

    if not parent_exists and not parents:
        return ("", f"Parent directory does not exist. Use --parents to create it.")

    # Always use Unix mkdir since we execute via Git Bash
    if parents:
        return (f'mkdir -p "{path}"', "Unix mkdir -p creates parent directories")
    else:
        return (f'mkdir "{path}"', "Unix mkdir command")


def _backup_file(path: str) -> Optional[str]:
    """
    Create backup of file to ~/.sfa/backups with timestamp.
    Returns backup path on success, None on failure.
    """
    if not Path(path).exists():
        return None

    # Ensure backup directory exists
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    # Generate backup filename with timestamp
    path_obj = Path(path).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{path_obj.name}.{timestamp}.backup"
    backup_path = BACKUP_DIR / backup_name

    # Copy file
    try:
        import shutil

        shutil.copy2(str(path_obj), str(backup_path))
        return str(backup_path)
    except Exception as e:
        return None


# --- High-Level Operations (Layer 1) ---


def create(path: str, parents: bool = False) -> Tuple[bool, str, str]:
    """
    Create directory with full BDA verification.

    Returns: (success, stdout_data, error_message)
    Action command: silent on success, outputs nothing to STDOUT
    """
    path_obj = Path(path).resolve()

    _log(f"Creating {path}", level=2)
    _log(f"Resolved: {path_obj}", level=2)

    # CHECK STATE (Before)
    exists, type_before = _check_exists(str(path_obj))
    parent_exists, _ = _check_exists(str(path_obj.parent))

    _log(f"Before: exists={exists}, parent_exists={parent_exists}", level=2)

    if exists:
        return (False, "", f"Path already exists: {path} (Resolved: {path_obj})")

    # DETERMINE COMMAND
    command, explanation = _determine_create_command(str(path_obj), parents)

    if not command:
        return (False, "", explanation)

    _log(f"Command: {command} ({explanation})", level=2)

    # EXECUTE
    success, stdout, stderr, exit_code = _run_bash(command)

    if not success:
        return (False, "", f"Command failed: {stderr}")

    # CONFIRM (After)
    exists_after, type_after = _check_exists(str(path_obj))

    _log(f"After: exists={exists_after}, type={type_after}", level=2)

    if not exists_after:
        return (False, "", "Path does not exist after creation command (unexpected)")

    _log(f"Created: {path}", level=1)

    return (True, "", "")


def verify(path: str) -> Tuple[bool, str, str]:
    """
    Verify path exists and show detailed information.

    Returns: (success, stdout_data, error_message)
    Outputs to STDOUT: "exists <type>" or "missing"
    """
    path_obj = Path(path).resolve()

    _log(f"Verifying {path}", level=2)

    exists, type_found = _check_exists(str(path_obj))

    if exists:
        details = _get_details(str(path_obj))
        _log(f"Path exists: {path} (type={type_found})", level=1)
        if details and VERBOSITY >= 2:
            _log(f"Details: {details['ls_output']}", level=2)
        return (True, f"exists {type_found}", "")
    else:
        return (False, "missing", f"Path does not exist: {path} (Resolved: {path_obj})")


def list_path(path: str, mode: str = "tree") -> Tuple[bool, str, str]:
    """
    List directory contents with ground truth verification.
    mode: 'tree', 'flat', or 'detailed'

    Returns: (success, stdout_data, error_message)
    """
    path_obj = Path(path).resolve()

    _log(f"Listing {path} (mode: {mode})", level=2)

    exists, type_found = _check_exists(str(path_obj))

    if not exists:
        return (False, "", f"Path does not exist: {path} (Resolved: {path_obj})")

    if type_found != "dir":
        return (False, "", f"Path is not a directory: {path}")

    _log(f"Verified: {path} exists (type={type_found})", level=2)

    contents = _list_directory(str(path_obj), mode)

    _log(f"Found {len(contents.splitlines())} items", level=1)

    return (True, contents, "")


def delete(
    path: str, recursive: bool = False, backup: bool = True
) -> Tuple[bool, str, str]:
    """
    Delete file or directory with optional backup.

    Returns: (success, stdout_data, error_message)
    Action command: silent on success
    """
    path_obj = Path(path).resolve()

    _log(f"Deleting {path}", level=2)

    # Check state before
    exists, type_before = _check_exists(str(path_obj))

    if not exists:
        return (False, "", f"Path does not exist: {path} (Resolved: {path_obj})")

    _log(f"Before: exists={exists}, type={type_before}", level=2)

    # Backup if requested and it's a file
    backup_path = None
    if backup and type_before == "file":
        backup_path = _backup_file(str(path_obj))
        if not backup_path:
            return (False, "", "Failed to create backup")
        _log(f"Backup: {backup_path}", level=1)

    # Determine command
    if type_before == "file":
        command = f'rm "{path}"'
    elif type_before == "dir":
        if recursive:
            command = f'rm -rf "{path}"'
        else:
            command = f'rmdir "{path}"'
    else:
        return (False, "", f"Cannot delete path of type: {type_before}")

    _log(f"Command: {command}", level=2)

    # Execute
    success, stdout, stderr, exit_code = _run_bash(command)

    if not success:
        return (False, "", f"Command failed: {stderr}")

    # Verify deletion
    exists_after, type_after = _check_exists(str(path_obj))

    if exists_after:
        return (False, "", "Path still exists after deletion (unexpected)")

    _log(f"Deleted: {path}", level=1)

    return (True, "", "")


def move(source: str, destination: str, backup: bool = True) -> Tuple[bool, str, str]:
    """
    Move/rename file or directory with optional backup.

    Returns: (success, stdout_data, error_message)
    Action command: silent on success
    """
    source_obj = Path(source).resolve()
    dest_obj = Path(destination).resolve()

    _log(f"Moving {source} → {destination}", level=2)
    _log(f"Resolved: {source_obj} → {dest_obj}", level=2)

    # Check source exists
    src_exists, src_type = _check_exists(str(source_obj))
    dest_exists, dest_type = _check_exists(str(dest_obj))

    _log(
        f"Before: src_exists={src_exists} (type={src_type}), dest_exists={dest_exists}",
        level=2,
    )

    if not src_exists:
        return (False, "", f"Source does not exist: {source} (Resolved: {source_obj})")

    # Backup destination if it exists and is a file
    if backup and dest_exists and dest_type == "file":
        backup_path = _backup_file(str(dest_obj))
        if not backup_path:
            return (False, "", "Failed to create backup of destination")
        _log(f"Backup: {backup_path}", level=1)

    # Execute move (use original paths for Git Bash)
    command = f'mv "{source}" "{destination}"'
    _log(f"Command: {command}", level=2)

    success, stdout, stderr, exit_code = _run_bash(command)

    if not success:
        return (False, "", f"Command failed: {stderr}")

    # Verify
    src_exists_after, _ = _check_exists(str(source_obj))
    dest_exists_after, dest_type_after = _check_exists(str(dest_obj))

    _log(
        f"After: src_exists={src_exists_after}, dest_exists={dest_exists_after}",
        level=2,
    )

    if src_exists_after:
        return (False, "", "Source still exists after move")

    if not dest_exists_after:
        return (False, "", "Destination does not exist after move")

    _log(f"Moved: {source} → {destination}", level=1)

    return (True, "", "")


def copy(
    source: str, destination: str, recursive: bool = False
) -> Tuple[bool, str, str]:
    """
    Copy file or directory.

    Returns: (success, stdout_data, error_message)
    Action command: silent on success
    """
    source_obj = Path(source).resolve()
    dest_obj = Path(destination).resolve()

    _log(f"Copying {source} → {destination}", level=2)
    _log(f"Resolved: {source_obj} → {dest_obj}", level=2)

    # Check source exists
    src_exists, src_type = _check_exists(str(source_obj))
    dest_exists, dest_type = _check_exists(str(dest_obj))

    _log(
        f"Before: src_exists={src_exists} (type={src_type}), dest_exists={dest_exists}",
        level=2,
    )

    if not src_exists:
        return (False, "", f"Source does not exist: {source} (Resolved: {source_obj})")

    # Determine command (use original paths for Git Bash)
    if src_type == "dir" and recursive:
        command = f'cp -r "{source}" "{destination}"'
    elif src_type == "file":
        command = f'cp "{source}" "{destination}"'
    else:
        return (False, "", "Cannot copy directory without --recursive flag")

    _log(f"Command: {command}", level=2)

    # Execute
    success, stdout, stderr, exit_code = _run_bash(command)

    if not success:
        return (False, "", f"Command failed: {stderr}")

    # Verify
    dest_exists_after, dest_type_after = _check_exists(str(dest_obj))

    _log(f"After: dest_exists={dest_exists_after} (type={dest_type_after})", level=2)

    if not dest_exists_after:
        return (False, "", "Destination does not exist after copy")

    _log(f"Copied: {source} → {destination}", level=1)

    return (True, "", "")


def tree(path: str, max_depth: Optional[int] = None) -> Tuple[bool, str, str]:
    """
    Display directory tree structure.

    Returns: (success, stdout_data, error_message)
    Query command: outputs tree structure to STDOUT
    """
    path_obj = Path(path).resolve()

    _log(f"Tree view of {path}", level=2)

    exists, type_found = _check_exists(str(path_obj))

    if not exists:
        return (False, "", f"Path does not exist: {path} (Resolved: {path_obj})")

    if type_found != "dir":
        return (False, "", f"Path is not a directory: {path}")

    _log(f"Verified: {path} is a directory", level=2)

    # Use tree command if available, otherwise ls -R
    if max_depth:
        command = f'tree -L {max_depth} "{path}" 2>/dev/null || ls -R "{path}"'
    else:
        command = f'tree "{path}" 2>/dev/null || ls -R "{path}"'

    _log(f"Command: {command}", level=2)

    success, stdout, stderr, _ = _run_bash(command)

    if not success:
        return (False, "", f"Command failed: {stderr}")

    _log(f"Tree output: {len(stdout.splitlines())} lines", level=1)

    return (True, stdout, "")


def find_files(
    path: str,
    pattern: Optional[str] = None,
    type_filter: Optional[str] = None,
) -> Tuple[bool, str, str]:
    """
    Find files matching pattern.

    Args:
        path: Directory to search
        pattern: Filename pattern (e.g., "*.py")
        type_filter: 'f' for files, 'd' for directories

    Returns: (success, stdout_data, error_message)
    Query command: outputs file paths to STDOUT (one per line)
    """
    path_obj = Path(path).resolve()

    _log(f"Finding files in {path}", level=2)
    if pattern:
        _log(f"Pattern: {pattern}", level=2)
    exists, type_found = _check_exists(str(path_obj))

    if not exists:
        return (False, "", f"Path does not exist: {path} (Resolved: {path_obj})")

    if type_found != "dir":
        return (False, "", f"Path is not a directory: {path}")

    # Build find command
    command = f'find "{path}"'

    if type_filter:
        command += f" -type {type_filter}"

    if pattern:
        command += f' -name "{pattern}"'

    _log(f"Command: {command}", level=2)

    success, stdout, stderr, _ = _run_bash(command)

    if not success:
        return (False, "", f"Command failed: {stderr}")

    # Output file list to STDOUT (one per line)
    file_list = stdout.strip()
    num_matches = len([line for line in file_list.split("\n") if line.strip()])

    _log(f"Found {num_matches} matches", level=1)

    return (True, file_list, "")


# --- Reading Commands ---


def read_file(
    file_path: str,
    offset: int = 0,
    limit: Optional[int] = None,
    raw: bool = False,
    encoding: str = "utf-8",
) -> Tuple[bool, str, str]:
    """Read file contents with line numbers."""
    path_obj = Path(file_path).resolve()
    _log(f"Reading {file_path}", level=2)

    exists, type_found = _check_exists(str(path_obj))
    if not exists:
        return (False, "", f"File does not exist: {file_path}")
    if type_found != "file":
        return (False, "", f"Path is not a file: {file_path}")

    try:
        with open(path_obj, "r", encoding=encoding, errors="replace") as f:
            lines = f.readlines()

        total_lines = len(lines)
        start_idx = offset
        if start_idx >= total_lines:
            return (True, "", "")

        end_idx = total_lines if limit is None else min(start_idx + limit, total_lines)
        selected_lines = lines[start_idx:end_idx]

        if raw:
            output = "".join(selected_lines)
        else:
            output_lines = []
            for i, line in enumerate(selected_lines, start=start_idx + 1):
                line_content = line.rstrip("\n")
                output_lines.append(f"{i:>6}→{line_content}")
            output = "\n".join(output_lines)

        _log(f"Read {len(selected_lines)} lines", level=1)
        return (True, output, "")
    except Exception as e:
        return (False, "", f"Read error: {str(e)}")


def search(
    path: str,
    pattern: str,
    before_context: int = 0,
    after_context: int = 0,
    context: int = 0,
    count_only: bool = False,
    files_only: bool = False,
    ignore_case: bool = False,
    recursive: bool = True,
) -> Tuple[bool, str, str]:
    """Search for regex pattern in file(s)."""
    import re
    path_obj = Path(path).resolve()
    _log(f"Searching {path} for: {pattern}", level=2)

    exists, type_found = _check_exists(str(path_obj))
    if not exists:
        return (False, "", f"Path does not exist: {path}")

    try:
        flags = re.IGNORECASE if ignore_case else 0
        regex = re.compile(pattern, flags)
    except re.error as e:
        return (False, "", f"Invalid regex: {e}")

    ctx_before = context if context > 0 else before_context
    ctx_after = context if context > 0 else after_context

    if type_found == "file":
        files_to_search = [path_obj]
    else:
        files_to_search = list(path_obj.rglob("*") if recursive else path_obj.glob("*"))
        files_to_search = [f for f in files_to_search if f.is_file()]

    results = []
    files_with_matches = set()
    total_matches = 0

    for fp in files_to_search:
        try:
            with open(fp, "r", encoding="utf-8", errors="replace") as f:
                file_lines = f.readlines()
        except:
            continue

        file_matches = [i for i, line in enumerate(file_lines) if regex.search(line)]
        if not file_matches:
            continue

        total_matches += len(file_matches)
        files_with_matches.add(fp)

        if count_only:
            results.append(f"{fp}:{len(file_matches)}")
        elif files_only:
            results.append(str(fp))
        else:
            shown = set()
            for idx in file_matches:
                for j in range(max(0, idx - ctx_before), min(len(file_lines), idx + ctx_after + 1)):
                    if j not in shown:
                        shown.add(j)
                        marker = ":" if j == idx else "-"
                        results.append(f"{fp}:{j+1}{marker}{file_lines[j].rstrip()}")

    if files_only:
        results = list(dict.fromkeys(results))

    _log(f"Found {total_matches} matches in {len(files_with_matches)} files", level=1)
    return (True, "\n".join(results), "")


def _parse_line_range(range_str: str, total_lines: int):
    """Parse line range: 10, 10-20, 10-, -20"""
    range_str = range_str.strip()
    try:
        if "-" not in range_str:
            n = int(range_str)
            return (n, n)
        parts = range_str.split("-", 1)
        if parts[0] == "":
            return (1, int(parts[1]))
        elif parts[1] == "":
            return (int(parts[0]), total_lines)
        else:
            start, end = int(parts[0]), int(parts[1])
            if start < 1 or end < start:
                return (None, None)
            return (start, min(end, total_lines))
    except ValueError:
        return (None, None)


def line_get(file_path: str, line_range: str) -> Tuple[bool, str, str]:
    """Get specific line(s) from file."""
    path_obj = Path(file_path).resolve()
    exists, type_found = _check_exists(str(path_obj))
    if not exists or type_found != "file":
        return (False, "", f"Not a file: {file_path}")

    try:
        with open(path_obj, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        start, end = _parse_line_range(line_range, len(lines))
        if start is None:
            return (False, "", f"Invalid range: {line_range}")
        if start > len(lines):
            return (False, "", f"Start {start} exceeds file length")

        selected = lines[start - 1 : end]
        output = "\n".join(f"{i:>6}→{line.rstrip()}" for i, line in enumerate(selected, start=start))
        return (True, output, "")
    except Exception as e:
        return (False, "", str(e))


def line_delete_range(file_path: str, line_range: str, backup: bool = True) -> Tuple[bool, str, str]:
    """Delete range of lines from file."""
    path_obj = Path(file_path).resolve()
    exists, type_found = _check_exists(str(path_obj))
    if not exists or type_found != "file":
        return (False, "", f"Not a file: {file_path}")

    if backup:
        bp = _backup_file(str(path_obj))
        if not bp:
            return (False, "", "Backup failed")
        _log(f"Backup: {bp}", level=1)

    try:
        with open(path_obj, "r", encoding="utf-8") as f:
            lines = f.readlines()

        start, end = _parse_line_range(line_range, len(lines))
        if start is None:
            return (False, "", f"Invalid range: {line_range}")

        del lines[start - 1 : end]
        with open(path_obj, "w", encoding="utf-8") as f:
            f.writelines(lines)

        _log(f"Deleted lines {line_range}", level=1)
        return (True, "", "")
    except Exception as e:
        return (False, "", str(e))


# --- Content Manipulation Commands ---


def char_replace(
    file_path: str,
    old_char: str,
    new_char: str,
    backup: bool = True,
) -> Tuple[bool, str, str]:
    """
    Replace all occurrences of a character.

    Returns: (success, stdout_data, error_message)
    Action command: silent on success
    """
    path_obj = Path(file_path).resolve()

    _log(f"Replacing '{old_char}' with '{new_char}' in {file_path}", level=2)

    exists, type_found = _check_exists(str(path_obj))

    if not exists or type_found != "file":
        return (False, "", f"File does not exist or is not a file: {file_path} (Resolved: {path_obj})")

    # Backup
    if backup:
        backup_path = _backup_file(str(path_obj))
        if not backup_path:
            return (False, "", "Failed to create backup")
        _log(f"Backup: {backup_path}", level=1)

    # Read content
    try:
        with open(path_obj, "r", encoding="utf-8") as f:
            content = f.read()

        # Count replacements
        replacements = content.count(old_char)
        _log(f"Found {replacements} occurrences", level=2)

        # Replace
        new_content = content.replace(old_char, new_char)

        # Write back
        with open(path_obj, "w", encoding="utf-8") as f:
            f.write(new_content)

        _log(f"Replaced {replacements} occurrence(s)", level=1)
        return (True, "", "")

    except Exception as e:
        return (False, "", str(e))


def line_replace(
    file_path: str,
    line_num: int,
    new_content: str,
    backup: bool = True,
) -> Tuple[bool, str, str]:
    """
    Replace entire line at given line number (1-indexed).

    Returns: (success, stdout_data, error_message)
    Action command: silent on success
    """
    path_obj = Path(file_path).resolve()

    exists, type_found = _check_exists(str(path_obj))

    if not exists or type_found != "file":
        return (False, "", f"File does not exist or is not a file: {file_path} (Resolved: {path_obj})")

    if backup:
        backup_path = _backup_file(str(path_obj))
        if not backup_path:
            return (False, "", "Failed to create backup")
        _log(f"Backup: {backup_path}", level=1)

    try:
        with open(path_obj, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if line_num < 1 or line_num > len(lines):
            return (False, "", f"Line number {line_num} out of range (1-{len(lines)})")

        old_content = lines[line_num - 1].rstrip("\n")
        _log(f"Old: {old_content}", level=2)

        lines[line_num - 1] = new_content + "\n"

        with open(path_obj, "w", encoding="utf-8") as f:
            f.writelines(lines)

        _log(f"Replaced line {line_num}", level=1)
        return (True, "", "")

    except Exception as e:
        return (False, "", str(e))


def line_insert(
    file_path: str,
    line_num: int,
    content: str,
    backup: bool = True,
) -> Tuple[bool, str, str]:
    """
    Insert new line at given position (1-indexed).

    Returns: (success, stdout_data, error_message)
    Action command: silent on success
    """
    path_obj = Path(file_path).resolve()

    exists, type_found = _check_exists(str(path_obj))

    if not exists or type_found != "file":
        return (False, "", f"File does not exist or is not a file: {file_path} (Resolved: {path_obj})")

    if backup:
        backup_path = _backup_file(str(path_obj))
        if not backup_path:
            return (False, "", "Failed to create backup")
        _log(f"Backup: {backup_path}", level=1)

    try:
        with open(path_obj, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if line_num < 1 or line_num > len(lines) + 1:
            return (
                False,
                "",
                f"Line number {line_num} out of range (1-{len(lines) + 1})",
            )

        lines.insert(line_num - 1, content + "\n")

        with open(path_obj, "w", encoding="utf-8") as f:
            f.writelines(lines)

        _log(f"Inserted line at position {line_num}", level=1)
        return (True, "", "")

    except Exception as e:
        return (False, "", str(e))


def line_delete(
    file_path: str, line_num: int, backup: bool = True
) -> Tuple[bool, str, str]:
    """
    Delete line at given position (1-indexed).

    Returns: (success, stdout_data, error_message)
    Action command: silent on success
    """
    path_obj = Path(file_path).resolve()

    exists, type_found = _check_exists(str(path_obj))

    if not exists or type_found != "file":
        return (False, "", f"File does not exist or is not a file: {file_path} (Resolved: {path_obj})")

    if backup:
        backup_path = _backup_file(str(path_obj))
        if not backup_path:
            return (False, "", "Failed to create backup")
        _log(f"Backup: {backup_path}", level=1)

    try:
        with open(path_obj, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if line_num < 1 or line_num > len(lines):
            return (False, "", f"Line number {line_num} out of range (1-{len(lines)})")

        deleted_content = lines[line_num - 1].rstrip("\n")
        _log(f"Deleted: {deleted_content}", level=2)

        del lines[line_num - 1]

        with open(path_obj, "w", encoding="utf-8") as f:
            f.writelines(lines)

        _log(f"Deleted line {line_num}", level=1)
        return (True, "", "")

    except Exception as e:
        return (False, "", str(e))


def line_append(
    file_path: str, content: str, backup: bool = True
) -> Tuple[bool, str, str]:
    """
    Append line to end of file.

    Returns: (success, stdout_data, error_message)
    Action command: silent on success
    """
    path_obj = Path(file_path).resolve()

    _log(f"Appending line to {file_path}", level=2)

    exists, type_found = _check_exists(str(path_obj))

    if not exists or type_found != "file":
        return (False, "", f"File does not exist or is not a file: {file_path} (Resolved: {path_obj})")

    if backup:
        backup_path = _backup_file(str(path_obj))
        if not backup_path:
            return (False, "", "Failed to create backup")
        _log(f"Backup: {backup_path}", level=1)

    try:
        with open(path_obj, "a", encoding="utf-8") as f:
            f.write(content + "\n")

        _log(f"Appended line to end of file", level=1)
        return (True, "", "")

    except Exception as e:
        return (False, "", str(e))


def line_prepend(
    file_path: str, content: str, backup: bool = True
) -> Tuple[bool, str, str]:
    """
    Prepend line to beginning of file.

    Returns: (success, stdout_data, error_message)
    Action command: silent on success
    """
    path_obj = Path(file_path).resolve()

    _log(f"Prepending line to {file_path}", level=2)

    exists, type_found = _check_exists(str(path_obj))

    if not exists or type_found != "file":
        return (False, "", f"File does not exist or is not a file: {file_path} (Resolved: {path_obj})")

    if backup:
        backup_path = _backup_file(str(path_obj))
        if not backup_path:
            return (False, "", "Failed to create backup")
        _log(f"Backup: {backup_path}", level=1)

    try:
        with open(path_obj, "r", encoding="utf-8") as f:
            lines = f.readlines()

        lines.insert(0, content + "\n")

        with open(path_obj, "w", encoding="utf-8") as f:
            f.writelines(lines)

        _log(f"Prepended line to beginning of file", level=1)
        return (True, "", "")

    except Exception as e:
        return (False, "", str(e))


def block_replace(
    file_path: str,
    old_block: str,
    new_block: str,
    backup: bool = True,
    unique: bool = False,
    count_only: bool = False,
    preview: bool = False,
) -> Tuple[bool, str, str]:
    """
    Replace multi-line block of text.

    Args:
        file_path: Path to file
        old_block: Block to find
        new_block: Replacement block
        backup: Create backup before modifying
        unique: Fail if block appears more than once (like Edit tool)
        count_only: Just count occurrences, don't modify
        preview: Show diff of changes, don't apply

    Returns: (success, stdout_data, error_message)
    """
    path_obj = Path(file_path).resolve()

    _log(f"Block replace in {file_path}", level=2)

    exists, type_found = _check_exists(str(path_obj))

    if not exists or type_found != "file":
        return (False, "", f"File does not exist or is not a file: {file_path} (Resolved: {path_obj})")

    try:
        with open(path_obj, "r", encoding="utf-8") as f:
            file_content = f.read()

        occurrences = file_content.count(old_block)

        _log(f"Found {occurrences} occurrence(s)", level=2)

        # Count-only mode
        if count_only:
            return (True, f"{occurrences}", "")

        # Validation
        if occurrences == 0:
            return (False, "", "Block not found in file")

        if unique and occurrences > 1:
            return (False, "", f"Block not unique ({occurrences} occurrences). Use --count to see.")

        # Preview mode
        if preview:
            pos = file_content.find(old_block)
            line_num = file_content[:pos].count('\n') + 1

            preview_lines = [
                f"--- {file_path}",
                f"+++ {file_path} (modified)",
                f"@@ line {line_num} @@",
            ]
            for line in old_block.split('\n'):
                preview_lines.append(f"-{line}")
            for line in new_block.split('\n'):
                preview_lines.append(f"+{line}")

            if occurrences > 1 and not unique:
                preview_lines.append(f"\n({occurrences} total occurrences will be replaced)")

            return (True, '\n'.join(preview_lines), "")

        # Actual replacement
        if backup:
            backup_path = _backup_file(str(path_obj))
            if not backup_path:
                return (False, "", "Failed to create backup")
            _log(f"Backup: {backup_path}", level=1)

        new_content = file_content.replace(old_block, new_block)

        with open(path_obj, "w", encoding="utf-8") as f:
            f.write(new_content)

        _log(f"Replaced {occurrences} block(s)", level=1)
        return (True, "", "")

    except Exception as e:
        return (False, "", str(e))


# --- Structure-Aware Reading ---


def read_imports(file_path: str) -> Tuple[bool, str, str]:
    """
    Extract import statements from Python file.

    Returns: (success, stdout_data, error_message)
    Query command: outputs imports to STDOUT (line_num: content format)
    """
    path_obj = Path(file_path).resolve()

    exists, type_found = _check_exists(str(path_obj))

    if not exists or type_found != "file":
        return (False, "", f"File does not exist or is not a file: {file_path} (Resolved: {path_obj})")

    try:
        with open(path_obj, "r", encoding="utf-8") as f:
            lines = f.readlines()

        imports = []
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                imports.append(f"{i}: {stripped}")

        _log(f"Found {len(imports)} import statements", level=1)

        # Output to STDOUT (one per line)
        output = "\n".join(imports)
        return (True, output, "")

    except Exception as e:
        return (False, "", str(e))


def read_functions(file_path: str) -> Tuple[bool, str, str]:
    """
    Extract function definitions from Python file.

    Returns: (success, stdout_data, error_message)
    Query command: outputs functions to STDOUT (line_num: name format)
    """
    path_obj = Path(file_path).resolve()

    exists, type_found = _check_exists(str(path_obj))

    if not exists or type_found != "file":
        return (False, "", f"File does not exist or is not a file: {file_path} (Resolved: {path_obj})")

    try:
        with open(path_obj, "r", encoding="utf-8") as f:
            lines = f.readlines()

        functions = []
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("def ") or stripped.startswith("async def "):
                # Extract function name
                func_sig = stripped.split("(")[0]
                func_name = func_sig.replace("def ", "").replace("async ", "").strip()
                functions.append(f"{i}: {func_name}")

        _log(f"Found {len(functions)} functions", level=1)

        # Output to STDOUT (one per line)
        output = "\n".join(functions)
        return (True, output, "")

    except Exception as e:
        return (False, "", str(e))


def read_docstrings(file_path: str) -> Tuple[bool, str, str]:
    """
    Extract docstrings from Python file.

    Returns: (success, stdout_data, error_message)
    Query command: outputs docstrings to STDOUT (lines X-Y format)
    """
    path_obj = Path(file_path).resolve()

    _log(f"Extracting docstrings from {file_path}", level=2)

    exists, type_found = _check_exists(str(path_obj))

    if not exists or type_found != "file":
        return (False, "", f"File does not exist or is not a file: {file_path} (Resolved: {path_obj})")

    try:
        with open(path_obj, "r", encoding="utf-8") as f:
            lines = f.readlines()

        in_docstring = False
        docstring_start = 0
        docstring_content = []
        docstrings = []

        for i, line in enumerate(lines, 1):
            if '"""' in line or "'''" in line:
                if not in_docstring:
                    in_docstring = True
                    docstring_start = i
                    docstring_content = [line]
                else:
                    docstring_content.append(line)
                    docstrings.append(f"Lines {docstring_start}-{i}")
                    in_docstring = False
                    docstring_content = []
            elif in_docstring:
                docstring_content.append(line)

        _log(f"Found {len(docstrings)} docstrings", level=1)

        # Output to STDOUT (one per line)
        output = "\n".join(docstrings)
        return (True, output, "")

    except Exception as e:
        return (False, "", str(e))


# --- Processing Commands ---


def sort_lines(file_path: str, backup: bool = True) -> Tuple[bool, str, str]:
    """
    Sort lines alphabetically.

    Returns: (success, stdout_data, error_message)
    Action command: silent on success
    """
    path_obj = Path(file_path).resolve()

    exists, type_found = _check_exists(str(path_obj))

    if not exists or type_found != "file":
        return (False, "", f"File does not exist or is not a file: {file_path} (Resolved: {path_obj})")

    if backup:
        backup_path = _backup_file(str(path_obj))
        if not backup_path:
            return (False, "", "Failed to create backup")
        _log(f"Backup: {backup_path}", level=1)

    try:
        with open(path_obj, "r", encoding="utf-8") as f:
            lines = f.readlines()

        line_count = len(lines)
        sorted_lines = sorted(lines)

        with open(path_obj, "w", encoding="utf-8") as f:
            f.writelines(sorted_lines)

        _log(f"Sorted {line_count} lines", level=1)
        return (True, "", "")

    except Exception as e:
        return (False, "", str(e))


def unique_lines(file_path: str, backup: bool = True) -> Tuple[bool, str, str]:
    """
    Remove duplicate lines while preserving order.

    Returns: (success, stdout_data, error_message)
    Action command: silent on success
    """
    path_obj = Path(file_path).resolve()

    _log(f"Removing duplicate lines from {file_path}", level=2)

    exists, type_found = _check_exists(str(path_obj))

    if not exists or type_found != "file":
        return (False, "", f"File does not exist or is not a file: {file_path} (Resolved: {path_obj})")

    if backup:
        backup_path = _backup_file(str(path_obj))
        if not backup_path:
            return (False, "", "Failed to create backup")
        _log(f"Backup: {backup_path}", level=1)

    try:
        with open(path_obj, "r", encoding="utf-8") as f:
            lines = f.readlines()

        seen = set()
        unique = []
        for line in lines:
            if line not in seen:
                seen.add(line)
                unique.append(line)

        duplicates_removed = len(lines) - len(unique)

        with open(path_obj, "w", encoding="utf-8") as f:
            f.writelines(unique)

        _log(f"Removed {duplicates_removed} duplicate(s)", level=1)
        return (True, "", "")

    except Exception as e:
        return (False, "", str(e))


# --- MCP Tools ---

if mcp:

    @mcp.tool()
    async def fs_create(path: str, parents: bool = False) -> str:
        """Create directory with full verification."""
        success, stdout_data, error_msg = create(path, parents)
        if not success:
            return f"[FAIL] {error_msg}"
        return f"[OK] Created: {path}"

    @mcp.tool()
    async def fs_verify(path: str) -> str:
        """Verify path exists and show details."""
        success, stdout_data, error_msg = verify(path)
        if not success:
            return f"[FAIL] {error_msg}"
        return f"[OK] {stdout_data}"

    @mcp.tool()
    async def fs_list(path: str, mode: str = "tree") -> str:
        """
        List directory contents.
        mode: 'tree' (hierarchical), 'flat' (find), or 'detailed' (ls -la)
        """
        success, stdout_data, error_msg = list_path(path, mode)
        if not success:
            return f"[FAIL] {error_msg}"
        return stdout_data

    @mcp.tool()
    async def fs_delete(path: str, recursive: bool = False, backup: bool = True) -> str:
        """Delete file or directory with optional backup."""
        success, stdout_data, error_msg = delete(path, recursive, backup)
        if not success:
            return f"[FAIL] {error_msg}"
        return f"[OK] Deleted: {path}"

    @mcp.tool()
    async def fs_move(source: str, destination: str, backup: bool = True) -> str:
        """Move/rename file or directory."""
        success, stdout_data, error_msg = move(source, destination, backup)
        if not success:
            return f"[FAIL] {error_msg}"
        return f"[OK] Moved: {source} → {destination}"

    @mcp.tool()
    async def fs_copy(source: str, destination: str, recursive: bool = False) -> str:
        """Copy file or directory."""
        success, stdout_data, error_msg = copy(source, destination, recursive)
        if not success:
            return f"[FAIL] {error_msg}"
        return f"[OK] Copied: {source} → {destination}"

    @mcp.tool()
    async def fs_tree(path: str, max_depth: int = None) -> str:
        """Display directory tree structure."""
        success, stdout_data, error_msg = tree(path, max_depth)
        if not success:
            return f"[FAIL] {error_msg}"
        return stdout_data

    @mcp.tool()
    async def fs_find(path: str, pattern: str = None, type_filter: str = None) -> str:
        """Find files matching pattern. type_filter: 'f' for files, 'd' for directories"""
        success, stdout_data, error_msg = find_files(path, pattern, type_filter)
        if not success:
            return f"[FAIL] {error_msg}"
        return stdout_data

    @mcp.tool()
    async def fs_line_replace(
        file_path: str, line_num: int, new_content: str, backup: bool = True
    ) -> str:
        """Replace entire line at given line number (1-indexed)."""
        success, stdout_data, error_msg = line_replace(
            file_path, line_num, new_content, backup
        )
        if not success:
            return f"[FAIL] {error_msg}"
        return f"[OK] Replaced line {line_num} in {file_path}"

    @mcp.tool()
    async def fs_line_insert(
        file_path: str, line_num: int, content: str, backup: bool = True
    ) -> str:
        """Insert new line at given position (1-indexed)."""
        success, stdout_data, error_msg = line_insert(
            file_path, line_num, content, backup
        )
        if not success:
            return f"[FAIL] {error_msg}"
        return f"[OK] Inserted line at position {line_num} in {file_path}"

    @mcp.tool()
    async def fs_line_delete(file_path: str, line_num: int, backup: bool = True) -> str:
        """Delete line at given position (1-indexed)."""
        success, stdout_data, error_msg = line_delete(file_path, line_num, backup)
        if not success:
            return f"[FAIL] {error_msg}"
        return f"[OK] Deleted line {line_num} from {file_path}"

    @mcp.tool()
    async def fs_line_append(file_path: str, content: str, backup: bool = True) -> str:
        """Append line to end of file."""
        success, stdout_data, error_msg = line_append(file_path, content, backup)
        if not success:
            return f"[FAIL] {error_msg}"
        return f"[OK] Appended line to {file_path}"

    @mcp.tool()
    async def fs_line_prepend(file_path: str, content: str, backup: bool = True) -> str:
        """Prepend line to beginning of file."""
        success, stdout_data, error_msg = line_prepend(file_path, content, backup)
        if not success:
            return f"[FAIL] {error_msg}"
        return f"[OK] Prepended line to {file_path}"

    @mcp.tool()
    async def fs_block_replace(
        file_path: str, old_block: str, new_block: str, backup: bool = True
    ) -> str:
        """Replace multi-line block of text."""
        success, stdout_data, error_msg = block_replace(
            file_path, old_block, new_block, backup
        )
        if not success:
            return f"[FAIL] {error_msg}"
        return f"[OK] Replaced block in {file_path}"

    @mcp.tool()
    async def fs_read_imports(file_path: str) -> str:
        """Extract import statements from Python file."""
        success, stdout_data, error_msg = read_imports(file_path)
        if not success:
            return f"[FAIL] {error_msg}"
        return stdout_data

    @mcp.tool()
    async def fs_read_functions(file_path: str) -> str:
        """Extract function definitions from Python file."""
        success, stdout_data, error_msg = read_functions(file_path)
        if not success:
            return f"[FAIL] {error_msg}"
        return stdout_data

    @mcp.tool()
    async def fs_read_docstrings(file_path: str) -> str:
        """Extract docstrings from Python file."""
        success, stdout_data, error_msg = read_docstrings(file_path)
        if not success:
            return f"[FAIL] {error_msg}"
        return stdout_data

    @mcp.tool()
    async def fs_sort(file_path: str, backup: bool = True) -> str:
        """Sort lines alphabetically."""
        success, stdout_data, error_msg = sort_lines(file_path, backup)
        if not success:
            return f"[FAIL] {error_msg}"
        return f"[OK] Sorted lines in {file_path}"

    @mcp.tool()
    async def fs_unique(file_path: str, backup: bool = True) -> str:
        """Remove duplicate lines while preserving order."""
        success, stdout_data, error_msg = unique_lines(file_path, backup)
        if not success:
            return f"[FAIL] {error_msg}"
        return f"[OK] Removed duplicates from {file_path}"


# --- CLI ---


def cli_main():
    """CLI entrypoint."""
    global VERBOSITY

    parser = argparse.ArgumentParser(
        description="SFA Filesystem - Professional filesystem operations with verification"
    )
    parser.add_argument("--version", action="version", version=f"sfa_fs {__version__}")
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v=context, -vv=full BDA)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Operations")

    # Filesystem structure commands
    create_parser = subparsers.add_parser("create", help="Create directory")
    create_parser.add_argument("path", help="Path to create")
    create_parser.add_argument(
        "--parents", "-p", action="store_true", help="Create parent directories"
    )

    verify_parser = subparsers.add_parser("verify", help="Verify path exists")
    verify_parser.add_argument("path", help="Path to verify")

    list_parser = subparsers.add_parser("list", help="List directory contents")
    list_parser.add_argument("path", help="Directory to list")
    list_parser.add_argument(
        "--mode",
        choices=["tree", "flat", "detailed"],
        default="tree",
        help="Output format",
    )

    delete_parser = subparsers.add_parser("delete", help="Delete file or directory")
    delete_parser.add_argument("path", help="Path to delete")
    delete_parser.add_argument(
        "-r", "--recursive", action="store_true", help="Delete directories recursively"
    )
    delete_parser.add_argument("--no-backup", action="store_true", help="Skip backup")

    move_parser = subparsers.add_parser("move", help="Move/rename file or directory")
    move_parser.add_argument("source", help="Source path")
    move_parser.add_argument("destination", help="Destination path")
    move_parser.add_argument("--no-backup", action="store_true", help="Skip backup")

    copy_parser = subparsers.add_parser("copy", help="Copy file or directory")
    copy_parser.add_argument("source", help="Source path")
    copy_parser.add_argument("destination", help="Destination path")
    copy_parser.add_argument(
        "-r", "--recursive", action="store_true", help="Copy directories recursively"
    )

    tree_parser = subparsers.add_parser("tree", help="Display directory tree")
    tree_parser.add_argument("path", help="Directory path")
    tree_parser.add_argument("--depth", type=int, help="Maximum depth")

    find_parser = subparsers.add_parser("find", help="Find files matching pattern")
    find_parser.add_argument("path", help="Directory to search")
    find_parser.add_argument("--pattern", help="Filename pattern (e.g., '*.py')")
    find_parser.add_argument(
        "--type", choices=["f", "d"], help="f=files, d=directories"
    )

    # Reading commands
    read_parser = subparsers.add_parser("read", help="Read file contents with line numbers")
    read_parser.add_argument("file", help="File to read")
    read_parser.add_argument("--offset", type=int, default=0, help="Start from line N (1-indexed)")
    read_parser.add_argument("--limit", type=int, help="Max lines to read")
    read_parser.add_argument("--raw", action="store_true", help="No line numbers")
    read_parser.add_argument("--encoding", default="utf-8", help="File encoding")

    search_parser = subparsers.add_parser("search", help="Search for regex pattern in files")
    search_parser.add_argument("path", help="File or directory to search")
    search_parser.add_argument("pattern", help="Regex pattern")
    search_parser.add_argument("-A", "--after", type=int, default=0, help="Lines after match")
    search_parser.add_argument("-B", "--before", type=int, default=0, help="Lines before match")
    search_parser.add_argument("-C", "--context", type=int, default=0, help="Lines before and after")
    search_parser.add_argument("--count", action="store_true", help="Just show match counts")
    search_parser.add_argument("--files-only", action="store_true", help="Just show file paths")
    search_parser.add_argument("-i", "--ignore-case", action="store_true", help="Case insensitive")
    search_parser.add_argument("--no-recursive", action="store_true", help="Don't recurse directories")

    line_get_parser = subparsers.add_parser("line-get", help="Get specific line(s)")
    line_get_parser.add_argument("file", help="File path")
    line_get_parser.add_argument("range", help="Line range (10, 10-20, 10-, -20)")

    line_delete_range_parser = subparsers.add_parser("line-delete-range", help="Delete range of lines")
    line_delete_range_parser.add_argument("file", help="File path")
    line_delete_range_parser.add_argument("range", help="Line range (10, 10-20, 10-, -20)")
    line_delete_range_parser.add_argument("--no-backup", action="store_true", help="Skip backup")

    # Content manipulation commands
    line_replace_parser = subparsers.add_parser(
        "line-replace", help="Replace entire line"
    )
    line_replace_parser.add_argument("file", help="File path")
    line_replace_parser.add_argument(
        "line_num", type=int, help="Line number (1-indexed)"
    )
    line_replace_parser.add_argument("content", help="New content")
    line_replace_parser.add_argument(
        "--no-backup", action="store_true", help="Skip backup"
    )

    line_insert_parser = subparsers.add_parser("line-insert", help="Insert new line")
    line_insert_parser.add_argument("file", help="File path")
    line_insert_parser.add_argument(
        "line_num", type=int, help="Line number (1-indexed)"
    )
    line_insert_parser.add_argument("content", help="Content to insert")
    line_insert_parser.add_argument(
        "--no-backup", action="store_true", help="Skip backup"
    )

    line_delete_parser = subparsers.add_parser("line-delete", help="Delete line")
    line_delete_parser.add_argument("file", help="File path")
    line_delete_parser.add_argument(
        "line_num", type=int, help="Line number (1-indexed)"
    )
    line_delete_parser.add_argument(
        "--no-backup", action="store_true", help="Skip backup"
    )

    line_append_parser = subparsers.add_parser("line-append", help="Append line to end")
    line_append_parser.add_argument("file", help="File path")
    line_append_parser.add_argument("content", help="Content to append")
    line_append_parser.add_argument(
        "--no-backup", action="store_true", help="Skip backup"
    )

    line_prepend_parser = subparsers.add_parser(
        "line-prepend", help="Prepend line to beginning"
    )
    line_prepend_parser.add_argument("file", help="File path")
    line_prepend_parser.add_argument("content", help="Content to prepend")
    line_prepend_parser.add_argument(
        "--no-backup", action="store_true", help="Skip backup"
    )

    block_replace_parser = subparsers.add_parser(
        "block-replace", help="Replace multi-line block"
    )
    block_replace_parser.add_argument("file", help="File path")
    block_replace_parser.add_argument("old_block", help="Block to replace")
    block_replace_parser.add_argument("new_block", nargs="?", default="", help="Replacement block")
    block_replace_parser.add_argument("--no-backup", action="store_true", help="Skip backup")
    block_replace_parser.add_argument("--unique", action="store_true", help="Fail if >1 match")
    block_replace_parser.add_argument("--count", action="store_true", help="Just count, don't replace")
    block_replace_parser.add_argument("--preview", action="store_true", help="Show diff, don't apply")

    # Structure-aware reading
    read_imports_parser = subparsers.add_parser(
        "read-imports", help="Extract import statements"
    )
    read_imports_parser.add_argument("file", help="Python file path")

    read_functions_parser = subparsers.add_parser(
        "read-functions", help="Extract function definitions"
    )
    read_functions_parser.add_argument("file", help="Python file path")

    read_docstrings_parser = subparsers.add_parser(
        "read-docstrings", help="Extract docstrings"
    )
    read_docstrings_parser.add_argument("file", help="Python file path")

    # Processing commands
    sort_parser = subparsers.add_parser("sort", help="Sort lines alphabetically")
    sort_parser.add_argument("file", help="File path")
    sort_parser.add_argument("--no-backup", action="store_true", help="Skip backup")

    unique_parser = subparsers.add_parser("unique", help="Remove duplicate lines")
    unique_parser.add_argument("file", help="File path")
    unique_parser.add_argument("--no-backup", action="store_true", help="Skip backup")

    args = parser.parse_args()

    # Set global verbosity
    VERBOSITY = args.verbose

    if not args.command:
        parser.print_help()
        return

    # Execute command
    result = None

    if args.command == "create":
        result = create(args.path, args.parents)
    elif args.command == "verify":
        result = verify(args.path)
    elif args.command == "list":
        result = list_path(args.path, args.mode)
    elif args.command == "delete":
        result = delete(args.path, args.recursive, not args.no_backup)
    elif args.command == "move":
        result = move(args.source, args.destination, not args.no_backup)
    elif args.command == "copy":
        result = copy(args.source, args.destination, args.recursive)
    elif args.command == "tree":
        result = tree(args.path, args.depth)
    elif args.command == "find":
        result = find_files(args.path, args.pattern, args.type)
    elif args.command == "read":
        result = read_file(args.file, args.offset, args.limit, args.raw, args.encoding)
    elif args.command == "search":
        result = search(
            args.path, args.pattern, args.before, args.after, args.context,
            args.count, args.files_only, args.ignore_case, not args.no_recursive
        )
    elif args.command == "line-get":
        result = line_get(args.file, args.range)
    elif args.command == "line-delete-range":
        result = line_delete_range(args.file, args.range, not args.no_backup)
    elif args.command == "line-replace":
        result = line_replace(
            args.file, args.line_num, args.content, not args.no_backup
        )
    elif args.command == "line-insert":
        result = line_insert(args.file, args.line_num, args.content, not args.no_backup)
    elif args.command == "line-delete":
        result = line_delete(args.file, args.line_num, not args.no_backup)
    elif args.command == "line-append":
        result = line_append(args.file, args.content, not args.no_backup)
    elif args.command == "line-prepend":
        result = line_prepend(args.file, args.content, not args.no_backup)
    elif args.command == "block-replace":
        result = block_replace(
            args.file, args.old_block, args.new_block,
            backup=not args.no_backup,
            unique=args.unique,
            count_only=args.count,
            preview=args.preview
        )
    elif args.command == "read-imports":
        result = read_imports(args.file)
    elif args.command == "read-functions":
        result = read_functions(args.file)
    elif args.command == "read-docstrings":
        result = read_docstrings(args.file)
    elif args.command == "sort":
        result = sort_lines(args.file, not args.no_backup)
    elif args.command == "unique":
        result = unique_lines(args.file, not args.no_backup)
    else:
        parser.print_help()
        return

    # Output handling - all functions now return Tuple[bool, str, str]
    success, stdout_data, error_msg = result

    if not success:
        if error_msg:
            print(error_msg, file=sys.stderr)
        sys.exit(1)
    else:
        if stdout_data:
            # Ensure UTF-8 output on Windows
            import io

            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", errors="replace"
            )
            print(stdout_data)
        sys.exit(0)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cli_main()
    else:
        # MCP server mode
        if mcp:
            try:
                asyncio.run(mcp.run())
            except KeyboardInterrupt:
                sys.stderr.write("Server stopped by user.\n")
        else:
            print("FastMCP not available. Install with: uv pip install fastmcp")
            print("\nUsage: sfa_fs.py <command> [options]")
            print("Commands: create, verify, list")
            print("Run 'sfa_fs.py --help' for more information")
            sys.exit(1)
