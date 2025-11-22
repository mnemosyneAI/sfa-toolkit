#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "fastmcp"
# ]
# ///

"""
SFA Git Bash Tool (sfa_gitbash.py)
Execute Unix commands via Git Bash on Windows.
Provides both CLI and MCP server modes.
"""

import asyncio
import sys
import os
import json
import subprocess
import argparse
from typing import Optional, Tuple

try:
    from fastmcp import FastMCP

    mcp = FastMCP("sfa-gitbash")
except ImportError:
    mcp = None

# --- Tool Configuration ---
__version__ = "1.0.0"
GIT_BASH = os.getenv("SFA_GITBASH_PATH", r"C:\Program Files\Git\bin\bash.exe")

# --- Helpers ---


def _format_json_output(data: dict) -> str:
    return json.dumps(data, indent=None)


async def _run_bash_async(
    command: str, cwd: Optional[str] = None
) -> Tuple[bool, str, str, int]:
    """
    Run command in Git Bash asynchronously.
    Returns (success, stdout, stderr, exit_code)
    """
    if not os.path.exists(GIT_BASH):
        return (False, "", f"Git Bash not found at '{GIT_BASH}'", 127)

    try:
        # Windows-specific settings to hide console window
        creation_flags = 0
        if sys.platform == "win32":
            creation_flags = subprocess.CREATE_NO_WINDOW

        proc = await asyncio.create_subprocess_exec(
            GIT_BASH,
            "-c",
            command,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            creationflags=creation_flags,
            cwd=cwd,
        )

        stdout, stderr = await proc.communicate()

        return (
            proc.returncode == 0,
            stdout.decode("utf-8", errors="replace"),
            stderr.decode("utf-8", errors="replace"),
            proc.returncode,
        )
    except Exception as e:
        return (False, "", str(e), 1)


def _run_bash_sync(
    command: str, cwd: Optional[str] = None
) -> Tuple[bool, str, str, int]:
    """
    Run command in Git Bash synchronously (for CLI mode).
    Returns (success, stdout, stderr, exit_code)
    """
    if not os.path.exists(GIT_BASH):
        return (False, "", f"Git Bash not found at '{GIT_BASH}'", 127)

    try:
        creation_flags = 0
        if sys.platform == "win32":
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


# --- MCP Tools ---

if mcp:

    @mcp.tool()
    async def bash(command: str, cwd: Optional[str] = None) -> str:
        """
        Execute a Unix command via Git Bash on Windows.

        Args:
            command: Unix shell command to execute
            cwd: Working directory (optional)

        Returns:
            JSON with success status, stdout, stderr, and exit_code
        """
        success, stdout, stderr, exit_code = await _run_bash_async(command, cwd)

        return _format_json_output(
            {
                "success": success,
                "command": command,
                "cwd": cwd or os.getcwd(),
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": exit_code,
            }
        )

# --- CLI Dispatcher ---


def cli_main():
    """CLI entrypoint for direct command execution."""
    parser = argparse.ArgumentParser(
        description="SFA Git Bash - Execute Unix commands on Windows"
    )
    parser.add_argument("--cwd", help="Working directory")
    parser.add_argument("--json", action="store_true", help="Output JSON format")
    parser.add_argument(
        "--version", action="version", version=f"sfa_gitbash {__version__}"
    )
    parser.add_argument(
        "command", nargs=argparse.REMAINDER, help="Unix command to execute"
    )

    args = parser.parse_args()

    # REMAINDER captures all args after options
    # Windows CMD splits on spaces even with single quotes, so we need to reconstruct
    if not args.command:
        # Check if command is being piped via STDIN
        if not sys.stdin.isatty():
            command = sys.stdin.read().strip()
            # Strip outer quotes if present (echo adds them)
            if len(command) >= 2:
                if (command[0] == "'" and command[-1] == "'") or (
                    command[0] == '"' and command[-1] == '"'
                ):
                    command = command[1:-1]
        else:
            command = None
    else:
        # Join all parts - Windows splits incorrectly, we need to rejoin
        command = " ".join(args.command)
        # Strip outer quotes if the FIRST char is quote and LAST char matches
        if len(command) >= 2:
            if (command[0] == "'" and command[-1] == "'") or (
                command[0] == '"' and command[-1] == '"'
            ):
                command = command[1:-1]

    if not command:
        parser.print_help()
        return

    success, stdout, stderr, exit_code = _run_bash_sync(command, args.cwd)

    if args.json:
        print(
            _format_json_output(
                {
                    "success": success,
                    "command": command,
                    "cwd": args.cwd or os.getcwd(),
                    "stdout": stdout,
                    "stderr": stderr,
                    "exit_code": exit_code,
                }
            )
        )
    else:
        # Plain output mode - just print stdout/stderr
        # Use UTF-8 encoding for Windows console to handle Unicode
        if stdout:
            sys.stdout.buffer.write(stdout.encode("utf-8"))
            sys.stdout.flush()
        if stderr:
            sys.stderr.buffer.write(stderr.encode("utf-8"))
            sys.stderr.flush()
        sys.exit(exit_code)


if __name__ == "__main__":
    # CLI mode if: args provided OR stdin is piped
    if len(sys.argv) > 1 or not sys.stdin.isatty():
        cli_main()
    else:
        # MCP server mode (no args, no pipe - interactive mode)
        if mcp:
            try:
                asyncio.run(mcp.run())
            except KeyboardInterrupt:
                sys.stderr.write("Server stopped by user.\n")
        else:
            print("FastMCP not available. Install with: uv pip install fastmcp")
            sys.exit(1)
