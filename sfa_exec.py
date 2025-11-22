#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "fastmcp"
# ]
# ///

"""
SFA Exec Tool (sfa_exec.py)
System execution and command running capabilities.
"""

import asyncio
import sys
import os
import json
import subprocess
import shlex
from fastmcp import FastMCP
from typing import List, Dict, Any, Tuple

# --- Tool Configuration ---
__version__ = "1.0.0"
mcp = FastMCP("sfa-exec")
SFA_SUBPROC_TIMEOUT = int(os.getenv("SFA_SUBPROC_TIMEOUT", "60"))

# --- Helpers ---

def _format_json_output(data: Dict[str, Any]) -> str:
    return json.dumps(data, indent=None)

def _exec_format_error(error_msg: str) -> str:
    return _format_json_output({"success": False, "error": error_msg})

async def _run_async_proc(cmd: List[str], cwd: str = ".") -> Tuple[bool, str, str]:
    """Returns (success, stdout, stderr)"""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=SFA_SUBPROC_TIMEOUT)
        except asyncio.TimeoutError:
            try: proc.kill()
            except: pass
            return (False, "", f"Process timed out after {SFA_SUBPROC_TIMEOUT} seconds")
        
        return (proc.returncode == 0, stdout.decode('utf-8', errors='ignore'), stderr.decode('utf-8', errors='ignore'))
    except Exception as e:
        return (False, "", str(e))

# --- Tools ---

@mcp.tool()
async def run_command(command: str, cwd: str = ".") -> str:
    """
    Runs a shell command.
    On Windows, attempts to use Git Bash if available for POSIX compatibility.
    """
    # Strategy: Use Git Bash if on Windows to provide POSIX environment
    cmd_list = []
    if sys.platform == "win32":
        git_bash = r"C:\Program Files\Git\bin\bash.exe"
        if os.path.exists(git_bash):
            cmd_list = [git_bash, "-c", command]
        else:
            # Fallback to PowerShell if Git Bash missing
            cmd_list = ["powershell.exe", "-Command", command]
    else:
        # Linux/Mac: use sh
        cmd_list = ["/bin/sh", "-c", command]

    success, stdout, stderr = await _run_async_proc(cmd_list, cwd)
    
    return _format_json_output({
        "success": success,
        "command": command,
        "cwd": cwd,
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": 0 if success else 1
    })

@mcp.tool()
async def gh_cli(args: str) -> str:
    """
    Wraps the GitHub CLI (gh.exe).
    """
    gh_path = "gh"
    if sys.platform == "win32":
        gh_path = r"C:\Program Files\GitHub CLI\gh.exe"
        if not os.path.exists(gh_path):
            # Try PATH
            gh_path = "gh"
    
    # Parse args safely
    cmd_args = shlex.split(args)
    full_cmd = [gh_path] + cmd_args
    
    success, stdout, stderr = await _run_async_proc(full_cmd)
    
    return _format_json_output({
        "success": success,
        "tool": "gh",
        "args": args,
        "stdout": stdout,
        "stderr": stderr
    })

# --- CLI Dispatcher ---

def cli_main():
    'CLI Entrypoint for System Execution.'
    import argparse
    parser = argparse.ArgumentParser(description="SFA Exec Tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # run_command
    p_run = subparsers.add_parser("run", help="Run shell command")
    p_run.add_argument("cmd", help="Command string")
    p_run.add_argument("--cwd", default=".", help="Working directory")

    # gh_cli
    p_gh = subparsers.add_parser("gh", help="Run GitHub CLI")
    p_gh.add_argument("args", help="Arguments for gh")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == "run":
            print(asyncio.run(run_command(args.cmd, args.cwd)))
        elif args.command == "gh":
            print(asyncio.run(gh_cli(args.args)))
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        cli_main()
    else:
        try:
            asyncio.run(mcp.run())
        except KeyboardInterrupt:
            sys.stderr.write("Server stopped by user.\n")
