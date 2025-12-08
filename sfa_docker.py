#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "fastmcp",
# ]
# ///

"""
SFA Docker - Container Management Tool

Wraps Docker CLI to provide structured JSON output for agents.
Requires 'docker' to be installed and available in PATH.

Usage:
  uv run sfa_docker.py ps
  uv run sfa_docker.py run --image alpine --cmd "echo hello"
"""

import argparse
import json
import subprocess
import sys
from typing import Dict, Any, List, Optional

try:
    from fastmcp import FastMCP
    mcp = FastMCP("sfa-docker")
except ImportError:
    mcp = None

# --- Helpers ---

def _run_command(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')

def _check_docker() -> bool:
    return subprocess.run(["docker", "--version"], capture_output=True).returncode == 0

# --- Docker Logic ---

def list_containers(all: bool = False) -> List[Dict[str, Any]]:
    """List containers using docker ps --format json."""
    cmd = ["docker", "ps", "--format", "{{json .}}"]
    if all:
        cmd.append("-a")
        
    try:
        proc = _run_command(cmd)
    except FileNotFoundError:
        return [{"error": "Docker executable not found in PATH"}]
        
    if proc.returncode != 0:
        return [{"error": proc.stderr.strip() or "Unknown docker error"}]
        
    containers = []
    for line in proc.stdout.splitlines():
        if line.strip():
            try:
                containers.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return containers

def run_container(image: str, command: Optional[str] = None, name: Optional[str] = None, 
                 ports: Optional[List[str]] = None, env: Optional[List[str]] = None,
                 detach: bool = True, rm: bool = True) -> Dict[str, Any]:
    """Run a container."""
    cmd = ["docker", "run"]
    if detach: cmd.append("-d")
    if rm: cmd.append("--rm")
    if name: cmd.extend(["--name", name])
    
    if ports:
        for p in ports:
            cmd.extend(["-p", p])
            
    if env:
        for e in env:
            cmd.extend(["-e", e])
            
    cmd.append(image)
    if command:
        cmd.extend(command.split()) # Simple split, might need shlex for complex commands
        
    proc = _run_command(cmd)
    return {
        "success": proc.returncode == 0,
        "container_id": proc.stdout.strip(),
        "error": proc.stderr if proc.returncode != 0 else None
    }

def stop_container(name: str) -> Dict[str, Any]:
    """Stop a container."""
    proc = _run_command(["docker", "stop", name])
    return {
        "success": proc.returncode == 0,
        "output": proc.stdout.strip(),
        "error": proc.stderr if proc.returncode != 0 else None
    }

def get_logs(name: str, tail: int = 100) -> Dict[str, Any]:
    """Get container logs."""
    proc = _run_command(["docker", "logs", "--tail", str(tail), name])
    return {
        "success": proc.returncode == 0,
        "logs": proc.stdout + proc.stderr, # Docker logs often go to stderr
        "error": proc.stderr if proc.returncode != 0 else None
    }

# --- MCP Tools ---

if mcp:
    @mcp.tool()
    def ps(all: bool = False) -> str:
        """List containers."""
        return json.dumps(list_containers(all), indent=2)

    @mcp.tool()
    def run(image: str, command: str = None, name: str = None) -> str:
        """Run a container."""
        return json.dumps(run_container(image, command, name), indent=2)

    @mcp.tool()
    def stop(name: str) -> str:
        """Stop a container."""
        return json.dumps(stop_container(name), indent=2)

    @mcp.tool()
    def logs(name: str, tail: int = 100) -> str:
        """Get container logs."""
        return json.dumps(get_logs(name, tail), indent=2)

# --- CLI Dispatcher ---

def main():
    parser = argparse.ArgumentParser(description="SFA Docker - Container Management")
    subparsers = parser.add_subparsers(dest="command")

    # ps
    ps_parser = subparsers.add_parser("ps", help="List containers")
    ps_parser.add_argument("--all", "-a", action="store_true", help="Show all")

    # run
    run_parser = subparsers.add_parser("run", help="Run container")
    run_parser.add_argument("--image", required=True, help="Image name")
    run_parser.add_argument("--cmd", help="Command to run")
    run_parser.add_argument("--name", help="Container name")

    # stop
    stop_parser = subparsers.add_parser("stop", help="Stop container")
    stop_parser.add_argument("name", help="Container name/ID")

    # logs
    logs_parser = subparsers.add_parser("logs", help="Get logs")
    logs_parser.add_argument("name", help="Container name/ID")
    logs_parser.add_argument("--tail", type=int, default=100, help="Lines to show")

    args = parser.parse_args()

    if args.command == "ps":
        print(json.dumps(list_containers(args.all), indent=2))
    elif args.command == "run":
        print(json.dumps(run_container(args.image, args.cmd, args.name), indent=2))
    elif args.command == "stop":
        print(json.dumps(stop_container(args.name), indent=2))
    elif args.command == "logs":
        print(json.dumps(get_logs(args.name, args.tail), indent=2))
    else:
        if mcp:
            mcp.run()
        else:
            parser.print_help()

if __name__ == "__main__":
    main()
