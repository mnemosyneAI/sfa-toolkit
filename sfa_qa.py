#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "fastmcp",
# ]
# ///

"""
SFA QA - Quality Assurance Tool

Runs tests and linters, returning structured JSON output.
Currently supports:
- Tests: pytest (Python)
- Linting: ruff (Python)

Usage:
  uv run sfa_qa.py test --path ./tests
  uv run sfa_qa.py lint --path ./src --fix
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    from fastmcp import FastMCP
    mcp = FastMCP("sfa-qa")
except ImportError:
    mcp = None

# --- Helpers ---

def _run_command(cmd: List[str], cwd: Optional[str] = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding='utf-8', errors='replace')

# --- Pytest Logic ---

def _parse_junit_xml(xml_content: str) -> Dict[str, Any]:
    try:
        root = ET.fromstring(xml_content)
        summary = {
            "tests": int(root.attrib.get("tests", 0)),
            "failures": int(root.attrib.get("failures", 0)),
            "errors": int(root.attrib.get("errors", 0)),
            "skipped": int(root.attrib.get("skipped", 0)),
            "time": float(root.attrib.get("time", 0.0)),
            "cases": []
        }
        
        for testcase in root.findall(".//testcase"):
            case = {
                "name": testcase.attrib.get("name"),
                "classname": testcase.attrib.get("classname"),
                "file": testcase.attrib.get("file"),
                "line": testcase.attrib.get("line"),
                "status": "passed",
                "message": None
            }
            
            failure = testcase.find("failure")
            error = testcase.find("error")
            skipped = testcase.find("skipped")
            
            if failure is not None:
                case["status"] = "failed"
                case["message"] = failure.attrib.get("message")
                case["details"] = failure.text
            elif error is not None:
                case["status"] = "error"
                case["message"] = error.attrib.get("message")
                case["details"] = error.text
            elif skipped is not None:
                case["status"] = "skipped"
                case["message"] = skipped.attrib.get("message")
            
            summary["cases"].append(case)
            
        return summary
    except ET.ParseError:
        return {"error": "Failed to parse JUnit XML", "raw": xml_content}

def run_pytest(path: str = ".", command: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Run pytest and return structured results.
    
    Args:
        path: Path to test (default: ".")
        command: Custom command prefix (default: ["pytest"])
                 Example: ["uv", "run", "pytest"]
    """
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp:
        xml_path = tmp.name
    
    try:
        # Build command
        base_cmd = command if command else ["pytest"]
        cmd = base_cmd + [path, f"--junitxml={xml_path}"]
        
        proc = _run_command(cmd)
        
        # Read XML
        if os.path.exists(xml_path) and os.path.getsize(xml_path) > 0:
            with open(xml_path, 'r', encoding='utf-8') as f:
                xml_content = f.read()
            results = _parse_junit_xml(xml_content)
            results["raw_stdout"] = proc.stdout
            results["raw_stderr"] = proc.stderr
            results["exit_code"] = proc.returncode
            return results
        else:
            return {
                "success": False,
                "error": "pytest did not generate valid XML report",
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "exit_code": proc.returncode
            }
    finally:
        if os.path.exists(xml_path):
            try:
                os.unlink(xml_path)
            except OSError:
                pass

# --- Ruff Logic ---

def run_ruff(path: str = ".", fix: bool = False, command: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Run ruff linter and return structured results.
    
    Args:
        path: Path to lint
        fix: Whether to apply fixes
        command: Custom command prefix (default: ["ruff"])
    """
    base_cmd = command if command else ["ruff"]
    cmd = base_cmd + ["check", path, "--output-format=json"]
    if fix:
        cmd.append("--fix")
        
    proc = _run_command(cmd)
    
    try:
        # Ruff returns JSON on stdout even on failure (exit code 1)
        if not proc.stdout.strip():
             return {
                "success": proc.returncode == 0,
                "diagnostics": [],
                "fixed": fix,
                "stderr": proc.stderr
            }
            
        diagnostics = json.loads(proc.stdout)
        return {
            "success": proc.returncode == 0,
            "diagnostics": diagnostics,
            "fixed": fix,
            "stderr": proc.stderr
        }
    except json.JSONDecodeError:
        return {
            "success": False,
            "error": "Failed to parse ruff JSON output",
            "stdout": proc.stdout,
            "stderr": proc.stderr
        }

# --- MCP Tools ---

if mcp:
    @mcp.tool()
    def test(path: str = ".", command: str = None) -> str:
        """
        Run tests (pytest) and return JSON results.
        command: Optional custom command string (e.g. "uv run pytest")
        """
        cmd_list = command.split() if command else None
        return json.dumps(run_pytest(path, command=cmd_list), indent=2)

    @mcp.tool()
    def lint(path: str = ".", fix: bool = False, command: str = None) -> str:
        """
        Run linter (ruff) and return JSON results.
        command: Optional custom command string (e.g. "uv run ruff")
        """
        cmd_list = command.split() if command else None
        return json.dumps(run_ruff(path, fix, command=cmd_list), indent=2)

# --- CLI Dispatcher ---

def main():
    parser = argparse.ArgumentParser(description="SFA QA - Quality Assurance")
    subparsers = parser.add_subparsers(dest="command")

    # test command
    test_parser = subparsers.add_parser("test", help="Run tests")
    test_parser.add_argument("--path", default=".", help="Path to test")
    test_parser.add_argument("--cmd", help="Custom command (e.g. 'uv run pytest')")

    # lint command
    lint_parser = subparsers.add_parser("lint", help="Run linter")
    lint_parser.add_argument("--path", default=".", help="Path to lint")
    lint_parser.add_argument("--fix", action="store_true", help="Apply fixes")
    lint_parser.add_argument("--cmd", help="Custom command (e.g. 'uv run ruff')")

    args = parser.parse_args()

    if args.command == "test":
        cmd_list = args.cmd.split() if args.cmd else None
        print(json.dumps(run_pytest(args.path, command=cmd_list), indent=2))
    elif args.command == "lint":
        cmd_list = args.cmd.split() if args.cmd else None
        print(json.dumps(run_ruff(args.path, args.fix, command=cmd_list), indent=2))
    else:
        if mcp:
            mcp.run()
        else:
            parser.print_help()

if __name__ == "__main__":
    main()
