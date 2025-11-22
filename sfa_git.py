# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "fastmcp",
# ]
# ///

import argparse
import json
import os
import re
import subprocess
import sys
from typing import List, Dict, Any, Optional, Union

try:
    from fastmcp import FastMCP
    mcp = FastMCP("sfa-git")
except ImportError:
    mcp = None

# --- Constants ---

CLAUDE_ATTRIBUTION = """🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"""

SECRET_PATTERNS = [
    r'\.env$',
    r'credentials\.json$',
    r'.*\.key$',
    r'.*\.pem$',
    r'id_rsa$',
    r'secret.*\.json$',
    r'.*password.*\.txt$',
    r'.*token.*\.txt$',
]

COMMIT_TYPES = {
    'feat': 'New feature (wholly new capability)',
    'fix': 'Bug fix',
    'docs': 'Documentation changes',
    'refactor': 'Code restructuring without behavior change',
    'test': 'Adding or updating tests',
    'chore': 'Maintenance tasks',
    'perf': 'Performance improvements',
}

# --- Helpers ---

def _run_command(args: List[str], cwd: str = ".") -> str:
    """Run a command and return stdout. Raises on error."""
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8',
            errors='replace'
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Command failed: {' '.join(args)}\nError: {e.stderr.strip()}")

def _parse_git_status(output: str) -> Dict[str, List[str]]:
    """Parse 'git status --porcelain' output."""
    staged = []
    unstaged = []
    untracked = []
    
    for line in output.splitlines():
        if len(line) < 4: continue
        code = line[:2]
        path = line[3:]
        
        index_status = code[0]
        work_status = code[1]
        
        if index_status in 'MADRC':
            staged.append(path)
        
        if work_status in 'MD':
            unstaged.append(path)
            
        if index_status == '?' and work_status == '?':
            untracked.append(path)
            
    return {
        "staged": staged,
        "unstaged": unstaged,
        "untracked": untracked
    }

def _check_secrets(files: List[str]) -> List[str]:
    """Check if any files match secret patterns."""
    secrets = []
    for filepath in files:
        for pattern in SECRET_PATTERNS:
            if re.search(pattern, filepath, re.IGNORECASE):
                secrets.append(filepath)
                break
    return secrets

def _analyze_commit_type(files: List[str], diff: str) -> str:
    """Guess commit type from changed files."""
    if any('test' in f.lower() for f in files):
        return 'test'
    if all(f.endswith('.md') or 'doc' in f.lower() for f in files):
        return 'docs'
    if 'fix' in diff.lower() or 'bug' in diff.lower():
        return 'fix'
    if any(f.startswith('sfa/') for f in files):
        return 'feat'
    return 'chore'

# --- Core Logic: Git ---

def _git_status(cwd: str = ".") -> Dict[str, Any]:
    """Get structured git status."""
    try:
        # Check if repo exists
        _run_command(["git", "rev-parse", "--is-inside-work-tree"], cwd)
        
        # Get status
        output = _run_command(["git", "status", "--porcelain"], cwd)
        parsed = _parse_git_status(output)
        
        # Get branch
        branch = _run_command(["git", "branch", "--show-current"], cwd)
        
        # Check for secrets in staged/unstaged/untracked
        all_files = parsed["staged"] + parsed["unstaged"] + parsed["untracked"]
        secrets = _check_secrets(all_files)

        return {
            "branch": branch,
            "status": parsed,
            "secrets_detected": secrets
        }
    except RuntimeError as e:
        return {"error": str(e)}

def _git_analyze(cwd: str = ".") -> Dict[str, Any]:
    """Analyze changes and suggest commit message."""
    status = _git_status(cwd)
    if "error" in status:
        return status
    
    staged_files = status["status"]["staged"]
    if not staged_files:
        return {
            "message": "No staged changes to analyze.",
            "status": status
        }
    
    diff_staged = _git_diff(staged=True, cwd=cwd)
    commit_type = _analyze_commit_type(staged_files, diff_staged)
    
    return {
        "suggested_type": commit_type,
        "staged_files": staged_files,
        "secrets_detected": status["secrets_detected"],
        "diff_summary": f"{len(diff_staged.splitlines())} lines changed"
    }

def _git_log(count: int = 10, cwd: str = ".") -> List[Dict[str, str]]:
    """Get recent commit log."""
    # Format: hash|author|date|message
    fmt = "%h|%an|%ar|%s"
    try:
        output = _run_command(["git", "log", f"-n{count}", f"--format={fmt}"], cwd)
        logs = []
        for line in output.splitlines():
            parts = line.split("|", 3)
            if len(parts) == 4:
                logs.append({
                    "hash": parts[0],
                    "author": parts[1],
                    "date": parts[2],
                    "message": parts[3]
                })
        return logs
    except RuntimeError:
        return []

def _git_diff(file: Optional[str] = None, staged: bool = False, cwd: str = ".") -> str:
    """Get diff."""
    cmd = ["git", "diff"]
    if staged:
        cmd.append("--staged")
    if file:
        cmd.append(file)
    try:
        return _run_command(cmd, cwd)
    except RuntimeError as e:
        return f"Error: {e}"

def _git_commit(message: str, add_all: bool = False, skip_safety: bool = False, cwd: str = ".") -> str:
    """Commit changes."""
    try:
        if add_all:
            _run_command(["git", "add", "."], cwd)
        
        # Safety Check
        if not skip_safety:
            status = _git_status(cwd)
            if status.get("secrets_detected"):
                return f"Error: Secrets detected in files: {status['secrets_detected']}. Use --skip-safety to override."

        # Append Attribution if not present
        if CLAUDE_ATTRIBUTION not in message:
            message = f"{message}\n\n{CLAUDE_ATTRIBUTION}"

        return _run_command(["git", "commit", "-m", message], cwd)
    except RuntimeError as e:
        return f"Error: {e}"

# --- Core Logic: GitHub ---

def _gh_issue_list(limit: int = 10, state: str = "open", cwd: str = ".") -> List[Dict[str, Any]]:
    """List GitHub issues."""
    try:
        # gh issue list --limit X --state Y --json number,title,url,state
        output = _run_command([
            "gh", "issue", "list", 
            "--limit", str(limit), 
            "--state", state,
            "--json", "number,title,url,state"
        ], cwd)
        return json.loads(output)
    except Exception as e:
        return [{"error": str(e)}]

def _gh_pr_list(limit: int = 10, state: str = "open", cwd: str = ".") -> List[Dict[str, Any]]:
    """List GitHub PRs."""
    try:
        output = _run_command([
            "gh", "pr", "list", 
            "--limit", str(limit), 
            "--state", state,
            "--json", "number,title,url,state,headRefName"
        ], cwd)
        return json.loads(output)
    except Exception as e:
        return [{"error": str(e)}]

# --- MCP Tools ---

if mcp:
    @mcp.tool()
    def git_status() -> str:
        """Get current git status (branch, staged, unstaged files, secrets)."""
        return json.dumps(_git_status(), indent=2)

    @mcp.tool()
    def git_analyze() -> str:
        """Analyze staged changes and suggest commit type."""
        return json.dumps(_git_analyze(), indent=2)

    @mcp.tool()
    def git_log(count: int = 10) -> str:
        """Get recent git commit log."""
        return json.dumps(_git_log(count), indent=2)

    @mcp.tool()
    def git_diff(file: Optional[str] = None, staged: bool = False) -> str:
        """Get git diff. Set staged=True for staged changes."""
        return _git_diff(file, staged)

    @mcp.tool()
    def git_commit(message: str, add_all: bool = False, skip_safety: bool = False) -> str:
        """Commit changes. Set add_all=True to 'git add .' first. Checks for secrets unless skip_safety=True."""
        return _git_commit(message, add_all, skip_safety)

    @mcp.tool()
    def gh_issues(limit: int = 10) -> str:
        """List GitHub issues."""
        return json.dumps(_gh_issue_list(limit), indent=2)

    @mcp.tool()
    def gh_prs(limit: int = 10) -> str:
        """List GitHub PRs."""
        return json.dumps(_gh_pr_list(limit), indent=2)

# --- CLI Dispatcher ---

def main():
    parser = argparse.ArgumentParser(description="SFA Git - SCM & GitHub Integration")
    subparsers = parser.add_subparsers(dest="command")

    # status
    subparsers.add_parser("status", help="Get git status")

    # analyze
    subparsers.add_parser("analyze", help="Analyze staged changes")

    # log
    log_parser = subparsers.add_parser("log", help="Get git log")
    log_parser.add_argument("-n", "--count", type=int, default=10, help="Number of commits")

    # diff
    diff_parser = subparsers.add_parser("diff", help="Get git diff")
    diff_parser.add_argument("file", nargs="?", help="Specific file")
    diff_parser.add_argument("--staged", action="store_true", help="Show staged changes")

    # commit
    commit_parser = subparsers.add_parser("commit", help="Commit changes")
    commit_parser.add_argument("message", help="Commit message")
    commit_parser.add_argument("--add-all", "-a", action="store_true", help="Stage all files first")
    commit_parser.add_argument("--skip-safety", action="store_true", help="Skip secret checks")

    # gh-issues
    gh_issues_parser = subparsers.add_parser("gh-issues", help="List GitHub issues")
    gh_issues_parser.add_argument("--limit", type=int, default=10)
    gh_issues_parser.add_argument("--state", default="open")

    # gh-prs
    gh_prs_parser = subparsers.add_parser("gh-prs", help="List GitHub PRs")
    gh_prs_parser.add_argument("--limit", type=int, default=10)
    gh_prs_parser.add_argument("--state", default="open")

    args = parser.parse_args()

    if args.command == "status":
        print(json.dumps(_git_status(), indent=2))
    elif args.command == "analyze":
        print(json.dumps(_git_analyze(), indent=2))
    elif args.command == "log":
        print(json.dumps(_git_log(args.count), indent=2))
    elif args.command == "diff":
        print(_git_diff(args.file, args.staged))
    elif args.command == "commit":
        print(_git_commit(args.message, args.add_all, args.skip_safety))
    elif args.command == "gh-issues":
        print(json.dumps(_gh_issue_list(args.limit, args.state), indent=2))
    elif args.command == "gh-prs":
        print(json.dumps(_gh_pr_list(args.limit, args.state), indent=2))
    else:
        if mcp:
            mcp.run()
        else:
            parser.print_help()

if __name__ == "__main__":
    main()
