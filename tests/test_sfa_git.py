import pytest
import subprocess
import sys
import json
from pathlib import Path
import sfa_git

def test_git_status_repl(temp_git_repo):
    """Test sfa_git._git_status() in REPL mode (import)."""
    # Create a file
    (temp_git_repo / "test.txt").write_text("hello")
    
    # Check status (untracked)
    status = sfa_git._git_status(str(temp_git_repo))
    assert "test.txt" in status["status"]["untracked"]
    
    # Stage file
    subprocess.run(["git", "add", "test.txt"], cwd=temp_git_repo, check=True)
    
    # Check status (staged)
    status = sfa_git._git_status(str(temp_git_repo))
    assert "test.txt" in status["status"]["staged"]

def test_git_commit_repl(temp_git_repo):
    """Test sfa_git._git_commit() in REPL mode."""
    (temp_git_repo / "test.txt").write_text("hello")
    subprocess.run(["git", "add", "test.txt"], cwd=temp_git_repo, check=True)
    
    # Commit
    result = sfa_git._git_commit("feat: test commit", cwd=str(temp_git_repo))
    assert "feat: test commit" in result
    
    # Verify log
    log = sfa_git._git_log(cwd=str(temp_git_repo))
    assert len(log) > 0
    assert "feat: test commit" in log[0]["message"]

def test_git_cli_status(temp_git_repo, sfa_root):
    """Test 'uv run sfa_git.py status' (CLI mode)."""
    (temp_git_repo / "cli_test.txt").write_text("cli")
    
    # Run CLI command
    # We use sys.executable to ensure we use the same python, but 'uv run' is preferred for the toolkit
    # For testing, we can call the script directly with python if dependencies are installed, 
    # or use 'uv run' if we want to test the full stack.
    # Let's use 'uv run' to be authentic to the user request.
    
    cmd = ["uv", "run", "--script", str(sfa_root / "sfa_git.py"), "status"]
    result = subprocess.run(cmd, cwd=temp_git_repo, capture_output=True, text=True, check=True)
    
    output = json.loads(result.stdout)
    assert "cli_test.txt" in output["status"]["untracked"]

def test_git_safety_check(temp_git_repo):
    """Test that secrets are detected."""
    (temp_git_repo / ".env").write_text("SECRET=123")
    subprocess.run(["git", "add", ".env"], cwd=temp_git_repo, check=True)
    
    # Try to commit via REPL
    result = sfa_git._git_commit("feat: add secrets", cwd=str(temp_git_repo))
    assert "Error: Secrets detected" in result
