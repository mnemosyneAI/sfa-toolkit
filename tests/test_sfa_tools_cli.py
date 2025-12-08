import pytest
import subprocess
import sys
import json
import os
from pathlib import Path

# Helper to run tool
def run_tool(tool_name, args, cwd=None):
    script_path = Path(__file__).parent.parent / tool_name
    cmd = ["uv", "run", "--script", str(script_path)] + args
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)

def test_sfa_find_cli(tmp_path):
    """Test sfa_find.py CLI."""
    # Create some files
    (tmp_path / "test1.txt").write_text("content1")
    (tmp_path / "test2.log").write_text("content2")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "test3.txt").write_text("content3")

    # Test find
    result = run_tool("sfa_find.py", ["find", "*.txt", "--path", str(tmp_path)])
    assert result.returncode == 0
    files = json.loads(result.stdout)
    assert len(files) == 2
    assert any("test1.txt" in f for f in files)
    assert any("test3.txt" in f for f in files)

def test_sfa_read_cli(tmp_path):
    """Test sfa_read.py CLI."""
    f = tmp_path / "read_test.txt"
    f.write_text("line1\nline2\nline3")

    # Test read
    result = run_tool("sfa_read.py", ["read", str(f)])
    assert result.returncode == 0
    assert "line1" in result.stdout
    assert "line3" in result.stdout

    # Test read with lines
    result = run_tool("sfa_read.py", ["read", str(f), "--start", "2", "--end", "2"])
    assert result.returncode == 0
    assert "line1" not in result.stdout
    assert "line2" in result.stdout
    assert "line3" not in result.stdout

def test_sfa_edit_cli(tmp_path):
    """Test sfa_edit.py CLI."""
    f = tmp_path / "edit_test.txt"
    
    # Test create
    result = run_tool("sfa_edit.py", ["create", str(f), "hello world"])
    assert result.returncode == 0
    assert f.read_text() == "hello world"

    # Test replace
    result = run_tool("sfa_edit.py", ["replace", str(f), "world", "universe"])
    assert result.returncode == 0
    assert f.read_text() == "hello universe"

    # Test append
    result = run_tool("sfa_edit.py", ["append", str(f), "\nend"])
    assert result.returncode == 0
    assert f.read_text() == "hello universe\nend"

def test_sfa_repl_cli(tmp_path):
    """Test sfa_repl.py CLI."""
    result = run_tool("sfa_repl.py", ["print('hello')"])
    assert result.returncode == 0
    # Output format is string representation of dict
    assert "'success': True" in result.stdout
    assert "'stdout': 'hello\\n'" in result.stdout
