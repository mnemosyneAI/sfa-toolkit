import pytest
import sfa_repl
import os

def test_repl_basic_math(temp_dir):
    """Test executing simple python code."""
    code = "print(1 + 1)"
    result = sfa_repl._run_python(code, cwd=str(temp_dir))
    assert result["success"] is True
    assert result["stdout"].strip() == "2"

def test_repl_import_local(temp_dir, sfa_root):
    """Test importing a local module (sfa_git) from the root."""
    # We need to run the REPL in the sfa_root to import sfa_git
    code = """
import sfa_git
print(type(sfa_git._git_status))
"""
    result = sfa_repl._run_python(code, cwd=str(sfa_root))
    assert result["success"] is True
    assert "function" in result["stdout"] or "method" in result["stdout"]

def test_repl_persistence(temp_dir):
    """Test that file I/O persists."""
    code = """
with open('test.txt', 'w') as f:
    f.write('persisted')
"""
    result = sfa_repl._run_python(code, cwd=str(temp_dir))
    assert result["success"] is True
    
    assert (temp_dir / "test.txt").exists()
    assert (temp_dir / "test.txt").read_text() == "persisted"

def test_repl_error_handling(temp_dir):
    """Test that errors are captured in stderr."""
    code = "raise ValueError('oops')"
    result = sfa_repl._run_python(code, cwd=str(temp_dir))
    assert result["success"] is False
    assert "ValueError: oops" in result["stderr"]
