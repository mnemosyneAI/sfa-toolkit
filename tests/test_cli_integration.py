import pytest
import subprocess
import sys
from pathlib import Path

TOOLS_TO_TEST = [
    "sfa_git.py",
    "sfa_video.py",
    "sfa_repl.py",
    "sfa_find.py",
    "sfa_web.py"
]

@pytest.mark.parametrize("tool", TOOLS_TO_TEST)
def test_tool_help(tool, sfa_root):
    """Verify that every tool responds to --help."""
    cmd = ["uv", "run", "--script", str(sfa_root / tool), "--help"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    assert result.returncode == 0
    assert "usage:" in result.stdout
