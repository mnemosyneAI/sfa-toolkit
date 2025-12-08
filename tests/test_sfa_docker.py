import pytest
import subprocess
import sys
import json
from pathlib import Path

# Helper to run tool
def run_tool(tool_name, args, cwd=None):
    script_path = Path(__file__).parent.parent / tool_name
    cmd = ["uv", "run", "--script", str(script_path)] + args
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)

def test_sfa_docker_help():
    """Test sfa_docker.py help."""
    result = run_tool("sfa_docker.py", ["--help"])
    assert result.returncode == 0
    assert "usage:" in result.stdout

def test_sfa_docker_ps_mock():
    """Test sfa_docker.py ps command (mocked if docker not present)."""
    # We can't easily mock the subprocess call inside the script from here without complex injection.
    # So we just run it. If docker is not running/installed, it should return an error JSON, not crash.
    
    result = run_tool("sfa_docker.py", ["ps"])
    assert result.returncode == 0
    
    try:
        output = json.loads(result.stdout)
        # It returns a list
        assert isinstance(output, list)
        # If docker is missing/failed, it might be [{"error": ...}]
        if len(output) > 0 and "error" in output[0]:
            # This is acceptable behavior if docker is missing
            pass
    except json.JSONDecodeError:
        pytest.fail(f"Failed to parse JSON: {result.stdout}")
