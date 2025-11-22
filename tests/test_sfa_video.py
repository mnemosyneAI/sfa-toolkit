import pytest
import subprocess
import sys
from pathlib import Path

# Import the module under test
# Since conftest.py adds the root to sys.path, we can import directly
import sfa_video

# --- Unit Tests ---

def test_extract_video_id_valid():
    """Test extraction of video IDs from various URL formats."""
    # Standard URL
    assert sfa_video.extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    # Short URL
    assert sfa_video.extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    # Embed URL
    assert sfa_video.extract_video_id("https://www.youtube.com/embed/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    # Shorts URL
    assert sfa_video.extract_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    # Direct ID
    assert sfa_video.extract_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"

def test_extract_video_id_invalid():
    """Test that invalid URLs raise ValueError."""
    with pytest.raises(ValueError):
        sfa_video.extract_video_id("https://example.com/video")
    with pytest.raises(ValueError):
        sfa_video.extract_video_id("not_a_video_id")

def test_strip_vtt_formatting():
    """Test VTT cleanup logic."""
    vtt_content = """WEBVTT
Kind: captions
Language: en

00:00:00.000 --> 00:00:05.000
<c>Hello</c> world!

00:00:05.000 --> 00:00:10.000
This is a <00:00:07.000>test.
"""
    result = sfa_video.strip_vtt_formatting(vtt_content)
    assert "Hello world!" in result
    assert "This is a test." in result
    # The function joins with space
    assert result == "Hello world! This is a test."

# --- CLI Tests ---

def test_sfa_video_help():
    """Test that the CLI help command works."""
    script_path = Path(__file__).parent.parent / "sfa_video.py"
    cmd = ["uv", "run", "--script", str(script_path), "--help"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    assert "usage:" in result.stdout
