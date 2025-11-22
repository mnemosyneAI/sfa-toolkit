#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "fish-audio-sdk>=1.0.0",
#   "fastmcp",
# ]
# ///
"""
Voice Clone Creator - Create Fish.audio voice clone from audio samples

Usage:
  uv run --script sfa_clone_fish.py --title "My Voice" --description "Custom voice clone" --voices sample1.mp3 sample2.mp3

Creates a voice model on Fish.audio and returns the reference_id for use in TTS.
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from fish_audio_sdk import Session
except ImportError:
    Session = None

try:
    from fastmcp import FastMCP
    mcp = FastMCP("sfa-clone-fish")
except ImportError:
    mcp = None

# Default configuration
DEFAULT_API_KEY = os.environ.get("FISH_API_KEY")


def create_voice_clone(api_key: str, title: str, description: str, voice_files: list[Path]):
    """Create a voice clone model on Fish.audio"""
    if Session is None:
        raise RuntimeError('fish-audio-sdk not available in this environment')

    session = Session(api_key)

    # Read all voice samples
    voice_data = []
    for voice_file in voice_files:
        if not voice_file.exists():
            raise FileNotFoundError(f"Voice sample not found: {voice_file}")
        with open(voice_file, 'rb') as f:
            voice_data.append(f.read())

    # Create the model
    model = session.create_model(
        title=title,
        description=description,
        voices=voice_data,
    )

    return model


# --- MCP Tools ---

if mcp:
    @mcp.tool()
    def clone_voice(title: str, description: str, voice_file_paths: list[str]) -> str:
        """
        Create a voice clone on Fish Audio.
        
        Args:
            title: Name of the voice model.
            description: Description of the voice.
            voice_file_paths: List of absolute paths to audio sample files.
        """
        api_key = os.environ.get("FISH_API_KEY") or DEFAULT_API_KEY
        if not api_key:
            return "Error: FISH_API_KEY not set."

        voice_files = [Path(p) for p in voice_file_paths]
        
        try:
            model = create_voice_clone(api_key, title, description, voice_files)
            ref_id = model.id if hasattr(model, 'id') else str(model)
            return json.dumps({
                "success": True,
                "reference_id": ref_id,
                "title": title
            }, indent=2)
        except Exception as e:
            return f"Error cloning voice: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description="Create Fish.audio voice clone")
    parser.add_argument("--title", help="Voice model title")
    parser.add_argument("--description", help="Voice model description")
    parser.add_argument("--voices", nargs='+', help="Audio sample files")
    parser.add_argument("--api-key", dest="api_key", help="Fish.audio API key (overrides default)")

    args = parser.parse_args()

    # Check if running as MCP
    if len(sys.argv) == 1 and mcp:
        mcp.run()
        return

    if not args.title or not args.description or not args.voices:
        parser.print_help()
        sys.exit(1)

    # Get API key
    api_key = args.api_key or DEFAULT_API_KEY

    # Convert voice paths to Path objects
    voice_files = [Path(v) for v in args.voices]

    result = {
        "success": False,
        "reference_id": None,
        "model_id": None,
        "error": None,
        "error_code": None
    }

    try:
        model = create_voice_clone(api_key, args.title, args.description, voice_files)

        result["success"] = True
        result["reference_id"] = model.id if hasattr(model, 'id') else str(model)
        result["model_id"] = model.id if hasattr(model, 'id') else str(model)
        result["title"] = args.title
        result["description"] = args.description
        result["samples_count"] = len(voice_files)

    except Exception as e:
        result["error"] = str(e)
        result["error_code"] = "CLONE_FAILED"

    print(json.dumps(result, indent=2))
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
