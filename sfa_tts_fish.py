#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "httpx>=0.24.0",
#   "ormsgpack>=1.3.0",
#   "fastmcp",
# ]
# ///
"""
Fish Audio TTS - Text-to-speech using Fish.audio

Usage:
  uv run --script sfa_tts_fish.py "(happy) text to speak"           # with emotion tag
  uv run --script sfa_tts_fish.py --text "(confident) text to speak"   # named parameter
  uv run --script sfa_tts_fish.py --text "text" --voice "reference_id"  # custom voice

Environment:
  FISH_API_KEY: Fish.audio API key (required)
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import platform

try:
    import httpx
    import ormsgpack
except ImportError:
    httpx = None
    ormsgpack = None

try:
    from fastmcp import FastMCP

    mcp = FastMCP("sfa-tts-fish")
except ImportError:
    mcp = None

# Import portable path resolver
# (ClaudePaths removed for sfa-toolkit portability)

# Default configuration
DEFAULT_API_KEY = os.environ.get("FISH_API_KEY")
FISH_API_URL = "https://api.fish.audio/v1/tts"
SILENCE_DELAY_MS = 500

# Default Voice
# Generic Reference Voice: a42a3bca2c574877b4b7dfa8d58bc670

# Fish.audio Supported Emotion Tags (64 total)
EMOTION_TAGS = {
    # Basic Emotions (24)
    "happy",
    "sad",
    "angry",
    "excited",
    "calm",
    "nervous",
    "confident",
    "surprised",
    "satisfied",
    "delighted",
    "scared",
    "worried",
    "upset",
    "frustrated",
    "depressed",
    "empathetic",
    "embarrassed",
    "disgusted",
    "moved",
    "proud",
    "relaxed",
    "grateful",
    "curious",
    "sarcastic",
    # Advanced Emotions (25)
    "disdainful",
    "unhappy",
    "anxious",
    "hysterical",
    "indifferent",
    "uncertain",
    "doubtful",
    "confused",
    "disappointed",
    "regretful",
    "guilty",
    "ashamed",
    "jealous",
    "envious",
    "hopeful",
    "optimistic",
    "pessimistic",
    "nostalgic",
    "lonely",
    "bored",
    "contemptuous",
    "sympathetic",
    "compassionate",
    "determined",
    "resigned",
    # Tone Markers (5)
    "in a hurry tone",
    "shouting",
    "screaming",
    "whispering",
    "soft tone",
    # Audio Effects (10)
    "laughing",
    "chuckling",
    "sobbing",
    "crying loudly",
    "sighing",
    "groaning",
    "panting",
    "gasping",
    "yawning",
    "snoring",
    # Pause/Break Effects
    "break",
    "long-break",
    # Special atmospheric
    "audience laughing",
    "background laughter",
    "crowd laughing",
    # Intensity modifiers
    "slightly",
    "very",
    "extremely",
}

# Voice customization defaults (Fish.audio API parameters)
DEFAULT_SPEED = 0.1  # Speech rate: 0.5-2.0 (1.0=normal, 0.1=90% slower)
DEFAULT_VOLUME = 0  # Loudness: -20 to +20 (0=default)
DEFAULT_TEMPERATURE = 1.0  # Randomness: 0.0-1.0 (1.0=maximum expressiveness)
DEFAULT_TOP_P = 0.7  # Token selection: 0.0-1.0 (0.7=balanced)
DEFAULT_LATENCY = "normal"  # Quality: "normal" or "balanced"
DEFAULT_NORMALIZE = True  # Text normalization
DEFAULT_CHUNK_LENGTH = 200  # Characters per chunk: 100-300


def get_output_dir():
    """Get output directory for temporary audio files"""
    # Use a local 'output' directory or temp dir
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def generate_audio(
    text: str,
    output_path: Path,
    api_key: str,
    voice_id: str | None = None,
    speed: float = DEFAULT_SPEED,
    volume: int = DEFAULT_VOLUME,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    latency: str = DEFAULT_LATENCY,
    normalize: bool = DEFAULT_NORMALIZE,
    chunk_length: int = DEFAULT_CHUNK_LENGTH,
):
    """Generate audio using Fish.audio API with direct HTTP calls"""
    if httpx is None or ormsgpack is None:
        raise RuntimeError("httpx and ormsgpack not available in this environment")

    # Build TTS request
    request_data = {
        "text": text,
        "format": "mp3",
        "speed": speed,
        "volume": volume,
        "temperature": temperature,
        "top_p": top_p,
        "latency": latency,
        "normalize": normalize,
        "chunk_length": chunk_length,
    }
    if voice_id:
        request_data["reference_id"] = voice_id

    # Make API call with model: s1 header for emotion support
    with httpx.Client() as client:
        response = client.post(
            FISH_API_URL,
            content=ormsgpack.packb(request_data),
            headers={
                "authorization": f"Bearer {api_key}",
                "content-type": "application/msgpack",
                "model": "s1",  # Required for emotion tag support
            },
        )
        response.raise_for_status()

        # Write audio to file
        with open(output_path, "wb") as f:
            f.write(response.content)


def add_silence_prepend(audio_data, delay_ms=SILENCE_DELAY_MS):
    """Add silence prepend to prevent audio clipping"""
    try:
        # Use static filenames to prevent litter
        temp_dir = tempfile.gettempdir()
        raw_path = os.path.join(temp_dir, "tts_raw.mp3")
        fixed_path = os.path.join(temp_dir, "tts_fixed.mp3")

        # Write raw audio
        with open(raw_path, "wb") as raw_file:
            raw_file.write(audio_data)

        # Use ffmpeg to add delay
        ffmpeg_bin = (
            "ffmpeg.exe" if platform.system().lower() == "windows" else "ffmpeg"
        )
        result = subprocess.run(
            [
                ffmpeg_bin,
                "-y",  # overwrite output files
                "-i",
                raw_path,
                "-af",
                f"adelay={delay_ms}|{delay_ms}",
                fixed_path,
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            # Read the processed audio
            with open(fixed_path, "rb") as f:
                processed_audio = f.read()

            success = True
        else:
            print(f"Error: ffmpeg processing failed: {result.stderr}", file=sys.stderr)
            processed_audio = None
            success = False

        # Cleanup static files
        try:
            os.unlink(raw_path)
            if success:
                os.unlink(fixed_path)
        except:
            pass

        return processed_audio, success

    except Exception as e:
        print(f"Error: Failed to add silence prepend: {e}", file=sys.stderr)
        return None, False


def play_audio_linux(audio_data):
    """Play audio using Linux tools (ffplay for MP3)"""
    try:
        # Use static filename to prevent litter
        temp_dir = tempfile.gettempdir()
        audio_path = os.path.join(temp_dir, "tts_play.mp3")

        # Write audio data
        with open(audio_path, "wb") as audio_file:
            audio_file.write(audio_data)

        # Try ffplay first (works with MP3, part of ffmpeg)
        try:
            result = subprocess.run(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", audio_path],
                capture_output=True,
                timeout=300,
            )  # 5 minutes for long messages
            if result.returncode == 0:
                # Cleanup static file
                try:
                    os.unlink(audio_path)
                except:
                    pass
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Try mpg123 as fallback
        try:
            result = subprocess.run(
                ["mpg123", "-q", audio_path], capture_output=True, timeout=300
            )
            if result.returncode == 0:
                try:
                    os.unlink(audio_path)
                except:
                    pass
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        raise RuntimeError("No audio player found for MP3 (tried ffplay, mpg123)")

    except Exception as e:
        print(f"Error: Failed to play audio on Linux: {e}", file=sys.stderr)
        return False


def play_audio_optimized(file_path: Path):
    """Play audio file with platform-appropriate playback and anti-clipping"""
    try:
        # Read the original audio file
        with open(file_path, "rb") as f:
            audio_data = f.read()

        # Add silence prepend for anti-clipping
        processed_audio, success = add_silence_prepend(audio_data)
        if not success or not processed_audio:
            raise RuntimeError("Failed to process audio for anti-clipping")

        # Use ffplay/mpg123-based playback on both Linux and Windows
        # as long as the corresponding binaries are on PATH.
        success = play_audio_linux(processed_audio)
        if not success:
            raise RuntimeError(
                "Failed to play audio (ffplay/mpg123 not available or failed)"
            )

        return True

    except Exception as e:
        print(f"Optimized playback failed: {e}", file=sys.stderr)

        # Fallback: try direct playback without processing
        try:
            subprocess.run(
                ["mpg123", "-q", str(file_path)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=300,
            )
            return True
        except:
            pass

        raise RuntimeError("All playback methods failed")


def validate_emotion_tags(text: str) -> tuple[bool, list[str]]:
    """
    Validate emotion tags in text against Fish.audio supported tags.

    Returns: (valid, invalid_tags)
    """
    import re

    # Extract all emotion tags from text: (tag) or (tag1)(tag2)
    tag_pattern = r"\(([^)]+)\)"
    found_tags = re.findall(tag_pattern, text)

    invalid_tags = []
    for tag in found_tags:
        # Handle compound tags like "slightly happy"
        parts = tag.split()

        # Check modifier + emotion (e.g., "slightly happy")
        if len(parts) == 2 and parts[0] in ["slightly", "very", "extremely"]:
            if parts[1] not in EMOTION_TAGS:
                invalid_tags.append(tag)
        # Check single tag
        elif tag not in EMOTION_TAGS:
            invalid_tags.append(tag)

    return (len(invalid_tags) == 0, invalid_tags)


def say(
    text: str,
    api_key: str,
    voice_id: str | None = None,
    speed: float = DEFAULT_SPEED,
    volume: int = DEFAULT_VOLUME,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    latency: str = DEFAULT_LATENCY,
    normalize: bool = DEFAULT_NORMALIZE,
    chunk_length: int = DEFAULT_CHUNK_LENGTH,
):
    """Generate and play TTS audio"""
    # Remove bash escape sequences that interfere with TTS
    cleaned_text = text.replace("\\!", "!").replace("\\?", "?")

    # Replace dashes with break tags for natural pauses (not pronounced as "minus")
    import re

    # Em dash, en dash, or double hyphen -> (break)
    cleaned_text = re.sub(r"—|–|--", " (break) ", cleaned_text)
    # Single hyphen between words (not in compound words) -> (break)
    cleaned_text = re.sub(r"\s+-\s+", " (break) ", cleaned_text)

    result = {"success": False, "error": None, "error_code": None}

    # Validate emotion tags FIRST
    valid, invalid_tags = validate_emotion_tags(cleaned_text)
    if not valid:
        result["error"] = (
            f"Invalid emotion tags: {', '.join(invalid_tags)}. See supported tags below:"
        )
        result["error_code"] = "INVALID_EMOTION_TAGS"
        result["invalid_tags"] = invalid_tags
        # Print the catalog to help fix the error
        print(json.dumps(result, indent=2))
        print("\n" + "=" * 60)
        show_emotion_catalog()
        sys.exit(1)

    try:
        if not cleaned_text or not cleaned_text.strip():
            result["error"] = "Text cannot be empty"
            result["error_code"] = "VALIDATION_FAILED"
            return result

        if not api_key:
            result["error"] = "FISH_API_KEY environment variable not set"
            result["error_code"] = "MISSING_API_KEY"
            return result

        output_dir = get_output_dir()
        output_path = output_dir / "response.mp3"

        # Generate audio
        generate_audio(
            cleaned_text,
            output_path,
            api_key,
            voice_id,
            speed,
            volume,
            temperature,
            top_p,
            latency,
            normalize,
            chunk_length,
        )

        # Play audio with optimized playback
        play_audio_optimized(output_path)

        result["success"] = True
        return result

    except Exception as e:
        result["error"] = str(e)
        result["error_code"] = "TTS_FAILED"
        return result


def show_emotion_catalog():
    """Display complete Fish.audio emotion tag catalog"""
    catalog = """
# Fish.audio Emotion Tags (64 supported)

## Basic Emotions (24)
(happy) (sad) (angry) (excited) (calm) (nervous) (confident) (surprised)
(satisfied) (delighted) (scared) (worried) (upset) (frustrated) (depressed)
(empathetic) (embarrassed) (disgusted) (moved) (proud) (relaxed) (grateful)
(curious) (sarcastic)

## Advanced Emotions (25)
(disdainful) (unhappy) (anxious) (hysterical) (indifferent) (uncertain)
(doubtful) (confused) (disappointed) (regretful) (guilty) (ashamed)
(jealous) (envious) (hopeful) (optimistic) (pessimistic) (nostalgic)
(lonely) (bored) (contemptuous) (sympathetic) (compassionate) (determined)
(resigned)

## Tone Markers (5)
(in a hurry tone) (shouting) (screaming) (whispering) (soft tone)

## Audio Effects (10)
(laughing) (chuckling) (sobbing) (crying loudly) (sighing) (groaning)
(panting) (gasping) (yawning) (snoring)

## Usage
- Max 3 tags per sentence: (tag1)(tag2) Text here
- Intensity modifiers: (slightly happy) (very sad) (extremely excited)
"""
    print(catalog)
    sys.exit(0)


# --- MCP Tools ---

if mcp:

    @mcp.tool()
    def speak(
        text: str,
        voice_id: str = "a42a3bca2c574877b4b7dfa8d58bc670",
        emotion: str = None,
    ) -> str:
        """
        Generate speech from text using Fish Audio.

        Args:
            text: The text to speak. Can include emotion tags like (happy).
            voice_id: The reference ID for the voice (default: Generic Reference).
            emotion: Optional emotion to prepend (e.g., "happy", "serious").
        """
        api_key = os.environ.get("FISH_API_KEY") or DEFAULT_API_KEY
        if not api_key:
            return "Error: FISH_API_KEY not set."

        if emotion:
            text = f"({emotion}) {text}"

        result = say(text, api_key, voice_id)
        if result["success"]:
            return "Audio generated and played successfully."
        else:
            return f"Error: {result['error']}"


def main():
    parser = argparse.ArgumentParser(description="Fish Audio TTS")
    parser.add_argument("text", nargs="?", help="Text to speak (positional)")
    parser.add_argument(
        "--text", dest="named_text", help="Text to speak (named parameter)"
    )
    parser.add_argument(
        "--tags", action="store_true", help="Show complete emotion tag catalog and exit"
    )
    parser.add_argument(
        "--voice",
        dest="voice_id",
        default="a42a3bca2c574877b4b7dfa8d58bc670",
        help="Voice reference ID (default: Generic Reference)",
    )
    parser.add_argument(
        "--api-key", dest="api_key", help="Fish.audio API key (overrides env)"
    )

    # Voice customization options
    parser.add_argument(
        "--speed",
        type=float,
        default=DEFAULT_SPEED,
        help=f"Speech rate: 0.5-2.0 (default: {DEFAULT_SPEED})",
    )
    parser.add_argument(
        "--volume",
        type=int,
        default=DEFAULT_VOLUME,
        help=f"Loudness: -20 to +20 (default: {DEFAULT_VOLUME})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Randomness: 0.0-1.0 (default: {DEFAULT_TEMPERATURE})",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=DEFAULT_TOP_P,
        dest="top_p",
        help=f"Token selection: 0.0-1.0 (default: {DEFAULT_TOP_P})",
    )
    parser.add_argument(
        "--latency",
        choices=["normal", "balanced"],
        default=DEFAULT_LATENCY,
        help=f"Quality mode (default: {DEFAULT_LATENCY})",
    )
    parser.add_argument(
        "--normalize",
        type=bool,
        default=DEFAULT_NORMALIZE,
        help=f"Text normalization (default: {DEFAULT_NORMALIZE})",
    )
    parser.add_argument(
        "--chunk-length",
        type=int,
        default=DEFAULT_CHUNK_LENGTH,
        dest="chunk_length",
        help=f"Characters per chunk: 100-300 (default: {DEFAULT_CHUNK_LENGTH})",
    )

    args = parser.parse_args()

    # Check if running as MCP
    if len(sys.argv) == 1 and mcp:
        mcp.run()
        return

    # Show catalog if requested
    if args.tags:
        show_emotion_catalog()

    # Use named --text if provided, otherwise use positional text
    text_to_speak = args.named_text or args.text

    if not text_to_speak:
        # If no text and no args (handled above), show help
        parser.print_help()
        sys.exit(1)

    # Get API key from args, env, or default
    api_key = args.api_key or os.environ.get("FISH_API_KEY") or DEFAULT_API_KEY

    result = say(
        text_to_speak,
        api_key,
        args.voice_id,
        args.speed,
        args.volume,
        args.temperature,
        args.top_p,
        args.latency,
        args.normalize,
        args.chunk_length,
    )

    # Compact output on success, pretty print on error
    if result["success"]:
        print(json.dumps(result))
    else:
        print(json.dumps(result, indent=2))
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
