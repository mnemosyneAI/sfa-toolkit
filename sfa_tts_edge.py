#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "edge-tts>=6.1.0",
#   "fastmcp",
# ]
# ///
"""
SFA TTS Edge - Simple text-to-speech using Edge TTS

Usage:
  uv run --script sfa_tts_edge.py "text to speak"
    - Default behavior: Plays audio, saves to temp file

  uv run --script sfa_tts_edge.py "text to speak" --play
    - Explicitly plays audio.

  uv run --script sfa_tts_edge.py "text to speak" --no-play
    - Explicitly prevents audio playback.

  uv run --script sfa_tts_edge.py "text to speak" --output "path/to/file.mp3"
    - Saves audio to a specific file. Playback is disabled by default.
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import platform
from typing import Optional

try:
    from fastmcp import FastMCP
    mcp = FastMCP("sfa-tts-edge")
except ImportError:
    mcp = None

try:
    import edge_tts
except Exception:  # edge-tts resolved by uv when using shebang
    edge_tts = None

try:
    import winsound  # Windows only
    HAS_WINSOUND = (platform.system().lower() == 'windows')
except Exception:
    HAS_WINSOUND = False


# Default voice - Ava
DEFAULT_VOICE = "en-US-AvaMultilingualNeural"
DEFAULT_RATE = "+0%"
SILENCE_DELAY_MS = 500

# --- Global Security ---
ALLOWED_PATHS: list[Path] = []

def _normalize_path(path_str: str) -> Path:
    if not path_str:
        return Path.cwd()
    if sys.platform == "win32" and path_str.startswith("/") and len(path_str) > 2 and path_str[2] == "/":
        drive = path_str[1]
        rest = path_str[2:]
        path_str = f"{drive}:{rest}"
    
    path = Path(path_str).resolve()
    
    # Security Check
    if ALLOWED_PATHS:
        is_allowed = False
        for allowed in ALLOWED_PATHS:
            try:
                path.relative_to(allowed)
                is_allowed = True
                break
            except ValueError:
                continue
        
        if not is_allowed:
            raise PermissionError(f"Access denied: Path '{path}' is not in allowed paths.")
            
    return path

def get_output_dir():
    """Get output directory for temporary audio files"""
    output_dir = Path(tempfile.gettempdir()) / "sfa_tts"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


async def generate_audio(text: str, output_path: Path, voice: str = DEFAULT_VOICE, rate: str = DEFAULT_RATE, pitch: str = "+0Hz", volume: str = "+0%"):
    """Generate audio using Edge TTS with prosody control"""
    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch, volume=volume)
    await communicate.save(str(output_path))



def add_silence_prepend(audio_data, delay_ms=SILENCE_DELAY_MS):
    """Add silence prepend to prevent audio clipping"""
    try:
        # Use static filenames to prevent litter
        temp_dir = tempfile.gettempdir()
        raw_path = os.path.join(temp_dir, 'tts_raw.wav')
        fixed_path = os.path.join(temp_dir, 'tts_fixed.wav')

        # Write raw audio
        with open(raw_path, 'wb') as raw_file:
            raw_file.write(audio_data)

        # Use ffmpeg to add delay
        ffmpeg_bin = 'ffmpeg.exe' if platform.system().lower() == 'windows' else 'ffmpeg'
        result = subprocess.run([
            ffmpeg_bin, '-y',  # overwrite output files
            '-i', raw_path,
            '-af', f'adelay={delay_ms}|{delay_ms}',
            fixed_path
        ], capture_output=True, text=True)

        if result.returncode == 0:
            # Read the processed audio
            with open(fixed_path, 'rb') as f:
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


def play_audio_windows(audio_data):
    """Play audio using Windows winsound"""
    try:
        # Use static filename to prevent litter
        temp_dir = tempfile.gettempdir()
        audio_path = os.path.join(temp_dir, 'tts_play.wav')

        # Write audio data
        with open(audio_path, 'wb') as audio_file:
            audio_file.write(audio_data)

        if not HAS_WINSOUND:
            raise RuntimeError('winsound not available')
        # Play with winsound (Windows)
        winsound.PlaySound(audio_path, winsound.SND_FILENAME)

        # Cleanup static file
        try:
            os.unlink(audio_path)
        except:
            pass

        return True

    except Exception as e:
        print(f"Error: Failed to play audio: {e}", file=sys.stderr)
        return False


def play_audio_linux(audio_data):
    """Play audio using Linux tools (aplay or sox)"""
    try:
        # Use static filename to prevent litter
        temp_dir = tempfile.gettempdir()
        audio_path = os.path.join(temp_dir, 'tts_play.wav')

        # Write audio data
        with open(audio_path, 'wb') as audio_file:
            audio_file.write(audio_data)

        # Try aplay first (ALSA player, most common on Linux)
        try:
            result = subprocess.run(['aplay', '-q', audio_path],
                                    capture_output=True,
                                    timeout=300)  # 5 minutes for long messages
            if result.returncode == 0:
                # Cleanup static file
                try:
                    os.unlink(audio_path)
                except:
                    pass
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Try sox play as fallback
        try:
            result = subprocess.run(['play', '-q', audio_path],
                                    capture_output=True,
                                    timeout=300)  # 5 minutes for long messages
            if result.returncode == 0:
                # Cleanup static file
                try:
                    os.unlink(audio_path)
                except:
                    pass
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        raise RuntimeError('No audio player found (tried aplay, sox)')

    except Exception as e:
        print(f"Error: Failed to play audio on Linux: {e}", file=sys.stderr)
        return False


def play_audio_optimized(file_path: Path):
    """Play audio file with platform-appropriate playback and anti-clipping"""
    try:
        # Read the original audio file
        with open(file_path, 'rb') as f:
            audio_data = f.read()

        # Add silence prepend for anti-clipping
        processed_audio, success = add_silence_prepend(audio_data)
        if not success or not processed_audio:
            raise RuntimeError("Failed to process audio for anti-clipping")

        # Choose playback method based on platform
        system_name = platform.system().lower()
        if system_name == 'windows' or HAS_WINSOUND:
            success = play_audio_windows(processed_audio)
            if not success:
                raise RuntimeError("Failed to play audio with winsound")
        elif system_name == 'linux':
            success = play_audio_linux(processed_audio)
            if not success:
                raise RuntimeError("Failed to play audio on Linux")
        else:
            raise RuntimeError(f"Unsupported platform: {system_name}")

        return True

    except Exception as e:
        print(f"Optimized playback failed: {e}", file=sys.stderr)

        import subprocess

        # Convert WSL path to Windows path for Windows media players
        def to_windows_path(wsl_path: Path) -> str:
            result = subprocess.run(['wslpath', '-w', str(wsl_path)],
                                  capture_output=True, text=True, check=True)
            return result.stdout.strip()

        # Try Windows PowerShell first
        try:
            win_path = to_windows_path(file_path)
            subprocess.run([
                'powershell.exe', '-Command',
                f'(New-Object Media.SoundPlayer "{win_path}").PlaySync()'
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass

        # Try ffplay with high-quality resampling
        try:
            subprocess.run([
                "ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet",
                "-af", "aresample=resampler=soxr",
                str(file_path)
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

        # Fallback players
        players = [
            (["mpg123", "-q"], "mpg123"),
            (["aplay", "-q"], "aplay"),
            (["paplay"], "paplay")
        ]

        for player_cmd, player_name in players:
            try:
                subprocess.run(player_cmd + [str(file_path)],
                             check=True,
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
                return True
            except (FileNotFoundError, subprocess.CalledProcessError):
                continue

        raise RuntimeError("No audio player found. Install ffmpeg, mpg123, or pulseaudio")


def say(text: str, output_path: Path | None = None, play: bool = False):
    """Generate and optionally play TTS audio"""
    # Remove bash escape sequences that interfere with TTS
    # Windows/bash sometimes escapes ! as \! which TTS reads as "backslash exclamation"
    cleaned_text = text.replace('\\!', '!').replace('\\?', '?')

    result = {
        "success": False,
        "text": text,  # Keep original for logging
        "error": None,
        "error_code": None
    }

    try:
        if not cleaned_text or not cleaned_text.strip():
            result["error"] = "Text cannot be empty"
            result["error_code"] = "VALIDATION_FAILED"
            return result

        if output_path is None:
            output_dir = get_output_dir()
            output_path = output_dir / "response.mp3"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate audio
        if edge_tts is None:
            raise RuntimeError('edge-tts not available in this environment')
        asyncio.run(generate_audio(cleaned_text, output_path))

        # Play audio if requested
        if play:
            play_audio_optimized(output_path)

        result["success"] = True
        return result

    except Exception as e:
        result["error"] = str(e)
        result["error_code"] = "TTS_FAILED"
        return result


# --- MCP Tools ---

if mcp:
    @mcp.tool()
    def speak(text: str, play: bool = True) -> str:
        """Generate and play speech from text."""
        result = say(text, play=play)
        return json.dumps(result, indent=2)

    @mcp.tool()
    def save_speech(text: str, output_path: str) -> str:
        """Generate speech and save to file."""
        result = say(text, output_path=output_path, play=False)
        return json.dumps(result, indent=2)

def main():
    parser = argparse.ArgumentParser(description="SFA TTS Edge - Simple text-to-speech")
    parser.add_argument("text", nargs='?', help="Text to speak (positional)")
    parser.add_argument("--text", dest="named_text", help="Text to speak (named parameter)")
    parser.add_argument("--play", default=None, action='store_true', help="Play the generated audio")
    parser.add_argument("--no-play", dest='play', action='store_false', help="Do not play the generated audio")
    parser.add_argument("--output", help="Path to save the audio file")
    parser.add_argument("--allowed-paths", help="Comma-separated list of allowed paths (MCP security)")

    parser.set_defaults(play=None)

    args = parser.parse_args()

    if args.allowed_paths:
        global ALLOWED_PATHS
        for p in args.allowed_paths.split(","):
            if p.strip():
                ALLOWED_PATHS.append(Path(p.strip()).resolve())

    # Use named --text if provided, otherwise use positional text
    text_to_speak = args.named_text or args.text

    if not text_to_speak:
        if mcp:
            mcp.run()
            return
        else:
            print(json.dumps({
                "success": False,
                "text": None,
                "error": "No text provided. Use positional argument or --text parameter",
                "error_code": "NO_TEXT_PROVIDED"
            }, indent=2))
            sys.exit(1)

    play_action = args.play
    if play_action is None:
        # If --play/--no-play is not specified, decide based on --output
        # If --output is given, user likely just wants to save.
        # If no --output, user likely wants to hear it (original behavior).
        play_action = not args.output

    result = say(text_to_speak, output_path=args.output, play=play_action)

    print(json.dumps(result, indent=2))
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
