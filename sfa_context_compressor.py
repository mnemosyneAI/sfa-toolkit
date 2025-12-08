#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#     "fastmcp",
# ]
# ///
"""
SFA Context Compressor - Intelligent Context Reduction

Analyzes context dumps and creates compressed versions for reload.

Process:
1. Read full dump (everything preserved)
2. Remove cruft (system noise, verbose outputs, resolved errors)
3. Keep signal (conversation, decisions, working state)
4. Output compressed version (30-50k tokens vs 178k)

Usage:
    uv run --script sfa_context_compressor.py input.txt -o output.txt
"""

import re
import sys
from pathlib import Path
from datetime import datetime
from typing import List

try:
    from fastmcp import FastMCP
    mcp = FastMCP("sfa-context-compressor")
except ImportError:
    mcp = None

# --- Global Security ---
ALLOWED_PATHS: List[Path] = []

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

def remove_system_reminders(text):
    """Remove <system-reminder> tags and content."""
    # Remove entire system-reminder blocks
    pattern = r'<system-reminder>.*?</system-reminder>'
    return re.sub(pattern, '', text, flags=re.DOTALL)


def remove_background_bash_checks(text):
    """Remove background bash output availability checks."""
    pattern = r'Background Bash [a-f0-9]+ \(command:.*?\) \(status: \w+\) Has new output available.*?\n'
    return re.sub(pattern, '', text, flags=re.DOTALL)


def remove_hook_success_messages(text):
    """Remove hook success confirmations."""
    lines = text.split('\n')
    filtered = [line for line in lines if not re.match(r'.*hook success: Success', line)]
    return '\n'.join(filtered)


def remove_token_warnings(text):
    """Remove token usage warnings."""
    pattern = r'Token usage: \d+/\d+; \d+ remaining'
    return re.sub(pattern, '', text)


def remove_verbose_outputs(text):
    """
    Remove verbose command outputs that don't add value.
    Keep summaries, remove full listings.
    """
    # Remove long directory listings (keep one-line summaries)
    # This is aggressive - looks for patterns like:
    # total 16
    # drwxr-xr-x ... (multiple lines)
    pattern = r'total \d+\n(?:(?:d|-)(?:r|w|x|-){9}\s+\d+\s+\S+\s+\S+\s+\d+\s+\w+\s+\d+\s+[\d:]+\s+.*\n)+'
    text = re.sub(pattern, '[directory listing removed]\n', text)

    return text


def remove_resolved_errors(text):
    """
    Remove error messages that were subsequently fixed.
    This is tricky - for now, just remove common retry patterns.
    """
    # Remove "Exit code X" followed by eventual success
    # This needs to be smart - for now, keep it simple
    return text


def compress_repetitive_content(text):
    """Compress repetitive patterns."""
    # If same file read multiple times, keep only last
    # If same search run multiple times, keep only last
    # This requires more sophisticated analysis
    return text


def extract_conversation(text):
    """
    Extract the core conversation.
    This is the most important part - user/assistant dialogue.
    """
    # For now, return text as-is
    # TODO: Parse structured conversation history
    return text


def extract_decisions(text):
    """Extract key decisions and technical choices."""
    # Look for patterns indicating decisions:
    # - "decided to..."
    # - "chose..."
    # - "going with..."
    # - "the approach..."

    # For now, simple extraction
    return text


def compress_context(input_path: Path) -> tuple[str, dict]:
    """
    Compress a PreCompact context dump.

    Returns: (compressed_text, stats)
    """
    # Read full dump
    full_text = input_path.read_text(encoding='utf-8')
    original_size = len(full_text)

    # Apply compression steps
    text = full_text

    # Remove cruft
    text = remove_system_reminders(text)
    text = remove_background_bash_checks(text)
    text = remove_hook_success_messages(text)
    text = remove_token_warnings(text)
    text = remove_verbose_outputs(text)

    # Compress repetitive content
    text = compress_repetitive_content(text)

    # Calculate stats
    compressed_size = len(text)
    reduction = original_size - compressed_size
    reduction_pct = (reduction / original_size * 100) if original_size > 0 else 0

    stats = {
        'original_size': original_size,
        'compressed_size': compressed_size,
        'reduction_bytes': reduction,
        'reduction_percent': reduction_pct,
        'original_tokens_est': original_size // 4,  # Rough estimate
        'compressed_tokens_est': compressed_size // 4
    }

    return text, stats


# --- MCP Tools ---

if mcp:
    @mcp.tool()
    def compress(input_path: str, output_path: str = None) -> str:
        """Compress a context file."""
        in_path = _normalize_path(input_path)
        if output_path:
            out_path = _normalize_path(output_path)
        else:
            out_path = in_path.parent / f"{in_path.stem}_compressed.txt"
            
        compressed_text, stats = compress_context(in_path)
        out_path.write_text(compressed_text, encoding='utf-8')
        
        return f"Compressed {in_path.name} -> {out_path.name}\nReduction: {stats['reduction_percent']:.1f}%"

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="SFA Context Compressor")
    parser.add_argument("input", nargs="?", type=Path, help="Input context dump file")
    parser.add_argument("-o", "--output", type=Path, help="Output compressed file (default: input_compressed.txt)")
    parser.add_argument("-s", "--stats", action="store_true", help="Show compression statistics")
    parser.add_argument("--allowed-paths", help="Comma-separated list of allowed paths (MCP security)")

    args = parser.parse_args()

    if args.allowed_paths:
        global ALLOWED_PATHS
        for p in args.allowed_paths.split(","):
            if p.strip():
                ALLOWED_PATHS.append(Path(p.strip()).resolve())

    if not args.input:
        if mcp:
            mcp.run()
            return
        else:
            parser.print_help()
            sys.exit(1)

    # Compress
    print(f"Compressing: {args.input}")
    compressed_text, stats = compress_context(args.input)

    # Write compressed version
    if not args.output:
        args.output = args.input.parent / f"{args.input.stem}_compressed.txt"
        
    args.output.write_text(compressed_text, encoding='utf-8')
    print(f"Compressed: {args.output}")

    # Show stats
    if args.stats or True:  # Always show stats
        print(f"\nCompression Statistics:")
        print(f"  Original:   {stats['original_size']:,} bytes (~{stats['original_tokens_est']:,} tokens)")
        print(f"  Compressed: {stats['compressed_size']:,} bytes (~{stats['compressed_tokens_est']:,} tokens)")
        print(f"  Reduction:  {stats['reduction_bytes']:,} bytes ({stats['reduction_percent']:.1f}%)")

    sys.exit(0)


if __name__ == "__main__":
    main()
