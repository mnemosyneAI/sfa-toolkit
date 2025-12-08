# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "fastmcp",
# ]
# ///

import argparse
import json
import os
import shutil
import sys
import zipfile
import tarfile
from pathlib import Path
from typing import List, Optional, Dict, Any

try:
    from fastmcp import FastMCP
    mcp = FastMCP("sfa-ops")
except ImportError:
    mcp = None

# --- Global Security ---
ALLOWED_PATHS: List[Path] = []

# --- Standard Runtime Helpers ---

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

# --- Core Logic ---

def move_item(src: str, dst: str) -> str:
    """Move a file or directory."""
    src_path = _normalize_path(src)
    dst_path = _normalize_path(dst)
    
    if not src_path.exists():
        return f"Error: Source not found: {src}"
    
    try:
        # Create parent dir if it doesn't exist
        if not dst_path.parent.exists():
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
        shutil.move(str(src_path), str(dst_path))
        return f"Successfully moved {src} to {dst}"
    except Exception as e:
        return f"Error moving item: {str(e)}"

def copy_item(src: str, dst: str) -> str:
    """Copy a file or directory."""
    src_path = _normalize_path(src)
    dst_path = _normalize_path(dst)
    
    if not src_path.exists():
        return f"Error: Source not found: {src}"
    
    try:
        if not dst_path.parent.exists():
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
        if src_path.is_dir():
            shutil.copytree(str(src_path), str(dst_path), dirs_exist_ok=True)
        else:
            shutil.copy2(str(src_path), str(dst_path))
        return f"Successfully copied {src} to {dst}"
    except Exception as e:
        return f"Error copying item: {str(e)}"

def delete_item(path: str, force: bool = False) -> str:
    """Delete a file or directory."""
    target_path = _normalize_path(path)
    
    if not target_path.exists():
        return f"Error: Path not found: {path}"
    
    try:
        if target_path.is_dir():
            if force:
                shutil.rmtree(str(target_path))
            else:
                # Check if empty
                if any(target_path.iterdir()):
                    return f"Error: Directory not empty: {path}. Use force=True to delete."
                target_path.rmdir()
        else:
            target_path.unlink()
        return f"Successfully deleted {path}"
    except Exception as e:
        return f"Error deleting item: {str(e)}"

def archive_items(paths: str, output_path: str, format: str = "zip") -> str:
    """
    Archive files or directories.
    paths: Comma-separated list of paths to include.
    """
    out_path = _normalize_path(output_path)
    path_list = [p.strip() for p in paths.split(",")]
    
    try:
        if format == "zip":
            with zipfile.ZipFile(out_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for p_str in path_list:
                    p = _normalize_path(p_str)
                    if not p.exists():
                        continue
                    
                    if p.is_file():
                        zf.write(p, p.name)
                    else:
                        for root, _, files in os.walk(p):
                            for file in files:
                                file_path = Path(root) / file
                                arcname = file_path.relative_to(p.parent)
                                zf.write(file_path, arcname)
        else:
            return "Error: Only 'zip' format is currently supported."
            
        return f"Successfully created archive {output_path}"
    except Exception as e:
        return f"Error creating archive: {str(e)}"

def extract_archive(archive_path: str, dest_path: str) -> str:
    """Extract an archive."""
    arc_path = _normalize_path(archive_path)
    dst = _normalize_path(dest_path)
    
    if not arc_path.exists():
        return f"Error: Archive not found: {archive_path}"
        
    try:
        if arc_path.suffix == ".zip":
            with zipfile.ZipFile(arc_path, 'r') as zf:
                zf.extractall(dst)
        elif arc_path.suffix in [".tar", ".gz", ".tgz"]:
             with tarfile.open(arc_path, 'r') as tf:
                tf.extractall(dst)
        else:
             return f"Error: Unsupported archive format: {arc_path.suffix}"
             
        return f"Successfully extracted {archive_path} to {dest_path}"
    except Exception as e:
        return f"Error extracting archive: {str(e)}"

# --- MCP Tools ---

if mcp:
    @mcp.tool()
    def move(src: str, dst: str) -> str:
        """Move file or directory."""
        return move_item(src, dst)

    @mcp.tool()
    def copy(src: str, dst: str) -> str:
        """Copy file or directory."""
        return copy_item(src, dst)

    @mcp.tool()
    def delete(path: str, force: bool = False) -> str:
        """Delete file or directory."""
        return delete_item(path, force)

    @mcp.tool()
    def archive(paths: str, output: str) -> str:
        """Create zip archive from paths (comma-separated)."""
        return archive_items(paths, output)

    @mcp.tool()
    def unzip(archive: str, dest: str) -> str:
        """Extract zip/tar archive."""
        return extract_archive(archive, dest)

# --- CLI Dispatcher ---

def main():
    parser = argparse.ArgumentParser(description="SFA Ops - Logistics & Management")
    parser.add_argument("--allowed-paths", help="Comma-separated list of allowed paths (MCP security)")
    subparsers = parser.add_subparsers(dest="command")

    # move
    mv_parser = subparsers.add_parser("move", help="Move item")
    mv_parser.add_argument("src", help="Source path")
    mv_parser.add_argument("dst", help="Destination path")

    # copy
    cp_parser = subparsers.add_parser("copy", help="Copy item")
    cp_parser.add_argument("src", help="Source path")
    cp_parser.add_argument("dst", help="Destination path")

    # delete
    rm_parser = subparsers.add_parser("delete", help="Delete item")
    rm_parser.add_argument("path", help="Path to delete")
    rm_parser.add_argument("--force", action="store_true", help="Force delete non-empty dirs")

    # archive
    zip_parser = subparsers.add_parser("archive", help="Create archive")
    zip_parser.add_argument("paths", help="Comma-separated paths to include")
    zip_parser.add_argument("output", help="Output zip file path")

    # unzip
    unzip_parser = subparsers.add_parser("unzip", help="Extract archive")
    unzip_parser.add_argument("archive", help="Archive file path")
    unzip_parser.add_argument("dest", help="Destination directory")

    args = parser.parse_args()

    if args.allowed_paths:
        global ALLOWED_PATHS
        for p in args.allowed_paths.split(","):
            if p.strip():
                ALLOWED_PATHS.append(Path(p.strip()).resolve())

    if args.command == "move":
        print(move_item(args.src, args.dst))
    elif args.command == "copy":
        print(copy_item(args.src, args.dst))
    elif args.command == "delete":
        print(delete_item(args.path, args.force))
    elif args.command == "archive":
        print(archive_items(args.paths, args.output))
    elif args.command == "unzip":
        print(extract_archive(args.archive, args.dest))
    else:
        if mcp:
            mcp.run()
        else:
            parser.print_help()

if __name__ == "__main__":
    main()
