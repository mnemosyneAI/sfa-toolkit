import os
import sys
import glob
import shutil
from datetime import datetime
from pathlib import Path

def clean_windows_artifacts():
    """Removes '-p' folders and 'nul' files created by shell quirks."""
    artifacts = ["-p", "nul"]
    cleaned = []
    for art in artifacts:
        p = Path(art)
        if p.exists():
            try:
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()
                cleaned.append(art)
            except Exception as e:
                print(f"Error deleting {art}: {e}", file=sys.stderr)
    
    if cleaned:
        print(f"Removed Windows artifacts: {', '.join(cleaned)}")
    else:
        print("No Windows artifacts found.")

def clean_temp_dbs(keep_count=2):
    """Cleans old bootstrap_*.db files from .temp/, keeping the N most recent."""
    temp_dir = Path(".temp")
    if not temp_dir.exists():
        return

    # Find all bootstrap DBs
    dbs = list(temp_dir.glob("bootstrap_*.db"))
    
    if not dbs:
        print("No bootstrap DBs found in .temp/")
        return

    # Sort by modification time (newest first)
    dbs.sort(key=lambda f: f.stat().st_mtime, reverse=True)

    # Split into keep and delete
    to_keep = dbs[:keep_count]
    to_delete = dbs[keep_count:]

    if not to_delete:
        print(f"All {len(dbs)} bootstrap DBs are within the keep limit ({keep_count}).")
        return

    print(f"Found {len(dbs)} bootstrap DBs. Keeping {len(to_keep)} most recent.")
    
    count = 0
    for db in to_delete:
        try:
            db.unlink()
            count += 1
            # print(f"Deleted: {db.name}") # Verbose
        except Exception as e:
            print(f"Error deleting {db.name}: {e}", file=sys.stderr)
    
    print(f"Cleaned {count} old bootstrap databases.")

def main():
    print("--- Starting Workspace Cleanup ---")
    
    # 1. Windows Artifacts
    clean_windows_artifacts()
    
    # 2. Temp DB Cleanup
    clean_temp_dbs(keep_count=2)
    
    # 3. Report Root Status (Informational)
    root_files = [f.name for f in Path(".").iterdir()]
    print(f"Root contains {len(root_files)} items.")
    
    # Check for node_modules at root
    if "node_modules" in root_files:
        print("WARNING: 'node_modules' found at root! This should likely be deleted.", file=sys.stderr)

    print("--- Cleanup Complete ---")

if __name__ == "__main__":
    main()
