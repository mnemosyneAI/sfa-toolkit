# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "fastmcp",
# ]
# ///

import sys
import io
import contextlib
import traceback
import os
import multiprocessing
import queue
from typing import Dict, Any

try:
    from fastmcp import FastMCP
    mcp = FastMCP("sfa-repl")
except ImportError:
    mcp = None

# --- Core Logic ---

def _worker(code: str, cwd: str, out_queue: multiprocessing.Queue):
    """Worker process to execute code safely."""
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    # Prepare environment
    original_sys_path = sys.path[:]
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
        
    exec_globals = {}
    success = False
    
    try:
        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            original_cwd = os.getcwd()
            try:
                os.chdir(cwd)
                exec(code, exec_globals)
                success = True
            finally:
                os.chdir(original_cwd)
    except Exception:
        traceback.print_exc(file=stderr_capture)
        success = False
    finally:
        sys.path = original_sys_path
        
    out_queue.put({
        "success": success,
        "stdout": stdout_capture.getvalue(),
        "stderr": stderr_capture.getvalue()
    })

def _run_python(code: str, cwd: str = ".", timeout: int = 30) -> Dict[str, Any]:
    """
    Execute Python code in a separate process with timeout.
    """
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_worker, args=(code, cwd, q))
    p.start()
    
    try:
        result = q.get(timeout=timeout)
        p.join()
        return result
    except queue.Empty:
        p.terminate()
        p.join()
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Error: Execution timed out after {timeout} seconds."
        }

# --- MCP Tools ---

if mcp:
    @mcp.tool()
    def run_python(code: str) -> str:
        """
        Execute Python code. 
        Can import local sfa_* modules (e.g. 'import sfa_git').
        Stdout and Stderr are captured and returned.
        """
        # Use current working directory of the process
        cwd = os.getcwd()
        result = _run_python(code, cwd)
        return str(result) # Return string representation of dict

# --- CLI Dispatcher ---

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SFA REPL - Python Code Execution")
    parser.add_argument("code", nargs="?", help="Python code to execute")
    parser.add_argument("--file", "-f", help="Python file to execute")
    
    args = parser.parse_args()

    if args.file:
        with open(args.file, 'r') as f:
            code = f.read()
        print(_run_python(code, os.getcwd()))
    elif args.code:
        print(_run_python(args.code, os.getcwd()))
    else:
        if mcp:
            mcp.run()
        else:
            parser.print_help()
