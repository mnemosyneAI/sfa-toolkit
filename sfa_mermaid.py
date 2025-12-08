# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "fastmcp",
# ]
# ///
"""
SFA Mermaid - Render Mermaid diagrams to HTML

This tool takes Mermaid diagram syntax (from a file or string) and generates 
a standalone HTML file that renders the diagram using the Mermaid.js library.
It automatically opens the result in the default browser.

Usage:
  uv run --script sfa_mermaid.py --file diagram.mmd
  uv run --script sfa_mermaid.py --code "graph TD; A-->B;"
"""

import argparse
import json
import sys
import webbrowser
from pathlib import Path
from typing import Optional

try:
    from fastmcp import FastMCP
    mcp = FastMCP("sfa-mermaid")
except ImportError:
    mcp = None

# --- HTML Template ---
# Uses standard Mermaid.js from CDN. 
# In a strictly offline environment, this script would need to bundle the JS.
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SFA Mermaid Render</title>
    <style>
        body {{
            font-family: sans-serif;
            margin: 20px;
            background-color: #fafafa;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        .container {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            max-width: 100%;
            overflow: auto;
        }}
        h1 {{
            color: #333;
            font-size: 1.2rem;
            margin-bottom: 1rem;
        }}
    </style>
</head>
<body>
    <h1>Mermaid Diagram Preview</h1>
    <div class="container">
        <pre class="mermaid">
{code}
        </pre>
    </div>

    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{ startOnLoad: true }});
    </script>
</body>
</html>
"""

def render_mermaid(code: str, output_path: str = "mermaid_output.html", show: bool = True) -> str:
    """Generates HTML file with embedded Mermaid code"""
    
    html_content = HTML_TEMPLATE.format(code=code)
    
    out_file = Path(output_path).resolve()
    out_file.write_text(html_content, encoding="utf-8")
    
    if show:
        webbrowser.open(out_file.as_uri())
        
    return str(out_file)

# --- MCP Tools ---

if mcp:
    @mcp.tool()
    def render_mermaid_file(file_path: str, output_path: str = "mermaid_output.html") -> str:
        """Render a Mermaid diagram from a file to HTML."""
        try:
            path = Path(file_path)
            if not path.exists():
                return f"Error: File not found: {file_path}"
            
            code = path.read_text(encoding="utf-8")
            result = render_mermaid(code, output_path=output_path, show=True)
            return f"Mermaid diagram rendered to: {result}"
        except Exception as e:
            return f"Error: {str(e)}"

    @mcp.tool()
    def render_mermaid_code(code: str, output_path: str = "mermaid_output.html") -> str:
        """Render a Mermaid diagram from a code string to HTML."""
        try:
            result = render_mermaid(code, output_path=output_path, show=True)
            return f"Mermaid diagram rendered to: {result}"
        except Exception as e:
            return f"Error: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="SFA Mermaid - Render Mermaid diagrams")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="Path to file containing Mermaid syntax (.mmd, .txt)")
    group.add_argument("--code", help="Mermaid syntax string")
    
    parser.add_argument("--output", default="mermaid_output.html", help="Output HTML file path")
    parser.add_argument("--no-show", action="store_true", help="Do not open browser automatically")
    
    args = parser.parse_args()
    
    try:
        code = ""
        if args.file:
            path = Path(args.file)
            if not path.exists():
                print(json.dumps({"success": False, "error": f"File not found: {args.file}"}))
                sys.exit(1)
            code = path.read_text(encoding="utf-8")
        else:
            code = args.code

        output = render_mermaid(code, args.output, not args.no_show)
        print(json.dumps({"success": True, "output": output}))
        
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    if mcp and len(sys.argv) == 1:
        mcp.run()
    else:
        main()
