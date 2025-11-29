# SFA Toolkit (Python Edition)

**Single File Agent Toolkit** - Autonomous, self-contained Python tools for the SFA ecosystem.

## Acknowledgements & Credits

### The Single File Agent (SFA) Pattern
The architecture of this toolkit is heavily inspired by the work of **IndyDevDan (Daniel Disler)**.
- **Concept:** Single File Agents (SFA) - Self-contained, uv-powered Python scripts
- **Source:** [Disler/single-file-agents](https://github.com/Disler/single-file-agents)

### Model Context Protocol (MCP)
This project aligns with the **Model Context Protocol** standards for tool interoperability.
- **Source:** [Anthropic MCP Announcement](https://www.anthropic.com/news/model-context-protocol)

## Philosophy: Living Software

This toolkit is built on the **Single File Agent (SFA)** pattern:
- **Zero-Install:** No pip install. Just `uv run --script tool.py`
- **Self-Contained:** Each tool is a single .py file with inline PEP 723 dependencies
- **Deterministic:** Structured JSON output for all tools
- **Cross-Platform:** Windows/WSL/Linux compatible

### Tri-Mode Architecture (CLI + MCP + Code)
Tools function in three modes:
1. **Standalone CLI:** For human use or shell scripting
2. **MCP Server:** For AI agent use via Model Context Protocol
3. **Code Execution:** Import tools as libraries for complex workflows

## Serverless RAG Ecosystem

Fully local, serverless RAG system powered by the [YMJ Format](https://github.com/baldsam/ymj-spec).

- **No Vector DB Required:** Embeddings stored in file footer
- **Zero Config:** Uses fastembed with local models

### Workflow
1. **Capture:** `sfa_web.py fetch ... --save-to ./memory`
2. **Manage:** `sfa_ymj.py migrate ./docs`
3. **Search:** `sfa_find.py semantic "query"`

## Usage

**Prerequisite:** [uv](https://github.com/astral-sh/uv) must be installed.

```bash
uv run --script sfa_find.py find "*.json"
uv run --script sfa_find.py --help
```

## Toolkit Catalog

| Tool | Mission |
|------|---------|
| `sfa_find.py` | File discovery, glob patterns, semantic search |
| `sfa_grep.py` | Content search with coordinates |
| `sfa_read.py` | File reading, tree structure, AST parsing |
| `sfa_edit.py` | Surgical file modifications |
| `sfa_exec.py` | Shell command execution |
| `sfa_ops.py` | File operations (move, copy, delete, archive) |
| `sfa_git.py` | Git operations + GitHub integration |
| `sfa_web.py` | Web search, fetch, YouTube transcripts |
| `sfa_db.py` | SQLite-based persistent storage |
| `sfa_ymj.py` | YMJ document management |
| `sfa_tts_edge.py` | Text-to-speech via Edge |
| `sfa_tts_fish.py` | Text-to-speech via Fish Audio |
| `sfa_clone_fish.py` | Voice cloning |
| `sfa_graph.py` | Serverless knowledge graphs |
| `sfa_viz.py` | Data visualization (Bokeh) |
| `sfa_mermaid.py` | Diagram generation |
| `sfa_chrome_ai.py` | Chrome automation |
| `sfa_video.py` | Video extraction & transcription |
| `sfa_repl.py` | Python code execution sandbox |
| `sfa_fs.py` | Complete filesystem operations |
| `sfa_unified_search.py` | Semantic search across sources |
| `sfa_context_compressor.py` | Context size reduction |

## Requirements

- **Python 3.11+**
- **uv** (for script execution)
- **Optional:** git, rg (ripgrep), gh (GitHub CLI)

## Running as MCP Servers

Add tools to your MCP client configuration:

```json
{
  "mcpServers": {
    "sfa-git": {
      "command": "uv",
      "args": ["run", "--script", "/path/to/sfa_git.py"]
    }
  }
}
```

## Contribution Guide

1. **One File:** Keep logic in a single file
2. **Inline Deps:** Use PEP 723 headers
3. **JSON Output:** Always return structured JSON
4. **No Global State:** Tools must be stateless
