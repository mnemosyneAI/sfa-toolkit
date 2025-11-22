# SFA Toolkit (Python Edition)

**Single File Agent Toolkit** - Autonomous, self-contained Python tools for the SFA ecosystem.

> 🤖 **Are you an AI Agent?**
> Please read [AGENTS.md](./AGENTS.md) for your operational manual and "Code Execution" protocols.

## Acknowledgements & Credits

### The Single File Agent (SFA) Pattern
The architecture of this toolkit is heavily inspired by the work of **IndyDevDan (Daniel Disler)**.
- **Concept:** "Single File Agents" (SFA) — Self-contained, `uv`-powered Python scripts that act as autonomous agents.
- **Philosophy:** The "Recon" (Reconnaissance) and "Scout" patterns for gathering context before execution.
- **Source:** [Disler/single-file-agents](https://github.com/Disler/single-file-agents) | [YouTube Channel](https://www.youtube.com/@indydevdan)

### Model Context Protocol (MCP)
This project aligns with the **Model Context Protocol** standards for tool interoperability.
- **Definition:** An open standard that enables AI assistants to connect to data sources (filesystem, GitHub, databases) via a universal protocol.
- **Source:** [Anthropic MCP Announcement](https://www.anthropic.com/news/model-context-protocol)

### Text-to-Speech
Special thanks to the open source community for enabling high-quality local TTS.
- **Edge TTS:** [rany2/edge-tts](https://github.com/rany2/edge-tts)
- **OpenAI Edge TTS:** [travisvn/openai-edge-tts](https://github.com/travisvn/openai-edge-tts)

## Philosophy: Living Software

> **"Give an agent mission + autonomy + tools + context, let it execute"**

This toolkit is built on the **Single File Agent (SFA)** pattern:
- **Zero-Install:** No `pip install`. Just `uv run --script <tool>.py`.
- **Self-Contained:** Each tool is a single `.py` file with inline PEP 723 dependencies.
- **Deterministic:** Structured JSON output for all tools.
- **Cross-Platform:** Windows/WSL/Linux compatible.

### Tri-Mode Architecture (CLI + MCP + Code)
New tools in this ecosystem are designed to function in three modes:
1.  **Standalone CLI:** For human use or shell scripting (e.g., `uv run sfa_tool.py arg`).
2.  **MCP Server:** For AI agent use. If run without arguments (and `fastmcp` is installed), the tool starts an MCP server exposing its functions to the AI.
3.  **Code Execution:** Agents can use `sfa_repl.py` to import tools as libraries (e.g., `import sfa_git`). This enables loops, conditionals, and data processing *before* returning results, following the [Anthropic Code Execution Pattern](https://www.anthropic.com/engineering/code-execution-with-mcp).

**Tri-Mode Tools:**
*   `sfa_video.py`
*   `sfa_tts_fish.py`
*   `sfa_clone_fish.py`
*   `sfa_git.py`
*   `sfa_repl.py`

## 🌟 Serverless RAG Ecosystem

The toolkit now features a fully local, serverless **Retrieval-Augmented Generation (RAG)** system powered by the **[YMJ Format](https://github.com/baldsam/ymj-spec)**.

*   **No Vector DB Required:** Embeddings are stored directly in the file footer.
*   **Zero Config:** Uses `fastembed` to run local models (Nomic) automatically.

### The Workflow
1.  **Capture:** `sfa_web.py fetch ... --save-to ./memory` (Saves page + embeddings)
2.  **Manage:** `sfa_ymj.py migrate ./docs` (Adds embeddings to existing docs)
3.  **Search:** `sfa_find.py semantic "query"` (Finds relevant files by meaning)
4.  **Chat:** `sfa_ai.py ask "question" --context-dir ./memory` (Answers using your data)

## Usage

**Prerequisite:** [uv](https://github.com/astral-sh/uv) must be installed.

```powershell
# Run a tool directly
uv run --script sfa_recon.py find "*.json"

# Get help
uv run --script sfa_recon.py --help
```

## Toolkit Catalog

### 1. `sfa_find.py` (The Navigator)
**Mission:** Locate files and map the terrain.
- **Key Functions:** `find` (glob), `semantic` (vector search), `check_runtime`.
- **Philosophy:** Uses `ripgrep` (`rg`) for speed, `fastembed` for meaning.

### 2. `sfa_grep.py` (The Investigator)
**Mission:** Search content and verify changes (Battle Damage Assessment).
- **Key Functions:** `grep` (search text), `count` (verify occurrences).
- **Philosophy:** Provides "coordinates" (File:Line) and context without reading full files.

### 3. `sfa_read.py` (The Reader)
**Mission:** Consume content and structure.
- **Key Functions:** `read` (file content), `tree` (directory structure), `structure` (AST parsing), `stats` (metadata).
- **Philosophy:** Smart reading. Can parse Python AST to show classes/functions without reading the whole file.

### 4. `sfa_edit.py` (The Surgeon)
**Mission:** Modify files with precision.
- **Key Functions:** `create`, `replace`, `replace_line`, `append`.
- **Philosophy:** Surgical edits. Supports line-based replacement to save tokens.

### 5. `sfa_exec.py` (The Executor)
**Mission:** Interact with the system shell.
- **Key Functions:** `run` (shell commands).
- **Philosophy:** Safe execution wrapper.

### 6. `sfa_ops.py` (The Logistics)
**Mission:** Manage file operations.
- **Key Functions:** `move`, `copy`, `delete`, `archive` (zip), `unzip`.
- **Philosophy:** High-level file management.

### 7. `sfa_git.py` (The Guardian)
**Mission:** Source Control, GitHub integration, and Safety.
- **Key Functions:** `status`, `analyze`, `commit`, `diff`, `log`, `gh-issues`, `gh-prs`.
- **Features:** Enforces conventional commits, checks for secrets, and drafts commit messages.
- **Mode:** Dual (CLI + MCP).

### 8. `sfa_web.py` (The Researcher)
**Mission:** Web retrieval and intelligence.
- **Key Functions:** `search` (DuckDuckGo/Brave), `fetch` (Markdown).
- **New:** `--save-to` flag creates "Web Memory" (.ymj files with embeddings).
- **Philosophy:** Zero-config defaults with optional API power.

### 9. `sfa_db.py` (The Memory)
**Mission:** Persistent state and structured storage.
- **Key Functions:** `set`/`get` (KV Store), `insert`/`query` (SQL/Doc Store).
- **Philosophy:** Serverless "Brain" for agents. SQLite-based.

### 10. `sfa_ai.py` (The Delegate)
**Mission:** Offload tasks to other models.
- **Key Functions:** `ask` (Gemini, OpenAI, Anthropic, Local).
- **New:** `--context-dir` flag enables RAG over local .ymj files.
- **Philosophy:** Model agnostic. "Phone a friend" for your agent.

### 11. `sfa_ymj.py` (The Librarian)
**Mission:** Manage YMJ documents and embeddings.
- **Key Functions:** `parse`, `lint`, `enrich` (add embeddings), `migrate` (md -> ymj), `upgrade`.
- **Philosophy:** The file system is the database.

### 12. `sfa_tts_edge.py` (The Voice)
**Mission:** Text-to-speech generation.
- **Key Functions:** `speak`, `save_speech`.
- **Philosophy:** High-quality, free neural TTS via Edge.

### 13. `sfa_synergy.py` (The Tracker)
**Mission:** Work tracking and partnership management.
- **Key Functions:** `add`, `update`, `list`, `report`.
- **Philosophy:** Terminal-native, zero-friction task management.

### 14. `sfa_context_compressor.py` (The Compressor)
**Mission:** Reduce context size for LLMs.
- **Key Functions:** `compress`.
- **Philosophy:** Signal > Noise.

### 15. `sfa_graph.py` (The Cartographer)
**Mission:** Manage Serverless Knowledge Graphs (SKG).
- **Key Functions:** `init`, `add-node`, `add-link`, `update`, `query`.
- **Philosophy:** Append-only, flat-file knowledge graph. Tracks "Epistemic State" (Fact vs Opinion) over time.
- **Spec:** See [Serverless Knowledge Graph Spec](https://github.com/baldsam/graph_spec) (Coming Soon).

### 16. `sfa_viz.py` (The Artist)
**Mission:** Generate data visualizations.
- **Key Functions:** `histogram`, `line`, `bar`, `scatter`, `pie`.
- **Philosophy:** Beautiful, interactive HTML charts using Bokeh.

### 17. `sfa_mermaid.py` (The Architect)
**Mission:** Generate diagrams from text.
- **Key Functions:** `render`.
- **Philosophy:** Text-to-diagram conversion using Mermaid.js.

### 18. `sfa_chrome_ai.py` (The Browser)
**Mission:** Unified Chrome automation and AI browsing.
- **Key Functions:** `navigate`, `extract`, `interact`, `search`.
- **Philosophy:** Complete browser control via DevTools Protocol (CDP). Replaces legacy `sfa_chrome.py`.

### 19. `sfa_chrome_shared.py` (The Persona)
**Mission:** Browser automation with persistent identity.
- **Key Functions:** Same as `sfa_chrome_ai.py`.
- **Philosophy:** Runs on a dedicated port (9223) with a persistent "user-ai" profile. Ideal for tasks requiring login state (e.g., YouTube Music, Email) without interfering with the main AI browser.

### 20. `sfa_edge_shared.py` (The Edge Persona)
**Mission:** Edge automation with persistent identity.
- **Key Functions:** Same as `sfa_chrome_shared.py`.
- **Philosophy:** Runs on port 9225 with a persistent "user-ai" profile.
- **Note:** A "clean" `sfa_edge_ai.py` is not provided because Microsoft Edge on Windows aggressively attaches to the user's active personal session/profile, making reliable isolation difficult. Use `sfa_edge_shared.py` for all Edge automation.

### 21. `sfa_tts_fish.py` (The Narrator)
**Mission:** High-quality Text-to-Speech using Fish Audio.
- **Key Functions:** `speak` (MCP), CLI usage.
- **Features:** Supports 64+ emotion tags (e.g., `(happy)`, `(whispering)`).
- **Mode:** Dual (CLI + MCP).

### 22. `sfa_clone_fish.py` (The Mimic)
**Mission:** Create custom voice clones.
- **Key Functions:** `clone_voice` (MCP), CLI usage.
- **Features:** Creates Fish Audio models from local audio samples.
- **Mode:** Dual (CLI + MCP).

### 23. `sfa_video.py` (The Watcher)
**Mission:** Unified Video Extraction & Transcription.
- **Key Functions:** `extract` (MCP), CLI usage.
- **Features:**
    - **YouTube:** Hybrid extraction (API -> yt-dlp -> VTT parsing) for maximum reliability.
    - **Generic:** Chrome DevTools Protocol (CDP) extraction for other sites.
- **Mode:** Dual (CLI + MCP).

### 24. `sfa_repl.py` (The Sandbox)
**Mission:** Python Code Execution Environment.
- **Key Functions:** `run_python` (MCP), CLI usage.
- **Philosophy:** Enables the "Code Execution" pattern where agents import tools as libraries instead of calling them via MCP.
- **Mode:** Dual (CLI + MCP).

## Dynamic Agents (The Cortex)
Located in `dynamic-agents/`, these scripts act as "Meta-Agents" that orchestrate other tools to perform higher-level cognitive tasks.

### 25. `sfa_DA_scout.py` (The Scout)
**Mission:** Intelligent Codebase Reconnaissance.
- **Role:** Adds judgment to file discovery. Filters, scores, and explains *why* a file matters.
- **Invokes:** `sfa_find.py`

### 26. `sfa_DA_radar.py` (The Radar)
**Mission:** System Knowledge Base.
- **Role:** Answers "Where is X?" and "How do we do Y?" by maintaining a dynamic map of the system.

### 27. `sfa_DA_quartermaster.py` (The Quartermaster)
**Mission:** Context Assembly.
- **Role:** Builds "Briefcases" (Objective, Instructions, Context) for other agents to enable autonomous execution.

### 28. `sfa_DA_forge.py` (The Forge)
**Mission:** Agent Builder.
- **Role:** Creates new specialized agents on demand.

## Standard Runtime
The toolkit expects the following tools in PATH for maximum performance (but will degrade gracefully):
- `git`
- `rg` (ripgrep)
- `gh` (GitHub CLI)

## Requirements

- **Python 3.11+**
- **uv** (for script execution)
- **Standard Runtime (Recommended):**
    - `git` (Version Control)
    - `gh` (GitHub CLI)
    - `rg` (Ripgrep - for fast search)

## Running as MCP Servers

To use these tools with Claude Desktop or other MCP clients, add them to your configuration file (e.g., `claude_desktop_config.json`).

**Note:** You must use the absolute path to the python script.

```json
{
  "mcpServers": {
    "sfa-git": {
      "command": "uv",
      "args": [
        "run",
        "--script",
        "D:/repo_ours/sfa-toolkit/sfa_git.py"
      ]
    },
    "sfa-video": {
      "command": "uv",
      "args": [
        "run",
        "--script",
        "D:/repo_ours/sfa-toolkit/sfa_video.py"
      ]
    },
    "sfa-repl": {
      "command": "uv",
      "args": [
        "run",
        "--script",
        "D:/repo_ours/sfa-toolkit/sfa_repl.py"
      ]
    }
  }
}
```

## Development Status

*   **Active Development:** `D:\repo_ours\sfa-toolkit`
*   **Examples:** See `example_data/` for sample inputs (CSV, JSON, Mermaid) and outputs.
*   **Specs:** See `*.ymj` files for detailed specifications.

## Contribution Guide

1.  **One File:** Keep logic in a single file.
2.  **Inline Deps:** Use PEP 723 headers.
3.  **JSON Output:** Always return structured JSON.
4.  **No Global State:** Tools must be stateless.
