#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "fastmcp",
#     "pathlib",
# ]
# ///

"""
SFA Tmux Tool (sfa_tmux.py) v1.1.0
Agent orchestration via tmux sessions with file-based messaging.

PHILOSOPHY:
  STDOUT: Data (session-id, output text, JSON)
  STDERR: Context (controlled by -v)
  EXIT:   0=success, 1=failure

COMMANDS:
  Agent Lifecycle:
    spawn [--id ID] [--cwd PATH] [COMMAND...]
                            Create agent session, returns session-id
    kill [--id ID]          Kill agent session
    list [--json]           List all agent sessions

  Terminal Communication:
    send [--id ID] TEXT     Send text/keys to agent
    read [--id ID] [--lines N]
                            Read agent output (capture-pane)
    pipe [--id ID] FILE     Pipe file contents to agent

  Message Broker:
    msg --to ID [--from ID] MESSAGE
                            Post message to agent's inbox
    inbox --id ID [--json]  Read agent's inbox
    ack --id ID [--all|--msg MSG_ID]
                            Acknowledge/archive messages

  Control:
    attach [--id ID]        Attach to agent session (interactive)

AGENT DIRECTORY STRUCTURE:
  {AGENTS_DIR}/{id}/
    context.md              - Initial context/prompt
    output.log              - Captured output
    artifacts/              - Work products
    status                  - Current status (running/complete/failed)
    inbox/                  - Incoming messages
    inbox/archive/          - Acknowledged messages
"""

import subprocess
import sys
import json
import argparse
import os
from pathlib import Path
from datetime import datetime
import time
from typing import Optional, List, Dict, Any, Tuple

# --- Configuration ---

# Safe default for public release, overrideable via env var
DEFAULT_AGENTS_DIR = Path.home() / ".sfa" / "agents"
AGENTS_DIR = Path(os.environ.get("SFA_AGENTS_DIR", DEFAULT_AGENTS_DIR))

# --- MCP Support ---

try:
    from fastmcp import FastMCP  # type: ignore

    mcp = FastMCP("sfa-tmux")
except ImportError:
    mcp = None

# --- Core Logic ---


def _run_tmux(*args: str, capture: bool = True) -> Tuple[int, str, str]:
    """Run tmux command. Returns (returncode, stdout, stderr)."""
    cmd = ["tmux"] + list(args)
    try:
        result = subprocess.run(cmd, capture_output=capture, text=True)
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except FileNotFoundError:
        return 1, "", "tmux not found in PATH"


def _session_name(agent_id: str) -> str:
    """Convert agent ID to tmux session name."""
    return f"agent-{agent_id}"


def _agent_dir(agent_id: str) -> Path:
    """Get agent's working directory."""
    return AGENTS_DIR / agent_id


def _ensure_agent_dirs(agent_id: str) -> Path:
    """Ensure agent directory structure exists."""
    agent_dir = _agent_dir(agent_id)
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "artifacts").mkdir(exist_ok=True)
    (agent_dir / "inbox").mkdir(exist_ok=True)
    (agent_dir / "inbox" / "archive").mkdir(exist_ok=True)
    return agent_dir


def spawn_agent(
    agent_id: str | None = None,
    command: list[str] | None = None,
    cwd: str | None = None,
) -> Tuple[bool, str]:
    """Spawn new agent session. Returns (success, session-id or error)."""
    if not agent_id:
        agent_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    session = _session_name(agent_id)
    agent_dir = _ensure_agent_dirs(agent_id)
    (agent_dir / "status").write_text("running\n")

    work_dir = cwd or str(agent_dir)
    # Ensure work_dir exists
    if cwd and not os.path.isdir(cwd):
        return False, f"Working directory does not exist: {cwd}"

    args = ["new-session", "-d", "-s", session, "-c", work_dir]

    if command:
        args.extend(command)

    code, stdout, stderr = _run_tmux(*args)
    if code == 0:
        return True, agent_id

    if "duplicate session" in stderr:
        return True, agent_id

    return False, stderr or "Failed to spawn agent"


def kill_agent(agent_id: str) -> Tuple[bool, str]:
    """Kill agent session. Returns (success, message)."""
    session = _session_name(agent_id)
    agent_dir = _agent_dir(agent_id)

    code, stdout, stderr = _run_tmux("kill-session", "-t", session)

    if agent_dir.exists():
        (agent_dir / "status").write_text("killed\n")

    if code == 0:
        return True, f"Agent {agent_id} killed"
    return False, stderr or "Failed to kill agent"


def send_to_agent(agent_id: str, text: str, literal: bool = False) -> Tuple[bool, str]:
    """Send text/keys to agent. Returns (success, message)."""
    session = _session_name(agent_id)

    args = ["send-keys", "-t", session]
    if literal:
        args.append("-l")
    args.append(text)

    code, stdout, stderr = _run_tmux(*args)
    if code == 0:
        return True, "Sent"
    return False, stderr or "Failed to send"


def read_agent(
    agent_id: str, lines: int | None = None, save: bool = False
) -> Tuple[bool, str]:
    """Read agent output. Returns (success, text or error)."""
    session = _session_name(agent_id)

    args = ["capture-pane", "-t", session, "-p"]
    if lines:
        args.extend(["-S", str(-lines)])

    code, stdout, stderr = _run_tmux(*args)
    if code == 0:
        if save:
            agent_dir = _agent_dir(agent_id)
            if agent_dir.exists():
                log_file = agent_dir / "output.log"
                with log_file.open("a") as f:
                    f.write(f"\n--- {datetime.now().isoformat()} ---\n")
                    f.write(stdout)
                    f.write("\n")
        return True, stdout
    return False, stderr or "Failed to read agent"


def pipe_to_agent(agent_id: str, filepath: str) -> Tuple[bool, str]:
    """Pipe file contents to agent. Returns (success, message)."""
    session = _session_name(agent_id)

    try:
        with open(filepath) as f:
            content = f.read()
    except Exception as e:
        return False, f"Failed to read file: {e}"

    # Load content into tmux buffer
    code, _, stderr = _run_tmux("load-buffer", "-b", "pipe", filepath)
    if code != 0:
        return False, stderr or "Failed to load buffer"

    # Paste buffer to session
    code, _, stderr = _run_tmux("paste-buffer", "-b", "pipe", "-t", session)
    if code != 0:
        return False, stderr or "Failed to paste buffer"

    return True, f"Piped {len(content)} bytes"


def list_agents_data() -> List[Dict[str, Any]]:
    """List all agent sessions as data objects."""
    code, stdout, stderr = _run_tmux(
        "list-sessions", "-F", "#{session_name}|#{session_created}|#{session_windows}"
    )

    if code != 0:
        return []

    agents = []
    for line in stdout.split("\n"):
        if line.startswith("agent-"):
            parts = line.split("|")
            agent_id = parts[0].replace("agent-", "")
            agent_dir = _agent_dir(agent_id)
            status = "unknown"
            if agent_dir.exists() and (agent_dir / "status").exists():
                status = (agent_dir / "status").read_text().strip()

            # Count inbox messages
            inbox_count = 0
            inbox_dir = agent_dir / "inbox"
            if inbox_dir.exists():
                inbox_count = len([f for f in inbox_dir.glob("*.msg")])

            agents.append(
                {
                    "id": agent_id,
                    "session": parts[0],
                    "created": parts[1] if len(parts) > 1 else "",
                    "windows": parts[2] if len(parts) > 2 else "1",
                    "status": status,
                    "inbox": inbox_count,
                    "dir": str(agent_dir),
                }
            )
    return agents


def attach_agent(agent_id: str) -> Tuple[bool, str]:
    """Attach to agent session. Returns (success, message)."""
    session = _session_name(agent_id)
    code = subprocess.call(["tmux", "attach-session", "-t", session])
    if code == 0:
        return True, "Detached"
    return False, f"Failed to attach (code {code})"


# ============ MESSAGE BROKER ============


def post_message(
    to_agent: str, message: str, from_agent: str = "sfa"
) -> Tuple[bool, str]:
    """Post a message to agent's inbox. Returns (success, msg_id or error)."""
    agent_dir = _ensure_agent_dirs(to_agent)
    inbox_dir = agent_dir / "inbox"

    # Generate message ID (timestamp-based for ordering)
    ts = datetime.now()
    msg_id = ts.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
    msg_file = inbox_dir / f"{msg_id}.msg"

    # Write message with metadata
    content = {
        "id": msg_id,
        "from": from_agent,
        "to": to_agent,
        "timestamp": ts.isoformat(),
        "message": message,
    }

    try:
        msg_file.write_text(json.dumps(content, indent=2))
        return True, msg_id
    except Exception as e:
        return False, f"Failed to write message: {e}"


def read_inbox_data(agent_id: str) -> List[Dict[str, Any]]:
    """Read agent's inbox as data objects."""
    agent_dir = _agent_dir(agent_id)
    inbox_dir = agent_dir / "inbox"

    if not inbox_dir.exists():
        return []

    messages = []
    for msg_file in sorted(inbox_dir.glob("*.msg")):
        try:
            content = json.loads(msg_file.read_text())
            messages.append(content)
        except Exception:
            continue
    return messages


def ack_messages(
    agent_id: str, msg_id: str | None = None, ack_all: bool = False
) -> Tuple[bool, str]:
    """Acknowledge (archive) messages. Returns (success, message)."""
    agent_dir = _agent_dir(agent_id)
    inbox_dir = agent_dir / "inbox"
    archive_dir = inbox_dir / "archive"

    if not inbox_dir.exists():
        return True, "No messages to ack"

    archive_dir.mkdir(exist_ok=True)
    count = 0

    if ack_all:
        for msg_file in inbox_dir.glob("*.msg"):
            msg_file.rename(archive_dir / msg_file.name)
            count += 1
    elif msg_id:
        msg_file = inbox_dir / f"{msg_id}.msg"
        if msg_file.exists():
            msg_file.rename(archive_dir / msg_file.name)
            count = 1
        else:
            return False, f"Message {msg_id} not found"
    else:
        return False, "Specify --all or --msg MSG_ID"

    return True, f"Acknowledged {count} message(s)"


# --- MCP Tools Implementation ---

if mcp:

    @mcp.tool()
    def tmux_spawn(cwd: str | None = None, command: list[str] | None = None) -> str:
        """Spawn a new agent session. Returns the agent ID."""
        success, result = spawn_agent(command=command, cwd=cwd)
        if not success:
            raise RuntimeError(result)
        return result

    @mcp.tool()
    def tmux_list_sessions() -> str:
        """List all active agent sessions as JSON."""
        agents = list_agents_data()
        return json.dumps(agents, indent=2)

    @mcp.tool()
    def tmux_kill(agent_id: str) -> str:
        """Kill an agent session."""
        success, msg = kill_agent(agent_id)
        if not success:
            raise RuntimeError(msg)
        return msg

    @mcp.tool()
    def tmux_send(agent_id: str, text: str) -> str:
        """Send text to an agent's terminal."""
        success, msg = send_to_agent(agent_id, text)
        if not success:
            raise RuntimeError(msg)
        return msg

    @mcp.tool()
    def tmux_read(agent_id: str, lines: int = 20) -> str:
        """Read the output of an agent's terminal."""
        success, output = read_agent(agent_id, lines=lines)
        if not success:
            raise RuntimeError(output)
        return output

    @mcp.tool()
    def tmux_msg_post(to_agent: str, message: str, from_agent: str = "sfa") -> str:
        """Post a structured message to an agent's inbox."""
        success, result = post_message(to_agent, message, from_agent)
        if not success:
            raise RuntimeError(result)
        return json.dumps({"status": "sent", "msg_id": result})

    @mcp.tool()
    def tmux_msg_read(agent_id: str) -> str:
        """Read messages from an agent's inbox."""
        msgs = read_inbox_data(agent_id)
        return json.dumps(msgs, indent=2)

# --- CLI Dispatcher ---


def main():
    parser = argparse.ArgumentParser(
        description="Tmux agent orchestration with messaging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    # If no arguments, and MCP is available, run MCP
    if len(sys.argv) == 1 and mcp:
        mcp.run()
        return

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # spawn
    p_spawn = subparsers.add_parser("spawn", help="Spawn agent session")
    p_spawn.add_argument("--id", help="Agent ID (auto-generated if not provided)")
    p_spawn.add_argument("--cwd", help="Working directory")
    p_spawn.add_argument("prog", nargs="*", help="Command to run")

    # kill
    p_kill = subparsers.add_parser("kill", help="Kill agent session")
    p_kill.add_argument("--id", required=True, help="Agent ID")

    # send
    p_send = subparsers.add_parser("send", help="Send text to agent")
    p_send.add_argument("--id", required=True, help="Agent ID")
    p_send.add_argument("--literal", "-l", action="store_true")
    p_send.add_argument("text", help="Text to send")

    # read
    p_read = subparsers.add_parser("read", help="Read agent output")
    p_read.add_argument("--id", required=True, help="Agent ID")
    p_read.add_argument("--lines", type=int, help="Number of lines")
    p_read.add_argument("--save", action="store_true", help="Save to output.log")

    # pipe
    p_pipe = subparsers.add_parser("pipe", help="Pipe file to agent")
    p_pipe.add_argument("--id", required=True, help="Agent ID")
    p_pipe.add_argument("file", help="File to pipe")

    # list
    p_list = subparsers.add_parser("list", help="List agent sessions")
    p_list.add_argument("--json", action="store_true")

    # attach
    p_attach = subparsers.add_parser("attach", help="Attach to agent session")
    p_attach.add_argument("--id", required=True, help="Agent ID")

    # === MESSAGE BROKER ===

    # msg
    p_msg = subparsers.add_parser("msg", help="Post message to agent")
    p_msg.add_argument("--to", required=True, help="Target agent ID")
    p_msg.add_argument("--from", dest="from_agent", default="sfa", help="Sender ID")
    p_msg.add_argument("message", help="Message content")

    # inbox
    p_inbox = subparsers.add_parser("inbox", help="Read agent inbox")
    p_inbox.add_argument("--id", required=True, help="Agent ID")
    p_inbox.add_argument("--json", action="store_true")

    # ack
    p_ack = subparsers.add_parser("ack", help="Acknowledge messages")
    p_ack.add_argument("--id", required=True, help="Agent ID")
    p_ack.add_argument("--all", action="store_true", help="Ack all messages")
    p_ack.add_argument("--msg", help="Specific message ID to ack")

    args = parser.parse_args()

    if args.command == "spawn":
        # 'prog' is a list[str] from argparse (nargs='*') which can be empty list.
        # Ensure we pass the list or None if empty.
        cmd_arg: list[str] | None = args.prog if args.prog else None
        success, output = spawn_agent(agent_id=args.id, command=cmd_arg, cwd=args.cwd)

    elif args.command == "kill":
        success, output = kill_agent(args.id)

    elif args.command == "send":
        success, output = send_to_agent(args.id, args.text, args.literal)

    elif args.command == "read":
        success, output = read_agent(args.id, args.lines, args.save)

    elif args.command == "pipe":
        success, output = pipe_to_agent(args.id, args.file)

    elif args.command == "list":
        agents = list_agents_data()
        success = True
        if args.json:
            output = json.dumps(agents, indent=2)
        else:
            if not agents:
                output = "No agent sessions"
            else:
                lines_out = []
                for a in agents:
                    inbox_str = f" [{a['inbox']} msg]" if a["inbox"] > 0 else ""
                    lines_out.append(f"{a['id']}: {a['status']}{inbox_str}")
                output = "\n".join(lines_out)

    elif args.command == "attach":
        success, output = attach_agent(args.id)

    elif args.command == "msg":
        success, output = post_message(args.to, args.message, args.from_agent)

    elif args.command == "inbox":
        msgs = read_inbox_data(args.id)
        success = True
        if args.json:
            output = json.dumps(msgs, indent=2)
        else:
            if not msgs:
                output = "No messages"
            else:
                lines_out = []
                for m in msgs:
                    lines_out.append(
                        f"[{m['id']}] from {m['from']}: {m['message'][:80]}..."
                    )
                output = "\n".join(lines_out)

    elif args.command == "ack":
        success, output = ack_messages(args.id, args.msg, args.all)

    else:
        # Fallback if command not matched (should satisfy argparse)
        parser.print_help()
        sys.exit(1)

    if success:
        print(output)
        sys.exit(0)
    else:
        print(f"Error: {output}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
