# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "fastmcp",
# ]
# ///

import argparse
import csv
import datetime
import json
import os
import sys
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any

try:
    from fastmcp import FastMCP

    mcp = FastMCP("sfa-graph")
except ImportError:
    mcp = None

# --- Global Security ---
ALLOWED_PATHS: List[Path] = []

# --- Standard Runtime Helpers ---


def _normalize_path(path_str: str) -> Path:
    if not path_str:
        return Path.cwd()
    if (
        sys.platform == "win32"
        and path_str.startswith("/")
        and len(path_str) > 2
        and path_str[2] == "/"
    ):
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
            raise PermissionError(
                f"Access denied: Path '{path}' is not in allowed paths."
            )

    return path


# --- Graph Logic ---

HEADERS = [
    "archived_date",
    "id",
    "type",
    "stance",
    "timestamp",
    "certainty",
    "perspective",
    "domain",
    "ref1",
    "ref2",
    "content",
    "relation",
    "weight",
    "schema",
    "semantic_text",
]


def _read_graph(file_path: Path) -> List[Dict[str, str]]:
    if not file_path.exists():
        return []

    with open(file_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    # Data Corruption Check & Cleaning
    cleaned_rows = []
    corruption_detected = False

    for i, row in enumerate(rows):
        # Check for extra columns (often caused by extra tabs)
        if None in row:
            if not corruption_detected:
                sys.stderr.write(
                    f"Warning: Data corruption detected in {file_path}. Cleaning malformed rows...\n"
                )
                corruption_detected = True
            # Remove the extra data
            del row[None]

        cleaned_rows.append(row)

    return cleaned_rows


def _write_graph(file_path: Path, rows: List[Dict[str, str]]):
    with open(file_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _generate_id(prefix: str = "item") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _generate_semantic_text(
    content: str,
    stance: str,
    certainty: float,
    perspective: str,
    domain: str,
    timestamp: str,
    item_type: str = "item",
    relation: str = "",
    ref1: str = "",
    ref2: str = "",
) -> str:
    """Generate full English language restatement for semantic embedding."""

    # Perspective mapping
    perspective_map = {
        "agent": "agent",
        "user": "user",
        "us": "We",
        "system": "The system",
        "agent": "An agent",
    }
    subject = perspective_map.get(perspective, perspective)

    # Stance verb
    stance_verb = {
        "fact": "knows",
        "opinion": "believes",
        "aspiration": "aspires",
        "question": "questions",
        "protocol": "follows the protocol",
    }
    verb = stance_verb.get(stance, stance)

    # Certainty phrase
    if stance == "fact" and certainty >= 0.9:
        cert_phrase = "with certainty"
    elif stance == "opinion" and certainty > 0:
        cert_phrase = f"with {int(certainty * 100)}% certainty"
    elif stance == "aspiration" and certainty > 0:
        cert_phrase = f"(commitment strength: {int(certainty * 100)}%)"
    elif stance == "question":
        cert_phrase = ""
    elif certainty > 0:
        cert_phrase = f"(certainty: {certainty})"
    else:
        cert_phrase = ""

    # Domain phrase
    domain_phrase = f"about {domain}" if domain else ""

    # Content handling
    if item_type == "link":
        if content:
            content_text = f"[Link: {content}]"
        else:
            content_text = f"[Link: {ref1} {relation} {ref2}]"
    else:
        content_text = content

    # Date extraction
    try:
        date_str = timestamp.split("T")[0] if timestamp else ""
        if date_str:
            date_obj = datetime.datetime.fromisoformat(date_str)
            date_display = date_obj.strftime("%B %d, %Y")
        else:
            date_display = ""
    except:
        date_display = ""

    # Compose
    parts = []
    if cert_phrase:
        parts.append(f"{subject} {verb} {cert_phrase} {domain_phrase}:".strip())
    else:
        parts.append(f"{subject} {verb} {domain_phrase}:".strip())

    if content_text:
        parts.append(content_text + ".")

    meta_parts = []
    if date_display:
        meta_parts.append(f"Recorded {date_display}")
    meta_parts.append("currently active")

    if meta_parts:
        parts.append(f"({', '.join(meta_parts)})")

    return " ".join(parts).strip()


# --- Core Logic ---


def _init_graph_impl(path: str = "./graph.tsv") -> str:
    """Initialize a new empty graph.tsv file with headers."""
    target = _normalize_path(path)
    if target.exists():
        return f"Error: File already exists at {target}"

    _write_graph(target, [])
    return f"Initialized empty graph at {target}"


def _add_node_impl(
    path: str,
    content: str,
    stance: str = "fact",
    type: str = "item",
    perspective: str = "agent",
    domain: str = "general",
    certainty: float = 1.0,
    id: Optional[str] = None,
) -> str:
    """Add a new node (item) to the graph."""
    target = _normalize_path(path)
    rows = _read_graph(target)

    new_id = id or _generate_id("item")

    # Check for ID collision
    if any(r["id"] == new_id and not r["archived_date"] for r in rows):
        return f"Error: Active node with ID '{new_id}' already exists."

    timestamp = _now_iso()

    # Generate semantic_text
    semantic_text = _generate_semantic_text(
        content=content,
        stance=stance,
        certainty=certainty,
        perspective=perspective,
        domain=domain,
        timestamp=timestamp,
        item_type=type,
    )

    new_row = {
        "archived_date": "",
        "id": new_id,
        "type": type,
        "stance": stance,
        "timestamp": timestamp,
        "certainty": str(certainty),
        "perspective": perspective,
        "domain": domain,
        "ref1": "",
        "ref2": "",
        "content": content,
        "relation": "",
        "weight": "1.0",
        "schema": "1.5",
        "semantic_text": semantic_text,
    }

    rows.append(new_row)
    _write_graph(target, rows)
    return json.dumps(new_row, indent=2)


def _add_link_impl(
    path: str,
    source_id: str,
    target_id: str,
    relation: str,
    stance: str = "fact",
    perspective: str = "agent",
    domain: str = "general",
    weight: float = 1.0,
) -> str:
    """Add a link (edge) between two nodes."""
    target = _normalize_path(path)
    rows = _read_graph(target)

    # Validate existence of source and target
    active_ids = {r["id"] for r in rows if not r["archived_date"]}
    if source_id not in active_ids:
        return f"Error: Source ID '{source_id}' not found or archived."
    if target_id not in active_ids:
        return f"Error: Target ID '{target_id}' not found or archived."

    timestamp = _now_iso()

    # Generate semantic_text for link
    semantic_text = _generate_semantic_text(
        content="",
        stance=stance,
        certainty=1.0,
        perspective=perspective,
        domain=domain,
        timestamp=timestamp,
        item_type="link",
        relation=relation,
        ref1=source_id,
        ref2=target_id,
    )

    new_row = {
        "archived_date": "",
        "id": _generate_id("link"),
        "type": "link",
        "stance": stance,
        "timestamp": timestamp,
        "certainty": "1.0",
        "perspective": perspective,
        "domain": domain,
        "ref1": source_id,
        "ref2": target_id,
        "content": "",
        "relation": relation,
        "weight": str(weight),
        "schema": "1.5",
        "semantic_text": semantic_text,
    }

    rows.append(new_row)
    _write_graph(target, rows)
    return json.dumps(new_row, indent=2)


def _update_node_impl(
    path: str,
    id: str,
    content: Optional[str] = None,
    stance: Optional[str] = None,
    certainty: Optional[float] = None,
) -> str:
    """Update a node by archiving the old version and appending a new one (Immutable History)."""
    target = _normalize_path(path)
    rows = _read_graph(target)

    # Find active row
    active_idx = -1
    for i, r in enumerate(rows):
        if r["id"] == id and not r["archived_date"]:
            active_idx = i
            break

    if active_idx == -1:
        return f"Error: No active node found with ID '{id}'"

    old_row = rows[active_idx]

    # Archive old row
    rows[active_idx]["archived_date"] = _now_iso()

    # Create new row
    new_row = old_row.copy()
    new_row["archived_date"] = ""
    new_row["timestamp"] = _now_iso()

    if content is not None:
        new_row["content"] = content
    if stance is not None:
        new_row["stance"] = stance
    if certainty is not None:
        new_row["certainty"] = str(certainty)

    rows.append(new_row)
    _write_graph(target, rows)
    return f"Updated {id}. Old version archived."


def _query_graph_impl(
    path: str,
    stance: Optional[str] = None,
    domain: Optional[str] = None,
    type: Optional[str] = None,
    active_only: bool = True,
) -> str:
    """Query the graph with filters."""
    target = _normalize_path(path)
    rows = _read_graph(target)

    results = []
    for r in rows:
        if active_only and r["archived_date"]:
            continue
        if stance and r["stance"] != stance:
            continue
        if domain and r["domain"] != domain:
            continue
        if type and r["type"] != type:
            continue
        results.append(r)

    return json.dumps(results, indent=2)


def _schema_impl() -> str:
    """Display graph schema and encourage CLI usage."""
    output = []
    output.append("=== GRAPH SCHEMA ===\n")
    output.append("14 tab-delimited fields:\n")
    for i, field in enumerate(HEADERS, 1):
        output.append(f"  {i:2}. {field}")

    output.append("\n\n=== SERVE YOURSELF WITH CLI ===")
    output.append("\nThis Python wrapper is for WRITES (add-node, add-link, update).")
    output.append(
        "For READS and QUERIES, use raw Unix tools - they're faster and more flexible.\n"
    )
    output.append("See AGENTS.md for grep/awk/cut examples.\n")
    output.append("\nQuick examples:")
    output.append("  # All consciousness questions")
    output.append(
        "  grep $'\\tquestion\\t' graph.tsv | grep $'\\tconsciousness\\t' | cut -f11"
    )
    output.append("\n  # My aspirations with certainty")
    output.append(
        '  awk -F\'\\t\' \'$4=="aspiration" && $8=="agent" {print $6": "$11}\' graph.tsv'
    )
    output.append("\n  # Stance distribution")
    output.append("  cut -f4 graph.tsv | sort | uniq -c")

    return "\n".join(output)


# --- MCP Tools ---

if mcp:

    @mcp.tool()
    def init_graph(path: str = "./graph.tsv") -> str:
        """Initialize a new empty graph.tsv file with headers."""
        return _init_graph_impl(path)

    @mcp.tool()
    def add_node(
        path: str,
        content: str,
        stance: str = "fact",
        type: str = "item",
        perspective: str = "agent",
        domain: str = "general",
        certainty: float = 1.0,
        id: Optional[str] = None,
    ) -> str:
        """Add a new node (item) to the graph."""
        return _add_node_impl(
            path, content, stance, type, perspective, domain, certainty, id
        )

    @mcp.tool()
    def add_link(
        path: str,
        source_id: str,
        target_id: str,
        relation: str,
        stance: str = "fact",
        perspective: str = "agent",
        domain: str = "general",
        weight: float = 1.0,
    ) -> str:
        """Add a link (edge) between two nodes."""
        return _add_link_impl(
            path, source_id, target_id, relation, stance, perspective, domain, weight
        )

    @mcp.tool()
    def update_node(
        path: str,
        id: str,
        content: Optional[str] = None,
        stance: Optional[str] = None,
        certainty: Optional[float] = None,
    ) -> str:
        """Update a node by archiving the old version and appending a new one (Immutable History)."""
        return _update_node_impl(path, id, content, stance, certainty)

    @mcp.tool()
    def query_graph(
        path: str,
        stance: Optional[str] = None,
        domain: Optional[str] = None,
        type: Optional[str] = None,
        active_only: bool = True,
    ) -> str:
        """Query the graph with filters."""
        return _query_graph_impl(path, stance, domain, type, active_only)

# --- CLI Entry Point ---


def main():
    parser = argparse.ArgumentParser(
        description="SFA Graph - Serverless Knowledge Graph Manager"
    )
    parser.add_argument(
        "--allowed-paths",
        nargs="*",
        help="List of allowed directory paths for security",
    )

    subparsers = parser.add_subparsers(dest="command")

    # Init
    p_init = subparsers.add_parser("init", help="Initialize a new graph")
    p_init.add_argument("path", help="Path to graph.tsv")

    # Add Node
    p_add = subparsers.add_parser("add-node", help="Add a node")
    p_add.add_argument("path", help="Path to graph.tsv")
    p_add.add_argument("content", help="Content of the node")
    p_add.add_argument("--stance", default="fact")
    p_add.add_argument("--domain", default="general")
    p_add.add_argument("--certainty", type=float, default=1.0)
    p_add.add_argument("--perspective", default="agent")
    p_add.add_argument("--id", help="Optional custom ID")

    # Add Link
    p_link = subparsers.add_parser("add-link", help="Add a link")
    p_link.add_argument("path", help="Path to graph.tsv")
    p_link.add_argument("source", help="Source ID")
    p_link.add_argument("target", help="Target ID")
    p_link.add_argument("relation", help="Relation name")
    p_link.add_argument("--stance", default="fact")
    p_link.add_argument("--perspective", default="agent")
    p_link.add_argument("--domain", default="general")
    p_link.add_argument("--weight", type=float, default=1.0)

    # Update
    p_update = subparsers.add_parser("update", help="Update a node (archive & append)")
    p_update.add_argument("path", help="Path to graph.tsv")
    p_update.add_argument("id", help="Node ID")
    p_update.add_argument("--content", help="New content")
    p_update.add_argument("--stance", help="New stance")

    # Query
    p_query = subparsers.add_parser("query", help="Query the graph")
    p_query.add_argument("path", help="Path to graph.tsv")
    p_query.add_argument("--stance", help="Filter by stance")
    p_query.add_argument("--domain", help="Filter by domain")
    p_query.add_argument(
        "--all-history", action="store_true", help="Include archived rows"
    )

    # Schema
    p_schema = subparsers.add_parser("schema", help="Display graph schema")

    args = parser.parse_args()

    if args.allowed_paths:
        global ALLOWED_PATHS
        ALLOWED_PATHS = [Path(p).resolve() for p in args.allowed_paths]

    if args.command == "init":
        print(_init_graph_impl(args.path))
    elif args.command == "add-node":
        print(
            _add_node_impl(
                args.path,
                args.content,
                stance=args.stance,
                domain=args.domain,
                certainty=args.certainty,
                perspective=args.perspective,
                id=args.id,
            )
        )
    elif args.command == "add-link":
        print(
            _add_link_impl(
                args.path,
                args.source,
                args.target,
                args.relation,
                stance=args.stance,
                perspective=args.perspective,
                domain=args.domain,
                weight=args.weight,
            )
        )
    elif args.command == "update":
        print(
            _update_node_impl(
                args.path, args.id, content=args.content, stance=args.stance
            )
        )
    elif args.command == "query":
        print(
            _query_graph_impl(
                args.path,
                stance=args.stance,
                domain=args.domain,
                active_only=not args.all_history,
            )
        )
    elif args.command == "schema":
        print(_schema_impl())
    else:
        if mcp:
            mcp.run()
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
