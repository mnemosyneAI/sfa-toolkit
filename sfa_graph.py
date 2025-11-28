# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "fastmcp",
#     "torch>=2.0.0",
#     "onnxruntime-gpu>=1.18.0,<1.20.0",
#     "fastembed-gpu",
#     "numpy",
#     "sqlite-utils",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cu121"
# url = "https://download.pytorch.org/whl/cu121"
# explicit = true
#
# [[tool.uv.index]]
# name = "onnxruntime-cu12"
# url = "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/"
# explicit = true
#
# [tool.uv.sources]
# torch = [
#     { index = "pytorch-cu121", marker = "sys_platform == 'win32'" },
# ]
# onnxruntime-gpu = [
#     { index = "onnxruntime-cu12", marker = "sys_platform == 'win32'" },
# ]
# ///
#
# GPU ACCELERATION CONFIGURATION (RTX 4090):
# - fastembed-gpu includes onnxruntime-gpu for CUDA support
# - PyTorch CUDA 12.1 wheel from pytorch.org index (not PyPI)
# - explicit=true ensures only torch/torchvision use this index
# - marker limits to Windows (CUDA builds not available on macOS)
# - Must call TextEmbedding(..., providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
# - Result: 10-14x speedup (32 items/sec CPU -> 447 items/sec GPU @ batch=20)

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
        "syne": "Syne",
        "john": "John",
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

# Lazy-load embedding model (expensive initialization)
_embedding_model = None


def _get_embedding_model():
    """Get or initialize the GPU-accelerated embedding model."""
    global _embedding_model
    if _embedding_model is None:
        try:
            # Preload CUDA/cuDNN DLLs from PyTorch
            # This makes PyTorch's bundled CUDA libs available to ONNX Runtime
            import torch  # This loads CUDA DLLs into process memory

            from fastembed import TextEmbedding

            _embedding_model = TextEmbedding(
                model_name="nomic-ai/nomic-embed-text-v1.5",
                providers=[
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                ],  # GPU -> CPU fallback
            )
        except ImportError as e:
            raise RuntimeError(f"fastembed-gpu not installed: {e}")
    return _embedding_model


def _generate_embedding(text: str) -> List[float]:
    """Generate embedding vector from text using GPU-accelerated model."""
    model = _get_embedding_model()
    embeddings = list(model.embed([text]))
    return embeddings[0].tolist()


def _write_semantics_row(
    semantics_path: Path, node_id: str, semantic_text: str, embedding: List[float],
    archived_date: str = "ACTIVE"
) -> None:
    """Write or update a row in graph_semantics.tsv."""
    import csv

    SEMANTICS_FIELDS = ["archived_date", "id", "semantic_text", "embedding"]

    # Read existing rows
    rows = []
    if semantics_path.exists():
        with open(semantics_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for r in reader:
                if r["id"] != node_id:
                    # Migrate old rows without archived_date
                    if "archived_date" not in r or not r.get("archived_date"):
                        r["archived_date"] = "ACTIVE"
                    rows.append(r)

    # Add new row
    rows.append(
        {
            "archived_date": archived_date,
            "id": node_id,
            "semantic_text": semantic_text,
            "embedding": json.dumps(embedding) if isinstance(embedding, list) else embedding,
        }
    )

    # Write all rows
    with open(semantics_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SEMANTICS_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


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
    if any(r["id"] == new_id and r.get("archived_date") == "ACTIVE" for r in rows):
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
        "archived_date": "ACTIVE",
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

    # DUAL-WRITE: Generate embedding and write to graph_semantics.tsv
    try:
        embedding = _generate_embedding(semantic_text)
        semantics_path = target.parent / "graph_semantics.tsv"
        _write_semantics_row(semantics_path, new_id, semantic_text, embedding)
    except Exception as e:
        # Rollback: remove the node we just added
        rows = [r for r in rows if r["id"] != new_id]
        _write_graph(target, rows)
        return json.dumps({"error": f"Failed to generate embedding: {e}"}, indent=2)

    # NOTE: No cache invalidation needed - per-row embeddings in graph_semantics.tsv

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
    include_archived: bool = False,
) -> str:
    """Add a link (edge) between two nodes."""
    target = _normalize_path(path)
    rows = _read_graph(target)

    # Validate existence of source and target
    if include_archived:
        valid_ids = {r["id"] for r in rows}
    else:
        valid_ids = {r["id"] for r in rows if r.get("archived_date") == "ACTIVE"}
    if source_id not in valid_ids:
        return f"Error: Source ID '{source_id}' not found" + (" or archived." if not include_archived else ".")
    if target_id not in valid_ids:
        return f"Error: Target ID '{target_id}' not found" + (" or archived." if not include_archived else ".")

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
        "archived_date": "ACTIVE",
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

    # DUAL-WRITE: Generate embedding and write to graph_semantics.tsv
    link_id = new_row["id"]
    try:
        embedding = _generate_embedding(semantic_text)
        semantics_path = target.parent / "graph_semantics.tsv"
        _write_semantics_row(semantics_path, link_id, semantic_text, embedding)
    except Exception as e:
        # Rollback: remove the link we just added
        rows = [r for r in rows if r["id"] != link_id]
        _write_graph(target, rows)
        return json.dumps({"error": f"Failed to generate embedding: {e}"}, indent=2)

    # NOTE: No cache invalidation needed - per-row embeddings in graph_semantics.tsv

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
        if r["id"] == id and r.get("archived_date") == "ACTIVE":
            active_idx = i
            break

    if active_idx == -1:
        return f"Error: No active node found with ID '{id}'"

    old_row = rows[active_idx]

    # Archive old row
    rows[active_idx]["archived_date"] = _now_iso()

    # Create new row
    new_row = old_row.copy()
    new_row["archived_date"] = "ACTIVE"
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


def _archive_node_impl(path: str, id: str, reason: Optional[str] = None) -> str:
    """Archive a node without creating a new version. For compaction/superseded entries."""
    import csv

    target = _normalize_path(path)
    rows = _read_graph(target)

    # Find active row
    active_idx = -1
    for i, r in enumerate(rows):
        if r["id"] == id and r.get("archived_date") == "ACTIVE":
            active_idx = i
            break

    if active_idx == -1:
        return f"Error: No active node found with ID '{id}'"

    # Archive the row (no new version created)
    archive_timestamp = _now_iso()
    rows[active_idx]["archived_date"] = archive_timestamp

    # Optionally append reason to content
    if reason:
        old_content = rows[active_idx].get("content", "")
        rows[active_idx]["content"] = f"{old_content} [ARCHIVED: {reason}]"

    _write_graph(target, rows)

    # SYNC: Also update graph_semantics.tsv
    semantics_path = target.parent / "graph_semantics.tsv"
    if semantics_path.exists():
        SEMANTICS_FIELDS = ["archived_date", "id", "semantic_text", "embedding"]
        sem_rows = []
        with open(semantics_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for r in reader:
                # Migrate old rows without archived_date
                if "archived_date" not in r or not r.get("archived_date"):
                    r["archived_date"] = "ACTIVE"
                # Update the archived entry
                if r["id"] == id:
                    r["archived_date"] = archive_timestamp
                sem_rows.append(r)

        with open(semantics_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SEMANTICS_FIELDS, delimiter="\t")
            writer.writeheader()
            writer.writerows(sem_rows)

    return f"Archived {id}. No replacement created."


def _unarchive_node_impl(path: str, id: str) -> str:
    """Restore an archived node to active status."""
    import csv

    target = _normalize_path(path)
    rows = _read_graph(target)

    # Find archived row
    archived_idx = -1
    for i, r in enumerate(rows):
        if r["id"] == id and r.get("archived_date") != "ACTIVE":
            archived_idx = i
            break

    if archived_idx == -1:
        return f"Error: No archived node found with ID '{id}'"

    # Unarchive the row
    rows[archived_idx]["archived_date"] = "ACTIVE"

    _write_graph(target, rows)

    # SYNC: Also update graph_semantics.tsv
    semantics_path = target.parent / "graph_semantics.tsv"
    if semantics_path.exists():
        SEMANTICS_FIELDS = ["archived_date", "id", "semantic_text", "embedding"]
        sem_rows = []
        with open(semantics_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for r in reader:
                if r["id"] == id:
                    r["archived_date"] = "ACTIVE"
                sem_rows.append(r)

        with open(semantics_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SEMANTICS_FIELDS, delimiter="\t")
            writer.writeheader()
            writer.writerows(sem_rows)

    return f"Unarchived {id}. Entry is now active."


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
        if active_only and r.get("archived_date") != "ACTIVE":
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
        '  awk -F\'\\t\' \'$4=="aspiration" && $8=="syne" {print $6": "$11}\' graph.tsv'
    )
    output.append("\n  # Stance distribution")
    output.append("  cut -f4 graph.tsv | sort | uniq -c")

    return "\n".join(output)


def _semantic_query_impl(path: str, query: str, top_k: int = 10) -> str:
    """
    Semantic search on graph using embeddings.
    Searches graph_semantics.tsv and returns matching graph IDs with full data.
    """
    import numpy as np

    target = _normalize_path(path)
    semantics_path = target.parent / "graph_semantics.tsv"

    if not semantics_path.exists():
        return json.dumps(
            {"error": "graph_semantics.tsv not found. No embeddings available."},
            indent=2,
        )

    # Generate query embedding
    model = _get_embedding_model()
    query_embedding = _generate_embedding(query)
    query_vec = np.array(query_embedding)

    # Read graph_semantics.tsv and compute similarities
    import csv

    candidates = []

    with open(semantics_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            # Skip archived entries (filter early for efficiency)
            archived = row.get("archived_date", "ACTIVE")
            if archived != "ACTIVE":
                continue

            node_id = row["id"]
            semantic_text = row["semantic_text"]
            embedding = json.loads(row["embedding"])
            embedding_vec = np.array(embedding)

            # Cosine similarity
            similarity = np.dot(query_vec, embedding_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(embedding_vec)
            )

            candidates.append(
                {
                    "id": node_id,
                    "semantic_text": semantic_text,
                    "score": float(similarity),
                }
            )

    # Sort by similarity
    candidates.sort(key=lambda x: x["score"], reverse=True)
    top_matches = candidates[:top_k]

    # Lookup full graph entries
    rows = _read_graph(target)
    id_to_row = {r["id"]: r for r in rows if r.get("archived_date") == "ACTIVE"}

    results = []
    for match in top_matches:
        node_id = match["id"]
        if node_id in id_to_row:
            full_data = id_to_row[node_id].copy()
            full_data["_semantic_score"] = match["score"]
            results.append(full_data)

    return json.dumps(results, indent=2)


def _regenerate_semantics_impl(graph_path: str, force: bool = False) -> str:
    """
    Regenerate graph_semantics.tsv from graph.tsv.
    If force=True, regenerate all embeddings.
    If force=False, only add missing embeddings.
    """
    import csv

    target = _normalize_path(graph_path)

    if not target.exists():
        return json.dumps({"error": "Graph file not found"})

    semantics_path = target.parent / "graph_semantics.tsv"

    # Read existing semantics if not forcing
    existing_ids = set()
    if not force and semantics_path.exists():
        with open(semantics_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            existing_ids = {row["id"] for row in reader}

    # Read graph.tsv
    rows = _read_graph(target)
    active_rows = [r for r in rows if r.get("archived_date") == "ACTIVE"]

    total = len(active_rows)
    if total == 0:
        return json.dumps({"status": "no_active_rows", "total": 0})

    # Initialize embedding model
    print(f"Initializing GPU-accelerated embedding model...", file=sys.stderr)
    model = _get_embedding_model()
    print(f"✓ Embedding model ready", file=sys.stderr)

    # Determine which rows need embeddings
    to_process = []
    for row in active_rows:
        row_id = row["id"]
        if force or row_id not in existing_ids:
            to_process.append(row)

    if len(to_process) == 0:
        return json.dumps(
            {"status": "all_current", "total": total, "processed": 0, "skipped": total}
        )

    print(f"Processing {len(to_process)}/{total} rows...", file=sys.stderr)

    # If forcing, recreate file from scratch
    if force:
        with open(semantics_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["archived_date", "id", "semantic_text", "embedding"])

    # Process rows and write
    processed = 0
    errors = 0

    for idx, row in enumerate(to_process, 1):
        try:
            row_id = row["id"]
            semantic_text = row["semantic_text"]

            # Generate embedding
            embedding = _generate_embedding(semantic_text)

            # Write to semantics file (ACTIVE since we only process active rows)
            archived_date = row.get("archived_date", "ACTIVE")
            _write_semantics_row(semantics_path, row_id, semantic_text, embedding, archived_date)

            processed += 1
            if processed % 10 == 0:
                print(
                    f"[{processed}/{len(to_process)}] Processed {row_id}",
                    file=sys.stderr,
                )

        except Exception as e:
            errors += 1
            print(
                f"✗ Error processing {row.get('id', 'unknown')}: {e}", file=sys.stderr
            )

    result = {
        "status": "complete",
        "total": total,
        "processed": processed,
        "skipped": total - len(to_process),
        "errors": errors,
    }

    print(
        f"\n✓ Regeneration complete: {processed} embeddings generated", file=sys.stderr
    )
    return json.dumps(result, indent=2)


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
    p_link.add_argument("--include-archived", action="store_true", help="Allow linking to archived entries")

    # Update
    p_update = subparsers.add_parser("update", help="Update a node (archive & append)")
    p_update.add_argument("path", help="Path to graph.tsv")
    p_update.add_argument("id", help="Node ID")
    p_update.add_argument("--content", help="New content")
    p_update.add_argument("--stance", help="New stance")

    # Archive (pure archival, no replacement)
    p_archive = subparsers.add_parser("archive", help="Archive a node (no replacement)")
    p_archive.add_argument("path", help="Path to graph.tsv")
    p_archive.add_argument("id", help="Node ID to archive")
    p_archive.add_argument("--reason", help="Optional reason for archival")

    # Unarchive (restore archived entry to active)
    p_unarchive = subparsers.add_parser("unarchive", help="Restore an archived node to active")
    p_unarchive.add_argument("path", help="Path to graph.tsv")
    p_unarchive.add_argument("id", help="Node ID to unarchive")

    # Query
    p_query = subparsers.add_parser("query", help="Query the graph")
    p_query.add_argument("path", help="Path to graph.tsv")
    p_query.add_argument("--stance", help="Filter by stance")
    p_query.add_argument("--domain", help="Filter by domain")
    p_query.add_argument(
        "--all-history", action="store_true", help="Include archived rows"
    )

    # Semantic Query
    p_semantic = subparsers.add_parser(
        "semantic-query", help="Semantic search on graph"
    )
    p_semantic.add_argument("path", help="Path to graph.tsv")
    p_semantic.add_argument("query", help="Natural language query")
    p_semantic.add_argument("--top", type=int, default=10, help="Number of results")
    p_semantic.add_argument(
        "--db-output",
        metavar="DB_PATH",
        help="Write results to SQLite database instead of stdout (silent mode)",
    )
    p_semantic.add_argument(
        "--db-table",
        default="graph_results",
        help="Database table name (default: graph_results)",
    )

    # Regenerate Semantics
    p_regen = subparsers.add_parser(
        "regenerate-semantics", help="Rebuild graph_semantics.tsv from graph.tsv"
    )
    p_regen.add_argument("path", help="Path to graph.tsv")
    p_regen.add_argument(
        "--force",
        action="store_true",
        help="Regenerate all embeddings (not just missing)",
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
                include_archived=getattr(args, "include_archived", False),
            )
        )
    elif args.command == "update":
        print(
            _update_node_impl(
                args.path, args.id, content=args.content, stance=args.stance
            )
        )
    elif args.command == "archive":
        print(_archive_node_impl(args.path, args.id, reason=args.reason))
    elif args.command == "unarchive":
        print(_unarchive_node_impl(args.path, args.id))
    elif args.command == "query":
        print(
            _query_graph_impl(
                args.path,
                stance=args.stance,
                domain=args.domain,
                active_only=not args.all_history,
            )
        )
    elif args.command == "semantic-query":
        results_json = _semantic_query_impl(args.path, args.query, top_k=args.top)

        if hasattr(args, "db_output") and args.db_output:
            # Silent mode: write to database
            import sqlite_utils

            results = json.loads(results_json)

            # Add metadata
            enriched_results = []
            for r in results:
                r_copy = r.copy()
                r_copy["query"] = args.query
                r_copy["timestamp"] = __import__("datetime").datetime.now().isoformat()
                enriched_results.append(r_copy)

            # Insert into DB
            db = sqlite_utils.Database(args.db_output)
            table = db[args.db_table]
            table.insert_all(enriched_results, alter=True, replace=True)
            # Silent: no output
        else:
            # Normal mode: print to stdout
            print(results_json)
    elif args.command == "regenerate-semantics":
        print(_regenerate_semantics_impl(args.path, force=args.force))
    elif args.command == "schema":
        print(_schema_impl())
    else:
        if mcp:
            mcp.run()
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
