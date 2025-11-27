#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "torch>=2.0.0",
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
# [tool.uv.sources]
# torch = [
#     { index = "pytorch-cu121", marker = "sys_platform == 'win32'" },
# ]
# ///
#
# GPU ACCELERATION NOTES:
# - fastembed-gpu includes onnxruntime-gpu for CUDA support
# - PyTorch CUDA 12.1 wheel from pytorch.org index (not PyPI)
# - Must call TextEmbedding with providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
# - Result: 10-14x speedup (32 items/sec CPU -> 447 items/sec GPU @ batch=20)
"""
Unified Semantic Search - Query graph + knowledge + reference by meaning.

Search across all three data sources using natural language semantic queries.
Results ranked by similarity, formatted for readability.

Usage:
  uv run --script sfa_unified_search.py "how does Syne feel about John"
  uv run --script sfa_unified_search.py "consciousness exploration methods" --top 15
  uv run --script sfa_unified_search.py "partnership chemistry patterns" --sources graph,knowledge
  uv run --script sfa_unified_search.py "what is love" --format json
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

# --- Embedding & Similarity ---


def _cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def _embed_text(text: str, model) -> np.ndarray:
    """Embed text using provided model."""
    return list(model.embed([text]))[0]


# --- Graph Search ---


def search_graph(
    query_embedding: np.ndarray, graph_path: str, top_k: int = 20
) -> List[Dict[str, Any]]:
    """
    Search graph items by semantic similarity using per-row embeddings from graph_semantics.tsv.
    """
    import csv

    graph_file = Path(graph_path)
    if not graph_file.exists():
        return []

    # Check for graph_semantics.tsv
    semantics_file = graph_file.parent / "graph_semantics.tsv"
    if not semantics_file.exists():
        print(
            "Warning: graph_semantics.tsv not found, skipping graph search",
            file=sys.stderr,
        )
        return []

    # Read graph.tsv for metadata
    with open(graph_file, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        id_idx = header.index("id")
        content_idx = header.index("content")
        stance_idx = header.index("stance")
        certainty_idx = header.index("certainty")
        domain_idx = header.index("domain")
        archived_idx = header.index("archived_date")
        perspective_idx = header.index("perspective")
        semantic_idx = header.index("semantic_text")

        row_data = {}
        for line in f:
            if not line.strip():
                continue
            fields = line.rstrip("\r\n").split("\t")

            # Skip archived
            if fields[archived_idx] and fields[archived_idx] != "ACTIVE":
                continue

            row_id = fields[id_idx] if len(fields) > id_idx else ""
            row_data[row_id] = {
                "type": fields[stance_idx] if len(fields) > stance_idx else "item",
                "content": fields[content_idx] if len(fields) > content_idx else "",
                "semantic_text": fields[semantic_idx]
                if len(fields) > semantic_idx
                else "",
                "certainty": float(fields[certainty_idx])
                if len(fields) > certainty_idx and fields[certainty_idx]
                else 0.0,
                "domain": fields[domain_idx] if len(fields) > domain_idx else "",
                "perspective": fields[perspective_idx]
                if len(fields) > perspective_idx
                else "",
            }

    # Read embeddings from graph_semantics.tsv and score
    candidates = []
    with open(semantics_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            row_id = row["id"]
            if row_id not in row_data:
                continue

            embedding = json.loads(row["embedding"])
            embedding_vec = np.array(embedding)
            score = _cosine_similarity(query_embedding, embedding_vec)

            candidates.append(
                {
                    "source": "graph",
                    "id": row_id,
                    "score": float(score),
                    **row_data[row_id],
                }
            )

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_k]


# --- YMJ Search ---


def search_ymj(
    query_embedding: np.ndarray,
    root_path: str,
    doc_type: Optional[str] = None,
    top_k: int = 20,
) -> List[Dict[str, Any]]:
    """
    Search YMJ documents by semantic similarity.

    Args:
        query_embedding: Query vector
        root_path: Root directory to search
        doc_type: Filter by doc_type ('knowledge' or 'reference'), None for both
        top_k: Max results
    """
    root = Path(root_path)
    if not root.exists():
        return []

    candidates = []

    for fpath in root.rglob("*.ymj"):
        try:
            content = fpath.read_text(encoding="utf-8")

            # Extract footer embedding
            footer_end = content.rfind("```")
            if footer_end == -1:
                continue

            footer_start = content.rfind("```json", 0, footer_end)
            if footer_start == -1:
                continue

            footer_json = content[footer_start + 7 : footer_end].strip()
            footer = json.loads(footer_json)

            if "index" not in footer or "embedding" not in footer["index"]:
                continue

            vec = footer["index"]["embedding"]
            if not vec:
                continue

            # Extract header metadata
            summary = "No summary"
            file_doc_type = None

            if content.startswith("---"):
                try:
                    header_end = content.index("\n---", 3)
                    header_lines = content[3:header_end].splitlines()

                    for line in header_lines:
                        if line.strip().startswith("doc_summary:"):
                            summary = line.split(":", 1)[1].strip()
                        elif line.strip().startswith("doc_type:"):
                            file_doc_type = line.split(":", 1)[1].strip()
                except:
                    pass

            # Filter by doc_type if specified
            if doc_type and file_doc_type != doc_type:
                continue

            # Calculate similarity
            score = _cosine_similarity(query_embedding, vec)

            # Determine source category
            source = "knowledge" if "knowledge" in str(fpath) else "reference"

            candidates.append(
                {
                    "source": source,
                    "type": "ymj",
                    "path": str(fpath),
                    "filename": fpath.name,
                    "summary": summary,
                    "doc_type": file_doc_type or source,
                    "score": float(score),
                }
            )

        except Exception:
            continue

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_k]


# --- Unified Search ---


def unified_search(
    query: str,
    root_path: str = ".",
    sources: str = "graph,knowledge,reference",
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Unified semantic search across all three data sources.

    Args:
        query: Natural language query
        root_path: Root directory (default: current)
        sources: Comma-separated list of sources to search
        top_k: Total results to return (distributed across sources)
    """
    try:
        # CRITICAL: Import torch first to preload CUDA DLLs for ONNX Runtime
        import torch
        from fastembed import TextEmbedding
    except ImportError:
        return [{"error": "fastembed not installed"}]

    root = Path(root_path).resolve()
    source_list = [s.strip() for s in sources.split(",")]

    # Embed query with GPU acceleration
    print(f"Embedding query: '{query}'...", file=sys.stderr)
    model = TextEmbedding(
        model_name="nomic-ai/nomic-embed-text-v1.5",
        providers=[
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],  # GPU -> CPU fallback
    )
    query_embedding = _embed_text(query, model)

    all_results = []

    # Search graph
    if "graph" in source_list:
        # Try multiple possible locations
        possible_graph_paths = [
            root / "brain" / "graph.tsv",
            root / "graph" / "graph.tsv",
            root / ".opencode" / "graph" / "graph.tsv",
            root / "graph.tsv",
        ]
        graph_path = None
        for p in possible_graph_paths:
            if p.exists():
                graph_path = p
                break

        if graph_path:
            print(f"Searching graph...", file=sys.stderr)
            graph_results = search_graph(
                query_embedding, str(graph_path), top_k=top_k * 2
            )
            all_results.extend(graph_results)
            print(f"  Found {len(graph_results)} graph items", file=sys.stderr)
        else:
            print(
                f"  Graph not found (tried: brain/graph.tsv, graph/graph.tsv, .opencode/graph/graph.tsv)",
                file=sys.stderr,
            )

    # Search knowledge
    if "knowledge" in source_list:
        # Try multiple possible locations
        possible_knowledge_paths = [
            root / "brain" / "knowledge",
            root / "knowledge",
            root / ".opencode" / "knowledge",
        ]
        knowledge_path = None
        for p in possible_knowledge_paths:
            if p.exists() and p.is_dir():
                knowledge_path = p
                break

        if knowledge_path:
            print(f"Searching knowledge...", file=sys.stderr)
            knowledge_results = search_ymj(
                query_embedding, str(knowledge_path), doc_type="knowledge", top_k=top_k
            )
            all_results.extend(knowledge_results)
            print(f"  Found {len(knowledge_results)} knowledge docs", file=sys.stderr)
        else:
            print(
                f"  Knowledge directory not found (tried: brain/knowledge, knowledge, .opencode/knowledge)",
                file=sys.stderr,
            )

    # Search reference
    if "reference" in source_list:
        # Try multiple possible locations
        possible_reference_paths = [
            root / "brain" / "reference",
            root / "reference",
            root / ".opencode" / "reference",
        ]
        reference_path = None
        for p in possible_reference_paths:
            if p.exists() and p.is_dir():
                reference_path = p
                break

        if reference_path:
            print(f"Searching reference...", file=sys.stderr)
            reference_results = search_ymj(
                query_embedding, str(reference_path), doc_type="reference", top_k=top_k
            )
            all_results.extend(reference_results)
            print(f"  Found {len(reference_results)} reference docs", file=sys.stderr)
        else:
            print(
                f"  Reference directory not found (tried: brain/reference, reference, .opencode/reference)",
                file=sys.stderr,
            )

    # Sort all results by score
    all_results.sort(key=lambda x: x["score"], reverse=True)

    return all_results[:top_k]


# --- Output Formatting ---


def format_results(results: List[Dict[str, Any]], format_type: str = "readable") -> str:
    """Format search results for display."""

    if not results:
        return "No results found."

    if format_type == "json":
        return json.dumps(results, indent=2, default=str)

    # Readable format
    output = []
    output.append("\n" + "=" * 100)
    output.append(f"  UNIFIED SEMANTIC SEARCH RESULTS - {len(results)} items")
    output.append("=" * 100 + "\n")

    for i, result in enumerate(results, 1):
        score = result["score"]
        source = result["source"].upper()

        # Score bar visualization (ASCII safe)
        bar_length = int(score * 50)
        bar = "#" * bar_length + "-" * (50 - bar_length)

        output.append(f"#{i:2d}  {score:.4f}  [{bar}]")
        output.append(
            f"     SOURCE: {source:12s}  TYPE: {result.get('type', 'unknown')}"
        )

        if source == "GRAPH":
            # Graph item
            content = result.get("content", "")
            semantic = result.get("semantic_text", "")
            certainty = result.get("certainty", 0.0)
            domain = result.get("domain", "")
            perspective = result.get("perspective", "")

            output.append(
                f"     CERTAINTY: {certainty:.2f}  DOMAIN: {domain}  PERSPECTIVE: {perspective}"
            )
            output.append(f"     CONTENT: {content[:90]}...")
            output.append(f"     SEMANTIC: {semantic[:100]}...")

        else:
            # YMJ document
            filename = result.get("filename", result.get("path", "unknown"))
            summary = result.get("summary", "No summary")

            output.append(f"     FILE: {filename}")
            output.append(f"     SUMMARY: {summary[:85]}...")

        output.append("")  # Blank line between results

    output.append("=" * 100)

    return "\n".join(output)


# --- CLI ---


def main():
    parser = argparse.ArgumentParser(
        description="Unified Semantic Search - Query graph + knowledge + reference by meaning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "how does Syne feel about John"
  %(prog)s "consciousness exploration methods" --top 15
  %(prog)s "partnership chemistry patterns" --sources graph,knowledge
  %(prog)s "what is love" --format json
        """,
    )

    parser.add_argument("query", help="Natural language search query")
    parser.add_argument(
        "--path", default=".", help="Root path to search (default: current directory)"
    )
    parser.add_argument(
        "--sources",
        default="graph,knowledge,reference",
        help="Comma-separated sources to search (default: all)",
    )
    parser.add_argument(
        "--top", "-k", type=int, default=10, help="Max results to return (default: 10)"
    )
    parser.add_argument(
        "--format",
        choices=["readable", "json"],
        default="readable",
        help="Output format (default: readable)",
    )
    parser.add_argument(
        "--db-output",
        metavar="DB_PATH",
        help="Write results to SQLite database instead of stdout (silent mode)",
    )
    parser.add_argument(
        "--db-table",
        default="search_results",
        help="Database table name (default: search_results)",
    )

    args = parser.parse_args()

    # Run search
    results = unified_search(
        query=args.query, root_path=args.path, sources=args.sources, top_k=args.top
    )

    # Output results
    if args.db_output:
        # Silent mode: write to database
        import sqlite_utils

        db = sqlite_utils.Database(args.db_output)

        # Add metadata to each result
        enriched_results = []
        for r in results:
            r_copy = r.copy()
            r_copy["query"] = args.query
            r_copy["timestamp"] = __import__("datetime").datetime.now().isoformat()
            enriched_results.append(r_copy)

        # Insert into table (auto-creates/alters schema)
        table = db[args.db_table]
        table.insert_all(enriched_results, alter=True, replace=True)

        # Silent: no output to stdout
    else:
        # Normal mode: print to stdout
        output = format_results(results, args.format)
        print(output)


if __name__ == "__main__":
    main()
