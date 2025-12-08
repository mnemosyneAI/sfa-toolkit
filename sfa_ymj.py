# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "fastmcp",
#     "pyyaml",
#     "torch>=2.0.0",
#     "fastembed-gpu",
#     "numpy",
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
# - Must import torch BEFORE TextEmbedding to preload CUDA DLLs for ONNX Runtime
# - Must call TextEmbedding with providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
# - Result: 10-14x speedup (32 items/sec CPU -> 447 items/sec GPU @ batch=20)

import argparse
import hashlib
import json
import re
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

try:
    from fastmcp import FastMCP

    mcp = FastMCP("sfa-ymj")
except ImportError:
    mcp = None

import yaml

# --- Global Security ---
ALLOWED_PATHS: List[Path] = []


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


# --- Constants ---

YAML_FENCE = "---\n"
JSON_FENCE_START = "```json"
JSON_FENCE_END = "```"

# --- Core Logic: Parsing ---


class YMJDocument:
    def __init__(self, path: str):
        self.path = _normalize_path(str(path))
        self.raw_content = ""
        self.header: Dict[str, Any] = {}
        self.body: str = ""
        self.footer: Dict[str, Any] = {}
        self.valid_structure = False
        self.errors = []

        if self.path.exists():
            self.raw_content = self.path.read_text(encoding="utf-8")
            self._parse()

    def _parse(self):
        """Parse the Sandwich: YAML --- Markdown --- JSON"""
        content = self.raw_content

        # 1. Extract YAML Header
        # Must start with ---
        if not content.startswith("---"):
            self.errors.append(
                "Missing start YAML fence. Expected format:\n---\nkey: value\n---\n"
            )
            return

        # Find end of YAML
        # We look for the second ---
        # content[3:] skips the first ---
        try:
            header_end_idx = content.index("\n---", 3)
        except ValueError:
            self.errors.append(
                "Missing end YAML fence. Expected format:\n---\nkey: value\n---\n"
            )
            return

        header_str = content[3:header_end_idx].strip()
        try:
            self.header = yaml.safe_load(header_str) or {}
        except yaml.YAMLError as e:
            self.errors.append(f"Invalid YAML header: {e}")
            return

        # 2. Extract JSON Footer
        # Must be the last block
        # We search from the end
        footer_end_idx = content.rfind(JSON_FENCE_END)
        if footer_end_idx == -1:
            self.errors.append("Missing JSON footer end fence")
            return

        footer_start_idx = content.rfind(JSON_FENCE_START, 0, footer_end_idx)
        if footer_start_idx == -1:
            self.errors.append("Missing JSON footer start fence")
            return

        # Verify it's actually at the end (ignoring whitespace)
        if content[footer_end_idx + 3 :].strip() != "":
            self.errors.append("Content found after JSON footer")
            # We continue anyway for resilience

        footer_str = content[
            footer_start_idx + len(JSON_FENCE_START) : footer_end_idx
        ].strip()
        try:
            self.footer = json.loads(footer_str)
        except json.JSONDecodeError as e:
            self.errors.append(f"Invalid JSON footer: {e}")
            return

        # 3. Extract Markdown Body
        # Everything between YAML end fence and JSON start fence
        # YAML end fence is at header_end_idx. The fence itself is "\n---" (length 4) or "\n---\n"
        # We need to be careful with indices.

        # header_end_idx points to the newline before ---
        body_start = header_end_idx + 4  # Skip \n---
        if content[body_start] == "\n":
            body_start += 1  # Skip following newline if present

        body_end = footer_start_idx

        self.body = content[body_start:body_end].strip()
        self.valid_structure = len(self.errors) == 0

        # Validate header fields after structure check
        if self.valid_structure:
            self._validate_header()

    def _validate_header(self):
        """Validate required header fields based on file location."""
        # Check for doc_type field based on path
        path_str = str(self.path)

        if "knowledge" in path_str:
            expected_doc_type = "knowledge"
        elif "reference" in path_str:
            expected_doc_type = "reference"
        else:
            # Not in knowledge or reference directory, skip doc_type check
            return

        actual_doc_type = self.header.get("doc_type")

        if not actual_doc_type:
            self.errors.append(
                f"Missing required field: doc_type (expected '{expected_doc_type}')"
            )
            self.valid_structure = False
        elif actual_doc_type != expected_doc_type:
            self.errors.append(
                f"Invalid doc_type: '{actual_doc_type}' (expected '{expected_doc_type}' based on path)"
            )
            self.valid_structure = False

    def calculate_hash(self) -> str:
        """Calculate SHA256 of the body."""
        return hashlib.sha256(self.body.encode("utf-8")).hexdigest()

    def save(self):
        """Reconstruct and save the file."""
        new_content = "---\n"
        new_content += yaml.dump(self.header, sort_keys=False, allow_unicode=True)
        new_content += "---\n\n"
        new_content += self.body + "\n\n"
        new_content += "```json\n"
        new_content += json.dumps(self.footer, indent=2, ensure_ascii=False)
        new_content += "\n```\n"

        self.path.write_text(new_content, encoding="utf-8")

    def get_markdown(self) -> str:
        """Extract markdown body only (no YAML header or JSON footer)."""
        return self.body

    def get_yaml(self) -> str:
        """Extract YAML header as formatted string."""
        return yaml.dump(self.header, sort_keys=False, allow_unicode=True)

    def get_json(self) -> str:
        """Extract JSON footer as formatted string."""
        return json.dumps(self.footer, indent=2, ensure_ascii=False)


# --- Core Logic: Enrichment ---


def _enrich_document(
    doc: YMJDocument, use_embeddings: bool = True, force_embedding: bool = False
) -> Dict[str, Any]:
    """Update document footer with hash and embeddings."""
    if not doc.valid_structure:
        return {"error": "Invalid structure", "details": doc.errors}

    updates = []

    # 1. Hash (Always update if stale)
    current_hash = doc.calculate_hash()
    stored_hash = doc.footer.get("payload_hash")

    if current_hash != stored_hash:
        doc.footer["payload_hash"] = current_hash
        updates.append("hash")

    # 2. Embeddings (if hash changed, missing, or forced)
    # We check if we need to re-embed
    needs_embedding = (
        force_embedding
        or "hash" in updates
        or "index" not in doc.footer
        or "embedding" not in doc.footer.get("index", {})
    )

    if use_embeddings and needs_embedding:
        try:
            # CRITICAL: Import torch first to preload CUDA DLLs for ONNX Runtime
            import torch
            from fastembed import TextEmbedding

            # Initialize model (downloads if needed)
            embedding_model = TextEmbedding(
                model_name="nomic-ai/nomic-embed-text-v1.5",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )

            # Embed the body (truncate if too large? fastembed handles chunking usually, but for doc embedding we might want a summary or full text)
            # For now, embed the whole body
            vectors = list(embedding_model.embed([doc.body]))
            vector = vectors[0].tolist()  # Convert numpy to list

            if "index" not in doc.footer:
                doc.footer["index"] = {}

            doc.footer["index"]["embedding"] = vector
            doc.footer["index"]["model"] = "nomic-embed-text-v1.5"
            updates.append("embedding")

        except Exception as e:
            print(f"Warning: Embedding failed: {e}", file=sys.stderr)

    # 3. Keywords (Simple Frequency)
    # TODO: Implement TF-IDF or similar if needed. For now, skip to keep it simple or use LLM later.

    if updates:
        doc.save()
        return {"status": "updated", "updates": updates}
    else:
        return {"status": "no_changes"}


def _upgrade_document(doc: YMJDocument) -> Dict[str, Any]:
    """Upgrade YMJ document to latest spec compliance."""
    if not doc.valid_structure:
        return {"error": "Invalid structure", "details": doc.errors}

    updates = []

    # 1. Move payload_sha256 from header to footer payload_hash
    if "payload_sha256" in doc.header:
        del doc.header["payload_sha256"]
        updates.append("removed_header_hash")

    # 2. Ensure footer schema version
    if "schema" not in doc.footer:
        doc.footer["schema"] = "1"
        updates.append("added_schema_version")

    # 3. Ensure ymj_spec in header
    if "ymj_spec" not in doc.header:
        doc.header["ymj_spec"] = "1.0.0"
        updates.append("added_ymj_spec_version")

    # 4. Run enrichment (calculates hash, adds embeddings)
    enrich_res = _enrich_document(doc, use_embeddings=True)
    if enrich_res.get("status") == "updated":
        updates.extend(enrich_res.get("updates", []))

    if updates:
        doc.save()
        return {"status": "upgraded", "updates": updates}

    return {"status": "already_compliant"}


# --- Core Logic: Reading ---


def _read_markdown(path: str) -> str:
    """
    Extract markdown content from .ymj file(s), excluding YAML header and JSON footer.

    If path is a file: extracts markdown from that file
    If path is a folder: extracts markdown from all .ymj files in folder

    Returns markdown content as string
    """
    path_obj = _normalize_path(path)

    if not path_obj.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    # Collect files to process
    files_to_process = []

    if path_obj.is_file():
        if path_obj.suffix != ".ymj":
            raise ValueError("File is not a .ymj file")
        files_to_process.append(path_obj)
    elif path_obj.is_dir():
        # Find all .ymj files in directory
        files_to_process = list(path_obj.glob("*.ymj"))
        if not files_to_process:
            raise FileNotFoundError("No .ymj files found in directory")
    else:
        raise ValueError(f"Path type not supported: {path}")

    # Extract markdown from each file
    results = []

    for file_path in files_to_process:
        doc = YMJDocument(str(file_path))
        if not doc.valid_structure:
            print(
                f"Warning: Skipping {file_path.name} - invalid structure",
                file=sys.stderr,
            )
            continue

        markdown_content = doc.get_markdown()

        # Add file header if processing multiple files
        if len(files_to_process) > 1:
            results.append(f"# {file_path.name}\n\n{markdown_content}")
        else:
            results.append(markdown_content)

    if not results:
        raise ValueError("Failed to extract markdown from any files")

    # Combine all results
    return "\n\n---\n\n".join(results)


def _read_yaml(path: str) -> str:
    """
    Extract YAML header from .ymj file(s).

    Returns YAML content as string
    """
    path_obj = _normalize_path(path)

    if not path_obj.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    # Collect files to process
    files_to_process = []

    if path_obj.is_file():
        if path_obj.suffix != ".ymj":
            raise ValueError("File is not a .ymj file")
        files_to_process.append(path_obj)
    elif path_obj.is_dir():
        files_to_process = list(path_obj.glob("*.ymj"))
        if not files_to_process:
            raise FileNotFoundError("No .ymj files found in directory")
    else:
        raise ValueError(f"Path type not supported: {path}")

    # Extract YAML from each file
    results = []

    for file_path in files_to_process:
        doc = YMJDocument(str(file_path))
        if not doc.valid_structure:
            print(
                f"Warning: Skipping {file_path.name} - invalid structure",
                file=sys.stderr,
            )
            continue

        yaml_content = doc.get_yaml()

        # Add file header if processing multiple files
        if len(files_to_process) > 1:
            results.append(f"# {file_path.name}\n---\n{yaml_content}---")
        else:
            results.append(yaml_content)

    if not results:
        raise ValueError("Failed to extract YAML from any files")

    return "\n\n".join(results)


def _read_json(path: str) -> str:
    """
    Extract JSON footer from .ymj file(s).

    Returns JSON content as string
    """
    path_obj = _normalize_path(path)

    if not path_obj.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    # Collect files to process
    files_to_process = []

    if path_obj.is_file():
        if path_obj.suffix != ".ymj":
            raise ValueError("File is not a .ymj file")
        files_to_process.append(path_obj)
    elif path_obj.is_dir():
        files_to_process = list(path_obj.glob("*.ymj"))
        if not files_to_process:
            raise FileNotFoundError("No .ymj files found in directory")
    else:
        raise ValueError(f"Path type not supported: {path}")

    # Extract JSON from each file
    results = []

    for file_path in files_to_process:
        doc = YMJDocument(str(file_path))
        if not doc.valid_structure:
            print(
                f"Warning: Skipping {file_path.name} - invalid structure",
                file=sys.stderr,
            )
            continue

        json_content = doc.get_json()

        # Add file header if processing multiple files
        if len(files_to_process) > 1:
            results.append(f"# {file_path.name}\n{json_content}")
        else:
            results.append(json_content)

    if not results:
        raise ValueError("Failed to extract JSON from any files")

    return "\n\n".join(results)


# --- Core Logic: Migration ---


def _migrate_md(path: str) -> str:
    """Convert MD to YMJ."""
    p = Path(path)
    if p.suffix == ".ymj":
        return "Already YMJ"

    content = p.read_text(encoding="utf-8")

    # Basic heuristic to strip existing frontmatter if present
    header = {
        "doc_summary": "Auto-migrated document",
        "kind": "note",
        "version": "1.0.0",
        "created": "2025-11-19",  # Should use file creation time
    }

    # Check for existing YAML frontmatter
    body = content
    if content.startswith("---"):
        try:
            end_idx = content.index("\n---", 3)
            fm = content[3:end_idx]
            existing_header = yaml.safe_load(fm)
            if isinstance(existing_header, dict):
                header.update(existing_header)
            body = content[end_idx + 4 :].strip()
        except:
            pass  # Failed to parse existing frontmatter, treat as body

    # Create YMJ
    new_path = p.with_suffix(".ymj")

    doc = YMJDocument(str(new_path))
    # Manually construct since file doesn't exist or we are overwriting
    doc.header = header
    doc.body = body
    doc.footer = {
        "schema": "1",
        "payload_hash": "",  # Will be filled by enrich
        "index": {},
    }
    doc.save()

    # Enrich immediately to set hash
    doc = YMJDocument(str(new_path))
    _enrich_document(
        doc, use_embeddings=False
    )  # Skip embeddings for speed on migration

    return f"Migrated to {new_path}"


# --- MCP Tools ---

if mcp:

    @mcp.tool()
    def ymj_read(path: str) -> str:
        """Read a YMJ file and return parsed content."""
        doc = YMJDocument(path)
        if not doc.valid_structure:
            return json.dumps({"error": "Invalid YMJ", "details": doc.errors})

        return json.dumps(
            {"header": doc.header, "body": doc.body, "footer": doc.footer}, indent=2
        )

    @mcp.tool()
    def ymj_read_markdown(path: str) -> str:
        """Extract markdown content from .ymj file(s), excluding YAML header and JSON footer."""
        try:
            return _read_markdown(path)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    def ymj_read_yaml(path: str) -> str:
        """Extract YAML header from .ymj file(s)."""
        try:
            return _read_yaml(path)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    def ymj_read_json(path: str) -> str:
        """Extract JSON footer from .ymj file(s)."""
        try:
            return _read_json(path)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    def ymj_lint(path: str) -> str:
        """Check YMJ file validity."""
        doc = YMJDocument(path)
        status = {
            "valid": doc.valid_structure,
            "errors": doc.errors,
            "stale_hash": False,
        }
        if doc.valid_structure:
            calc_hash = doc.calculate_hash()
            stored_hash = doc.footer.get("payload_hash")
            if calc_hash != stored_hash:
                status["stale_hash"] = True
                status["errors"].append("Payload hash mismatch (stale footer)")

        return json.dumps(status, indent=2)

# --- CLI Dispatcher ---


def main():
    parser = argparse.ArgumentParser(description="SFA YMJ - Document Manager")
    parser.add_argument(
        "--allowed-paths", help="Comma-separated list of allowed paths (MCP security)"
    )
    subparsers = parser.add_subparsers(dest="command")

    # parse
    parse_parser = subparsers.add_parser("parse", help="Parse YMJ file")
    parse_parser.add_argument("path", help="File path")

    # read-markdown
    read_md_parser = subparsers.add_parser(
        "read-markdown", help="Extract markdown content (no YAML/JSON)"
    )
    read_md_parser.add_argument("path", help="File or directory path")

    # read-yaml
    read_yaml_parser = subparsers.add_parser("read-yaml", help="Extract YAML header")
    read_yaml_parser.add_argument("path", help="File or directory path")

    # read-json
    read_json_parser = subparsers.add_parser("read-json", help="Extract JSON footer")
    read_json_parser.add_argument("path", help="File or directory path")

    # lint
    lint_parser = subparsers.add_parser("lint", help="Validate YMJ file")
    lint_parser.add_argument("path", help="File path")

    # enrich
    enrich_parser = subparsers.add_parser("enrich", help="Update hash and embeddings")
    enrich_parser.add_argument("path", help="File path")
    enrich_parser.add_argument(
        "--no-embed", action="store_true", help="Skip vector embeddings"
    )

    # enrich-all
    enrich_all_parser = subparsers.add_parser(
        "enrich-all", help="Bulk enrich all YMJ files in directory"
    )
    enrich_all_parser.add_argument("path", help="Directory path")
    enrich_all_parser.add_argument(
        "--no-embed", action="store_true", help="Skip vector embeddings"
    )
    enrich_all_parser.add_argument(
        "--force", action="store_true", help="Regenerate embeddings even if they exist"
    )

    # migrate
    migrate_parser = subparsers.add_parser("migrate", help="Convert MD to YMJ")
    migrate_parser.add_argument("path", help="File or directory path")

    # upgrade
    upgrade_parser = subparsers.add_parser("upgrade", help="Upgrade YMJ to latest spec")
    upgrade_parser.add_argument("path", help="File or directory path")

    args = parser.parse_args()

    if args.allowed_paths:
        global ALLOWED_PATHS
        for p in args.allowed_paths.split(","):
            if p.strip():
                ALLOWED_PATHS.append(Path(p.strip()).resolve())

    if args.command == "parse":
        doc = YMJDocument(args.path)
        if doc.valid_structure:
            print(
                json.dumps(
                    {
                        "header": doc.header,
                        "body_preview": doc.body[:100] + "...",
                        "footer": doc.footer,
                    },
                    indent=2,
                    default=str,
                )
            )
        else:
            print(json.dumps({"error": doc.errors}, indent=2))

    elif args.command == "read-markdown":
        try:
            # Ensure UTF-8 output on Windows
            import io

            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", errors="replace"
            )
            markdown = _read_markdown(args.path)
            print(markdown)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "read-yaml":
        try:
            # Ensure UTF-8 output on Windows
            import io

            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", errors="replace"
            )
            yaml_content = _read_yaml(args.path)
            print(yaml_content)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "read-json":
        try:
            # Ensure UTF-8 output on Windows
            import io

            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", errors="replace"
            )
            json_content = _read_json(args.path)
            print(json_content)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "lint":
        doc = YMJDocument(args.path)
        calc_hash = doc.calculate_hash() if doc.valid_structure else None
        stored_hash = doc.footer.get("payload_hash") if doc.valid_structure else None

        print(f"Structure Valid: {doc.valid_structure}")
        if doc.errors:
            print("Errors:")
            for e in doc.errors:
                print(f"  - {e}")

        if doc.valid_structure:
            if calc_hash == stored_hash:
                print("Hash: OK")
            else:
                print(
                    f"Hash: STALE (Calculated: {calc_hash[:8]}..., Stored: {stored_hash[:8]}...)"
                )

    elif args.command == "enrich":
        doc = YMJDocument(args.path)
        res = _enrich_document(doc, use_embeddings=not args.no_embed)
        print(json.dumps(res, indent=2))

    elif args.command == "enrich-all":
        p = Path(args.path)
        if not p.is_dir():
            print(json.dumps({"error": "Path must be a directory"}), file=sys.stderr)
            sys.exit(1)

        ymj_files = list(p.rglob("*.ymj"))
        total = len(ymj_files)

        if total == 0:
            print(json.dumps({"status": "no_files_found"}))
            sys.exit(0)

        print(f"Found {total} YMJ files. Starting bulk enrichment...", file=sys.stderr)

        # Pre-initialize embedding model if needed (for GPU warmup)
        if not args.no_embed:
            print("Initializing GPU-accelerated embedding model...", file=sys.stderr)
            try:
                # CRITICAL: Import torch first to preload CUDA DLLs
                import torch
                from fastembed import TextEmbedding

                _temp_model = TextEmbedding(
                    model_name="nomic-ai/nomic-embed-text-v1.5",
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                )
                print("✓ Embedding model ready", file=sys.stderr)
            except Exception as e:
                print(
                    f"Warning: GPU initialization failed, will use CPU: {e}",
                    file=sys.stderr,
                )

        results = {"total": total, "updated": 0, "skipped": 0, "errors": 0}

        for idx, file_path in enumerate(ymj_files, 1):
            try:
                doc = YMJDocument(str(file_path))

                # Check if we should skip (has embedding and not --force)
                has_embedding = (
                    doc.valid_structure
                    and "index" in doc.footer
                    and "embedding" in doc.footer.get("index", {})
                )

                if has_embedding and not args.force and not args.no_embed:
                    results["skipped"] += 1
                    print(
                        f"[{idx}/{total}] SKIP {file_path.name} (has embedding, use --force)",
                        file=sys.stderr,
                    )
                    continue

                res = _enrich_document(
                    doc, use_embeddings=not args.no_embed, force_embedding=args.force
                )

                if "error" in res:
                    results["errors"] += 1
                    error_msg = res.get("error", "unknown")
                    print(
                        f"[{idx}/{total}] ✗ {file_path.name}: {error_msg}",
                        file=sys.stderr,
                    )
                elif res.get("status") == "updated":
                    results["updated"] += 1
                    updates = ", ".join(res.get("updates", []))
                    print(
                        f"[{idx}/{total}] ✓ {file_path.name} ({updates})",
                        file=sys.stderr,
                    )
                else:
                    results["skipped"] += 1
                    print(
                        f"[{idx}/{total}] - {file_path.name} (no changes)",
                        file=sys.stderr,
                    )

            except Exception as e:
                results["errors"] += 1
                print(f"[{idx}/{total}] ✗ {file_path.name}: {e}", file=sys.stderr)

        print(json.dumps(results, indent=2))

    elif args.command == "migrate":
        p = Path(args.path)
        if p.is_file():
            print(_migrate_md(str(p)))
        elif p.is_dir():
            for f in p.rglob("*.md"):
                print(_migrate_md(str(f)))

    elif args.command == "upgrade":
        p = Path(args.path)
        if p.is_file():
            doc = YMJDocument(str(p))
            print(f"{p.name}: {json.dumps(_upgrade_document(doc))}")
        elif p.is_dir():
            for f in p.rglob("*.ymj"):
                doc = YMJDocument(str(f))
                res = _upgrade_document(doc)
                print(f"{f.name}: {res.get('status')} {res.get('updates', [])}")

    else:
        if mcp:
            mcp.run()
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
