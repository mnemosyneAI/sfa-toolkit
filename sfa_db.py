# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "fastmcp",
#     "sqlite-utils",
# ]
# ///

import argparse
import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional, Union

try:
    from fastmcp import FastMCP
    mcp = FastMCP("sfa-db")
except ImportError:
    mcp = None

import sqlite_utils

# --- Configuration ---

DEFAULT_DB_NAME = "sfa_data.db"

def _get_db_path() -> str:
    return os.environ.get("SFA_DB_PATH", DEFAULT_DB_NAME)

def _get_db(memory: bool = False) -> sqlite_utils.Database:
    """Get database connection."""
    if memory:
        return sqlite_utils.Database(memory=True)
    return sqlite_utils.Database(_get_db_path())

# --- Core Logic: Key-Value Store ---

def _kv_set(key: str, value: Any, namespace: str = "default", memory: bool = False):
    """Set a value in the KV store."""
    db = _get_db(memory)
    table = db["kv_store"]
    
    # Ensure table exists with correct schema
    if not table.exists():
        table.create(
            {
                "key": str,
                "value": str, # JSON string
                "namespace": str,
                "updated_at": str
            },
            pk="key"
        )
        table.create_index(["namespace"])

    # Upsert
    table.upsert({
        "key": key,
        "value": json.dumps(value),
        "namespace": namespace,
        "updated_at": datetime.now().isoformat()
    }, pk="key")

def _kv_get(key: str, namespace: str = "default", memory: bool = False) -> Optional[Any]:
    """Get a value from the KV store."""
    db = _get_db(memory)
    table = db["kv_store"]
    
    if not table.exists():
        return None
        
    try:
        row = table.get(key)
        # Optional: Check namespace if we want strict scoping, 
        # but PK is key, so it's unique anyway. 
        # We'll just return the value.
        return json.loads(row["value"])
    except sqlite_utils.db.NotFoundError:
        return None

def _kv_delete(key: str, memory: bool = False):
    """Delete a value from the KV store."""
    db = _get_db(memory)
    table = db["kv_store"]
    if table.exists():
        table.delete(key)

# --- Core Logic: Document Store ---

def _doc_insert(table_name: str, data: List[Dict[str, Any]], memory: bool = False):
    """Insert a list of dictionaries into a table."""
    db = _get_db(memory)
    table = db[table_name]
    
    # sqlite-utils automatically creates/alters tables to fit data
    table.insert_all(data, alter=True)
    
    # Attempt to enable FTS if text columns exist and FTS not yet enabled
    # This is a heuristic; might want to make it explicit later.
    # For now, we just insert.

def _doc_query(sql: str, memory: bool = False) -> List[Dict[str, Any]]:
    """Run a raw SQL query."""
    db = _get_db(memory)
    return list(db.query(sql))

def _doc_search(table_name: str, query: str, memory: bool = False) -> List[Dict[str, Any]]:
    """Run a Full Text Search on a table."""
    db = _get_db(memory)
    table = db[table_name]
    
    if not table.exists():
        return []
        
    # Check if FTS is enabled, if not, try to enable it on all text columns
    if not table.detect_fts():
        # Get text columns
        text_cols = [
            c.name for c in table.columns 
            if c.type == "TEXT" and c.name not in ["id", "pk"]
        ]
        if text_cols:
            table.enable_fts(text_cols)
            
    try:
        return list(table.search(query))
    except Exception:
        # Fallback or error if FTS fails (e.g. no text columns)
        return []

# --- MCP Tools ---

if mcp:
    @mcp.tool()
    def db_query(sql: str) -> str:
        """Run a SQL query against the persistent database."""
        try:
            results = _doc_query(sql)
            return json.dumps(results, indent=2, default=str)
        except Exception as e:
            return f"Error: {str(e)}"

    @mcp.tool()
    def db_insert(table: str, data: str) -> str:
        """Insert JSON data (list of dicts) into a table."""
        try:
            parsed_data = json.loads(data)
            if isinstance(parsed_data, dict):
                parsed_data = [parsed_data]
            _doc_insert(table, parsed_data)
            return f"Inserted {len(parsed_data)} records into '{table}'."
        except Exception as e:
            return f"Error: {str(e)}"

    @mcp.tool()
    def kv_set(key: str, value: str) -> str:
        """Set a value in the Key-Value store."""
        try:
            # Try to parse as JSON, otherwise store as string
            try:
                val = json.loads(value)
            except json.JSONDecodeError:
                val = value
            _kv_set(key, val)
            return f"Set '{key}'."
        except Exception as e:
            return f"Error: {str(e)}"

    @mcp.tool()
    def kv_get(key: str) -> str:
        """Get a value from the Key-Value store."""
        val = _kv_get(key)
        return json.dumps(val, indent=2, default=str) if val is not None else "null"

# --- CLI Dispatcher ---

def main():
    parser = argparse.ArgumentParser(description="SFA DB - Serverless Agent Memory")
    parser.add_argument("--memory", action="store_true", help="Use in-memory database (ephemeral)")
    subparsers = parser.add_subparsers(dest="command")

    # kv-set
    kv_set_parser = subparsers.add_parser("set", help="Set Key-Value pair")
    kv_set_parser.add_argument("key", help="Key")
    kv_set_parser.add_argument("value", help="Value (JSON string or raw string)")
    kv_set_parser.add_argument("--namespace", "-n", default="default", help="Namespace")

    # kv-get
    kv_get_parser = subparsers.add_parser("get", help="Get Key-Value pair")
    kv_get_parser.add_argument("key", help="Key")

    # insert
    insert_parser = subparsers.add_parser("insert", help="Insert documents into a table")
    insert_parser.add_argument("table", help="Table name")
    insert_parser.add_argument("data", help="JSON data (list of objects)")

    # query
    query_parser = subparsers.add_parser("query", help="Run SQL query")
    query_parser.add_argument("sql", help="SQL Query")

    # search
    search_parser = subparsers.add_parser("search", help="Full Text Search")
    search_parser.add_argument("table", help="Table name")
    search_parser.add_argument("query", help="Search query")

    args = parser.parse_args()

    try:
        if args.command == "set":
            try:
                val = json.loads(args.value)
            except json.JSONDecodeError:
                val = args.value
            _kv_set(args.key, val, args.namespace, args.memory)
            print(f"OK: {args.key} set.")

        elif args.command == "get":
            val = _kv_get(args.key, memory=args.memory)
            print(json.dumps(val, indent=2, default=str))

        elif args.command == "insert":
            data = json.loads(args.data)
            if isinstance(data, dict):
                data = [data]
            _doc_insert(args.table, data, args.memory)
            print(f"OK: Inserted {len(data)} records into {args.table}.")

        elif args.command == "query":
            res = _doc_query(args.sql, args.memory)
            print(json.dumps(res, indent=2, default=str))

        elif args.command == "search":
            res = _doc_search(args.table, args.query, args.memory)
            print(json.dumps(res, indent=2, default=str))

        else:
            if mcp:
                mcp.run()
            else:
                parser.print_help()

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
