#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = []
# ///
"""
sfa_bus.py - In-memory pub/sub message bus for agent coordination.

No external dependencies. Pure Python stdlib.

Usage:
    sfa_bus.py serve [--socket PATH] [--background]
    sfa_bus.py pub --channel CHANNEL MESSAGE
    sfa_bus.py sub --channel CHANNEL [--pattern]
    sfa_bus.py status
    sfa_bus.py stop

Examples:
    # Start bus server
    sfa_bus.py serve --background
    
    # L2 publishes completion
    sfa_bus.py pub --channel l2:events '{"type": "complete", "task_id": "1838"}'
    
    # L1 subscribes (blocks, prints messages)
    sfa_bus.py sub --channel l2:events
    
    # Subscribe with pattern matching
    sfa_bus.py sub --channel "l2:*" --pattern
"""

import argparse
import json
import os
import select
import signal
import socket
import sys
import threading
from typing import Dict, Set

DEFAULT_SOCKET = "/tmp/sfa-bus.sock"
PID_FILE = "/tmp/sfa-bus.pid"


class BusServer:
    """Simple pub/sub server using Unix domain socket."""
    
    def __init__(self, socket_path: str):
        self.socket_path = socket_path
        self.server_socket = None
        self.clients: Dict[socket.socket, dict] = {}
        self.channels: Dict[str, Set[socket.socket]] = {}
        self.running = False
        self.lock = threading.Lock()
    
    def start(self):
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        
        self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(self.socket_path)
        self.server_socket.listen(50)
        self.server_socket.setblocking(False)
        self.running = True
        
        with open(PID_FILE, 'w') as f:
            f.write(str(os.getpid()))
        
        print(f"Bus server listening on {self.socket_path}", file=sys.stderr)
        
        try:
            self._run_loop()
        finally:
            self._cleanup()
    
    def _run_loop(self):
        while self.running:
            readable = [self.server_socket] + list(self.clients.keys())
            try:
                ready, _, _ = select.select(readable, [], [], 1.0)
            except (ValueError, OSError):
                continue
            
            for sock in ready:
                if sock is self.server_socket:
                    self._accept_client()
                else:
                    self._handle_client(sock)
    
    def _accept_client(self):
        try:
            client, _ = self.server_socket.accept()
            client.setblocking(False)
            with self.lock:
                self.clients[client] = {"subscriptions": set(), "buffer": ""}
        except Exception:
            pass
    
    def _handle_client(self, sock: socket.socket):
        try:
            data = sock.recv(4096).decode('utf-8')
            if not data:
                self._remove_client(sock)
                return
            
            with self.lock:
                self.clients[sock]["buffer"] += data
            
            while '\n' in self.clients[sock]["buffer"]:
                line, self.clients[sock]["buffer"] = self.clients[sock]["buffer"].split('\n', 1)
                if line.strip():
                    self._process_message(sock, line.strip())
                    
        except (ConnectionResetError, BrokenPipeError, OSError):
            self._remove_client(sock)
    
    def _process_message(self, sock: socket.socket, line: str):
        try:
            msg = json.loads(line)
            cmd = msg.get("cmd")
            
            if cmd == "pub":
                self._handle_pub(msg)
                self._send(sock, {"type": "ok"})
                
            elif cmd == "sub":
                channel = msg.get("channel", "")
                pattern = msg.get("pattern", False)
                with self.lock:
                    self.clients[sock]["subscriptions"].add((channel, pattern))
                    if channel not in self.channels:
                        self.channels[channel] = set()
                    self.channels[channel].add(sock)
                self._send(sock, {"type": "ok", "subscribed": channel})
                
            elif cmd == "unsub":
                channel = msg.get("channel", "")
                with self.lock:
                    self.clients[sock]["subscriptions"].discard((channel, False))
                    self.clients[sock]["subscriptions"].discard((channel, True))
                    if channel in self.channels:
                        self.channels[channel].discard(sock)
                self._send(sock, {"type": "ok", "unsubscribed": channel})
                
            elif cmd == "ping":
                self._send(sock, {"type": "pong"})
                
            elif cmd == "stats":
                with self.lock:
                    stats = {
                        "type": "stats",
                        "clients": len(self.clients),
                        "channels": list(self.channels.keys()),
                        "subscriptions": sum(len(c["subscriptions"]) for c in self.clients.values())
                    }
                self._send(sock, stats)
                
        except json.JSONDecodeError:
            self._send(sock, {"type": "error", "message": "Invalid JSON"})
        except Exception as e:
            self._send(sock, {"type": "error", "message": str(e)})
    
    def _handle_pub(self, msg: dict):
        channel = msg.get("channel", "")
        payload = msg.get("payload", {})
        
        to_notify = set()
        with self.lock:
            for client_sock, client_info in list(self.clients.items()):
                for sub_channel, is_pattern in client_info["subscriptions"]:
                    if is_pattern:
                        if sub_channel.endswith('*'):
                            prefix = sub_channel[:-1]
                            if channel.startswith(prefix):
                                to_notify.add(client_sock)
                        elif sub_channel == channel:
                            to_notify.add(client_sock)
                    else:
                        if sub_channel == channel:
                            to_notify.add(client_sock)
        
        out_msg = {"type": "msg", "channel": channel, "payload": payload}
        for sock in to_notify:
            self._send(sock, out_msg)
    
    def _send(self, sock: socket.socket, msg: dict):
        try:
            sock.sendall((json.dumps(msg) + '\n').encode('utf-8'))
        except (BrokenPipeError, ConnectionResetError, OSError):
            self._remove_client(sock)
    
    def _remove_client(self, sock: socket.socket):
        with self.lock:
            if sock in self.clients:
                for channel in self.channels.values():
                    channel.discard(sock)
                del self.clients[sock]
        try:
            sock.close()
        except Exception:
            pass
    
    def _cleanup(self):
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        if os.path.exists(PID_FILE):
            os.unlink(PID_FILE)
        for sock in list(self.clients.keys()):
            try:
                sock.close()
            except Exception:
                pass


class BusClient:
    """Client for connecting to the bus."""
    
    def __init__(self, socket_path: str = DEFAULT_SOCKET):
        self.socket_path = socket_path
        self.sock = None
        self.buffer = ""
    
    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(self.socket_path)
    
    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None
    
    def _send(self, msg: dict):
        self.sock.sendall((json.dumps(msg) + '\n').encode('utf-8'))
    
    def _recv(self, timeout: float = None) -> dict:
        if timeout is not None:
            self.sock.settimeout(timeout)
        else:
            self.sock.setblocking(True)
            
        while '\n' not in self.buffer:
            data = self.sock.recv(4096).decode('utf-8')
            if not data:
                raise ConnectionError("Server closed connection")
            self.buffer += data
        
        line, self.buffer = self.buffer.split('\n', 1)
        return json.loads(line)
    
    def publish(self, channel: str, payload: dict):
        self._send({"cmd": "pub", "channel": channel, "payload": payload})
        resp = self._recv(timeout=5.0)
        return resp.get("type") == "ok"
    
    def subscribe(self, channel: str, pattern: bool = False, callback=None):
        self._send({"cmd": "sub", "channel": channel, "pattern": pattern})
        resp = self._recv(timeout=5.0)
        if resp.get("type") != "ok":
            raise RuntimeError(f"Subscribe failed: {resp}")
        
        if callback:
            while True:
                msg = self._recv()
                if msg.get("type") == "msg":
                    callback(msg)
    
    def ping(self) -> bool:
        try:
            self._send({"cmd": "ping"})
            resp = self._recv(timeout=2.0)
            return resp.get("type") == "pong"
        except Exception:
            return False
    
    def stats(self) -> dict:
        self._send({"cmd": "stats"})
        return self._recv(timeout=5.0)


def cmd_serve(args):
    if args.background:
        pid = os.fork()
        if pid > 0:
            print(f"Bus server started (pid {pid})")
            return
        os.setsid()
        sys.stdout = open('/dev/null', 'w')
        sys.stderr = open('/dev/null', 'w')
    
    server = BusServer(args.socket)
    
    def handle_signal(signum, frame):
        server.running = False
    
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    server.start()


def cmd_pub(args):
    try:
        payload = json.loads(args.message)
    except json.JSONDecodeError:
        payload = {"text": args.message}
    
    client = BusClient(args.socket)
    try:
        client.connect()
        if client.publish(args.channel, payload):
            print(json.dumps({"status": "published", "channel": args.channel}))
        else:
            print(json.dumps({"status": "error"}))
            sys.exit(1)
    finally:
        client.close()


def cmd_sub(args):
    client = BusClient(args.socket)
    
    def on_message(msg):
        print(json.dumps(msg))
        sys.stdout.flush()
    
    try:
        client.connect()
        client.subscribe(args.channel, pattern=args.pattern, callback=on_message)
    except KeyboardInterrupt:
        pass
    finally:
        client.close()


def cmd_status(args):
    client = BusClient(args.socket)
    try:
        client.connect()
        if client.ping():
            stats = client.stats()
            print(json.dumps(stats, indent=2))
        else:
            print(json.dumps({"status": "not responding"}))
            sys.exit(1)
    except (ConnectionRefusedError, FileNotFoundError):
        print(json.dumps({"status": "not running"}))
        sys.exit(1)
    finally:
        client.close()


def cmd_stop(args):
    if os.path.exists(PID_FILE):
        with open(PID_FILE) as f:
            pid = int(f.read().strip())
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"Stopped bus server (pid {pid})")
        except ProcessLookupError:
            print("Server not running (stale PID file)")
            os.unlink(PID_FILE)
    else:
        print("Server not running")


def main():
    parser = argparse.ArgumentParser(description="sfa_bus - In-memory pub/sub message bus")
    parser.add_argument("--socket", default=DEFAULT_SOCKET, help=f"Socket path (default: {DEFAULT_SOCKET})")
    subparsers = parser.add_subparsers(dest="command")
    
    serve_parser = subparsers.add_parser("serve", help="Start the bus server")
    serve_parser.add_argument("--background", "-b", action="store_true", help="Run in background")
    
    pub_parser = subparsers.add_parser("pub", help="Publish a message")
    pub_parser.add_argument("--channel", "-c", required=True, help="Channel to publish to")
    pub_parser.add_argument("message", help="Message (JSON or plain text)")
    
    sub_parser = subparsers.add_parser("sub", help="Subscribe to a channel")
    sub_parser.add_argument("--channel", "-c", required=True, help="Channel to subscribe to")
    sub_parser.add_argument("--pattern", "-p", action="store_true", help="Use pattern matching (e.g., 'l2:*')")
    
    subparsers.add_parser("status", help="Check server status")
    subparsers.add_parser("stop", help="Stop the server")
    
    args = parser.parse_args()
    
    if args.command == "serve":
        cmd_serve(args)
    elif args.command == "pub":
        cmd_pub(args)
    elif args.command == "sub":
        cmd_sub(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "stop":
        cmd_stop(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
