"""Execute code in a persistent Jupyter kernel, so state survives across calls.

Each CLI invocation attaches to a long-running background kernel via its
connection file, sends a single block of code, prints outputs, and detaches.
Variables, imports, and open resources persist between invocations — the
experience mirrors running cells one at a time in a notebook.

Usage
-----
Pipe a cell on stdin (auto-starts the kernel on first use):

    python kexec.py <<'EOF'
    import numpy as np
    x = np.arange(5)
    print(x.sum())
    EOF

Explicit lifecycle:

    python kexec.py --start     # idempotent; prints the connection file path
    python kexec.py --status    # exit 0 if running, 1 otherwise
    python kexec.py --stop      # graceful shutdown

Connection-file location (in order of precedence):
    1. --connection-file <path>
    2. $CLAUDE_KERNEL_CONNECTION_FILE
    3. <tempdir>/claude_kernel/kernel.json

Exit codes
----------
    0 — executed successfully (or lifecycle command succeeded)
    1 — code raised an exception in the kernel (traceback printed to stderr),
        or --status found no running kernel
    2 — infrastructure failure (kernel failed to start, timeout, etc.)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from jupyter_client import BlockingKernelClient

DEFAULT_CONNECTION_FILE = Path(tempfile.gettempdir()) / "claude_kernel" / "kernel.json"
ENV_VAR = "CLAUDE_KERNEL_CONNECTION_FILE"

STARTUP_TIMEOUT = 20  # seconds to wait for a freshly spawned kernel to become ready
EXEC_TIMEOUT = 120  # seconds to wait for a single iopub message during execute
READY_PROBE_TIMEOUT = 2  # short probe used when checking if a kernel is alive


def resolve_connection_file(cli_arg: str | None) -> Path:
    """Pick the connection-file path from CLI arg, env var, or default."""
    if cli_arg:
        return Path(cli_arg).expanduser().resolve()
    env = os.environ.get(ENV_VAR)
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_CONNECTION_FILE


def kernel_alive(conn: Path) -> bool:
    """Return True iff a kernel is listening on the given connection file."""
    if not conn.exists():
        return False
    kc = BlockingKernelClient()
    try:
        kc.load_connection_file(str(conn))
        kc.start_channels()
        try:
            kc.wait_for_ready(timeout=READY_PROBE_TIMEOUT)
            return True
        finally:
            kc.stop_channels()
    except Exception:
        return False


def _spawn_detached_kernel(conn: Path) -> None:
    """Launch `jupyter kernel` as a detached background process.

    The kernel outlives this script: on POSIX we start a new session; on
    Windows we use DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP. Stdio is
    redirected to DEVNULL so the child doesn't keep our terminal handles.
    """
    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "kernel",
        f"--KernelManager.connection_file={conn}",
    ]
    popen_kwargs: dict = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
    }
    if sys.platform == "win32":
        DETACHED_PROCESS = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        popen_kwargs["creationflags"] = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True

    subprocess.Popen(cmd, **popen_kwargs)


def start_kernel(conn: Path) -> None:
    """Ensure a kernel is running at `conn`. Idempotent."""
    if kernel_alive(conn):
        return

    conn.parent.mkdir(parents=True, exist_ok=True)
    # Remove any stale connection file so we can detect when the fresh one appears.
    if conn.exists():
        conn.unlink()

    _spawn_detached_kernel(conn)

    # Phase 1: wait for the connection file to appear on disk.
    deadline = time.time() + STARTUP_TIMEOUT
    while time.time() < deadline:
        if conn.exists():
            break
        time.sleep(0.2)
    else:
        raise RuntimeError(
            f"Kernel connection file not written to {conn} within {STARTUP_TIMEOUT}s. "
            "Is `jupyter_client` installed in the active Python environment?"
        )

    # Phase 2: wait for the kernel itself to answer.
    kc = BlockingKernelClient()
    kc.load_connection_file(str(conn))
    kc.start_channels()
    try:
        remaining = max(1.0, deadline - time.time())
        kc.wait_for_ready(timeout=remaining)
    finally:
        kc.stop_channels()


def stop_kernel(conn: Path) -> None:
    """Send a shutdown request to the kernel and clean up the connection file."""
    if not kernel_alive(conn):
        if conn.exists():
            conn.unlink(missing_ok=True)
        return

    kc = BlockingKernelClient()
    kc.load_connection_file(str(conn))
    kc.start_channels()
    try:
        kc.shutdown(restart=False)
    finally:
        kc.stop_channels()

    # The kernel deletes its own connection file on shutdown on most platforms,
    # but don't rely on it — clean up defensively.
    for _ in range(10):
        if not conn.exists():
            break
        try:
            conn.unlink()
            break
        except (OSError, PermissionError):
            time.sleep(0.1)


def exec_code(conn: Path, code: str) -> int:
    """Execute `code` in the kernel at `conn`. Returns 0 on success, 1 on traceback."""
    kc = BlockingKernelClient()
    kc.load_connection_file(str(conn))
    kc.start_channels()

    exit_code = 0
    try:
        kc.wait_for_ready(timeout=10)
        msg_id = kc.execute(code)

        # Drain iopub messages belonging to our execute request until the
        # kernel reports `idle` for it. Messages for other requests (e.g. from
        # concurrent clients) are ignored via the parent_header check.
        while True:
            try:
                msg = kc.get_iopub_msg(timeout=EXEC_TIMEOUT)
            except Exception as e:
                print(
                    f"[kexec] timeout after {EXEC_TIMEOUT}s waiting for kernel output: {e}",
                    file=sys.stderr,
                )
                return 2

            if msg["parent_header"].get("msg_id") != msg_id:
                continue

            msg_type = msg["header"]["msg_type"]
            content = msg["content"]

            if msg_type == "stream":
                # "stdout" / "stderr" streams from print(), sys.stderr.write(), etc.
                stream = sys.stderr if content.get("name") == "stderr" else sys.stdout
                stream.write(content["text"])
            elif msg_type in ("execute_result", "display_data"):
                # Value of the final expression, or IPython display() call.
                # We only render the text/plain representation — images, HTML,
                # and DataFrames come back as their repr string.
                text = content.get("data", {}).get("text/plain", "")
                if text:
                    sys.stdout.write(text + "\n")
            elif msg_type == "error":
                sys.stderr.write("\n".join(content["traceback"]) + "\n")
                exit_code = 1
            elif msg_type == "status" and content["execution_state"] == "idle":
                break
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        kc.stop_channels()

    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execute code in a persistent Jupyter kernel.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-c",
        "--connection-file",
        help=f"Path to the kernel connection JSON (default: ${ENV_VAR} or {DEFAULT_CONNECTION_FILE})",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--start", action="store_true", help="Start a kernel (idempotent)")
    mode.add_argument("--stop", action="store_true", help="Shut down the kernel")
    mode.add_argument("--status", action="store_true", help="Report whether a kernel is running")
    args = parser.parse_args()

    conn = resolve_connection_file(args.connection_file)

    if args.status:
        if kernel_alive(conn):
            print(f"running (connection: {conn})")
            return 0
        print(f"not running (expected connection: {conn})")
        return 1

    if args.stop:
        stop_kernel(conn)
        return 0

    if args.start:
        try:
            start_kernel(conn)
        except Exception as e:
            print(f"[kexec] failed to start kernel: {e}", file=sys.stderr)
            return 2
        print(conn)
        return 0

    # Default mode: read code from stdin, auto-starting the kernel if needed.
    code = sys.stdin.read()
    if not code.strip():
        print("[kexec] no code on stdin; nothing to execute", file=sys.stderr)
        return 2

    try:
        start_kernel(conn)
    except Exception as e:
        print(f"[kexec] failed to start kernel: {e}", file=sys.stderr)
        return 2

    return exec_code(conn, code)


if __name__ == "__main__":
    sys.exit(main())
