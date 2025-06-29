"""
Daemon management CLI commands.

AIDEV-NOTE: Provides commands to start, stop, and check the status of the SteadyText daemon server. It also allows preloading specific model sizes.
"""

import os
import sys
import subprocess
import time
import signal
import click
import json
from pathlib import Path
from typing import Optional

from ...daemon.client import DaemonClient
from ...daemon.protocol import DEFAULT_DAEMON_HOST, DEFAULT_DAEMON_PORT
from ...utils import get_cache_dir


def get_pid_file() -> Path:
    """Get the daemon PID file path."""
    cache_dir = get_cache_dir()
    return cache_dir / "daemon.pid"


def is_daemon_running(pid_file: Path) -> bool:
    """Check if daemon process is running."""
    if not pid_file.exists():
        return False

    try:
        with open(pid_file, "r") as f:
            pid = int(f.read().strip())

        # Check if process exists
        os.kill(pid, 0)
        return True
    except (ValueError, OSError):
        # PID file is invalid or process doesn't exist
        pid_file.unlink(missing_ok=True)
        return False


@click.group()
def daemon():
    """Manage the SteadyText daemon server."""
    pass


@daemon.command()
@click.option("--host", default=DEFAULT_DAEMON_HOST, help="Host to bind to")
@click.option("--port", type=int, default=DEFAULT_DAEMON_PORT, help="Port to bind to")
@click.option(
    "--foreground", "-f", is_flag=True, help="Run in foreground (don't daemonize)"
)
@click.option("--no-preload", is_flag=True, help="Don't preload models on startup")
@click.option(
    "--force", is_flag=True, help="Force start even if daemon appears to be running"
)
@click.option(
    "--size",
    type=click.Choice(["small", "large"]),
    help="Model size to preload (small=2B, large=4B)",
)
def start(
    host: str,
    port: int,
    foreground: bool,
    no_preload: bool,
    force: bool,
    size: Optional[str],
):
    """Start the SteadyText daemon server."""
    pid_file = get_pid_file()

    # Check if already running
    if not force and is_daemon_running(pid_file):
        click.echo("Daemon is already running. Use --force to override.", err=True)
        sys.exit(1)

    # AIDEV-NOTE: Set environment variables for daemon configuration
    env = os.environ.copy()
    env["STEADYTEXT_DAEMON_HOST"] = host
    env["STEADYTEXT_DAEMON_PORT"] = str(port)
    if size:
        env["STEADYTEXT_DAEMON_SIZE"] = size

    if foreground:
        # Run in foreground
        click.echo(f"Starting SteadyText daemon in foreground on {host}:{port}...")
        from ...daemon.server import DaemonServer

        server = DaemonServer(
            host=host, port=port, preload_models=not no_preload, size=size
        )
        try:
            server.run()
        except KeyboardInterrupt:
            click.echo("\nShutting down daemon...")
    else:
        # Daemonize the process
        click.echo(f"Starting SteadyText daemon on {host}:{port}...")

        # Build command
        cmd = [
            sys.executable,
            "-m",
            "steadytext.daemon.server",
            "--host",
            host,
            "--port",
            str(port),
        ]
        if no_preload:
            cmd.append("--no-preload")
        if size:
            cmd.extend(["--size", size])

        # AIDEV-NOTE: Start the daemon as a background process. On Unix, this could use fork(), but subprocess is cross-platform.
        try:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                start_new_session=True,  # Detach from parent process group
            )

            # Write PID file
            pid_file.parent.mkdir(parents=True, exist_ok=True)
            with open(pid_file, "w") as f:
                f.write(str(process.pid))

            # Wait a moment and check if daemon started successfully
            time.sleep(2)
            client = DaemonClient(host=host, port=port, timeout_ms=5000)
            if client.connect():
                client.disconnect()
                click.echo(f"Daemon started successfully (PID: {process.pid})")
            else:
                click.echo(
                    "Warning: Daemon process started but not responding", err=True
                )

        except Exception as e:
            click.echo(f"Failed to start daemon: {e}", err=True)
            sys.exit(1)


@daemon.command()
@click.option("--force", is_flag=True, help="Force kill the daemon process")
def stop(force: bool):
    """Stop the SteadyText daemon server."""
    pid_file = get_pid_file()

    if not is_daemon_running(pid_file):
        click.echo("Daemon is not running.")
        return

    try:
        with open(pid_file, "r") as f:
            pid = int(f.read().strip())

        # First try graceful shutdown via client
        if not force:
            click.echo("Requesting daemon shutdown...")
            client = DaemonClient()
            if client.connect() and client.shutdown():
                # Wait for process to exit
                for _ in range(10):
                    time.sleep(0.5)
                    try:
                        os.kill(pid, 0)
                    except OSError:
                        # Process has exited
                        pid_file.unlink(missing_ok=True)
                        click.echo("Daemon stopped successfully.")
                        return

        # Force kill if graceful shutdown failed or --force was specified
        click.echo(f"Force stopping daemon (PID: {pid})...")
        os.kill(pid, signal.SIGKILL)
        pid_file.unlink(missing_ok=True)
        click.echo("Daemon force stopped.")

    except Exception as e:
        click.echo(f"Failed to stop daemon: {e}", err=True)
        sys.exit(1)


@daemon.command()
@click.option("--json", "output_json", is_flag=True, help="Output status as JSON")
def status(output_json: bool):
    """Check the status of the SteadyText daemon."""
    pid_file = get_pid_file()

    # Check PID file
    if is_daemon_running(pid_file):
        with open(pid_file, "r") as f:
            pid = int(f.read().strip())
    else:
        if output_json:
            click.echo(json.dumps({"running": False}))
        else:
            click.echo("Daemon is not running.")
        return

    # Try to connect and get more info
    client = DaemonClient()
    connected = client.connect()

    status_info = {
        "running": True,
        "pid": pid,
        "responsive": connected,
        "host": DEFAULT_DAEMON_HOST,
        "port": DEFAULT_DAEMON_PORT,
    }

    if connected:
        client.disconnect()

    if output_json:
        click.echo(json.dumps(status_info))
    else:
        click.echo(f"Daemon is running (PID: {pid})")
        click.echo(f"Host: {status_info['host']}")
        click.echo(f"Port: {status_info['port']}")
        click.echo(f"Responsive: {'Yes' if status_info['responsive'] else 'No'}")


@daemon.command()
@click.option("--host", default=DEFAULT_DAEMON_HOST, help="Host to bind to")
@click.option("--port", type=int, default=DEFAULT_DAEMON_PORT, help="Port to bind to")
@click.option("--no-preload", is_flag=True, help="Don't preload models on startup")
@click.option(
    "--size",
    type=click.Choice(["small", "large"]),
    help="Model size to preload (small=2B, large=4B)",
)
def restart(host: str, port: int, no_preload: bool, size: Optional[str]):
    """Restart the SteadyText daemon server."""
    # Stop existing daemon if running
    pid_file = get_pid_file()
    if is_daemon_running(pid_file):
        click.echo("Stopping existing daemon...")
        ctx = click.get_current_context()
        ctx.invoke(stop)
        time.sleep(1)

    # Start new daemon
    click.echo("Starting new daemon...")
    ctx = click.get_current_context()
    ctx.invoke(
        start,
        host=host,
        port=port,
        foreground=False,
        no_preload=no_preload,
        force=False,
        size=size,
    )
