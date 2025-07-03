# Daemon Usage Guide

Learn how to use SteadyText's daemon mode for persistent model serving and 160x faster response times.

## Overview

The SteadyText daemon is a background service that keeps models loaded in memory, eliminating the 2-3 second startup overhead for each operation. It provides:

- **160x faster first response** - No model loading delay
- **Shared cache** - All clients benefit from cached results
- **Automatic fallback** - Operations work without daemon
- **Zero configuration** - Used by default when available
- **Thread-safe** - Handles concurrent requests efficiently

## Table of Contents

- [Understanding the Daemon](#understanding-the-daemon)
- [Starting and Stopping](#starting-and-stopping)
- [Configuration](#configuration)
- [Python SDK Usage](#python-sdk-usage)
- [CLI Integration](#cli-integration)
- [Production Deployment](#production-deployment)
- [Monitoring and Debugging](#monitoring-and-debugging)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

## Understanding the Daemon

### Architecture

```
┌─────────────────────────────────────────────┐
│              Client Applications            │
├─────────────────────────────────────────────┤
│  Python SDK  │  CLI Tools  │  Custom Apps   │
├─────────────────────────────────────────────┤
│           ZeroMQ Client Layer               │
│         (Automatic Fallback)                │
├─────────────────────────────────────────────┤
│              ZeroMQ REP Server              │
│            (TCP Port 5557)                  │
├─────────────────────────────────────────────┤
│           Daemon Server Process             │
│  ┌─────────────────────────────────────┐   │
│  │  Loaded Models  │  Shared Cache     │   │
│  │  - Gemma-3n     │  - Generation     │   │
│  │  - Qwen3        │  - Embeddings     │   │
│  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

### How It Works

1. **First Request**: Client checks if daemon is running
2. **Daemon Available**: Sends request via ZeroMQ
3. **Daemon Unavailable**: Falls back to direct model loading
4. **Response**: Client receives result (cached or generated)

## Starting and Stopping

### Basic Commands

```bash
# Start daemon in background (default)
st daemon start

# Start with custom settings
st daemon start --host 0.0.0.0 --port 5557 --seed 42

# Check status
st daemon status

# Stop daemon
st daemon stop

# Restart daemon
st daemon restart
```

### Foreground Mode (Debugging)

```bash
# Run in foreground to see logs
st daemon start --foreground

# Output:
# SteadyText daemon starting...
# Loading generation model...
# Loading embedding model...
# Daemon ready on tcp://127.0.0.1:5557
# [2024-01-15 10:23:45] Request: generate (seed=42)
# [2024-01-15 10:23:45] Cache hit for generation
```

### Systemd Service (Production)

```ini
# /etc/systemd/system/steadytext.service
[Unit]
Description=SteadyText Daemon
After=network.target

[Service]
Type=simple
User=steadytext
Group=steadytext
WorkingDirectory=/var/lib/steadytext
ExecStart=/usr/local/bin/st daemon start --foreground
ExecStop=/usr/local/bin/st daemon stop
Restart=always
RestartSec=10
StandardOutput=append:/var/log/steadytext/daemon.log
StandardError=append:/var/log/steadytext/daemon.error.log

# Environment
Environment="STEADYTEXT_GENERATION_CACHE_CAPACITY=1024"
Environment="STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB=200"
Environment="PYTHONUNBUFFERED=1"

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable steadytext
sudo systemctl start steadytext
sudo systemctl status steadytext
```

## Configuration

### Environment Variables

```bash
# Daemon settings
export STEADYTEXT_DAEMON_HOST=0.0.0.0      # Bind address
export STEADYTEXT_DAEMON_PORT=5557         # Port number
export STEADYTEXT_DISABLE_DAEMON=1         # Disable daemon usage

# Cache settings (shared by daemon)
export STEADYTEXT_GENERATION_CACHE_CAPACITY=1024
export STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB=200
export STEADYTEXT_EMBEDDING_CACHE_CAPACITY=2048
export STEADYTEXT_EMBEDDING_CACHE_MAX_SIZE_MB=400

# Model settings
export STEADYTEXT_DEFAULT_SEED=42
export STEADYTEXT_MODEL_DIR=/path/to/models
```

### Configuration File

```python
# steadytext_config.py
import os

# Daemon configuration
DAEMON_CONFIG = {
    "host": os.getenv("STEADYTEXT_DAEMON_HOST", "127.0.0.1"),
    "port": int(os.getenv("STEADYTEXT_DAEMON_PORT", 5557)),
    "timeout": 5000,  # milliseconds
    "max_retries": 3,
    "retry_delay": 0.1  # seconds
}

# Cache configuration
CACHE_CONFIG = {
    "generation": {
        "capacity": int(os.getenv("STEADYTEXT_GENERATION_CACHE_CAPACITY", 256)),
        "max_size_mb": float(os.getenv("STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB", 50.0))
    },
    "embedding": {
        "capacity": int(os.getenv("STEADYTEXT_EMBEDDING_CACHE_CAPACITY", 512)),
        "max_size_mb": float(os.getenv("STEADYTEXT_EMBEDDING_CACHE_MAX_SIZE_MB", 100.0))
    }
}

# Apply configuration
os.environ.update({
    "STEADYTEXT_DAEMON_HOST": DAEMON_CONFIG["host"],
    "STEADYTEXT_DAEMON_PORT": str(DAEMON_CONFIG["port"]),
    "STEADYTEXT_GENERATION_CACHE_CAPACITY": str(CACHE_CONFIG["generation"]["capacity"]),
    "STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB": str(CACHE_CONFIG["generation"]["max_size_mb"]),
    "STEADYTEXT_EMBEDDING_CACHE_CAPACITY": str(CACHE_CONFIG["embedding"]["capacity"]),
    "STEADYTEXT_EMBEDDING_CACHE_MAX_SIZE_MB": str(CACHE_CONFIG["embedding"]["max_size_mb"])
})
```

## Python SDK Usage

### Automatic Daemon Usage

```python
import steadytext

# Daemon is used automatically if available
text = steadytext.generate("Hello world", seed=42)  # Fast if daemon running
embedding = steadytext.embed("test text", seed=123)  # Uses daemon

# Check if daemon was used
from steadytext.daemon.client import is_daemon_running
if is_daemon_running():
    print("Using daemon for fast responses")
else:
    print("Daemon not available, using direct mode")
```

### Explicit Daemon Context

```python
from steadytext.daemon import use_daemon
import steadytext

# Force daemon usage (raises error if not available)
with use_daemon():
    text = steadytext.generate("Hello world", seed=42)
    embedding = steadytext.embed("test", seed=123)
    
    # All operations in this context use daemon
    for i in range(100):
        result = steadytext.generate(f"Item {i}", seed=i)
```

### Connection Management

```python
from steadytext.daemon.client import DaemonClient
import time

class ManagedDaemonClient:
    """Daemon client with connection pooling and retries."""
    
    def __init__(self, max_retries=3, timeout=5000):
        self.max_retries = max_retries
        self.timeout = timeout
        self._client = None
    
    def _get_client(self):
        """Get or create daemon client."""
        if self._client is None:
            self._client = DaemonClient(timeout=self.timeout)
        return self._client
    
    def generate_with_retry(self, prompt, **kwargs):
        """Generate with automatic retry on failure."""
        for attempt in range(self.max_retries):
            try:
                client = self._get_client()
                return client.generate(prompt, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                print(f"Retry {attempt + 1}/{self.max_retries} after error: {e}")
                time.sleep(0.1 * (attempt + 1))
                self._client = None  # Reset connection
    
    def close(self):
        """Close daemon connection."""
        if self._client:
            self._client.close()
            self._client = None

# Usage
client = ManagedDaemonClient()
try:
    text = client.generate_with_retry("Hello world", seed=42)
    print(text)
finally:
    client.close()
```

### Streaming with Daemon

```python
import steadytext
from steadytext.daemon import use_daemon

# Streaming works identically with daemon
with use_daemon():
    print("Streaming with daemon:")
    for token in steadytext.generate_iter("Tell me a story", seed=42):
        print(token, end="", flush=True)
    print()

# The daemon handles streaming efficiently:
# 1. Client sends streaming request
# 2. Daemon generates tokens
# 3. Tokens sent with acknowledgment protocol
# 4. Client controls flow with ACK messages
```

### Batch Operations

```python
import concurrent.futures
import steadytext
from steadytext.daemon import use_daemon
import time

def benchmark_daemon_performance():
    """Compare daemon vs direct performance."""
    prompts = [f"Generate text for item {i}" for i in range(20)]
    
    # Test without daemon
    start = time.time()
    results_direct = []
    for prompt in prompts:
        # Force direct mode
        import os
        os.environ["STEADYTEXT_DISABLE_DAEMON"] = "1"
        result = steadytext.generate(prompt, seed=42)
        results_direct.append(result)
        del os.environ["STEADYTEXT_DISABLE_DAEMON"]
    direct_time = time.time() - start
    
    # Test with daemon
    start = time.time()
    results_daemon = []
    with use_daemon():
        for prompt in prompts:
            result = steadytext.generate(prompt, seed=42)
            results_daemon.append(result)
    daemon_time = time.time() - start
    
    print(f"Direct mode: {direct_time:.2f}s")
    print(f"Daemon mode: {daemon_time:.2f}s")
    print(f"Speedup: {direct_time/daemon_time:.1f}x")

# Parallel batch processing
def process_batch_parallel(prompts, max_workers=4):
    """Process prompts in parallel using daemon."""
    with use_daemon():
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(steadytext.generate, prompt, seed=idx): (prompt, idx)
                for idx, prompt in enumerate(prompts)
            }
            
            # Collect results
            results = {}
            for future in concurrent.futures.as_completed(futures):
                prompt, idx = futures[future]
                try:
                    result = future.result()
                    results[idx] = result
                except Exception as e:
                    print(f"Error processing {prompt}: {e}")
                    results[idx] = None
            
            # Return in order
            return [results[i] for i in range(len(prompts))]

# Usage
prompts = ["Task 1", "Task 2", "Task 3", "Task 4"]
results = process_batch_parallel(prompts)
```

## CLI Integration

### Automatic Daemon Usage

```bash
# CLI automatically uses daemon if available
st generate "Hello world" --seed 42

# Check if daemon is being used
st daemon status && echo "Daemon active" || echo "No daemon"

# Force direct mode (bypass daemon)
STEADYTEXT_DISABLE_DAEMON=1 st generate "Hello world"
```

### Shell Script Integration

```bash
#!/bin/bash
# daemon_batch.sh - Batch processing with daemon

# Ensure daemon is running
ensure_daemon() {
    if ! st daemon status >/dev/null 2>&1; then
        echo "Starting daemon..."
        st daemon start
        sleep 2  # Wait for startup
    fi
}

# Process files with daemon
process_files() {
    local files=("$@")
    
    ensure_daemon
    
    for file in "${files[@]}"; do
        echo "Processing: $file"
        
        # Generate summary using daemon
        summary=$(cat "$file" | st generate "Summarize this text" --wait --seed 42)
        
        # Generate embedding using daemon  
        embedding=$(cat "$file" | st embed --format json --seed 42)
        
        # Save results
        echo "$summary" > "${file%.txt}_summary.txt"
        echo "$embedding" > "${file%.txt}_embedding.json"
    done
}

# Main
if [ $# -eq 0 ]; then
    echo "Usage: $0 file1.txt file2.txt ..."
    exit 1
fi

process_files "$@"

echo "Processing complete!"
```

### Performance Monitoring

```bash
#!/bin/bash
# monitor_daemon.sh - Monitor daemon performance

# Function to time operations
time_operation() {
    local operation="$1"
    local start=$(date +%s.%N)
    eval "$operation" >/dev/null 2>&1
    local end=$(date +%s.%N)
    echo "$(echo "$end - $start" | bc)"
}

# Monitor daemon performance
monitor_daemon() {
    echo "Daemon Performance Monitor"
    echo "========================="
    
    # Check daemon status
    if st daemon status --json | jq -e '.running' >/dev/null; then
        echo "✓ Daemon is running"
    else
        echo "✗ Daemon is not running"
        return 1
    fi
    
    # Test generation speed
    echo -e "\nGeneration Performance:"
    for i in {1..5}; do
        time=$(time_operation "echo 'test' | st --seed $i")
        echo "  Request $i: ${time}s"
    done
    
    # Test embedding speed
    echo -e "\nEmbedding Performance:"
    for i in {1..5}; do
        time=$(time_operation "st embed 'test text' --seed $i")
        echo "  Request $i: ${time}s"
    done
    
    # Cache statistics
    echo -e "\nCache Statistics:"
    st cache --status
}

# Run monitoring
monitor_daemon
```

## Production Deployment

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -s /bin/bash steadytext

# Install SteadyText
RUN pip install steadytext

# Create directories
RUN mkdir -p /var/log/steadytext /var/lib/steadytext && \
    chown -R steadytext:steadytext /var/log/steadytext /var/lib/steadytext

# Switch to app user
USER steadytext
WORKDIR /home/steadytext

# Download models during build
RUN st models download --all

# Expose daemon port
EXPOSE 5557

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD st daemon status || exit 1

# Start daemon
CMD ["st", "daemon", "start", "--foreground", "--host", "0.0.0.0"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  steadytext:
    build: .
    ports:
      - "5557:5557"
    volumes:
      - steadytext-cache:/home/steadytext/.cache/steadytext
      - ./logs:/var/log/steadytext
    environment:
      - STEADYTEXT_GENERATION_CACHE_CAPACITY=1024
      - STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB=200
      - STEADYTEXT_EMBEDDING_CACHE_CAPACITY=2048
      - STEADYTEXT_EMBEDDING_CACHE_MAX_SIZE_MB=400
      - STEADYTEXT_DEFAULT_SEED=42
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "st", "daemon", "status"]
      interval: 30s
      timeout: 3s
      retries: 3

volumes:
  steadytext-cache:
```

### Kubernetes Deployment

```yaml
# steadytext-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: steadytext-daemon
  labels:
    app: steadytext
spec:
  replicas: 3
  selector:
    matchLabels:
      app: steadytext
  template:
    metadata:
      labels:
        app: steadytext
    spec:
      containers:
      - name: steadytext
        image: steadytext:latest
        ports:
        - containerPort: 5557
        env:
        - name: STEADYTEXT_GENERATION_CACHE_CAPACITY
          value: "2048"
        - name: STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB
          value: "500"
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        livenessProbe:
          exec:
            command:
            - st
            - daemon
            - status
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          tcpSocket:
            port: 5557
          initialDelaySeconds: 15
          periodSeconds: 5
        volumeMounts:
        - name: cache
          mountPath: /home/steadytext/.cache/steadytext
      volumes:
      - name: cache
        persistentVolumeClaim:
          claimName: steadytext-cache-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: steadytext-service
spec:
  selector:
    app: steadytext
  ports:
    - protocol: TCP
      port: 5557
      targetPort: 5557
  type: LoadBalancer

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: steadytext-cache-pvc
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 50Gi
```

### High Availability Setup

```python
# ha_daemon_client.py - High availability daemon client
import random
import time
from typing import List, Optional
import steadytext
from steadytext.daemon.client import DaemonClient

class HADaemonClient:
    """High availability client with multiple daemon endpoints."""
    
    def __init__(self, endpoints: List[tuple]):
        """
        Initialize with multiple endpoints.
        
        Args:
            endpoints: List of (host, port) tuples
        """
        self.endpoints = endpoints
        self.clients = {}
        self.failed_endpoints = set()
        self.last_health_check = 0
        self.health_check_interval = 60  # seconds
    
    def _get_client(self, endpoint: tuple) -> Optional[DaemonClient]:
        """Get or create client for endpoint."""
        if endpoint not in self.clients:
            try:
                host, port = endpoint
                client = DaemonClient(host=host, port=port, timeout=2000)
                # Test connection
                client._send_request({"type": "ping"})
                self.clients[endpoint] = client
            except Exception:
                return None
        return self.clients.get(endpoint)
    
    def _health_check(self):
        """Periodic health check of failed endpoints."""
        if time.time() - self.last_health_check > self.health_check_interval:
            recovered = set()
            for endpoint in self.failed_endpoints:
                if self._get_client(endpoint):
                    recovered.add(endpoint)
            self.failed_endpoints -= recovered
            self.last_health_check = time.time()
    
    def _get_available_endpoint(self) -> Optional[tuple]:
        """Get random available endpoint."""
        self._health_check()
        available = [ep for ep in self.endpoints if ep not in self.failed_endpoints]
        return random.choice(available) if available else None
    
    def generate(self, prompt: str, **kwargs):
        """Generate with automatic failover."""
        attempts = 0
        endpoints_tried = set()
        
        while attempts < len(self.endpoints):
            endpoint = self._get_available_endpoint()
            if not endpoint or endpoint in endpoints_tried:
                break
            
            endpoints_tried.add(endpoint)
            client = self._get_client(endpoint)
            
            if client:
                try:
                    return client.generate(prompt, **kwargs)
                except Exception as e:
                    print(f"Failed on {endpoint}: {e}")
                    self.failed_endpoints.add(endpoint)
                    if endpoint in self.clients:
                        del self.clients[endpoint]
            
            attempts += 1
        
        # All endpoints failed, fall back to direct mode
        print("All daemon endpoints failed, using direct mode")
        return steadytext.generate(prompt, **kwargs)
    
    def embed(self, text: str, **kwargs):
        """Embed with automatic failover."""
        # Similar implementation to generate
        pass

# Usage
ha_client = HADaemonClient([
    ("daemon1.example.com", 5557),
    ("daemon2.example.com", 5557),
    ("daemon3.example.com", 5557)
])

# Automatic failover
result = ha_client.generate("Hello world", seed=42)
```

## Monitoring and Debugging

### Logging Configuration

```python
# logging_config.py
import logging
import sys
from pathlib import Path

def setup_daemon_logging(log_dir="/var/log/steadytext"):
    """Configure comprehensive daemon logging."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # File handler for all logs
    all_handler = logging.FileHandler(log_dir / "daemon.log")
    all_handler.setLevel(logging.DEBUG)
    all_handler.setFormatter(detailed_formatter)
    
    # File handler for errors only
    error_handler = logging.FileHandler(log_dir / "daemon.error.log")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(all_handler)
    root_logger.addHandler(error_handler)
    root_logger.addHandler(console_handler)
    
    # Configure specific loggers
    logging.getLogger("steadytext.daemon").setLevel(logging.DEBUG)
    logging.getLogger("zmq").setLevel(logging.WARNING)
    
    return root_logger

# Request logging middleware
class RequestLogger:
    """Log all daemon requests for debugging."""
    
    def __init__(self, daemon_server):
        self.daemon_server = daemon_server
        self.logger = logging.getLogger("steadytext.daemon.requests")
    
    def log_request(self, request_id, request_type, request_data):
        """Log incoming request."""
        self.logger.info(f"Request {request_id}: {request_type}", extra={
            "request_id": request_id,
            "request_type": request_type,
            "seed": request_data.get("seed"),
            "prompt_length": len(request_data.get("prompt", "")),
            "timestamp": time.time()
        })
    
    def log_response(self, request_id, response_data, duration):
        """Log outgoing response."""
        self.logger.info(f"Response {request_id}: {duration:.3f}s", extra={
            "request_id": request_id,
            "success": response_data.get("success"),
            "cached": response_data.get("cached", False),
            "duration": duration,
            "timestamp": time.time()
        })
```

### Performance Metrics

```python
# metrics.py - Daemon performance metrics
import time
import psutil
import json
from collections import deque
from datetime import datetime
from typing import Dict, Any

class DaemonMetrics:
    """Collect and report daemon performance metrics."""
    
    def __init__(self, window_size=1000):
        self.request_times = deque(maxlen=window_size)
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_requests = 0
        self.errors = 0
        self.start_time = time.time()
        self.process = psutil.Process()
    
    def record_request(self, duration: float, cached: bool, success: bool):
        """Record request metrics."""
        self.total_requests += 1
        self.request_times.append(duration)
        
        if cached:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        
        if not success:
            self.errors += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        uptime = time.time() - self.start_time
        
        # Calculate percentiles
        if self.request_times:
            sorted_times = sorted(self.request_times)
            p50 = sorted_times[len(sorted_times) // 2]
            p95 = sorted_times[int(len(sorted_times) * 0.95)]
            p99 = sorted_times[int(len(sorted_times) * 0.99)]
            avg_time = sum(sorted_times) / len(sorted_times)
        else:
            p50 = p95 = p99 = avg_time = 0
        
        # System metrics
        cpu_percent = self.process.cpu_percent()
        memory_info = self.process.memory_info()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": uptime,
            "total_requests": self.total_requests,
            "requests_per_second": self.total_requests / uptime if uptime > 0 else 0,
            "cache_hit_rate": self.cache_hits / self.total_requests if self.total_requests > 0 else 0,
            "error_rate": self.errors / self.total_requests if self.total_requests > 0 else 0,
            "response_times": {
                "average": avg_time,
                "p50": p50,
                "p95": p95,
                "p99": p99
            },
            "system": {
                "cpu_percent": cpu_percent,
                "memory_mb": memory_info.rss / 1024 / 1024,
                "threads": self.process.num_threads()
            }
        }
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        metrics = self.get_metrics()
        lines = []
        
        # Request metrics
        lines.append(f'steadytext_requests_total {metrics["total_requests"]}')
        lines.append(f'steadytext_requests_per_second {metrics["requests_per_second"]:.2f}')
        lines.append(f'steadytext_cache_hit_rate {metrics["cache_hit_rate"]:.4f}')
        lines.append(f'steadytext_error_rate {metrics["error_rate"]:.4f}')
        
        # Response time metrics
        lines.append(f'steadytext_response_time_seconds{{quantile="0.5"}} {metrics["response_times"]["p50"]:.4f}')
        lines.append(f'steadytext_response_time_seconds{{quantile="0.95"}} {metrics["response_times"]["p95"]:.4f}')
        lines.append(f'steadytext_response_time_seconds{{quantile="0.99"}} {metrics["response_times"]["p99"]:.4f}')
        
        # System metrics
        lines.append(f'steadytext_cpu_percent {metrics["system"]["cpu_percent"]:.2f}')
        lines.append(f'steadytext_memory_megabytes {metrics["system"]["memory_mb"]:.2f}')
        lines.append(f'steadytext_threads {metrics["system"]["threads"]}')
        
        return '\n'.join(lines)

# HTTP metrics endpoint
from flask import Flask, Response

app = Flask(__name__)
metrics = DaemonMetrics()

@app.route('/metrics')
def prometheus_metrics():
    return Response(metrics.export_prometheus(), mimetype='text/plain')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9090)
```

### Debug Tools

```bash
#!/bin/bash
# debug_daemon.sh - Comprehensive daemon debugging

# Function to trace daemon requests
trace_requests() {
    echo "Tracing daemon requests..."
    
    # Start tcpdump on daemon port
    sudo tcpdump -i lo -w daemon_trace.pcap port 5557 &
    TCPDUMP_PID=$!
    
    # Run test requests
    for i in {1..10}; do
        st generate "Test $i" --seed $i &
    done
    wait
    
    # Stop tcpdump
    sudo kill $TCPDUMP_PID
    
    echo "Trace saved to daemon_trace.pcap"
}

# Function to profile daemon
profile_daemon() {
    echo "Profiling daemon performance..."
    
    # Get daemon PID
    DAEMON_PID=$(st daemon status --json | jq -r '.pid')
    
    if [ -z "$DAEMON_PID" ]; then
        echo "Daemon not running"
        return 1
    fi
    
    # CPU profiling
    echo "CPU profiling for 30 seconds..."
    sudo perf record -F 99 -p $DAEMON_PID -g -- sleep 30
    sudo perf report > daemon_cpu_profile.txt
    
    # Memory profiling
    echo "Memory snapshot..."
    sudo gcore -o daemon_memory $DAEMON_PID
    
    # Strace
    echo "System call trace for 10 seconds..."
    sudo strace -p $DAEMON_PID -o daemon_strace.log -f -T &
    STRACE_PID=$!
    sleep 10
    sudo kill $STRACE_PID
    
    echo "Profiling complete"
}

# Function to stress test daemon
stress_test() {
    local concurrent=${1:-10}
    local requests=${2:-100}
    
    echo "Stress testing with $concurrent concurrent clients, $requests requests each"
    
    # Start monitoring
    st daemon status --json > stress_test_before.json
    
    # Run concurrent requests
    for i in $(seq 1 $concurrent); do
        (
            for j in $(seq 1 $requests); do
                st generate "Stress test $i-$j" --seed $((i*1000+j)) >/dev/null 2>&1
            done
            echo "Client $i completed"
        ) &
    done
    
    # Wait for completion
    wait
    
    # Get final status
    st daemon status --json > stress_test_after.json
    
    echo "Stress test complete"
}

# Main menu
echo "SteadyText Daemon Debug Tools"
echo "1. Trace requests"
echo "2. Profile daemon"
echo "3. Stress test"
echo "4. View logs"
echo "5. Export metrics"

read -p "Select option: " choice

case $choice in
    1) trace_requests ;;
    2) profile_daemon ;;
    3) 
        read -p "Concurrent clients (default 10): " concurrent
        read -p "Requests per client (default 100): " requests
        stress_test ${concurrent:-10} ${requests:-100}
        ;;
    4) 
        tail -f /var/log/steadytext/daemon.log
        ;;
    5)
        curl -s http://localhost:9090/metrics
        ;;
    *) echo "Invalid option" ;;
esac
```

## Performance Optimization

### Cache Warming

```python
# cache_warmer.py - Pre-populate daemon cache
import steadytext
from steadytext.daemon import use_daemon
import json
from pathlib import Path

class DaemonCacheWarmer:
    """Warm up daemon cache with common requests."""
    
    def __init__(self, warmup_file="warmup_prompts.json"):
        self.warmup_file = Path(warmup_file)
        self.load_prompts()
    
    def load_prompts(self):
        """Load warmup prompts from file."""
        if self.warmup_file.exists():
            with open(self.warmup_file) as f:
                self.warmup_data = json.load(f)
        else:
            # Default warmup prompts
            self.warmup_data = {
                "generation": [
                    {"prompt": "Hello", "seed": 42},
                    {"prompt": "Write a summary", "seed": 42},
                    {"prompt": "Explain this concept", "seed": 42},
                    {"prompt": "Generate code", "seed": 42},
                    {"prompt": "Create documentation", "seed": 42}
                ],
                "embedding": [
                    {"text": "search query", "seed": 42},
                    {"text": "document text", "seed": 42},
                    {"text": "user input", "seed": 42}
                ]
            }
    
    def warm_generation_cache(self):
        """Warm up generation cache."""
        print("Warming generation cache...")
        
        with use_daemon():
            for item in self.warmup_data["generation"]:
                try:
                    result = steadytext.generate(
                        item["prompt"],
                        seed=item.get("seed", 42),
                        max_new_tokens=item.get("max_tokens", 512)
                    )
                    print(f"✓ Cached: {item['prompt'][:30]}...")
                except Exception as e:
                    print(f"✗ Failed: {item['prompt'][:30]}... - {e}")
    
    def warm_embedding_cache(self):
        """Warm up embedding cache."""
        print("\nWarming embedding cache...")
        
        with use_daemon():
            for item in self.warmup_data["embedding"]:
                try:
                    result = steadytext.embed(
                        item["text"],
                        seed=item.get("seed", 42)
                    )
                    print(f"✓ Cached: {item['text'][:30]}...")
                except Exception as e:
                    print(f"✗ Failed: {item['text'][:30]}... - {e}")
    
    def run(self):
        """Run complete cache warming."""
        print("Starting daemon cache warming...")
        self.warm_generation_cache()
        self.warm_embedding_cache()
        print("\nCache warming complete!")
    
    def save_common_prompts(self, prompts_file="access.log"):
        """Extract common prompts from access logs."""
        # Parse access logs to find common prompts
        prompt_counts = {}
        
        with open(prompts_file) as f:
            for line in f:
                # Extract prompt from log line
                # Adjust parsing based on your log format
                if "prompt:" in line:
                    prompt = line.split("prompt:")[1].strip()
                    prompt_counts[prompt] = prompt_counts.get(prompt, 0) + 1
        
        # Get top prompts
        top_prompts = sorted(
            prompt_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:50]
        
        # Update warmup data
        self.warmup_data["generation"] = [
            {"prompt": prompt, "seed": 42}
            for prompt, _ in top_prompts
        ]
        
        # Save to file
        with open(self.warmup_file, 'w') as f:
            json.dump(self.warmup_data, f, indent=2)

# Usage
if __name__ == "__main__":
    warmer = DaemonCacheWarmer()
    warmer.run()
```

### Connection Pooling

```python
# connection_pool.py - Daemon connection pooling
import queue
import threading
from contextlib import contextmanager
from steadytext.daemon.client import DaemonClient

class DaemonConnectionPool:
    """Thread-safe connection pool for daemon clients."""
    
    def __init__(self, host="127.0.0.1", port=5557, pool_size=10, timeout=5000):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.pool_size = pool_size
        self._pool = queue.Queue(maxsize=pool_size)
        self._all_connections = []
        self._lock = threading.Lock()
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Create initial connections."""
        for _ in range(self.pool_size):
            conn = self._create_connection()
            if conn:
                self._pool.put(conn)
                self._all_connections.append(conn)
    
    def _create_connection(self):
        """Create new daemon connection."""
        try:
            return DaemonClient(
                host=self.host,
                port=self.port,
                timeout=self.timeout
            )
        except Exception as e:
            print(f"Failed to create connection: {e}")
            return None
    
    @contextmanager
    def get_connection(self, timeout=None):
        """Get connection from pool."""
        connection = None
        try:
            connection = self._pool.get(timeout=timeout)
            yield connection
        finally:
            if connection:
                self._pool.put(connection)
    
    def close_all(self):
        """Close all connections."""
        with self._lock:
            while not self._pool.empty():
                try:
                    conn = self._pool.get_nowait()
                    conn.close()
                except:
                    pass
            self._all_connections.clear()

# Global connection pool
_connection_pool = None

def get_connection_pool():
    """Get or create global connection pool."""
    global _connection_pool
    if _connection_pool is None:
        _connection_pool = DaemonConnectionPool()
    return _connection_pool

# Usage example
def parallel_generate(prompts):
    """Generate text in parallel using connection pool."""
    pool = get_connection_pool()
    results = {}
    
    def process_prompt(idx, prompt):
        with pool.get_connection() as conn:
            if conn:
                try:
                    result = conn.generate(prompt, seed=idx)
                    results[idx] = result
                except Exception as e:
                    results[idx] = f"Error: {e}"
    
    threads = []
    for idx, prompt in enumerate(prompts):
        t = threading.Thread(target=process_prompt, args=(idx, prompt))
        t.start()
        threads.append(t)
    
    for t in threads:
        t.join()
    
    return [results[i] for i in range(len(prompts))]
```

### Memory Optimization

```python
# memory_optimization.py - Optimize daemon memory usage
import gc
import resource
import psutil
from steadytext import get_cache_manager

class DaemonMemoryOptimizer:
    """Optimize memory usage for long-running daemons."""
    
    def __init__(self, max_memory_mb=4096):
        self.max_memory_mb = max_memory_mb
        self.process = psutil.Process()
        self.cache_manager = get_cache_manager()
    
    def set_memory_limits(self):
        """Set process memory limits."""
        # Convert MB to bytes
        max_memory_bytes = self.max_memory_mb * 1024 * 1024
        
        # Set soft and hard limits
        resource.setrlimit(
            resource.RLIMIT_AS,
            (max_memory_bytes, max_memory_bytes)
        )
        
        print(f"Memory limit set to {self.max_memory_mb}MB")
    
    def get_memory_usage(self):
        """Get current memory usage."""
        memory_info = self.process.memory_info()
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": self.process.memory_percent()
        }
    
    def optimize_caches(self):
        """Optimize cache sizes based on memory usage."""
        usage = self.get_memory_usage()
        
        if usage["percent"] > 80:
            # Reduce cache sizes
            print("High memory usage, reducing cache sizes...")
            
            # Get current stats
            stats = self.cache_manager.get_cache_stats()
            
            # Clear least recently used entries
            self.cache_manager.clear_old_entries(keep_ratio=0.5)
            
            # Force garbage collection
            gc.collect()
    
    def periodic_optimization(self, interval=300):
        """Run periodic memory optimization."""
        import time
        import threading
        
        def optimize():
            while True:
                try:
                    self.optimize_caches()
                    usage = self.get_memory_usage()
                    print(f"Memory: {usage['rss_mb']:.1f}MB ({usage['percent']:.1f}%)")
                except Exception as e:
                    print(f"Optimization error: {e}")
                
                time.sleep(interval)
        
        thread = threading.Thread(target=optimize, daemon=True)
        thread.start()

# Apply optimizations at daemon startup
optimizer = DaemonMemoryOptimizer(max_memory_mb=4096)
optimizer.set_memory_limits()
optimizer.periodic_optimization()
```

## Troubleshooting

### Common Issues and Solutions

#### Daemon Won't Start

```bash
# Problem: Address already in use
$ st daemon start
Error: Address already in use (127.0.0.1:5557)

# Solution 1: Check for existing process
$ lsof -i :5557
$ kill -9 <PID>

# Solution 2: Use different port
$ st daemon start --port 5558
```

#### Connection Timeouts

```python
# Problem: Timeout errors with daemon

# Solution 1: Increase timeout
from steadytext.daemon.client import DaemonClient
client = DaemonClient(timeout=10000)  # 10 seconds

# Solution 2: Check daemon health
import requests
try:
    response = requests.get("http://localhost:9090/metrics", timeout=1)
    print("Daemon healthy")
except:
    print("Daemon unhealthy")

# Solution 3: Restart daemon
import subprocess
subprocess.run(["st", "daemon", "restart"])
```

#### Memory Issues

```bash
# Problem: Daemon using too much memory

# Solution 1: Clear caches
$ st cache --clear

# Solution 2: Reduce cache sizes
$ export STEADYTEXT_GENERATION_CACHE_CAPACITY=128
$ export STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB=25
$ st daemon restart

# Solution 3: Monitor memory usage
$ watch -n 1 'ps aux | grep "st daemon" | grep -v grep'
```

#### Performance Degradation

```python
# diagnose_performance.py
import time
import statistics
import steadytext
from steadytext.daemon import use_daemon

def diagnose_daemon_performance():
    """Diagnose daemon performance issues."""
    
    # Test direct mode
    direct_times = []
    for i in range(10):
        start = time.time()
        steadytext.generate("test", seed=i)
        direct_times.append(time.time() - start)
    
    # Test daemon mode
    daemon_times = []
    with use_daemon():
        for i in range(10):
            start = time.time()
            steadytext.generate("test", seed=i+100)
            daemon_times.append(time.time() - start)
    
    print("Direct mode:")
    print(f"  Mean: {statistics.mean(direct_times):.3f}s")
    print(f"  Stdev: {statistics.stdev(direct_times):.3f}s")
    
    print("\nDaemon mode:")
    print(f"  Mean: {statistics.mean(daemon_times):.3f}s")
    print(f"  Stdev: {statistics.stdev(daemon_times):.3f}s")
    
    if statistics.mean(daemon_times) > statistics.mean(direct_times):
        print("\nWARNING: Daemon is slower than direct mode!")
        print("Possible causes:")
        print("- Network latency")
        print("- Daemon overloaded")
        print("- Cache thrashing")

diagnose_daemon_performance()
```

### Debug Checklist

1. **Check daemon status**
   ```bash
   st daemon status --json | jq .
   ```

2. **Verify connectivity**
   ```bash
   nc -zv 127.0.0.1 5557
   ```

3. **Check logs**
   ```bash
   tail -f /var/log/steadytext/daemon.log
   grep ERROR /var/log/steadytext/daemon.error.log
   ```

4. **Monitor resources**
   ```bash
   htop -p $(pgrep -f "st daemon")
   ```

5. **Test basic operations**
   ```python
   from steadytext.daemon.client import DaemonClient
   client = DaemonClient()
   print(client._send_request({"type": "ping"}))
   ```

## Best Practices

### 1. Production Configuration

```bash
# production.env
STEADYTEXT_DAEMON_HOST=0.0.0.0
STEADYTEXT_DAEMON_PORT=5557
STEADYTEXT_GENERATION_CACHE_CAPACITY=2048
STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB=500
STEADYTEXT_EMBEDDING_CACHE_CAPACITY=4096
STEADYTEXT_EMBEDDING_CACHE_MAX_SIZE_MB=1000
STEADYTEXT_DEFAULT_SEED=42
PYTHONUNBUFFERED=1
```

### 2. Health Monitoring

```python
# health_check.py
def health_check():
    """Comprehensive daemon health check."""
    checks = {
        "daemon_running": False,
        "response_time": None,
        "cache_available": False,
        "memory_ok": False
    }
    
    # Check if daemon is running
    try:
        result = subprocess.run(
            ["st", "daemon", "status", "--json"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            status = json.loads(result.stdout)
            checks["daemon_running"] = status.get("running", False)
    except:
        pass
    
    # Check response time
    if checks["daemon_running"]:
        start = time.time()
        try:
            steadytext.generate("health check", seed=42)
            checks["response_time"] = time.time() - start
        except:
            pass
    
    # Check cache
    try:
        cache_manager = get_cache_manager()
        stats = cache_manager.get_cache_stats()
        checks["cache_available"] = True
    except:
        pass
    
    # Check memory
    if checks["daemon_running"]:
        memory = psutil.Process().memory_info().rss / 1024 / 1024
        checks["memory_ok"] = memory < 4096  # 4GB limit
    
    return checks
```

### 3. Graceful Degradation

```python
# graceful_degradation.py
class ResilientClient:
    """Client with graceful degradation."""
    
    def __init__(self):
        self.use_daemon = True
        self.fallback_count = 0
        self.max_fallbacks = 3
    
    def generate(self, prompt, **kwargs):
        """Generate with automatic fallback."""
        if self.use_daemon and self.fallback_count < self.max_fallbacks:
            try:
                with use_daemon():
                    return steadytext.generate(prompt, **kwargs)
            except Exception as e:
                self.fallback_count += 1
                print(f"Daemon failed ({self.fallback_count}/{self.max_fallbacks}): {e}")
                
                if self.fallback_count >= self.max_fallbacks:
                    self.use_daemon = False
                    print("Disabling daemon due to repeated failures")
        
        # Direct mode fallback
        return steadytext.generate(prompt, **kwargs)
```

### 4. Security Considerations

```python
# secure_daemon.py
import ssl
import secrets

class SecureDaemonConfig:
    """Secure daemon configuration."""
    
    @staticmethod
    def generate_auth_token():
        """Generate secure auth token."""
        return secrets.token_urlsafe(32)
    
    @staticmethod
    def configure_tls(cert_path, key_path):
        """Configure TLS for daemon."""
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.load_cert_chain(cert_path, key_path)
        return context
    
    @staticmethod
    def restrict_bind_address():
        """Restrict daemon to localhost only."""
        return {
            "host": "127.0.0.1",  # Never use 0.0.0.0 in production
            "port": 5557
        }
```

This comprehensive guide covers all aspects of using SteadyText's daemon mode, from basic usage to advanced production deployments. The daemon provides significant performance benefits while maintaining the simplicity and reliability that SteadyText is known for.