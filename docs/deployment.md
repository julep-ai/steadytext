# Deployment Guide

This guide covers deploying SteadyText in various production environments, from simple servers to cloud-native architectures.

## Table of Contents

- [Deployment Options](#deployment-options)
- [System Requirements](#system-requirements)
- [Basic Server Deployment](#basic-server-deployment)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Cloud Deployments](#cloud-deployments)
  - [AWS](#aws-deployment)
  - [Google Cloud](#google-cloud-deployment)
  - [Azure](#azure-deployment)
- [PostgreSQL Extension Deployment](#postgresql-extension-deployment)
- [Production Configuration](#production-configuration)
- [Monitoring and Observability](#monitoring-and-observability)
- [Security Considerations](#security-considerations)
- [High Availability](#high-availability)
- [Troubleshooting](#troubleshooting)

## Deployment Options

### Overview of Deployment Methods

| Method | Best For | Complexity | Scalability |
|--------|----------|------------|-------------|
| Direct Install | Development | Low | Limited |
| Systemd Service | Single server | Medium | Vertical |
| Docker | Containerized apps | Medium | Horizontal |
| Kubernetes | Cloud-native | High | Auto-scaling |
| PostgreSQL Extension | Database-integrated | Medium | With database |

## System Requirements

### Minimum Requirements

- **CPU**: 2 cores (4+ recommended)
- **RAM**: 4GB (8GB+ recommended)
- **Storage**: 10GB (for models and cache)
- **OS**: Linux (Ubuntu 20.04+), macOS, Windows Server
- **Python**: 3.8+ (3.10+ recommended)

### Recommended Production Specs

```yaml
# Production server specifications
production:
  cpu: 8 cores
  ram: 16GB
  storage: 50GB SSD
  network: 1Gbps
  os: Ubuntu 22.04 LTS
```

### Resource Planning

```python
# Calculate resource requirements
def calculate_resources(concurrent_users, cache_size_gb, model_size):
    """Estimate resource requirements."""
    
    # Memory calculation
    base_memory_gb = 2  # OS and services
    model_memory_gb = {
        'small': 2,
        'large': 4
    }[model_size]
    cache_memory_gb = cache_size_gb
    worker_memory_gb = concurrent_users * 0.1  # 100MB per concurrent user
    
    total_memory_gb = (
        base_memory_gb + 
        model_memory_gb + 
        cache_memory_gb + 
        worker_memory_gb
    )
    
    # CPU calculation
    cpu_cores = max(4, concurrent_users // 10)
    
    return {
        'memory_gb': total_memory_gb,
        'cpu_cores': cpu_cores,
        'storage_gb': 10 + cache_size_gb * 2  # 2x cache for growth
    }

# Example: 100 concurrent users, 10GB cache, large model
resources = calculate_resources(100, 10, 'large')
print(f"Required: {resources['memory_gb']}GB RAM, {resources['cpu_cores']} cores")
```

## Basic Server Deployment

### 1. System Setup

```bash
# Ubuntu/Debian setup
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3-pip

# Create dedicated user
sudo useradd -m -s /bin/bash steadytext
sudo mkdir -p /opt/steadytext
sudo chown steadytext:steadytext /opt/steadytext

# Switch to steadytext user
sudo su - steadytext
cd /opt/steadytext
```

### 2. Python Environment

```bash
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install SteadyText
pip install steadytext

# Preload models
st models download --all
st models preload
```

### 3. Systemd Service

Create `/etc/systemd/system/steadytext.service`:

```ini
[Unit]
Description=SteadyText Daemon Service
After=network.target

[Service]
Type=simple
User=steadytext
Group=steadytext
WorkingDirectory=/opt/steadytext
Environment="PATH=/opt/steadytext/venv/bin"
Environment="STEADYTEXT_GENERATION_CACHE_CAPACITY=2048"
Environment="STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB=1024"
Environment="STEADYTEXT_EMBEDDING_CACHE_CAPACITY=4096"
Environment="STEADYTEXT_EMBEDDING_CACHE_MAX_SIZE_MB=2048"
ExecStart=/opt/steadytext/venv/bin/st daemon start --foreground --host 0.0.0.0 --port 5557
Restart=always
RestartSec=10

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/steadytext /home/steadytext/.cache

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable steadytext
sudo systemctl start steadytext
sudo systemctl status steadytext
```

### 4. Nginx Reverse Proxy

Install and configure Nginx:

```nginx
# /etc/nginx/sites-available/steadytext
upstream steadytext_backend {
    server 127.0.0.1:5557;
    keepalive 32;
}

server {
    listen 80;
    server_name steadytext.example.com;

    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name steadytext.example.com;

    # SSL configuration
    ssl_certificate /etc/letsencrypt/live/steadytext.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/steadytext.example.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # WebSocket support for streaming
    location /ws {
        proxy_pass http://steadytext_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 3600s;
    }

    # Regular HTTP API
    location / {
        proxy_pass http://steadytext_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
```

Enable site:

```bash
sudo ln -s /etc/nginx/sites-available/steadytext /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

## Docker Deployment

### 1. Dockerfile

```dockerfile
# Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -s /bin/bash steadytext

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download models during build (optional)
RUN python -c "import steadytext; steadytext.preload_models()"

# Copy application code
COPY . .

# Change ownership
RUN chown -R steadytext:steadytext /app

# Switch to non-root user
USER steadytext

# Expose port
EXPOSE 5557

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD st daemon status || exit 1

# Start daemon
CMD ["st", "daemon", "start", "--foreground", "--host", "0.0.0.0"]
```

### 2. Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  steadytext:
    build: .
    image: steadytext:latest
    container_name: steadytext-daemon
    ports:
      - "5557:5557"
    environment:
      - STEADYTEXT_GENERATION_CACHE_CAPACITY=2048
      - STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB=1024
      - STEADYTEXT_EMBEDDING_CACHE_CAPACITY=4096
      - STEADYTEXT_EMBEDDING_CACHE_MAX_SIZE_MB=2048
    volumes:
      - steadytext-cache:/home/steadytext/.cache
      - steadytext-models:/app/models
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G

  nginx:
    image: nginx:alpine
    container_name: steadytext-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./certs:/etc/nginx/certs:ro
    depends_on:
      - steadytext
    restart: unless-stopped

volumes:
  steadytext-cache:
  steadytext-models:
```

### 3. Build and Run

```bash
# Build image
docker build -t steadytext:latest .

# Run with docker-compose
docker-compose up -d

# Check logs
docker-compose logs -f steadytext

# Scale horizontally
docker-compose up -d --scale steadytext=3
```

## Kubernetes Deployment

### 1. ConfigMap

```yaml
# steadytext-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: steadytext-config
  namespace: steadytext
data:
  STEADYTEXT_GENERATION_CACHE_CAPACITY: "2048"
  STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB: "1024"
  STEADYTEXT_EMBEDDING_CACHE_CAPACITY: "4096"
  STEADYTEXT_EMBEDDING_CACHE_MAX_SIZE_MB: "2048"
  DAEMON_HOST: "0.0.0.0"
  DAEMON_PORT: "5557"
```

### 2. Deployment

```yaml
# steadytext-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: steadytext
  namespace: steadytext
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
          name: daemon
        envFrom:
        - configMapRef:
            name: steadytext-config
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
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          exec:
            command:
            - st
            - daemon
            - status
          initialDelaySeconds: 30
          periodSeconds: 10
        volumeMounts:
        - name: cache
          mountPath: /home/steadytext/.cache
        - name: models
          mountPath: /app/models
      volumes:
      - name: cache
        persistentVolumeClaim:
          claimName: steadytext-cache-pvc
      - name: models
        persistentVolumeClaim:
          claimName: steadytext-models-pvc
```

### 3. Service

```yaml
# steadytext-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: steadytext-service
  namespace: steadytext
spec:
  selector:
    app: steadytext
  ports:
  - port: 5557
    targetPort: 5557
    name: daemon
  type: ClusterIP
```

### 4. Horizontal Pod Autoscaler

```yaml
# steadytext-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: steadytext-hpa
  namespace: steadytext
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: steadytext
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### 5. Ingress

```yaml
# steadytext-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: steadytext-ingress
  namespace: steadytext
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - steadytext.example.com
    secretName: steadytext-tls
  rules:
  - host: steadytext.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: steadytext-service
            port:
              number: 5557
```

### 6. Deploy to Kubernetes

```bash
# Create namespace
kubectl create namespace steadytext

# Apply configurations
kubectl apply -f steadytext-config.yaml
kubectl apply -f steadytext-pvc.yaml  # Create PVCs first
kubectl apply -f steadytext-deployment.yaml
kubectl apply -f steadytext-service.yaml
kubectl apply -f steadytext-hpa.yaml
kubectl apply -f steadytext-ingress.yaml

# Check status
kubectl -n steadytext get pods
kubectl -n steadytext logs -f deployment/steadytext
```

## Cloud Deployments

### AWS Deployment

#### 1. EC2 Instance

```bash
# Launch EC2 instance (via AWS CLI)
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type t3.xlarge \
  --key-name your-key \
  --security-group-ids sg-xxxxxx \
  --subnet-id subnet-xxxxxx \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=steadytext-server}]' \
  --user-data file://setup.sh
```

Setup script (`setup.sh`):

```bash
#!/bin/bash
# Update system
apt update && apt upgrade -y

# Install dependencies
apt install -y python3.10 python3.10-venv python3-pip nginx

# Install SteadyText
pip3 install steadytext

# Configure and start daemon
cat > /etc/systemd/system/steadytext.service << EOF
[Unit]
Description=SteadyText Daemon
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/st daemon start --foreground
Restart=always
Environment="STEADYTEXT_GENERATION_CACHE_CAPACITY=2048"

[Install]
WantedBy=multi-user.target
EOF

systemctl enable steadytext
systemctl start steadytext
```

#### 2. ECS Fargate

```json
// task-definition.json
{
  "family": "steadytext",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "8192",
  "containerDefinitions": [
    {
      "name": "steadytext",
      "image": "your-ecr-repo/steadytext:latest",
      "portMappings": [
        {
          "containerPort": 5557,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "STEADYTEXT_GENERATION_CACHE_CAPACITY",
          "value": "2048"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/steadytext",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "st daemon status || exit 1"],
        "interval": 30,
        "timeout": 10,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

#### 3. Lambda Function

```python
# lambda_function.py
import json
import steadytext

def lambda_handler(event, context):
    """AWS Lambda handler for SteadyText."""
    
    # Parse request
    body = json.loads(event.get('body', '{}'))
    prompt = body.get('prompt', '')
    seed = body.get('seed', 42)
    
    # Generate text
    result = steadytext.generate(prompt, seed=seed)
    
    if result is None:
        return {
            'statusCode': 503,
            'body': json.dumps({'error': 'Model not available'})
        }
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'text': result,
            'seed': seed
        })
    }
```

### Google Cloud Deployment

#### 1. Compute Engine

```bash
# Create instance
gcloud compute instances create steadytext-server \
  --machine-type=n2-standard-4 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=50GB \
  --metadata-from-file startup-script=setup.sh
```

#### 2. Cloud Run

```dockerfile
# Dockerfile for Cloud Run
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Cloud Run sets PORT environment variable
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app
```

Deploy:

```bash
# Build and push
gcloud builds submit --tag gcr.io/PROJECT-ID/steadytext

# Deploy
gcloud run deploy steadytext \
  --image gcr.io/PROJECT-ID/steadytext \
  --platform managed \
  --memory 8Gi \
  --cpu 4 \
  --timeout 300 \
  --concurrency 10
```

#### 3. Kubernetes Engine (GKE)

```bash
# Create cluster
gcloud container clusters create steadytext-cluster \
  --num-nodes=3 \
  --machine-type=n2-standard-4 \
  --enable-autoscaling \
  --min-nodes=2 \
  --max-nodes=10

# Deploy application
kubectl apply -f k8s/
```

### Azure Deployment

#### 1. Virtual Machine

```bash
# Create VM
az vm create \
  --resource-group steadytext-rg \
  --name steadytext-vm \
  --image UbuntuLTS \
  --size Standard_D4s_v3 \
  --admin-username azureuser \
  --generate-ssh-keys \
  --custom-data setup.sh
```

#### 2. Container Instances

```bash
# Deploy container
az container create \
  --resource-group steadytext-rg \
  --name steadytext \
  --image your-acr.azurecr.io/steadytext:latest \
  --cpu 4 \
  --memory 8 \
  --port 5557 \
  --environment-variables \
    STEADYTEXT_GENERATION_CACHE_CAPACITY=2048
```

#### 3. App Service

```bash
# Create App Service plan
az appservice plan create \
  --name steadytext-plan \
  --resource-group steadytext-rg \
  --sku P2v3 \
  --is-linux

# Deploy container
az webapp create \
  --resource-group steadytext-rg \
  --plan steadytext-plan \
  --name steadytext-app \
  --deployment-container-image-name your-acr.azurecr.io/steadytext:latest
```

## PostgreSQL Extension Deployment

### 1. Standalone PostgreSQL

```bash
# Install PostgreSQL and dependencies
sudo apt install postgresql-15 postgresql-server-dev-15 python3-dev

# Install pg_steadytext
cd pg_steadytext
sudo ./install.sh

# Configure PostgreSQL
sudo -u postgres psql << EOF
CREATE DATABASE steadytext_db;
\c steadytext_db
CREATE EXTENSION plpython3u;
CREATE EXTENSION vector;
CREATE EXTENSION pg_steadytext;
EOF
```

### 2. Docker PostgreSQL

```dockerfile
# Dockerfile.postgres
FROM postgres:15

# Install dependencies
RUN apt-get update && apt-get install -y \
    postgresql-plpython3-15 \
    python3-pip \
    build-essential

# Install Python packages
RUN pip3 install steadytext pyzmq numpy

# Copy extension files
COPY pg_steadytext /tmp/pg_steadytext
RUN cd /tmp/pg_steadytext && ./install.sh

# Initialize script
COPY init.sql /docker-entrypoint-initdb.d/
```

### 3. Managed PostgreSQL (RDS/CloudSQL)

Most managed PostgreSQL services don't support custom extensions. Options:

1. **Use separate daemon**: Run SteadyText daemon separately
2. **API wrapper**: Create REST API that PostgreSQL can call
3. **Self-managed**: Use EC2/GCE with PostgreSQL

## Production Configuration

### 1. Environment Configuration

```bash
# /etc/environment or .env file
# Performance
STEADYTEXT_GENERATION_CACHE_CAPACITY=4096
STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB=2048
STEADYTEXT_EMBEDDING_CACHE_CAPACITY=8192
STEADYTEXT_EMBEDDING_CACHE_MAX_SIZE_MB=4096

# Security
STEADYTEXT_DAEMON_AUTH_TOKEN=your-secret-token
STEADYTEXT_ALLOWED_HOSTS=steadytext.example.com

# Monitoring
STEADYTEXT_METRICS_ENABLED=true
STEADYTEXT_METRICS_PORT=9090

# Logging
STEADYTEXT_LOG_LEVEL=INFO
STEADYTEXT_LOG_FILE=/var/log/steadytext/daemon.log
```

### 2. Resource Limits

```yaml
# systemd resource limits
[Service]
# Memory limits
MemoryMax=16G
MemoryHigh=12G

# CPU limits
CPUQuota=400%  # 4 cores

# File descriptor limits
LimitNOFILE=65536

# Process limits
TasksMax=1024
```

### 3. Logging Configuration

```python
# logging_config.py
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
        'json': {
            'class': 'pythonjsonlogger.jsonlogger.JsonFormatter',
            'format': '%(asctime)s %(name)s %(levelname)s %(message)s'
        }
    },
    'handlers': {
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': '/var/log/steadytext/daemon.log',
            'maxBytes': 100_000_000,  # 100MB
            'backupCount': 10,
            'formatter': 'json'
        },
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard'
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['file', 'console']
    }
}
```

## Monitoring and Observability

### 1. Prometheus Metrics

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Define metrics
generation_requests = Counter('steadytext_generation_requests_total', 
                            'Total generation requests')
generation_duration = Histogram('steadytext_generation_duration_seconds',
                              'Generation request duration')
cache_hits = Counter('steadytext_cache_hits_total', 
                    'Cache hit count', ['cache_type'])
active_connections = Gauge('steadytext_active_connections',
                          'Number of active connections')

# Start metrics server
start_http_server(9090)
```

### 2. Grafana Dashboard

```json
{
  "dashboard": {
    "title": "SteadyText Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(steadytext_generation_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Response Time",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, steadytext_generation_duration_seconds)"
          }
        ]
      },
      {
        "title": "Cache Hit Rate",
        "targets": [
          {
            "expr": "rate(steadytext_cache_hits_total[5m]) / rate(steadytext_generation_requests_total[5m])"
          }
        ]
      }
    ]
  }
}
```

### 3. Health Checks

```python
# healthcheck.py
from flask import Flask, jsonify
import steadytext

app = Flask(__name__)

@app.route('/health')
def health():
    """Basic health check."""
    return jsonify({'status': 'healthy'})

@app.route('/ready')
def ready():
    """Readiness check with model test."""
    try:
        result = steadytext.generate("test", seed=42)
        if result:
            return jsonify({'status': 'ready'})
        else:
            return jsonify({'status': 'not ready'}), 503
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 503

@app.route('/metrics')
def metrics():
    """Custom metrics endpoint."""
    from steadytext import get_cache_manager
    stats = get_cache_manager().get_cache_stats()
    
    return jsonify({
        'cache': stats,
        'models': {
            'loaded': True,
            'generation_model': 'gemma-3n',
            'embedding_model': 'qwen3'
        }
    })
```

## Security Considerations

### 1. Network Security

```nginx
# Rate limiting in Nginx
limit_req_zone $binary_remote_addr zone=steadytext:10m rate=10r/s;

server {
    location / {
        limit_req zone=steadytext burst=20 nodelay;
        # ... proxy settings
    }
}
```

### 2. Authentication

```python
# Simple token authentication
import hmac
import hashlib

def verify_token(request_token, secret_key):
    """Verify API token."""
    expected = hmac.new(
        secret_key.encode(),
        b"steadytext",
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(request_token, expected)

# Middleware
def require_auth(func):
    def wrapper(*args, **kwargs):
        token = request.headers.get('X-API-Token')
        if not verify_token(token, SECRET_KEY):
            abort(401)
        return func(*args, **kwargs)
    return wrapper
```

### 3. Input Validation

```python
def validate_input(prompt: str) -> bool:
    """Validate user input."""
    # Length check
    if len(prompt) > 10000:
        return False
    
    # Character validation
    if not prompt.isprintable():
        return False
    
    # Rate limiting per user
    if check_rate_limit(user_id):
        return False
    
    return True
```

## High Availability

### 1. Load Balancing

```nginx
# Nginx load balancing
upstream steadytext_cluster {
    least_conn;
    server steadytext1.internal:5557 max_fails=3 fail_timeout=30s;
    server steadytext2.internal:5557 max_fails=3 fail_timeout=30s;
    server steadytext3.internal:5557 max_fails=3 fail_timeout=30s;
    keepalive 32;
}
```

### 2. Failover Configuration

```python
# Client with failover
class FailoverClient:
    def __init__(self, servers):
        self.servers = servers
        self.current_server = 0
    
    def generate(self, prompt, max_retries=3):
        """Generate with automatic failover."""
        for attempt in range(max_retries):
            try:
                server = self.servers[self.current_server]
                return self._call_server(server, prompt)
            except Exception as e:
                logger.warning(f"Server {server} failed: {e}")
                self.current_server = (self.current_server + 1) % len(self.servers)
        
        raise Exception("All servers failed")
```

### 3. Backup and Recovery

```bash
#!/bin/bash
# backup.sh - Backup cache and models

BACKUP_DIR="/backup/steadytext/$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"

# Backup cache
rsync -av ~/.cache/steadytext/ "$BACKUP_DIR/cache/"

# Backup models
rsync -av ~/.cache/steadytext/models/ "$BACKUP_DIR/models/"

# Backup configuration
cp /etc/steadytext/* "$BACKUP_DIR/config/"

# Compress
tar -czf "$BACKUP_DIR.tar.gz" "$BACKUP_DIR"
rm -rf "$BACKUP_DIR"

# Upload to S3
aws s3 cp "$BACKUP_DIR.tar.gz" s3://backup-bucket/steadytext/
```

## Troubleshooting

### Common Deployment Issues

#### 1. Model Loading Failures

```bash
# Check model directory
ls -la ~/.cache/steadytext/models/

# Download models manually
st models download --all

# Verify model integrity
st models status --verify
```

#### 2. Memory Issues

```bash
# Check memory usage
free -h
ps aux | grep steadytext

# Adjust cache sizes
export STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB=500
export STEADYTEXT_EMBEDDING_CACHE_MAX_SIZE_MB=1000
```

#### 3. Performance Issues

```bash
# Check daemon status
st daemon status --verbose

# Monitor resource usage
htop -p $(pgrep -f steadytext)

# Check cache performance
st cache --status --detailed
```

### Debugging Production Issues

```python
# debug_helper.py
import logging
import traceback
from functools import wraps

def debug_on_error(func):
    """Decorator to help debug production issues."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.__name__}:")
            logging.error(f"Args: {args}")
            logging.error(f"Kwargs: {kwargs}")
            logging.error(traceback.format_exc())
            raise
    return wrapper

# Use in production
@debug_on_error
def generate_text(prompt):
    return steadytext.generate(prompt)
```

## Deployment Checklist

### Pre-Deployment

- [ ] System requirements verified
- [ ] Python 3.8+ installed
- [ ] Sufficient disk space (10GB+)
- [ ] Network connectivity tested
- [ ] Security groups/firewalls configured

### Deployment

- [ ] SteadyText installed
- [ ] Models downloaded
- [ ] Daemon configured
- [ ] Service scripts created
- [ ] Reverse proxy configured
- [ ] SSL certificates installed

### Post-Deployment

- [ ] Health checks passing
- [ ] Monitoring configured
- [ ] Logs being collected
- [ ] Backup strategy implemented
- [ ] Performance benchmarked
- [ ] Documentation updated

### Production Readiness

- [ ] High availability configured
- [ ] Auto-scaling enabled
- [ ] Rate limiting active
- [ ] Security hardened
- [ ] Disaster recovery tested
- [ ] Team trained

## Support

- **Documentation**: [steadytext.readthedocs.io](https://steadytext.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/diwank/steadytext/issues)
- **Community**: [Discussions](https://github.com/diwank/steadytext/discussions)