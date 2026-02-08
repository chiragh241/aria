# Aria - Personal AI Assistant
# Main Application Container

FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build dependencies
    build-essential \
    # Git for package installation
    git \
    # For Playwright/Chromium
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libdbus-1-3 \
    libxkbcommon0 \
    libatspi2.0-0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    # For audio processing
    ffmpeg \
    # For document processing
    libmagic1 \
    # Node.js for frontend build
    nodejs \
    npm \
    # curl for Docker CLI install
    curl \
    ca-certificates \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Install Docker CLI + Compose plugin from official apt repo
RUN install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc && \
    chmod a+r /etc/apt/keyrings/docker.asc && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
    > /etc/apt/sources.list.d/docker.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends docker-ce-cli docker-compose-plugin && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user and add to root group for Docker socket access
RUN useradd -m -u 1000 -G root aria && \
    mkdir -p /app /data && \
    chown -R aria:aria /app /data

WORKDIR /app

# Install uv for faster package installation
RUN pip install uv

# -- Dependency layer (cached unless pyproject.toml changes) --
COPY --chown=aria:aria pyproject.toml ./

# Install dependencies only (no editable install yet — source isn't copied)
# Use a two-step approach: first install deps, then the package
RUN uv pip install --system hatchling && \
    uv pip install --system $(python -c "\
import tomllib, pathlib; \
d = tomllib.loads(pathlib.Path('pyproject.toml').read_text()); \
print(' '.join(d['project']['dependencies']))")

# Install Playwright browsers
RUN playwright install chromium && \
    playwright install-deps chromium

# -- Source layer --
COPY --chown=aria:aria . .

# Build frontend
RUN cd src/web/frontend && \
    npm install && \
    npm run build

# Install the package itself (not editable in container)
RUN uv pip install --system .

# Switch to non-root user
USER aria

# Create data directories
RUN mkdir -p /data/chromadb /data/logs /data/whatsapp-session /data/conversations /data/cognee

# Expose ports
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8080/api/system/health')" || exit 1

# Default command (--skip-setup since Docker env is pre-configured)
CMD ["python", "-m", "src.main", "--skip-setup"]
