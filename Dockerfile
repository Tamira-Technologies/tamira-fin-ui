# syntax=docker/dockerfile:1.6

# ─────────────────────────────────────────────────────────────────────────────
# Base image
# Use official Python 3.12 slim for a lightweight runtime
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Avoid writing .pyc files, ensure unbuffered logs, and disable pip cache
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Streamlit default port
ENV PORT=8501

# Set a sensible default storage path for portfolio persistence (can be overridden)
ENV TAMIRA_STORAGE_PATH=/data/portfolios.json

# Ensure our source path is importable without installing as a package
ENV PYTHONPATH=/app/src

# ─────────────────────────────────────────────────────────────────────────────
# System dependencies
# - curl/ca-certificates for healthcheck and potential runtime needs
# - libgomp1 for scikit-learn (OpenMP) runtime
# ─────────────────────────────────────────────────────────────────────────────
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────────────────────────────────────────
# Create non-root user and directories
# ─────────────────────────────────────────────────────────────────────────────
RUN useradd -m -u 10001 appuser
WORKDIR /app

# Data directory for local persistence (bind mount or volume recommended)
RUN mkdir -p /data && chown -R appuser:appuser /data

# ─────────────────────────────────────────────────────────────────────────────
# Install Python dependencies
# Note:
# - Torch CPU wheels are installed from PyPI by default; if needed, adjust index.
# - OpenBB is included per project dependency; you can remove it if not required.
# ─────────────────────────────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    torch \
    numpy \
    pandas \
    matplotlib \
    scikit-learn \
    yfinance \
    streamlit \
    plotly \
    "pydantic>=2" \
    "pydantic-settings>=2" \
    "openbb>=4.1.3"

# ─────────────────────────────────────────────────────────────────────────────
# Copy application source
# ─────────────────────────────────────────────────────────────────────────────
# Copy only the src directory to keep image lean; if you need other files (e.g., README),
# you can add them as needed.
COPY src/ ./src/

# Adjust ownership
RUN chown -R appuser:appuser /app

# ─────────────────────────────────────────────────────────────────────────────
# Expose Streamlit port and define healthcheck
# ─────────────────────────────────────────────────────────────────────────────
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail http://localhost:${PORT}/ || exit 1

# ─────────────────────────────────────────────────────────────────────────────
# Switch to non-root user
# ─────────────────────────────────────────────────────────────────────────────
USER appuser

# ─────────────────────────────────────────────────────────────────────────────
# Default command:
# - Run the Streamlit dashboard
# - Bind to 0.0.0.0 and specified port
# ─────────────────────────────────────────────────────────────────────────────
CMD ["streamlit", "run", "src/tamira_fin_ui/dashboard_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
