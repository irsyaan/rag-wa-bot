# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Copy requirements first for layer caching
COPY requirements.txt .

# If wheelhouse exists (offline builds), use it; otherwise pip install normally
COPY wheelhouse/ wheelhouse/ 2>/dev/null || true

RUN if [ -d "wheelhouse" ] && [ "$(ls -A wheelhouse 2>/dev/null)" ]; then \
        pip install --no-cache-dir --no-index --find-links=wheelhouse -r requirements.txt; \
    else \
        pip install --no-cache-dir -r requirements.txt; \
    fi

# ── Runtime stage ────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy application code
COPY app/ ./app/
COPY migrations/ ./migrations/

# Create non-root user
RUN useradd --create-home --shell /bin/bash botuser
USER botuser

# Default command
CMD ["python", "-m", "app.main"]
