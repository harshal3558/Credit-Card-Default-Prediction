
# =========================
# 1. Base image
# =========================
FROM python:3.10-slim

# =========================
# 2. Environment settings
# =========================
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV MLFLOW_DISABLE_AGENT_HINT=1
ENV MLFLOW_TRACKING_URI=""

# =========================
# 3. Set working directory
# =========================
WORKDIR /app

# =========================
# 4. System dependencies
# =========================
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# =========================
# 5. Install Python deps (cached layer)
# =========================
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# =========================
# 6. Copy project source
# =========================
COPY . .

# Install the CCDP package in editable mode
RUN pip install --no-cache-dir -e .

# =========================
# 7. Train model at build time
#    (generates artifacts/model.pkl & artifacts/preprocessor.pkl)
# =========================
RUN python main.py

# =========================
# 8. Expose port
# =========================
EXPOSE 5000

# =========================
# 9. Run with Gunicorn (production WSGI server)
# =========================
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "application:app"]
