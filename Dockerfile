# Slim base image: much smaller than the full python:3.10 image
FROM python:3.10-slim

# Don't write .pyc files; flush logs straight to `docker logs`
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OUTPUT_DIR=/app/artifacts

WORKDIR /app

# Copy requirements first so the pip layer stays cached when only code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ .

# Run as a non-root user
RUN useradd --create-home appuser && mkdir -p /app/artifacts && chown -R appuser /app
USER appuser

# Mount a volume here to get the model and metrics back on the host
VOLUME ["/app/artifacts"]

CMD ["python", "train.py"]
