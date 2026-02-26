# Build context must be the repository root (parent of server/)
# docker build -f server/Dockerfile -t bulkocr-server .
FROM python:3.12-slim

# System deps for PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
        libmupdf-dev \
        mupdf-tools \
        gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cache layer)
COPY server/requirements.txt /app/server/requirements.txt
RUN pip install --no-cache-dir -r server/requirements.txt

# Copy the whole repository so that config.py is at /app/config.py
COPY . /app

# Jobs output dir
RUN mkdir -p /app/server/jobs

WORKDIR /app/server

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
