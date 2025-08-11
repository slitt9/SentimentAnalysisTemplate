FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
# Install CPU-only PyTorch first, then other requirements
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create artifacts directory
RUN mkdir -p artifacts

# Copy your trained model artifacts
COPY artifacts/best_model.ckpt artifacts/
COPY artifacts/vocab.json artifacts/
COPY artifacts/embeddings.pt artifacts/

# Expose port
EXPOSE 8000

# Start the API - Railway handles health checks
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]