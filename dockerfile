# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the sentence transformer model during build
# This saves time on container startup
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application code
COPY Phase4.py phase4.py

# Note: ChromaDB directory will be mounted as volume
# No need to create it here since it needs the actual data

# Expose port
EXPOSE 8002

# Run the application
CMD ["uvicorn", "phase4:app", "--host", "0.0.0.0", "--port", "8002"]