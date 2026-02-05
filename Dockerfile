FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port 7860 (HuggingFace Spaces default)
EXPOSE 7860

# Run the application
CMD ["python", "run.py", "--host", "0.0.0.0", "--port", "7860"]
