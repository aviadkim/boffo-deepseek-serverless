# RunPod Serverless - DeepSeek OCR
# Optimized for serverless execution with fast cold starts

FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y poppler-utils && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    transformers==4.36.0 \
    torch==2.1.0 \
    pillow==10.1.0 \
    pdf2image==1.16.3 \
    runpod==1.5.0

# Note: Model will be downloaded on first run to avoid build timeout
# This makes the Docker image much smaller and builds faster

# Copy handler code
COPY handler.py /app/handler.py

# Set the entrypoint
CMD ["python", "-u", "handler.py"]
