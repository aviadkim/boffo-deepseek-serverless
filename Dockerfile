# RunPod Serverless - DeepSeek OCR
# Optimized for serverless execution with fast cold starts

FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies (Tesseract OCR + Poppler for PDF)
RUN apt-get update && \
    apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    pytesseract==0.3.10 \
    pillow==10.1.0 \
    pdf2image==1.16.3 \
    runpod==1.5.0 \
    opencv-python-headless==4.8.1.78

# Tesseract is pre-installed (~100MB) - fast startup!

# Copy handler code
COPY handler.py /app/handler.py

# Set the entrypoint
CMD ["python", "-u", "handler.py"]
