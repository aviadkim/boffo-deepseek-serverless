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

# Download DeepSeek OCR model (baked into image for fast cold starts)
RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    import torch; \
    print('Downloading DeepSeek OCR model...'); \
    model = AutoModelForCausalLM.from_pretrained('deepseek-ai/DeepSeek-OCR', trust_remote_code=True, torch_dtype=torch.bfloat16); \
    tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-OCR', trust_remote_code=True); \
    print('Model cached in image')"

# Copy handler code
COPY handler.py /app/handler.py

# Set the entrypoint
CMD ["python", "-u", "handler.py"]
