# BOFFO DeepSeek OCR - Serverless Deployment

**Serverless PDF extraction using DeepSeek OCR on RunPod**

This repository contains the Docker configuration for deploying DeepSeek OCR as a serverless function on RunPod.

## What This Does

Extracts financial portfolio data from bank statements (PDF) using AI vision:
- Portfolio summary (total value, currency, dates)
- Holdings table (securities, ISINs, quantities, values)
- Asset allocation breakdown

## Deployment

### RunPod Serverless

1. Go to [RunPod Serverless Console](https://www.runpod.io/console/serverless/user/templates)
2. Click "New Template"
3. Select "Build from GitHub"
4. Use this repository URL
5. Set Dockerfile path: `Dockerfile`
6. Click "Build"
7. Create endpoint using this template

### Configuration

- **GPU Required:** 24GB+ (RTX A5000, RTX 4090)
- **Container Disk:** 50 GB
- **Idle Timeout:** 5 seconds
- **Execution Timeout:** 120 seconds

## API Usage

```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "pdf_base64": "BASE64_ENCODED_PDF_DATA",
      "filename": "statement.pdf"
    }
  }'
```

## Response Format

```json
{
  "status": "success",
  "pages_processed": 5,
  "summary": {
    "total_portfolio_value": 1234567.89,
    "currency": "USD",
    "client_id": "123456",
    "statement_date": "2025-09-30"
  },
  "holdings": [
    {
      "security_name": "Apple Inc",
      "isin": "US0378331005",
      "quantity": 100,
      "market_value": 15000
    }
  ],
  "processing_time_seconds": 28.5
}
```

## Cost

- **Processing:** ~$0.01 per PDF (30 seconds on RTX A5000)
- **Idle:** $0.00 (automatic shutdown)
- **Monthly:** $1-2 for 100 PDFs

## Files

- `Dockerfile` - Container with DeepSeek OCR model
- `handler.py` - RunPod serverless handler function

## Model

Uses [deepseek-ai/DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR) from Hugging Face.

## License

Part of the BOFFO financial platform.
