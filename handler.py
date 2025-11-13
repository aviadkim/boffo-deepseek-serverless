"""
RunPod Serverless Handler for DeepSeek OCR
Processes PDF extractions on-demand with automatic scaling
"""

import runpod
import torch
import base64
import json
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from pdf2image import convert_from_bytes
from io import BytesIO

# Load model once at startup (cached in container)
print("Loading DeepSeek OCR model...")
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-OCR",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True
)

tokenizer = AutoTokenizer.from_pretrained(
    "deepseek-ai/DeepSeek-OCR",
    trust_remote_code=True
)

print(f"✅ Model loaded on {model.device}")
print(f"GPU available: {torch.cuda.is_available()}")


def generate_extraction(prompt, image):
    """Generate extraction using DeepSeek OCR"""
    try:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                images=image,
                max_new_tokens=2048,
                temperature=0.1,
                do_sample=False
            )

        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        result = result.replace(prompt, "").strip()

        # Try to parse as JSON
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {"raw_text": result}

    except Exception as e:
        return {"error": str(e)}


def extract_portfolio_summary(image):
    """Extract portfolio summary"""
    prompt = """<image>
<|grounding|>This is a bank portfolio statement.

Extract the portfolio summary in JSON format:
{
  "client_id": "string (account number)",
  "client_name": "string (if visible)",
  "statement_date": "string (YYYY-MM-DD)",
  "total_portfolio_value": "number",
  "currency": "string (USD, EUR, CHF, etc.)",
  "ytd_performance_pct": "number (percentage)",
  "ytd_gain_loss": "number",
  "bank_name": "string"
}

Look for labels like: Total Assets, Portfolio Value, NAV, Performance, Return, YTD.
Be smart about currency symbols and number formats.
Output ONLY valid JSON, no other text.
"""
    return generate_extraction(prompt, image)


def extract_holdings(image):
    """Extract holdings table"""
    prompt = """<image>
<|grounding|>This is a bank portfolio statement with holdings.

Extract ALL securities in JSON format:
{
  "holdings": [
    {
      "security_name": "string",
      "isin": "string (12 chars, or null)",
      "ticker": "string (or null)",
      "asset_class": "string (Bond/Equity/Structured Product/Cash/Other)",
      "quantity": "number",
      "currency": "string",
      "market_value": "number",
      "percentage": "number (if shown)",
      "maturity_date": "string (YYYY-MM-DD, or null)"
    }
  ]
}

Look for table headers: Security, ISIN, Ticker, Quantity, Value, Price.
ISINs usually start with 2 letters followed by 10 alphanumeric characters.
Be smart about currency symbols and number formats.
Output ONLY valid JSON, no other text.
"""
    return generate_extraction(prompt, image)


def extract_asset_allocation(image):
    """Extract asset allocation"""
    prompt = """<image>
<|grounding|>This is a bank portfolio statement.

Extract asset allocation in JSON format:
{
  "asset_allocation": {
    "bonds": "number (percentage)",
    "equities": "number (percentage)",
    "structured_products": "number (percentage)",
    "cash": "number (percentage)",
    "alternatives": "number (percentage)",
    "other": "number (percentage)"
  }
}

Look for: Asset Allocation, Portfolio Breakdown, Investment Mix.
Categories: Bonds, Fixed Income, Equities, Stocks, Structured Products, Cash, Liquidity.
Output ONLY valid JSON, no other text.
"""
    return generate_extraction(prompt, image)


def handler(job):
    """
    RunPod serverless handler

    Input format:
    {
        "input": {
            "pdf_base64": "base64 encoded PDF data",
            "filename": "optional filename"
        }
    }

    Output format:
    {
        "status": "success",
        "summary": {...},
        "holdings": [...],
        "asset_allocation": {...},
        "processing_time_seconds": 28.5,
        "pages_processed": 5
    }
    """
    try:
        job_input = job['input']

        # Get PDF data
        pdf_base64 = job_input.get('pdf_base64')
        filename = job_input.get('filename', 'unknown.pdf')

        if not pdf_base64:
            return {"error": "No PDF data provided"}

        print(f"Processing: {filename}")
        start_time = time.time()

        # Decode PDF
        pdf_bytes = base64.b64decode(pdf_base64)

        # Convert to images
        print("Converting PDF to images...")
        images = convert_from_bytes(pdf_bytes, dpi=300)
        print(f"Converted {len(images)} pages")

        # Initialize results
        results = {
            "status": "success",
            "pdf_filename": filename,
            "pages_processed": len(images),
            "summary": None,
            "holdings": [],
            "asset_allocation": None,
            "confidence_score": 0.90,
            "extraction_method": "deepseek_ocr_serverless"
        }

        # Process each page
        for i, image in enumerate(images):
            print(f"Processing page {i+1}/{len(images)}")

            # Extract summary (page 1)
            if i == 0:
                print("  Extracting portfolio summary...")
                summary = extract_portfolio_summary(image)
                if summary and not summary.get("error"):
                    results["summary"] = summary
                    print("  ✅ Found summary")

            # Extract holdings (any page)
            print("  Extracting holdings...")
            holdings = extract_holdings(image)
            if holdings and "holdings" in holdings and len(holdings["holdings"]) > 0:
                results["holdings"].extend(holdings["holdings"])
                print(f"  ✅ Found {len(holdings['holdings'])} holdings")

            # Extract asset allocation (page 1)
            if i == 0 and results["asset_allocation"] is None:
                print("  Extracting asset allocation...")
                allocation = extract_asset_allocation(image)
                if allocation and "asset_allocation" in allocation:
                    results["asset_allocation"] = allocation["asset_allocation"]
                    print("  ✅ Found asset allocation")

        results["processing_time_seconds"] = time.time() - start_time

        print(f"✅ Extraction complete in {results['processing_time_seconds']:.2f}s")
        print(f"   Summary: {'Yes' if results['summary'] else 'No'}")
        print(f"   Holdings: {len(results['holdings'])}")
        print(f"   Allocation: {'Yes' if results['asset_allocation'] else 'No'}")

        return results

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }


# Start the serverless worker
if __name__ == "__main__":
    print("🚀 Starting RunPod serverless worker...")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    runpod.serverless.start({"handler": handler})
