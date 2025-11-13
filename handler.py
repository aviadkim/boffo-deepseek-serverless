"""
RunPod Serverless Handler - Tesseract OCR for Financial PDFs
Extracts portfolio data with confidence scoring
"""

import runpod
import pytesseract
import base64
import json
import time
import re
from pdf2image import convert_from_bytes
from PIL import Image
import cv2
import numpy as np


def preprocess_image(image):
    """Enhance image quality for better OCR"""
    # Convert PIL to OpenCV
    img = np.array(image)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply thresholding to make text clearer
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Noise removal
    denoised = cv2.fastNlMeansDenoising(thresh, h=10)

    return Image.fromarray(denoised)


def extract_text_from_image(image):
    """Extract text using Tesseract OCR"""
    # Preprocess for better accuracy
    processed = preprocess_image(image)

    # Extract text with config for financial documents
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(processed, config=custom_config)

    return text


def extract_isin_pattern(text):
    """Extract ISINs using regex (2 letters + 10 alphanumeric)"""
    isin_pattern = r'\b[A-Z]{2}[A-Z0-9]{10}\b'
    isins = re.findall(isin_pattern, text)
    return list(set(isins))  # Remove duplicates


def extract_currency(text):
    """Detect currency from text"""
    currencies = ['USD', 'EUR', 'CHF', 'GBP', 'JPY']
    for curr in currencies:
        if curr in text:
            return curr
    return 'USD'  # Default


def extract_numbers(text):
    """Extract numbers from text (handles Swiss format with apostrophes)"""
    # Match numbers like: 1'234'567.89 or 1234567.89
    number_pattern = r"[\d']+(?:\.\d{2})?"
    matches = re.findall(number_pattern, text)

    numbers = []
    for match in matches:
        try:
            # Remove apostrophes and convert
            clean = match.replace("'", "")
            num = float(clean)
            if num > 100:  # Filter out small numbers (likely dates/counts)
                numbers.append(num)
        except:
            pass

    return numbers


def extract_dates(text):
    """Extract dates in various formats"""
    # Matches: 30.09.2025, 2025-09-30, etc.
    date_patterns = [
        r'\d{2}\.\d{2}\.\d{4}',
        r'\d{4}-\d{2}-\d{2}',
        r'\d{2}/\d{2}/\d{4}'
    ]

    for pattern in date_patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[0]

    return None


def parse_portfolio_summary(all_text):
    """Extract portfolio summary from OCR text"""
    summary = {
        "currency": extract_currency(all_text),
        "statement_date": extract_dates(all_text),
        "total_portfolio_value": None,
        "client_id": None
    }

    # Extract total value (largest number found)
    numbers = extract_numbers(all_text)
    if numbers:
        summary["total_portfolio_value"] = max(numbers)

    # Extract client ID if present
    client_patterns = [
        r'Client[:\s]+([A-Z0-9]+)',
        r'Account[:\s]+([A-Z0-9]+)',
        r'Portfolio[:\s]+([A-Z0-9]+)'
    ]

    for pattern in client_patterns:
        match = re.search(pattern, all_text, re.IGNORECASE)
        if match:
            summary["client_id"] = match.group(1)
            break

    return summary


def parse_holdings(all_text, isins):
    """Parse holdings from OCR text using ISINs as anchors"""
    holdings = []

    # Split text into lines
    lines = all_text.split('\n')

    for isin in isins:
        # Find lines containing this ISIN
        for i, line in enumerate(lines):
            if isin in line:
                # Extract data from this line and nearby lines
                context = ' '.join(lines[max(0, i-1):min(len(lines), i+3)])

                # Extract numbers from context
                numbers = extract_numbers(context)

                holding = {
                    "isin": isin,
                    "security_name": extract_security_name(line, isin),
                    "quantity": numbers[0] if len(numbers) > 0 else None,
                    "price": numbers[1] if len(numbers) > 1 else None,
                    "market_value": numbers[2] if len(numbers) > 2 else (numbers[1] if len(numbers) > 1 else None),
                    "currency": extract_currency(context),
                    "asset_class": classify_asset(line)
                }

                holdings.append(holding)
                break

    return holdings


def extract_security_name(line, isin):
    """Extract security name from line (text before ISIN)"""
    isin_pos = line.find(isin)
    if isin_pos > 0:
        name = line[:isin_pos].strip()
        # Clean up
        name = re.sub(r'\s+', ' ', name)
        name = name[:100]  # Max 100 chars
        return name if name else "Unknown Security"
    return "Unknown Security"


def classify_asset(line):
    """Classify asset type based on keywords"""
    line_lower = line.lower()

    if any(word in line_lower for word in ['bond', 'note', 'treasury', 'govt']):
        return 'BOND'
    elif any(word in line_lower for word in ['struct', 'product', 'certif']):
        return 'STRUCTURED_PRODUCT'
    elif any(word in line_lower for word in ['equity', 'stock', 'share']):
        return 'EQUITY'
    elif any(word in line_lower for word in ['cash', 'liquidity']):
        return 'CASH'
    else:
        return 'OTHER'


def calculate_confidence(summary, holdings, all_text):
    """
    Calculate extraction confidence score (0-100%)

    Scoring:
    - ISINs found: 40 points (0-10 ISINs = 0-40 points)
    - Summary completeness: 30 points
    - Holdings data quality: 20 points
    - Text quality: 10 points
    """
    score = 0

    # 1. ISINs found (40 points max)
    isin_count = len(holdings)
    score += min(40, isin_count * 4)

    # 2. Summary completeness (30 points)
    if summary.get('total_portfolio_value'):
        score += 10
    if summary.get('statement_date'):
        score += 10
    if summary.get('client_id'):
        score += 10

    # 3. Holdings data quality (20 points)
    if holdings:
        holdings_with_value = sum(1 for h in holdings if h.get('market_value'))
        score += min(20, (holdings_with_value / len(holdings)) * 20)

    # 4. Text quality (10 points)
    # Check for garbled text or too many special characters
    text_quality = len(re.findall(r'[A-Za-z0-9]', all_text)) / max(1, len(all_text))
    score += text_quality * 10

    return min(100, int(score))


def handler(job):
    """
    RunPod serverless handler for Tesseract OCR extraction

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
        "confidence_score": 85,
        "requires_review": false (if < 90%),
        "summary": {...},
        "holdings": [...],
        "processing_time_seconds": 12.5
    }
    """
    try:
        job_input = job['input']

        # Get PDF data
        pdf_base64 = job_input.get('pdf_base64')
        filename = job_input.get('filename', 'unknown.pdf')

        if not pdf_base64:
            return {"error": "No PDF data provided"}

        print(f"✓ Processing: {filename}")
        start_time = time.time()

        # Decode PDF
        pdf_bytes = base64.b64decode(pdf_base64)

        # Convert to images
        print("✓ Converting PDF to images (300 DPI)...")
        images = convert_from_bytes(pdf_bytes, dpi=300)
        print(f"✓ Converted {len(images)} pages")

        # Extract text from all pages
        all_text = ""
        print("✓ Extracting text with Tesseract OCR...")

        for i, image in enumerate(images):
            print(f"  Page {i+1}/{len(images)}...")
            text = extract_text_from_image(image)
            all_text += text + "\n\n"

        print(f"✓ Extracted {len(all_text)} characters")

        # Parse data
        print("✓ Parsing portfolio data...")

        # Extract ISINs first (these are our anchors)
        isins = extract_isin_pattern(all_text)
        print(f"✓ Found {len(isins)} ISINs")

        # Extract summary
        summary = parse_portfolio_summary(all_text)

        # Extract holdings
        holdings = parse_holdings(all_text, isins)

        # Calculate confidence
        confidence = calculate_confidence(summary, holdings, all_text)
        print(f"✓ Confidence score: {confidence}%")

        # Prepare results
        results = {
            "status": "success",
            "pdf_filename": filename,
            "pages_processed": len(images),
            "confidence_score": confidence / 100.0,  # 0.0 to 1.0
            "requires_review": confidence < 90,
            "summary": summary,
            "holdings": holdings,
            "asset_allocation": {
                "bonds": sum(1 for h in holdings if h.get('asset_class') == 'BOND'),
                "structured_products": sum(1 for h in holdings if h.get('asset_class') == 'STRUCTURED_PRODUCT'),
                "equities": sum(1 for h in holdings if h.get('asset_class') == 'EQUITY'),
                "cash": sum(1 for h in holdings if h.get('asset_class') == 'CASH'),
                "other": sum(1 for h in holdings if h.get('asset_class') == 'OTHER')
            },
            "extraction_method": "tesseract_ocr",
            "processing_time_seconds": time.time() - start_time
        }

        # If low confidence, include PDF for Claude Code review
        if confidence < 90:
            print(f"⚠ Low confidence ({confidence}%) - flagging for review")
            results["review_data"] = {
                "pdf_base64": pdf_base64,
                "extracted_text_sample": all_text[:1000],  # First 1000 chars
                "message": "Extraction confidence below 90% - PDF saved for Claude Code assistance"
            }

        print(f"✓ Extraction complete in {results['processing_time_seconds']:.2f}s")
        print(f"  Summary: {'Yes' if summary.get('total_portfolio_value') else 'No'}")
        print(f"  Holdings: {len(holdings)}")
        print(f"  Requires Review: {'Yes' if results['requires_review'] else 'No'}")

        return results

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

        return {
            "status": "error",
            "error": str(e),
            "requires_review": True
        }


# Start the serverless worker
if __name__ == "__main__":
    print("🚀 Starting RunPod serverless worker (Tesseract OCR)...")
    print("✓ Tesseract version:", pytesseract.get_tesseract_version())
    runpod.serverless.start({"handler": handler})
