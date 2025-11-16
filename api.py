#!/usr/bin/env python3
"""
FastAPI application for document processing pipeline.
Accepts PDF files and returns detection results in JSON format.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
import asyncio

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import httpx

from pipeline import process_pdf_pipeline, PDF_SUPPORT

app = FastAPI(
    title="Document Processing Pipeline API",
    description="API for QR code, signature, and stamp detection in PDF documents",
    version="1.0.0"
)

# Enable CORS for all origins (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool executor for running blocking CPU/GPU operations concurrently
# This allows multiple PDFs to be processed in parallel
executor = ThreadPoolExecutor(max_workers=4)  # Adjust based on your GPU/CPU capacity


@app.on_event("startup")
async def startup_event():
    """Authenticate with Hugging Face and pre-load models if possible."""
    # Authenticate with Hugging Face if token is available
    # HF Spaces automatically provides HF_TOKEN, but we also check HUGGINGFACE_TOKEN
    hf_token = os.environ.get(
        "HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if hf_token:
        try:
            from huggingface_hub import login
            login(token=hf_token)
            print("✓ Authenticated with Hugging Face")
        except Exception as e:
            print(f"⚠ Warning: Failed to authenticate with HF: {e}")
    else:
        print("⚠ Warning: No HF_TOKEN found. Gated models may not work.")
        print("  Set HF_TOKEN in Space Settings → Secrets for gated model access.")

    # Check if stamp model exists
    stamp_model_path = Path("stamp_detector/stamp_model.pt")
    if stamp_model_path.exists():
        print("✓ Stamp model found")
    else:
        print("⚠ Warning: Stamp model not found at stamp_detector/stamp_model.pt")
        print("  Please upload stamp_model.pt to the Space.")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "Document Processing Pipeline API",
        "pdf_support": PDF_SUPPORT
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "pdf_support": PDF_SUPPORT}


@app.post("/process-pdf")
async def process_pdf(
    file: UploadFile = File(..., description="PDF file to process"),
    dpi: int = 200,
    stamp_conf: float = 0.25
):
    """
    Process a PDF file and return detection results.

    Args:
        file: PDF file to upload
        dpi: DPI for PDF to image conversion (default: 200)
        stamp_conf: Confidence threshold for stamp detection (default: 0.25)

    Returns:
        JSON response with detection results
    """
    # Check if PDF support is available
    if not PDF_SUPPORT:
        raise HTTPException(
            status_code=503,
            detail="PDF processing is not available. Please install PyMuPDF: pip install PyMuPDF"
        )

    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only PDF files are supported."
        )

    # Create temporary file for uploaded PDF
    temp_pdf = None
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
            content = await file.read()
            temp_pdf.write(content)
            temp_pdf_path = temp_pdf.name

        # Process the PDF in a thread pool to allow concurrent requests
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                executor,
                process_pdf_pipeline,
                temp_pdf_path,
                tempfile.gettempdir(),  # Use temp directory
                "stamp_detector/stamp_model.pt",
                stamp_conf,
                dpi,
                False  # save_intermediate
            )

            # Return the result as JSON
            return JSONResponse(content=result)

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error processing PDF: {str(e)}"
            )

    finally:
        # Clean up temporary file
        if temp_pdf and os.path.exists(temp_pdf_path):
            try:
                os.unlink(temp_pdf_path)
            except Exception:
                pass


@app.post("/process-pdf-advanced")
async def process_pdf_advanced(
    file: UploadFile = File(..., description="PDF file to process"),
    dpi: int = 200,
    stamp_conf: float = 0.25,
    stamp_model: Optional[str] = None
):
    """
    Process a PDF file with advanced options.

    Args:
        file: PDF file to upload
        dpi: DPI for PDF to image conversion (default: 200)
        stamp_conf: Confidence threshold for stamp detection (default: 0.25)
        stamp_model: Path to custom stamp model (optional)

    Returns:
        JSON response with detection results
    """
    # Check if PDF support is available
    if not PDF_SUPPORT:
        raise HTTPException(
            status_code=503,
            detail="PDF processing is not available. Please install PyMuPDF: pip install PyMuPDF"
        )

    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only PDF files are supported."
        )

    # Use default stamp model if not provided
    stamp_model_path = stamp_model or "stamp_detector/stamp_model.pt"

    # Validate stamp model exists
    if not Path(stamp_model_path).exists():
        raise HTTPException(
            status_code=404,
            detail=f"Stamp model not found: {stamp_model_path}"
        )

    # Create temporary file for uploaded PDF
    temp_pdf = None
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
            content = await file.read()
            temp_pdf.write(content)
            temp_pdf_path = temp_pdf.name

        # Process the PDF in a thread pool to allow concurrent requests
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                executor,
                process_pdf_pipeline,
                temp_pdf_path,
                tempfile.gettempdir(),  # Use temp directory
                stamp_model_path,
                stamp_conf,
                dpi,
                False  # save_intermediate
            )

            # Return the result as JSON
            return JSONResponse(content=result)

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error processing PDF: {str(e)}"
            )

    finally:
        # Clean up temporary file
        if temp_pdf and os.path.exists(temp_pdf_path):
            try:
                os.unlink(temp_pdf_path)
            except Exception:
                pass


@app.post("/process-pdf-from-url")
async def process_pdf_from_url(
    pdf_url: str = Query(...,
                         description="URL to PDF file (S3 or HTTP/HTTPS)"),
    dpi: int = Query(200, description="DPI for PDF to image conversion"),
    stamp_conf: float = Query(
        0.25, description="Confidence threshold for stamp detection"),
    stamp_model: Optional[str] = Query(
        None, description="Path to custom stamp model")
):
    """
    Process a PDF file from a URL (S3 or HTTP/HTTPS) and return detection results.

    Args:
        pdf_url: URL to the PDF file (e.g., s3://bucket/key or https://example.com/file.pdf)
        dpi: DPI for PDF to image conversion (default: 200)
        stamp_conf: Confidence threshold for stamp detection (default: 0.25)
        stamp_model: Path to custom stamp model (optional)

    Returns:
        JSON response with detection results
    """
    # Check if PDF support is available
    if not PDF_SUPPORT:
        raise HTTPException(
            status_code=503,
            detail="PDF processing is not available. Please install PyMuPDF: pip install PyMuPDF"
        )

    # Validate URL
    parsed_url = urlparse(pdf_url)
    if not parsed_url.scheme:
        raise HTTPException(
            status_code=400,
            detail="Invalid URL format. Must include scheme (http://, https://, or s3://)"
        )

    # Use default stamp model if not provided
    stamp_model_path = stamp_model or "stamp_detector/stamp_model.pt"

    # Validate stamp model exists
    if not Path(stamp_model_path).exists():
        raise HTTPException(
            status_code=404,
            detail=f"Stamp model not found: {stamp_model_path}"
        )

    temp_pdf_path = None
    try:
        # Download PDF from URL
        print(f"Downloading PDF from: {pdf_url}")

        if parsed_url.scheme == 's3':
            # Handle S3 URLs
            # For S3, we'll use boto3 if available, otherwise try presigned URL
            try:
                import boto3
                from botocore.exceptions import ClientError

                # Parse S3 URL: s3://bucket/key
                bucket = parsed_url.netloc
                key = parsed_url.path.lstrip('/')

                # Download from S3
                s3_client = boto3.client('s3')
                temp_pdf_path = tempfile.mktemp(suffix='.pdf')

                try:
                    s3_client.download_file(bucket, key, temp_pdf_path)
                    print(f"✓ Downloaded PDF from S3: s3://{bucket}/{key}")
                except ClientError as e:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Failed to download from S3: {str(e)}"
                    )

            except ImportError:
                # If boto3 is not available, try treating S3 URL as presigned URL
                # Convert s3:// to https:// (assuming it's a presigned URL)
                if pdf_url.startswith('s3://'):
                    raise HTTPException(
                        status_code=400,
                        detail="S3 URLs require boto3. Install with: pip install boto3, or use a presigned HTTPS URL"
                    )
                # Fall through to HTTP handling
                pdf_url = pdf_url.replace('s3://', 'https://', 1)

        # Handle HTTP/HTTPS URLs (including presigned S3 URLs)
        if parsed_url.scheme in ('http', 'https') or temp_pdf_path is None:
            if temp_pdf_path is None:
                temp_pdf_path = tempfile.mktemp(suffix='.pdf')

            # 5 minute timeout
            async with httpx.AsyncClient(timeout=300.0) as client:
                try:
                    response = await client.get(pdf_url)
                    response.raise_for_status()

                    # Validate content type
                    content_type = response.headers.get(
                        'content-type', '').lower()
                    if 'pdf' not in content_type and not pdf_url.lower().endswith('.pdf'):
                        raise HTTPException(
                            status_code=400,
                            detail=f"URL does not point to a PDF file. Content-Type: {content_type}"
                        )

                    # Save to temporary file
                    with open(temp_pdf_path, 'wb') as f:
                        f.write(response.content)
                    print(f"✓ Downloaded PDF from URL: {pdf_url}")

                except httpx.HTTPStatusError as e:
                    raise HTTPException(
                        status_code=e.response.status_code,
                        detail=f"Failed to download PDF from URL: {str(e)}"
                    )
                except httpx.RequestError as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Error fetching PDF from URL: {str(e)}"
                    )

        # Process the PDF in a thread pool to allow concurrent requests
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                executor,
                process_pdf_pipeline,
                temp_pdf_path,
                tempfile.gettempdir(),
                stamp_model_path,
                stamp_conf,
                dpi,
                False  # save_intermediate
            )

            # Return the result as JSON
            return JSONResponse(content=result)

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error processing PDF: {str(e)}"
            )

    finally:
        # Clean up temporary file
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            try:
                os.unlink(temp_pdf_path)
            except Exception:
                pass


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )
