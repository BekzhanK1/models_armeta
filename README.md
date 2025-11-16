---
title: Document Processing Pipeline API
emoji: 📄
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# Document Processing Pipeline API

A production-ready FastAPI service for automated detection and extraction of QR codes, signatures, and stamps from PDF documents. The pipeline processes multi-page PDFs sequentially through three specialized detection models and returns consolidated JSON results.

## Overview

This API provides a unified interface for document analysis, combining multiple computer vision models to extract structured information from PDF documents. It supports concurrent processing of multiple documents and can handle both file uploads and remote PDF URLs.

## Detection Models

### 1. QR Code Detection
- **Method**: OpenCV `QRCodeDetector` (native implementation)
- **Library**: OpenCV Python (`cv2`)
- **Approach**: Multi-preprocessing pipeline with adaptive thresholding
- **Features**:
  - Detects multiple QR codes per page
  - Decodes QR code data automatically
  - Uses CLAHE (Contrast Limited Adaptive Histogram Equalization) for enhanced detection
  - Tests multiple preprocessing approaches (grayscale, binary, Otsu thresholding, inverted)
- **Output**: Bounding box coordinates, decoded data, corner points

### 2. Signature Detection
- **Model**: YOLOv8s (Small variant)
- **Source**: `tech4humans/yolov8s-signature-detector` (Hugging Face Hub)
- **Framework**: Ultralytics YOLO
- **Architecture**: YOLOv8s - optimized for speed and accuracy balance
- **Access**: Gated model (requires Hugging Face authentication token)
- **Features**:
  - Real-time signature detection
  - Confidence scoring for each detection
  - Bounding box coordinates with normalized values
- **Output**: Signature locations, confidence scores, bounding boxes

### 3. Stamp Detection
- **Model**: Custom YOLOv8 model
- **Framework**: Ultralytics YOLO
- **Model File**: `stamp_model.pt` (custom trained)
- **Default Confidence Threshold**: 0.25
- **Features**:
  - Detects stamps and seals on documents
  - Configurable confidence threshold
  - Supports custom model paths
- **Output**: Stamp locations, confidence scores, bounding boxes

## API Endpoints

### `POST /process-pdf`
Upload and process a PDF file directly.

**Parameters**:
- `file` (multipart/form-data): PDF file to process
- `dpi` (int, default: 200): Resolution for PDF to image conversion
- `stamp_conf` (float, default: 0.25): Confidence threshold for stamp detection

**Example**:
```bash
curl -X POST "https://bekzhanK1-armeta-hackaton.hf.space/process-pdf" \
  -F "file=@document.pdf" \
  -F "dpi=200" \
  -F "stamp_conf=0.25"
```

### `POST /process-pdf-advanced`
Process PDF with advanced options including custom model paths.

**Parameters**:
- `file` (multipart/form-data): PDF file to process
- `dpi` (int, default: 200): Resolution for PDF to image conversion
- `stamp_conf` (float, default: 0.25): Confidence threshold for stamp detection
- `stamp_model` (str, optional): Path to custom stamp model

### `POST /process-pdf-from-url`
Process PDF from a remote URL (S3, HTTP, or HTTPS).

**Parameters**:
- `pdf_url` (query string): URL to PDF file
- `dpi` (int, default: 200): Resolution for PDF to image conversion
- `stamp_conf` (float, default: 0.25): Confidence threshold for stamp detection
- `stamp_model` (str, optional): Path to custom stamp model

**Example**:
```bash
curl -X POST "https://bekzhanK1-armeta-hackaton.hf.space/process-pdf-from-url?pdf_url=https://example.com/document.pdf&dpi=200"
```

### `GET /health`
Health check endpoint.

### `GET /docs`
Interactive API documentation (Swagger UI).

## Batch Processing

For processing multiple PDF files locally, use the `process_all_pdfs.py` script to batch process all PDFs in a folder and generate a single JSON file with annotations.

### Basic Usage

Process all PDFs in the `documents` folder:
```bash
python process_all_pdfs.py
```

This will:
- Process all PDF files in the `documents/` folder
- Detect signatures and stamps on each page
- Generate a single JSON file: `all_annotations.json`
- Only include pages that have annotations

### Advanced Options

```bash
python process_all_pdfs.py \
  --documents-dir documents \
  --output results.json \
  --dpi 300 \
  --stamp-conf 0.3
```

**Parameters**:
- `--documents-dir`: Directory containing PDF files (default: `documents`)
- `--output`: Output JSON file path (default: `all_annotations.json`)
- `--dpi`: DPI for PDF to image conversion (default: 200)
- `--stamp-conf`: Confidence threshold for stamp detection (default: 0.25)
- `--stamp-model`: Path to stamp model (default: `stamp_detector/stamp_model.pt`)

### Output Format

The script generates a JSON file with the following structure:
```json
{
  "filename.pdf": {
    "page_1": {
      "annotations": [
        {
          "annotation_1": {
            "category": "signature",
            "bbox": {
              "x": 500,
              "y": 800,
              "width": 200,
              "height": 100
            },
            "area": 20000
          }
        }
      ],
      "page_size": {
        "width": 1654,
        "height": 2339
      }
    }
  }
}
```

## Response Format

The API returns a JSON object with the following structure:

```json
{
  "pdf_file": "document.pdf",
  "total_pages": 1,
  "summary": {
    "total_pages": 1,
    "total_qr_codes": 2,
    "total_signatures": 1,
    "total_stamps": 1,
    "total_detections": 4
  },
  "pages": [
    {
      "page_number": 1,
      "image": "document_page_1.jpg",
      "image_dimensions": {
        "width": 1654,
        "height": 2339
      },
      "qr_codes": [
        {
          "id": 1,
          "x": 100,
          "y": 200,
          "width": 150,
          "height": 150,
          "data": "https://example.com"
        }
      ],
      "signatures": [
        {
          "id": 1,
          "confidence": 0.95,
          "bbox": {
            "x1": 500,
            "y1": 800,
            "x2": 700,
            "y2": 900
          }
        }
      ],
      "stamps": [
        {
          "id": 1,
          "confidence": 0.87,
          "bbox": {
            "x1": 1200,
            "y1": 100,
            "x2": 1400,
            "y2": 300
          }
        }
      ]
    }
  ]
}
```

## Configuration

### DPI Settings
The DPI parameter controls the resolution when converting PDF pages to images:
- **150 DPI**: Fast processing, suitable for documents with large elements
- **200 DPI** (default): Balanced speed and accuracy
- **300 DPI**: Higher accuracy for small signatures/stamps, slower processing

**Impact on Detection**:
- **QR Codes**: Moderate impact - very low DPI may miss small QR codes
- **Signatures**: High impact - small signatures require higher DPI (200-300)
- **Stamps**: High impact - small stamps require higher DPI (200-300)

### Model Requirements

1. **Signature Model**: Automatically downloaded from Hugging Face Hub on first use
   - Requires `HF_TOKEN` environment variable for gated model access
   - Set in Space Settings → Secrets

2. **Stamp Model**: Must be uploaded to `stamp_detector/stamp_model.pt`
   - Upload via Hugging Face Space web interface or Git LFS

## Performance

- **Concurrent Processing**: Supports up to 4 parallel requests (configurable)
- **Processing Time**: Varies by document size and DPI (typically 2-10 seconds per page)
- **Memory**: Optimized for efficient model loading and image processing

## Deployment

This API is containerized using Docker and can be deployed on:
- Hugging Face Spaces (current deployment)
- Any Docker-compatible platform
- Local development with GPU support

## License

MIT License
