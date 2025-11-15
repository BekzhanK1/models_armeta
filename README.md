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

FastAPI service for detecting QR codes, signatures, and stamps in PDF documents.

## Features

- **QR Code Detection**: Detects and decodes QR codes in documents
- **Signature Detection**: Uses YOLOv8s to detect signatures
- **Stamp Detection**: Uses YOLOv8 to detect stamps/seals
- **PDF Support**: Processes multi-page PDF documents

## API Endpoints

- `POST /process-pdf` - Upload and process PDF file
- `POST /process-pdf-from-url` - Process PDF from URL (S3 or HTTP/HTTPS)
- `GET /docs` - Interactive API documentation
- `GET /health` - Health check

Visit `/docs` for interactive API documentation.

## Usage

### Process PDF via API

```bash
curl -X POST "https://bekzhanK1-armeta-hackaton.hf.space/process-pdf" \
  -F "file=@document.pdf" \
  -F "dpi=200" \
  -F "stamp_conf=0.25"
```

### Process PDF from URL

```bash
curl -X POST "https://bekzhanK1-armeta-hackaton.hf.space/process-pdf-from-url?pdf_url=https://example.com/document.pdf"
```

## Model Requirements

- Signature model: Automatically downloaded from Hugging Face
- Stamp model: Must be uploaded to `stamp_detector/stamp_model.pt` in this repository

