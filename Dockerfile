# Dockerfile for Hugging Face Spaces
FROM python:3.10-slim

# Install system dependencies for OpenCV and PyMuPDF
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Create user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy requirements and install Python dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user --upgrade -r requirements.txt

# Copy all application code
COPY --chown=user . .

# Create directories for models if needed
RUN mkdir -p stamp_detector signature qr

# Note: stamp_model.pt should be uploaded via HF Hub web interface or upload_model.py script
# The model will be available at stamp_detector/stamp_model.pt after upload

# Expose port (HF Spaces uses port 7860)
EXPOSE 7860

# Run FastAPI on port 7860
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]

