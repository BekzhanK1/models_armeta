# YOLOv8 Signature Detector

This repository implements signature detection using the YOLOv8s model from [tech4humans/yolov8s-signature-detector](https://huggingface.co/tech4humans/yolov8s-signature-detector).

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

### Authentication

The model repository is gated and requires Hugging Face authentication. You need to:

1. **Login via CLI** (recommended):
   ```bash
   huggingface-cli login
   ```
   Enter your Hugging Face token when prompted. Get your token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

2. **Or set environment variable**:
   ```bash
   export HF_TOKEN=your_token_here
   ```

3. **Or manually download the model**:
   ```bash
   huggingface-cli download tech4humans/yolov8s-signature-detector yolov8s.pt
   ```
   Then place `yolov8s.pt` in the project root directory.

## Usage

### Python Script

Process all images in the `inputs/` directory:

```bash
python inference.py
```

The script will:
1. Check for a local `yolov8s.pt` file first
2. If not found, download the model from Hugging Face (requires authentication)
3. Process all images in the `inputs/` directory
4. Save annotated images with detected signatures to the `outputs/` directory
5. **Save signature coordinates to `outputs/signature_coordinates.json`**
6. **Crop and save individual signatures to `outputs/signatures/` directory**

### CLI (Alternative)

You can also use the Ultralytics CLI:

```bash
huggingface-cli download tech4humans/yolov8s-signature-detector yolov8s.pt
yolo predict model=yolov8s.pt source=inputs/
```

## Model Formats

The model is available in multiple formats:
- `yolov8s.pt` (PyTorch format) - used by default
- `yolov8s.onnx` (ONNX format) - for ONNX Runtime
- `yolov8s.engine` (TensorRT format) - for TensorRT inference

## Output

The script generates several outputs:

1. **Annotated images**: Images with bounding boxes around detected signatures saved to `outputs/` with the prefix `detected_`
2. **Signature coordinates JSON**: All detection coordinates saved to `outputs/signature_coordinates.json` with the following structure:
   ```json
   [
     {
       "image": "image1.jpg",
       "image_width": 1920,
       "image_height": 1080,
       "signatures": [
         {
           "signature_id": 1,
           "confidence": 0.95,
           "bbox": {
             "x1": 100.5,
             "y1": 200.3,
             "x2": 300.7,
             "y2": 400.9,
             "width": 200.2,
             "height": 200.6
           },
           "class_id": 0,
           "cropped_path": "outputs/signatures/image1_signature_1.jpg"
         }
       ]
     }
   ]
   ```
   
   The `image_width` and `image_height` fields allow the frontend to properly scale coordinates when displaying images at different sizes. Coordinates are in pixels relative to the original image dimensions.
3. **Cropped signatures**: Individual signature images saved to `outputs/signatures/` directory

## Extracting Signatures from Coordinates

If you need to re-extract signatures using the saved coordinates, use the helper script:

```bash
python extract_signatures.py
```

Or specify a custom JSON file:

```bash
python extract_signatures.py outputs/signature_coordinates.json
```

This is useful if you want to extract signatures again without running inference, or if you need to adjust the extraction parameters.

