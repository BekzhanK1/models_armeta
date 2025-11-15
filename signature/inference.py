import cv2
import os
import sys
import json
import supervision as sv
from huggingface_hub import hf_hub_download, login
from ultralytics import YOLO
from pathlib import Path


def detect_signatures(image_path, model=None, output_dir=None, signatures_dir=None, save_crops=True):
    """
    Detect signatures in a single image.

    Args:
        image_path: Path to the input image
        model: YOLO model instance (if None, will load/create one)
        output_dir: Directory for output files (optional)
        signatures_dir: Directory for cropped signatures (optional)
        save_crops: Whether to save cropped signature images

    Returns:
        dict: Detection results with structure:
            {
                "image": image_filename,
                "image_width": int,
                "image_height": int,
                "signatures": [...]
            }
    """
    # Load model if not provided
    if model is None:
        local_model_path = Path("yolov8s.pt")
        if local_model_path.exists():
            model_path = str(local_model_path)
        else:
            try:
                # Get HF token from environment (for gated models)
                hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
                model_path = hf_hub_download(
                    repo_id="tech4humans/yolov8s-signature-detector",
                    filename="yolov8s.pt",
                    token=hf_token  # Pass token for gated repos
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load signature model: {e}")
        model = YOLO(model_path)

    # Set up paths (only if we need to save crops)
    image_file = Path(image_path)
    if save_crops:
        if output_dir is None:
            output_dir = Path("outputs")
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        if signatures_dir is None:
            signatures_dir = output_dir / "signatures"
        else:
            signatures_dir = Path(signatures_dir)
        signatures_dir.mkdir(exist_ok=True)
    else:
        # Dummy paths when not saving
        output_dir = None
        signatures_dir = None

    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Get image dimensions
    image_height, image_width = image.shape[:2]

    # Run inference
    results = model(str(image_path))
    detections = sv.Detections.from_ultralytics(results[0])

    # Store detection data
    image_detections = {
        "image": image_file.name,
        "image_width": int(image_width),
        "image_height": int(image_height),
        "signatures": []
    }

    # Process detections
    if len(detections) > 0:
        for i, (xyxy, confidence, class_id) in enumerate(zip(
            detections.xyxy, detections.confidence, detections.class_id
        )):
            x1, y1, x2, y2 = xyxy

            # Store detection data
            detection_data = {
                "signature_id": i + 1,
                "confidence": float(confidence),
                "bbox": {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "width": float(x2 - x1),
                    "height": float(y2 - y1)
                },
                "class_id": int(class_id)
            }

            # Crop and save individual signature if requested
            if save_crops and signatures_dir is not None:
                x1_int, y1_int, x2_int, y2_int = int(
                    x1), int(y1), int(x2), int(y2)
                x1_int = max(0, x1_int)
                y1_int = max(0, y1_int)
                x2_int = min(image.shape[1], x2_int)
                y2_int = min(image.shape[0], y2_int)

                signature_crop = image[y1_int:y2_int, x1_int:x2_int]
                signature_filename = f"{image_file.stem}_signature_{i+1}.jpg"
                signature_path = signatures_dir / signature_filename
                cv2.imwrite(str(signature_path), signature_crop)
                detection_data["cropped_path"] = str(signature_path)

            image_detections["signatures"].append(detection_data)

    return image_detections


def main():
    # Check if model file exists locally first
    local_model_path = Path("yolov8s.pt")

    if local_model_path.exists():
        print(f"Using local model file: {local_model_path}", flush=True)
        model_path = str(local_model_path)
    else:
        # Try to download model from Hugging Face
        print("Downloading model from Hugging Face...", flush=True)
        try:
            model_path = hf_hub_download(
                repo_id="tech4humans/yolov8s-signature-detector",
                filename="yolov8s.pt"
            )
        except Exception as e:
            if "401" in str(e) or "GatedRepoError" in str(type(e).__name__) or "Unauthorized" in str(e):
                print("\n" + "="*70)
                print("ERROR: Authentication required to access this model.")
                print("="*70)
                print(
                    "\nThis repository is gated and requires Hugging Face authentication.")
                print("\nTo authenticate, run one of the following:")
                print("  1. huggingface-cli login")
                print("  2. Or set your token: export HF_TOKEN=your_token_here")
                print("\nAfter authentication, run this script again.")
                print("="*70)
                sys.exit(1)
            else:
                print(f"\nError downloading model: {e}")
                print("\nYou can also download the model manually:")
                print(
                    "  huggingface-cli download tech4humans/yolov8s-signature-detector yolov8s.pt")
                print("\nOr place yolov8s.pt in the current directory.")
                sys.exit(1)

    # Load the model
    print("Loading model...")
    model = YOLO(model_path)

    # Set up paths
    input_dir = Path("inputs")
    output_dir = Path("outputs")
    signatures_dir = output_dir / "signatures"  # Directory for cropped signatures
    output_dir.mkdir(exist_ok=True)
    signatures_dir.mkdir(exist_ok=True)

    # Store all detections for JSON export
    all_detections = []

    # Get all image files from inputs directory
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = [f for f in input_dir.iterdir()
                   if f.suffix.lower() in image_extensions]

    if not image_files:
        print(f"No images found in {input_dir}/")
        return

    print(f"Found {len(image_files)} image(s) to process")

    # Process each image
    box_annotator = sv.BoxAnnotator()

    for image_file in image_files:
        print(f"\nProcessing: {image_file.name}")

        try:
            # Use the reusable function
            image_detections = detect_signatures(
                str(image_file),
                model=model,
                output_dir=output_dir,
                signatures_dir=signatures_dir,
                save_crops=True
            )

            # Read image for annotation
            image = cv2.imread(str(image_file))
            results = model(str(image_file))
            detections = sv.Detections.from_ultralytics(results[0])

            if len(detections) > 0:
                print(f"  Found {len(detections)} signature(s)")
                for i, sig in enumerate(image_detections["signatures"]):
                    bbox = sig["bbox"]
                    print(
                        f"    Signature {i+1}: confidence={sig['confidence']:.2f}, bbox=[{bbox['x1']:.1f}, {bbox['y1']:.1f}, {bbox['x2']:.1f}, {bbox['y2']:.1f}]")
                    if "cropped_path" in sig:
                        print(
                            f"      Saved cropped signature to: {sig['cropped_path']}")
            else:
                print("  No signatures detected")

            all_detections.append(image_detections)

            # Annotate image with bounding boxes
            annotated_image = box_annotator.annotate(
                scene=image.copy(),
                detections=detections
            )

            # Save annotated image
            output_path = output_dir / f"detected_{image_file.name}"
            cv2.imwrite(str(output_path), annotated_image)
            print(f"  Saved annotated image to: {output_path}")
        except Exception as e:
            print(f"  Error processing {image_file.name}: {str(e)}")
            continue

    # Save all coordinates to JSON file
    json_path = output_dir / "signature_coordinates.json"
    with open(json_path, 'w') as f:
        json.dump(all_detections, f, indent=2)
    print(f"\n{'='*70}")
    print(f"Saved all signature coordinates to: {json_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
