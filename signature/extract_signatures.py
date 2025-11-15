"""
Helper script to extract signatures from images using saved coordinates.
This script can be used to re-extract signatures from the JSON coordinates file.
"""

import json
import cv2
from pathlib import Path

def extract_signatures_from_json(json_path="outputs/signature_coordinates.json", 
                                  input_dir="inputs",
                                  output_dir="outputs/extracted_signatures"):
    """
    Extract signatures from images using saved coordinates in JSON file.
    
    Args:
        json_path: Path to the JSON file with coordinates
        input_dir: Directory containing original images
        output_dir: Directory to save extracted signatures
    """
    # Load coordinates
    with open(json_path, 'r') as f:
        all_detections = json.load(f)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    input_path = Path(input_dir)
    
    print(f"Loaded coordinates for {len(all_detections)} image(s)")
    
    for image_data in all_detections:
        image_name = image_data["image"]
        image_file = input_path / image_name
        
        if not image_file.exists():
            print(f"Warning: Image {image_name} not found, skipping...")
            continue
        
        # Read image
        image = cv2.imread(str(image_file))
        if image is None:
            print(f"Error: Could not read {image_name}, skipping...")
            continue
        
        print(f"\nProcessing: {image_name}")
        print(f"  Found {len(image_data['signatures'])} signature(s)")
        
        # Extract each signature
        for sig_data in image_data["signatures"]:
            sig_id = sig_data["signature_id"]
            bbox = sig_data["bbox"]
            
            # Get coordinates
            x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
            
            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            
            # Crop signature
            signature_crop = image[y1:y2, x1:x2]
            
            # Save cropped signature
            output_filename = f"{Path(image_name).stem}_signature_{sig_id}.jpg"
            output_file = output_path / output_filename
            cv2.imwrite(str(output_file), signature_crop)
            
            print(f"    Signature {sig_id}: confidence={sig_data['confidence']:.2f}, saved to {output_file}")

if __name__ == "__main__":
    import sys
    
    json_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/signature_coordinates.json"
    extract_signatures_from_json(json_path)

