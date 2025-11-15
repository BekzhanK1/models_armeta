"""Extract QR codes from images and save labeled images and JSON data."""
# ----------------------------------------------
# --- Author         : Ahmet Ozlu
# --- Mail           : ahmetozlu93@gmail.com
# --- Date           : 17th September 2018
# --- Modified       : QR code extraction only
# ----------------------------------------------

import cv2
import numpy as np
import json
import os
from pathlib import Path


def detect_qr_codes(img_original):
    """
    Detect QR codes in an image using multiple preprocessing approaches.

    Parameters:
    -----------
    img_original : numpy.ndarray
        Original BGR image

    Returns:
    --------
    list
        List of QR code dictionaries with 'x', 'y', 'width', 'height', 'data', 'points'
    """
    qr_detector = cv2.QRCodeDetector()
    qr_codes = []
    seen_qr_boxes = set()

    def add_qr_code(qr_points, info, seen_set):
        """Helper function to add QR code if not already detected"""
        if qr_points is None or len(qr_points) == 0:
            return False

        qr_points = qr_points.astype(int)
        x_coords = qr_points[:, 0]
        y_coords = qr_points[:, 1]
        x_min, x_max = int(x_coords.min()), int(x_coords.max())
        y_min, y_max = int(y_coords.min()), int(y_coords.max())

        # Check if we've already detected this QR code (within 10 pixels tolerance)
        box_key = (x_min // 10, y_min // 10, x_max // 10, y_max // 10)
        if box_key in seen_set:
            return False

        seen_set.add(box_key)
        qr_codes.append({
            'x': x_min,
            'y': y_min,
            'width': x_max - x_min,
            'height': y_max - y_min,
            'data': info if info else '',
            'points': qr_points.tolist()
        })
        return True

    # Try multiple preprocessing approaches for better QR code detection
    test_images = [("original", img_original)]
    gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    test_images.append(("grayscale", gray))

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    test_images.append(("clahe", gray_clahe))

    # Add thresholded versions
    _, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    _, thresh2 = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    test_images.append(("binary", thresh1))
    test_images.append(("otsu", thresh2))

    # Add inverted versions (QR codes might be white on black)
    test_images.append(("inverted", cv2.bitwise_not(gray)))
    test_images.append(("inverted_clahe", cv2.bitwise_not(gray_clahe)))

    # Try detection on each preprocessed image
    for img_name, test_img in test_images:
        if len(qr_codes) > 0:
            print(f"  QR code detected using: {img_name}")
            break  # Stop if we found QR codes

        # Ensure image is in correct format (3-channel for color, 1-channel for grayscale)
        if len(test_img.shape) == 2:
            # Grayscale - convert to 3-channel for detection
            test_img_3ch = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)
        else:
            test_img_3ch = test_img

        # Try detectAndDecodeMulti first (for multiple QR codes)
        try:
            retval, decoded_info, points, straight_qrcode = qr_detector.detectAndDecodeMulti(
                test_img_3ch)

            if retval and points is not None:
                # Handle both single and multiple QR codes
                if isinstance(decoded_info, str):
                    decoded_info = [decoded_info]
                    points = [points]

                for info, qr_points in zip(decoded_info, points):
                    if add_qr_code(qr_points, info, seen_qr_boxes):
                        print(f"  QR code detected using: {img_name} (multi)")
        except Exception as e:
            pass

        # Try single QR code detection as fallback
        if len(qr_codes) == 0:
            try:
                retval, decoded_info, points, straight_qrcode = qr_detector.detectAndDecode(
                    test_img_3ch)
                if retval and points is not None and len(points) > 0:
                    if add_qr_code(points, decoded_info, seen_qr_boxes):
                        print(f"  QR code detected using: {img_name} (single)")
            except Exception as e:
                pass

    return qr_codes


def process_image_no_save(input_path):
    """
    Process a single image and detect QR codes without saving images or JSON files.

    Parameters:
    -----------
    input_path : str
        Path to input image

    Returns:
    --------
    dict
        Dictionary with detection results (no files saved)
    """
    # Read the input image
    img_original = cv2.imread(input_path)
    if img_original is None:
        print(f"Error: Could not read image {input_path}")
        return None

    # Detect QR codes
    qr_codes = detect_qr_codes(img_original)

    # Prepare QR codes for JSON
    qr_codes_json = []
    for i, qr in enumerate(qr_codes):
        qr_json = {
            "id": i + 1,
            "x": qr['x'],
            "y": qr['y'],
            "width": qr['width'],
            "height": qr['height'],
            "data": qr['data']
        }
        # Optionally include corner points if needed
        if 'points' in qr and len(qr['points']) > 0:
            qr_json['corner_points'] = qr['points']
        qr_codes_json.append(qr_json)

    # Create output JSON structure
    output_json = {
        "image": Path(input_path).name,
        "image_dimensions": {
            "width": img_original.shape[1],
            "height": img_original.shape[0]
        },
        "qr_codes": {
            "count": len(qr_codes_json),
            "items": qr_codes_json
        }
    }

    return output_json


def process_image(input_path, output_folder='labelled', json_folder='outputs'):
    """
    Process a single image and detect QR codes.

    Parameters:
    -----------
    input_path : str
        Path to input image
    output_folder : str
        Folder to save labeled images
    json_folder : str
        Folder to save JSON files

    Returns:
    --------
    dict
        Dictionary with detection results
    """
    # Get filename without extension
    filename = Path(input_path).stem
    file_ext = Path(input_path).suffix

    print(f"\n{'='*60}")
    print(f"Processing: {Path(input_path).name}")
    print(f"{'='*60}")

    # Read the input image
    img_original = cv2.imread(input_path)
    if img_original is None:
        print(f"Error: Could not read image {input_path}")
        return None

    # Detect QR codes
    qr_codes = detect_qr_codes(img_original)

    print(f"Found {len(qr_codes)} QR code(s)")

    # Create labeled image
    labeled_img = img_original.copy()

    # Draw QR codes in blue color
    for i, qr in enumerate(qr_codes):
        # Draw bounding box
        cv2.rectangle(labeled_img, (qr['x'], qr['y']),
                      (qr['x'] + qr['width'], qr['y'] + qr['height']),
                      (255, 0, 0), 2)  # Blue color (BGR format)

        # Draw QR code points/polygon
        if len(qr['points']) >= 4:
            pts = np.array(qr['points'], np.int32)
            cv2.polylines(labeled_img, [pts], True, (255, 0, 0), 2)

        # Add label
        cv2.putText(labeled_img, f"QR {i+1}", (qr['x'], qr['y'] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Add QR data text (if not too long)
        if qr['data'] and len(qr['data']) < 50:
            cv2.putText(labeled_img, qr['data'][:30],
                        (qr['x'], qr['y'] + qr['height'] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Create output folders
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(json_folder, exist_ok=True)

    # Save labeled image
    output_image_path = os.path.join(
        output_folder, f'qr_labelled_{filename}{file_ext}')
    cv2.imwrite(output_image_path, labeled_img)

    # Prepare QR codes for JSON
    qr_codes_json = []
    for i, qr in enumerate(qr_codes):
        qr_json = {
            "id": i + 1,
            "x": qr['x'],
            "y": qr['y'],
            "width": qr['width'],
            "height": qr['height'],
            "data": qr['data']
        }
        # Optionally include corner points if needed
        if 'points' in qr and len(qr['points']) > 0:
            qr_json['corner_points'] = qr['points']
        qr_codes_json.append(qr_json)

    # Create output JSON
    output_json = {
        "image": Path(input_path).name,
        "image_dimensions": {
            "width": img_original.shape[1],
            "height": img_original.shape[0]
        },
        "qr_codes": {
            "count": len(qr_codes_json),
            "items": qr_codes_json
        }
    }

    # Save JSON
    output_json_path = os.path.join(
        json_folder, f'qr_detection_{filename}.json')
    with open(output_json_path, 'w') as f:
        json.dump(output_json, f, indent=2)

    # Print summary
    print(f"✓ Found {len(qr_codes_json)} QR code(s)")
    print(f"✓ Labeled image saved: {output_image_path}")
    print(f"✓ Detection data saved: {output_json_path}")

    return output_json


def process_folder(input_folder='inputs', output_folder='labelled', json_folder='outputs'):
    """
    Process all images in the input folder.

    Parameters:
    -----------
    input_folder : str
        Folder containing input images
    output_folder : str
        Folder to save labeled images
    json_folder : str
        Folder to save JSON files
    """
    # Create output folders
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(json_folder, exist_ok=True)

    # Supported image formats
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

    # Get all image files
    input_path = Path(input_folder)
    if not input_path.exists():
        print(f"Error: Input folder '{input_folder}' does not exist!")
        return

    image_files = [f for f in input_path.iterdir()
                   if f.is_file() and f.suffix.lower() in image_extensions]

    if not image_files:
        print(f"No image files found in '{input_folder}'")
        return

    print(f"\n{'='*60}")
    print(f"Found {len(image_files)} image(s) to process")
    print(f"{'='*60}\n")

    # Process each image
    all_results = []
    for i, image_file in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {image_file.name}")
        try:
            result = process_image(
                str(image_file),
                output_folder=output_folder,
                json_folder=json_folder
            )
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"✗ Error processing {image_file.name}: {str(e)}")
            continue

    # Save summary JSON with all results
    if all_results:
        summary_path = os.path.join(json_folder, 'qr_detection_summary.json')
        summary = {
            "total_images": len(all_results),
            "total_qr_codes": sum(r['qr_codes']['count'] for r in all_results),
            "images": all_results
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*60}")
        print(f"PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"✓ Processed {len(all_results)} image(s)")
        print(f"✓ Total QR codes detected: {summary['total_qr_codes']}")
        print(f"✓ Summary saved: {summary_path}")
        print(f"✓ Labeled images saved in: {output_folder}/")
        print(f"✓ JSON files saved in: {json_folder}/")


if __name__ == "__main__":
    # Process all images in the 'inputs' folder
    process_folder(
        input_folder='inputs',
        output_folder='labelled',
        json_folder='outputs'
    )
