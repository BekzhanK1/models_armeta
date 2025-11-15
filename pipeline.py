#!/usr/bin/env python3
"""
Unified Pipeline for Document Processing
Runs QR code detection, signature detection, and stamp detection in sequence
and combines all results into a single JSON file.
"""

import sys
import json
import argparse
import cv2
import numpy as np
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List

# Try to import PyMuPDF for PDF processing
try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("Warning: PyMuPDF not installed. PDF support disabled.")
    print("Install with: pip install PyMuPDF")

# Add subdirectories to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import detection functions
from qr.qr_extraction import process_image_no_save as process_qr
from signature.inference import detect_signatures
from stamp_detector.detect import detect_stamps_no_save


def pdf_to_images(pdf_path: str, dpi: int = 200) -> List[np.ndarray]:
    """
    Convert PDF pages to images.
    
    Args:
        pdf_path: Path to PDF file
        dpi: Resolution for conversion (default: 200)
        
    Returns:
        List of images as numpy arrays (BGR format for OpenCV)
    """
    if not PDF_SUPPORT:
        raise ImportError("PyMuPDF is required for PDF processing. Install with: pip install PyMuPDF")
    
    doc = fitz.open(pdf_path)
    images = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        # Convert to image with specified DPI
        mat = fitz.Matrix(dpi / 72, dpi / 72)  # 72 is default DPI
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to numpy array
        img_data = pix.tobytes("ppm")
        # Use cv2 to decode PPM
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is not None:
            images.append(img)
    
    doc.close()
    return images


def process_pdf_pipeline(
    pdf_path: str,
    output_dir: str = "pipeline_outputs",
    stamp_model_path: str = "stamp_detector/stamp_model.pt",
    stamp_conf: float = 0.25,
    dpi: int = 200,
    save_intermediate: bool = False
) -> Dict[str, Any]:
    """
    Process a PDF file by converting each page to an image and running the pipeline.
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Directory for output files
        stamp_model_path: Path to stamp model
        stamp_conf: Confidence threshold for stamp detection
        dpi: DPI for PDF to image conversion
        save_intermediate: Whether to save intermediate results
        
    Returns:
        Combined results dictionary for all pages
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    if not PDF_SUPPORT:
        raise ImportError("PyMuPDF is required for PDF processing. Install with: pip install PyMuPDF")
    
    print(f"\n{'='*70}")
    print(f"Processing PDF: {pdf_path.name}")
    print(f"{'='*70}\n")
    
    # Convert PDF to images
    print(f"📄 Converting PDF pages to images (DPI: {dpi})...")
    try:
        page_images = pdf_to_images(str(pdf_path), dpi=dpi)
        print(f"✓ Converted {len(page_images)} page(s) to images\n")
    except Exception as e:
        raise RuntimeError(f"Failed to convert PDF to images: {e}")
    
    # Process each page
    all_pages = []
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        for page_num, img in enumerate(page_images, 1):
            print(f"\n{'='*70}")
            print(f"Processing Page {page_num}/{len(page_images)}")
            print(f"{'='*70}\n")
            
            # Save temporary image for processing
            temp_img_path = temp_dir / f"page_{page_num}.jpg"
            cv2.imwrite(str(temp_img_path), img)
            
            # Process the page
            try:
                page_result = process_image_pipeline(
                    str(temp_img_path),
                    output_dir=output_dir,
                    stamp_model_path=stamp_model_path,
                    stamp_conf=stamp_conf,
                    save_intermediate=save_intermediate
                )
                
                # Add page number to result
                page_result["page_number"] = page_num
                page_result["image"] = f"{pdf_path.stem}_page_{page_num}.jpg"
                all_pages.append(page_result)
                
            except Exception as e:
                print(f"✗ Error processing page {page_num}: {str(e)}")
                all_pages.append({
                    "page_number": page_num,
                    "image": f"{pdf_path.stem}_page_{page_num}.jpg",
                    "error": str(e)
                })
    finally:
        # Clean up temporary directory
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Create combined summary
    summary = {
        "total_pages": len(all_pages),
        "total_qr_codes": sum(p.get("summary", {}).get("qr_codes", 0) for p in all_pages),
        "total_signatures": sum(p.get("summary", {}).get("signatures", 0) for p in all_pages),
        "total_stamps": sum(p.get("summary", {}).get("stamps", 0) for p in all_pages),
        "total_detections": sum(p.get("summary", {}).get("total", 0) for p in all_pages)
    }
    
    result = {
        "pdf": pdf_path.name,
        "pdf_path": str(pdf_path),
        "summary": summary,
        "pages": all_pages
    }
    
    print(f"\n{'='*70}")
    print("PDF PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Total Pages:   {summary['total_pages']}")
    print(f"QR Codes:      {summary['total_qr_codes']}")
    print(f"Signatures:    {summary['total_signatures']}")
    print(f"Stamps:        {summary['total_stamps']}")
    print(f"Total:         {summary['total_detections']}")
    print(f"{'='*70}\n")
    
    return result


def process_image_pipeline(
    image_path: str,
    output_dir: str = "pipeline_outputs",
    qr_model_path: Optional[str] = None,
    signature_model_path: Optional[str] = None,
    stamp_model_path: str = "stamp_detector/stamp_model.pt",
    stamp_conf: float = 0.25,
    save_intermediate: bool = False
) -> Dict[str, Any]:
    """
    Process a single image through all three detection models.
    
    Args:
        image_path: Path to input image
        output_dir: Directory for output files
        qr_model_path: Path to QR model (not used, kept for compatibility)
        signature_model_path: Path to signature model (optional)
        stamp_model_path: Path to stamp model
        stamp_conf: Confidence threshold for stamp detection
        save_intermediate: Whether to save intermediate results
        
    Returns:
        Combined results dictionary
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    print(f"\n{'='*70}")
    print(f"Processing: {image_path.name}")
    print(f"{'='*70}\n")
    
    # Get image dimensions once (will be used to consolidate)
    img_sample = cv2.imread(str(image_path))
    if img_sample is None:
        raise ValueError(f"Could not read image: {image_path}")
    img_height, img_width = img_sample.shape[:2]
    
    # Initialize result structure with consolidated image info
    result = {
        "image": image_path.name,
        "image_dimensions": {
            "width": img_width,
            "height": img_height
        },
        "qr_codes": [],
        "signatures": [],
        "stamps": []
    }
    
    # Step 1: QR Code Detection
    print("🔷 Step 1/3: QR Code Detection")
    print("-" * 70)
    try:
        qr_result = process_qr(str(image_path))
        
        if qr_result and qr_result.get("qr_codes", {}).get("items"):
            result["qr_codes"] = qr_result["qr_codes"]["items"]
            print(f"✓ Found {len(result['qr_codes'])} QR code(s)")
        else:
            print("✓ No QR codes detected")
    except Exception as e:
        print(f"✗ Error in QR detection: {str(e)}")
        result["qr_error"] = str(e)
    
    # Step 2: Signature Detection
    print(f"\n🔷 Step 2/3: Signature Detection")
    print("-" * 70)
    try:
        sig_result = detect_signatures(
            str(image_path),
            model=None,  # Will auto-load
            output_dir=None,  # Don't save
            signatures_dir=None,  # Don't save
            save_crops=False  # Don't save crops
        )
        
        if sig_result and sig_result.get("signatures"):
            # Clean up signature items (remove cropped_path if present, keep only essential data)
            cleaned_signatures = []
            for sig in sig_result["signatures"]:
                cleaned_sig = {
                    "id": sig.get("signature_id"),
                    "confidence": sig.get("confidence"),
                    "bbox": sig.get("bbox")
                }
                cleaned_signatures.append(cleaned_sig)
            result["signatures"] = cleaned_signatures
            print(f"✓ Found {len(result['signatures'])} signature(s)")
        else:
            print("✓ No signatures detected")
    except Exception as e:
        print(f"✗ Error in signature detection: {str(e)}")
        result["signature_error"] = str(e)
    
    # Step 3: Stamp Detection
    print(f"\n🔷 Step 3/3: Stamp Detection")
    print("-" * 70)
    try:
        if not Path(stamp_model_path).exists():
            raise FileNotFoundError(f"Stamp model not found: {stamp_model_path}")
        
        stamp_result = detect_stamps_no_save(
            str(image_path),
            model_path=stamp_model_path,
            conf=stamp_conf
        )
        
        if stamp_result and stamp_result.get("detections"):
            # Clean up stamp items (keep only essential data, remove normalized bbox)
            cleaned_stamps = []
            for stamp in stamp_result["detections"]:
                cleaned_stamp = {
                    "confidence": stamp.get("confidence"),
                    "bbox": stamp.get("bbox")
                }
                cleaned_stamps.append(cleaned_stamp)
            result["stamps"] = cleaned_stamps
            print(f"✓ Found {len(result['stamps'])} stamp(s)")
        else:
            print("✓ No stamps detected")
    except Exception as e:
        print(f"✗ Error in stamp detection: {str(e)}")
        result["stamp_error"] = str(e)
    
    # Create summary
    result["summary"] = {
        "qr_codes": len(result.get("qr_codes", [])),
        "signatures": len(result.get("signatures", [])),
        "stamps": len(result.get("stamps", [])),
        "total": len(result.get("qr_codes", [])) + len(result.get("signatures", [])) + len(result.get("stamps", []))
    }
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"QR Codes:      {result['summary']['qr_codes']}")
    print(f"Signatures:    {result['summary']['signatures']}")
    print(f"Stamps:        {result['summary']['stamps']}")
    print(f"Total:         {result['summary']['total']}")
    print(f"{'='*70}\n")
    
    return result


def process_folder_pipeline(
    input_folder: str,
    output_dir: str = "pipeline_outputs",
    stamp_model_path: str = "stamp_detector/stamp_model.pt",
    stamp_conf: float = 0.25,
    save_intermediate: bool = False
) -> Dict[str, Any]:
    """
    Process all images in a folder through the pipeline.
    
    Args:
        input_folder: Folder containing input images
        output_dir: Directory for output files
        stamp_model_path: Path to stamp model
        stamp_conf: Confidence threshold for stamp detection
        save_intermediate: Whether to save intermediate results
        
    Returns:
        Combined results for all images
    """
    input_folder = Path(input_folder)
    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    
    # Supported image formats
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    image_files = [f for f in input_folder.iterdir()
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No image files found in '{input_folder}'")
        return {"images": [], "summary": {}}
    
    print(f"\n{'='*70}")
    print(f"Found {len(image_files)} image(s) to process")
    print(f"{'='*70}\n")
    
    all_results = []
    for i, image_file in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}]")
        try:
            result = process_image_pipeline(
                str(image_file),
                output_dir=output_dir,
                stamp_model_path=stamp_model_path,
                stamp_conf=stamp_conf,
                save_intermediate=save_intermediate
            )
            all_results.append(result)
        except Exception as e:
            print(f"✗ Error processing {image_file.name}: {str(e)}")
            all_results.append({
                "image": image_file.name,
                "image_path": str(image_file),
                "error": str(e)
            })
    
    # Create summary
    summary = {
        "total_images": len(all_results),
        "total_qr_codes": sum(r.get("summary", {}).get("qr_codes", 0) for r in all_results),
        "total_signatures": sum(r.get("summary", {}).get("signatures", 0) for r in all_results),
        "total_stamps": sum(r.get("summary", {}).get("stamps", 0) for r in all_results),
        "total_detections": sum(r.get("summary", {}).get("total", 0) for r in all_results)
    }
    
    final_result = {
        "summary": summary,
        "images": all_results
    }
    
    # Save combined JSON
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    json_path = output_dir / "pipeline_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"Processed:     {summary['total_images']} image(s)")
    print(f"QR Codes:      {summary['total_qr_codes']}")
    print(f"Signatures:    {summary['total_signatures']}")
    print(f"Stamps:        {summary['total_stamps']}")
    print(f"Total:         {summary['total_detections']}")
    print(f"\nResults saved to: {json_path}")
    print(f"{'='*70}\n")
    
    return final_result


def main():
    parser = argparse.ArgumentParser(
        description="Unified pipeline for QR code, signature, and stamp detection"
    )
    parser.add_argument(
        "input",
        help="Input image file, PDF file, or folder containing images"
    )
    parser.add_argument(
        "--output",
        default="pipeline_outputs",
        help="Output directory (default: pipeline_outputs)"
    )
    parser.add_argument(
        "--stamp-model",
        default="stamp_detector/stamp_model.pt",
        help="Path to stamp model (default: stamp_detector/stamp_model.pt)"
    )
    parser.add_argument(
        "--stamp-conf",
        type=float,
        default=0.25,
        help="Confidence threshold for stamp detection (default: 0.25)"
    )
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Save intermediate results from each detection step"
    )
    
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for PDF to image conversion (default: 200)"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Check if it's a PDF
        if input_path.suffix.lower() == '.pdf':
            if not PDF_SUPPORT:
                print("Error: PyMuPDF is required for PDF processing.")
                print("Install with: pip install PyMuPDF")
                sys.exit(1)
            
            # Process PDF
            result = process_pdf_pipeline(
                str(input_path),
                output_dir=args.output,
                stamp_model_path=args.stamp_model,
                stamp_conf=args.stamp_conf,
                dpi=args.dpi,
                save_intermediate=args.save_intermediate
            )
            
            # Save JSON
            output_dir = Path(args.output)
            output_dir.mkdir(exist_ok=True)
            json_path = output_dir / f"{input_path.stem}_pipeline_result.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"Results saved to: {json_path}")
            
        else:
            # Process single image
            result = process_image_pipeline(
                str(input_path),
                output_dir=args.output,
                stamp_model_path=args.stamp_model,
                stamp_conf=args.stamp_conf,
                save_intermediate=args.save_intermediate
            )
            
            # Save JSON
            output_dir = Path(args.output)
            output_dir.mkdir(exist_ok=True)
            json_path = output_dir / f"{input_path.stem}_pipeline_result.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"Results saved to: {json_path}")
        
    elif input_path.is_dir():
        # Process folder
        process_folder_pipeline(
            str(input_path),
            output_dir=args.output,
            stamp_model_path=args.stamp_model,
            stamp_conf=args.stamp_conf,
            save_intermediate=args.save_intermediate
        )
    else:
        print(f"Error: '{args.input}' is not a valid file or directory")
        sys.exit(1)


if __name__ == "__main__":
    main()

