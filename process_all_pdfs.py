#!/usr/bin/env python3
"""
Process all PDF files in the documents folder and generate a single JSON file
in the required format with annotations for signatures and stamps.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

from pipeline import process_pdf_pipeline

def convert_to_annotation_format(
    pipeline_result: Dict[str, Any],
    annotation_id_start: int = 1
) -> Tuple[Dict[str, Any], int]:
    """
    Convert pipeline result to the required annotation format.
    
    Args:
        pipeline_result: Result from process_pdf_pipeline
        annotation_id_start: Starting annotation ID number
        
    Returns:
        Tuple of (formatted_result, next_annotation_id)
    """
    pdf_filename = pipeline_result["pdf"]
    pages_data = pipeline_result["pages"]
    
    result = {}
    annotation_counter = annotation_id_start
    
    for page_data in pages_data:
        page_num = page_data.get("page_number", 1)
        page_key = f"page_{page_num}"
        
        # Get image dimensions
        img_dims = page_data.get("image_dimensions", {})
        width = img_dims.get("width", 0)
        height = img_dims.get("height", 0)
        
        # Collect all annotations (signatures and stamps only, no QR codes)
        annotations = []
        
        # Process signatures
        signatures = page_data.get("signatures", [])
        for sig in signatures:
            bbox = sig.get("bbox", {})
            if bbox:
                x1 = bbox.get("x1", 0)
                y1 = bbox.get("y1", 0)
                width_bbox = bbox.get("width", 0)
                height_bbox = bbox.get("height", 0)
                
                # Calculate area
                area = width_bbox * height_bbox
                
                annotation = {
                    f"annotation_{annotation_counter}": {
                        "category": "signature",
                        "bbox": {
                            "x": float(x1),
                            "y": float(y1),
                            "width": float(width_bbox),
                            "height": float(height_bbox)
                        },
                        "area": float(area)
                    }
                }
                annotations.append(annotation)
                annotation_counter += 1
        
        # Process stamps
        stamps = page_data.get("stamps", [])
        for stamp in stamps:
            bbox = stamp.get("bbox", {})
            if bbox:
                x1 = bbox.get("x1", 0)
                y1 = bbox.get("y1", 0)
                width_bbox = bbox.get("width", 0)
                height_bbox = bbox.get("height", 0)
                
                # Calculate area
                area = width_bbox * height_bbox
                
                annotation = {
                    f"annotation_{annotation_counter}": {
                        "category": "stamp",
                        "bbox": {
                            "x": float(x1),
                            "y": float(y1),
                            "width": float(width_bbox),
                            "height": float(height_bbox)
                        },
                        "area": float(area)
                    }
                }
                annotations.append(annotation)
                annotation_counter += 1
        
        # Only include pages that have annotations
        if annotations:
            result[page_key] = {
                "annotations": annotations,
                "page_size": {
                    "width": int(width),
                    "height": int(height)
                }
            }
    
    return result, annotation_counter


def process_all_pdfs(
    documents_dir: str = "documents",
    output_file: str = "all_annotations.json",
    stamp_model_path: str = "stamp_detector/stamp_model.pt",
    stamp_conf: float = 0.25,
    dpi: int = 200
) -> None:
    """
    Process all PDF files in the documents folder and generate a single JSON file.
    
    Args:
        documents_dir: Directory containing PDF files
        output_file: Output JSON file path
        stamp_model_path: Path to stamp model
        stamp_conf: Confidence threshold for stamp detection
        dpi: DPI for PDF to image conversion
    """
    documents_path = Path(documents_dir)
    
    if not documents_path.exists():
        print(f"Error: Documents directory '{documents_dir}' not found!")
        sys.exit(1)
    
    # Find all PDF files
    pdf_files = sorted(list(documents_path.glob("*.pdf")))
    
    if not pdf_files:
        print(f"No PDF files found in '{documents_dir}' directory!")
        sys.exit(1)
    
    print(f"Found {len(pdf_files)} PDF file(s) to process\n")
    print("=" * 70)
    
    # Final result dictionary
    final_result = {}
    annotation_counter = 1
    
    # Process each PDF
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_file.name}")
        print("-" * 70)
        
        try:
            # Process PDF using pipeline
            pipeline_result = process_pdf_pipeline(
                pdf_path=str(pdf_file),
                output_dir="pipeline_outputs",
                stamp_model_path=stamp_model_path,
                stamp_conf=stamp_conf,
                dpi=dpi,
                save_intermediate=False
            )
            
            # Convert to annotation format
            pdf_annotations, annotation_counter = convert_to_annotation_format(
                pipeline_result,
                annotation_id_start=annotation_counter
            )
            
            # Only add to result if there are annotations
            if pdf_annotations:
                final_result[pdf_file.name] = pdf_annotations
                print(f"✓ Processed: {len(pdf_annotations)} page(s) with annotations")
            else:
                print(f"⚠ No annotations found in {pdf_file.name}")
            
        except Exception as e:
            print(f"✗ Error processing {pdf_file.name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save to JSON file
    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    print(f"Total PDFs processed: {len(pdf_files)}")
    print(f"PDFs with annotations: {len(final_result)}")
    print(f"Total annotations: {annotation_counter - 1}")
    print(f"Output saved to: {output_path.absolute()}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process all PDF files in documents folder and generate annotations JSON"
    )
    parser.add_argument(
        "--documents-dir",
        default="documents",
        help="Directory containing PDF files (default: documents)"
    )
    parser.add_argument(
        "--output",
        default="all_annotations.json",
        help="Output JSON file path (default: all_annotations.json)"
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
        "--dpi",
        type=int,
        default=200,
        help="DPI for PDF to image conversion (default: 200)"
    )
    
    args = parser.parse_args()
    
    process_all_pdfs(
        documents_dir=args.documents_dir,
        output_file=args.output,
        stamp_model_path=args.stamp_model,
        stamp_conf=args.stamp_conf,
        dpi=args.dpi
    )

