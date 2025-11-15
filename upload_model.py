#!/usr/bin/env python3
"""
Script to upload stamp_model.pt to Hugging Face Space.
Run this after the Space is created to upload the model file.
"""
from huggingface_hub import HfApi, login
from pathlib import Path

# Login (will prompt for token if not already logged in)
# Or set HF_TOKEN environment variable
login()

api = HfApi()
model_path = Path("stamp_detector/stamp_model.pt")

if not model_path.exists():
    print(f"Error: {model_path} not found!")
    exit(1)

print(f"Uploading {model_path} to bekzhanK1/armeta_hackaton...")
api.upload_file(
    path_or_fileobj=str(model_path),
    path_in_repo="stamp_detector/stamp_model.pt",
    repo_id="bekzhanK1/armeta_hackaton",
    repo_type="space"
)
print("✓ Model uploaded successfully!")

