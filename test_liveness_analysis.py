#!/usr/bin/env python3
"""
Test script to analyze why static images are being flagged as live
"""
import cv2
import numpy as np
import torch
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_liveness_utils import FrequencyDomainAnalyzer
from face_auth_utils import compute_liveness_enhanced, load_enhanced_model

def analyze_image_liveness(image_path):
    """Analyze liveness detection on a static image"""
    print(f"\n=== Analyzing: {image_path} ===")

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not load image: {image_path}")
        return

    print(f"Image shape: {img.shape}")

    # Test frequency analysis
    analyzer = FrequencyDomainAnalyzer()
    freq_score = analyzer.analyze_frequency_spectrum(img)
    print(f"Frequency score: {freq_score:.3f}")
    # Test with model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        model, model_type = load_enhanced_model('ensemble', device)
        print(f"Loaded model type: {model_type}")

        # Crop to face area (approximate)
        h, w = img.shape[:2]
        crop = img[h//4:h*3//4, w//4:w*3//4]  # Rough face crop
        print(f"Crop shape: {crop.shape}")

        final_score, detailed_scores = compute_liveness_enhanced(
            crop, model, device, model_type=model_type,
            use_frequency_analysis=True,
            use_temporal_check=False,
            temporal_checker=None,
            blink_detector=None
        )

        print(f"Final liveness score: {final_score:.3f}")
        print(f"Threshold (0.55): {'✅ PASS' if final_score >= 0.55 else '❌ FAIL'}")
        print(f"Detailed scores: {detailed_scores}")

        # Analyze why it might be passing
        if final_score >= 0.55:
            print("⚠️  WARNING: Static image flagged as LIVE!")
            if 'model' in detailed_scores:
                print(f"  Model score: {detailed_scores['model']:.3f}")
            if 'frequency' in detailed_scores:
                print(f"  Frequency score: {detailed_scores['frequency']:.3f}")

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("🔍 Liveness Detection Analysis Tool")
    print("=" * 50)

    # Test on known face images
    known_faces_dir = "known_faces"
    if os.path.exists(known_faces_dir):
        image_files = [f for f in os.listdir(known_faces_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        print(f"Found {len(image_files)} images in {known_faces_dir}")

        for img_file in image_files[:3]:  # Test first 3 images
            img_path = os.path.join(known_faces_dir, img_file)
            analyze_image_liveness(img_path)
    else:
        print(f"❌ Directory {known_faces_dir} not found")

if __name__ == "__main__":
    main()
