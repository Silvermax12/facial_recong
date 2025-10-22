#!/usr/bin/env python3
"""
Direct test of enhanced liveness detection (bypassing Flask server)
This simulates what happens when Flutter sends an image to the API
"""
import cv2
import torch
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from face_auth_utils import compute_liveness_enhanced, load_enhanced_model
from enhanced_liveness_utils import StaticImageDetector, InjectionAttackDetector, FrequencyDomainAnalyzer
from face_auth_utils import load_known_faces

def test_enhanced_liveness():
    """Test the enhanced liveness detection directly"""
    print("🧪 TESTING ENHANCED LIVENESS DETECTION")
    print("=" * 50)

    # Load model and known faces
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        known_encodings, known_names = load_known_faces("known_faces")
        model, model_type = load_enhanced_model('ensemble', device)
        print("✅ Model and known faces loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model/faces: {e}")
        return

    # Initialize detectors
    static_detector = StaticImageDetector()
    injection_detector = InjectionAttackDetector()
    freq_analyzer = FrequencyDomainAnalyzer()

    # Test with a static image (should be flagged as non-live)
    test_image = "known_faces/femi_1760021370_0.jpg"
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        return

    print(f"\n📸 Testing with static image: {test_image}")

    # Read and process image (simulating what Flask does)
    img = cv2.imread(test_image)
    if img is None:
        print("❌ Failed to read image")
        return

    # Convert to RGB for face detection
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Face detection (simplified - in real Flask this is done)
    import face_recognition
    locs = face_recognition.face_locations(rgb)
    if not locs:
        print("❌ No faces detected in image")
        return

    print(f"✅ Found {len(locs)} face(s)")

    # Crop face (simulating what Flask does)
    top, right, bottom, left = locs[0]
    h, w = img.shape[:2]
    top = max(0, top); right = min(w, right); bottom = min(h, bottom); left = max(0, left)
    crop = img[top:bottom, left:right]

    print(f"Crop shape: {crop.shape}")

    # Test enhanced liveness detection
    try:
        score, is_live, detailed_scores = compute_liveness_enhanced(
            face_bgr=crop,
            model=model,
            device=device,
            model_type=model_type,
            static_detector=static_detector
        )

        print("\n🎯 LIVENESS DETECTION RESULTS:")
        print(f"  Score: {score:.3f}")
        print(f"  Is Live: {is_live}")
        print(f"  Threshold: 0.70")

        print("\n📊 DETAILED SCORES:")
        for key, value in detailed_scores.items():
            print(f"  {key}: {value:.3f}")

        # Analyze results
        print("\n🔍 ANALYSIS:")
        if score >= 0.70:
            print("⚠️  WARNING: Static image passed liveness check!")
            print("   This suggests the enhanced detection needs tuning.")
        else:
            print("✅ SUCCESS: Static image correctly flagged as non-live!")

        # Check individual components
        static_score = detailed_scores.get('static_image', 0)
        if static_score > 0.5:
            print(f"⚠️  Static detector score ({static_score:.3f}) indicates photo detected")
        else:
            print(f"ℹ️  Static detector score ({static_score:.3f}) - may need tuning")

        freq_score = detailed_scores.get('frequency', 0)
        if freq_score > 0.5:
            print(f"✅ Frequency analysis ({freq_score:.3f}) correctly identifies static content")
        else:
            print(f"⚠️  Frequency analysis ({freq_score:.3f}) may be too permissive")

    except Exception as e:
        print(f"❌ Liveness detection failed: {e}")
        import traceback
        traceback.print_exc()

def test_component_breakdown():
    """Test individual components in detail"""
    print("\n" + "=" * 50)
    print("🔧 TESTING COMPONENT BREAKDOWN")
    print("=" * 50)

    static_detector = StaticImageDetector()
    freq_analyzer = FrequencyDomainAnalyzer()

    img = cv2.imread("known_faces/femi_1760021370_0.jpg")
    if img is None:
        return

    print("STATIC IMAGE DETECTOR BREAKDOWN:")
    static_score = static_detector.detect_static_image(img, "known_faces/femi_1760021370_0.jpg")
    print(f"Overall static score: {static_score:.3f}")

    # Test individual static detector methods
    exif = static_detector._analyze_exif_metadata("known_faces/femi_1760021370_0.jpg")
    format_score = static_detector._analyze_image_format(img)
    color = static_detector._analyze_color_space_artifacts(img)
    temporal = static_detector._detect_temporal_artifacts(img)
    resolution = static_detector._analyze_resolution_patterns(img)

    print(f"  EXIF metadata: {exif:.3f}")
    print(f"  Image format: {format_score:.3f}")
    print(f"  Color space: {color:.3f}")
    print(f"  Temporal artifacts: {temporal:.3f}")
    print(f"  Resolution patterns: {resolution:.3f}")

    print("\nFREQUENCY ANALYSIS BREAKDOWN:")
    freq_overall = freq_analyzer.analyze_frequency_spectrum(img)
    print(f"Overall frequency score: {freq_overall:.3f}")

    # Test individual frequency methods
    magnitude = freq_analyzer.analyze_frequency_spectrum(img)  # This returns score, not magnitude
    # We can't easily test the individual methods without modifying the code

if __name__ == "__main__":
    test_enhanced_liveness()
    test_component_breakdown()

    print("\n" + "=" * 50)
    print("🏁 TESTING COMPLETE")
    print("=" * 50)
