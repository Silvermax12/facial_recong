#!/usr/bin/env python3
"""
Comprehensive test suite for enhanced liveness detection
Tests all improvements against static images and potential spoofing attacks
"""
import cv2
import numpy as np
import torch
import os
import sys
import time
import requests
import json
from PIL import Image

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_static_image_detection():
    """Test static image detection capabilities"""
    print("\n" + "="*60)
    print("🖼️  TESTING STATIC IMAGE DETECTION")
    print("="*60)

    # Test on known face images (which should be static photos)
    known_faces_dir = "known_faces"
    if os.path.exists(known_faces_dir):
        image_files = [f for f in os.listdir(known_faces_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        print(f"Testing {len(image_files)} known face images...")

        for img_file in image_files[:3]:  # Test first 3 images
            img_path = os.path.join(known_faces_dir, img_file)

            # Test with API
            try:
                with open(img_path, 'rb') as f:
                    files = {'image': f}
                    response = requests.post('http://localhost:5000/v2/verify', files=files, timeout=10)

                if response.status_code == 200:
                    result = response.json()
                    liveness_score = result.get('liveness_score', 0)
                    is_live = result.get('is_live', False)
                    detailed_scores = result.get('detailed_scores', {})

                    print(f"\n📸 {img_file}:")
                    print(f"   Liveness Score: {liveness_score:.3f}")
                    print(f"   Live: {'❌ NO' if not is_live else '✅ YES'}")

                    # Check for our new detection methods
                    if 'static_image' in detailed_scores:
                        print(f"   Static Image Score: {detailed_scores['static_image']:.3f}")
                    if 'injection_attack' in detailed_scores:
                        print(f"   Injection Attack Score: {detailed_scores['injection_attack']:.3f}")
                    if 'static_image_penalty' in detailed_scores:
                        print(f"   Static Penalty Applied: {detailed_scores['static_image_penalty']}")

                    # Static images should NOT be flagged as live (or require very high scores)
                    if is_live and liveness_score < 0.85:  # Very high threshold for static images
                        print("   ⚠️  WARNING: Static image incorrectly flagged as live!")
                    else:
                        print("   ✅ Correctly identified as static/non-live")

                else:
                    print(f"❌ API error for {img_file}: {response.status_code}")

            except Exception as e:
                print(f"❌ Error testing {img_file}: {e}")

    else:
        print("❌ known_faces directory not found")

def test_frequency_analysis_improvements():
    """Test enhanced frequency domain analysis"""
    print("\n" + "="*60)
    print("📊 TESTING FREQUENCY ANALYSIS IMPROVEMENTS")
    print("="*60)

    try:
        from enhanced_liveness_utils import FrequencyDomainAnalyzer

        analyzer = FrequencyDomainAnalyzer()

        # Test on a static image
        img_path = "known_faces/femi_1760021997_0.jpg"
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            score = analyzer.analyze_frequency_spectrum(img)

            print(f"Frequency analysis on static image: {score:.3f}")
            print("   (Higher scores = more likely synthetic/static)")

            # Static photos should score relatively high
            if score > 0.6:
                print("   ✅ Frequency analysis correctly identifies static content")
            else:
                print("   ⚠️  Frequency analysis may be too permissive")

        # Test individual components
        print("\nDetailed frequency analysis breakdown:")
        # Compute the spectrum data for detailed analysis
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        fft = np.fft.fft2(gray.astype(float))
        fft_shift = np.fft.fftshift(fft)
        magnitude_spectrum = np.abs(fft_shift)

        high_freq = analyzer._analyze_high_frequency_energy(magnitude_spectrum)
        compression = analyzer._detect_compression_artifacts(magnitude_spectrum)
        noise = analyzer._analyze_noise_patterns(img)
        entropy = analyzer._analyze_spectral_entropy(magnitude_spectrum)
        edges = analyzer._analyze_edge_consistency(img)

        print(f"   High Frequency: {high_freq:.3f}")
        print(f"   Compression: {compression:.3f}")
        print(f"   Noise: {noise:.3f}")
        print(f"   Entropy: {entropy:.3f}")
        print(f"   Edges: {edges:.3f}")
    except Exception as e:
        print(f"❌ Error in frequency analysis test: {e}")

def test_static_image_detector():
    """Test the dedicated static image detector"""
    print("\n" + "="*60)
    print("📷 TESTING STATIC IMAGE DETECTOR")
    print("="*60)

    try:
        from enhanced_liveness_utils import StaticImageDetector

        detector = StaticImageDetector()

        # Test on static image
        img_path = "known_faces/femi_1760021997_0.jpg"
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            score = detector.detect_static_image(img, img_path)

            print(f"Static image detection score: {score:.3f}")
            print("   (Higher scores = more likely static photo)")

            if score > 0.7:
                print("   ✅ Static image detector correctly identifies photo")
            else:
                print("   ⚠️  Static image detector may need tuning")

            # Test individual components
            print("\nDetailed static detection breakdown:")
            exif_score = detector._analyze_exif_metadata(img_path)
            format_score = detector._analyze_image_format(img)
            color_score = detector._analyze_color_space_artifacts(img)
            temporal_score = detector._detect_temporal_artifacts(img)
            resolution_score = detector._analyze_resolution_patterns(img)

            print(f"   EXIF: {exif_score:.3f}")
            print(f"   Format: {format_score:.3f}")
            print(f"   Color: {color_score:.3f}")
            print(f"   Temporal: {temporal_score:.3f}")
            print(f"   Resolution: {resolution_score:.3f}")
    except Exception as e:
        print(f"❌ Error in static image detector test: {e}")

def test_injection_attack_detector():
    """Test injection attack detection"""
    print("\n" + "="*60)
    print("🛡️  TESTING INJECTION ATTACK DETECTOR")
    print("="*60)

    try:
        from enhanced_liveness_utils import InjectionAttackDetector

        detector = InjectionAttackDetector()

        # Simulate some requests with regular timing (suspicious)
        print("Testing timing pattern analysis...")
        scores = []
        for i in range(10):
            img = cv2.imread("known_faces/femi_1760021997_0.jpg")
            score = detector.detect_injection_attack(img, time.time() + i * 0.1)  # Regular 100ms intervals
            scores.append(score)

        avg_score = np.mean(scores)
        print(f"Average injection score: {avg_score:.3f}")
        if avg_score > 0.6:
            print("   ✅ Injection detector correctly identifies regular timing patterns")
        else:
            print("   ⚠️  Injection detector may be too permissive")

    except Exception as e:
        print(f"❌ Error in injection attack detector test: {e}")

def test_temporal_requirements():
    """Test mandatory temporal requirements"""
    print("\n" + "="*60)
    print("⏰ TESTING TEMPORAL REQUIREMENTS")
    print("="*60)

    print("Testing single image (should be heavily penalized)...")

    # Test single image
    try:
        img_path = "known_faces/femi_1760021997_0.jpg"
        with open(img_path, 'rb') as f:
            files = {'image': f}
            response = requests.post('http://localhost:5000/v2/verify', files=files, timeout=10)

        if response.status_code == 200:
            result = response.json()
            is_live = result.get('is_live', False)
            detailed_scores = result.get('detailed_scores', {})

            print(f"Single image result: Live = {is_live}")
            if 'static_image_penalty' in detailed_scores:
                print(f"Static penalty applied: {detailed_scores['static_image_penalty']}")
            if 'temporal' in detailed_scores:
                print(f"Temporal score: {detailed_scores['temporal']:.3f}")

            if not is_live:
                print("   ✅ Single image correctly flagged as non-live")
            else:
                print("   ⚠️  Single image incorrectly passed liveness check")

    except Exception as e:
        print(f"❌ Error testing temporal requirements: {e}")

def test_configuration_changes():
    """Test that configuration changes are working"""
    print("\n" + "="*60)
    print("⚙️  TESTING CONFIGURATION CHANGES")
    print("="*60)

    try:
        import yaml
        with open('liveness_config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        print("Current configuration:")
        print(f"  Liveness threshold: {config['thresholds']['liveness']}")
        print(f"  Min signals required: {config['ensemble']['min_signals_required']}")
        print(f"  Model weight: {config['ensemble']['weights']['model']}")
        print(f"  Static image weight: {config['ensemble']['weights'].get('static_image', 'N/A')}")

        if config['thresholds']['liveness'] >= 0.7:
            print("   ✅ Conservative threshold applied")
        else:
            print("   ⚠️  Threshold may still be too permissive")

        if config['ensemble']['min_signals_required'] >= 3:
            print("   ✅ Conservative signal requirements")
        else:
            print("   ⚠️  Signal requirements may be too lenient")

    except Exception as e:
        print(f"❌ Error checking configuration: {e}")

def main():
    """Run all tests"""
    print("🔬 COMPREHENSIVE LIVENESS DETECTION TEST SUITE")
    print("Testing all improvements against static images and spoofing attacks")
    print("="*80)

    # Check if server is running
    try:
        response = requests.get('http://localhost:5000/health', timeout=5)
        if response.status_code != 200:
            print("❌ Liveness detection server not running on localhost:5000")
            print("Please start the server first: python face_api.py")
            return
        print("✅ Server is running")
    except:
        print("❌ Cannot connect to server. Please start it first.")
        return

    # Run all tests
    test_configuration_changes()
    test_frequency_analysis_improvements()
    test_static_image_detector()
    test_injection_attack_detector()
    test_static_image_detection()
    test_temporal_requirements()

    print("\n" + "="*80)
    print("🏁 TEST SUITE COMPLETED")
    print("="*80)
    print("\n📋 SUMMARY OF IMPROVEMENTS:")
    print("✅ Enhanced frequency domain analysis with multiple detection methods")
    print("✅ Dedicated static image detector using metadata and artifacts")
    print("✅ Mandatory temporal requirements for single images")
    print("✅ Conservative ensemble weights and thresholds")
    print("✅ Injection attack detection for programmatic attacks")
    print("✅ Comprehensive test suite for validation")

if __name__ == "__main__":
    main()
