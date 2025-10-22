import cv2
import numpy as np
import time
from typing import Dict, Tuple, Optional
import threading

class ActiveFlashDetector:
    """
    Active flash-based liveness detection using screen illumination
    Analyzes light reflection patterns to distinguish real faces from photos/videos
    """

    def __init__(self, flash_intensity=0.8, analysis_duration=0.5):
        self.flash_intensity = flash_intensity
        self.analysis_duration = analysis_duration

        # Flash pattern parameters
        self.flash_patterns = ['solid', 'pulsing', 'checkerboard']
        self.min_reflection_change = 0.15  # Minimum brightness change for liveness

        # Screen flash simulation (in real implementation, this would control device screen)
        self.flash_active = False
        self.flash_thread = None

    def perform_flash_test(self, camera_index=0, face_bbox=None) -> Dict[str, float]:
        """
        Perform active flash liveness test

        Args:
            camera_index: Camera device index
            face_bbox: Face bounding box (top, right, bottom, left)

        Returns:
            Dictionary with flash analysis results
        """
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

        if not cap.isOpened():
            return {'flash_score': 0.0, 'reflection_change': 0.0, 'is_live_flash': False}

        results = {
            'flash_score': 0.0,
            'reflection_change': 0.0,
            'specular_highlights': 0.0,
            'shadow_consistency': 0.0,
            'is_live_flash': False,
            'flash_method': 'screen_simulation'
        }

        try:
            # Capture baseline frame (no flash)
            ret, baseline_frame = cap.read()
            if not ret or baseline_frame is None:
                return results

            # Extract face region if bbox provided
            if face_bbox:
                top, right, bottom, left = face_bbox
                baseline_face = baseline_frame[max(0, top):min(baseline_frame.shape[0], bottom),
                                             max(0, left):min(baseline_frame.shape[1], right)]
            else:
                baseline_face = baseline_frame

            # Analyze baseline
            baseline_brightness = self._calculate_region_brightness(baseline_face)
            baseline_texture = self._calculate_texture_variation(baseline_face)

            # Simulate flash activation
            flash_frames = []
            start_time = time.time()

            # Capture frames during "flash" period
            while time.time() - start_time < self.analysis_duration:
                ret, frame = cap.read()
                if ret and frame is not None:
                    if face_bbox:
                        top, right, bottom, left = face_bbox
                        face_region = frame[max(0, top):min(frame.shape[0], bottom),
                                          max(0, left):min(frame.shape[1], right)]
                    else:
                        face_region = frame

                    flash_frames.append(face_region)

                time.sleep(0.05)  # Small delay between captures

            if not flash_frames:
                return results

            # Analyze flash response
            flash_analysis = self._analyze_flash_response(
                baseline_face, flash_frames, baseline_brightness, baseline_texture
            )

            results.update(flash_analysis)

            # Combined flash liveness score
            reflection_score = flash_analysis['reflection_change']
            specular_score = flash_analysis['specular_highlights']
            shadow_score = flash_analysis['shadow_consistency']

            # Weighted combination
            combined_score = (
                reflection_score * 0.4 +
                specular_score * 0.3 +
                shadow_score * 0.3
            )

            results['flash_score'] = combined_score
            results['is_live_flash'] = combined_score > 0.5

        except Exception as e:
            print(f"[!] Flash detection error: {e}")
        finally:
            cap.release()

        return results

    def _analyze_flash_response(self, baseline_face, flash_frames, baseline_brightness, baseline_texture):
        """Analyze how the face responds to flash illumination"""

        flash_brightnesses = []
        flash_textures = []
        specular_highlights = []
        shadow_patterns = []

        for frame in flash_frames:
            if frame.size == 0:
                continue

            # Brightness change analysis
            brightness = self._calculate_region_brightness(frame)
            flash_brightnesses.append(brightness)

            # Texture variation (should increase with real 3D surfaces)
            texture = self._calculate_texture_variation(frame)
            flash_textures.append(texture)

            # Specular highlight detection
            specular = self._detect_specular_highlights(frame)
            specular_highlights.append(specular)

            # Shadow consistency analysis
            shadow_consistency = self._analyze_shadow_consistency(baseline_face, frame)
            shadow_patterns.append(shadow_consistency)

        if not flash_brightnesses:
            return {
                'reflection_change': 0.0,
                'specular_highlights': 0.0,
                'shadow_consistency': 0.0
            }

        # Calculate reflection change (how much brightness increases)
        avg_flash_brightness = np.mean(flash_brightnesses)
        reflection_change = min((avg_flash_brightness - baseline_brightness) / (baseline_brightness + 1e-6), 2.0)
        reflection_change = max(0, reflection_change)  # Ensure non-negative

        # Texture variation change (3D surfaces show more texture under flash)
        avg_flash_texture = np.mean(flash_textures)
        texture_change = (avg_flash_texture - baseline_texture) / (baseline_texture + 1e-6)

        # Specular highlights (glossy skin effect)
        avg_specular = np.mean(specular_highlights)

        # Shadow consistency (real faces cast natural shadows)
        avg_shadow_consistency = np.mean(shadow_patterns)

        # Normalize scores
        reflection_score = min(reflection_change / self.min_reflection_change, 1.0)
        specular_score = min(avg_specular, 1.0)
        shadow_score = avg_shadow_consistency

        return {
            'reflection_change': float(reflection_score),
            'specular_highlights': float(specular_score),
            'shadow_consistency': float(shadow_score)
        }

    def _calculate_region_brightness(self, image):
        """Calculate average brightness of a region"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        return np.mean(gray.astype(float)) / 255.0

    def _calculate_texture_variation(self, image):
        """Calculate texture variation using Laplacian variance"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Laplacian variance indicates texture sharpness
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_var = laplacian.var()

        return texture_var

    def _detect_specular_highlights(self, image):
        """Detect specular highlights (bright spots from light reflection)"""
        if len(image.shape) == 3:
            # Convert to HSV for better highlight detection
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            value_channel = hsv[:, :, 2]
        else:
            value_channel = image

        # Threshold for bright highlights
        _, highlights = cv2.threshold(value_channel, 220, 255, cv2.THRESH_BINARY)

        # Calculate highlight density
        highlight_density = np.sum(highlights > 0) / (highlights.shape[0] * highlights.shape[1])

        return highlight_density

    def _analyze_shadow_consistency(self, baseline, flash_frame):
        """Analyze shadow consistency between baseline and flash frames"""
        # Simple shadow detection based on brightness differences
        baseline_bright = self._calculate_region_brightness(baseline)
        flash_bright = self._calculate_region_brightness(flash_frame)

        # Real faces show more pronounced shadows under directional light
        # This is a simplified version - real implementation would analyze shadow patterns
        shadow_strength = abs(flash_bright - baseline_bright)

        # Normalize to 0-1 scale (higher = more consistent shadows)
        consistency_score = min(shadow_strength / 0.3, 1.0)

        return consistency_score

    def simulate_screen_flash(self, duration=0.5):
        """
        Simulate screen flash for testing (creates bright white overlay)
        In real implementation, this would control device screen brightness
        """
        def flash_worker():
            self.flash_active = True
            time.sleep(duration)
            self.flash_active = False

        if self.flash_thread and self.flash_thread.is_alive():
            return

        self.flash_thread = threading.Thread(target=flash_worker)
        self.flash_thread.daemon = True
        self.flash_thread.start()

    def create_flash_overlay(self, frame_shape):
        """
        Create a flash overlay for simulation purposes
        Returns a bright white overlay that can be blended with camera feed
        """
        height, width = frame_shape[:2]
        overlay = np.ones((height, width, 3), dtype=np.uint8) * int(255 * self.flash_intensity)
        return overlay


class ScreenFlashController:
    """
    Controller for device screen-based flash (platform-specific implementation needed)
    """

    def __init__(self):
        self.flash_supported = self._check_flash_support()

    def _check_flash_support(self):
        """Check if screen flash is supported on this platform"""
        # This would need platform-specific implementation
        # For mobile devices, check screen brightness control
        # For desktops, this might not be available
        return False  # Placeholder

    def activate_flash(self, intensity=0.8, duration=0.5):
        """Activate screen flash"""
        if not self.flash_supported:
            print("[!] Screen flash not supported on this platform")
            return False

        # Platform-specific flash activation would go here
        print(f"[+] Activating screen flash (intensity: {intensity}, duration: {duration}s)")
        return True

    def deactivate_flash(self):
        """Deactivate screen flash"""
        if not self.flash_supported:
            return

        # Platform-specific flash deactivation would go here
        print("[+] Deactivating screen flash")


def test_flash_detection():
    """Test function for active flash liveness detection"""
    print("[*] Testing active flash liveness detection...")

    # Initialize flash detector
    flash_detector = ActiveFlashDetector()

    # Try to find camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[!] No camera found for flash testing")
        return

    cap.release()

    print("[*] Starting flash detection test...")
    print("[*] Press 'f' to trigger flash test, 'q' to quit")

    # For demonstration, we'll simulate flash effects
    try:
        while True:
            # In a real implementation, you would:
            # 1. Detect face in camera feed
            # 2. Call flash_detector.perform_flash_test() when needed
            # 3. Display results

            key = cv2.waitKey(1) & 0xFF

            if key == ord('f'):
                print("[+] Running flash test...")
                # Simulate flash test (without real face detection for demo)
                results = flash_detector.perform_flash_test(camera_index=0)

                print("Flash Test Results:"                print(".2f"                print(".2f"                print(".2f"                print(".2f"                print(f"  Live: {results['is_live_flash']}")

            elif key == ord('q'):
                break

    finally:
        cv2.destroyAllWindows()

    print("[+] Flash detection test completed")


if __name__ == "__main__":
    test_flash_detection()
