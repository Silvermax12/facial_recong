import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
import time

class StereoDepthAnalyzer:
    """
    Stereo vision-based depth analysis for liveness detection
    Uses two camera feeds to create depth maps and detect 3D facial features
    """

    def __init__(self, baseline_distance=6.0, focal_length=800):
        """
        Args:
            baseline_distance: Distance between stereo cameras in cm
            focal_length: Camera focal length in pixels
        """
        self.baseline = baseline_distance / 100.0  # Convert to meters
        self.focal_length = focal_length

        # Stereo matching parameters
        self.stereo = cv2.StereoBM.create(numDisparities=16*5, blockSize=15)
        self.min_disparity = 0
        self.max_depth = 3.0  # Maximum depth in meters

        # Face region depth analysis
        self.depth_threshold = 0.15  # Minimum depth variation for live faces

    def compute_depth_map(self, left_frame: np.ndarray, right_frame: np.ndarray) -> np.ndarray:
        """
        Compute depth map from stereo image pair
        """
        try:
            # Convert to grayscale
            left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

            # Ensure images are same size
            height, width = left_gray.shape[:2]
            right_gray = cv2.resize(right_gray, (width, height))

            # Compute disparity map
            disparity = self.stereo.compute(left_gray, right_gray).astype(np.float32)

            # Convert disparity to depth (meters)
            # depth = (baseline * focal_length) / disparity
            with np.errstate(divide='ignore', invalid='ignore'):
                depth_map = (self.baseline * self.focal_length) / (disparity + 1e-6)
                depth_map = np.clip(depth_map, 0, self.max_depth)

            # Normalize depth map for visualization/analysis
            depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
            depth_normalized = depth_normalized.astype(np.uint8)

            return depth_map, depth_normalized

        except Exception as e:
            print(f"[!] Stereo depth computation error: {e}")
            return np.zeros_like(left_frame[:, :, 0], dtype=np.float32), np.zeros_like(left_frame[:, :, 0], dtype=np.uint8)

    def analyze_face_depth(self, depth_map: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Dict[str, float]:
        """
        Analyze depth characteristics within face region

        Args:
            depth_map: Depth map in meters
            face_bbox: (top, right, bottom, left) face bounding box

        Returns:
            Dictionary with depth analysis metrics
        """
        top, right, bottom, left = face_bbox

        # Extract face region from depth map
        face_depth = depth_map[max(0, top):min(depth_map.shape[0], bottom),
                              max(0, left):min(depth_map.shape[1], right)]

        if face_depth.size == 0:
            return {
                'depth_variation': 0.0,
                'face_depth_score': 0.0,
                'is_3d': False
            }

        # Calculate depth statistics
        valid_depths = face_depth[face_depth > 0]  # Filter out invalid depths

        if len(valid_depths) == 0:
            return {
                'depth_variation': 0.0,
                'face_depth_score': 0.0,
                'is_3d': False
            }

        depth_mean = np.mean(valid_depths)
        depth_std = np.std(valid_depths)
        depth_variation = depth_std / (depth_mean + 1e-6)  # Coefficient of variation

        # 3D face detection score
        # Live faces have natural depth variation due to nose, eyes, etc.
        face_depth_score = min(depth_variation / self.depth_threshold, 1.0)
        is_3d = depth_variation > self.depth_threshold

        return {
            'depth_variation': float(depth_variation),
            'face_depth_score': float(face_depth_score),
            'mean_depth': float(depth_mean),
            'depth_std': float(depth_std),
            'is_3d': bool(is_3d)
        }


class StructuredLightDepthAnalyzer:
    """
    Structured light-based depth sensing (simulated)
    Uses projected light patterns to detect surface geometry
    """

    def __init__(self):
        self.pattern_frequency = 50  # Lines per frame
        self.min_modulation_depth = 0.1

    def simulate_structured_light(self, frame: np.ndarray) -> np.ndarray:
        """
        Simulate structured light depth sensing
        In real implementation, this would use projected light patterns
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Simulate structured light pattern distortion
        height, width = gray.shape

        # Create synthetic light pattern
        x_coords = np.arange(width)
        y_coords = np.arange(height)
        X, Y = np.meshgrid(x_coords, y_coords)

        # Simulate curved surface distortion
        distortion = 0.01 * np.sin(2 * np.pi * X / self.pattern_frequency)
        distorted_pattern = np.sin(2 * np.pi * (X + distortion * Y) / self.pattern_frequency)

        # Analyze pattern modulation
        modulation = np.abs(np.sin(2 * np.pi * X / self.pattern_frequency) - distorted_pattern)
        modulation = cv2.GaussianBlur(modulation.astype(np.float32), (5, 5), 0)

        # Convert to depth-like representation
        depth_map = modulation * 2.0  # Scale to reasonable depth range

        return depth_map.astype(np.float32)

    def analyze_structured_light_depth(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Dict[str, float]:
        """Analyze depth using structured light simulation"""
        depth_map = self.simulate_structured_light(frame)

        # Analyze face region
        top, right, bottom, left = face_bbox
        face_depth = depth_map[max(0, top):min(depth_map.shape[0], bottom),
                              max(0, left):min(depth_map.shape[1], right)]

        if face_depth.size == 0:
            return {'structured_light_score': 0.0, 'surface_modulation': 0.0}

        # Calculate surface modulation
        surface_modulation = np.std(face_depth) / (np.mean(face_depth) + 1e-6)

        # Structured light score (higher = more 3D-like surface)
        structured_light_score = min(surface_modulation / self.min_modulation_depth, 1.0)

        return {
            'structured_light_score': float(structured_light_score),
            'surface_modulation': float(surface_modulation)
        }


class DepthBasedLivenessDetector:
    """
    Main depth-based liveness detection system
    Combines multiple depth sensing approaches
    """

    def __init__(self, use_stereo=True, use_structured_light=True):
        self.use_stereo = use_stereo
        self.use_structured_light = use_structured_light

        if use_stereo:
            self.stereo_analyzer = StereoDepthAnalyzer()
        if use_structured_light:
            self.light_analyzer = StructuredLightDepthAnalyzer()

        # Ensemble weights
        self.depth_weights = {
            'stereo_3d_score': 0.5,
            'structured_light_score': 0.3,
            'combined_depth_score': 0.2
        }

    def analyze_depth_liveness(self, left_frame: Optional[np.ndarray] = None,
                             right_frame: Optional[np.ndarray] = None,
                             face_bbox: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        """
        Comprehensive depth-based liveness analysis

        Args:
            left_frame: Left camera frame (for stereo)
            right_frame: Right camera frame (for stereo)
            face_bbox: Face bounding box (top, right, bottom, left)

        Returns:
            Dictionary with depth analysis results
        """

        results = {
            'depth_available': False,
            'stereo_3d_score': 0.0,
            'structured_light_score': 0.0,
            'combined_depth_score': 0.0,
            'is_live_depth': False,
            'depth_method': 'none'
        }

        if face_bbox is None:
            return results

        # Stereo vision analysis
        if self.use_stereo and left_frame is not None and right_frame is not None:
            try:
                depth_map, depth_vis = self.stereo_analyzer.compute_depth_map(left_frame, right_frame)
                stereo_results = self.stereo_analyzer.analyze_face_depth(depth_map, face_bbox)

                results['stereo_3d_score'] = stereo_results['face_depth_score']
                results['depth_available'] = True
                results['depth_method'] = 'stereo'

            except Exception as e:
                print(f"[!] Stereo depth analysis failed: {e}")

        # Structured light analysis (can work with single camera)
        if self.use_structured_light and left_frame is not None:
            try:
                light_results = self.light_analyzer.analyze_structured_light_depth(left_frame, face_bbox)
                results['structured_light_score'] = light_results['structured_light_score']
                results['depth_available'] = True

                if results['depth_method'] == 'none':
                    results['depth_method'] = 'structured_light'
                else:
                    results['depth_method'] = 'combined'

            except Exception as e:
                print(f"[!] Structured light analysis failed: {e}")

        # Combine depth scores
        if results['depth_available']:
            combined_score = (
                results['stereo_3d_score'] * self.depth_weights['stereo_3d_score'] +
                results['structured_light_score'] * self.depth_weights['structured_light_score']
            )

            # Add cross-validation bonus
            if results['stereo_3d_score'] > 0.5 and results['structured_light_score'] > 0.5:
                combined_score += self.depth_weights['combined_depth_score']

            results['combined_depth_score'] = min(combined_score, 1.0)
            results['is_live_depth'] = combined_score > 0.6  # Threshold for liveness

        return results


def create_depth_camera_pair():
    """
    Initialize dual camera setup for depth sensing
    Returns camera indices for left and right cameras
    """
    # Try to find available cameras
    left_cam = None
    right_cam = None

    for i in range(10):  # Check first 10 camera indices
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            if left_cam is None:
                left_cam = i
            elif right_cam is None:
                right_cam = i
            cap.release()

            if left_cam is not None and right_cam is not None:
                break

    return left_cam, right_cam


def test_depth_system():
    """Test function for depth-based liveness detection"""
    print("[*] Testing depth-based liveness detection...")

    # Initialize depth detector
    depth_detector = DepthBasedLivenessDetector()

    # Try to find stereo cameras
    left_idx, right_idx = create_depth_camera_pair()

    if left_idx is None:
        print("[!] No cameras found for depth sensing")
        return

    print(f"[+] Using cameras: Left={left_idx}, Right={right_idx}")

    # Initialize cameras
    left_cap = cv2.VideoCapture(left_idx, cv2.CAP_DSHOW)
    right_cap = cv2.VideoCapture(right_idx, cv2.CAP_DSHOW) if right_idx is not None else None

    if not left_cap.isOpened():
        print("[!] Cannot open left camera")
        return

    print("[*] Press 'q' to quit depth sensing test")

    try:
        while True:
            # Capture frames
            ret_left, left_frame = left_cap.read()
            ret_right, right_frame = right_cap.read() if right_cap else (False, None)

            if not ret_left:
                continue

            # Face detection (simplified)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) > 0:
                # Use first detected face
                x, y, w, h = faces[0]
                face_bbox = (y, x + w, y + h, x)  # Convert to (top, right, bottom, left)

                # Analyze depth
                depth_results = depth_detector.analyze_depth_liveness(
                    left_frame=left_frame,
                    right_frame=right_frame,
                    face_bbox=face_bbox
                )

                # Display results
                status = "LIVE" if depth_results['is_live_depth'] else "SPOOF"
                color = (0, 255, 0) if depth_results['is_live_depth'] else (0, 0, 255)

                cv2.rectangle(left_frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(left_frame, f"Depth: {status}", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Show depth info
                info = ".2f"
                cv2.putText(left_frame, info, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Depth Liveness Detection", left_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        left_cap.release()
        if right_cap:
            right_cap.release()
        cv2.destroyAllWindows()

    print("[+] Depth sensing test completed")


if __name__ == "__main__":
    test_depth_system()
