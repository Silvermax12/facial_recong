"""
Mouth Detection Module for Open Mouth Challenge
Uses MediaPipe face mesh to detect mouth opening gestures
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple


class MouthDetector:
    """Detects mouth opening using MediaPipe face mesh landmarks"""
    
    # Mouth landmark indices from MediaPipe face mesh
    UPPER_LIP_INDICES = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
    LOWER_LIP_INDICES = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
    
    # Simplified mouth corners and top/bottom points
    MOUTH_TOP = 13
    MOUTH_BOTTOM = 14
    MOUTH_LEFT = 78
    MOUTH_RIGHT = 308
    
    def __init__(self, mar_threshold: float = 0.6):
        """
        Initialize mouth detector
        
        Args:
            mar_threshold: Mouth Aspect Ratio threshold for open mouth detection
        """
        self.mar_threshold = mar_threshold
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def calculate_mar(self, landmarks, img_shape) -> float:
        """
        Calculate Mouth Aspect Ratio (MAR)
        MAR = (vertical distance) / (horizontal distance)
        
        Args:
            landmarks: MediaPipe face mesh landmarks
            img_shape: Image shape (height, width, channels)
            
        Returns:
            Mouth Aspect Ratio value
        """
        h, w = img_shape[:2]
        
        # Get key mouth points
        mouth_top = landmarks[self.MOUTH_TOP]
        mouth_bottom = landmarks[self.MOUTH_BOTTOM]
        mouth_left = landmarks[self.MOUTH_LEFT]
        mouth_right = landmarks[self.MOUTH_RIGHT]
        
        # Convert normalized coordinates to pixel coordinates
        top_y = mouth_top.y * h
        bottom_y = mouth_bottom.y * h
        left_x = mouth_left.x * w
        right_x = mouth_right.x * w
        
        # Calculate distances
        vertical_dist = abs(bottom_y - top_y)
        horizontal_dist = abs(right_x - left_x)
        
        # Avoid division by zero
        if horizontal_dist < 1:
            return 0.0
            
        mar = vertical_dist / horizontal_dist
        return mar
    
    def detect_open_mouth(self, frame: np.ndarray) -> Tuple[bool, float, Optional[np.ndarray]]:
        """
        Detect if mouth is open in the given frame
        
        Args:
            frame: BGR image from camera
            
        Returns:
            Tuple of (is_open, mar_value, annotated_frame)
            - is_open: True if mouth is detected as open
            - mar_value: Calculated MAR value
            - annotated_frame: Frame with mouth landmarks drawn (optional)
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return False, 0.0, None
        
        # Get first face landmarks
        face_landmarks = results.multi_face_landmarks[0]
        
        # Calculate MAR
        mar = self.calculate_mar(face_landmarks.landmark, frame.shape)
        
        # Determine if mouth is open
        is_open = mar > self.mar_threshold
        
        # Create annotated frame (optional)
        annotated_frame = frame.copy()
        
        # Draw mouth landmarks
        h, w = frame.shape[:2]
        mouth_points = [self.MOUTH_TOP, self.MOUTH_BOTTOM, self.MOUTH_LEFT, self.MOUTH_RIGHT]
        
        for idx in mouth_points:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            color = (0, 255, 0) if is_open else (0, 0, 255)
            cv2.circle(annotated_frame, (x, y), 3, color, -1)
        
        # Draw MAR value
        cv2.putText(annotated_frame, f"MAR: {mar:.2f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        status = "OPEN" if is_open else "CLOSED"
        color = (0, 255, 0) if is_open else (0, 0, 255)
        cv2.putText(annotated_frame, status, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return is_open, mar, annotated_frame
    
    def analyze_sequence(self, frames: list) -> dict:
        """
        Analyze a sequence of frames for mouth opening
        
        Args:
            frames: List of BGR frames
            
        Returns:
            Dictionary with analysis results
        """
        open_count = 0
        mar_values = []
        
        for frame in frames:
            is_open, mar, _ = self.detect_open_mouth(frame)
            if is_open:
                open_count += 1
            mar_values.append(mar)
        
        avg_mar = np.mean(mar_values) if mar_values else 0.0
        max_mar = np.max(mar_values) if mar_values else 0.0
        
        # Consider sequence valid if at least 30% of frames show open mouth
        # and max MAR is above threshold
        is_valid = (open_count / len(frames) >= 0.3 and 
                   max_mar > self.mar_threshold)
        
        return {
            'is_valid': is_valid,
            'open_count': open_count,
            'total_frames': len(frames),
            'open_percentage': (open_count / len(frames) * 100) if frames else 0,
            'avg_mar': avg_mar,
            'max_mar': max_mar,
            'mar_values': mar_values
        }
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
    
    def __del__(self):
        """Destructor to ensure resources are cleaned up"""
        self.close()


def detect_mouth_opening_in_frames(frames: list, mar_threshold: float = 0.6) -> dict:
    """
    Convenience function to detect mouth opening in a sequence of frames
    
    Args:
        frames: List of image frames (BGR format)
        mar_threshold: Threshold for mouth opening detection
        
    Returns:
        Dictionary with detection results
    """
    detector = MouthDetector(mar_threshold=mar_threshold)
    results = detector.analyze_sequence(frames)
    detector.close()
    return results

