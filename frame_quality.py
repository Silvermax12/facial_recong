"""
Frame Quality Analysis Module
Provides frame enhancement, quality assessment, and filtering for liveness detection
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List


class FrameQualityAnalyzer:
    """
    Analyzes and enhances frame quality for liveness detection
    - Auto brightness/contrast normalization
    - Blur detection
    - Multi-face detection and rejection
    - Lighting quality assessment
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Quality thresholds (configurable)
        self.blur_threshold = self.config.get('blur_threshold', 100.0)
        self.min_brightness = self.config.get('min_brightness', 40)
        self.max_brightness = self.config.get('max_brightness', 220)
        self.target_brightness = self.config.get('target_brightness', 128)
        self.min_face_size = self.config.get('min_face_size', 80)  # pixels
        self.max_faces_allowed = self.config.get('max_faces_allowed', 1)
        
    def analyze_and_enhance(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Analyze frame quality and apply enhancements
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Tuple of (enhanced_frame, quality_metrics)
        """
        quality_metrics = {
            'original_brightness': 0.0,
            'enhanced_brightness': 0.0,
            'blur_score': 0.0,
            'is_blurry': False,
            'num_faces': 0,
            'lighting_quality': 'unknown',
            'frame_quality_score': 0.0,
            'enhancements_applied': [],
            'rejection_reasons': []
        }
        
        if frame is None or frame.size == 0:
            quality_metrics['rejection_reasons'].append('empty_frame')
            return frame, quality_metrics
        
        enhanced_frame = frame.copy()
        
        # 1. Brightness analysis and normalization
        brightness_metrics = self._analyze_brightness(frame)
        quality_metrics['original_brightness'] = brightness_metrics['avg_brightness']
        quality_metrics['lighting_quality'] = brightness_metrics['quality']
        
        if brightness_metrics['needs_adjustment']:
            enhanced_frame = self._normalize_brightness_contrast(enhanced_frame)
            quality_metrics['enhancements_applied'].append('brightness_normalization')
        
        quality_metrics['enhanced_brightness'] = self._calculate_brightness(enhanced_frame)
        
        # 2. Blur detection
        blur_metrics = self._detect_blur(enhanced_frame)
        quality_metrics['blur_score'] = float(blur_metrics['score'])
        quality_metrics['is_blurry'] = bool(blur_metrics['is_blurry'])
        
        if blur_metrics['is_blurry']:
            quality_metrics['rejection_reasons'].append('blurry_frame')
        
        # 3. Multi-face detection
        face_count = self._count_faces(enhanced_frame)
        quality_metrics['num_faces'] = int(face_count)
        
        if face_count == 0:
            quality_metrics['rejection_reasons'].append('no_face_detected')
        elif face_count > self.max_faces_allowed:
            quality_metrics['rejection_reasons'].append(f'multiple_faces_detected:{face_count}')
        
        # 4. Calculate overall quality score (0-100)
        quality_metrics['frame_quality_score'] = float(self._calculate_quality_score(quality_metrics))
        
        return enhanced_frame, quality_metrics
    
    def _analyze_brightness(self, frame: np.ndarray) -> Dict:
        """Analyze brightness levels and determine if adjustment needed"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        avg_brightness = np.mean(gray)
        
        quality = 'good'
        needs_adjustment = False
        
        if avg_brightness < self.min_brightness:
            quality = 'too_dark'
            needs_adjustment = True
        elif avg_brightness > self.max_brightness:
            quality = 'too_bright'
            needs_adjustment = True
        elif avg_brightness < 60 or avg_brightness > 200:
            quality = 'poor'
            needs_adjustment = True
        
        return {
            'avg_brightness': float(avg_brightness),
            'quality': quality,
            'needs_adjustment': bool(needs_adjustment)
        }
    
    def _calculate_brightness(self, frame: np.ndarray) -> float:
        """Calculate average brightness of frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        return float(np.mean(gray))
    
    def _normalize_brightness_contrast(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply adaptive brightness and contrast normalization
        Uses CLAHE (Contrast Limited Adaptive Histogram Equalization) for better results
        """
        # Convert to LAB color space for better brightness normalization
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # Merge channels and convert back
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Additional brightness adjustment if needed
        current_brightness = np.mean(cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY))
        if abs(current_brightness - self.target_brightness) > 20:
            # Apply gamma correction
            gamma = self.target_brightness / max(current_brightness, 1)
            gamma = np.clip(gamma, 0.5, 2.0)  # Limit gamma range
            
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
            enhanced_bgr = cv2.LUT(enhanced_bgr, table)
        
        return enhanced_bgr
    
    def _detect_blur(self, frame: np.ndarray) -> Dict:
        """
        Detect blur using Laplacian variance method
        Lower variance = more blurry
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        is_blurry = laplacian_var < self.blur_threshold
        
        return {
            'score': float(laplacian_var),
            'is_blurry': bool(is_blurry),
            'threshold': float(self.blur_threshold)
        }
    
    def _count_faces(self, frame: np.ndarray) -> int:
        """Count number of faces in frame using Haar Cascade (fast)"""
        try:
            # Use Haar Cascade for quick face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(self.min_face_size, self.min_face_size)
            )
            
            return len(faces)
        except Exception as e:
            print(f"[!] Face detection error: {e}")
            return 0
    
    def _calculate_quality_score(self, metrics: Dict) -> float:
        """
        Calculate overall quality score (0-100)
        Higher is better
        """
        score = 100.0
        
        # Deduct points for rejections
        if metrics['is_blurry']:
            score -= 30
        
        if metrics['num_faces'] == 0:
            score -= 50
        elif metrics['num_faces'] > self.max_faces_allowed:
            score -= 40
        
        # Deduct for poor lighting
        if metrics['lighting_quality'] in ['too_dark', 'too_bright']:
            score -= 20
        elif metrics['lighting_quality'] == 'poor':
            score -= 10
        
        # Blur penalty (proportional)
        if metrics['blur_score'] < self.blur_threshold:
            blur_ratio = metrics['blur_score'] / self.blur_threshold
            score -= (1 - blur_ratio) * 20
        
        return max(0.0, min(100.0, score))
    
    def batch_analyze(self, frames: List[np.ndarray]) -> Tuple[List[np.ndarray], List[Dict]]:
        """Analyze and enhance multiple frames"""
        enhanced_frames = []
        all_metrics = []
        
        for frame in frames:
            enhanced, metrics = self.analyze_and_enhance(frame)
            enhanced_frames.append(enhanced)
            all_metrics.append(metrics)
        
        return enhanced_frames, all_metrics
    
    def filter_quality_frames(
        self, 
        frames: List[np.ndarray], 
        min_quality_score: float = 50.0
    ) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        Filter frames by quality score, returning only good frames
        
        Args:
            frames: List of input frames
            min_quality_score: Minimum quality score to keep (0-100)
            
        Returns:
            Tuple of (good_frames, metrics_for_good_frames)
        """
        enhanced_frames, all_metrics = self.batch_analyze(frames)
        
        good_frames = []
        good_metrics = []
        
        for frame, metrics in zip(enhanced_frames, all_metrics):
            if metrics['frame_quality_score'] >= min_quality_score and not metrics['rejection_reasons']:
                good_frames.append(frame)
                good_metrics.append(metrics)
        
        return good_frames, good_metrics


class EnhancedPreprocessor:
    """Enhanced preprocessing for DeepPixBis and other models"""
    
    @staticmethod
    def preprocess_for_deeppixbis(frame: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Preprocess frame for DeepPixBis model with quality enhancements
        
        Args:
            frame: Input BGR frame
            target_size: Target size for model input
            
        Returns:
            Preprocessed frame ready for model
        """
        # Apply quality enhancement first
        quality_analyzer = FrameQualityAnalyzer()
        enhanced_frame, _ = quality_analyzer.analyze_and_enhance(frame)
        
        # Resize
        resized = cv2.resize(enhanced_frame, target_size, interpolation=cv2.INTER_CUBIC)
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized
    
    @staticmethod
    def apply_data_augmentation(frame: np.ndarray, augmentation_type: str = 'none') -> np.ndarray:
        """Apply data augmentation for robustness"""
        if augmentation_type == 'horizontal_flip':
            return cv2.flip(frame, 1)
        elif augmentation_type == 'rotation':
            angle = np.random.uniform(-15, 15)
            h, w = frame.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            return cv2.warpAffine(frame, M, (w, h))
        elif augmentation_type == 'brightness':
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * np.random.uniform(0.8, 1.2), 0, 255)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        else:
            return frame


def get_frame_statistics(frames: List[np.ndarray]) -> Dict:
    """Calculate statistics across frame sequence"""
    if not frames:
        return {'error': 'no_frames'}
    
    brightnesses = []
    blur_scores = []
    
    analyzer = FrameQualityAnalyzer()
    
    for frame in frames:
        _, metrics = analyzer.analyze_and_enhance(frame)
        brightnesses.append(metrics['enhanced_brightness'])
        blur_scores.append(metrics['blur_score'])
    
    return {
        'avg_brightness': float(np.mean(brightnesses)),
        'brightness_variance': float(np.var(brightnesses)),
        'brightness_std': float(np.std(brightnesses)),
        'avg_blur_score': float(np.mean(blur_scores)),
        'min_blur_score': float(np.min(blur_scores)),
        'max_blur_score': float(np.max(blur_scores)),
        'num_frames': len(frames)
    }

