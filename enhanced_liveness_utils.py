import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from scipy import fftpack
import mediapipe as mp
from collections import deque
import time
import os
import io
try:
    from PIL import Image as PILImage
    import PIL.ExifTags
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

class FrequencyDomainAnalyzer:
    """
    Enhanced frequency domain analysis to detect synthetic/AI-generated content and static images
    Analyzes multiple spectral characteristics including compression artifacts, noise patterns, and texture consistency
    """
    def __init__(self):
        self.spectrum_threshold = 0.6  # Threshold for synthetic content detection
        self.static_image_threshold = 0.7  # Higher threshold for static image detection

    def analyze_frequency_spectrum(self, image):
        """
        Enhanced frequency spectrum analysis for spoofing detection
        Returns: synthetic_probability (0-1, higher = more likely synthetic/static)
        """
        try:
            # Convert to grayscale for frequency analysis
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Apply 2D FFT
            fft = fftpack.fft2(gray.astype(float))
            fft_shift = fftpack.fftshift(fft)
            magnitude = np.abs(fft_shift)

            # Multiple analysis methods for robust detection
            synthetic_indicators = []

            # 1. High-frequency energy analysis (original method, enhanced)
            high_freq_score = self._analyze_high_frequency_energy(magnitude)
            synthetic_indicators.append(high_freq_score)

            # 2. Compression artifact detection (JPEG artifacts are periodic)
            compression_score = self._detect_compression_artifacts(magnitude)
            synthetic_indicators.append(compression_score)

            # 3. Noise pattern analysis (real images have natural noise)
            noise_score = self._analyze_noise_patterns(gray)
            synthetic_indicators.append(noise_score)

            # 4. Spectral entropy analysis (synthetic images often have unnatural entropy patterns)
            entropy_score = self._analyze_spectral_entropy(magnitude)
            synthetic_indicators.append(entropy_score)

            # 5. Edge consistency analysis (static images may have unnatural edges)
            edge_score = self._analyze_edge_consistency(gray)
            synthetic_indicators.append(edge_score)

            # Combine indicators with weights (static images score higher)
            weights = [0.25, 0.20, 0.20, 0.15, 0.20]  # Favor noise and edge analysis for static detection
            final_score = sum(score * weight for score, weight in zip(synthetic_indicators, weights))

            # Apply sigmoid to bound between 0-1
            final_score = 1.0 / (1.0 + np.exp(-(final_score - 0.5) * 8))

            return float(final_score)

        except Exception as e:
            print(f"[!] Frequency analysis error: {e}")
            return 0.5

    def _analyze_high_frequency_energy(self, magnitude):
        """Analyze high-frequency energy distribution"""
        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2

        # High frequency regions (corners of spectrum)
        high_freq_regions = [
            magnitude[:center_h//2, :center_w//2],      # Top-left
            magnitude[:center_h//2, center_w + center_w//2:],  # Top-right
            magnitude[center_h + center_h//2:, :center_w//2],  # Bottom-left
            magnitude[center_h + center_h//2:, center_w + center_w//2:]  # Bottom-right
        ]

        high_freq_energy = sum(np.sum(region**2) for region in high_freq_regions)
        total_energy = np.sum(magnitude**2)

        if total_energy == 0:
            return 0.5

        high_freq_ratio = high_freq_energy / total_energy

        # Enhanced heuristic: synthetic images often have either too much or too little high-frequency content
        if high_freq_ratio < 0.1:  # Too smooth (over-processed)
            return 0.8
        elif high_freq_ratio > 0.6:  # Too noisy (artificial generation)
            return 0.7
        else:
            return 0.3  # Normal range

    def _detect_compression_artifacts(self, magnitude):
        """Detect periodic artifacts from JPEG compression"""
        h, w = magnitude.shape

        # Look for periodic patterns in frequency domain (JPEG blocks are 8x8)
        # Check for peaks at frequencies corresponding to 8-pixel intervals
        block_size_indicators = []

        for block_size in [8, 16, 32]:  # Common JPEG block sizes
            freq_x = w // block_size
            freq_y = h // block_size

            # Check if there are strong periodic components
            roi = magnitude[h//2-freq_y:h//2+freq_y, w//2-freq_x:w//2+freq_x]
            avg_roi = np.mean(roi)
            avg_center = np.mean(magnitude[h//2-5:h//2+5, w//2-5:w//2+5])

            if avg_roi > avg_center * 1.5:  # Strong periodic component
                block_size_indicators.append(0.8)
            else:
                block_size_indicators.append(0.2)

        return max(block_size_indicators) if block_size_indicators else 0.3

    def _analyze_noise_patterns(self, gray_image):
        """Analyze noise patterns - real images have natural noise, synthetics often have artificial patterns"""
        try:
            # Estimate noise using local variance
            kernel = np.ones((3,3), np.uint8)
            local_var = cv2.filter2D(gray_image.astype(np.float32)**2, -1, kernel/9.0) - \
                       (cv2.filter2D(gray_image.astype(np.float32), -1, kernel/9.0))**2

            noise_std = np.std(local_var)

            # Real images typically have noise std between 10-100
            # Very low noise suggests over-processing (synthetic)
            # Very high noise suggests artificial generation
            if noise_std < 5:
                return 0.9  # Too smooth, likely synthetic
            elif noise_std > 200:
                return 0.8  # Too noisy, likely artificial
            else:
                return 0.2  # Normal noise level

        except:
            return 0.5

    def _analyze_spectral_entropy(self, magnitude):
        """Analyze spectral entropy - synthetic images often have unnatural entropy distributions"""
        try:
            # Normalize magnitude spectrum
            magnitude_norm = magnitude / (np.sum(magnitude) + 1e-10)

            # Calculate spectral entropy
            entropy = -np.sum(magnitude_norm * np.log2(magnitude_norm + 1e-10))

            # Normalize entropy (typical range: 10-18 for natural images)
            # Too low entropy = unnatural spectral distribution
            # Too high entropy = artificial noise patterns
            if entropy < 8:
                return 0.8  # Unnatural spectral distribution
            elif entropy > 20:
                return 0.7  # Artificial entropy pattern
            else:
                return 0.2  # Normal entropy

        except:
            return 0.5

    def _analyze_edge_consistency(self, gray_image):
        """Analyze edge consistency - static images may have unnatural edge patterns"""
        try:
            # Detect edges
            edges = cv2.Canny(gray_image, 50, 150)

            # Analyze edge distribution
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

            # Check for unnatural edge patterns
            # Too few edges = over-smoothed (synthetic)
            # Too many edges = artificial sharpening
            if edge_density < 0.02:
                return 0.9  # Too few edges, likely synthetic
            elif edge_density > 0.25:
                return 0.7  # Too many edges, likely artificial
            else:
                return 0.2  # Normal edge density

        except:
            return 0.5


class StaticImageDetector:
    """
    Detects static images/photos that are not live video frames
    Uses multiple heuristics to identify pre-recorded content
    """
    def __init__(self):
        self.static_threshold = 0.75  # Threshold for flagging as static

    def detect_static_image(self, image_bgr, image_path=None, image_bytes=None):
        """
        Comprehensive static image detection
        Returns: static_probability (0-1, higher = more likely static image)
        
        Args:
            image_bgr: BGR image array
            image_path: Optional file path (deprecated, use image_bytes)
            image_bytes: Optional raw image bytes for EXIF analysis
        """
        static_indicators = []

        # 1. EXIF metadata analysis (photos often have rich EXIF data)
        # FIXED: MAJOR-4 - Use bytes instead of path for uploaded images
        if image_bytes is not None:
            exif_score = self._analyze_exif_from_bytes(image_bytes)
        elif image_path is not None:
            exif_score = self._analyze_exif_metadata(image_path)
        else:
            exif_score = 0.5  # Neutral if no metadata source
        static_indicators.append(exif_score)

        # 2. Image format analysis (JPEG compression artifacts)
        format_score = self._analyze_image_format(image_bgr)
        static_indicators.append(format_score)

        # 3. Color space analysis (photos often have sRGB color space artifacts)
        color_score = self._analyze_color_space_artifacts(image_bgr)
        static_indicators.append(color_score)

        # 4. Temporal artifacts (static images lack temporal coherence)
        temporal_score = self._detect_temporal_artifacts(image_bgr)
        static_indicators.append(temporal_score)

        # 5. Resolution and aspect ratio analysis
        resolution_score = self._analyze_resolution_patterns(image_bgr)
        static_indicators.append(resolution_score)

        # Weighted combination
        weights = [0.25, 0.20, 0.20, 0.20, 0.15]
        final_score = sum(score * weight for score, weight in zip(static_indicators, weights))

        # Apply sigmoid for final probability
        final_score = 1.0 / (1.0 + np.exp(-(final_score - 0.5) * 6))

        return float(final_score)

    def _analyze_exif_metadata(self, image_path):
        """Analyze EXIF metadata - photos typically have extensive metadata"""
        if not HAS_PIL or not image_path or not os.path.exists(image_path):
            return 0.3  # Neutral if no metadata available

        try:
            img = PILImage.open(image_path)
            exif_data = img._getexif()

            if exif_data is None:
                return 0.2  # No EXIF data, might be a video frame

            # Count EXIF tags (photos typically have many)
            exif_tags = len(exif_data)

            # Check for photo-specific EXIF tags
            photo_tags = [PIL.ExifTags.TAGS.get(tag, '') for tag in exif_data.keys()]
            camera_tags = ['Make', 'Model', 'DateTime', 'ISOSpeedRatings', 'FNumber', 'ExposureTime']
            camera_tag_count = sum(1 for tag in photo_tags if tag in camera_tags)

            # High EXIF tag count + camera tags = likely a photo
            if exif_tags > 10 and camera_tag_count >= 2:
                return 0.9  # Very likely a photo
            elif exif_tags > 5:
                return 0.7  # Likely a photo
            else:
                return 0.4  # Could be either

        except Exception:
            return 0.3  # Error, neutral score
    
    def _analyze_exif_from_bytes(self, image_bytes):
        """Analyze EXIF metadata from raw bytes (for uploaded images)"""
        if not HAS_PIL or image_bytes is None:
            return 0.5  # Neutral if no PIL or no bytes
        
        try:
            import io
            img = PILImage.open(io.BytesIO(image_bytes))
            exif_data = img._getexif()
            
            if exif_data is None:
                return 0.2  # No EXIF data, might be a video frame
            
            # Count EXIF tags (photos typically have many)
            exif_tags = len(exif_data)
            
            # Check for photo-specific EXIF tags
            photo_tags = [PIL.ExifTags.TAGS.get(tag, '') for tag in exif_data.keys()]
            camera_tags = ['Make', 'Model', 'DateTime', 'ISOSpeedRatings', 'FNumber', 'ExposureTime']
            camera_tag_count = sum(1 for tag in photo_tags if tag in camera_tags)
            
            # High EXIF tag count + camera tags = likely a photo
            if exif_tags > 10 and camera_tag_count >= 2:
                return 0.9  # Very likely a photo
            elif exif_tags > 5:
                return 0.7  # Likely a photo
            else:
                return 0.4  # Could be either
        
        except Exception:
            return 0.5  # Error, neutral score

    def _analyze_image_format(self, image_bgr):
        """Analyze image format characteristics - JPEG compression creates specific artifacts"""
        try:
            # Convert to JPEG in memory and check for compression artifacts
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            result, enc_img = cv2.imencode('.jpg', image_bgr, encode_param)

            if not result:
                return 0.3

            # Decode back
            dec_img = cv2.imdecode(enc_img, cv2.IMREAD_COLOR)

            # Calculate compression artifacts (difference between original and re-encoded)
            diff = cv2.absdiff(image_bgr.astype(np.float32), dec_img.astype(np.float32))
            artifact_level = np.mean(diff) / 255.0

            # Low artifact level suggests already heavily compressed (like a photo)
            # High artifact level suggests uncompressed or different format
            if artifact_level < 0.01:
                return 0.8  # Likely already compressed (photo)
            elif artifact_level > 0.05:
                return 0.3  # Likely uncompressed (video frame)
            else:
                return 0.5  # Ambiguous

        except Exception:
            return 0.3

    def _analyze_color_space_artifacts(self, image_bgr):
        """Analyze color space artifacts - photos often have sRGB gamma correction"""
        try:
            # Convert to different color spaces and look for artifacts
            hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

            # Analyze HSV histogram for unnatural distributions
            h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
            v_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])

            # Normalize histograms
            h_hist = h_hist / (np.sum(h_hist) + 1e-10)
            s_hist = s_hist / (np.sum(s_hist) + 1e-10)
            v_hist = v_hist / (np.sum(v_hist) + 1e-10)

            # Calculate entropy for each channel
            h_entropy = -np.sum(h_hist * np.log2(h_hist + 1e-10))
            s_entropy = -np.sum(s_hist * np.log2(s_hist + 1e-10))
            v_entropy = -np.sum(v_hist * np.log2(v_hist + 1e-10))

            # Photos often have specific entropy patterns due to sRGB conversion
            avg_entropy = (h_entropy + s_entropy + v_entropy) / 3

            if avg_entropy < 5.0:  # Low entropy suggests processed image
                return 0.8
            elif avg_entropy > 7.0:  # High entropy suggests raw/unprocessed
                return 0.3
            else:
                return 0.5

        except Exception:
            return 0.3

    def _detect_temporal_artifacts(self, image_bgr):
        """Detect artifacts that indicate lack of temporal coherence (static image)"""
        try:
            # Analyze for motion blur (real video frames often have slight motion blur)
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

            # Calculate Laplacian variance (measure of focus/sharpness)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Very high sharpness often indicates a static photo
            # Real video frames often have slight blur from motion/camera shake
            if laplacian_var > 500:  # Too sharp, likely a static photo
                return 0.8
            elif laplacian_var < 50:  # Too blurry, might be poor quality
                return 0.4
            else:
                return 0.3  # Normal sharpness

        except Exception:
            return 0.3

    def _analyze_resolution_patterns(self, image_bgr):
        """Analyze resolution and aspect ratio patterns typical of photos vs video frames"""
        try:
            h, w = image_bgr.shape[:2]
            aspect_ratio = w / h

            # Common photo aspect ratios
            photo_ratios = [4/3, 3/2, 16/9, 1/1, 5/4, 5/3]
            closest_photo_ratio = min(photo_ratios, key=lambda x: abs(x - aspect_ratio))

            ratio_deviation = abs(aspect_ratio - closest_photo_ratio) / closest_photo_ratio

            # Check if dimensions are common photo resolutions
            common_resolutions = [
                (1920, 1080), (1280, 720), (1024, 768), (800, 600),  # Video-like
                (4000, 3000), (3000, 2000), (2048, 1536), (1600, 1200),  # Photo-like
                (3264, 2448), (2592, 1944), (2048, 1536)  # Common camera resolutions
            ]

            # Check if close to common photo resolutions
            is_photo_resolution = any(
                abs(w - pw) / pw < 0.1 and abs(h - ph) / ph < 0.1
                for pw, ph in common_resolutions[4:]  # Photo resolutions
            )

            if is_photo_resolution and ratio_deviation < 0.1:
                return 0.8  # Very likely a photo
            elif ratio_deviation > 0.3:
                return 0.3  # Unusual ratio, might be video crop
            else:
                return 0.5  # Ambiguous

        except Exception:
            return 0.3


class InjectionAttackDetector:
    """
    Detects injection attacks where synthetic images/videos are fed directly to the system
    Analyzes patterns that indicate programmatic image injection rather than camera capture
    """
    def __init__(self):
        self.injection_threshold = 0.8
        self.request_timestamps = deque(maxlen=100)  # Track recent requests
        self.suspicious_patterns = {
            'identical_images': 0,      # Same image sent multiple times
            'perfect_timing': 0,        # Too regular timing between requests
            'uniform_metadata': 0,      # All images have identical properties
            'synthetic_artifacts': 0,   # Clear synthetic generation markers
        }

    def detect_injection_attack(self, image_bgr, request_timestamp=None, client_info=None):
        """
        Comprehensive injection attack detection
        Returns: injection_probability (0-1, higher = more likely injection attack)
        """
        injection_indicators = []

        # 1. Timing pattern analysis
        timing_score = self._analyze_timing_patterns(request_timestamp)
        injection_indicators.append(timing_score)

        # 2. Image uniformity analysis (synthetic images often have uniform properties)
        uniformity_score = self._analyze_image_uniformity(image_bgr)
        injection_indicators.append(uniformity_score)

        # 3. Synthetic artifact detection
        artifact_score = self._detect_synthetic_artifacts(image_bgr)
        injection_indicators.append(artifact_score)

        # 4. Metadata consistency analysis
        metadata_score = self._analyze_metadata_consistency(image_bgr, client_info)
        injection_indicators.append(metadata_score)

        # 5. Request pattern analysis
        pattern_score = self._analyze_request_patterns()
        injection_indicators.append(pattern_score)

        # Weighted combination
        weights = [0.15, 0.25, 0.25, 0.15, 0.20]
        final_score = sum(score * weight for score, weight in zip(injection_indicators, weights))

        # Apply sigmoid for final probability
        final_score = 1.0 / (1.0 + np.exp(-(final_score - 0.5) * 6))

        return float(final_score)

    def _analyze_timing_patterns(self, current_timestamp):
        """Analyze timing patterns between requests"""
        if current_timestamp is None:
            return 0.3

        self.request_timestamps.append(current_timestamp)

        if len(self.request_timestamps) < 5:
            return 0.3

        # Calculate intervals between requests
        intervals = []
        for i in range(1, len(self.request_timestamps)):
            intervals.append(self.request_timestamps[i] - self.request_timestamps[i-1])

        if not intervals:
            return 0.3

        # Check for suspiciously regular timing (programmatic injection)
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)

        # Very low standard deviation indicates machine-generated timing
        regularity_score = 1.0 - min(std_interval / mean_interval, 1.0) if mean_interval > 0 else 0.5

        # Too regular timing is suspicious
        if regularity_score > 0.8:
            return 0.9  # Very likely programmatic injection
        elif regularity_score > 0.6:
            return 0.7  # Likely injection
        else:
            return 0.3  # Normal timing variation

    def _analyze_image_uniformity(self, image_bgr):
        """Analyze if image properties are suspiciously uniform (synthetic)"""
        try:
            # Convert to different color spaces
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

            uniformity_scores = []

            # Check histogram uniformity for each channel
            for channel_data in [gray, hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]]:
                hist = cv2.calcHist([channel_data], [0], None, [256], [0, 256])
                hist = hist / (np.sum(hist) + 1e-10)

                # Calculate entropy (uniform distributions have high entropy)
                entropy = -np.sum(hist * np.log2(hist + 1e-10))

                # Perfectly uniform distribution would have entropy = log2(256) ≈ 8
                # Real images have lower entropy due to natural variations
                uniformity_score = entropy / 8.0
                uniformity_scores.append(uniformity_score)

            avg_uniformity = np.mean(uniformity_scores)

            # Too uniform suggests synthetic generation
            if avg_uniformity > 0.9:
                return 0.9  # Very uniform, likely synthetic
            elif avg_uniformity > 0.7:
                return 0.7  # Suspiciously uniform
            else:
                return 0.2  # Normal variation

        except Exception:
            return 0.3

    def _detect_synthetic_artifacts(self, image_bgr):
        """Detect specific artifacts common in synthetic images"""
        try:
            artifacts_detected = []

            # 1. Check for pixel-perfect regularity (common in generated images)
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            # Look for perfectly straight lines (uncommon in real camera images)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
            if lines is not None and len(lines) > 10:
                # Too many straight lines might indicate synthetic content
                artifacts_detected.append(0.8)
            else:
                artifacts_detected.append(0.2)

            # 2. Check for color banding (common in low-quality synthetic images)
            hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
            for channel in range(3):
                channel_data = hsv[:, :, channel]
                # Look for repeated values (banding)
                unique_vals = len(np.unique(channel_data))
                if unique_vals < 50:  # Too few unique values
                    artifacts_detected.append(0.9)
                elif unique_vals < 100:
                    artifacts_detected.append(0.6)
                else:
                    artifacts_detected.append(0.2)

            # 3. Check for unnatural color correlations
            b, g, r = cv2.split(image_bgr)
            # Real images have complex color correlations, synthetics may not
            bg_corr = np.corrcoef(b.flatten(), g.flatten())[0, 1]
            br_corr = np.corrcoef(b.flatten(), r.flatten())[0, 1]
            gr_corr = np.corrcoef(g.flatten(), r.flatten())[0, 1]

            # Perfect correlations are suspicious
            max_corr = max(abs(bg_corr), abs(br_corr), abs(gr_corr))
            if max_corr > 0.95:
                artifacts_detected.append(0.8)
            elif max_corr > 0.85:
                artifacts_detected.append(0.6)
            else:
                artifacts_detected.append(0.2)

            return max(artifacts_detected) if artifacts_detected else 0.3

        except Exception:
            return 0.3

    def _analyze_metadata_consistency(self, image_bgr, client_info):
        """Analyze metadata consistency across requests"""
        # This would typically check HTTP headers, user agent, etc.
        # For now, we use image properties as a proxy
        try:
            h, w = image_bgr.shape[:2]

            # Suspiciously consistent dimensions across requests could indicate injection
            # In a real implementation, this would track client behavior patterns

            # For now, check if dimensions are powers of 2 (common in synthetic generation)
            is_power_of_2 = (h & (h - 1)) == 0 and (w & (w - 1)) == 0
            if is_power_of_2 and h >= 256 and w >= 256:
                return 0.7  # Suspiciously perfect dimensions
            else:
                return 0.3  # Normal dimensions

        except Exception:
            return 0.3

    def _analyze_request_patterns(self):
        """Analyze patterns in request frequency and characteristics"""
        # This would track sophisticated patterns like:
        # - Burst requests followed by pauses
        # - Identical image hashes
        # - Sequential parameter modifications
        # For now, return neutral score
        return 0.3


class TemporalConsistencyChecker:
    """
    Analyzes temporal consistency across multiple frames to detect video spoofing
    """
    def __init__(self, buffer_size=10):
        self.frame_buffer = deque(maxlen=buffer_size)
        self.consistency_threshold = 0.8

    def add_frame(self, face_crop, timestamp=None):
        """Add a frame to the temporal analysis buffer"""
        if timestamp is None:
            timestamp = time.time()

        # Extract facial landmarks and features
        features = self._extract_temporal_features(face_crop)
        self.frame_buffer.append({
            'features': features,
            'timestamp': timestamp,
            'frame': face_crop.copy()
        })

    def check_temporal_consistency(self):
        """
        Check if recent frames show natural temporal consistency
        Returns: consistency_score (0-1, higher = more consistent) or None if insufficient data
        """
        if len(self.frame_buffer) < 3:
            return None  # Not enough data

        try:
            frames = list(self.frame_buffer)
            consistency_scores = []

            # Check feature consistency across time
            for i in range(1, len(frames)):
                prev_features = frames[i-1]['features']
                curr_features = frames[i]['features']

                # Calculate feature differences
                if prev_features is not None and curr_features is not None:
                    diff = np.linalg.norm(curr_features - prev_features)
                    # Natural movement should have gradual changes
                    time_diff = frames[i]['timestamp'] - frames[i-1]['timestamp']
                    normalized_diff = diff / max(time_diff, 0.01)  # Avoid division by zero

                    # Lower normalized_diff = more consistent
                    score = 1.0 / (1.0 + normalized_diff)
                    consistency_scores.append(score)

            if not consistency_scores:
                return None

            return float(np.mean(consistency_scores))

        except Exception as e:
            print(f"[!] Temporal consistency check error: {e}")
            return None

    def _extract_temporal_features(self, face_crop):
        """Extract features useful for temporal analysis"""
        try:
            # Simple feature extraction - can be enhanced
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

            # Edge density (should be relatively stable)
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])

            # Color histogram consistency
            hist = cv2.calcHist([face_crop], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            return np.concatenate([[edge_density], hist])

        except Exception as e:
            return None


class ActiveFlashDetector:
    """
    Uses screen flash to detect 3D facial properties
    Requires camera with flash/screen control capability
    """
    def __init__(self):
        self.flash_intensity = 0.8
        self.analysis_frames = 3

    def perform_flash_test(self, camera_index=0):
        """
        Perform active flash test (requires camera with controllable flash)
        This is a placeholder - actual implementation depends on camera hardware
        """
        # Note: This requires specific camera hardware support
        # Most webcams don't support flash control

        cap = cv2.VideoCapture(camera_index)
        flash_results = []

        try:
            # Capture baseline frame
            ret, baseline = cap.read()
            if not ret:
                return False

            # In a real implementation, you would:
            # 1. Turn on screen flash (bright white screen)
            # 2. Capture frames during flash
            # 3. Analyze light reflection patterns
            # 4. Turn off flash

            # For now, return a placeholder result
            # Real implementation would analyze:
            # - Light reflection patterns on facial surfaces
            # - Shadow consistency
            # - Specular highlights on eyes/skin

            return True  # Placeholder

        except Exception as e:
            print(f"[!] Flash detection error: {e}")
            return False
        finally:
            cap.release()


class EnhancedBlinkDetector:
    """
    Enhanced blink detection with pattern analysis and anti-spoofing
    """
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

        self.ear_threshold = 0.23
        self.min_consecutive_frames = 2
        self.blink_history = deque(maxlen=50)  # Track recent blinks

        # Anti-spoofing: natural blink patterns
        self.expected_blink_rate = 0.3  # blinks per second (normal range: 0.1-0.4)
        self.blink_duration_range = (0.1, 0.4)  # seconds

    def detect_blink_enhanced(self, frame_rgb, face_mesh):
        """Enhanced blink detection with anti-spoofing"""
        if face_mesh.multi_face_landmarks:
            landmarks = face_mesh.multi_face_landmarks[0].landmark

            # Calculate EAR for both eyes
            left_ear = self._calculate_ear(landmarks, self.LEFT_EYE_IDX, frame_rgb.shape[1], frame_rgb.shape[0])
            right_ear = self._calculate_ear(landmarks, self.RIGHT_EYE_IDX, frame_rgb.shape[1], frame_rgb.shape[0])
            ear = (left_ear + right_ear) / 2.0

            current_time = time.time()
            is_blinking = ear < self.ear_threshold

            # Track blink state
            if is_blinking:
                if not hasattr(self, 'blink_start_time'):
                    self.blink_start_time = current_time
                self.consecutive_low_frames = getattr(self, 'consecutive_low_frames', 0) + 1
            else:
                if hasattr(self, 'blink_start_time') and hasattr(self, 'consecutive_low_frames'):
                    if self.consecutive_low_frames >= self.min_consecutive_frames:
                        blink_duration = current_time - self.blink_start_time
                        self._record_blink(blink_duration, current_time)

                self.blink_start_time = None
                self.consecutive_low_frames = 0

            return ear, is_blinking

        return None, False

    def _calculate_ear(self, landmarks, eye_indices, img_w, img_h):
        """Calculate Eye Aspect Ratio"""
        points = [(int(landmarks[i].x * img_w), int(landmarks[i].y * img_h)) for i in eye_indices]
        p1, p2, p3, p4, p5, p6 = points

        # Vertical distances
        v1 = np.linalg.norm(np.array(p2) - np.array(p6))
        v2 = np.linalg.norm(np.array(p3) - np.array(p5))

        # Horizontal distance
        h = np.linalg.norm(np.array(p1) - np.array(p4))

        if h == 0:
            return 1.0

        return (v1 + v2) / (2.0 * h)

    def _record_blink(self, duration, timestamp):
        """Record blink event with anti-spoofing analysis"""
        self.blink_history.append({
            'duration': duration,
            'timestamp': timestamp,
            'is_natural': self._is_natural_blink(duration)
        })

    def _is_natural_blink(self, duration):
        """Check if blink duration is within natural range"""
        return self.blink_duration_range[0] <= duration <= self.blink_duration_range[1]

    def get_blink_pattern_score(self):
        """Analyze blink pattern for natural behavior"""
        if len(self.blink_history) < 3:
            return None

        recent_blinks = list(self.blink_history)[-10:]  # Last 10 blinks

        # Calculate blink rate
        if len(recent_blinks) >= 2:
            time_span = recent_blinks[-1]['timestamp'] - recent_blinks[0]['timestamp']
            blink_rate = len(recent_blinks) / time_span if time_span > 0 else 0
            rate_score = 1.0 / (1.0 + abs(blink_rate - self.expected_blink_rate) * 5)
        else:
            rate_score = 0.5

        # Check natural blink durations
        natural_blinks = sum(1 for blink in recent_blinks if blink['is_natural'])
        duration_score = natural_blinks / len(recent_blinks)

        return float((rate_score + duration_score) / 2.0)


def enhanced_liveness_check(face_crop, model, device, freq_analyzer=None,
                          temporal_checker=None, blink_detector=None, static_detector=None,
                          image_path=None, image_bytes=None, weights=None):
    """
    Comprehensive liveness check combining multiple detection methods
    """
    liveness_scores = {}

    # 1. Deep learning model score (ViT or traditional CNN)
    try:
        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        img_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        tensor = transform(img_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            pixel_out, model_score = model(tensor)
            liveness_scores['model'] = float(model_score.item())
    except Exception as e:
        print(f"[!] Model inference error: {e}")
        liveness_scores['model'] = 0.5

    # 2. Frequency domain analysis for synthetic content
    if freq_analyzer:
        freq_score = freq_analyzer.analyze_frequency_spectrum(face_crop)
        try:
            liveness_scores['frequency'] = float(1.0 - freq_score)  # Invert (higher = more live)
        except Exception:
            pass

    # 3. Temporal consistency (requires multiple frames)
    if temporal_checker:
        temp_score = temporal_checker.check_temporal_consistency()
        if temp_score is not None:
            liveness_scores['temporal'] = float(temp_score)

    # 4. Blink pattern analysis
    if blink_detector:
        blink_score = blink_detector.get_blink_pattern_score()
        if blink_score is not None:
            liveness_scores['blink_pattern'] = float(blink_score)

    # 5. Static image detection (NEW: specifically targets photos/videos)
    if static_detector:
        static_score = static_detector.detect_static_image(face_crop, image_path, image_bytes)
        if static_score is not None:
            # Invert: higher static score = lower liveness score
            liveness_scores['static_image'] = float(1.0 - static_score)

    # Weighted ensemble
    default_weights = {
        'model': 0.25,         # Primary model prediction (reduced weight)
        'frequency': 0.2,      # Frequency analysis
        'temporal': 0.15,      # Temporal consistency
        'blink_pattern': 0.15, # Blink behavior
        'static_image': 0.25   # Static image detection (NEW - high weight for photos)
    }
    if weights is None:
        weights = default_weights

    final_score = 0.0
    total_weight = 0.0
    expected_total_weight = sum(weights.values())  # Should be 1.0

    for method, score in liveness_scores.items():
        if method in weights:
            final_score += score * weights[method]
            total_weight += weights[method]

    # FIXED: CRITICAL-5 - If signals missing, apply penalty instead of normalizing
    if total_weight < expected_total_weight * 0.8:  # Missing >20% of signals
        missing_weight = expected_total_weight - total_weight
        final_score = final_score - (missing_weight * 0.3)  # Penalty for missing signals
        final_score = max(0.0, final_score)  # Clamp to 0
    else:
        # All signals present - normalize properly
        final_score = final_score / expected_total_weight if expected_total_weight > 0 else 0.5

    return final_score, liveness_scores
