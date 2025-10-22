import cv2
import numpy as np


def analyze_motion_coherence(frame_sequence):
    """
    Detect natural vs artificial motion using optical flow
    
    Args:
        frame_sequence: List of BGR frames
    
    Returns:
        Dictionary with motion analysis results
    """
    if len(frame_sequence) < 2:
        return {
            'motion_score': 0.0,
            'is_natural': False,
            'reason': 'insufficient_frames',
            'magnitude_variance': 0.0
        }
    
    flows = []
    
    for i in range(len(frame_sequence) - 1):
        prev_gray = cv2.cvtColor(frame_sequence[i], cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(frame_sequence[i+1], cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, next_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        flows.append(flow)
    
    # Analyze flow characteristics
    motion_magnitudes = []
    direction_variances = []
    
    for flow in flows:
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_magnitudes.append(np.mean(mag))
        direction_variances.append(np.var(ang))
    
    mag_variance = np.var(motion_magnitudes)
    avg_dir_variance = np.mean(direction_variances)
    
    # Video loops have near-zero variance in both magnitude and direction
    if mag_variance < 0.01:
        return {
            'motion_score': 0.0,
            'is_natural': False,
            'reason': 'static_or_loop',
            'magnitude_variance': float(mag_variance)
        }
    
    # Natural motion has variance in magnitude and direction
    # Combine both metrics for final score
    motion_score = min(mag_variance * 50, 1.0) * 0.7 + min(avg_dir_variance * 10, 1.0) * 0.3
    
    return {
        'motion_score': float(motion_score),
        'is_natural': motion_score > 0.3,
        'magnitude_variance': float(mag_variance),
        'direction_variance': float(avg_dir_variance),
        'reason': 'natural_motion' if motion_score > 0.3 else 'suspicious_motion'
    }


def detect_video_loop(frame_sequence, similarity_threshold=0.95):
    """
    Detect if frames are from a looped video by checking for repeating patterns
    
    Args:
        frame_sequence: List of BGR frames
        similarity_threshold: Threshold for considering frames as similar
    
    Returns:
        Dictionary with loop detection results
    """
    if len(frame_sequence) < 4:
        return {'is_loop': False, 'confidence': 0.0}
    
    # Calculate frame-to-frame similarity using histograms
    similarities = []
    
    for i in range(len(frame_sequence) - 1):
        hist1 = cv2.calcHist([frame_sequence[i]], [0, 1, 2], None, 
                             [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([frame_sequence[i+1]], [0, 1, 2], None,
                             [8, 8, 8], [0, 256, 0, 256, 0, 256])
        
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()
        
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        similarities.append(similarity)
    
    # Video loops show very high similarity between all frames
    avg_similarity = np.mean(similarities)
    similarity_std = np.std(similarities)
    
    # High average similarity + low std deviation = likely loop
    is_loop = avg_similarity > similarity_threshold and similarity_std < 0.05
    confidence = avg_similarity if is_loop else 0.0
    
    return {
        'is_loop': bool(is_loop),
        'confidence': float(confidence),
        'avg_similarity': float(avg_similarity),
        'similarity_std': float(similarity_std)
    }


def analyze_temporal_consistency(frame_sequence):
    """
    Analyze temporal consistency across frames
    
    Args:
        frame_sequence: List of BGR frames
    
    Returns:
        Dictionary with temporal consistency metrics
    """
    if len(frame_sequence) < 3:
        return {
            'is_consistent': False,
            'consistency_score': 0.0,
            'reason': 'insufficient_frames'
        }
    
    # Calculate frame differences
    frame_diffs = []
    
    for i in range(len(frame_sequence) - 1):
        diff = cv2.absdiff(frame_sequence[i], frame_sequence[i+1])
        mean_diff = np.mean(diff)
        frame_diffs.append(mean_diff)
    
    # Analyze difference pattern
    diff_variance = np.var(frame_diffs)
    diff_mean = np.mean(frame_diffs)
    
    # Natural video: moderate differences with some variance
    # Static images/loops: very low differences or very consistent
    # Synthetic: unusual patterns
    
    if diff_mean < 5:
        return {
            'is_consistent': False,
            'consistency_score': 0.0,
            'reason': 'frames_too_similar'
        }
    
    if diff_variance < 1:
        return {
            'is_consistent': False,
            'consistency_score': 0.3,
            'reason': 'unnaturally_consistent'
        }
    
    # Score based on natural variation
    consistency_score = min(diff_variance / 100, 1.0) * 0.6 + min(diff_mean / 50, 1.0) * 0.4
    
    return {
        'is_consistent': consistency_score > 0.5,
        'consistency_score': float(consistency_score),
        'diff_mean': float(diff_mean),
        'diff_variance': float(diff_variance),
        'reason': 'natural_temporal_pattern' if consistency_score > 0.5 else 'suspicious_pattern'
    }


def comprehensive_motion_analysis(frame_sequence):
    """
    Comprehensive motion analysis combining multiple methods
    
    Args:
        frame_sequence: List of BGR frames
    
    Returns:
        Dictionary with complete motion analysis
    """
    motion_result = analyze_motion_coherence(frame_sequence)
    loop_result = detect_video_loop(frame_sequence)
    temporal_result = analyze_temporal_consistency(frame_sequence)
    
    # Combine all results for final decision
    combined_score = (
        motion_result['motion_score'] * 0.4 +
        (1.0 - loop_result['confidence']) * 0.3 +
        temporal_result['consistency_score'] * 0.3
    )
    
    is_live_motion = (
        motion_result['is_natural'] and
        not loop_result['is_loop'] and
        temporal_result['is_consistent']
    )
    
    return {
        'combined_score': float(combined_score),
        'is_live_motion': bool(is_live_motion),
        'motion_coherence': motion_result,
        'loop_detection': loop_result,
        'temporal_consistency': temporal_result
    }

