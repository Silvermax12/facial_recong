"""
Performance Optimization Utilities
Provides async/threading support and optimizations for liveness detection
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Callable, Any, Tuple
import numpy as np


class PerformanceOptimizer:
    """
    Performance optimization utilities for frame processing
    Supports multi-threading, batching, and parallel processing
    """
    
    def __init__(self, max_workers: int = 4, enable_threading: bool = True):
        self.max_workers = max_workers
        self.enable_threading = enable_threading
        self.executor = ThreadPoolExecutor(max_workers=max_workers) if enable_threading else None
    
    def process_frames_parallel(
        self, 
        frames: List[np.ndarray],
        process_func: Callable[[np.ndarray], Any],
        timeout: float = 5.0
    ) -> List[Any]:
        """
        Process multiple frames in parallel using threading
        
        Args:
            frames: List of frames to process
            process_func: Function to apply to each frame
            timeout: Maximum time to wait for processing (seconds)
            
        Returns:
            List of results in same order as input frames
        """
        if not self.enable_threading or len(frames) <= 1:
            # Sequential processing
            return [process_func(frame) for frame in frames]
        
        results = [None] * len(frames)
        futures = {}
        
        # Submit all frames for processing
        for idx, frame in enumerate(frames):
            future = self.executor.submit(process_func, frame)
            futures[future] = idx
        
        # Collect results with timeout
        try:
            for future in as_completed(futures.keys(), timeout=timeout):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"[!] Frame {idx} processing error: {e}")
                    results[idx] = None
        except TimeoutError:
            print(f"[!] Processing timeout after {timeout}s")
        
        return results
    
    def batch_process_with_timing(
        self,
        frames: List[np.ndarray],
        process_func: Callable[[List[np.ndarray]], List[Any]],
        batch_size: int = 4
    ) -> Tuple[List[Any], Dict]:
        """
        Process frames in batches and measure timing
        
        Returns:
            Tuple of (results, timing_info)
        """
        start_time = time.time()
        results = []
        batch_times = []
        
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            batch_start = time.time()
            
            batch_results = process_func(batch)
            results.extend(batch_results)
            
            batch_times.append(time.time() - batch_start)
        
        total_time = time.time() - start_time
        
        timing_info = {
            'total_time': total_time,
            'avg_batch_time': np.mean(batch_times) if batch_times else 0,
            'frames_per_second': len(frames) / total_time if total_time > 0 else 0,
            'num_batches': len(batch_times)
        }
        
        return results, timing_info
    
    def optimize_frame_sequence(
        self,
        frames: List[np.ndarray],
        target_count: int = 8,
        strategy: str = 'uniform'
    ) -> List[np.ndarray]:
        """
        Optimize frame sequence by selecting best frames
        
        Args:
            frames: Input frames
            target_count: Desired number of frames
            strategy: 'uniform' (evenly spaced) or 'quality' (best quality)
            
        Returns:
            Optimized list of frames
        """
        if len(frames) <= target_count:
            return frames
        
        if strategy == 'uniform':
            # Select evenly spaced frames
            indices = np.linspace(0, len(frames) - 1, target_count, dtype=int)
            return [frames[i] for i in indices]
        
        elif strategy == 'quality':
            # Select frames with best quality (based on sharpness)
            from frame_quality import FrameQualityAnalyzer
            analyzer = FrameQualityAnalyzer()
            
            quality_scores = []
            for frame in frames:
                _, metrics = analyzer.analyze_and_enhance(frame)
                quality_scores.append(metrics['frame_quality_score'])
            
            # Get indices of top quality frames
            top_indices = np.argsort(quality_scores)[-target_count:]
            top_indices = sorted(top_indices)  # Maintain temporal order
            
            return [frames[i] for i in top_indices]
        
        return frames[:target_count]
    
    def async_frame_capture(
        self,
        capture_func: Callable[[], np.ndarray],
        num_frames: int,
        delay_ms: int = 100,
        callback: Callable[[int, np.ndarray], None] = None
    ) -> List[np.ndarray]:
        """
        Asynchronously capture frames with optional callback
        
        Args:
            capture_func: Function that captures a single frame
            num_frames: Number of frames to capture
            delay_ms: Delay between captures in milliseconds
            callback: Optional callback(frame_idx, frame) for each captured frame
            
        Returns:
            List of captured frames
        """
        frames = []
        
        for i in range(num_frames):
            frame = capture_func()
            frames.append(frame)
            
            if callback:
                callback(i, frame)
            
            if i < num_frames - 1:
                time.sleep(delay_ms / 1000.0)
        
        return frames
    
    def measure_processing_speed(
        self,
        process_func: Callable[[np.ndarray], Any],
        test_frame: np.ndarray,
        iterations: int = 10
    ) -> Dict:
        """
        Benchmark processing speed
        
        Returns:
            Dictionary with timing statistics
        """
        times = []
        
        for _ in range(iterations):
            start = time.time()
            process_func(test_frame)
            times.append(time.time() - start)
        
        return {
            'avg_time': np.mean(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'std_time': np.std(times),
            'fps': 1.0 / np.mean(times) if np.mean(times) > 0 else 0
        }
    
    def cleanup(self):
        """Cleanup executor resources"""
        if self.executor:
            self.executor.shutdown(wait=False)


class FrameBuffer:
    """
    Thread-safe circular buffer for frame storage
    Useful for continuous capture scenarios
    """
    
    def __init__(self, max_size: int = 30):
        self.max_size = max_size
        self.buffer = []
        self.lock = threading.Lock()
    
    def add(self, frame: np.ndarray, metadata: Dict = None):
        """Add frame to buffer (thread-safe)"""
        with self.lock:
            if len(self.buffer) >= self.max_size:
                self.buffer.pop(0)
            
            self.buffer.append({
                'frame': frame,
                'timestamp': time.time(),
                'metadata': metadata or {}
            })
    
    def get_latest(self, n: int = 1) -> List[Dict]:
        """Get latest N frames"""
        with self.lock:
            return self.buffer[-n:] if len(self.buffer) >= n else self.buffer.copy()
    
    def get_all(self) -> List[Dict]:
        """Get all frames in buffer"""
        with self.lock:
            return self.buffer.copy()
    
    def clear(self):
        """Clear buffer"""
        with self.lock:
            self.buffer.clear()
    
    def size(self) -> int:
        """Get current buffer size"""
        with self.lock:
            return len(self.buffer)


def optimize_capture_settings(
    target_fps: int = 10,
    target_frame_count: int = 8,
    available_time_budget: float = 2.0
) -> Dict:
    """
    Calculate optimal capture settings based on constraints
    
    Args:
        target_fps: Desired frames per second
        target_frame_count: Number of frames to capture
        available_time_budget: Maximum time available for capture (seconds)
        
    Returns:
        Dictionary with optimized settings
    """
    # Calculate delay between frames
    delay_ms = int(1000 / target_fps)
    
    # Calculate total capture time
    total_time = (target_frame_count - 1) * (delay_ms / 1000.0)
    
    # Adjust if exceeds time budget
    if total_time > available_time_budget:
        # Reduce frame count
        max_frames = int(available_time_budget * target_fps) + 1
        actual_frame_count = min(target_frame_count, max_frames)
        
        # Recalculate delay
        if actual_frame_count > 1:
            delay_ms = int((available_time_budget / (actual_frame_count - 1)) * 1000)
        
        return {
            'fps': target_fps,
            'frame_count': actual_frame_count,
            'delay_ms': delay_ms,
            'total_time': available_time_budget,
            'adjusted': True,
            'warning': f'Reduced frame count from {target_frame_count} to {actual_frame_count}'
        }
    
    return {
        'fps': target_fps,
        'frame_count': target_frame_count,
        'delay_ms': delay_ms,
        'total_time': total_time,
        'adjusted': False
    }


def cpu_optimize_inference(use_mkldnn: bool = True):
    """
    Optimize PyTorch for CPU inference
    
    Args:
        use_mkldnn: Enable MKL-DNN optimizations (Intel CPUs)
    """
    import torch
    
    # Set number of threads
    num_threads = min(4, torch.get_num_threads())
    torch.set_num_threads(num_threads)
    
    # Enable MKL-DNN if available
    if use_mkldnn and hasattr(torch.backends, 'mkldnn'):
        torch.backends.mkldnn.enabled = True
    
    print(f"[+] CPU optimization: {num_threads} threads, MKL-DNN: {use_mkldnn}")
    
    return {
        'num_threads': num_threads,
        'mkldnn_enabled': use_mkldnn and hasattr(torch.backends, 'mkldnn')
    }

