import random
import secrets
import time
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from mouth_detection import MouthDetector


class LivenessChallengeManager:
    """
    Interactive liveness challenge system
    Implements random challenges (blink, smile, head turn, open mouth) to verify live user presence
    """
    
    def __init__(self):
        self.challenges = {}
        self.challenge_timeout = 30  # seconds
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mouth_detector = MouthDetector(mar_threshold=0.6)
        
    def create_challenge(self, session_id):
        """Create random liveness challenge (single challenge - legacy method)"""
        challenge_type = random.choice(['blink', 'smile', 'head_turn_left', 'head_turn_right', 'open_mouth'])
        challenge_id = secrets.token_urlsafe(16)
        
        self.challenges[challenge_id] = {
            'type': challenge_type,
            'session_id': session_id,
            'created_at': time.time(),
            'completed': False
        }
        
        instructions = {
            'blink': 'Please blink 2-3 times',
            'smile': 'Please smile naturally',
            'head_turn_left': 'Turn your head slowly to the left',
            'head_turn_right': 'Turn your head slowly to the right',
            'open_mouth': 'Please open your mouth wide'
        }
        
        return {
            'challenge_id': challenge_id,
            'type': challenge_type,
            'instruction': instructions[challenge_type],
            'expires_in': self.challenge_timeout
        }
    
    def create_challenge_sequence(self, session_id, num_challenges=None):
        """
        Create a sequence of 2-3 random challenges for enhanced security
        
        Args:
            session_id: Unique session identifier
            num_challenges: Number of challenges (2-3), if None will be random
            
        Returns:
            Dictionary with challenge sequence info
        """
        if num_challenges is None:
            num_challenges = random.randint(2, 3)
        
        num_challenges = max(2, min(3, num_challenges))  # Ensure 2-3 range
        
        # All available challenge types
        all_types = ['blink', 'smile', 'head_turn_left', 'head_turn_right', 'open_mouth']
        
        # Randomly select unique challenges
        selected_types = random.sample(all_types, num_challenges)
        
        sequence_id = secrets.token_urlsafe(16)
        
        instructions = {
            'blink': 'Please blink 2-3 times',
            'smile': 'Please smile naturally',
            'head_turn_left': 'Turn your head slowly to the left',
            'head_turn_right': 'Turn your head slowly to the right',
            'open_mouth': 'Please open your mouth wide'
        }
        
        challenges = []
        for challenge_type in selected_types:
            challenge_id = secrets.token_urlsafe(16)
            
            self.challenges[challenge_id] = {
                'type': challenge_type,
                'session_id': session_id,
                'sequence_id': sequence_id,
                'created_at': time.time(),
                'completed': False
            }
            
            challenges.append({
                'challenge_id': challenge_id,
                'type': challenge_type,
                'instruction': instructions[challenge_type]
            })
        
        return {
            'sequence_id': sequence_id,
            'challenges': challenges,
            'total_challenges': len(challenges),
            'expires_in': self.challenge_timeout
        }
    
    def verify_challenge(self, challenge_id, response_frames):
        """Verify user completed the challenge"""
        challenge = self.challenges.get(challenge_id)
        if not challenge:
            return {'success': False, 'error': 'challenge_not_found'}
        
        if time.time() - challenge['created_at'] > self.challenge_timeout:
            return {'success': False, 'error': 'challenge_expired'}
        
        # Verify based on challenge type
        if challenge['type'] == 'blink':
            return self._verify_blink_challenge(response_frames)
        elif challenge['type'] == 'smile':
            return self._verify_smile_challenge(response_frames)
        elif challenge['type'] in ['head_turn_left', 'head_turn_right']:
            return self._verify_head_turn_challenge(response_frames, challenge['type'])
        elif challenge['type'] == 'open_mouth':
            return self._verify_open_mouth_challenge(response_frames)
        
        return {'success': False, 'error': 'unknown_challenge_type'}
    
    def _verify_blink_challenge(self, frames):
        """Verify user blinked 2-3 times"""
        blink_count = 0
        prev_ear = None
        blink_threshold = 0.23
        
        with self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
            for frame in frames:
                if isinstance(frame, bytes):
                    npimg = np.frombuffer(frame, np.uint8)
                    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
                else:
                    img = frame
                
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)
                
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    ear = self._calculate_eye_aspect_ratio(landmarks, img.shape[1], img.shape[0])
                    
                    if prev_ear is not None:
                        # Detect blink (EAR drops below threshold then rises)
                        if prev_ear > blink_threshold and ear < blink_threshold:
                            blink_count += 1
                    
                    prev_ear = ear
        
        success = 2 <= blink_count <= 5
        return {
            'success': success,
            'blink_count': blink_count,
            'expected_range': '2-3 blinks'
        }
    
    def _verify_smile_challenge(self, frames):
        """Verify user smiled"""
        smile_detected = False
        max_mouth_width = 0
        
        with self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
            for frame in frames:
                if isinstance(frame, bytes):
                    npimg = np.frombuffer(frame, np.uint8)
                    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
                else:
                    img = frame
                
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)
                
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    mouth_width = self._calculate_mouth_width(landmarks, img.shape[1], img.shape[0])
                    max_mouth_width = max(max_mouth_width, mouth_width)
                    
                    # Smile threshold (mouth width ratio)
                    if mouth_width > 0.5:
                        smile_detected = True
        
        return {
            'success': smile_detected,
            'max_mouth_width': float(max_mouth_width),
            'threshold': 0.5
        }
    
    def _verify_head_turn_challenge(self, frames, direction):
        """Verify user turned head left or right"""
        face_angles = []
        
        with self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
            for frame in frames:
                if isinstance(frame, bytes):
                    npimg = np.frombuffer(frame, np.uint8)
                    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
                else:
                    img = frame
                
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)
                
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    angle = self._calculate_head_angle(landmarks, img.shape[1], img.shape[0])
                    face_angles.append(angle)
        
        if len(face_angles) < 2:
            return {'success': False, 'error': 'insufficient_frames'}
        
        # Check if head turned in correct direction
        angle_change = face_angles[-1] - face_angles[0]
        
        if direction == 'head_turn_left':
            success = angle_change < -10  # Negative angle = left turn
        else:  # head_turn_right
            success = angle_change > 10  # Positive angle = right turn
        
        return {
            'success': success,
            'angle_change': float(angle_change),
            'direction_detected': 'left' if angle_change < 0 else 'right'
        }
    
    def _verify_open_mouth_challenge(self, frames):
        """Verify user opened their mouth"""
        open_mouth_count = 0
        mar_values = []
        
        for frame in frames:
            if isinstance(frame, bytes):
                npimg = np.frombuffer(frame, np.uint8)
                img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            else:
                img = frame
            
            # Detect open mouth
            is_open, mar, _ = self.mouth_detector.detect_open_mouth(img)
            
            if is_open:
                open_mouth_count += 1
            mar_values.append(mar)
        
        # Consider success if mouth was open in at least 30% of frames
        total_frames = len(frames)
        open_percentage = (open_mouth_count / total_frames * 100) if total_frames > 0 else 0
        max_mar = max(mar_values) if mar_values else 0.0
        
        success = open_percentage >= 30.0 and max_mar > self.mouth_detector.mar_threshold
        
        return {
            'success': success,
            'open_mouth_count': open_mouth_count,
            'total_frames': total_frames,
            'open_percentage': open_percentage,
            'max_mar': float(max_mar),
            'threshold': self.mouth_detector.mar_threshold
        }
    
    def _calculate_eye_aspect_ratio(self, landmarks, img_w, img_h):
        """Calculate Eye Aspect Ratio (EAR) for blink detection"""
        # Left eye indices
        left_eye = [33, 160, 158, 133, 153, 144]
        # Right eye indices
        right_eye = [362, 385, 387, 263, 373, 380]
        
        def ear(eye_indices):
            points = [(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in eye_indices]
            # Vertical distances
            v1 = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
            v2 = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
            # Horizontal distance
            h = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
            return (v1 + v2) / (2.0 * h) if h > 0 else 1.0
        
        left_ear = ear(left_eye)
        right_ear = ear(right_eye)
        return (left_ear + right_ear) / 2.0
    
    def _calculate_mouth_width(self, landmarks, img_w, img_h):
        """Calculate mouth width ratio for smile detection"""
        # Mouth corner indices
        left_corner = landmarks[61]
        right_corner = landmarks[291]
        
        # Face width (approximate using cheek landmarks)
        left_cheek = landmarks[234]
        right_cheek = landmarks[454]
        
        mouth_width = abs(right_corner.x - left_corner.x) * img_w
        face_width = abs(right_cheek.x - left_cheek.x) * img_w
        
        return mouth_width / face_width if face_width > 0 else 0
    
    def _calculate_head_angle(self, landmarks, img_w, img_h):
        """Calculate head rotation angle for head turn detection"""
        # Use nose tip and face center to estimate angle
        nose_tip = landmarks[1]
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        
        # Calculate face center
        center_x = (left_eye.x + right_eye.x) / 2
        
        # Angle based on nose position relative to center
        offset = (nose_tip.x - center_x) * img_w
        # Convert to approximate angle in degrees
        angle = np.arctan(offset / 100) * 180 / np.pi
        
        return angle
    
    def cleanup_expired_challenges(self):
        """Remove expired challenges"""
        current_time = time.time()
        expired = [cid for cid, challenge in self.challenges.items()
                   if current_time - challenge['created_at'] > self.challenge_timeout]
        for cid in expired:
            del self.challenges[cid]

