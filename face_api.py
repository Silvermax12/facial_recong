from flask import Flask, request, jsonify
import face_recognition
import torch, cv2, numpy as np, os, time
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables from .env file
load_dotenv()
from Model import DeePixBiS
import json
import io
try:
    import yaml
except Exception:
    yaml = None
try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
from face_auth_utils import (
    load_deeppixbis,
    load_enhanced_model,
    compute_liveness_enhanced,
    compute_liveness,
    MATCH_THRESHOLD
)
from database_utils import db_manager
from cloudinary_utils import cloudinary_manager
from auth_backend_utils import (
    init_jwt,
    authenticate_user,
    register_user,
    token_required,
    admin_required,
    user_allowed_required
)
from flask_jwt_extended import get_jwt_identity
from enhanced_liveness_utils import (
    TemporalConsistencyChecker,
    EnhancedBlinkDetector,
    StaticImageDetector,
    InjectionAttackDetector
)
from face_challenge import LivenessChallengeManager
from motion_analysis import comprehensive_motion_analysis
from adaptive_threshold import AdaptiveThresholdManager
from frame_quality import FrameQualityAnalyzer, EnhancedPreprocessor, get_frame_statistics
from verification_logger import get_logger

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize JWT
jwt = init_jwt(app)

# ---- Config loading ----
DEFAULT_CONFIG = {
    "ensemble": {
        "weights": {"model": 0.5, "frequency": 0.2, "temporal": 0.2, "blink_pattern": 0.1},
        "min_signals_required": 2
    },
    "thresholds": {
        "liveness": 0.55,
        "spectrum": 0.62,
        "blink_ear": 0.23
    },
    "profiles": {
        "default": {"model_type": "ensemble"}
    }
}

def load_config(path="liveness_config.yaml"):
    if yaml is None or not os.path.exists(path):
        return DEFAULT_CONFIG
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        # Merge with defaults
        cfg = DEFAULT_CONFIG.copy()
        cfg.update({k: v for k, v in (data or {}).items() if v is not None})
        return cfg
    except Exception:
        return DEFAULT_CONFIG

CONFIG = load_config()
_FLASH_CHALLENGES = {}
try:
    from secrets import token_urlsafe
except Exception:
    token_urlsafe = None

# Load enhanced models and analyzers
# Note: Face encodings are now loaded from database on-demand
model, model_type = load_enhanced_model(model_type=CONFIG.get("profiles",{}).get("default",{}).get("model_type","ensemble"), device=device)

# Check if database and Cloudinary are configured
print(f"[+] Database configured: {db_manager.connection is not None}")
print(f"[+] Cloudinary configured: {cloudinary_manager.configured}")

# Session-based detection components (FIXED: CRITICAL-3, CRITICAL-4)
_session_detectors = {}  # Store per-session detectors
_session_timeout = 300  # 5 minutes

# Initialize challenge manager and adaptive threshold manager
challenge_manager = LivenessChallengeManager()
threshold_manager = AdaptiveThresholdManager()

# Initialize verification logger
verification_logger = get_logger(CONFIG)

def get_or_create_session_detectors(session_id):
    """Get or create session-specific detectors"""
    cleanup_old_sessions()  # Remove expired sessions
    if session_id not in _session_detectors:
        _session_detectors[session_id] = {
            'temporal': TemporalConsistencyChecker(buffer_size=15),
            'blink': EnhancedBlinkDetector(),
            'injection': InjectionAttackDetector(),
            'created_at': time.time()
        }
    return _session_detectors[session_id]

def cleanup_old_sessions():
    """Remove sessions older than timeout"""
    current_time = time.time()
    expired = [sid for sid, data in _session_detectors.items() 
               if current_time - data['created_at'] > _session_timeout]
    for sid in expired:
        del _session_detectors[sid]

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


# ========================================
# Authentication Endpoints
# ========================================

@app.route("/auth/signup", methods=["POST"])
def signup():
    """User registration endpoint"""
    data = request.get_json() or {}
    username = data.get('username', '').strip()
    password = data.get('password', '')
    display_name = data.get('display_name', '').strip()
    
    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400
    
    result = register_user(username, password, display_name or username)
    
    if result['success']:
        return jsonify(result), 201
    else:
        return jsonify(result), 400


@app.route("/auth/login", methods=["POST"])
def login():
    """User login endpoint"""
    data = request.get_json() or {}
    username = data.get('username', '').strip()
    password = data.get('password', '')
    
    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400
    
    result = authenticate_user(username, password)
    
    if result['success']:
        return jsonify(result), 200
    else:
        status_code = 403 if result.get('unauthorized') else 401
        return jsonify(result), status_code


@app.route("/admin/login", methods=["POST"])
def admin_login():
    """Admin login endpoint"""
    data = request.get_json() or {}
    username = data.get('username', '').strip()
    password = data.get('password', '')
    
    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400
    
    result = authenticate_user(username, password)
    
    if result['success']:
        # Verify admin role
        if result.get('role') != 'admin':
            return jsonify({
                "success": False,
                "message": "Admin access required"
            }), 403
        
        return jsonify(result), 200
    else:
        status_code = 403 if result.get('unauthorized') else 401
        return jsonify(result), status_code


@app.route("/check_access", methods=["GET"])
@token_required
def check_access():
    """Check if user has access (not blocked)"""
    username = get_jwt_identity()
    
    if not db_manager.is_user_allowed(username):
        return jsonify({
            "allowed": False,
            "message": "Account blocked"
        }), 403
    
    return jsonify({
        "allowed": True,
        "username": username,
        "message": "Access granted"
    }), 200


@app.route("/admin/users", methods=["GET"])
@admin_required
def get_users():
    """Get all users (admin only)"""
    users = db_manager.get_all_auth_users()
    
    # Format users for response
    formatted_users = []
    for user in users:
        formatted_users.append({
            "id": user['username'],
            "username": user['username'],
            "display_name": user.get('display_name', user['username']),
            "role": user['role'],
            "allowed": user['allowed'],
            "created_at": user['created_at'].isoformat() if user.get('created_at') else None,
            "last_login": user['last_login'].isoformat() if user.get('last_login') else None
        })
    
    return jsonify({"users": formatted_users}), 200


@app.route("/admin/users/<username>", methods=["PATCH"])
@admin_required
def toggle_user_status(username):
    """Toggle user allowed status (admin only)"""
    data = request.get_json() or {}
    allowed = data.get('allowed')
    
    if allowed is None:
        return jsonify({"error": "allowed field required"}), 400
    
    # Prevent admins from blocking themselves
    admin_username = get_jwt_identity()
    if username == admin_username:
        return jsonify({"error": "Cannot modify your own status"}), 400
    
    # Check if user exists
    user = db_manager.get_auth_user(username)
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    # Prevent blocking other admins
    if user['role'] == 'admin' and not allowed:
        return jsonify({"error": "Cannot block admin users"}), 400
    
    # Update status
    success = db_manager.set_user_allowed(username, allowed)
    
    if not success:
        return jsonify({"error": "Failed to update user status"}), 500
    
    # Return updated user info
    updated_user = db_manager.get_auth_user(username)
    action = "allowed" if allowed else "blocked"
    
    return jsonify({
        "message": f"User {username} {action} successfully",
        "user": {
            "id": updated_user['username'],
            "username": updated_user['username'],
            "display_name": updated_user.get('display_name', updated_user['username']),
            "role": updated_user['role'],
            "allowed": updated_user['allowed'],
            "created_at": updated_user['created_at'].isoformat() if updated_user.get('created_at') else None,
            "last_login": updated_user['last_login'].isoformat() if updated_user.get('last_login') else None
        }
    }), 200


@app.route("/admin/users/<username>/skip-challenges", methods=["PATCH"])
@admin_required
def toggle_user_skip_challenges(username):
    """Toggle user challenge skipping (admin only)"""
    data = request.get_json() or {}
    skip_challenges = data.get('skip_challenges')

    if skip_challenges is None:
        return jsonify({"error": "skip_challenges field required"}), 400

    # Check if user exists
    user = db_manager.get_auth_user(username)
    if not user:
        return jsonify({"error": "User not found"}), 404

    # Update challenge skipping setting
    success = db_manager.set_user_skip_challenges(username, skip_challenges)

    if not success:
        return jsonify({"error": "Failed to update user challenge settings"}), 500

    # Return updated user info
    updated_user = db_manager.get_auth_user(username)
    action = "enabled" if skip_challenges else "disabled"

    return jsonify({
        "message": f"Challenge skipping {action} for user {username}",
        "user": {
            "id": updated_user['username'],
            "username": updated_user['username'],
            "display_name": updated_user.get('display_name', updated_user['username']),
            "role": updated_user['role'],
            "allowed": updated_user['allowed'],
            "skip_challenges": updated_user['skip_challenges'],
            "created_at": updated_user['created_at'].isoformat() if updated_user.get('created_at') else None,
            "last_login": updated_user['last_login'].isoformat() if updated_user.get('last_login') else None
        }
    }), 200


# ========================================
# Face Recognition Endpoints (Protected)
# ========================================

@app.route("/verify", methods=["POST"])
@token_required
@user_allowed_required
def verify():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "no image"}), 400

    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    locs = face_recognition.face_locations(rgb)
    encs = face_recognition.face_encodings(rgb, locs)
    if not encs:
        return jsonify({"status": "no_face"})

    # Username is now REQUIRED for verification - users must enroll first
    username = (request.form.get("username") or request.args.get("username") or "").strip()
    if not username:
        return jsonify({"error": "username required for verification - please enroll first"}), 400

    face_enc = encs[0]

    # Load user's face encodings from database
    user_face_data = db_manager.get_user_encodings(username)
    if not user_face_data:
        # User has not enrolled their face yet
        return jsonify({
            "error": "user not enrolled",
            "message": f"User '{username}' has not enrolled their face yet. Please enroll before verification.",
            "username": username,
            "enrollment_required": True
        }), 403

    # Extract encodings and URLs
    cand_encs = [encoding for encoding, url in user_face_data]
    cand_names = [username] * len(cand_encs)

    # User is enrolled - perform face matching
    dists = face_recognition.face_distance(cand_encs, face_enc)
    idx = int(np.argmin(dists))
    dmin = float(dists[idx])
    name = cand_names[idx] if dmin < MATCH_THRESHOLD else "Unknown"

    # Check if this appears to be an imposter attempt
    is_imposter = (name == "Unknown" and len(cand_encs) > 0)

    # Update last verified timestamp
    db_manager.update_last_verified(username)

    top, right, bottom, left = locs[0]
    h, w = frame.shape[:2]
    top = max(0, top); right = min(w, right); bottom = min(h, bottom); left = max(0, left)
    crop = frame[top:bottom, left:right]

    # Create static detector locally to avoid global variable issues
    static_detector = StaticImageDetector()

    # Use enhanced liveness detection (v1 response shape)
    score, is_live, detailed_scores = compute_liveness_enhanced(
        crop, model, device, model_type=model_type,
        use_frequency_analysis=True,
        use_temporal_check=True,
        temporal_checker=temporal_checker,
        blink_detector=blink_detector,
        static_detector=static_detector,
        image_path=None,  # No file path for uploaded images
        weights=CONFIG.get("ensemble",{}).get("weights")
    )

    # INJECTION ATTACK DETECTION
    injection_score = injection_detector.detect_injection_attack(frame, request_timestamp=time.time())
    detailed_scores['injection_attack'] = injection_score

    # If injection attack is highly likely, mark as not live
    if injection_score > 0.8:
        is_live = False
        detailed_scores['injection_blocked'] = True

    # MANDATORY TEMPORAL REQUIREMENT: Single images are suspicious
    # Require temporal evidence or very strong liveness score for single images
    has_temporal_evidence = (
        len(temporal_checker.frame_buffer) >= 3 and
        detailed_scores.get('temporal', 0) > 0.6
    )

    has_blink_evidence = (
        len(getattr(blink_detector, 'blink_history', [])) >= 2 and
        detailed_scores.get('blink_pattern', 0) > 0.6
    )

    # For single images without temporal context, require much higher liveness score
    if not (has_temporal_evidence or has_blink_evidence):
        # Single static image - apply stricter threshold
        single_image_threshold = CONFIG.get("thresholds",{}).get("liveness", 0.70) + 0.15  # Add 15% penalty
        is_live = score >= single_image_threshold
        if 'static_image_penalty' not in detailed_scores:
            detailed_scores['static_image_penalty'] = float(score >= single_image_threshold)

    # Add frame to temporal analysis for future requests
    temporal_checker.add_frame(crop, time.time())

    # Analyze blink patterns if face mesh is available
    try:
        import mediapipe as mp
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1
        )
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            blink_detector.detect_blink_enhanced(rgb, face_mesh)
        face_mesh.close()
    except Exception:
        pass  # Skip blink analysis if MediaPipe not available

    response = {
        "user": name,
        "distance": dmin,
        "liveness_score": score,
        "is_live": is_live,
        "detection_method": model_type,
        "detailed_scores": detailed_scores,
        "is_imposter": is_imposter,
        "username_provided": username
    }

    # Add imposter alert if detected
    if is_imposter:
        response["imposter_alert"] = True
        response["message"] = f"Imposter detected! Face does not match enrolled user '{username}'"

    return jsonify(response)


@app.route("/v2/verify", methods=["POST"])
@token_required
@user_allowed_required
def verify_v2():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "no image"}), 400

    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    locs = face_recognition.face_locations(rgb)
    encs = face_recognition.face_encodings(rgb, locs)
    if not encs:
        return jsonify({"status": "no_face"})

    # Username is now REQUIRED for verification - users must enroll first
    username = (request.form.get("username") or request.args.get("username") or "").strip()
    if not username:
        return jsonify({"error": "username required for verification - please enroll first"}), 400

    face_enc = encs[0]

    # Load user's face encodings from database
    user_face_data = db_manager.get_user_encodings(username)
    if not user_face_data:
        # User has not enrolled their face yet
        return jsonify({
            "error": "user not enrolled",
            "message": f"User '{username}' has not enrolled their face yet. Please enroll before verification.",
            "username": username,
            "enrollment_required": True
        }), 403

    # Extract encodings and URLs
    cand_encs = [encoding for encoding, url in user_face_data]
    cand_names = [username] * len(cand_encs)

    # User is enrolled - perform face matching
    dists = face_recognition.face_distance(cand_encs, face_enc)
    idx = int(np.argmin(dists))
    dmin = float(dists[idx])
    name = cand_names[idx] if dmin < MATCH_THRESHOLD else "Unknown"

    # Check if this appears to be an imposter attempt
    is_imposter = (name == "Unknown" and len(cand_encs) > 0)

    # Update last verified timestamp
    db_manager.update_last_verified(username)

    t, r, b, l = locs[0]
    H, W = frame.shape[:2]
    t = max(0, t); r = min(W, r); b = min(H, b); l = max(0, l)
    crop = frame[t:b, l:r]

    # Determine which analyzers can run
    methods_run = ["model"]
    signals_missing = []
    use_frequency = True
    use_temporal = True
    actually_used = ["model"]
    if use_frequency:
        actually_used.append("frequency")
    if use_temporal:
        if len(temporal_checker.frame_buffer) >= 3:
            actually_used.append("temporal")
        else:
            signals_missing.append("temporal")
    if getattr(blink_detector, 'blink_history', None) and len(blink_detector.blink_history) >= 3:
        actually_used.append("blink")
    else:
        signals_missing.append("blink")

    # Create static detector locally to avoid global variable issues
    static_detector = StaticImageDetector()

    score, is_live, detailed_scores = compute_liveness_enhanced(
        crop, model, device, model_type=model_type,
        use_frequency_analysis=use_frequency,
        use_temporal_check=use_temporal,
        temporal_checker=temporal_checker,
        blink_detector=blink_detector,
        static_detector=static_detector,
        image_path=None,  # No file path for uploaded images
        weights=CONFIG.get("ensemble",{}).get("weights")
    )

    # INJECTION ATTACK DETECTION
    injection_score = injection_detector.detect_injection_attack(frame, request_timestamp=time.time())
    detailed_scores['injection_attack'] = injection_score

    # If injection attack is highly likely, mark as not live
    if injection_score > 0.8:
        is_live = False
        detailed_scores['injection_blocked'] = True

    # MANDATORY TEMPORAL REQUIREMENT: Single images are suspicious
    # Require temporal evidence or very strong liveness score for single images
    has_temporal_evidence = (
        len(temporal_checker.frame_buffer) >= 3 and
        detailed_scores.get('temporal', 0) > 0.6
    )

    has_blink_evidence = (
        len(getattr(blink_detector, 'blink_history', [])) >= 2 and
        detailed_scores.get('blink_pattern', 0) > 0.6
    )

    # For single images without temporal context, require much higher liveness score
    if not (has_temporal_evidence or has_blink_evidence):
        # Single static image - apply stricter threshold
        single_image_threshold = CONFIG.get("thresholds",{}).get("liveness", 0.70) + 0.15  # Add 15% penalty
        is_live = score >= single_image_threshold
        detailed_scores['static_image_penalty'] = float(is_live)

    temporal_checker.add_frame(crop, time.time())

    # Confidence heuristic based on score margin and number of methods_run
    margin = abs(score - CONFIG.get("thresholds",{}).get("liveness", 0.55))
    if score >= CONFIG.get("thresholds",{}).get("liveness", 0.55) and margin > 0.15 and len(methods_run) >= CONFIG.get("ensemble",{}).get("min_signals_required",2):
        confidence = "high"
    elif margin > 0.07:
        confidence = "medium"
    else:
        confidence = "low"

    # Sanitize JSON types to avoid numpy/torch types leaking into responses
    def _to_float(x):
        try:
            return float(x)
        except Exception:
            return None
    sanitized_scores = {}
    for k, v in (detailed_scores or {}).items():
        fv = _to_float(v)
        if fv is not None:
            sanitized_scores[str(k)] = fv

    response = {
        "api_version": "2.0",
        "user": str(name),
        "distance": _to_float(dmin),
        "liveness_score": _to_float(score),
        "is_live": bool(is_live),
        "confidence": str(confidence),
        "detection_method": str(model_type),
        "methods_run": [str(m) for m in actually_used],
        "signals_missing": [str(s) for s in signals_missing],
        "detailed_scores": sanitized_scores,
        "device_profile": "default",
        "is_imposter": bool(is_imposter),
        "username_provided": str(username)
    }

    # Add imposter alert if detected
    if is_imposter:
        response["imposter_alert"] = True
        response["message"] = f"Imposter detected! Face does not match enrolled user '{username}'"

    return jsonify(response)


@app.route("/v2/flash/challenge", methods=["POST"])  # Issue randomized flash pattern challenge
def flash_challenge():
    # Create a simple randomized pattern: list of {color, duration_ms}
    now = int(time.time())
    pattern = [
        {"color": "white", "duration_ms": 160},
        {"color": "black", "duration_ms": 120},
        {"color": "white", "duration_ms": 200},
        {"color": "black", "duration_ms": 140},
    ]
    cid = token_urlsafe(16) if token_urlsafe else f"cid_{now}_{np.random.randint(0, 1_000_000)}"
    _FLASH_CHALLENGES[cid] = {"pattern": pattern, "issued_at": now, "ttl": 20}
    return jsonify({
        "api_version": "2.0",
        "challenge_id": cid,
        "pattern": pattern,
        "expires_in": 20
    })


@app.route("/v2/flash/verify", methods=["POST"])  # Verify observed response to challenge
def flash_verify():
    data = request.get_json(silent=True) or {}
    challenge_id = data.get("challenge_id")
    observations = data.get("observations")  # e.g., [{"t": ms, "brightness": 0..1}]
    if not challenge_id or not isinstance(observations, list):
        return jsonify({"error": "invalid_request"}), 400
    entry = _FLASH_CHALLENGES.get(challenge_id)
    if not entry:
        return jsonify({"error": "challenge_not_found"}), 404
    if int(time.time()) - entry["issued_at"] > entry["ttl"]:
        return jsonify({"error": "challenge_expired"}), 410

    # Very simple heuristic: check that brightness toggles align with pattern count
    try:
        brightness_values = [float(o.get("brightness", 0)) for o in observations]
        if not brightness_values:
            return jsonify({"error": "no_observations"}), 400
        # Compute number of significant transitions
        transitions = sum(
            1 for i in range(1, len(brightness_values)) if abs(brightness_values[i] - brightness_values[i-1]) > 0.15
        )
        expected = len(entry["pattern"]) - 1
        # Score by closeness of transitions
        diff = abs(transitions - expected)
        flash_score = max(0.0, 1.0 - (diff / max(1.0, expected)))
        is_live_flash = flash_score > 0.6
        return jsonify({
            "api_version": "2.0",
            "challenge_id": challenge_id,
            "flash_score": flash_score,
            "is_live_flash": is_live_flash
        })
    except Exception:
        return jsonify({"error": "verification_error"}), 500


@app.route("/v3/verify/sequence", methods=["POST"])
@token_required
@user_allowed_required
def verify_sequence():
    """Multi-frame verification - MANDATORY for security (FIXED: CRITICAL-1, CRITICAL-2)"""
    files = request.files.getlist("frames")
    username = request.form.get("username", "").strip()
    session_id = request.form.get("session_id") or (token_urlsafe(16) if token_urlsafe else f"sess_{int(time.time())}")
    
    if not username:
        return jsonify({"error": "username required"}), 400
    if len(files) < 5:
        return jsonify({"error": "minimum 5 frames required"}), 400
    
    session_detectors = get_or_create_session_detectors(session_id)
    frame_results = []
    face_matched = False
    
    for idx, file in enumerate(files):
        file_bytes = file.read()
        npimg = np.frombuffer(file_bytes, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if frame is None:
            continue
            
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Face detection & matching (only first frame needs matching)
        locs = face_recognition.face_locations(rgb)
        encs = face_recognition.face_encodings(rgb, locs)
        
        if not encs:
            continue
            
        if idx == 0:  # Verify identity on first frame
            face_enc = encs[0]
            user_face_data = db_manager.get_user_encodings(username)
            if not user_face_data:
                return jsonify({"error": "user not enrolled"}), 403
            cand_encs = [encoding for encoding, url in user_face_data]
            dists = face_recognition.face_distance(cand_encs, face_enc)
            dmin = float(np.min(dists))
            if dmin >= MATCH_THRESHOLD:
                return jsonify({"error": "face not matched", "distance": dmin}), 403
            face_matched = True
            # Update last verified timestamp
            db_manager.update_last_verified(username)
        
        # Liveness detection with session detectors
        top, right, bottom, left = locs[0]
        crop = frame[max(0,top):min(frame.shape[0],bottom), 
                    max(0,left):min(frame.shape[1],right)]
        
        static_detector = StaticImageDetector()
        score, is_live, detailed = compute_liveness_enhanced(
            crop, model, device, model_type=model_type,
            temporal_checker=session_detectors['temporal'],
            blink_detector=session_detectors['blink'],
            static_detector=static_detector,
            image_path=None,
            weights=CONFIG.get("ensemble",{}).get("weights")
        )
        
        # Add frame to temporal buffer
        session_detectors['temporal'].add_frame(crop, time.time())
        
        frame_results.append({
            'frame_idx': idx,
            'liveness_score': float(score),
            'is_live': bool(is_live),
            'detailed_scores': detailed
        })
    
    # Aggregate: ALL frames must show liveness
    if len(frame_results) < 3:
        return jsonify({"error": "insufficient valid frames", "frames_processed": len(frame_results)}), 400
    
    all_live = all(r['is_live'] for r in frame_results)
    avg_score = np.mean([r['liveness_score'] for r in frame_results])
    score_variance = np.var([r['liveness_score'] for r in frame_results])
    
    # Video loops have very low variance
    if score_variance < 0.005:
        return jsonify({
            "error": "suspicious score consistency", 
            "reason": "possible video loop",
            "variance": float(score_variance)
        }), 403
    
    # Apply threshold
    threshold = CONFIG.get("thresholds", {}).get("multi_frame", 
                CONFIG.get("thresholds", {}).get("liveness", 0.75))
    passed = all_live and avg_score >= threshold
    
    return jsonify({
        "api_version": "3.0",
        "session_id": session_id,
        "success": passed,
        "username": username,
        "avg_liveness_score": float(avg_score),
        "score_variance": float(score_variance),
        "frames_analyzed": len(frame_results),
        "threshold_used": threshold,
        "all_frames_live": all_live,
        "frame_results": frame_results
    })


@app.route("/v3/challenge/create", methods=["POST"])
def create_challenge():
    """Create random liveness challenge(s) - supports single or sequence"""
    # Support both form and JSON requests
    if request.is_json:
        data = request.get_json() or {}
        session_id = data.get("session_id") or (token_urlsafe(16) if token_urlsafe else f"sess_{int(time.time())}")
        num_challenges = data.get("num_challenges")
        username = data.get("username", "").strip()
    else:
        session_id = request.form.get("session_id") or (token_urlsafe(16) if token_urlsafe else f"sess_{int(time.time())}")
        num_challenges = request.form.get("num_challenges")
        username = request.form.get("username", "").strip()
    
    # Check if user should skip challenges
    if username and db_manager.get_user_skip_challenges(username):
        print(f"[+] User {username} configured to skip challenges - returning empty challenge list")
        return jsonify({
            "challenges": [],
            "session_id": session_id,
            "skip_challenges": True,
            "message": "Challenge skipping enabled for this user"
        })
    
    # If num_challenges is provided, create a sequence
    if num_challenges:
        try:
            num_challenges = int(num_challenges)
            challenge_sequence = challenge_manager.create_challenge_sequence(session_id, num_challenges)
            return jsonify(challenge_sequence)
        except (ValueError, TypeError):
            pass  # Fall back to single challenge
    
    # Default: single challenge
    challenge = challenge_manager.create_challenge(session_id)
    return jsonify(challenge)


@app.route("/v3/challenge/verify", methods=["POST"])
def verify_challenge_response():
    """Verify challenge response"""
    challenge_id = request.form.get("challenge_id")
    if not challenge_id:
        return jsonify({"error": "challenge_id required"}), 400
    
    frames = request.files.getlist("response_frames")
    if not frames:
        return jsonify({"error": "no response frames provided"}), 400
    
    # Convert uploaded frames to images
    frame_images = []
    for f in frames:
        file_bytes = f.read()
        npimg = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if img is not None:
            frame_images.append(img)
    
    result = challenge_manager.verify_challenge(challenge_id, frame_images)
    return jsonify(result)


@app.route("/v3/verify/enhanced", methods=["POST"])
@token_required
@user_allowed_required
def verify_enhanced():
    """Enhanced verification with motion analysis and adaptive thresholds"""
    files = request.files.getlist("frames")
    username = request.form.get("username", "").strip()
    session_id = request.form.get("session_id") or (token_urlsafe(16) if token_urlsafe else f"sess_{int(time.time())}")
    
    if not username:
        return jsonify({"error": "username required"}), 400
    if len(files) < 5:
        return jsonify({"error": "minimum 5 frames required"}), 400
    
    session_detectors = get_or_create_session_detectors(session_id)
    frame_results = []
    frame_images = []  # Store for motion analysis
    
    for idx, file in enumerate(files):
        file_bytes = file.read()
        npimg = np.frombuffer(file_bytes, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if frame is None:
            continue
        
        frame_images.append(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Face detection & matching
        locs = face_recognition.face_locations(rgb)
        encs = face_recognition.face_encodings(rgb, locs)
        
        if not encs:
            continue
        
        if idx == 0:  # Verify identity on first frame
            face_enc = encs[0]
            user_face_data = db_manager.get_user_encodings(username)
            if not user_face_data:
                return jsonify({"error": "user not enrolled"}), 403
            cand_encs = [encoding for encoding, url in user_face_data]
            dists = face_recognition.face_distance(cand_encs, face_enc)
            dmin = float(np.min(dists))
            if dmin >= MATCH_THRESHOLD:
                return jsonify({"error": "face not matched", "distance": dmin}), 403
            # Update last verified timestamp
            db_manager.update_last_verified(username)
        
        # Liveness detection
        top, right, bottom, left = locs[0]
        crop = frame[max(0,top):min(frame.shape[0],bottom), 
                    max(0,left):min(frame.shape[1],right)]
        
        static_detector = StaticImageDetector()
        score, is_live, detailed = compute_liveness_enhanced(
            crop, model, device, model_type=model_type,
            temporal_checker=session_detectors['temporal'],
            blink_detector=session_detectors['blink'],
            static_detector=static_detector,
            weights=CONFIG.get("ensemble",{}).get("weights")
        )
        
        session_detectors['temporal'].add_frame(crop, time.time())
        
        frame_results.append({
            'frame_idx': idx,
            'liveness_score': float(score),
            'is_live': bool(is_live)
        })
    
    if len(frame_results) < 3:
        return jsonify({"error": "insufficient valid frames"}), 400
    
    # Motion coherence analysis
    if len(frame_images) >= 2:
        motion_result = comprehensive_motion_analysis(frame_images)
        
        if not motion_result['is_live_motion']:
            return jsonify({
                "error": "unnatural_motion_detected",
                "motion_analysis": motion_result
            }), 403
    else:
        motion_result = {'combined_score': 0.0, 'is_live_motion': False}
    
    # Calculate metrics
    all_live = all(r['is_live'] for r in frame_results)
    avg_score = np.mean([r['liveness_score'] for r in frame_results])
    score_variance = np.var([r['liveness_score'] for r in frame_results])
    
    # Adaptive threshold based on context
    risk_context = {
        'device_trusted': request.form.get('device_trusted', 'false').lower() == 'true',
        'device_rooted': request.form.get('device_rooted', 'false').lower() == 'true',
        'user_login_count': int(request.form.get('user_login_count', '0')),
        'recent_failed_attempts': int(request.form.get('recent_failed_attempts', '0')),
        'timestamp': time.time()
    }
    
    threshold = threshold_manager.get_threshold(risk_context)
    risk_level = threshold_manager.get_risk_level(risk_context)
    
    passed = all_live and avg_score >= threshold and motion_result['is_live_motion']
    
    return jsonify({
        "api_version": "3.0-enhanced",
        "session_id": session_id,
        "success": passed,
        "username": username,
        "avg_liveness_score": float(avg_score),
        "score_variance": float(score_variance),
        "threshold_used": threshold,
        "risk_level": risk_level,
        "motion_analysis": motion_result,
        "frames_analyzed": len(frame_results)
    })


@app.route("/v3/verify/guided", methods=["POST"])
@token_required
@user_allowed_required
def verify_guided():
    """
    Guided verification with real-time quality feedback and instructions
    Designed for interactive UI with circular overlay and step-by-step guidance
    """
    files = request.files.getlist("frames")
    username = request.form.get("username", "").strip()
    session_id = request.form.get("session_id") or (token_urlsafe(16) if token_urlsafe else f"sess_{int(time.time())}")
    step = request.form.get("step", "quality_check")  # quality_check, liveness_verify, complete
    
    if not username:
        return jsonify({"error": "username required"}), 400
    
    # Initialize quality analyzer with config
    quality_config = {
        'blur_threshold': CONFIG.get('quality', {}).get('blur_threshold', 100.0),
        'min_brightness': CONFIG.get('quality', {}).get('min_brightness', 40),
        'max_brightness': CONFIG.get('quality', {}).get('max_brightness', 220),
        'target_brightness': CONFIG.get('quality', {}).get('target_brightness', 128),
        'min_face_size': CONFIG.get('quality', {}).get('min_face_size', 80),
        'max_faces_allowed': CONFIG.get('quality', {}).get('max_faces_allowed', 1)
    }
    quality_analyzer = FrameQualityAnalyzer(quality_config)
    
    # Step 1: Quality Check (can be called with 1+ frames for real-time feedback)
    if step == "quality_check":
        if not files:
            return jsonify({
                "error": "no_frames",
                "instruction": "Position your face in the circle",
                "status": "waiting"
            }), 400
        
        # Analyze first/latest frame
        file_bytes = files[0].read()
        npimg = np.frombuffer(file_bytes, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({
                "error": "invalid_frame",
                "instruction": "Camera error - please retry",
                "status": "error"
            }), 400
        
        enhanced_frame, metrics = quality_analyzer.analyze_and_enhance(frame)
        
        # Sanitize metrics for JSON serialization (convert numpy types to native Python)
        def sanitize_value(v):
            if isinstance(v, (np.integer, np.floating)):
                return float(v)
            elif isinstance(v, np.bool_):
                return bool(v)
            elif isinstance(v, (list, tuple)):
                return [sanitize_value(item) for item in v]
            elif isinstance(v, dict):
                return {k: sanitize_value(val) for k, val in v.items()}
            return v
        
        sanitized_metrics = {k: sanitize_value(v) for k, v in metrics.items()}
        
        # Determine feedback and instruction
        feedback = {
            "api_version": "3.0-guided",
            "step": "quality_check",
            "quality_score": float(metrics['frame_quality_score']),
            "metrics": sanitized_metrics,
            "passed": bool(len(metrics['rejection_reasons']) == 0 and metrics['frame_quality_score'] >= 60),
            "instruction": "",
            "guidance": {},
            "visual_feedback": "neutral"
        }
        
        # Generate specific instructions based on issues
        if metrics['num_faces'] == 0:
            feedback['instruction'] = "Position your face in the circle"
            feedback['guidance'] = {"type": "face_alignment", "severity": "critical"}
            feedback['visual_feedback'] = "red"
        elif metrics['num_faces'] > 1:
            feedback['instruction'] = "Multiple faces detected - please ensure only you are visible"
            feedback['guidance'] = {"type": "multiple_faces", "severity": "critical"}
            feedback['visual_feedback'] = "red"
        elif metrics['is_blurry']:
            feedback['instruction'] = "Hold steady - image is too blurry"
            feedback['guidance'] = {"type": "blur", "severity": "high", "blur_score": float(metrics['blur_score'])}
            feedback['visual_feedback'] = "orange"
        elif metrics['lighting_quality'] == 'too_dark':
            feedback['instruction'] = "Increase lighting - face is too dark"
            feedback['guidance'] = {"type": "lighting", "severity": "high", "issue": "too_dark"}
            feedback['visual_feedback'] = "orange"
        elif metrics['lighting_quality'] == 'too_bright':
            feedback['instruction'] = "Reduce brightness - face is overexposed"
            feedback['guidance'] = {"type": "lighting", "severity": "high", "issue": "too_bright"}
            feedback['visual_feedback'] = "orange"
        else:
            feedback['instruction'] = "Perfect! Hold steady for verification"
            feedback['guidance'] = {"type": "ready", "severity": "none"}
            feedback['visual_feedback'] = "green"
        
        return jsonify(feedback)
    
    # Step 2: Liveness Verification with Challenge
    elif step == "liveness_verify":
        # Helper function to sanitize all numpy types
        def sanitize_for_json(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: sanitize_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [sanitize_for_json(item) for item in obj]
            return obj
        
        if len(files) < 5:
            return jsonify({
                "error": "insufficient_frames",
                "instruction": "Continue capturing frames...",
                "frames_required": 5,
                "frames_received": len(files)
            }), 400
        
        session_detectors = get_or_create_session_detectors(session_id)
        
        # Process all frames
        frame_images = []
        enhanced_frames = []
        quality_metrics_list = []
        frame_results = []
        
        for idx, file in enumerate(files):
            file_bytes = file.read()
            npimg = np.frombuffer(file_bytes, np.uint8)
            frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue
            
            frame_images.append(frame)
            
            # Quality check and enhancement
            enhanced_frame, q_metrics = quality_analyzer.analyze_and_enhance(frame)
            enhanced_frames.append(enhanced_frame)
            quality_metrics_list.append(q_metrics)
            
            # Skip liveness check if quality too low
            if q_metrics['frame_quality_score'] < 50:
                continue
            
            rgb = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
            
            # Face detection & matching
            import face_recognition
            locs = face_recognition.face_locations(rgb)
            encs = face_recognition.face_encodings(rgb, locs)
            
            if not encs:
                continue
            
            # Verify identity on first valid frame
            if idx == 0:
                face_enc = encs[0]
                user_face_data = db_manager.get_user_encodings(username)
                if not user_face_data:
                    return jsonify({
                        "error": "user_not_enrolled",
                        "instruction": "User not found - please enroll first"
                    }), 403
                cand_encs = [encoding for encoding, url in user_face_data]
                dists = face_recognition.face_distance(cand_encs, face_enc)
                dmin = float(np.min(dists))
                if dmin >= MATCH_THRESHOLD:
                    return jsonify({
                        "error": "face_not_matched",
                        "instruction": "Face doesn't match enrolled user",
                        "distance": dmin
                    }), 403
            
            # Liveness detection
            top, right, bottom, left = locs[0]
            crop = enhanced_frame[max(0,top):min(enhanced_frame.shape[0],bottom), 
                                  max(0,left):min(enhanced_frame.shape[1],right)]
            
            static_detector = StaticImageDetector()
            score, is_live, detailed = compute_liveness_enhanced(
                crop, model, device, model_type=model_type,
                temporal_checker=session_detectors['temporal'],
                blink_detector=session_detectors['blink'],
                static_detector=static_detector,
                image_bytes=file_bytes,
                weights=CONFIG.get("ensemble",{}).get("weights")
            )
            
            session_detectors['temporal'].add_frame(crop, time.time())
            
            frame_results.append({
                'frame_idx': idx,
                'liveness_score': float(score),
                'is_live': bool(is_live),
                'quality_score': q_metrics['frame_quality_score']
            })
        
        # Check if we have enough valid frames
        if len(frame_results) < 3:
            return jsonify({
                "error": "insufficient_quality_frames",
                "instruction": "Please improve lighting and hold steady",
                "frames_processed": len(frame_results),
                "quality_issues": [m['rejection_reasons'] for m in quality_metrics_list if m['rejection_reasons']]
            }), 400
        
        # Motion coherence analysis
        if len(frame_images) >= 2:
            motion_result_raw = comprehensive_motion_analysis(frame_images)
            # Sanitize motion result
            motion_result = sanitize_for_json(motion_result_raw)
            
            if not motion_result['is_live_motion']:
                return jsonify({
                    "status": "fail",
                    "reason": "static_video_detected",
                    "instruction": "Video replay or static image detected - please use live camera",
                    "motion_analysis": motion_result
                }), 403
        else:
            motion_result = {'combined_score': 0.0, 'is_live_motion': False}
        
        # Calculate metrics
        all_live = all(r['is_live'] for r in frame_results)
        avg_score = np.mean([r['liveness_score'] for r in frame_results])
        score_variance = np.var([r['liveness_score'] for r in frame_results])
        avg_quality = np.mean([r['quality_score'] for r in frame_results])
        
        # Check for suspicious patterns
        suspicious_flags = []
        
        # Very low variance = possible video loop
        if score_variance < 0.005:
            suspicious_flags.append("low_score_variance")
        
        # Check blink detection
        blink_count = len(getattr(session_detectors['blink'], 'blink_history', []))
        if blink_count == 0 and len(frame_results) > 5:
            suspicious_flags.append("no_blink_detected")
        
        # Adaptive threshold
        risk_context = {
            'device_trusted': request.form.get('device_trusted', 'false').lower() == 'true',
            'user_login_count': int(request.form.get('user_login_count', '0')),
            'recent_failed_attempts': int(request.form.get('recent_failed_attempts', '0')),
            'timestamp': time.time()
        }
        threshold = threshold_manager.get_threshold(risk_context)
        
        # Determine pass/fail
        passed = (
            all_live and 
            avg_score >= threshold and 
            motion_result['is_live_motion'] and
            len(suspicious_flags) == 0 and
            avg_quality >= 60
        )
        
        # Generate detailed feedback
        if not passed:
            if not all_live:
                reason = "liveness_check_failed"
                instruction = "Liveness verification failed - please retry"
            elif avg_score < threshold:
                reason = "low_liveness_score"
                instruction = f"Liveness score too low ({avg_score:.2f} < {threshold:.2f})"
            elif not motion_result['is_live_motion']:
                reason = "no_natural_motion"
                instruction = "No natural motion detected - please move slightly"
            elif "no_blink_detected" in suspicious_flags:
                reason = "no_blink_detected"
                instruction = "No blink detected - please blink naturally"
            elif avg_quality < 60:
                reason = "poor_frame_quality"
                instruction = "Frame quality too low - improve lighting and stability"
            else:
                reason = "suspicious_pattern_detected"
                instruction = f"Suspicious patterns: {', '.join(suspicious_flags)}"
            
            # Log failed attempt
            verification_logger.log_attempt(
                username=username,
                success=False,
                liveness_score=avg_score,
                quality_score=avg_quality,
                reason=reason,
                ip_address=request.remote_addr,
                metadata={
                    'session_id': session_id,
                    'frames_analyzed': len(frame_results),
                    'suspicious_flags': suspicious_flags,
                    'blink_count': blink_count,
                    'motion_score': motion_result.get('combined_score', 0.0)
                }
            )
            
            return jsonify({
                "status": "fail",
                "reason": reason,
                "instruction": instruction,
                "avg_liveness_score": float(avg_score),
                "threshold_used": float(threshold),
                "suspicious_flags": suspicious_flags,
                "avg_quality_score": float(avg_quality),
                "blink_count": int(blink_count)
            }), 403
        
        # Log successful attempt
        verification_logger.log_attempt(
            username=username,
            success=True,
            liveness_score=avg_score,
            quality_score=avg_quality,
            reason="verification_success",
            ip_address=request.remote_addr,
            metadata={
                'session_id': session_id,
                'frames_analyzed': len(frame_results),
                'score_variance': float(score_variance),
                'blink_count': blink_count,
                'motion_score': motion_result.get('combined_score', 0.0)
            }
        )

        # Update last verified timestamp
        db_manager.update_last_verified(username)

        # Success!
        return jsonify({
            "api_version": "3.0-guided",
            "step": "complete",
            "status": "success",
            "instruction": "Verification successful!",
            "session_id": session_id,
            "username": username,
            "avg_liveness_score": float(avg_score),
            "score_variance": float(score_variance),
            "avg_quality_score": float(avg_quality),
            "threshold_used": float(threshold),
            "frames_analyzed": len(frame_results),
            "motion_analysis": motion_result,
            "blink_count": int(blink_count),
            "all_frames_live": bool(all_live)
        })
    
    else:
        return jsonify({"error": "invalid_step", "valid_steps": ["quality_check", "liveness_verify"]}), 400


# ========================================
# UNIFIED MULTI-ENDPOINT VERIFICATION
# ========================================

def _quick_liveness_check(frame, model, device):
    """
    Quick liveness check on a single frame
    Returns confidence score 0-1
    """
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb)
        if not locs:
            return 0.0
        
        top, right, bottom, left = locs[0]
        crop = frame[max(0,top):min(frame.shape[0],bottom), 
                    max(0,left):min(frame.shape[1],right)]
        
        score, is_live = compute_liveness(crop, model, device)
        return score
    except Exception:
        return 0.0


def _calculate_weighted_score(results, challenge_ran):
    """
    Calculate weighted final score from multiple endpoint results
    Weights: sequence=0.30, enhanced=0.30, guided=0.25, challenge=0.15
    If challenge not ran, redistribute its weight proportionally
    """
    config = CONFIG.get("unified_verification", {})
    weights = config.get("weights", {
        "sequence": 0.30,
        "enhanced": 0.30,
        "guided": 0.25,
        "challenge": 0.15
    })
    
    # Extract scores from results
    scores = {}
    for key in ["sequence", "enhanced", "guided"]:
        if key in results and results[key].get("score") is not None:
            scores[key] = results[key]["score"]
    
    # Add challenge score if it ran
    if challenge_ran and "challenge" in results:
        scores["challenge"] = results["challenge"].get("score", 0.0)
    
    # Calculate effective weights
    total_weight = sum(weights.get(k, 0) for k in scores.keys())
    if total_weight == 0:
        return 0.0
    
    # Normalize weights if challenge didn't run
    effective_weights = {}
    for key in scores.keys():
        effective_weights[key] = weights.get(key, 0) / total_weight
    
    # Calculate weighted average
    final_score = sum(scores[k] * effective_weights[k] for k in scores.keys())
    
    return final_score, effective_weights, scores


@app.route("/v3/verify/unified", methods=["POST"])
@token_required
@user_allowed_required
def verify_unified():
    """
    Unified verification using all v3 endpoints with weighted scoring
    Flow: Quick check → Conditional challenge → Parallel verification
    """
    files = request.files.getlist("frames")
    username = request.form.get("username", "").strip()
    session_id = request.form.get("session_id") or (token_urlsafe(16) if token_urlsafe else f"sess_{int(time.time())}")
    
    if not username:
        return jsonify({"error": "username required"}), 400
    if len(files) < 5:
        return jsonify({"error": "minimum 5 frames required"}), 400
    
    config = CONFIG.get("unified_verification", {})
    challenge_threshold = config.get("challenge_threshold", 0.8)
    adaptive_threshold = config.get("adaptive_threshold", 0.75)
    
    # Decode all frames first
    frames = []
    for file in files:
        file_bytes = file.read()
        npimg = np.frombuffer(file_bytes, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if frame is not None:
            frames.append(frame)
    
    if len(frames) < 5:
        return jsonify({"error": "failed to decode frames"}), 400
    
    # Step 1: Quick liveness check on first frame
    quick_score = _quick_liveness_check(frames[0], model, device)
    
    # Step 2: Challenge verification (always run if challenges provided)
    challenge_result = None
    challenge_ran = False

    # Check if user should skip challenges
    user_should_skip = db_manager.get_user_skip_challenges(username)
    if user_should_skip:
        print(f"[+] User {username} configured to skip challenges")
        challenge_result = {"score": 1.0, "skipped": True, "reason": "user_exempt"}
        challenge_ran = False  # Mark as not run, but give full score
    else:
        # Check if challenges were provided in the request
        # Handle both direct list and indexed format
        challenge_ids = request.form.getlist("challenge_ids")
        if not challenge_ids:
            # Try indexed format: challenge_ids[0], challenge_ids[1], etc.
            challenge_ids = []
            index = 0
            while True:
                key = f"challenge_ids[{index}]"
                value = request.form.get(key)
                if value is None:
                    break
                challenge_ids.append(value)
                index += 1

        if challenge_ids:
            challenge_ran = True
            challenge_scores = []

            # Get session detectors for enhanced verification
            session_detectors = get_or_create_session_detectors(session_id)

            # Verify each challenge individually
            for i, challenge_id in enumerate(challenge_ids):
                try:
                    # Extract frames for this specific challenge (divide frames among challenges)
                    frames_per_challenge = max(3, len(frames) // len(challenge_ids))
                    start_idx = i * frames_per_challenge
                    end_idx = min(start_idx + frames_per_challenge, len(frames))
                    challenge_frames = frames[start_idx:end_idx]

                    # Verify the challenge
                    verification_result = challenge_manager.verify_challenge(challenge_id, challenge_frames)
                    score = 1.0 if verification_result.get("success", False) else 0.0
                    challenge_scores.append(score)

                    print(f"[+] Challenge {challenge_id}: {verification_result}")

                except Exception as e:
                    print(f"[!] Challenge verification failed for {challenge_id}: {e}")
                    challenge_scores.append(0.0)  # Failed challenge = 0 score

            # Calculate average challenge score
            avg_challenge_score = np.mean(challenge_scores) if challenge_scores else 0.0
            challenge_result = {
                "score": float(avg_challenge_score),
                "challenges_verified": len(challenge_scores),
                "challenges_passed": sum(1 for s in challenge_scores if s > 0.5)
            }

            print(f"[+] Challenge verification complete: {challenge_result}")
        elif quick_score < challenge_threshold:
            # Fallback: trigger single challenge if score is low (legacy behavior)
            challenge_ran = True
            challenge_result = {"score": 0.5, "triggered": True, "fallback": True}
    
    # Step 3: Verify identity on first frame
    rgb = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)
    locs = face_recognition.face_locations(rgb)
    encs = face_recognition.face_encodings(rgb, locs)
    
    if not encs:
        return jsonify({"error": "no face detected"}), 400
    
    face_enc = encs[0]
    user_face_data = db_manager.get_user_encodings(username)

    if not user_face_data:
        return jsonify({"error": "user not enrolled"}), 403

    cand_encs = [encoding for encoding, url in user_face_data]
    dists = face_recognition.face_distance(cand_encs, face_enc)
    dmin = float(np.min(dists))

    if dmin >= MATCH_THRESHOLD:
        return jsonify({"error": "face not matched", "distance": dmin}), 403

    # Update last verified timestamp
    db_manager.update_last_verified(username)
    
    # Step 4: Run parallel verification across endpoints
    results = {}
    
    # 4a. Sequence verification
    try:
        session_detectors = get_or_create_session_detectors(session_id)
        temporal_checker = session_detectors['temporal']
        blink_detector = session_detectors['blink']
        injection_detector = session_detectors['injection']
        
        frame_scores = []
        for frame in frames[:5]:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locs_frame = face_recognition.face_locations(rgb_frame)
            if locs_frame:
                t, r, b, l = locs_frame[0]
                crop = frame[max(0,t):min(frame.shape[0],b), max(0,l):min(frame.shape[1],r)]
                score, _ = compute_liveness(crop, model, device)
                frame_scores.append(score)
        
        avg_sequence_score = np.mean(frame_scores) if frame_scores else 0.0
        results["sequence"] = {
            "score": float(avg_sequence_score),
            "frames_analyzed": len(frame_scores)
        }
    except Exception as e:
        results["sequence"] = {"score": 0.0, "error": str(e)}
    
    # 4b. Enhanced verification (motion analysis)
    try:
        motion_scores = []
        for i in range(1, min(len(frames), 5)):
            # Simple motion detection between frames
            diff = cv2.absdiff(frames[i-1], frames[i])
            motion_metric = np.mean(diff) / 255.0
            motion_scores.append(min(1.0, motion_metric * 10))  # Scale to 0-1
        
        avg_motion_score = np.mean(motion_scores) if motion_scores else 0.0
        # Combine with liveness
        enhanced_score = (avg_motion_score * 0.3 + avg_sequence_score * 0.7)
        results["enhanced"] = {
            "score": float(enhanced_score),
            "motion_detected": len(motion_scores) > 0
        }
    except Exception as e:
        results["enhanced"] = {"score": 0.0, "error": str(e)}
    
    # 4c. Guided verification (use last frames with quality check)
    try:
        quality_scores = []
        for frame in frames[-3:]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            quality_scores.append(min(1.0, blur_score / 100.0))
        
        avg_quality = np.mean(quality_scores) if quality_scores else 0.0
        guided_score = (avg_quality * 0.4 + avg_sequence_score * 0.6)
        results["guided"] = {
            "score": float(guided_score),
            "quality_check": True
        }
    except Exception as e:
        results["guided"] = {"score": 0.0, "error": str(e)}
    
    # 4d. Add challenge result if it ran
    if challenge_ran and challenge_result:
        results["challenge"] = challenge_result
    
    # Step 5: Calculate weighted final score
    final_score, effective_weights, endpoint_scores = _calculate_weighted_score(results, challenge_ran)
    
    # Determine pass/fail
    success = final_score >= adaptive_threshold
    
    return jsonify({
        "api_version": "3.0-unified",
        "session_id": session_id,
        "success": success,
        "username": username,
        "final_score": float(final_score),
        "threshold": adaptive_threshold,
        "challenge_ran": challenge_ran,
        "quick_check_score": float(quick_score),
        "endpoint_scores": endpoint_scores,
        "effective_weights": effective_weights,
        "endpoints_used": list(endpoint_scores.keys()),
        "breakdown": {
            "sequence": results.get("sequence", {}),
            "enhanced": results.get("enhanced", {}),
            "guided": results.get("guided", {}),
            "challenge": results.get("challenge", {}) if challenge_ran else None
        }
    })


@app.route("/enroll", methods=["POST"])
@token_required
@user_allowed_required
def enroll():
    username = (request.form.get("username") or "").strip()
    if not username:
        return jsonify({"error": "username required"}), 400

    # Check if services are configured
    if not cloudinary_manager.configured:
        return jsonify({"error": "Cloudinary not configured"}), 500
    if not db_manager.connection:
        return jsonify({"error": "Database not configured"}), 500

    files = request.files.getlist("images")
    if not files:
        # allow repeated 'image' fields too
        files = request.files.getlist("image")
    if not files:
        return jsonify({"error": "no images"}), 400

    enrolled = 0
    uploaded = 0
    ts = int(time.time())

    # Step 1: Process all frames to extract face encodings (fast, sequential)
    frames_data = []  # List of (index, file_bytes, encoding)
    
    for i, f in enumerate(files):
        try:
            # Read image bytes
            file_bytes = f.read()

            # Decode image
            npimg = np.frombuffer(file_bytes, np.uint8)
            frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            if frame is None:
                print(f"[!] Frame {i}: Failed to decode image")
                continue

            print(f"[*] Frame {i}: Image decoded successfully, shape: {frame.shape}")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Try with different models for better detection
            locs = face_recognition.face_locations(rgb, model="hog")
            print(f"[*] Frame {i}: Found {len(locs)} face location(s)")

            if not locs:
                # Retry with CNN model if HOG fails
                print(f"[*] Frame {i}: Retrying with CNN model...")
                locs = face_recognition.face_locations(rgb, model="cnn") if torch.cuda.is_available() else []
                print(f"[*] Frame {i}: CNN model found {len(locs)} face(s)")

            encs = face_recognition.face_encodings(rgb, locs)
            if not encs:
                print(f"[!] Frame {i}: No face encodings generated")
                continue

            # Store for parallel upload
            frames_data.append((i, file_bytes, encs[0]))

        except Exception as e:
            print(f"[!] Frame {i}: Exception during processing: {e}")
            continue

    if not frames_data:
        print(f"[!] WARNING: No faces detected in any of the {len(files)} frames!")
        return jsonify({"status": "no_face", "enrolled": 0, "message": "No faces detected in any frames. Please ensure good lighting and face camera directly."}), 200

    # Step 2: Upload to Cloudinary in parallel (slow, parallel)
    print(f"[*] Uploading {len(frames_data)} frames to Cloudinary in parallel...")
    
    def upload_frame(frame_data):
        """Upload a single frame to Cloudinary"""
        i, file_bytes, encoding = frame_data
        try:
            url = cloudinary_manager.upload_face_image(file_bytes, username, i, ts)
            if url:
                print(f"[+] Uploaded image for {username}: {url}")
                return (encoding, url, i)
            else:
                print(f"[!] Frame {i}: Failed to upload to Cloudinary")
                return None
        except Exception as e:
            print(f"[!] Frame {i}: Upload exception: {e}")
            return None
    
    face_encodings = []
    cloudinary_urls = []
    
    # Use ThreadPoolExecutor for parallel uploads (max 5 concurrent uploads)
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_frame = {executor.submit(upload_frame, frame_data): frame_data for frame_data in frames_data}
        
        for future in as_completed(future_to_frame):
            result = future.result()
            if result:
                encoding, url, i = result
                face_encodings.append(encoding)
                cloudinary_urls.append(url)
                enrolled += 1
                uploaded += 1
                print(f"[+] Frame {i}: Successfully processed and uploaded")

    print(f"[*] Enrollment complete for '{username}': {enrolled} faces enrolled out of {len(files)} frames received")

    # Save to database
    success = db_manager.enroll_user(username, face_encodings, cloudinary_urls)

    if not success:
        return jsonify({"error": "Failed to save enrollment data"}), 500

    return jsonify({
        "status": "ok",
        "user": username,
        "enrolled": enrolled,
        "uploaded": uploaded,
        "message": f"Successfully enrolled {enrolled} face(s) for user '{username}'"
    }), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
