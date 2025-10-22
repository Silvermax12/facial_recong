# face_auth_utils.py
import os
import cv2
import torch
import numpy as np
import face_recognition
import torchvision.transforms as T
from Model import DeePixBiS
from Model_ViT import ViTLivenessDetector, EnhancedLivenessDetector
from enhanced_liveness_utils import (
    FrequencyDomainAnalyzer,
    TemporalConsistencyChecker,
    EnhancedBlinkDetector,
    StaticImageDetector,
    InjectionAttackDetector,
    enhanced_liveness_check
)
from frame_quality import EnhancedPreprocessor

MATCH_THRESHOLD = 0.45
LIVENESS_THRESHOLD = 0.5

# -------------------------
# Enhanced Model Loading
# -------------------------
def load_enhanced_model(model_type='ensemble', device=None):
    """Load enhanced liveness detection model with multiple detection methods"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[+] Loading enhanced {model_type} model")

    if model_type == 'vit':
        try:
            model = ViTLivenessDetector(pretrained=True).to(device)
            model_path = find_model_path(['vit_liveness_model.pth', 'liveness_model.pth'])
            if model_path:
                sd = torch.load(model_path, map_location=device)
                if isinstance(sd, dict) and 'state_dict' in sd:
                    sd = sd['state_dict']
                model.load_state_dict(sd, strict=False)
                print("[+] Loaded ViT liveness model")
            else:
                print("[!] ViT model weights not found, using pretrained backbone only")
        except Exception as e:
            print(f"[!] Failed to load ViT model: {e}, falling back to CNN")
            model = load_deeppixbis("liveness_model.pth", device)
            model_type = 'cnn'

    elif model_type == 'ensemble':
        try:
            vit_model = ViTLivenessDetector(pretrained=True).to(device)
            cnn_model = load_deeppixbis("liveness_model.pth", device)
            model = EnhancedLivenessDetector(vit_model=vit_model, cnn_model=cnn_model)
            print("[+] Loaded ensemble model (ViT + CNN)")
        except Exception as e:
            print(f"[!] Failed to load ensemble: {e}, falling back to CNN")
            model = load_deeppixbis("liveness_model.pth", device)
            model_type = 'cnn'

    else:  # Default to CNN
        model = load_deeppixbis("liveness_model.pth", device)
        model_type = 'cnn'

    model.eval()
    return model, model_type

def find_model_path(candidates):
    """Find the first available model path from candidates"""
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None

# -------------------------
# Load DeePixBiS model (legacy support)
# -------------------------
def load_deeppixbis(weights_path, device):
    print(f"[+] Loading DeePixBiS weights from: {weights_path}")
    model = DeePixBiS(pretrained=False).to(device)
    sd = torch.load(weights_path, map_location=device)
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model

# -------------------------
# Load known faces
# -------------------------
def load_known_faces(folder):
    print(f"[*] Loading known faces from: {folder}")
    known_encodings = []
    known_names = []
    if os.path.isdir(folder):
        for fname in os.listdir(folder):
            if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                path = os.path.join(folder, fname)
                img = face_recognition.load_image_file(path)
                encs = face_recognition.face_encodings(img)
                if encs:
                    # Normalize username: take prefix before first underscore to match enrollment naming pattern
                    base = os.path.splitext(fname)[0]
                    uname = base.split('_', 1)[0]
                    known_encodings.append(encs[0])
                    known_names.append(uname)
                    print(f"  - Loaded {fname} as '{uname}'")
                else:
                    print(f"  ! No face found in {fname}")
    else:
        print("[!] known_faces folder missing.")
    return known_encodings, known_names

# -------------------------
# Enhanced liveness detection
# -------------------------
_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

def compute_liveness_enhanced(face_bgr, model, device, model_type='cnn',
                             use_frequency_analysis=True, use_temporal_check=True,
                             temporal_checker=None, blink_detector=None, static_detector=None,
                             image_path=None, image_bytes=None, weights=None,
                             use_enhanced_preprocessing=True):    
    """
    Enhanced liveness detection with multiple anti-spoofing methods
    Now includes quality-enhanced preprocessing for better model performance
    """

    # Apply enhanced preprocessing if enabled
    preprocessed_face = face_bgr
    if use_enhanced_preprocessing:
        try:
            # Use EnhancedPreprocessor for better quality
            preprocessed_face = EnhancedPreprocessor.preprocess_for_deeppixbis(
                face_bgr, target_size=(224, 224)
            )
            # Convert back to uint8 for other analyzers
            preprocessed_face = (preprocessed_face * 255).astype(np.uint8)
        except Exception as e:
            print(f"[!] Preprocessing warning: {e}, using original")
            preprocessed_face = face_bgr

    # Initialize analyzers if needed
    freq_analyzer = FrequencyDomainAnalyzer() if use_frequency_analysis else None

    # Use enhanced detection pipeline with preprocessed face
    final_score, detailed_scores = enhanced_liveness_check(
        face_crop=preprocessed_face, model=model, device=device, freq_analyzer=freq_analyzer,
        temporal_checker=temporal_checker, blink_detector=blink_detector,
        static_detector=static_detector, image_path=image_path, 
        image_bytes=image_bytes, weights=weights
    )

    is_live = final_score > LIVENESS_THRESHOLD

    return final_score, is_live, detailed_scores

# -------------------------
# Legacy function for backward compatibility
# -------------------------
def compute_liveness(face_bgr, model, device):
    """Legacy liveness detection for backward compatibility"""
    try:
        img_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        tensor = _transform(img_rgb).unsqueeze(0).to(device)
        # Prefer CNN-only path if an ensemble was provided
        model_for_inference = getattr(model, 'cnn_model', None) or model
        with torch.no_grad():
            out_map, out = model_for_inference(tensor)
            score = float(out.cpu().squeeze().item())
        is_live = (score > LIVENESS_THRESHOLD)
        return score, is_live
    except Exception as e:
        print("[!] Liveness inference error:", e)
        return 0.0, False
