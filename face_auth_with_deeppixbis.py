"""
face_auth_with_deeppixbis.py
Full prototype: registration + authentication + DeepPixBiS anti-spoof integration.

Usage:
  python face_auth_with_deeppixbis.py --register alice
  python face_auth_with_deeppixbis.py --auth alice --challenge
"""

import os
import time
import argparse
import pickle
import random
from collections import deque
from math import hypot

import cv2
import numpy as np
import face_recognition
import mediapipe as mp
import torch
import torch.nn as nn
import torchvision.transforms as T

# ---------------- CONFIG ----------------
ENCODINGS_DIR = "encodings"
FAS_MODELS_DIR = "fas_models"
DEEPPB_PATH = os.path.join(FAS_MODELS_DIR, "DeePixBiS.pth")  # expected location of pretrained weights
CAM_INDEX = 0

FRAME_SKIP = 2
MATCH_THRESHOLD = 0.45
REQUIRED_MATCH_FRAMES = 5
BLINK_EAR_THRESH = 0.23
BLINK_MIN_CONSEC_FRAMES = 2
HEAD_MOVEMENT_THRESHOLD = 8.0
CHALLENGE_TIMEOUT = 6.0
# ----------------------------------------

mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
NOSE_TIP_IDX = 1

def ensure_dirs():
    os.makedirs(ENCODINGS_DIR, exist_ok=True)
    os.makedirs(FAS_MODELS_DIR, exist_ok=True)

def save_encoding(username, encoding):
    ensure_dirs()
    fname = os.path.join(ENCODINGS_DIR, f"{username}.pkl")
    with open(fname, "wb") as f:
        pickle.dump(encoding, f)
    print(f"[+] Saved encoding to {fname}")

def load_encoding(username):
    fname = os.path.join(ENCODINGS_DIR, f"{username}.pkl")
    if not os.path.exists(fname):
        return None
    with open(fname, "rb") as f:
        return pickle.load(f)

def compute_ear(landmarks, eye_idx, img_w, img_h):
    pts = [(int(landmarks[i].x * img_w), int(landmarks[i].y * img_h)) for i in eye_idx]
    p1, p2, p3, p4, p5, p6 = pts
    v1 = hypot(p2[0]-p6[0], p2[1]-p6[1])
    v2 = hypot(p3[0]-p5[0], p3[1]-p5[1])
    h = hypot(p1[0]-p4[0], p1[1]-p4[1]) + 1e-6
    return (v1 + v2) / (2.0 * h)

# ---------------- Registration (same as before) ----------------
def register_user(username, cam_index=CAM_INDEX, capture_count=5):
    print(f"[*] Registering '{username}'. Look at camera.")
    cap = cv2.VideoCapture(cam_index)
    encs = []
    frames = 0
    while len(encs) < capture_count and frames < 300:
        ret, frame = cap.read()
        if not ret:
            continue
        frames += 1
        if frames % FRAME_SKIP != 0:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="hog")
        feats = face_recognition.face_encodings(rgb, boxes)
        if feats:
            encs.append(feats[0])
            cv2.putText(frame, f"Captured {len(encs)}/{capture_count}", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        else:
            cv2.putText(frame, "No face detected", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        cv2.imshow("Register", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release(); cv2.destroyAllWindows()
    if not encs:
        print("[-] No faces captured.")
        return False
    mean_enc = np.mean(encs, axis=0)
    save_encoding(username, mean_enc)
    print("[+] Registration complete.")
    return True

# ---------------- DeepPixBiS model loader (minimal wrapper) ----------------
# NOTE: this is a minimal wrapper to load a DeepPixBiS-style model for inference.
# Many public repos implement DeepPixBiS. This wrapper expects a PyTorch state_dict
# compatible with the model architecture defined here.
class SimpleDeeppixBiS(nn.Module):
    """
    Minimal model structure for inference:
    - uses a small backbone + pixel-wise output -> aggregated score.
    - This is a best-effort wrapper for common DeepPixBiS forks.
    If you use a different repository, adapt the class to match their architecture.
    """
    def __init__(self):
        super().__init__()
        # simple conv backbone (placeholder) - will be replaced by loaded weights
        # real repo uses a richer backbone. We keep a simple container and rely on state_dict loading.
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        f = self.features(x)
        out = self.classifier(f)
        return out.squeeze(1)

def load_deeppixbis_model(weight_path):
    """
    Attempt to load a DeepPixBiS-style state_dict.
    If the repo you downloaded defines a different architecture, replace this
    function with loading their model class and weights accordingly.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleDeeppixBiS().to(device)
    if not os.path.exists(weight_path):
        print(f"[-] FAS weights not found at {weight_path}. Place weights there and retry.")
        return None
    try:
        sd = torch.load(weight_path, map_location=device)
        # If repo saved dict with 'state_dict', extract it
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
        # Try load, allow missing keys
        model.load_state_dict(sd, strict=False)
        model.eval()
        print("[+] Loaded DeepPixBiS weights (partial load allowed).")
    except Exception as e:
        print("[!] Failed to load weights directly into wrapper model:", e)
        print("[!] You may need to adapt `load_deeppixbis_model` to the exact repo architecture.")
        return None
    return model

# ---------------- FAS hook function used by authenticate_user ----------------
class DeepPixBISHook:
    def __init__(self, weight_path=DEEPPB_PATH, device=None):
        self.model = None
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        if os.path.exists(weight_path):
            self.model = load_deeppixbis_model(weight_path)
            if self.model is not None:
                self.model.to(self.device)
        # transform
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

    def __call__(self, frame_bgr):
        """
        frame_bgr: full color frame in BGR (OpenCV)
        returns: {'live_score': float (0..1), 'is_live': bool}
        """
        if self.model is None:
            return {'live_score': 0.0, 'is_live': None}
        try:
            # Preprocess: crop center face area to feed model (optional: detect face and crop)
            h,w = frame_bgr.shape[:2]
            # naive center-crop square
            s = min(h,w)
            cy, cx = h//2, w//2
            y0, x0 = cy - s//2, cx - s//2
            crop = frame_bgr[max(0,y0):max(0,y0)+s, max(0,x0):max(0,x0)+s].copy()
            img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = self.model(tensor)  # expects scalar sigmoid output ~ probability of real
                if isinstance(out, torch.Tensor):
                    score = float(out.squeeze().cpu().item())
                else:
                    score = float(out)
            is_live = score > 0.5  # threshold tuneable
            return {'live_score': score, 'is_live': bool(is_live)}
        except Exception as e:
            # If model fails, return unknown
            print("[!] FAS inference error:", e)
            return {'live_score': 0.0, 'is_live': None}

# ---------------- Challenge helpers (unchanged, compact) ----------------
def challenge_blink(expected_blinks=1, timeout=CHALLENGE_TIMEOUT, cam_index=CAM_INDEX):
    print(f"[?] Challenge: Please blink {expected_blinks} time(s).")
    start_ts = time.time()
    blink_count = 0
    consec_low = 0
    cap = cv2.VideoCapture(cam_index)
    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1) as fm:
        while time.time() - start_ts < timeout:
            ret, frame = cap.read()
            if not ret:
                continue
            h,w = frame.shape[:2]
            res = fm.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ear = None
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                left = compute_ear(lm, LEFT_EYE_IDX, w, h)
                right = compute_ear(lm, RIGHT_EYE_IDX, w, h)
                ear = (left+right)/2.0
                if ear < BLINK_EAR_THRESH:
                    consec_low += 1
                else:
                    if consec_low >= BLINK_MIN_CONSEC_FRAMES:
                        blink_count += 1
                    consec_low = 0
            if blink_count >= expected_blinks:
                cap.release(); cv2.destroyAllWindows(); return True
            cv2.imshow("Challenge", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release(); cv2.destroyAllWindows()
    return False

def optional_challenge_random(cam_index=CAM_INDEX):
    choices = ["blink","look_left","look_right"]
    pick = random.choice(choices)
    if pick == "blink":
        return challenge_blink(expected_blinks=1, timeout=CHALLENGE_TIMEOUT, cam_index=cam_index)
    print(f"[?] Challenge: Please quickly look {pick.split('_')[1]} within {CHALLENGE_TIMEOUT}s.")
    start = time.time()
    cap = cv2.VideoCapture(cam_index)
    start_x = None
    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1) as fm:
        while time.time() - start < CHALLENGE_TIMEOUT:
            ret, frame = cap.read()
            if not ret:
                continue
            h,w = frame.shape[:2]
            res = fm.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                nose = lm[NOSE_TIP_IDX]
                nx = nose.x * w
                if start_x is None:
                    start_x = nx
                dx = nx - start_x
                if pick.endswith("left") and dx < -HEAD_MOVEMENT_THRESHOLD:
                    cap.release(); cv2.destroyAllWindows(); return True
                if pick.endswith("right") and dx > HEAD_MOVEMENT_THRESHOLD:
                    cap.release(); cv2.destroyAllWindows(); return True
            cv2.imshow("Challenge", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release(); cv2.destroyAllWindows()
    return False

# ---------------- Authentication (integrates fas hook) ----------------
def authenticate_user(username, cam_index=CAM_INDEX, require_challenge=False, fas_hook=None):
    known = load_encoding(username)
    if known is None:
        print(f"[-] No encoding for {username}. Register first.")
        return False

    if require_challenge:
        ok = optional_challenge_random(cam_index=cam_index)
        if not ok:
            print("[-] Interactive challenge failed.")
            return False
        print("[+] Challenge passed.")

    cap = cv2.VideoCapture(cam_index)
    fm = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
    start_time = time.time()
    matched_frames = 0
    processed = 0
    blink_count = 0
    consec_low = 0
    nose_positions = deque(maxlen=30)
    face_match_history = deque(maxlen=40)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            if (int((time.time()-start_time)*30) % FRAME_SKIP) != 0:
                cv2.imshow("Auth", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            processed += 1
            small = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(small, model="hog")
            encs = face_recognition.face_encodings(small, boxes)
            is_face_matched = False
            if encs:
                d = float(np.linalg.norm(encs[0] - known))
                is_face_matched = (d <= MATCH_THRESHOLD)
                face_match_history.append(1 if is_face_matched else 0)
                if is_face_matched:
                    matched_frames += 1
                top, right, bottom, left = boxes[0]
                cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0) if is_face_matched else (0,0,255), 2)
                cv2.putText(frame, f"dist:{d:.3f}", (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0) if is_face_matched else (0,0,255), 2)

            mesh = fm.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ear = None
            head_movement = 0.0
            if mesh.multi_face_landmarks:
                lm = mesh.multi_face_landmarks[0].landmark
                left = compute_ear(lm, LEFT_EYE_IDX, frame.shape[1], frame.shape[0])
                right = compute_ear(lm, RIGHT_EYE_IDX, frame.shape[1], frame.shape[0])
                ear = (left+right)/2.0
                if ear < BLINK_EAR_THRESH:
                    consec_low += 1
                else:
                    if consec_low >= BLINK_MIN_CONSEC_FRAMES:
                        blink_count += 1
                    consec_low = 0
                nose = lm[NOSE_TIP_IDX]
                nx, ny = nose.x*frame.shape[1], nose.y*frame.shape[0]
                nose_positions.append((nx, ny))
                if len(nose_positions) >= 2:
                    xs = [p[0] for p in nose_positions]
                    ys = [p[1] for p in nose_positions]
                    head_movement = (max(xs)-min(xs)) + (max(ys)-min(ys))

            fas_out = None
            if fas_hook is not None:
                try:
                    fas_out = fas_hook(frame)
                except Exception as e:
                    print("[!] FAS hook error:", e)
                    fas_out = {'live_score':0.0,'is_live':None}

            passive_live = (blink_count > 0) or (head_movement > HEAD_MOVEMENT_THRESHOLD) or (fas_out and fas_out.get('is_live') is True)
            match_ratio = sum(face_match_history)/len(face_match_history) if face_match_history else 0.0
            status = f"MR:{match_ratio:.2f} Matches:{matched_frames} Blinks:{blink_count} Motion:{head_movement:.1f}"
            if fas_out:
                status += f" FAS:{fas_out.get('live_score'):.2f},{fas_out.get('is_live')}"
            cv2.putText(frame, status, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
            cv2.imshow("Auth", frame)

            if match_ratio > 0.45 and (passive_live or (fas_out and fas_out.get('is_live') is True)) and matched_frames >= REQUIRED_MATCH_FRAMES:
                print("[+] Authentication success: live face matched.")
                cap.release(); fm.close(); cv2.destroyAllWindows()
                return True

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if time.time() - start_time > 45:
                print("[-] Timeout")
                break
    finally:
        cap.release(); fm.close(); cv2.destroyAllWindows()

    print("[-] Authentication failed.")
    return False

# ---------------- CLI entry ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--register", type=str, help="Register username")
    parser.add_argument("--auth", type=str, help="Authenticate username")
    parser.add_argument("--challenge", action="store_true", help="Require interactive challenge")
    parser.add_argument("--cam", type=int, default=CAM_INDEX, help="camera index")
    parser.add_argument("--use-fas", action="store_true", help="Enable DeepPixBiS FAS model (must have weights in fas_models/DeePixBiS.pth)")
    args = parser.parse_args()

    if args.register:
        register_user(args.register, cam_index=args.cam)
        return
    fas_hook = None
    if args.use_fas:
        ensure_dirs()
        fas_hook = DeepPixBISHook(weight_path=DEEPPB_PATH)
        if fas_hook.model is None:
            print("[-] FAS model not loaded. Running without FAS.")
            fas_hook = None
    if args.auth:
        authenticate_user(args.auth, cam_index=args.cam, require_challenge=args.challenge, fas_hook=fas_hook)
    else:
        print("Use --register name or --auth name")

if __name__ == "__main__":
    main()
