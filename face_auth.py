# face_auth.py
# Full script: face recognition + DeePixBiS liveness
# Safe version with improved camera and face encoding checks

import os
import cv2
import torch
import numpy as np
import face_recognition
import torchvision.transforms as T
from Model import DeePixBiS  # your provided model.py

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "DeePixBiS.pth"),
    os.path.join(PROJECT_ROOT, "liveness_model.pth"),
    os.path.join(PROJECT_ROOT, "model.pth"),
]

KNOWN_FACES_DIR = os.path.join(PROJECT_ROOT, "known_faces")
CAM_INDEX = 0
MATCH_THRESHOLD = 0.45   # face-recognition euclidean distance threshold (tune)
LIVENESS_THRESHOLD = 0.5 # DeePixBiS score threshold (tune)

# -------------------------
# Utilities
# -------------------------
def find_model_path():
    for p in MODEL_CANDIDATES:
        if os.path.exists(p):
            return p
    return None

def load_deeppixbis(weights_path, device):
    print(f"[+] Loading DeePixBiS weights from: {weights_path}")
    model = DeePixBiS(pretrained=False).to(device)
    sd = torch.load(weights_path, map_location=device)
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    try:
        model.load_state_dict(sd, strict=False)
    except Exception as e:
        print("[!] Warning: model.load_state_dict(strict=False) raised:", e)
        try:
            model.load_state_dict(sd)
        except Exception as e2:
            print("[!] Failed to load state dict exactly:", e2)
            raise
    model.eval()
    print("[+] Model loaded and set to eval()")
    return model

# Preprocess transform (DenseNet backbone expected ~224x224)
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

def safe_crop(frame, top, right, bottom, left, pad=10):
    h, w = frame.shape[:2]
    top = max(0, top - pad)
    left = max(0, left - pad)
    bottom = min(h, bottom + pad)
    right = min(w, right + pad)
    if bottom <= top or right <= left:
        return None
    return frame[top:bottom, left:right]

def compute_liveness(face_bgr, model, device):
    """face_bgr: BGR crop from OpenCV -> returns (score, is_live)"""
    try:
        img_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        tensor = transform(img_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            out_map, out = model(tensor)
            score = float(out.cpu().squeeze().item()) if isinstance(out, torch.Tensor) else float(out)
        return score, (score > LIVENESS_THRESHOLD)
    except Exception as e:
        print("[!] Liveness inference error:", e)
        return 0.0, False

# -------------------------
# Load known faces
# -------------------------
print("[*] Loading known faces from:", KNOWN_FACES_DIR)
known_face_encodings, known_face_names = [], []
if os.path.isdir(KNOWN_FACES_DIR):
    for fname in os.listdir(KNOWN_FACES_DIR):
        if fname.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(KNOWN_FACES_DIR, fname)
            img = face_recognition.load_image_file(path)
            encs = face_recognition.face_encodings(img)
            if encs:
                known_face_encodings.append(encs[0])
                known_face_names.append(os.path.splitext(fname)[0])
                print(f"  - Loaded {fname} as '{os.path.splitext(fname)[0]}'")
            else:
                print(f"  ! No face found in {fname}; skipping.")
else:
    print("[!] known_faces folder missing. Create known_faces/ and put images in it.")

# -------------------------
# Load liveness model
# -------------------------
model_path = find_model_path()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if model_path is None:
    print("[!] No DeePixBiS weights found. Put 'DeePixBiS.pth' in project folder.")
    raise SystemExit(1)

deeppix_model = load_deeppixbis(model_path, device)

# -------------------------
# Start camera + main loop
# -------------------------
print("[*] Starting camera on index", CAM_INDEX)
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)  # <-- Added backend flag
if not cap.isOpened():
    print("[-] Cannot open camera. Exiting.")
    raise SystemExit(1)

print("Press 'q' to quit.")
frame_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            frame_count += 1
            if frame_count > 10:
                print("[-] Failed to read camera frames. Exiting.")
                break
            continue

        # Show raw feed to verify camera works
        cv2.imshow("DEBUG Camera Feed", frame)

        # Optional: brighten a bit for low light
        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=30)

        frame_count = 0
        small_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        rgb_small = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        if not face_locations:
            print("[!] No faces detected in frame.")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        for (top_s, right_s, bottom_s, left_s), face_encoding in zip(face_locations, face_encodings):
            scale = 2
            top, right, bottom, left = [int(v*scale) for v in (top_s, right_s, bottom_s, left_s)]
            name = "Unknown"

            if known_face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=MATCH_THRESHOLD)
                dists = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_idx = np.argmin(dists) if len(dists) > 0 else None
                if best_idx is not None and matches[best_idx]:
                    name = known_face_names[best_idx]

            crop = safe_crop(frame, top, right, bottom, left, pad=12)
            if crop is None or crop.size == 0:
                continue

            score, is_live = compute_liveness(crop, deeppix_model, device)
            color = (0,255,0) if (is_live and name != "Unknown") else (0,0,255)
            label = f"{name} | Liveness:{score:.2f} {'(LIVE)' if is_live else '(SPOOF)'}"
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("FaceAuth (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("[*] Exited cleanly.")
