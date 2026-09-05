# 08. Configuration Engine, Adaptive Thresholds & Profiles

## 1. Configuration Architecture

The system uses a two-tier configuration model:
1. **Built-in Defaults**: Hardcoded in `face_api.py` (`DEFAULT_CONFIG`) as a fail-safe fallback.
2. **Declarative YAML Configuration**: Loaded from `liveness_config.yaml` using PyYAML. If `liveness_config.yaml` is present, it dynamically overrides defaults with fine-tuned parameters.

```
┌─────────────────────────┐
│ face_api.py             │
│ (DEFAULT_CONFIG dict)   │
└───────────┬─────────────┘
            │ Fallback
            ▼
┌─────────────────────────┐      Deep Merge      ┌─────────────────────────┐
│ liveness_config.yaml    ├─────────────────────►│ Active CONFIG Object    │
│ (Declarative Overrides) │                      │ (In-memory system state)│
└─────────────────────────┘                      └─────────────────────────┘
```

---

## 2. Parameter Reference (`liveness_config.yaml`)

### 2.1 `ensemble` (Passive PAD Weights)
Controls the relative importance of each signal in the multi-modal PAD calculation:

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `weights.model` | Float | `0.35` | Weight for DeepPixBiS CNN or ViT prediction |
| `weights.frequency` | Float | `0.25` | Weight for 2D-FFT spectral anomaly analysis |
| `weights.temporal` | Float | `0.20` | Weight for inter-frame temporal consistency |
| `weights.blink_pattern`| Float | `0.10` | Weight for blink dynamics and frequency |
| `weights.static_image` | Float | `0.10` | Weight for screen moir\u00e9 and glare detection |
| `min_signals_required` | Integer | `4` | Minimum number of individual sub-checks that must pass |

---

### 2.2 `thresholds` (Decision Boundaries)

| Key | Type | Default | Operational Impact |
| :--- | :--- | :--- | :--- |
| `liveness` | Float | `0.75` | Final ensemble score needed to mark a face as LIVE |
| `spectrum` | Float | `0.70` | Cutoff for synthetic/replay frequency spectrum flags |
| `blink_ear` | Float | `0.23` | Eye Aspect Ratio threshold (values below indicate eye closure) |
| `single_frame` | Float | `0.90` | Penalty threshold for single static frames (near-impossible for photos) |
| `multi_frame` | Float | `0.75` | Normal threshold for multi-frame video sequences |

---

### 2.3 `unified_verification` (Master Pipeline)

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `challenge_threshold` | Float | `0.80` | If quick liveness score < 0.80, interactive challenges are required |
| `weights.sequence` | Float | `0.30` | Weight of sequence analysis in unified score |
| `weights.enhanced` | Float | `0.30` | Weight of optical flow motion analysis |
| `weights.guided` | Float | `0.25` | Weight of frame quality assessment |
| `weights.challenge` | Float | `0.15` | Weight of interactive challenge-response |
| `adaptive_threshold` | Float | `0.75` | Minimum unified score to pass authentication |
| `require_identity_match`| Boolean | `true` | Requires Euclidean distance < 0.45 against enrolled face |

---

### 2.4 `quality` (Frame Ingestion Standards)

| Key | Type | Default | Purpose |
| :--- | :--- | :--- | :--- |
| `blur_threshold` | Float | `100.0` | Laplacian variance cutoff (values below flag blurry frames) |
| `min_brightness` | Integer | `40` | Rejects under-exposed/dark frames (0-255 scale) |
| `max_brightness` | Integer | `220` | Rejects over-exposed/washed-out frames (0-255 scale) |
| `target_brightness` | Integer | `128` | Normalization midpoint for preprocessor gamma adjustment |
| `min_face_size` | Integer | `80` | Minimum bounding box width/height in pixels |
| `max_faces_allowed` | Integer | `1` | Rejects frames with multiple people in view |
| `min_quality_score` | Float | `60.0` | Overall composite quality score cutoff (0-100) |

---

## 3. Dynamic Context Adaptive Thresholding (`adaptive_threshold.py`)

Rather than relying on static decision boundaries, the `AdaptiveThresholdManager` calculates dynamic risk-weighted thresholds based on the transaction context:

### 3.1 Context Risk Factors
- `device_trusted`: Registered hardware token or established device fingerprint (-0.05 threshold discount).
- `device_rooted`: Jailbroken iOS or rooted Android device (+0.10 threshold penalty).
- `user_login_count`: Mature user accounts with established biometric baselines (-0.03 discount).
- `recent_failed_attempts`: Recent consecutive failures (+0.05 penalty per failure).

### 3.2 Dynamic Operating Tiers
```
Low Risk Profile    ────────► Threshold: 0.70 (Frictionless login)
Normal Profile      ────────► Threshold: 0.75 (Standard security)
Elevated Risk       ────────► Threshold: 0.85 (Strict verification required)
Critical Risk       ────────► Threshold: 0.92 (Challenges + multi-frame required)
```

---

## 4. Hardware Profiles

Hardware profiles allow instant system reconfiguration to match target environments:

```yaml
profiles:
  default:
    model_type: ensemble
    
  low_light:
    liveness: 0.60
    quality:
      min_brightness: 20
      target_brightness: 100
      
  mobile_cpu:
    model_type: cnn
    liveness: 0.58
    performance:
      capture_fps: 8
      frame_count: 5
      
  high_security:
    liveness: 0.85
    quality:
      blur_threshold: 150.0
      min_quality_score: 70.0
    guided:
      require_blink: true
      require_head_movement: true
      challenge_probability: 0.7
```

---

## 5. Environment Variables Reference

| Variable | Required | Default | Description |
| :--- | :--- | :--- | :--- |
| `DATABASE_URL` | **Yes** | None | PostgreSQL connection URI (`postgresql://user:pass@host:port/db`) |
| `CLOUDINARY_CLOUD_NAME` | **Yes** | None | Cloudinary cloud account name |
| `CLOUDINARY_API_KEY` | **Yes** | None | Cloudinary public API key |
| `CLOUDINARY_API_SECRET` | **Yes** | None | Cloudinary private API secret |
| `JWT_SECRET_KEY` | **Yes** | `dev-secret-key...` | Cryptographic secret for signing JWTs (32+ chars) |
| `JWT_ACCESS_TOKEN_EXPIRES`| No | `24` | Token expiration duration in hours |
| `FLASK_ENV` | No | `production` | Environment mode (`production` or `development`) |
| `PYTHON_VERSION` | No | `3.9.18` | Python runtime version pinned on Render |
| `PORT` | No | `5000` | Port for web service binding (Render sets this dynamically) |
