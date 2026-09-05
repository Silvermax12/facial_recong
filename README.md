# 🛡️ Face Recognition & Multi-Modal Anti-Spoofing (PAD) API

[![Python 3.9](https://img.shields.io/badge/Python-3.9.18-blue.svg)](https://www.python.org/)
[![Flask 3.0](https://img.shields.io/badge/Framework-Flask_3.0-green.svg)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch_2.1-red.svg)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/Facial%20Mesh-MediaPipe_0.9.3-orange.svg)](https://mediapipe.dev/)
[![Database](https://img.shields.io/badge/PostgreSQL-Render_Managed-blue.svg)](https://render.com/)
[![CDN](https://img.shields.io/badge/Storage-Cloudinary_CDN-blueviolet.svg)](https://cloudinary.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An biometric identity verification and Presentation Attack Detection (PAD) platform. Designed for testing cloud deployments (Render, AWS) and seamless integration with Flutter mobile applications and web clients.

---

## 🌟 Key Features

- 👤 **High-Precision Biometric Recognition**: 128-dimensional deep metric face embeddings (dlib ResNet-34) with strict Euclidean matching (`threshold = 0.45`) and automated imposter detection.
- 🔬 **Multi-Modal Anti-Spoofing Ensemble**: Combines DeepPixBiS (DenseNet-161), Vision Transformer (ViT-Base with DINO), 2D Fast Fourier Transform (FFT) spectral decomposition, and screen moiré/glare analysis.
- ⚡ **Interactive Challenge-Response**: Real-time physiological challenge system using MediaPipe 468-point 3D FaceMesh, tracking Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), and head turn yaw angles.
- ☁️ **Serverless-Resilient Cloud Architecture**: Stateless Flask WSGI backend with on-demand PostgreSQL connection lifecycle, eliminating broken pipe / socket disconnects during container idle sleep.
- 🖼️ **Cloudinary Media Pipeline**: High-throughput parallel biometric image ingestion via `ThreadPoolExecutor`, perceptual optimization (800x800 cap), and complete asset lifecycle management.
- 🔒 **Defense-in-Depth IAM & Forensics**: Granular RBAC (`admin` / `user`), PBKDF2:SHA256 password hashing, JWT Bearer tokens, and a forensic verification logger with suspicious activity heuristics.
- ⚙️ **Dynamic Context Adaptation**: Adaptive threshold manager that scales security requirements based on device trust, rooting flags, user history, and consecutive failure counts.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    Client[Client App: Flutter / Web / Mobile] -->|HTTPS REST / JWT| Gateway[Reverse Proxy / Gunicorn Gateway]
    
    subgraph Core_API [Flask Application Layer - face_api.py]
        Gateway --> Auth[JWT Auth & RBAC Manager]
        Gateway --> Endpoints[API Route Controllers]
        
        Endpoints --> Pipeline[Unified Multi-Stage Verification Pipeline]
        Pipeline --> Quality[Frame Quality Analyzer]
        Pipeline --> FaceMatch[Face Recognition Engine]
        Pipeline --> Liveness[Multi-Modal Liveness Engine]
        Pipeline --> Challenge[Challenge-Response Engine]
    end

    subgraph Biometrics_PAD [Biometric & PAD Subsystems]
        FaceMatch --> Dlib[dlib 128D Face Encodings]
        Liveness --> CNN[DeepPixBiS DenseNet-161]
        Liveness --> ViT[Vision Transformer + DINO]
        Liveness --> Freq[2D-FFT Spectral Analysis]
        Liveness --> Static[Screen Moiré / Glare Detector]
        Liveness --> Inject[Virtual Cam / Injection Detector]
        Liveness --> Motion[Farneback Optical Flow Motion]
        Challenge --> Mesh[MediaPipe 468 Landmark 3D Mesh]
    end

    subgraph Storage_Layer [Data & Cloud Persistence Layer]
        Auth --> Postgres[(Render PostgreSQL Database)]
        FaceMatch --> Postgres
        Endpoints --> Cloudinary[(Cloudinary Media CDN)]
        Endpoints --> AuditLogger[(Verification & Security Audit Logs)]
    end
```

---

## ⚡ Quick Start (< 5 Minutes)

### 1. Clone & Environment Setup
```bash
git clone https://github.com/Silvermax12/facial_recong.git
cd facial_recong

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `env_template.txt` to `.env` and fill in your credentials:
```bash
cp env_template.txt .env
```
```env
DATABASE_URL=postgresql://user:password@host:port/dbname
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret
JWT_SECRET_KEY=generate_a_secure_random_key_min_32_chars
FLASK_ENV=development
```

### 3. Initialize Database & Create Admin
```bash
python init_auth_db.py
```

### 4. Run Development Server
```bash
python face_api.py
# Server starts at http://localhost:5000
```

---

## 📡 Core API Reference

| Endpoint | Method | Auth | Description |
| :--- | :--- | :--- | :--- |
| `/health` | `GET` | None | Service heartbeat and health check |
| `/auth/signup` | `POST` | None | Register new user account with hashed password |
| `/auth/login` | `POST` | None | Authenticate user credentials and issue JWT |
| `/admin/login` | `POST` | None | Authenticate admin credentials and issue admin JWT |
| `/admin/users` | `GET` | Admin | List all registered users (status, roles, dates) |
| `/admin/users/<user>/skip-challenges` | `PATCH` | Admin | Toggle interactive challenge requirement |
| `/enroll` | `POST` | User | Enroll 5-10 facial frames to Cloudinary and DB |
| `/v3/challenge/create` | `POST` | None | Generate randomized timed challenge or sequence |
| `/v3/challenge/verify` | `POST` | None | Verify response frames against challenge ID |
| `/v3/verify/guided` | `POST` | User | Real-time circular UI feedback & quality scoring |
| `/v3/verify/unified` | `POST` | User | **Master Verification Pipeline** (Recommended) |

> 📖 **Full API Reference**: For exhaustive request/response schemas, status codes, query parameters, and curl examples, see **[docs/05_api_specification.md](docs/05_api_specification.md)**.

---

## 📁 Repository Directory Structure

```text
facial_recong/
├── docs/                             # 📚 Exhaustive Technical Documentation
│   ├── README.md                     # Documentation index and navigation hub
│   ├── 01_system_architecture.md     # Architecture, design principles, dataflow
│   ├── 02_face_recognition_and_enrollment.md # Biometrics, embeddings, Cloudinary
│   ├── 03_liveness_detection_and_antispofing.md # PAD models, 2D-FFT, moiré, injection
│   ├── 04_interactive_challenges.md  # Challenge-response, EAR, MAR, head pose
│   ├── 05_api_specification.md       # Complete REST API reference (19 endpoints)
│   ├── 06_database_and_storage.md    # PostgreSQL schemas, migrations, Cloudinary
│   ├── 07_security_and_auditing.md   # Security model, IAM, JWT, audit logging
│   ├── 08_configuration_and_tuning.md # liveness_config.yaml, weights, profiles
│   ├── 09_deployment_and_devops.md   # Render setup, render.yaml, Gunicorn
│   ├── 10_testing_and_benchmarks.md  # Test runners, benchmarks, ISO/IEC PAD
│   └── legacy/                       # Preserved historical deployment guides
├── active_flash.py                   # Active screen illumination analysis
├── adaptive_threshold.py             # Context-aware dynamic risk thresholding
├── auth_backend_utils.py             # JWT token lifecycle and RBAC decorators
├── cloudinary_utils.py               # Cloudinary CDN management and uploads
├── database_utils.py                 # PostgreSQL client with on-demand connections
├── depth_sensing.py                  # Stereo vision and structured light analysis
├── enhanced_liveness_utils.py        # Multi-modal anti-spoofing engine
├── face_api.py                       # Main Flask web application & REST routes
├── face_auth.py                      # Biometric authentication routines
├── face_auth_utils.py                # Model loader & preprocessing utilities
├── face_auth_with_deeppixbis.py      # Standalone DeepPixBiS integration harness
├── face_challenge.py                 # Interactive challenge-response manager
├── frame_quality.py                  # Laplacian blur, brightness, contrast checks
├── init_auth_db.py                   # Database schema bootstrap & admin CLI
├── liveness_config.yaml              # Primary declarative configuration file
├── Model.py                          # DeepPixBiS (DenseNet-161) architecture
├── Model_ViT.py                      # Vision Transformer (ViT-DINO) architecture
├── motion_analysis.py                # Farneback optical flow & loop detection
├── mouth_detection.py                # Mouth Aspect Ratio (MAR) analyzer
├── performance_optimizer.py          # Frame downsampling & memory buffering
├── render.yaml                       # Render cloud Infrastructure-as-Code
├── requirements.txt                  # Production Python dependencies
├── start.py                          # Production startup entrypoint
├── train_enhanced_model.py           # Synthetic spoof generation & training
├── verification_logger.py            # Audit logging & suspicious activity engine
└── verify_setup.py                   # Hardware and environment verification
```

---

## 🚀 Cloud Deployment (Render)

This repository is optimized for deployment on **Render**:
1. Connect this repository to Render Web Services.
2. Select **Python 3** runtime (`PYTHON_VERSION=3.9.18` is preconfigured in `render.yaml`).
3. Set build command: `pip install -r requirements.txt`.
4. Set start command: `gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 face_api:app`.
5. Connect a managed **PostgreSQL** database and provide Cloudinary credentials in Environment variables.

For a detailed walkthrough, refer to **[docs/09_deployment_and_devops.md](docs/09_deployment_and_devops.md)**.

---
