# 01. System Architecture & Design Principles

## 1. Executive Summary & Mission
The **Face Recognition & Multi-Modal Anti-Spoofing (PAD) System** is an enterprise-grade biometric verification platform. Designed for mission-critical client-server ecosystems (including Flutter mobile apps and web clients), the system provides dual guarantees:
1. **Biometric Identity Authentication**: Accurately matches presented facial geometry against enrolled 128-dimensional identity vectors.
2. **Presentation Attack Detection (PAD)**: Protects against presentation attacks across ISO/IEC 30107-3 attack categories, including 2D print attacks, digital screen replay attacks, 3D masks, generative AI/deepfakes, and virtual camera injection attacks.

---

## 2. Core Architectural Principles

### 2.1 Zero-Trust Biometrics
No biometric input is trusted by default. Every transaction undergoes validation across multiple orthogonal physical and mathematical dimensions:
- **Spatial Texture**: Neural network feature extraction (CNN & ViT).
- **Frequency Domain**: 2D Fast Fourier Transform (FFT) analysis detecting artificial high-frequency cutoff and JPEG blocking artifacts.
- **Temporal Consistency**: Motion coherence and inter-frame structural variance across consecutive frames.
- **Physiological Response**: Biological cues including involuntary micro-blinks, pupil behavior, and voluntary challenge-responses.
- **Environmental Interaction**: Screen reflection, active flash illumination, and specular highlight response.

### 2.2 Stateless Server with Ephemeral Session Detectors
To support serverless and containerized cloud deployment (e.g. Render, AWS Fargate):
- All persistent data resides in PostgreSQL and Cloudinary.
- Interactive multi-frame verification workflows rely on isolated, time-decaying session buffers (_session_detectors) indexed by cryptographically secure session IDs with automatic garbage collection (5-minute TTL).
- Database connections are created **on-demand** to avoid stale connection pooling failures in serverless environments where services suspend or spin down on inactivity.

---

## 3. Layered Architectural Decomposition

The system is organized into five distinct architectural layers:

\\mermaid
graph TD
    subgraph Layer1 [1. Client & Ingestion Layer]
        Mobile[Flutter iOS / Android Client]
        Web[Web Applications / Admin UI]
    end

    subgraph Layer2 [2. API Gateway & Security Perimeter]
        Gunicorn[Gunicorn WSGI Server - 2 Workers]
        Flask[Flask REST Controller - face_api.py]
        JWTAuth[Flask-JWT-Extended RBAC]
        Audit[Verification & Security Audit Logger]
    end

    subgraph Layer3 [3. Biometric & PAD Processing Pipeline]
        Quality[Frame Quality Analyzer - Laplacian Blur & Luminance]
        Align[Face Landmark Detection & Alignment]
        dlib[dlib ResNet-34 128D Face Embedding]
        PAD_Ensemble[Multi-Modal PAD Ensemble]
        Challenge_Engine[Interactive Challenge-Response Engine]
    end

    subgraph Layer4 [4. AI / ML Inference Engine]
        CNN[DeepPixBiS - DenseNet-161 Backbone]
        ViT[ViTLivenessDetector - DINO-Pretrained ViT]
        FFT[Frequency Domain 2D-FFT Analyzer]
        Moiré[Screen Glare & Moiré Pattern Detector]
        OpticalFlow[Farneback Optical Flow Analyzer]
        MediaPipe[MediaPipe 468-Point 3D Face Mesh]
    end

    subgraph Layer5 [5. Persistence & Infrastructure Layer]
        PG[(PostgreSQL Database - Render)]
        Cloudinary[(Cloudinary Media CDN)]
        LocalConfig[YAML Configuration & Profiles]
    end

    Mobile -->|HTTPS / Multipart Form| Gunicorn
    Web -->|HTTPS / JSON / JWT| Gunicorn
    Gunicorn --> Flask
    Flask --> JWTAuth
    Flask --> Audit
    Flask --> Quality
    Quality --> Align
    Align --> dlib
    Align --> PAD_Ensemble
    Align --> Challenge_Engine
    PAD_Ensemble --> CNN
    PAD_Ensemble --> ViT
    PAD_Ensemble --> FFT
    PAD_Ensemble --> Moiré
    PAD_Ensemble --> OpticalFlow
    Challenge_Engine --> MediaPipe
    Flask --> PG
    Flask --> Cloudinary
    Flask --> LocalConfig
\
---

## 4. End-to-End Verification Dataflow

The unified verification workflow (/v3/verify/unified) coordinates quality checks, identity matching, passive PAD, and dynamic challenge-response:

\\mermaid
sequenceDiagram
    autonumber
    actor User as Mobile Client (Flutter)
    participant API as Flask API Gateway (/v3/verify/unified)
    participant Auth as JWT & RBAC Guard
    participant QA as Frame Quality Analyzer
    participant DB as PostgreSQL Database
    participant Bio as Face Recognition Engine
    participant PAD as Multi-Modal PAD Ensemble
    participant Chal as Challenge Manager

    User->>API: POST /v3/verify/unified (JWT, frames, username, challenge_ids)
    API->>Auth: Validate JWT & User Allowed Status
    Auth-->>API: Authorized
    
    API->>QA: Analyze First & Last Frames (Laplacian, Brightness)
    alt Poor Quality (Blurry / Dark / Multiple Faces)
        QA-->>API: Quality Failure (Quality Score < 60)
        API-->>User: 400 Bad Request with corrective instruction
    end

    API->>PAD: Quick Liveness Check on Frame 0
    PAD-->>API: quick_score
    
    API->>DB: Fetch Enrolled Face Encodings (username)
    DB-->>API: 128D Encodings List
    API->>Bio: Calculate Euclidean Distances against Frame 0
    alt Face Distance >= 0.45 (No Match)
        Bio-->>API: Imposter Detected (dmin >= 0.45)
        API-->>User: 403 Forbidden (Imposter Alert)
    end

    alt User Configured to Skip Challenges (Admin Bypass)
        API->>API: Mark Challenge as Skipped (Score = 1.0)
    else Challenges Provided
        API->>Chal: Verify Frames Against Challenge IDs (Blink, Smile, Turn)
        Chal-->>API: Challenge Score & Landmark Metrics
    end

    par Parallel PAD Verification
        API->>PAD: Sequence Analysis (Temporal Consistency)
        API->>PAD: Enhanced Motion Analysis (Farneback Optical Flow)
        API->>PAD: Guided Verification (Laplacian Variance Trend)
    end
    PAD-->>API: Component Scores (Sequence, Enhanced, Guided)

    API->>API: Compute Weighted Fusion Score
    alt Final Score >= Adaptive Threshold (Default: 0.75)
        API->>DB: Update last_verified Timestamp
        API->>API: Log Success Audit Record
        API-->>User: 200 OK (Verification Passed, Detailed Breakdown)
    else Final Score < Threshold
        API->>API: Log Failure Audit Record & Track Suspicious Heuristics
        API-->>User: 200 OK / 403 Forbidden (Verification Failed, Score Breakdown)
    end
\
---

## 5. Modular Deployment Modes

To guarantee optimal operation across constrained edge hardware and high-performance cloud clusters, the system dynamically activates components according to the selected hardware profile:

| Deployment Mode | Target Hardware | Active Subsystems | Latency Target | Security Level |
| :--- | :--- | :--- | :--- | :--- |
| **Full High-Security** | Server with GPU / Dual Stereo Camera | ViT-DINO + DeepPixBiS + Stereo Depth + Active Flash + Challenge Sequence | < 800 ms (GPU) | **Tier 4 (Banking / Gov)** |
| **Standard Cloud (Default)** | Render Cloud CPU (512MB RAM, 1-2 vCPU) | DeepPixBiS CNN + 2D-FFT + Temporal + MediaPipe Challenges | 1.2s - 2.5s (CPU) | **Tier 3 (Enterprise Auth)** |
| **Mobile / Lite CPU** | Low-end Cloud or Edge Microservice | Quantized CNN + Basic Motion + Single Challenge | < 600 ms (CPU) | **Tier 2 (Standard App)** |
| **Edge Embedded** | On-Device Android / iOS (CoreML/TFLite) | Embedded MobileNet/CNN + MediaPipe FaceMesh on-device | < 150 ms (NPU) | **Tier 2 (On-device Offline)** |

---

## 6. Graceful Degradation Hierarchy

When hardware or environmental conditions prevent full multi-modal analysis, the engine gracefully degrades down the hierarchy:

\Level 1: Full Ensemble (ViT + CNN + Depth + Flash + Challenges)
    │
    ▼ (No Stereo / Flash Hardware)
Level 2: Standard Multi-Modal (DeepPixBiS + 2D-FFT + Optical Flow + Challenges)
    │
    ▼ (Single Frame Upload / Network Constraint)
Level 3: Strict Static Mode (High-Threshold CNN + 2D-FFT + Penalty Factor +15%)
    │
    ▼ (Missing Database Encodings)
Level 4: Rejection & Enrollment Required (HTTP 403)
\
---

## 7. Technology Stack & Component Specifications

| Component | Library / Framework | Version | Functionality |
| :--- | :--- | :--- | :--- |
| **Web Framework** | Flask | 3.0.x | REST API routing, multipart file decoding, CORS |
| **WSGI Server** | Gunicorn | 21.x | Multi-worker process management for Linux/Render |
| **Authentication** | Flask-JWT-Extended | 4.6.x | JWT token creation, cryptographic verification, RBAC |
| **Computer Vision** | OpenCV (opencv-python-headless) | 4.8.x | Color transformations, Laplacian variance, Farneback optical flow |
| **Face Recognition** | face_recognition (dlib) | 1.3.x | HOG/CNN face location detection, 128D ResNet vector extraction |
| **Deep Learning** | PyTorch & Torchvision | 2.1.x | Tensor computation, DeepPixBiS DenseNet-161, ViT |
| **Facial Mesh** | MediaPipe | 0.9.3.1 / 0.10.x | 468 3D facial landmarks, EAR and MAR calculation |
| **Signal Processing** | SciPy | 1.11.x | 2D Fast Fourier Transform (FFT2) spectral decomposition |
| **Relational Database** | PostgreSQL via psycopg2-binary | 2.9.x | Encodings storage (DOUBLE PRECISION[]), auth users |
| **Cloud Object CDN** | Cloudinary SDK | 1.36.x | Secure cloud storage, image optimization (800x800, WebP/JPG) |
| **Configuration** | PyYAML | 6.0.x | System parameter configuration and hardware profiles |
