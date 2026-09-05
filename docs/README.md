# 📚 System Documentation Index

Welcome to the comprehensive technical documentation for the **Face Recognition & Multi-Modal Anti-Spoofing (PAD) API**. This documentation provides an in-depth reference for software architects, machine learning engineers, backend developers, mobile app integrators, and system administrators.

---

## 🧭 Documentation Map

| Chapter | Topic | Description | Key Source Modules |
| :--- | :--- | :--- | :--- |
| **[01. System Architecture](01_system_architecture.md)** | Core Architecture | High-level system design, modular layers, dataflow, deployment modes, and fallback hierarchy | ace_api.py, start.py |
| **[02. Face Recognition & Enrollment](02_face_recognition_and_enrollment.md)** | Biometrics & Identity | 128-D Euclidean embeddings, matching thresholds, multi-image enrollment pipeline, imposter detection | ace_recognition, ace_auth.py, ace_auth_utils.py |
| **[03. Liveness & Anti-Spoofing](03_liveness_detection_and_antispofing.md)** | Passive & Active PAD | DeepPixBiS CNN, Vision Transformer (ViT), 2D-FFT frequency spectrum, moiré/screen detection, injection defense | Model.py, Model_ViT.py, enhanced_liveness_utils.py |
| **[04. Interactive Challenges](04_interactive_challenges.md)** | Active Challenge-Response | MediaPipe 468-point FaceMesh, Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), head pose yaw/pitch, admin bypass | ace_challenge.py, mouth_detection.py |
| **[05. API Specification](05_api_specification.md)** | REST API Reference | Exhaustive reference for all 19 endpoints across Auth, Admin, v1, v2, and v3 APIs with request/response schemas | ace_api.py |
| **[06. Database & Storage Architecture](06_database_and_storage.md)** | Data Persistence | PostgreSQL relational schema, on-demand connection pooling, vector storage, automatic migrations, Cloudinary media CDN | database_utils.py, cloudinary_utils.py, init_auth_db.py |
| **[07. Security & Auditing](07_security_and_auditing.md)** | Security Architecture | JWT token lifecycle, PBKDF2:SHA256 password hashing, RBAC, audit logging, and suspicious activity heuristic engine | uth_backend_utils.py, erification_logger.py |
| **[08. Configuration & Tuning](08_configuration_and_tuning.md)** | Configuration Engine | Parameter breakdown of liveness_config.yaml, ensemble weights, risk context adaptive thresholding, hardware profiles | liveness_config.yaml, daptive_threshold.py |
| **[09. Deployment & DevOps](09_deployment_and_devops.md)** | Production Deployment | Render cloud deployment, 
ender.yaml, Gunicorn worker tuning, Python 3.9 MediaPipe ABI compatibility, cold start mitigation | 
ender.yaml, start.py, 
equirements.txt |
| **[10. Testing & Verification](10_testing_and_benchmarks.md)** | Test Suites & Quality | Synthetic attack augmentation, offline verification scripts, ISO/IEC 30107-3 compliance metrics (APCER, BPCER, ACER) | 	est_*.py, 	rain_enhanced_model.py |
| **[Legacy Documentation](legacy/)** | Historical Reference | Original deployment guides, checklists, and preliminary development observations | docs/legacy/* |

---

## 🏗️ High-Level System Architecture

\\mermaid
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
\
---

## 🚀 Quick Navigation by User Role

- **Frontend / Flutter Engineers**: Start with [05. API Specification](05_api_specification.md) and [04. Interactive Challenges](04_interactive_challenges.md).
- **Backend / DevOps Engineers**: Start with [01. System Architecture](01_system_architecture.md), [06. Database & Storage](06_database_and_storage.md), and [09. Deployment & DevOps](09_deployment_and_devops.md).
- **ML / Biometrics Engineers**: Start with [02. Face Recognition & Enrollment](02_face_recognition_and_enrollment.md), [03. Liveness & Anti-Spoofing](03_liveness_detection_and_antispofing.md), and [10. Testing & Verification](10_testing_and_benchmarks.md).
- **Security & Compliance Officers**: Start with [07. Security & Auditing](07_security_and_auditing.md) and [08. Configuration & Tuning](08_configuration_and_tuning.md).
