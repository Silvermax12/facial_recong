# 05. REST API Specification & Reference

This document provides the complete API specification for all 19 endpoints in the Face Recognition & PAD service.

- **Base URL**: `https://<your-render-domain>.onrender.com` (or `http://localhost:5000` locally)
- **Content Types**: `application/json` or `multipart/form-data`
- **Security Scheme**: HTTP Bearer JWT (`Authorization: Bearer <access_token>`)

---

## 1. System & Health Endpoints

### 1.1 GET `/health`
Check service health and availability.
- **Auth**: None
- **Response `200 OK`**:
```json
{
  "status": "ok"
}
```

---

## 2. Authentication & User Management

### 2.1 POST `/auth/signup`
Register a new user account with hashed password credentials.
- **Auth**: None
- **Content-Type**: `application/json`
- **Request Body**:
```json
{
  "username": "johndoe",
  "password": "SecurePassword123!",
  "display_name": "John Doe"
}
```
- **Response `201 Created`**:
```json
{
  "success": true,
  "message": "User registered successfully",
  "username": "johndoe"
}
```
- **Errors**: `400 Bad Request` (Missing fields or username already taken).

---

### 2.2 POST `/auth/login`
Authenticate with username and password to obtain a JWT Bearer token.
- **Auth**: None
- **Content-Type**: `application/json`
- **Request Body**:
```json
{
  "username": "johndoe",
  "password": "SecurePassword123!"
}
```
- **Response `200 OK`**:
```json
{
  "success": true,
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "username": "johndoe",
  "role": "user",
  "display_name": "John Doe"
}
```
- **Errors**: `401 Unauthorized` (Invalid credentials), `403 Forbidden` (Account blocked).

---

### 2.3 POST `/admin/login`
Specialized authentication for administrator portal access.
- **Auth**: None
- **Content-Type**: `application/json`
- **Request Body**: Same as `/auth/login`
- **Response `200 OK`**: Same as `/auth/login` (Guaranteed `role: "admin"`).
- **Errors**: `403 Forbidden` if user role is not `admin`.

---

### 2.4 GET `/check_access`
Validate whether the caller's JWT token is valid and the account is not blocked.
- **Auth**: `Bearer <token>`
- **Response `200 OK`**:
```json
{
  "allowed": true,
  "username": "johndoe",
  "message": "Access granted"
}
```
- **Errors**: `401 Unauthorized` (Token invalid/expired), `403 Forbidden` (Account blocked).

---

### 2.5 GET `/admin/users`
List all registered authentication accounts (Admin only).
- **Auth**: `Bearer <admin_token>`
- **Response `200 OK`**:
```json
{
  "users": [
    {
      "id": "johndoe",
      "username": "johndoe",
      "display_name": "John Doe",
      "role": "user",
      "allowed": true,
      "skip_challenges": false,
      "created_at": "2025-10-22T12:00:00",
      "last_login": "2025-10-22T13:30:00"
    }
  ]
}
```

---

### 2.6 PATCH `/admin/users/<username>`
Block or unblock a user account.
- **Auth**: `Bearer <admin_token>`
- **Content-Type**: `application/json`
- **Request Body**:
```json
{
  "allowed": false
}
```
- **Response `200 OK`**:
```json
{
  "message": "User johndoe blocked successfully",
  "user": { ... }
}
```

---

### 2.7 PATCH `/admin/users/<username>/skip-challenges`
Toggle whether a user skips interactive liveness challenges.
- **Auth**: `Bearer <admin_token>`
- **Content-Type**: `application/json`
- **Request Body**:
```json
{
  "skip_challenges": true
}
```
- **Response `200 OK`**:
```json
{
  "message": "Challenge skipping enabled for user johndoe",
  "user": { ... }
}
```

---

## 3. Biometric Enrollment

### 3.1 POST `/enroll`
Enroll multiple facial images for a user. Extracts 128D embeddings, uploads images to Cloudinary, and saves vectors to PostgreSQL.
- **Auth**: `Bearer <token>` (User must not be blocked)
- **Content-Type**: `multipart/form-data`
- **Form Fields**:
  - `username`: string (Required)
  - `images`: file[] (Multiple image files, recommended 5-10 frames)
- **Response `200 OK`**:
```json
{
  "status": "ok",
  "user": "johndoe",
  "enrolled": 8,
  "uploaded": 8,
  "message": "Successfully enrolled 8 face(s) for user 'johndoe'"
}
```
- **Errors**: `400 Bad Request` (No images, or missing username), `500 Internal Server Error` (Database or Cloudinary misconfigured).

---

## 4. Verification Endpoints

### 4.1 POST `/v3/verify/unified` (Recommended Production Endpoint)
The master verification pipeline executing quality checks, identity matching, multi-frame sequence analysis, motion coherence, guided checks, and challenge verification.
- **Auth**: `Bearer <token>`
- **Content-Type**: `multipart/form-data`
- **Form Fields**:
  - `username`: string (Required)
  - `frames`: file[] (Minimum 5 frames required)
  - `challenge_ids`: string[] (Optional list of challenge IDs completed)
  - `session_id`: string (Optional session identifier)
- **Response `200 OK`**:
```json
{
  "api_version": "3.0-unified",
  "session_id": "sess_k9fA82bL...",
  "success": true,
  "username": "johndoe",
  "final_score": 0.842,
  "threshold": 0.75,
  "challenge_ran": true,
  "quick_check_score": 0.88,
  "effective_weights": {
    "sequence": 0.30,
    "enhanced": 0.30,
    "guided": 0.25,
    "challenge": 0.15
  },
  "endpoint_scores": {
    "sequence": 0.86,
    "enhanced": 0.82,
    "guided": 0.89,
    "challenge": 1.00
  },
  "endpoints_used": ["sequence", "enhanced", "guided", "challenge"],
  "breakdown": {
    "sequence": { "score": 0.86, "frames_analyzed": 5 },
    "enhanced": { "score": 0.82, "motion_detected": true },
    "guided": { "score": 0.89, "quality_check": true },
    "challenge": { "score": 1.0, "challenges_verified": 2, "challenges_passed": 2 }
  }
}
```
- **Errors**:
  - `400 Bad Request`: Fewer than 5 frames or missing username.
  - `403 Forbidden`: Face distance $\ge 0.45$ (Imposter detected) or user not enrolled.

---

### 4.2 POST `/v3/challenge/create`
Create a randomized liveness challenge or sequence of challenges.
- **Auth**: None (or optional)
- **Content-Type**: `application/json` or form-data
- **Body / Form**:
  - `username`: string (Optional, checks `skip_challenges`)
  - `session_id`: string (Optional)
  - `num_challenges`: integer (Optional: 2 or 3 for sequence)
- **Response `200 OK` (Single Challenge)**:
```json
{
  "challenge_id": "chal_x87Hs9...",
  "type": "blink",
  "instruction": "Please blink 2-3 times",
  "expires_in": 30
}
```
- **Response `200 OK` (Sequence)**:
```json
{
  "sequence_id": "seq_m39Xq...",
  "challenges": [
    { "challenge_id": "c1...", "type": "head_turn_left", "instruction": "Turn your head slowly to the left" },
    { "challenge_id": "c2...", "type": "smile", "instruction": "Please smile naturally" }
  ],
  "total_challenges": 2,
  "expires_in": 30
}
```

---

### 4.3 POST `/v3/challenge/verify`
Verify user response frames against a specific challenge ID.
- **Auth**: None
- **Content-Type**: `multipart/form-data`
- **Form Fields**:
  - `challenge_id`: string (Required)
  - `response_frames`: file[] (Image frames captured during action)
- **Response `200 OK`**:
```json
{
  "success": true,
  "challenge_type": "blink",
  "details": { "blink_count": 2, "min_ear": 0.18 }
}
```

---

### 4.4 POST `/v3/verify/guided`
Provides real-time frame quality analysis and step-by-step instructions for circular UI overlays.
- **Auth**: `Bearer <token>`
- **Content-Type**: `multipart/form-data`
- **Form Fields**:
  - `username`: string (Required)
  - `step`: string (`quality_check` or `complete`)
  - `frames`: file[] (1+ frames)
- **Response `200 OK`**:
```json
{
  "api_version": "3.0-guided",
  "step": "quality_check",
  "quality_score": 82.5,
  "status": "good",
  "instruction": "Hold still, verifying...",
  "metrics": {
    "blur_score": 142.6,
    "brightness": 128.4,
    "contrast": 64.2,
    "faces_detected": 1,
    "is_sharp": true,
    "is_well_lit": true
  }
}
```

---

### 4.5 POST `/v3/verify/enhanced`
Enhanced verification with Farneback optical flow motion coherence and adaptive thresholding.
- **Auth**: `Bearer <token>`
- **Form Fields**: `username`, `frames` (min 5), `session_id`, optional risk context fields (`device_trusted`, `device_rooted`, `user_login_count`, `recent_failed_attempts`).
- **Response `200 OK`**: Includes `avg_liveness_score`, `threshold_used`, `risk_level`, and `motion_analysis`.

---

### 4.6 POST `/v3/verify/sequence`
Evaluates temporal consistency and blink dynamics across a multi-frame video sequence.
- **Auth**: `Bearer <token>`
- **Form Fields**: `username`, `frames` (min 3), `session_id`.

---

### 4.7 POST `/v2/verify`
Version 2 verification endpoint with full multi-modal feature scoring and detailed breakdown.
- **Auth**: `Bearer <token>`
- **Form Fields**: `username`, `image` (single file), `session_id`.

---

### 4.8 POST `/v2/flash/challenge` & `/v2/flash/verify`
- **`/v2/flash/challenge`**: Issues a server-generated randomized color flash pattern (e.g. Red-Blue-White pulse sequence).
- **`/v2/flash/verify`**: Evaluates facial skin surface reflection color changes corresponding to the commanded flash pattern.

---

### 4.9 POST `/verify` (v1 Legacy)
Original v1 verification endpoint executing DeepPixBiS inference and face matching against database encodings.
- **Auth**: `Bearer <token>`
- **Form Fields**: `username`, `image` (single file).
