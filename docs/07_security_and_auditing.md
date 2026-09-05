# 07. Security Architecture, IAM & Forensic Auditing

## 1. Security Philosophy & Threat Model

The Face Recognition & Anti-Spoofing system is designed around defense-in-depth principles:
- **Zero-Trust Identity**: Every biometric request requires cryptographic authentication and identity validation.
- **Least Privilege Access**: Granular Role-Based Access Control (RBAC) separates standard users from administrative operators.
- **Immutable Forensic Auditing**: Comprehensive logging captures timestamps, IP addresses, failure reasons, and biometric anomaly scores.

---

## 2. Identity & Access Management (IAM)

```mermaid
graph TD
    Client[Client App] -->|Credentials| Login[/auth/login or /admin/login]
    Login --> VerifyPass{Verify PBKDF2 Hash}
    VerifyPass -->|Valid| IssueJWT[Generate JWT Token with Role Claim]
    VerifyPass -->|Invalid| Deny401[401 Unauthorized]
    
    Client -->|Bearer JWT| ProtectedRoute[Protected Route: /v3/verify/unified]
    ProtectedRoute --> Guard1[@token_required: Verify Signature & Expiry]
    Guard1 --> Guard2[@user_allowed_required: Check Database allowed Flag]
    Guard2 --> Guard3[@admin_required: Check role == 'admin' if needed]
    Guard3 --> Execute[Execute Endpoint Controller]
```

### 2.1 Password Hashing Architecture
User passwords are never stored in plaintext. Passwords undergo cryptographic stretching via Werkzeug using **PBKDF2 with SHA-256**:
- **Algorithm**: `pbkdf2:sha256`
- **Salt Generation**: Random cryptographically secure 16-byte salt per user.
- **Iterations**: High-iteration key stretching mitigating offline dictionary and rainbow table attacks.

### 2.2 JSON Web Token (JWT) Lifecycle
- **Signature Algorithm**: HMAC-SHA256 (`HS256`).
- **Secret Key**: Defined by environment variable `JWT_SECRET_KEY` (minimum 32 characters).
- **Default TTL**: Configurable via `JWT_ACCESS_TOKEN_EXPIRES` (Default: 24 hours).
- **Custom Claims**: Includes user identity (`sub`) and role (`role: 'user'` | `'admin'`).

### 2.3 Route Guard Decorators
- `@token_required`: Verifies presence and cryptographic validity of Bearer tokens.
- `@admin_required`: Validates that the token carries the `role: "admin"` claim.
- `@user_allowed_required`: Queries the database to verify that the user account has not been suspended or blocked (`allowed == TRUE`).

---

## 3. Forensic Auditing & Verification Logger (`verification_logger.py`)

Every authentication and verification attempt is recorded in a thread-safe, structured JSON log file (`logs/verification.log`).

### 3.1 Log Schema
```json
{
  "timestamp": "2026-09-05T13:30:00.123456",
  "username": "johndoe",
  "success": false,
  "liveness_score": 0.42,
  "quality_score": 75.0,
  "reason": "liveness_failed_high_frequency_truncation",
  "ip_address": "192.168.1.100",
  "metadata": {
    "detection_method": "ensemble",
    "is_imposter": false
  }
}
```

---

## 4. Real-Time Suspicious Activity Heuristic Engine

The `VerificationLogger` incorporates an in-memory heuristic engine that detects and flags active attack patterns:

### 4.1 Rapid Consecutive Failures
- **Trigger**: $\ge 3$ failed verification attempts for a single username within a 5-minute rolling window.
- **Flag**: `rapid_failed_attempts`.
- **Indication**: Potential credential stuffing or presentation attack trial-and-error.

### 4.2 Distributed Brute-Force from Single IP
- **Trigger**: Single IP address initiating verification attempts across $\ge 5$ distinct usernames within 30 minutes.
- **Flag**: `multiple_users_same_ip`.
- **Indication**: Automated credential testing or botnet scanning.

### 4.3 Repeated Failure Reason
- **Trigger**: $\ge 5$ consecutive failures sharing the exact same failure reason (e.g. `unnatural_motion_detected` or `injection_attack`).
- **Flag**: `repeated_same_failure`.
- **Indication**: Replay script running without adapting to detection countermeasures.

### 4.4 Suspicious Activity Log Stream
When an anomaly is flagged, the event is immediately appended to `logs/verification_suspicious.log` for SIEM ingestion and administrative review:
```json
{
  "timestamp": "2026-09-05T13:32:00.000000",
  "type": "SUSPICIOUS_ACTIVITY",
  "pattern": "rapid_failed_attempts",
  "identifier": "johndoe",
  "details": {
    "count": 4,
    "window": "5_minutes"
  }
}
```
