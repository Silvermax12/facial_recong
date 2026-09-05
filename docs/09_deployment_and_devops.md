# 09. Deployment, Cloud Infrastructure & DevOps

## 1. Cloud Architecture Overview

The system is containerized and hosted as a cloud microservice on **Render**, backed by a managed **PostgreSQL** instance and **Cloudinary CDN**:

```mermaid
graph TD
    subgraph Client_Applications [Client Ecosystem]
        FlutterApp[Flutter Mobile App - iOS / Android]
        AdminDashboard[Web Administrative Console]
    end

    subgraph Render_Cloud [Render PaaS Infrastructure]
        LB[Render Managed Load Balancer / SSL Termination]
        
        subgraph Web_Service [Web Service: face-recognition-api]
            Gunicorn[Gunicorn WSGI Master - Port $PORT]
            Worker1[Worker 1 - Flask App Instance]
            Worker2[Worker 2 - Flask App Instance]
        end
        
        subgraph DB_Instance [Managed Database]
            PG[(Render PostgreSQL - face-recognition-db)]
        end
    end

    subgraph Third_Party_SaaS [External Cloud Services]
        Cloudinary[(Cloudinary Media CDN - Asset Delivery)]
    end

    FlutterApp -->|HTTPS / REST| LB
    AdminDashboard -->|HTTPS / REST| LB
    LB --> Gunicorn
    Gunicorn --> Worker1
    Gunicorn --> Worker2
    Worker1 -->|On-demand TCP| PG
    Worker2 -->|On-demand TCP| PG
    Worker1 -->|HTTPS Upload| Cloudinary
    Worker2 -->|HTTPS Upload| Cloudinary
```

---

## 2. Render Infrastructure as Code (`render.yaml`)

The repository includes a declarative `render.yaml` blueprint defining the service configuration:

```yaml
services:
  - type: web
    name: face-recognition-api
    runtime: python3
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 face_api:app
    envVars:
      - key: PYTHON_VERSION
        value: 3.9.18
      - key: FLASK_ENV
        value: production
```

### 2.1 Why Python 3.9.18 is Pinned
`PYTHON_VERSION` is explicitly pinned to **`3.9.18`**:
- MediaPipe releases for Linux (`mediapipe==0.9.3.1`) have binary dependencies compiled against glibc and Python 3.9 C-extension ABIs.
- Python 3.11+ and 3.12+ introduce breaking changes in Python C-API internals, causing segmentation faults when running MediaPipe FaceMesh on headless Linux containers.
- Pinned Python 3.9 guarantees 100% build repeatability across cloud redeployments.

---

## 3. Web Server & Concurrency Tuning

### 3.1 Gunicorn WSGI Configuration
```bash
gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 face_api:app
```
- `--workers 2`: On Render's 512 MB Free / Starter instances, 2 worker processes maximize CPU utilization during PyTorch and OpenCV inference without risking Out-Of-Memory (OOM) process termination.
- `--timeout 120`: Multi-frame liveness verification with face encoding extraction across 5-10 images can take 2-4 seconds on CPU. A 120-second timeout prevents Gunicorn worker killing during cold starts or transient load spikes.
- `--bind 0.0.0.0:$PORT`: Dynamically binds to the port assigned by Render's environment.

---

## 4. Step-by-Step Deployment Runbook

### Step 1: Create Managed PostgreSQL on Render
1. Navigate to [dashboard.render.com](https://dashboard.render.com).
2. Click **New +** → **PostgreSQL**.
3. Name: `face-recognition-db`.
4. Plan: **Free** (or Starter for permanent retention).
5. Copy the **Internal Database URL** (e.g. `postgresql://user:pass@dpg-xxxx-a:5432/db`).

### Step 2: Create Web Service on Render
1. Click **New +** → **Web Service**.
2. Connect your GitHub repository (`Silvermax12/facial_recong`).
3. Name: `face-recognition-api`.
4. Runtime: **Python 3**.
5. Build Command: `pip install -r requirements.txt`.
6. Start Command: `gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 face_api:app`.

### Step 3: Inject Environment Variables
Under the **Environment** tab of your Render Web Service, add:
- `DATABASE_URL`: *(Your Render PostgreSQL Internal URL)*
- `CLOUDINARY_CLOUD_NAME`: *(From Cloudinary Console)*
- `CLOUDINARY_API_KEY`: *(From Cloudinary Console)*
- `CLOUDINARY_API_SECRET`: *(From Cloudinary Console)*
- `JWT_SECRET_KEY`: *(Generate with `python -c "import secrets; print(secrets.token_hex(32))"`)*
- `FLASK_ENV`: `production`
- `PYTHON_VERSION`: `3.9.18`

### Step 4: Initialize the Database
Once the service shows **Live**:
1. Open the **Shell** tab in the Render Dashboard.
2. Run the interactive initialization CLI:
   ```bash
   python init_auth_db.py
   ```
3. When prompted, create your primary administrator credentials.

### Step 5: Validate Health Endpoint
```bash
curl https://face-recognition-api.onrender.com/health
# Expected Output: {"status": "ok"}
```

---

## 5. Free-Tier Behavior & Cold Start Mitigation

Render Free Tier instances spin down after 15 minutes of inactivity. When a new request arrives, a cold start occurs (taking ~30–50 seconds to spin up).

### Cold Start Mitigation Strategies:
1. **Automated Heartbeat Cron**: Configure a free external monitor (e.g. [UptimeRobot](https://uptimerobot.com) or [cron-job.org](https://cron-job.org)) to perform a `GET /health` request every **10 minutes**. This keeps the web container warm continuously.
2. **On-Demand DB Connections**: As documented in Chapter 06, our code creates PostgreSQL connections per transaction, completely eliminating `Broken pipe` or stale connection crashes after server sleeps.
3. **Database Expiry Management**: Render free-tier databases expire after 90 days. For production permanence, upgrade the database to the Starter plan ($7/mo) or point `DATABASE_URL` to Supabase / Neon / AWS RDS.
