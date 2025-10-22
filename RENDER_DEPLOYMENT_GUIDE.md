# 🚀 Complete Guide: Deploy Flask Face Recognition API to Render

## 📋 Table of Contents
1. [Render Free Tier Overview](#render-free-tier-overview)
2. [Prerequisites](#prerequisites)
3. [Step-by-Step Deployment](#step-by-step-deployment)
4. [Environment Variables Setup](#environment-variables-setup)
5. [Database Setup](#database-setup)
6. [Post-Deployment Configuration](#post-deployment-configuration)
7. [Troubleshooting](#troubleshooting)

---

## 🆓 Render Free Tier Overview

### ✅ What's Included (FREE)
- **750 hours/month** of web service runtime
- **512 MB RAM** per service
- **Automatic HTTPS** with SSL certificates
- **Automatic deployments** from GitHub
- **Custom domains** (optional)
- **PostgreSQL database** (90 days, then expires - need to upgrade)

### ⚠️ Free Tier Limitations
1. **Service Spin-down**: Your service will spin down after 15 minutes of inactivity
   - **First request after spin-down takes 30-60 seconds** to wake up
   - Good for personal use, but not for production apps needing instant response
2. **PostgreSQL**: Free for 90 days, then you need to upgrade ($7/month for starter)
3. **Build time**: Limited to 15 minutes

### 💡 Is Free Tier Good for Personal Use?
**YES!** Perfect for:
- ✅ Personal projects
- ✅ Development/testing
- ✅ Low-traffic applications
- ✅ Portfolio demonstrations

**Note**: If you need instant response times, consider upgrading to the $7/month plan.

---

## 📦 Prerequisites

### 1. GitHub Account
- Sign up at [github.com](https://github.com) if you don't have one

### 2. Render Account
- Sign up at [render.com](https://render.com) (use GitHub to sign up - it's easier!)

### 3. Cloudinary Account (for image storage)
- Sign up at [cloudinary.com](https://cloudinary.com)
- Free tier: 25GB storage, 25GB bandwidth/month

---

## 🔧 Step-by-Step Deployment

### Step 1: Prepare Your Code for Git

1. **Navigate to your project folder**:
   ```bash
   cd c:\Users\slive\Desktop\Address\facial_recong
   ```

2. **Initialize Git repository** (if not already done):
   ```bash
   git init
   ```

3. **Add all files**:
   ```bash
   git add .
   ```

4. **Commit your code**:
   ```bash
   git commit -m "Initial commit - Face Recognition API"
   ```

### Step 2: Create GitHub Repository

1. Go to [github.com](https://github.com) and click **"New repository"**
2. **Repository name**: `face-recognition-api` (or any name you prefer)
3. **Privacy**: Choose **Private** (recommended for API keys)
4. **DON'T** initialize with README (you already have code)
5. Click **"Create repository"**

### Step 3: Push Code to GitHub

1. **Copy the commands** from GitHub's "push an existing repository" section
2. Run in your terminal:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/face-recognition-api.git
   git branch -M main
   git push -u origin main
   ```

### Step 4: Create PostgreSQL Database on Render

1. Go to [dashboard.render.com](https://dashboard.render.com)
2. Click **"New +"** → **"PostgreSQL"**
3. **Name**: `face-recognition-db`
4. **Database**: `face_recognition`
5. **User**: Auto-generated (keep it)
6. **Region**: Choose closest to you
7. **Plan**: **Free**
8. Click **"Create Database"**
9. **IMPORTANT**: Copy the **"Internal Database URL"** - you'll need this!

### Step 5: Deploy Web Service on Render

1. Go to [dashboard.render.com](https://dashboard.render.com)
2. Click **"New +"** → **"Web Service"**
3. Click **"Connect GitHub account"** (if first time)
4. Select your repository: **face-recognition-api**
5. Fill in the details:
   - **Name**: `face-recognition-api`
   - **Region**: Same as database
   - **Branch**: `main`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 face_api:app`
6. **Instance Type**: **Free**
7. Click **"Create Web Service"** (Don't deploy yet!)

---

## 🔐 Environment Variables Setup

Before the first deployment, you need to add environment variables:

### Step 6: Add Environment Variables

1. In your Render web service dashboard, go to **"Environment"** tab
2. Click **"Add Environment Variable"**
3. Add the following variables:

#### Required Variables:

| Key | Value | Where to Get It |
|-----|-------|-----------------|
| `DATABASE_URL` | Your PostgreSQL Internal URL | From Step 4 - Render database dashboard |
| `CLOUDINARY_CLOUD_NAME` | Your cloud name | Cloudinary dashboard → Account Details |
| `CLOUDINARY_API_KEY` | Your API key | Cloudinary dashboard → Account Details |
| `CLOUDINARY_API_SECRET` | Your API secret | Cloudinary dashboard → Account Details |
| `JWT_SECRET_KEY` | Random secret (generate below) | Generate using command below |
| `FLASK_ENV` | `production` | Manual entry |
| `PYTHON_VERSION` | `3.11.0` | Manual entry |

#### Generate JWT Secret Key:
Run this in PowerShell to generate a secure random key:
```powershell
python -c "import secrets; print(secrets.token_hex(32))"
```

### Step 7: Deploy!

1. After adding all environment variables, click **"Save Changes"**
2. Your app will automatically deploy
3. Wait 5-10 minutes for the build to complete
4. Once deployed, you'll get a URL like: `https://face-recognition-api.onrender.com`

---

## 🗄️ Database Setup

### Step 8: Initialize Database Tables

After deployment, you need to create the database tables:

1. **Get your service URL**: `https://face-recognition-api.onrender.com`
2. **Run initialization script**:

   You can either:
   
   **Option A: Use Render Shell**
   - Go to your web service dashboard
   - Click **"Shell"** tab
   - Run:
     ```bash
     python init_auth_db.py
     ```

   **Option B: SSH from your computer**
   - Install Render CLI: `npm install -g render-cli`
   - Login: `render login`
   - SSH: `render ssh face-recognition-api`
   - Run: `python init_auth_db.py`

3. **Create admin user** when prompted

---

## 🔄 Post-Deployment Configuration

### Step 9: Update Flutter App with New URL

Update the base URL in your Flutter apps:

**File**: `here_address_validator/lib/services/api_service.dart`
```dart
static const String baseUrl = 'https://face-recognition-api.onrender.com';
```

**File**: `here_address_validator/lib/services/face_verify_service.dart`
```dart
return 'https://face-recognition-api.onrender.com';
```

### Step 10: Test Your Deployment

1. **Health Check**:
   ```
   https://face-recognition-api.onrender.com/health
   ```
   Should return: `{"status": "ok"}`

2. **Test from Flutter app**:
   - Open your app
   - Try to login
   - Try to enroll
   - Try to verify

---

## 🐛 Troubleshooting

### Common Issues:

#### 1. Build Failed
- Check build logs in Render dashboard
- Ensure all dependencies in `requirements.txt` are compatible
- Check Python version matches

#### 2. Service Crashes on Start
- Check the **Logs** tab in Render dashboard
- Common causes:
  - Missing environment variables
  - Database connection failed
  - Port binding issue

#### 3. "Application Error" or 503
- Service is probably sleeping (free tier)
- Wait 30-60 seconds and refresh
- Check logs for actual errors

#### 4. Database Connection Failed
- Verify `DATABASE_URL` is correct
- Use **Internal Database URL** (not External)
- Check database is running in Render dashboard

#### 5. Slow First Response (30-60 seconds)
- **This is normal** on free tier
- Service spins down after 15 minutes of inactivity
- Consider upgrading to paid tier ($7/month) for instant response

#### 6. Enrollment/Verification Timing Out
- Increased timeout to 90 seconds in Flutter app ✅
- Gunicorn timeout set to 120 seconds ✅
- If still timing out, check Cloudinary upload speed

---

## 📊 Monitoring Your Application

### View Logs
1. Go to your web service dashboard
2. Click **"Logs"** tab
3. See real-time logs

### View Metrics
1. Go to your web service dashboard
2. Click **"Metrics"** tab
3. See CPU, Memory, Request counts

### Set Up Alerts
1. Go to **"Settings"** → **"Notifications"**
2. Add email for deploy failures/crashes

---

## 💰 Cost Management

### Stay Within Free Tier:
- ✅ 1 web service (free)
- ✅ 1 PostgreSQL database (free for 90 days)
- ⚠️ After 90 days, upgrade database to $7/month or migrate data

### When to Consider Upgrading:
- Need instant response (no spin-down)
- Higher traffic
- Need more than 512 MB RAM
- Production application

**Starter Plan**: $7/month
- No spin-down
- 512 MB RAM
- Faster response times

---

## 🎉 Success!

Your Face Recognition API is now live on Render!

**Your API URL**: `https://face-recognition-api.onrender.com`

### Next Steps:
1. ✅ Test all endpoints
2. ✅ Update Flutter apps with new URL
3. ✅ Test enrollment and verification
4. ✅ Monitor logs for any issues
5. ✅ Set up email notifications

---

## 📝 Additional Resources

- [Render Documentation](https://render.com/docs)
- [Render Status Page](https://status.render.com)
- [Render Community](https://community.render.com)
- [PostgreSQL on Render](https://render.com/docs/databases)

---

## 🆘 Need Help?

If you encounter any issues:
1. Check the **Logs** in Render dashboard
2. Review **Troubleshooting** section above
3. Check Render [Community Forum](https://community.render.com)
4. Contact Render Support (they're very responsive!)

Good luck with your deployment! 🚀

