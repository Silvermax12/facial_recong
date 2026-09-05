# 📋 Render Deployment Checklist

## ✅ Pre-Deployment

- [ ] Create Cloudinary account and get credentials
  - [ ] Cloud name
  - [ ] API key
  - [ ] API secret

- [ ] Create GitHub account
- [ ] Create Render account (use GitHub login)

## ✅ Code Preparation

- [ ] Navigate to `facial_recong` folder
- [ ] Initialize Git: `git init`
- [ ] Add files: `git add .`
- [ ] Commit: `git commit -m "Initial commit"`

## ✅ GitHub Setup

- [ ] Create new private repository on GitHub
- [ ] Name: `face-recognition-api`
- [ ] Don't initialize with README
- [ ] Copy remote add command
- [ ] Push code to GitHub

## ✅ Render Database Setup

- [ ] Go to dashboard.render.com
- [ ] Click "New +" → "PostgreSQL"
- [ ] Name: `face-recognition-db`
- [ ] Plan: Free
- [ ] Region: Choose closest
- [ ] Create database
- [ ] **COPY Internal Database URL** (save it!)

## ✅ Render Web Service Setup

- [ ] Click "New +" → "Web Service"
- [ ] Connect GitHub account
- [ ] Select repository: `face-recognition-api`
- [ ] Name: `face-recognition-api`
- [ ] Region: Same as database
- [ ] Runtime: Python 3
- [ ] Build Command: `pip install -r requirements.txt`
- [ ] Start Command: `gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 face_api:app`
- [ ] Instance Type: Free
- [ ] Click "Create Web Service" (DON'T DEPLOY YET)

## ✅ Environment Variables (CRITICAL!)

Add these in Render → Environment tab:

- [ ] `DATABASE_URL` = (Paste Internal Database URL from above)
- [ ] `CLOUDINARY_CLOUD_NAME` = (Your Cloudinary cloud name)
- [ ] `CLOUDINARY_API_KEY` = (Your Cloudinary API key)
- [ ] `CLOUDINARY_API_SECRET` = (Your Cloudinary API secret)
- [ ] `JWT_SECRET_KEY` = (Generate: `python -c "import secrets; print(secrets.token_hex(32))"`)
- [ ] `FLASK_ENV` = production
- [ ] `PYTHON_VERSION` = 3.11.0

## ✅ Deploy

- [ ] Save environment variables
- [ ] Wait for automatic deployment (5-10 minutes)
- [ ] Check build logs for errors
- [ ] Wait for "Live" status

## ✅ Database Initialization

- [ ] Go to web service → Shell tab
- [ ] Run: `python init_auth_db.py`
- [ ] Create admin user when prompted
- [ ] Note admin credentials

## ✅ Testing

- [ ] Test health endpoint: `https://YOUR-APP.onrender.com/health`
- [ ] Should return: `{"status": "ok"}`
- [ ] Note your app URL

## ✅ Flutter App Update

- [ ] Update `api_service.dart`: Set `baseUrl` to Render URL
- [ ] Update `face_verify_service.dart`: Update `resolveBaseUrl()` default
- [ ] Rebuild Flutter apps
- [ ] Test login
- [ ] Test enrollment
- [ ] Test verification

## 🎉 Deployment Complete!

Your app URL: `https://______________________.onrender.com`

---

## ⚠️ Important Notes

**Free Tier Behavior:**
- Service sleeps after 15 minutes of inactivity
- First request after sleep takes 30-60 seconds
- Database free for 90 days only

**Costs After Free Tier:**
- Free: $0/month (with 15-min spin-down)
- Starter: $7/month (no spin-down, faster)
- Database: $7/month (after 90 days)

**For Personal Use:** Free tier is perfect! ✅

