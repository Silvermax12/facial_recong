# 🚀 Quick Start: Deploy to Render in 15 Minutes

## TL;DR - Render Free Tier for Personal Use

**YES! ✅** Render's free tier is **perfect for personal use**:
- ✅ **FREE** web service (750 hours/month)
- ✅ **FREE** PostgreSQL for 90 days
- ✅ Automatic HTTPS
- ✅ Automatic deployments
- ⚠️ Service sleeps after 15 minutes (30-60s wake up time)

**After 90 days**: Upgrade database to $7/month (keep web service free if you want)

---

## 🎯 5-Step Deployment

### 1️⃣ Push to GitHub (5 min)
```bash
cd c:\Users\slive\Desktop\Address\facial_recong
git init
git add .
git commit -m "Initial commit"
```
Create repo on GitHub (private), then:
```bash
git remote add origin https://github.com/YOUR_USERNAME/face-recognition-api.git
git branch -M main
git push -u origin main
```

### 2️⃣ Create Database on Render (2 min)
1. Go to [dashboard.render.com](https://dashboard.render.com)
2. New + → PostgreSQL → Free tier
3. **SAVE** the Internal Database URL!

### 3️⃣ Deploy Web Service (3 min)
1. New + → Web Service → Connect GitHub repo
2. Settings:
   - Build: `pip install -r requirements.txt`
   - Start: `gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 face_api:app`
   - Instance: Free

### 4️⃣ Add Environment Variables (3 min)
Go to Environment tab, add:
```
DATABASE_URL = (paste from step 2)
CLOUDINARY_CLOUD_NAME = (from cloudinary.com)
CLOUDINARY_API_KEY = (from cloudinary.com)
CLOUDINARY_API_SECRET = (from cloudinary.com)
JWT_SECRET_KEY = (generate: python -c "import secrets; print(secrets.token_hex(32))")
FLASK_ENV = production
```

### 5️⃣ Initialize Database (2 min)
1. Web Service → Shell tab
2. Run: `python init_auth_db.py`
3. Create admin user

---

## ✅ Done! Test It

**Health check**: `https://YOUR-APP.onrender.com/health`

**Update Flutter apps**: Change base URL to your Render URL

---

## 📚 Full Documentation

For detailed guide with screenshots and troubleshooting:
- See `RENDER_DEPLOYMENT_GUIDE.md`
- See `DEPLOYMENT_CHECKLIST.md`

---

## 💡 Tips

**First request slow?** Normal on free tier (service was sleeping)

**Need faster response?** Upgrade to $7/month Starter plan (no sleep)

**Running out of database days?** Upgrade to $7/month PostgreSQL

**Perfect for personal use!** ✅

