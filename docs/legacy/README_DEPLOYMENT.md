# Face Recognition API - Render Deployment Guide

This guide explains how to deploy the face recognition backend to Render using Cloudinary for image storage and Render's PostgreSQL database.

## 🚀 Quick Deployment Steps

### 1. Set Up Cloudinary (Free Tier)
1. Go to [cloudinary.com](https://cloudinary.com) and create a free account
2. Get your API credentials from the Dashboard:
   - Cloud Name
   - API Key
   - API Secret

### 2. Create Render PostgreSQL Database
1. Go to [render.com](https://render.com) and sign up
2. Create a new PostgreSQL database:
   - Click "New" → "PostgreSQL"
   - Choose "Free" plan
   - Copy the connection string (it looks like: `postgresql://user:password@host:port/database`)

### 3. Deploy to Render
1. Connect your GitHub repository to Render
2. Create a new Web Service:
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python start.py`

### 4. Configure Environment Variables
In your Render service settings, add these environment variables:

```bash
# Cloudinary Configuration
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret

# Render PostgreSQL Database
DATABASE_URL=postgresql://user:password@host:port/database

# Optional
FLASK_ENV=production
PORT=10000  # Render will set this automatically
```

## 📊 Free Tier Limits

### Cloudinary Free Plan
- ✅ 10GB storage
- ✅ 20,000 monthly transformations
- ✅ 20GB monthly bandwidth
- ✅ 300,000 assets
- ⚠️ For testing only - monitor usage

### Render Free Database
- ✅ 1GB storage
- ✅ Basic PostgreSQL features
- ⚠️ Expires after 30 days (upgrade to paid to prevent data loss)

### Render Free Web Service
- ✅ 750 hours/month
- ✅ Auto-scaling to zero
- ⚠️ 15-minute spin-down (cold starts)

## 🔧 API Endpoints

### Enrollment
```bash
POST /enroll
Form data:
- username: string
- images: file[] (multiple face images)
```

### Verification
```bash
POST /verify
Form data:
- username: string
- image: file

POST /v2/verify
POST /v3/verify/sequence
POST /v3/verify/enhanced
POST /v3/verify/guided
POST /v3/verify/unified
```

## 🏗️ Architecture Changes

### Before (Local Files)
- Face encodings stored in memory (`known_encodings`, `known_names`)
- Images saved to local `known_faces/` directory
- Data lost on container restart

### After (Cloud + Database)
- Face encodings stored in PostgreSQL database
- Images uploaded to Cloudinary with URLs stored in database
- Data persists across deployments

### Database Schema
```sql
-- Users table
CREATE TABLE users (
    username VARCHAR(255) PRIMARY KEY,
    enrolled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    face_count INTEGER DEFAULT 0,
    last_verified TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Face encodings table
CREATE TABLE face_encodings (
    id SERIAL PRIMARY KEY,
    username VARCHAR(255) REFERENCES users(username) ON DELETE CASCADE,
    encoding_vector DOUBLE PRECISION[],  -- 128-dimensional face vector
    cloudinary_url VARCHAR(500),  -- Cloudinary image URL
    enrolled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## 🧪 Testing Your Deployment

### 1. Health Check
```bash
curl https://your-app-name.onrender.com/health
# Should return: {"status": "ok"}
```

### 2. Test Enrollment
```bash
curl -X POST https://your-app-name.onrender.com/enroll \
  -F "username=testuser" \
  -F "images=@face1.jpg" \
  -F "images=@face2.jpg"
```

### 3. Test Verification
```bash
curl -X POST https://your-app-name.onrender.com/verify \
  -F "username=testuser" \
  -F "image=@verify.jpg"
```

## ⚠️ Important Notes

1. **Database Expiration**: Render free databases expire after 30 days. Upgrade to paid ($7/month) to keep your data.

2. **Cold Starts**: Free tier spins down after 15 minutes of inactivity, causing 30-60 second delays.

3. **Bandwidth Monitoring**: Cloudinary free tier has 20GB bandwidth limit. Monitor usage in your dashboard.

4. **Face Encoding Storage**: We're storing 128-dimensional vectors in PostgreSQL, which works but isn't optimized for vector similarity search. For production at scale, consider:
   - Pinecone (vector database)
   - Weaviate
   - Or upgrade to a paid PostgreSQL with vector extensions

## 🔍 Troubleshooting

### Database Connection Issues
- Check `DATABASE_URL` environment variable
- Ensure database hasn't expired (30-day limit)
- Verify connection string format

### Cloudinary Upload Failures
- Check API credentials
- Verify free tier limits not exceeded
- Check network connectivity

### Face Recognition Errors
- Ensure users are enrolled before verification
- Check that face encodings were properly saved to database
- Verify image quality and face detection

## 💰 Cost Estimation (Testing Only)

- **Render**: $0 (free tier) → $7/month (persistent database)
- **Cloudinary**: $0 (free tier) → $9/month (100GB bandwidth)
- **Total for small production**: ~$16/month

## 🎯 Next Steps

For production use:
1. Upgrade Render database to paid plan
2. Monitor Cloudinary bandwidth usage
3. Consider vector database for better performance
4. Implement proper logging and monitoring
5. Add rate limiting and security measures
