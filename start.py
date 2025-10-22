#!/usr/bin/env python3
"""
Production startup script for Render deployment
"""
import os
from face_api import app

if __name__ == "__main__":
    # Get port from environment (Render provides this)
    port = int(os.environ.get("PORT", 5000))

    # Run with production server settings
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,  # Never run debug in production
        threaded=True
    )
