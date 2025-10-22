import cloudinary
import cloudinary.uploader
import cloudinary.api
import os
from typing import List, Tuple, Optional
import requests
from io import BytesIO
import cv2
import numpy as np

class CloudinaryManager:
    def __init__(self):
        # Configure Cloudinary
        cloudinary.config(
            cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
            api_key=os.getenv('CLOUDINARY_API_KEY'),
            api_secret=os.getenv('CLOUDINARY_API_SECRET')
        )

        # Check if configuration is valid
        if not all([
            os.getenv('CLOUDINARY_CLOUD_NAME'),
            os.getenv('CLOUDINARY_API_KEY'),
            os.getenv('CLOUDINARY_API_SECRET')
        ]):
            print("[!] Cloudinary credentials not found in environment variables")
            print("   Required: CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET")
            self.configured = False
        else:
            self.configured = True
            print("[+] Cloudinary configured successfully")

    def upload_face_image(self, image_bytes: bytes, username: str,
                         image_index: int, timestamp: int) -> Optional[str]:
        """Upload a face image to Cloudinary and return the URL"""
        if not self.configured:
            print("[!] Cloudinary not configured")
            return None

        try:
            # Create a unique public_id for the image
            public_id = f"faces/{username}/{timestamp}_{image_index}"

            # Upload to Cloudinary
            result = cloudinary.uploader.upload(
                image_bytes,
                public_id=public_id,
                folder="face_recognition/faces",
                resource_type="image",
                format="jpg",
                quality="auto",  # Optimize quality
                width=800,  # Resize for storage efficiency
                height=800,
                crop="limit"
            )

            print(f"[+] Uploaded image for {username}: {result['secure_url']}")
            return result['secure_url']

        except Exception as e:
            print(f"[!] Cloudinary upload failed: {e}")
            return None

    def upload_multiple_faces(self, image_bytes_list: List[bytes], username: str,
                             timestamp: int) -> List[str]:
        """Upload multiple face images and return their URLs"""
        urls = []
        for i, image_bytes in enumerate(image_bytes_list):
            url = self.upload_face_image(image_bytes, username, i, timestamp)
            if url:
                urls.append(url)
            else:
                print(f"[!] Failed to upload image {i} for {username}")

        print(f"[+] Uploaded {len(urls)}/{len(image_bytes_list)} images for {username}")
        return urls

    def download_face_image(self, cloudinary_url: str) -> Optional[np.ndarray]:
        """Download a face image from Cloudinary and return as numpy array"""
        if not self.configured:
            return None

        try:
            # Download the image
            response = requests.get(cloudinary_url, timeout=10)
            response.raise_for_status()

            # Convert to numpy array
            image_bytes = BytesIO(response.content)
            image_array = np.frombuffer(image_bytes.getvalue(), np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if image is not None:
                return image
            else:
                print(f"[!] Failed to decode image from {cloudinary_url}")
                return None

        except Exception as e:
            print(f"[!] Failed to download image from {cloudinary_url}: {e}")
            return None

    def download_user_faces(self, cloudinary_urls: List[str]) -> List[np.ndarray]:
        """Download all face images for a user"""
        images = []
        for url in cloudinary_urls:
            image = self.download_face_image(url)
            if image is not None:
                images.append(image)

        print(f"[+] Downloaded {len(images)}/{len(cloudinary_urls)} face images")
        return images

    def delete_user_faces(self, username: str) -> bool:
        """Delete all face images for a user from Cloudinary"""
        if not self.configured:
            return False

        try:
            # Delete the entire folder for the user
            folder_path = f"face_recognition/faces/{username}/"
            result = cloudinary.api.delete_resources_by_prefix(folder_path)

            deleted_count = result.get('deleted_counts', {}).get('original', 0)
            print(f"[+] Deleted {deleted_count} images for user {username}")

            # Also delete the folder
            try:
                cloudinary.api.delete_folder(folder_path)
            except:
                pass  # Folder deletion might fail if not empty

            return True

        except Exception as e:
            print(f"[!] Failed to delete user faces: {e}")
            return False

    def get_storage_usage(self) -> dict:
        """Get current storage usage statistics"""
        if not self.configured:
            return {}

        try:
            usage = cloudinary.api.usage()
            return {
                'storage_used': usage.get('storage', {}).get('usage', 0),
                'storage_limit': usage.get('storage', {}).get('limit', 0),
                'bandwidth_used': usage.get('bandwidth', {}).get('usage', 0),
                'bandwidth_limit': usage.get('bandwidth', {}).get('limit', 0),
                'transformations_used': usage.get('transformations', {}).get('usage', 0),
                'transformations_limit': usage.get('transformations', {}).get('limit', 0)
            }

        except Exception as e:
            print(f"[!] Failed to get storage usage: {e}")
            return {}

# Global Cloudinary manager instance
cloudinary_manager = CloudinaryManager()
