import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import cv2
import numpy as np
from PIL import Image
import argparse
from Model_ViT import ViTLivenessDetector
from Model import DeePixBiS
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SpoofingDataset(Dataset):
    """
    Dataset for training liveness detection with data augmentation
    Includes synthetic spoofing attack generation
    """
    def __init__(self, real_images_dir, spoof_images_dir=None, transform=None, augment=True):
        self.real_images = self._load_images(real_images_dir, label=1)
        self.spoof_images = []

        if spoof_images_dir and os.path.exists(spoof_images_dir):
            self.spoof_images = self._load_images(spoof_images_dir, label=0)
        else:
            # Generate synthetic spoofing attacks if no real spoof data
            print("[+] Generating synthetic spoofing attacks...")
            self.spoof_images = self._generate_synthetic_spoofs(len(self.real_images))

        self.all_images = self.real_images + self.spoof_images
        self.transform = transform
        self.augment = augment

        # Data augmentation for robustness
        self.augmentation = A.Compose([
            A.Rotate(limit=15, p=0.5),
            A.GaussianBlur(blur_limit=3, p=0.3),
            A.GaussNoise(var_limit=(10, 50), p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Resize(224, 224),
            ToTensorV2()
        ]) if augment else None

    def _load_images(self, directory, label):
        images = []
        if os.path.exists(directory):
            for filename in os.listdir(directory):
                if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                    path = os.path.join(directory, filename)
                    images.append((path, label))
        return images

    def _generate_synthetic_spoofs(self, num_samples):
        """Generate synthetic spoofing attacks from real images"""
        spoof_images = []

        # Use some real images to create synthetic attacks
        real_samples = self.real_images[:min(num_samples, len(self.real_images))]

        for img_path, _ in real_samples:
            try:
                # Load image
                image = cv2.imread(img_path)
                if image is None:
                    continue

                # Apply various spoofing transformations
                spoof_variants = self._create_spoof_variants(image)

                for spoof_img in spoof_variants:
                    # Save temporary spoof image
                    temp_path = f"temp_spoof_{len(spoof_images)}.jpg"
                    cv2.imwrite(temp_path, spoof_img)
                    spoof_images.append((temp_path, 0))  # 0 = spoof

            except Exception as e:
                print(f"[!] Error generating spoof for {img_path}: {e}")

        return spoof_images[:num_samples]

    def _create_spoof_variants(self, image):
        """Create various spoofing attack variants"""
        variants = []

        # 1. Print attack simulation (add paper texture)
        print_attack = self._simulate_print_attack(image.copy())
        variants.append(print_attack)

        # 2. Screen replay attack simulation
        screen_attack = self._simulate_screen_attack(image.copy())
        variants.append(screen_attack)

        # 3. Photo attack (add shadows, perspective)
        photo_attack = self._simulate_photo_attack(image.copy())
        variants.append(photo_attack)

        # 4. Mask attack simulation (subtle distortions)
        mask_attack = self._simulate_mask_attack(image.copy())
        variants.append(mask_attack)

        return variants

    def _simulate_print_attack(self, image):
        """Simulate printed photo attack"""
        # Add paper texture and print artifacts
        noise = np.random.normal(0, 5, image.shape).astype(np.uint8)
        image = cv2.add(image, noise)

        # Add subtle color shifts (printing artifacts)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.9, 0, 255)  # Reduce saturation
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return image

    def _simulate_screen_attack(self, image):
        """Simulate screen replay attack"""
        # Add screen pixelation and RGB subpixel patterns
        height, width = image.shape[:2]

        # Pixelate image (screen resolution effect)
        pixel_size = 4
        temp = cv2.resize(image, (width // pixel_size, height // pixel_size),
                         interpolation=cv2.INTER_NEAREST)
        image = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)

        # Add subtle color fringing (RGB subpixel effect)
        b, g, r = cv2.split(image)
        r = np.roll(r, 1, axis=1)  # Slight shift in red channel
        image = cv2.merge([b, g, r])

        return image

    def _simulate_photo_attack(self, image):
        """Simulate photo attack with perspective and lighting"""
        height, width = image.shape[:2]

        # Add perspective distortion
        pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        pts2 = np.float32([[10, 5], [width-10, 0], [0, height], [width, height-5]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        image = cv2.warpPerspective(image, matrix, (width, height))

        # Add shadow/lighting effects
        shadow_mask = np.random.rand(height, width) > 0.8
        image[shadow_mask] = (image[shadow_mask] * 0.7).astype(np.uint8)

        return image

    def _simulate_mask_attack(self, image):
        """Simulate mask attack with facial distortions"""
        # Add subtle facial warping
        height, width = image.shape[:2]

        # Create a subtle warp field
        warp_field_x = np.random.normal(0, 2, (height, width)).astype(np.float32)
        warp_field_y = np.random.normal(0, 2, (height, width)).astype(np.float32)

        # Apply warp
        map_x = np.tile(np.arange(width), (height, 1)) + warp_field_x
        map_y = np.tile(np.arange(height), (width, 1)).T + warp_field_y

        image = cv2.remap(image, map_x.astype(np.float32), map_y.astype(np.float32),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        return image

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        img_path, label = self.all_images[idx]

        # Load image
        if img_path.startswith('temp_spoof_'):
            image = cv2.imread(img_path)
            os.remove(img_path)  # Clean up temp file
        else:
            image = cv2.imread(img_path)

        if image is None:
            # Return a placeholder if image loading fails
            image = np.zeros((224, 224, 3), dtype=np.uint8)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.augmentation:
            augmented = self.augmentation(image=image)
            image = augmented['image']
        elif self.transform:
            image = Image.fromarray(image)
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)


def train_enhanced_model(model_type='vit', epochs=50, batch_size=16, lr=1e-4):
    """Train enhanced liveness detection model"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Training on device: {device}")

    # Initialize model
    if model_type == 'vit':
        model = ViTLivenessDetector(pretrained=True)
    else:
        model = DeePixBiS(pretrained=True)

    model = model.to(device)

    # Create datasets
    train_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Note: You need to organize your data with real and spoof directories
    train_dataset = SpoofingDataset(
        real_images_dir="data/train/real",
        spoof_images_dir="data/train/spoof",
        transform=train_transform,
        augment=True
    )

    val_dataset = SpoofingDataset(
        real_images_dir="data/val/real",
        spoof_images_dir="data/val/spoof",
        transform=train_transform,
        augment=False
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Loss functions (pixel-wise + global classification)
    pixel_criterion = nn.BCEWithLogitsLoss()
    global_criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            if model_type == 'vit':
                pixel_out, global_out = model(images)
            else:
                pixel_out, global_out = model(images)

            # Pixel-wise supervision
            pixel_target = labels.view(-1, 1, 1, 1).expand_as(pixel_out)
            pixel_loss = pixel_criterion(pixel_out, pixel_target)

            # Global classification
            global_loss = global_criterion(global_out, labels)

            # Combined loss
            loss = pixel_loss + global_loss

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                if model_type == 'vit':
                    _, outputs = model(images)
                else:
                    _, outputs = model(images)

                predictions = (outputs > 0.5).float()
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs}")
        print(".4f")
        print(".4f")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            model_path = f"enhanced_{model_type}_liveness_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc,
            }, model_path)
            print(f"[+] Saved best model with accuracy: {val_acc:.4f}")

    print("[+] Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train enhanced liveness detection model')
    parser.add_argument('--model', type=str, default='vit', choices=['vit', 'cnn'],
                       help='Model type to train')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')

    args = parser.parse_args()

    train_enhanced_model(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
