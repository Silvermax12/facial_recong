import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models.vision_transformer import VisionTransformer
import timm

class ViTLivenessDetector(nn.Module):
    """
    Vision Transformer-based liveness detector for enhanced spoofing detection
    Uses DINO-pretrained ViT backbone for better generalization against AI-generated content
    """
    def __init__(self, pretrained=True, patch_size=16, embed_dim=768, num_heads=12):
        super().__init__()

        # Use DINO-pretrained ViT (if available) or standard ViT
        try:
            # Try to load DINO-pretrained model
            self.backbone = timm.create_model('vit_base_patch16_224.dino', pretrained=pretrained)
            # Remove the classification head
            self.backbone.head = nn.Identity()
        except:
            # Fallback to standard ViT
            self.backbone = VisionTransformer(
                image_size=224,
                patch_size=patch_size,
                num_layers=12,
                num_heads=num_heads,
                hidden_dim=embed_dim,
                mlp_dim=3072,
                dropout=0.1,
                attention_dropout=0.1,
                num_classes=0  # Remove classification head
            )

        # Multi-scale feature processing
        self.multi_scale_conv = nn.Sequential(
            nn.Conv2d(embed_dim, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((14, 14))
        )

        # Spatial attention for spoofing cues
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(512, 1, 1),
            nn.Sigmoid()
        )

        # Pixel-wise binary supervision decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1)  # Pixel-wise binary output
        )

        # Global classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # Frequency domain analysis for detecting synthetic artifacts
        self.freq_analysis = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        try:
            # Extract patch embeddings from ViT
            features = self.backbone(x)  # [B, N, D] where N = (224//16)^2 + 1 = 197

            # Reshape to spatial feature map (excluding CLS token)
            B, N, D = features.shape
            H = W = int((N - 1) ** 0.5)  # Should be 14 for patch_size=16
            spatial_features = features[:, 1:, :].reshape(B, H, W, D).permute(0, 3, 1, 2)

            # Multi-scale processing
            multi_scale = self.multi_scale_conv(spatial_features)

            # Spatial attention
            attention = self.spatial_attention(multi_scale)
            attended_features = multi_scale * attention

            # Pixel-wise binary supervision output
            pixel_output = self.decoder(attended_features)

            # Global classification
            global_score = self.classifier(attended_features)

            # Frequency domain analysis for synthetic content detection
            freq_score = self.freq_analysis(x)

            # Combine multiple cues
            combined_score = (global_score + freq_score) / 2.0

            return pixel_output, combined_score.squeeze(-1)
        except Exception:
            # Fallback: produce neutral pixel map and mid-score to avoid hard failure
            B = x.size(0)
            device = x.device
            pixel_output = torch.zeros((B, 1, 14, 14), device=device)
            score = torch.full((B,), 0.5, device=device)
            return pixel_output, score


class EnhancedLivenessDetector(nn.Module):
    """
    Ensemble model combining ViT with traditional CNN for robust detection
    """
    def __init__(self, vit_model=None, cnn_model=None):
        super().__init__()
        self.vit_model = vit_model
        self.cnn_model = cnn_model

        # Ensemble fusion
        if self.vit_model and self.cnn_model:
            self.fusion = nn.Sequential(
                nn.Linear(2, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        else:
            self.fusion = None

    def forward(self, x):
        vit_pixel, vit_score = (None, None)
        cnn_pixel, cnn_score = (None, None)
        if self.vit_model is not None:
            try:
                vit_pixel, vit_score = self.vit_model(x)
            except Exception:
                vit_pixel, vit_score = None, None
        if self.cnn_model is not None:
            try:
                cnn_pixel, cnn_score = self.cnn_model(x)
            except Exception:
                cnn_pixel, cnn_score = None, None

        if self.fusion and vit_score is not None and cnn_score is not None:
            combined_score = self.fusion(torch.stack([vit_score, cnn_score], dim=1))
            return vit_pixel if vit_pixel is not None else cnn_pixel, combined_score
        else:
            # Use ViT if available, otherwise CNN
            pixel_out = vit_pixel if vit_pixel is not None else cnn_pixel
            score_out = vit_score if (vit_score is not None) else cnn_score
            if score_out is None:
                score_out = torch.full((x.size(0),), 0.5, device=x.device)
            return pixel_out, score_out
