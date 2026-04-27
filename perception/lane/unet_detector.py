"""
U-Net based lane segmentation (from open-adas).
Uses a lightweight U-Net trained on Mapillary Vistas or BDD100K lane masks.
Outputs a binary segmentation mask of lane lines, which is then
post-processed to extract polylines.
"""

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ========================================================================= #
#  Lightweight U-Net architecture (from open-adas design)                    #
# ========================================================================= #

class _DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class _Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            _DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class _Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = _DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = _DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class _OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super( _OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class LaneUNet(nn.Module):
    """Standard U-Net for binary lane-line segmentation."""
    def __init__(self, n_channels=3, n_classes=1, bilinear=False):
        super(LaneUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = _DoubleConv(n_channels, 64)
        self.down1 = _Down(64, 128)
        self.down2 = _Down(128, 256)
        self.down3 = _Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = _Down(512, 1024 // factor)
        self.up1 = _Up(1024, 512 // factor, bilinear)
        self.up2 = _Up(512, 256 // factor, bilinear)
        self.up3 = _Up(256, 128 // factor, bilinear)
        self.up4 = _Up(128, 64, bilinear)
        self.outc = _OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return torch.sigmoid(logits)


# ========================================================================= #
#  Detector wrapper                                                          #
# ========================================================================= #

class UNetLaneDetector:
    """Wraps the U-Net model for lane mask prediction and polyline extraction."""

    def __init__(self, config):
        self.config = config
        self.input_size = tuple(config['models'].get('unet_input_size', [256, 256]))
        self.model = None
        self.device = "cpu"

        if TORCH_AVAILABLE:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            # Matching the checkpoint's 2-class output (e.g., background and lane)
            self.model = LaneUNet(n_channels=3, n_classes=2).to(self.device)
            model_path = config['models'].get('unet_lane_path')
            if model_path:
                try:
                    state = torch.load(model_path, map_location=self.device, weights_only=True)
                    self.model.load_state_dict(state)
                    print(f"[UNet-Lane] Loaded weights from {model_path}")
                except FileNotFoundError:
                    print(f"[UNet-Lane] Weights not found at {model_path}; using random init.")
            self.model.eval()
        else:
            print("[UNet-Lane] PyTorch not available — lane segmentation disabled.")

        print("[UNet-Lane] U-Net Lane Detector initialized.")

    def detect(self, frame):
        """Run lane segmentation on a BGR frame.

        Returns
        -------
        dict:
            mask      : binary lane mask resized to original frame size (uint8)
            polylines : list of np arrays, each shape (N,2) — extracted lane lines
            overlay   : frame with lane mask drawn on it
        """
        if frame is None or self.model is None:
            return {"mask": None, "polylines": [], "overlay": frame}

        orig_h, orig_w = frame.shape[:2]

        # Pre-process
        resized = cv2.resize(frame, self.input_size)
        blob = resized.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))  # HWC → CHW
        tensor = torch.from_numpy(blob).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            pred = self.model(tensor)

        # Post-process: If 2 classes, index 1 is usually the lane mask
        if pred.shape[1] == 2:
            # Using channel 1 as the positive class
            mask_small = (pred[0, 1, :, :].cpu().numpy() > 0.5).astype(np.uint8) * 255
        else:
            mask_small = (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255

        mask = cv2.resize(mask_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # Extract polylines via contours
        polylines = self._mask_to_polylines(mask)

        # Overlay
        overlay = frame.copy()
        color_mask = np.zeros_like(overlay)
        color_mask[:, :, 1] = mask  # green channel
        overlay = cv2.addWeighted(overlay, 1.0, color_mask, 0.4, 0)

        return {"mask": mask, "polylines": polylines, "overlay": overlay}

    def _mask_to_polylines(self, mask):
        """Extract polylines from a binary mask using contour finding."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polylines = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 200:
                continue
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            polylines.append(approx.reshape(-1, 2))
        return polylines
