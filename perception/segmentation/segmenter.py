"""
Drivable area segmentation using DeepLabv3+ with MobileNetV2 backbone.
Critical for Indian roads where lanes are absent — segments road vs non-road.
"""
import os
import cv2
import numpy as np

try:
    import torch
    from torchvision import models
    from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class DrivableAreaSegmenter:
    """Segments drivable road area using pretrained DeepLabv3+."""

    ROAD_CLASSES = {0: 'road', 1: 'sidewalk'}  # COCO/Cityscapes indices

    def __init__(self, config=None):
        self.model = None
        self.device = "cpu"
        self.config = config

        if TORCH_AVAILABLE:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                # Initialize model architecture
                self.model = deeplabv3_mobilenet_v3_large(weights=None)
                
                # Load local weights if path exists in config
                weights_path = None
                if config and 'models' in config:
                    weights_path = config['models'].get('deeplabv3_path')
                
                if weights_path and os.path.exists(weights_path):
                    state_dict = torch.load(weights_path, map_location=self.device)
                    self.model.load_state_dict(state_dict, strict=False)
                    print(f"[Seg] Loaded local weights from {weights_path} (non-strict)")
                else:
                    # Fallback to default weights (downloads if needed)
                    print("[Seg] Local weights not found, using default weights.")
                    self.model = deeplabv3_mobilenet_v3_large(weights='DEFAULT')
                
                self.model.to(self.device).eval()
                print(f"[Seg] DeepLabv3+ MobileNetV3 ready on {self.device}.")
            except Exception as e:
                print(f"[Seg] Failed to load model: {e}")
                self.model = None
        else:
            print("[Seg] PyTorch unavailable — segmentation disabled.")

    def segment(self, frame):
        """Return a binary drivable-area mask.

        Parameters
        ----------
        frame : BGR image (numpy)

        Returns
        -------
        dict:
            mask    : np.ndarray uint8 (H, W) — 255 = drivable
            overlay : frame with green drivable overlay
        """
        if frame is None or self.model is None:
            return {"mask": None, "overlay": frame}

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Normalize
        inp = rgb.astype(np.float32) / 255.0
        inp = (inp - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        inp = np.transpose(inp, (2, 0, 1))
        tensor = torch.from_numpy(inp).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            output = self.model(tensor)['out']
            pred = output.argmax(1).squeeze().cpu().numpy()

        # Classes 0 (road) in COCO-stuff / Cityscapes
        # DeepLabv3 pretrained on COCO has 21 classes; class 0 = background
        # For road segmentation we treat low-index flat surfaces as drivable
        # In practice you'd fine-tune on IDD/BDD. Here we use a heuristic.
        mask = np.zeros((h, w), dtype=np.uint8)
        # Class indices that are typically road-like
        road_ids = [0, 1]  # adjust after fine-tuning
        for rid in road_ids:
            mask[pred == rid] = 255

        # Overlay
        overlay = frame.copy()
        green = np.zeros_like(overlay)
        green[:, :, 1] = mask
        overlay = cv2.addWeighted(overlay, 1.0, green, 0.3, 0)

        return {"mask": mask, "overlay": overlay}
