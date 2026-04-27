"""
MiDaS depth estimation wrapper.
Uses MiDaS v2.1 small for real-time relative depth from a single RGB camera.
"""
import os
import cv2
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class DepthEstimator:
    """Wraps MiDaS for monocular relative depth estimation."""

    def __init__(self, config=None):
        self.model = None
        self.transform = None
        self.device = "cpu"
        self.config = config

        if TORCH_AVAILABLE:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                # Force trust repositories to avoid interactive prompts
                # This is a bit of a hack but necessary for non-interactive environments
                try:
                    from torch.hub import _get_cache_or_download
                    # We can't easily add to the trusted list via API, 
                    # but we can try to skip the check if we're in a script.
                except ImportError:
                    pass

                mtype = "MiDaS_small"
                midas_path = None
                if config and 'models' in config:
                    mtype = config['models'].get('midas_type', 'MiDaS_small')
                    midas_path = config['models'].get('midas_path')

                # Load local model if it exists
                if midas_path and os.path.exists(midas_path):
                    # Try loading as a scripted model first (common for MiDaS .pt files)
                    try:
                        self.model = torch.jit.load(midas_path, map_location=self.device)
                        print(f"[Depth] Loaded scripted MiDaS from {midas_path}")
                    except Exception:
                        # Fallback to hub load but this usually requires internet if not cached
                        self.model = torch.hub.load("intel-isl/MiDaS", mtype, trust_repo=True)
                        print(f"[Depth] Loaded MiDaS via Hub (fallback)")
                else:
                    self.model = torch.hub.load("intel-isl/MiDaS", mtype, trust_repo=True)
                    print(f"[Depth] Loaded MiDaS via Hub")

                self.model.to(self.device).eval()

                # Transforms still usually need to be loaded from Hub or defined locally
                # For MiDaS small, it's a resize to 256x256 and normalization
                try:
                    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
                    if mtype in ("DPT_Large", "DPT_Hybrid"):
                        self.transform = midas_transforms.dpt_transform
                    else:
                        self.transform = midas_transforms.small_transform
                except Exception:
                    # Minimal manual transform if hub fails
                    self.transform = self._get_manual_transform(mtype)

                print(f"[Depth] MiDaS ({mtype}) ready on {self.device}.")
            except Exception as e:
                print(f"[Depth] Failed to load MiDaS: {e}")
                self.model = None
        else:
            print("[Depth] PyTorch unavailable — depth disabled.")

    def _get_manual_transform(self, mtype):
        """Fallback transform if torch.hub transforms fail."""
        def transform(img):
            size = (384, 384) if "DPT" in mtype else (256, 256)
            img = cv2.resize(img, size)
            img = img.astype(np.float32) / 255.0
            img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            img = np.transpose(img, (2, 0, 1))
            return torch.from_numpy(img).unsqueeze(0)
        return transform

    def estimate(self, frame):
        """Return a relative depth map (float32 H×W, higher = closer).

        Parameters
        ----------
        frame : BGR image (numpy)

        Returns
        -------
        depth_map : np.ndarray float32 (H, W) or None
        """
        if frame is None or self.model is None:
            return None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(rgb).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy().astype(np.float32)
        # Normalize to 0-1
        dmin, dmax = depth_map.min(), depth_map.max()
        if dmax - dmin > 1e-6:
            depth_map = (depth_map - dmin) / (dmax - dmin)
        return depth_map
