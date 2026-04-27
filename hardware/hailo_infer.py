"""
Hailo HAT+ inference pipeline.
Designed for Raspberry Pi 5 with Hailo-8L (13 TOPS) or Hailo-8 (26 TOPS).
Uses HailoRT Python API + rpicam-apps integration via PCIe Gen 3.0.
"""
import numpy as np
import cv2

try:
    from hailo_platform import (
        HEF, Device, VDevice, HailoStreamInterface,
        InferVStreams, ConfigureParams, InputVStreamParams,
        OutputVStreamParams, FormatType,
    )
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False


class HailoInference:
    """Run compiled HEF models on Hailo AI accelerator."""

    def __init__(self, model_path, batch_size=1):
        self.model_path = model_path
        self.batch_size = batch_size
        self.device = None
        self.hef = None
        self.network_group = None
        self.input_vstreams = None
        self.output_vstreams = None
        self._input_shape = None

        if HAILO_AVAILABLE:
            self._init_hailo()
        else:
            print(f"[Hailo] HailoRT not available — running in CPU fallback.")

    def _init_hailo(self):
        """Initialize Hailo device, load HEF, configure streams."""
        try:
            self.device = VDevice()
            self.hef = HEF(self.model_path)

            configure_params = ConfigureParams.create_from_hef(
                self.hef, interface=HailoStreamInterface.PCIe)
            self.network_group = self.device.configure(
                self.hef, configure_params)[0]

            input_params = InputVStreamParams.make_from_network_group(
                self.network_group, quantized=False,
                format_type=FormatType.FLOAT32)
            output_params = OutputVStreamParams.make_from_network_group(
                self.network_group, quantized=False,
                format_type=FormatType.FLOAT32)

            self.input_vstreams = input_params
            self.output_vstreams = output_params

            # Get expected input shape
            vstream_info = self.hef.get_input_vstream_infos()
            if vstream_info:
                shape = vstream_info[0].shape
                self._input_shape = (shape[1], shape[2])  # H, W
                print(f"[Hailo] Model loaded: {self.model_path}")
                print(f"[Hailo] Input shape: {self._input_shape}")
            else:
                print("[Hailo] Warning: could not read input shape.")

        except Exception as e:
            print(f"[Hailo] Init failed: {e}")
            self.network_group = None

    def preprocess(self, frame):
        """Resize and normalize frame for Hailo input."""
        if self._input_shape is None:
            return frame.astype(np.float32) / 255.0

        resized = cv2.resize(frame, (self._input_shape[1],
                                      self._input_shape[0]))
        blob = resized.astype(np.float32) / 255.0
        return np.expand_dims(blob, axis=0)  # add batch dim

    def infer(self, frame):
        """Run inference on a single BGR frame.

        Returns
        -------
        dict: raw output tensors keyed by output layer name,
              or empty dict if Hailo is unavailable.
        """
        if not HAILO_AVAILABLE or self.network_group is None:
            return {"boxes": [], "classes": [], "scores": []}

        input_data = self.preprocess(frame)

        try:
            with InferVStreams(self.network_group,
                               self.input_vstreams,
                               self.output_vstreams) as pipeline:
                input_dict = {
                    self.hef.get_input_vstream_infos()[0].name: input_data
                }
                raw_output = pipeline.infer(input_dict)
                return raw_output
        except Exception as e:
            print(f"[Hailo] Inference error: {e}")
            return {"boxes": [], "classes": [], "scores": []}

    def infer_batch(self, frames):
        """Run batched inference for higher throughput."""
        results = []
        for frame in frames:
            results.append(self.infer(frame))
        return results
