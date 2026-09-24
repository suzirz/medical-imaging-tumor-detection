import numpy as np
import cv2
import torch
import torch.nn as nn

class GradCAMVisualizer:
    """
    Explainable AI (XAI) Grad-CAM visualizer for PyTorch MRI tumor models.
    Hooks into convolutional layers to compute exact gradient-weighted activation maps.
    """
    def __init__(self, model, target_layer=None):
        self.model = model
        self.target_layer = target_layer or self._find_target_conv_layer()
        self.gradients = None
        self.activations = None
        self.handles = []
        self._register_hooks()

    def _find_target_conv_layer(self):
        last_conv = None
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        return last_conv

    def _register_hooks(self):
        if self.target_layer is None:
            return

        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate_heatmap(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        """
        Computes spatial activation saliency using gradient attribution.
        Returns a normalized 2D numpy array [0.0, 1.0] matching the tensor spatial dimensions.
        """
        self.model.eval()
        tensor = input_tensor.clone().detach().requires_grad_(True)
        out = self.model(tensor)

        if out.shape[-1] == 1 or (len(out.shape) == 2 and out.shape[1] == 1):
            score = out[0, 0]
        else:
            if target_class is None:
                target_class = int(torch.argmax(out, dim=1).item())
            score = out[0, target_class]

        self.model.zero_grad()
        score.backward(retain_graph=True)

        h, w = input_tensor.shape[2], input_tensor.shape[3]

        if self.gradients is not None and self.activations is not None:
            g = self.gradients.detach()
            a = self.activations.detach()

            # Element-wise positive gradient-activation attribution
            saliency = torch.sum(torch.clamp(g * a, min=0), dim=1, keepdim=True)
            cam = saliency.squeeze().cpu().numpy()

            if cam.max() > 0:
                cam = cam / cam.max()

            # Gaussian smoothing for continuous clinical gradient visualization
            smoothed = cv2.GaussianBlur(cam.astype(np.float32), (25, 25), 9)
            if smoothed.max() > 0:
                smoothed = smoothed / smoothed.max()

            heatmap = cv2.resize(smoothed, (w, h))
            return heatmap
        else:
            if tensor.grad is not None:
                inp_g = tensor.grad[0].abs().mean(dim=0).detach().cpu().numpy()
                smoothed = cv2.GaussianBlur(inp_g.astype(np.float32), (25, 25), 9)
                if smoothed.max() > 0:
                    smoothed = smoothed / smoothed.max()
                return cv2.resize(smoothed, (w, h))
            return np.zeros((h, w), dtype=np.float32)

    def overlay_on_mri(self, mri_slice, heatmap: np.ndarray, alpha: float = 0.55, threshold: float = 0.15) -> np.ndarray:
        """
        Blends heatmap with MRI scan. Only pixels above threshold show color JET,
        leaving non-salient brain tissue in clear grayscale.
        """
        if mri_slice.dtype != np.uint8:
            norm = (mri_slice - mri_slice.min()) / (mri_slice.max() - mri_slice.min() + 1e-8)
            mri_uint8 = (norm * 255).astype(np.uint8)
        else:
            mri_uint8 = mri_slice.copy()

        if len(mri_uint8.shape) == 2:
            mri_rgb = cv2.cvtColor(mri_uint8, cv2.COLOR_GRAY2RGB)
        else:
            mri_rgb = mri_uint8

        h, w = mri_rgb.shape[:2]
        if heatmap.shape[:2] != (h, w):
            heatmap = cv2.resize(heatmap, (w, h))

        # Suppress background noise below threshold
        active_map = np.clip((heatmap - threshold) / (1.0 - threshold + 1e-8), 0.0, 1.0)

        heatmap_color = cv2.applyColorMap(np.uint8(255 * active_map), cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        mask = (active_map[:, :, np.newaxis] * alpha)
        blended = (mri_rgb * (1.0 - mask) + heatmap_color * mask).astype(np.uint8)
        return blended

    def get_clinical_action(self, class_id: int, confidence: float) -> dict:
        if confidence < 0.5:
            return {
                "status": "Indeterminate",
                "recommendation": "Borderline scan signal. Requires manual radiologist evaluation with contrast-enhanced T1 sequence."
            }

        actions = {
            0: {
                "status": "Normal (No Structural Lesion)",
                "recommendation": "No abnormal structural mass lesion detected. Routine clinical surveillance as indicated."
            },
            1: {
                "status": "Glioma Signature",
                "recommendation": "Intraparenchymal mass signature. Neurosurgical oncology consult and multi-sequence MRI evaluation recommended."
            },
            2: {
                "status": "Meningioma Signature",
                "recommendation": "Extra-axial dural-based mass signature. Neuro-oncology review and volumetric serial tracking advised."
            },
            3: {
                "status": "Pituitary Tumor Signature",
                "recommendation": "Sellar / parasellar mass signature. Endocrine hormonal panel and dedicated pituitary MRI recommended."
            }
        }
        return actions.get(class_id, {
            "status": "Suspicious Lesion",
            "recommendation": "Mass effect detected. Comprehensive neuroradiological evaluation required."
        })
