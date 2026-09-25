"""
Clinical DICOM PACS Protocol Simulator & 3D Multi-Planar Reconstruction (MPR) Engine.
NOTE: Network operations (C-ECHO, C-FIND, C-STORE) are simulated testbed methods
for UI workflow prototyping, not a production pynetdicom socket daemon.
The 3D MPR slicing operates locally on loaded image volumes.
"""
import os
import time
import glob
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import cv2

class DICOMPACSNode:
    """
    Simulated PACS Network Node & Testbed for workflow demonstration.
    """
    def __init__(self, ae_title: str = "NEUROSCAN_PACS", port: int = 11112, storage_dir: str = "data/pacs_storage"):
        self.ae_title = ae_title
        self.port = port
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    def ping_pacs(self, target_host: str = "192.168.1.120", target_ae: str = "HOSPITAL_CENTRAL_PACS") -> Dict[str, Any]:
        """
        Executes DICOM C-ECHO connectivity verification.
        """
        t0 = time.time()
        # Simulated network round-trip latency
        latency_ms = round(12.4 + (np.random.rand() * 4.2), 1)
        
        return {
            "status": "SUCCESS",
            "message": f"DICOM C-ECHO Verification Acknowledged by {target_ae}",
            "source_ae": self.ae_title,
            "target_ae": target_ae,
            "target_ip": target_host,
            "port": self.port,
            "latency_ms": latency_ms,
            "sop_class": "1.2.840.10008.1.1 (Verification SOP Class)",
            "presentation_context": "Accepted (Transfer Syntax: Explicit VR Little Endian)"
        }

    def query_worklist(self, patient_filter: str = "") -> List[Dict[str, Any]]:
        """
        Executes DICOM C-FIND (DICOMweb QIDO-RS) study query.
        """
        return [
            {
                "accession_no": "ACC-2026-9812",
                "patient_id": "MRN-8472910",
                "patient_name": "Rahman^Budi^Mr",
                "study_date": "2026-09-24 08:30",
                "modality": "MR",
                "series_desc": "BRAIN 3D MULTI-PARAMETRIC (T1, T1ce, T2, FLAIR)",
                "slice_count": 160,
                "scanner": "Siemens MAGNETOM Vida 3.0T",
                "status": "Ready for AI Analysis"
            },
            {
                "accession_no": "ACC-2026-9844",
                "patient_id": "MRN-1092837",
                "patient_name": "Siti^Aminah^Mrs",
                "study_date": "2026-09-24 09:15",
                "modality": "MR",
                "series_desc": "BRAIN T1 POST-CONTRAST AXIAL",
                "slice_count": 48,
                "scanner": "GE SIGNA Premier 3.0T",
                "status": "Acquiring..."
            },
            {
                "accession_no": "ACC-2026-9850",
                "patient_id": "MRN-3391028",
                "patient_name": "Wijaya^Hendra^Dr",
                "study_date": "2026-09-24 10:00",
                "modality": "MR",
                "series_desc": "BRAIN CEREBELLAR POSTERIOR FOSSA",
                "slice_count": 128,
                "scanner": "Philips Ingenia Elition 3.0T",
                "status": "Transferred"
            }
        ]


class MultiPlanarReconstruction:
    """
    3D Multi-Planar Reconstruction (MPR) Engine.
    Generates orthogonal views:
    - Axial Plane (Transverse, XY)
    - Coronal Plane (Frontal, XZ)
    - Sagittal Plane (Lateral, YZ)
    with synchronized crosshairs.
    """
    def __init__(self, base_slice: np.ndarray, num_depth_slices: int = 128):
        self.h, self.w = base_slice.shape[:2]
        self.d = num_depth_slices
        
        # Build 3D cranial volume V(z, y, x)
        self.volume = self._generate_synthetic_3d_volume(base_slice, num_depth_slices)

    def _generate_synthetic_3d_volume(self, base_slice: np.ndarray, depth: int) -> np.ndarray:
        """
        Extrapolates a 2D axial slice into a smooth 3D volumetric array V(z, y, x)
        incorporating cranial curvature and slice decay.
        """
        if len(base_slice.shape) == 3:
            gray = cv2.cvtColor(base_slice, cv2.COLOR_RGB2GRAY).astype(np.float32)
        else:
            gray = base_slice.astype(np.float32)

        vol = np.zeros((depth, self.h, self.w), dtype=np.float32)
        mid_z = depth // 2

        # Skull elliptical modulation along Z axis
        for z in range(depth):
            dz = (z - mid_z) / float(mid_z)
            # Ellipsoidal scaling factor
            scale = np.sqrt(max(0.01, 1.0 - (dz ** 2) * 0.75))
            intensity_factor = max(0.2, 1.0 - abs(dz) * 0.45)
            
            # Affine scale around center
            M = cv2.getRotationMatrix2D((self.w / 2.0, self.h / 2.0), 0, scale)
            scaled = cv2.warpAffine(gray, M, (self.w, self.h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            vol[z] = scaled * intensity_factor

        return vol

    def get_orthogonal_slices(self, x: int, y: int, z: int) -> Dict[str, np.ndarray]:
        """
        Extracts Axial, Coronal, and Sagittal orthogonal cuts at (x, y, z)
        and draws calibrated crosshairs.
        """
        x = min(max(0, x), self.w - 1)
        y = min(max(0, y), self.h - 1)
        z = min(max(0, z), self.d - 1)

        # 1. Axial Slice (XY at Z)
        axial = self.volume[z, :, :].copy()
        axial_rgb = cv2.cvtColor(cv2.normalize(axial, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        # Draw Crosshair
        cv2.line(axial_rgb, (x, 0), (x, self.h - 1), (56, 189, 248), 1)
        cv2.line(axial_rgb, (0, y), (self.w - 1, y), (56, 189, 248), 1)
        cv2.circle(axial_rgb, (x, y), 3, (244, 63, 94), -1)

        # 2. Coronal Slice (XZ at Y)
        # Slices across Z (vertical) and X (horizontal)
        coronal = self.volume[:, y, :].copy()
        # Aspect ratio correction (Z depth vs X width)
        coronal_resized = cv2.resize(coronal, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        coronal_rgb = cv2.cvtColor(cv2.normalize(coronal_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        cz = int((z / float(self.d)) * self.h)
        cv2.line(coronal_rgb, (x, 0), (x, self.h - 1), (16, 185, 129), 1)
        cv2.line(coronal_rgb, (0, cz), (self.w - 1, cz), (16, 185, 129), 1)
        cv2.circle(coronal_rgb, (x, cz), 3, (244, 63, 94), -1)

        # 3. Sagittal Slice (YZ at X)
        sagittal = self.volume[:, :, x].copy()
        sagittal_resized = cv2.resize(sagittal, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        sagittal_rgb = cv2.cvtColor(cv2.normalize(sagittal_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        sz = int((z / float(self.d)) * self.h)
        cv2.line(sagittal_rgb, (y, 0), (y, self.h - 1), (168, 85, 247), 1)
        cv2.line(sagittal_rgb, (0, sz), (self.w - 1, sz), (168, 85, 247), 1)
        cv2.circle(sagittal_rgb, (y, sz), 3, (244, 63, 94), -1)

        return {
            "axial": axial_rgb,
            "coronal": coronal_rgb,
            "sagittal": sagittal_rgb,
            "coordinates": {"x": x, "y": y, "z": z, "depth_total": self.d}
        }
