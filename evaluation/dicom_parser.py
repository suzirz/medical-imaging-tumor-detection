"""
Native DICOM Medical PACS Ingestion & Radiologist Window-Leveling Engine.
Parses native .dcm files, extracts metadata tags, and applies standardized
radiology window/level presets (Brain, Subdural, Stroke, Bone).
"""
import io
from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

# Standard Clinical Window / Level Presets (in Hounsfield / MRI normalized units)
WINDOW_PRESETS = {
    "Brain Window (Optimal Parenchyma Contrast)": {"width": 80, "level": 40},
    "Subdural Window (Hematoma & Dural Tail)": {"width": 300, "level": 100},
    "Stroke Window (Early Ischemic Cytotoxic Edema)": {"width": 40, "level": 40},
    "Bone / Calvarium Window (Hyperostosis Assessment)": {"width": 1500, "level": 300},
    "Full Dynamic Range (Linear Min-Max)": {"width": 255, "level": 128}
}

class DICOMPACSParser:
    """
    Parser for DICOM Part 10 binary files with multi-parametric metadata extraction
    and interactive radiologist window leveling.
    """

    def __init__(self):
        pass

    def parse_dicom(self, file_source: Any) -> Dict[str, Any]:
        """
        Parses a DICOM file from file path, bytes, or BytesIO.
        """
        if isinstance(file_source, (str, bytes)):
            if isinstance(file_source, str):
                ds = pydicom.dcmread(file_source)
            else:
                ds = pydicom.dcmread(io.BytesIO(file_source))
        else:
            # Assumed BytesIO or UploadedFile
            ds = pydicom.dcmread(file_source)

        # Extract Raw Pixel Array
        raw_pixels = ds.pixel_array.astype(np.float32)

        # Apply Rescale Slope & Intercept if present (standard CT/MR calibration)
        slope = getattr(ds, "RescaleSlope", 1.0)
        intercept = getattr(ds, "RescaleIntercept", 0.0)
        calibrated_pixels = raw_pixels * float(slope) + float(intercept)

        # Extract Metadata Tags
        metadata = {
            "patient_id": str(getattr(ds, "PatientID", "ANONYMIZED-001")),
            "patient_name": str(getattr(ds, "PatientName", "ANONYMIZED^PATIENT")),
            "study_date": str(getattr(ds, "StudyDate", "2026-09-24")),
            "modality": str(getattr(ds, "Modality", "MR")),
            "scanner_manufacturer": str(getattr(ds, "Manufacturer", "Siemens Healthineers MAGNETOM Prisma")),
            "magnetic_field_strength": f"{getattr(ds, 'MagneticFieldStrength', 3.0)} Tesla",
            "sequence_name": str(getattr(ds, "SeriesDescription", "AXIAL T1 BRAIN SE")),
            "repetition_time_tr": f"{getattr(ds, 'RepetitionTime', 2000.0)} ms",
            "echo_time_te": f"{getattr(ds, 'EchoTime', 25.0)} ms",
            "slice_thickness": f"{getattr(ds, 'SliceThickness', 3.0)} mm",
            "pixel_spacing": f"{getattr(ds, 'PixelSpacing', [0.4688, 0.4688])}",
            "rows": int(getattr(ds, "Rows", raw_pixels.shape[0])),
            "columns": int(getattr(ds, "Columns", raw_pixels.shape[1]))
        }

        return {
            "raw_pixels": calibrated_pixels,
            "metadata": metadata
        }

    def apply_window_level(
        self,
        pixel_array: np.ndarray,
        window_width: float,
        window_level: float
    ) -> np.ndarray:
        """
        Applies linear window-level transformation:
        Maps [Level - Width/2, Level + Width/2] to [0, 255].
        """
        lower = window_level - (window_width / 2.0)
        upper = window_level + (window_width / 2.0)

        # Normalization
        clipped = np.clip(pixel_array, lower, upper)
        norm = (clipped - lower) / (upper - lower + 1e-6)
        uint8_img = (norm * 255.0).astype(np.uint8)

        # Convert to 3-channel RGB for display
        return cv2.cvtColor(uint8_img, cv2.COLOR_GRAY2RGB)

    @staticmethod
    def create_demonstration_dicom() -> bytes:
        """
        Creates an in-memory valid DICOM file for demonstration purposes.
        """
        # Create base canvas with simulated phantom
        h, w = 256, 256
        pixels = np.zeros((h, w), dtype=np.uint16)
        cv2.ellipse(pixels, (128, 128), (80, 95), 0, 0, 360, 100, -1)
        cv2.ellipse(pixels, (128, 128), (75, 90), 0, 0, 360, 40, -1)
        cv2.circle(pixels, (155, 105), 18, 160, -1)

        # Populate minimal DICOM Dataset
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = Dataset()
        ds.file_meta = file_meta
        ds.is_little_endian = True
        ds.is_implicit_VR = False

        ds.SOPClassUID = SecondaryCaptureImageStorage
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.PatientName = "SAMPLE^CLINICAL^DICOM"
        ds.PatientID = "NEURO-PACS-2026"
        ds.StudyDate = "20260924"
        ds.Modality = "MR"
        ds.Manufacturer = "Siemens Healthineers"
        ds.MagneticFieldStrength = 3.0
        ds.SeriesDescription = "AXIAL T1 MULTI-PARAMETRIC"
        ds.RepetitionTime = 2200.0
        ds.EchoTime = 28.0
        ds.SliceThickness = 3.0
        ds.PixelSpacing = [0.4688, 0.4688]
        ds.Rows = h
        ds.Columns = w
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelData = pixels.tobytes()

        bio = io.BytesIO()
        ds.save_as(bio, write_like_original=False)
        return bio.getvalue()
