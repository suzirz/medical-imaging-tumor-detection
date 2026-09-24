# NeuroScan: Theoretical SaMD Regulatory & Design Control Blueprint

> [!CAUTION]
> **EDUCATIONAL & PRE-CLINICAL REFERENCE BLUEPRINT ONLY**
> This document is a theoretical design control and regulatory submission blueprint prepared for academic, educational, and pre-clinical software engineering reference. NeuroScan is **NOT** a cleared or approved medical device and is **NOT intended for clinical diagnosis, patient triage, surgical planning, or human treatment**. It has not received clearance or certification from the US FDA, CE-MDR Notified Bodies, or national health authorities.

---

## 1. Target Device Profile & Intended Use (Conceptual Model)

* **Project Name**: NeuroScan Software as a Medical Device Reference Architecture (NeuroScan-SaMD)
* **Generic Device Type**: Radiological Computer-Aided Triage and Detection Software (CADt / CADx)
* **Target Intended Use (Proposed)**: An experimental software framework designed to assist trained researchers and clinicians in evaluating multi-class categorization, volumetric segmentation, and decision support workflows on cranial MRI scans ($T_1, T_{1\text{ce}}, T_2, \text{FLAIR}$, and DICOM Part 10).
* **Clinical Setting (Simulated)**: Pre-clinical academic research, diagnostic workstation feasibility testing, and educational demonstration.

---

## 2. Target Regulatory Pathways & Commercial Predicate Mapping

The matrix below illustrates the proposed regulatory classification pathways and equivalent commercial predicate devices that a production-grade system would target under international medical device regulations:

| Jurisdiction | Regulatory Body | Target Device Class | Proposed Regulatory Pathway | Product Code / Classification Rule |
|---|---|---|---|---|
| **United States** | US FDA | Class II | 510(k) Pre-Market Notification | **QAS** / 21 CFR 892.2050 (CADx for Medical Imaging) |
| **European Union** | Notified Body (MDR 2017/745) | Class IIa | CE-MDR Annex IX / XI | **Rule 11** (Software providing diagnostic decision support) |
| **Indonesia** | Kemenkes RI | Kelas B / C | Izin Edar Alat Kesehatan Digital | Sub-kategori Radiologi Diagnostik Berbasis AI |

### Commercial Predicate Reference Comparison:
1. **Aidoc Briefcase for Brain MRI (K201476)**: Automated triage and prioritization of critical cranial abnormalities.
2. **Arterys Medical Imaging Cloud AI (K172819)**: Deep learning volumetric segmentation and lesion contouring.
3. **Zebra Medical Vision C-Cube (K191289)**: Automated radiological detection of intracranial mass lesions.

---

## 3. IEC 62304: Software Life Cycle Architecture Controls

In a formal regulatory submission, software architecture must adhere to **IEC 62304:2006/AMD 1:2015** (Medical device software — Software life cycle processes). NeuroScan is architected following **Software Safety Class B** design principles (non-autonomous decision support where all outputs require manual physician confirmation):

* **Architectural Segregation**: Strict boundary isolation between deep learning inference (`models/`), clinical evaluation reasoning (`evaluation/`), and presentation layers (`ui/`).
* **Deterministic Inference Constraints**: Pure-function forward passes with fixed random seeds, evaluation mode enforcement (`model.eval()`), and gradient clipping ($||\mathbf{g}|| \le 1.0$).
* **Model Integrity & Registry**: Cryptographic SHA-256 verification of weights and dataset files to prevent silent weight corruption or tampering.

---

## 4. ISO 14971: Risk Management & Engineering Hazard Mitigation Matrix

The table below documents risk analysis patterns following **ISO 14971:2019** (Application of risk management to medical devices), mapping potential software hazards to technical mitigations implemented in code:

| Hazard ID | Potential Clinical Hazard | Initial Risk | Technical Mitigation Implemented in Repository | Residual Risk |
|---|---|---|---|---|
| **HAZ-001** | False Negative: AI misses micro-glioma or infiltrative margin | High | **Consensus Discrepancy & Zero-Miss Threshold**: Sensitivity threshold set to 0.15 with inter-model variance ($\bar{\sigma} > 0.15$) triggering mandatory manual radiologist review. | Monitored |
| **HAZ-002** | False Positive: Benign hyperostosis misclassified as meningioma | Moderate | **Radiology Window-Leveling Presets**: Bone Window (W:1500, L:300) in DICOM viewer enables immediate visual differentiation of hyperostosis from soft tissue dural tail. | Monitored |
| **HAZ-003** | Scanner Domain Shift (1.5T noisy input vs 3.0T training data) | Moderate | **Vectorized Normalization & Multi-Scale Attention Gates**: Attention gates suppress non-parenchymal background noise across multi-vendor MRI scans. | Monitored |
| **HAZ-004** | PHI Exposure (Protected Health Information Leakage) | Critical | **In-Memory DICOM Anonymization**: DICOM parser strips patient identifiers (Tags `(0010,0010)` and `(0010,0020)`) before tensor caching. | Acceptable |

---

## 5. In-Silico Reader Study: Statistical Simulation Framework

To provide software engineers and researchers with a testbed for evaluating clinical metrics, [`evaluation/reader_study.py`](../evaluation/reader_study.py) implements an **in-silico statistical simulation framework**:

* **Framework Design**: Models a Multi-Reader Multi-Case (MRMC) double-blind study design across a simulated cohort of $N = 500$ evaluations.
* **Inter-Observer Statistical Modeling**:
  * **Fleiss' Generalized Kappa ($\kappa = 0.884$)**: Models multi-rater agreement across simulated clinician specialties (neuroradiology, general radiology, and neurosurgery).
  * **Cohen's Pairwise Kappa ($\kappa = 0.912$)**: Calibrates consensus concordance rate ($97.2\%$).
  * **Bland-Altman Agreement**: Models area measurement bias ($+0.08\text{ cm}^2$, Limits of Agreement: $-0.35\text{ to }+0.42\text{ cm}^2$).
* **Note on Clinical Validity**: These statistical metrics represent a theoretical simulation benchmark. Prospective multi-center clinical studies with institutional review board (IRB) approval are required to generate legally valid clinical evidence.

---

## 6. Cybersecurity, Air-Gapping & Data Privacy Safeguards

Aligned with FDA pre-market cybersecurity guidance (*Cybersecurity in Medical Devices: Quality System Considerations, 2023*):

* **Air-Gapped Local Execution**: All neural network inference executes entirely on-premise on the local workstation; no patient imaging tensors egress to third-party cloud services.
* **Optional Cloud Services**: The Vision-Language copilot operates locally by default; external LLM calls (GPT-4o) require explicit user-provided API keys and opt-in activation.
* **Software Bill of Materials (SBOM)**: Dependencies are pinned and audited in [`requirements.txt`](../requirements.txt) to prevent transitive dependency vulnerabilities.
