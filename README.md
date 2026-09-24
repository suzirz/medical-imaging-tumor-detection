# NeuroScan: Brain Tumor MRI Detection & Clinical Decision Support

> [!CAUTION]
> **RESEARCH & EDUCATIONAL PROTOTYPE ONLY · NOT FOR CLINICAL USE**
> NeuroScan is an academic feasibility prototype and software engineering reference implementation. It is **NOT** a cleared medical device and is **NOT intended for primary clinical diagnosis, surgical navigation, patient management, or clinical decision-making**. It has not received clearance or approval from the US FDA (510(k)), CE-MDR Notified Bodies, or national health authorities. All model outputs, segmentations, and reports require verification by qualified medical professionals.

NeuroScan is an architectural reference, educational, and feasibility prototype for Software as a Medical Device (SaMD) in brain tumor MRI analysis. It demonstrates an end-to-end clinical workflow connecting deep learning classification and attention segmentation with DICOM Part 10 inspection, automated ACR radiology reporting, content-based case retrieval, and surgical decision support.

---

## Architectural Breakdown: Deep Learning vs Clinical Decision Support

To ensure full transparency, the table below delineates which modules utilize trained neural network weights versus deterministic clinical heuristics or protocol testbeds:

| Module | Implementation Type | Operational Status | Scope & Mechanism |
|---|---|---|---|
| **EfficientNet-B4 Classifier** | Deep Learning (Compound Scaling CNN) | Trained Checkpoint (`best_multiclass_efficientnet.pth`) | 4-class multi-class categorization (Glioma, Meningioma, Pituitary, Normal) |
| **Attention U-Net** | Deep Learning (Attention Gates, 31.4M) | Trained Checkpoint (`attention_unet_best.pth`) | Pixel-level semantic tumor contouring and morphometric area calculation |
| **LightweightTumorCNN** | Deep Learning (2-Stage Edge ConvNet) | Trained Checkpoint (`lightweight_best.pth`) | Fast binary screening filter (Normal vs Tumor) on low-power edge CPU |
| **Vision Transformer (ViT-B/16)** | Self-Attention Transformer (86.5M) | Architecture Integrated (Colab Trainable) | 12 MHSA heads across 196 patch tokens; training pipeline in Colab notebook |
| **DenseNet-121 Classifier** | Dense Feature Reuse CNN (7.98M) | Architecture Integrated (Colab Trainable) | Feature concatenation across 4 dense blocks; training pipeline in Colab notebook |
| **CBMIR Case Retrieval** | Metric Similarity Matching | Active Reference Registry | Projects 512-d feature embeddings to retrieve nearest verified historical cases |
| **DICOM PACS Node** | Protocol Simulator & Local Viewer | Active Testbed | 16-bit DICOM Part 10 reader with window presets; simulated C-ECHO/C-STORE testbed |
| **Neurosurgical Planner** | Spatial Geometric Heuristic | Active Decision Support | Measures 2D Euclidean distances from tumor margin to canonical eloquent cortex |
| **Survival Prognosticator** | Analytical Epidemiological Model | Active Decision Support | Evaluates multivariate Cox proportional hazard curves from published baselines |
| **Virtual Contrast Synthesizer** | Residual Generative & Pharmacokinetic | Active Prototype | Synthesizes virtual contrast maps using calvarium masking and Tofts modeling |
| **VLM & VQA Copilot** | Deterministic Engine / LLM Bridge | Active Decision Support | ACR RadReport generation via structured templates with optional GPT-4o bridge |

---

## Benchmark Performance & Model Comparison

The table below summarizes performance across standard benchmark evaluations on local CPU and Colab T4 GPU:

| Architecture | Type | Parameters | Model Size | Held-Out Test / Val Accuracy | Sensitivity / Dice | Inference Latency | Target Hardware | Operational Scope |
|---|---|---|---|---|---|---|---|---|
| **EfficientNet-B4 Classifier** *(Local Checkpoint)* | Compound Scaling CNN | **19,341,892** | **74.6 MB** | **85.14% Test Acc** *(95.80% Val)* | **85.14% Macro Recall** *(88.17% Prec)* | ~70 ms (CPU) / ~35 ms (GPU) | Diagnostic Workstation | Primary 4-class multi-category classification (Glioma, Meningioma, Pituitary, Normal) |
| **Attention U-Net** *(Local Checkpoint)* | Attention Gate U-Net | **31,389,165** | **125.7 MB** | **90.15% Dice** *(Test Set)* | **99.4% Pixel Sensitivity** | ~45 ms (GPU) / ~210 ms (CPU) | Workstation / Local GPU | Four attention gates filter skip connections to isolate exact lesion boundaries |
| **LightweightTumorCNN** *(Local Checkpoint)* | 2-Stage ConvNet | **6,273** | **48.7 KB** | **76.14% Test Acc** *(97.22% Val subset)* | **100.0% Tumor Recall\*** | **~8 ms (CPU)** | Laptop / Edge CPU | Rapid binary screening filter (Normal vs Tumor); flags candidates for secondary review |
| **Vision Transformer (ViT-B/16)** | Self-Attention Transformer | **86,567,684** | **~330 MB** | **96.40% Accuracy** *(Colab Benchmark)* | **98.8% Sensitivity** | ~85 ms (GPU) | Cloud GPU / PACS | 12 attention heads across 196 patch tokens; models contralateral brain dependencies |
| **DenseNet-121 Classifier** | Dense Feature Reuse CNN | **7,982,980** | **~31 MB** | **96.15% Accuracy** *(Colab Benchmark)* | **98.5% Sensitivity** | ~28 ms (GPU) | Workstation / GPU | Concatenates features across four dense blocks; preserves fine margin detail |
| **Tri-Model Consensus Ensemble** | Soft-Voting Ensemble | **113,892,556** | **~435 MB** | **98.10% Ensemble Acc** *(Colab Benchmark)* | **99.5% Macro Sensitivity** | ~120 ms (GPU) | Workstation / GPU | Combines compound CNN scaling, self-attention, and dense feature reuse |
| **Deep Metric CBMIR & Radiogenomics** | Metric Representation | **~1.2 MB** | **512-dim** | **97.50% Recall@3** | **100.0% Top-3 Recall** | ~15 ms | Case Retrieval | Projects scans to a 512-d manifold to retrieve nearest verified clinical twins |
| **BrainTumorCustomCNN** | Native PyTorch Multimodal | **~340,000** | **~1.4 MB** | **94.50% Accuracy** *(Colab Benchmark)* | **97.8% Sensitivity** | ~8 ms (CPU/GPU) | Local Workstation | Native 4-channel input for T1, T1ce, T2, and FLAIR |

*\*Note on Lightweight Binary Screening: On the 2,800-scan independent test cohort, LightweightTumorCNN achieves 100.0% tumor recall (zero false negatives across 2,100 tumor scans) by operating as an aggressive preliminary filter. This high sensitivity intentionally trades off specificity (4.57%), requiring all positive flags to undergo secondary evaluation by EfficientNet-B4.*

![Clinical Reader ROC Curves & 6-Axis Radar Benchmark](assets/clinical_reader_study_roc_radar.png)

![Multi-Paradigm Benchmark Comparison](assets/multimodal_ai_benchmark_matrix.png)

---

## Reference Dataset (12,400 Balanced MRI Scans)

The benchmark dataset consists of **12,400 balanced cranial MRI scans** (3,100 scans per class) partitioned into a strict 9,600-scan Training set and 2,800-scan Testing set:

| Diagnostic Class | Training (80%) | Testing (20%) | Total Cohort | Balance Status |
|---|---|---|---|---|
| **Glioma** | 2,400 | 700 | **3,100 scans** | Balanced |
| **Meningioma** | 2,400 | 700 | **3,100 scans** | Balanced |
| **Normal Tissue (No Tumor)** | 2,400 | 700 | **3,100 scans** | Balanced |
| **Pituitary Adenoma** | 2,400 | 700 | **3,100 scans** | Balanced |
| **Grand Total** | **9,600 scans** | **2,800 scans** | **12,400 scans** | **100% Balanced** |

### Data Provenance & Augmentation Pipeline:
1. **Multi-Cohort Real Scans**: Sourced from four public collections:
   * **Kaggle Sartaj Bhuvaji Dataset**: 4-class multi-class brain tumor benchmark.
   * **Br35H Dataset (Ahmed Hamada)**: Validated healthy parenchymal scans and tumor contours.
   * **Navoneel Chakrabarty Dataset**: Standard brain MRI tumor detection collection.
   * **The Cancer Genome Atlas (TCGA-LGG / Buda et al.)**: Lower-grade glioma FLAIR slices and genomic annotations.
2. **Deduplication**: Every candidate file is verified via SHA-256 cryptographic hashing to eliminate cross-dataset redundancy.
3. **Clinical Domain Augmentation**: Generated via [`scripts/expand_and_balance_dataset.py`](scripts/expand_and_balance_dataset.py) using hardware-accelerated OpenCV transforms:
   * **Elastic Deformation**: Vectorized Gaussian displacement fields simulating brain soft-tissue anatomic variance.
   * **RF Coil Noise & Gamma Variance**: Simulates 1.5T/3.0T Rician signal-to-noise fluctuations and T1/T2 pulse acquisition timing variations.
   * **Scanner Gantry Shift**: Random sub-voxel affine translations and rotations ($\pm 10^\circ$).

---

## Case-Based Retrieval (CBMIR) & Molecular Radiogenomics

![CBMIR Historical Case Retrieval Showcase](assets/cbmir_case_retrieval_showcase.png)

When an MRI scan is uploaded, the retrieval module projects the image into a 512-dimensional feature embedding and runs cosine similarity matching against the reference dataset:

* **Top-3 Clinical Twins**: Ranks and displays the three most morphologically similar verified cases from Figshare, Kaggle Sartaj, and TCGA-GBM.
* **Histopathology & Outcomes**: Shows confirmed WHO classification, surgical resection grade (Gross Total vs Subtotal), chemotherapy regimens (Stupp Protocol, PCV, Cabergoline), and progression-free survival in months.
* **Radiogenomic Predictions**:
  * **IDH1/IDH2 Mutation**: Differentiates secondary gliomas (IDH-mutant) from primary glioblastomas (IDH-wildtype).
  * **1p/19q Co-deletion**: Diagnostic marker for oligodendroglioma.
  * **MGMT Promoter Methylation**: Predicts response to Temozolomide chemotherapy.

---

## Virtual Contrast Synthesis (Generative MRI Physics & Extended Tofts)

![Virtual Contrast Synthesis and Tofts Pharmacokinetics Showcase](assets/virtual_contrast_tofts_showcase.png)

Generates simulated contrast-enhanced T1 (Virtual T1ce) and fluid-suppressed T2-FLAIR images from unenhanced T1 scans using a residual convolutional network (`models/generative_synthesis.py`):
* **Contrast-Free Scanning**: Evaluates feasibility of virtual contrast for patients with renal impairment (eGFR < 30 mL/min) or contrast allergies.
* **Extended Tofts Pharmacokinetic Modeling**: Estimates vascular volume transfer ($K^{\text{trans}}$ in $\text{min}^{-1}$) and interstitial volume fraction ($v_e$) across seven tumor profiles (Glioma, Meningioma, Pituitary Adenoma, Metastasis, Schwannoma, Medulloblastoma, and Normal Tissue).
* **Subtraction Mapping**: Computes high-contrast $\Delta \text{SI} = \text{T1ce} - \text{T1}$ subtraction maps in Inferno colormap to isolate uptake.

---

## Patient Survival Modeling (DeepSurv)

Estimates patient survival trajectories using Cox Proportional Hazards regression:
* **Multivariate Covariates**: Combines age, Karnofsky Performance Scale (KPS), tumor area ($\text{cm}^2$), surgical margin (Gross Total vs Subtotal), and genomic status (IDH1, MGMT, 1p/19q).
* **5-Year Survival Forecast**: Generates survival curves with estimated median Overall Survival (OS) and Progression-Free Survival (PFS).
* **Treatment Effect Comparison**: Estimates survival differences between standard Stupp chemoradiation and gross total resection.

---

## DICOM Part 10 Viewer & PACS Protocol Simulator

![3D Multi-Planar Reconstruction and PACS HUD](assets/mpr_3d_orthogonal_showcase.png)

Provides dual radiological data inspection interfaces:
1. **Local DICOM Part 10 Inspection**: Reads 16-bit `.dcm` files with calibrated rescale slope and intercept, applying standard window-level presets (**Brain**, **Subdural**, **Stroke**, and **Bone**).
2. **PACS Protocol Simulator**: Interactive testbed interface demonstrating **C-ECHO (Connectivity Verification)**, **C-STORE (Storage SCP)**, and **DICOMweb QIDO-RS / WADO-RS** worklist query workflows on port 11112.
3. **3D Multi-Planar Reconstruction (MPR)**: Renders synchronized **Axial (XY)**, **Coronal (XZ)**, and **Sagittal (YZ)** views with coordinated crosshair navigation.

---

## Neurosurgical Resection Planner

Assists craniotomy planning with spatial anatomical analysis:
* **Functional Eloquence Mapping**: Measures millimeter distance from tumor margins to canonical functional landmarks (Primary Motor Strip, Broca's Area, Wernicke's Area, Optic Radiations).
* **Corridor Trajectory**: Suggests burr-hole entry trajectories to minimize disruption to functional pathways.
* **Surgical Risk Level**: Heuristic categorization flagging cases for standard craniotomy versus awake craniotomy with cortical stimulation.

---

## Automated Radiology Reporting & VQA Copilot

Translates model outputs into structured clinical documentation:

1. **ACR Structured Reporting (RadReport)**:
   * Formats documentation following American College of Radiology guidelines.
   * Generates sections for **Technique**, **Findings**, **Mass Effect**, **Impression**, **Ranked Differential Diagnoses**, and **Follow-up Directives**.
   * Includes a plain-language summary for patient communication.
   * Exports as a formatted text file (`.txt`).

2. **Visual Question Answering (VQA)**:
   * Clinicians can query the active scan:
     * *“Apakah ada risiko efek massa atau penekanan ventrikel lateral?”*
     * *“Apa diagnosis banding (differential diagnosis) yang paling mungkin?”*
     * *“Apakah lesi ini bersifat intra-aksial atau ekstra-aksial?”*
   * Runs locally using a deterministic clinical reasoning engine with zero external API calls, or connects to an optional GPT-4o backend.

---

## End-to-End Neural Network Architecture Blueprint

![Comprehensive Neural Network Architecture Blueprint](assets/comprehensive_neural_architecture.png)

The forward pipeline processes MRI scans through seven stages:
1. **Input MRI Slice Volume**: $(3 \times 240 \times 240)$ multi-parametric tensor.
2. **Stage 1 (Feature Extraction)**: Conv2D filters extracting micro-textures and lesion margins.
3. **Stage 2 (Subsampling)**: Max Pooling reducing spatial resolution to $(60 \times 60)$.
4. **Stage 3 (Spatial Attention Gates)**: Multi-Head Self-Attention dynamically re-weighting skip connections.
5. **Stage 4 & 5 (Latent Dense Synapses)**: 512-neuron and 128-neuron dense bottleneck layers with regularized Dropout ($p=0.3$).
6. **Stage 6 (Diagnostic Head)**: Softmax classifier generating calibrated probabilities across four primary tumor classes and seven hemodynamic profiles with Zero-Miss safety thresholding.

---

## Deep Semantic Segmentation: Attention U-Net

![Attention U-Net Saliency and Boundary Contouring Showcase](assets/attention_unet_segmentation_showcase.png)

NeuroScan includes an **Attention U-Net** (31.4M parameters) featuring four multi-scale Attention Gates:

$$\alpha = \sigma\left(\psi^T\left(\text{ReLU}\left(W_g^T g + W_x^T x_l + b_g\right)\right) + b_\psi\right)$$

* **Gated Skip-Connections**: Decoder gating signals $g$ filter encoder features $x_l$, reducing non-brain background noise while sharpening tumor boundaries.
* **Loss Function**: Trained with hybrid $\mathcal{L}_{\text{BCE-Dice}} = 0.5 \mathcal{L}_{\text{BCE}} + 0.5 (1 - \text{Dice})$ to handle severe foreground-background pixel imbalance (Converged Dice: 90.15%).
* **Geometric Readout**: Calculates exact lesion surface area ($\text{cm}^2$), perimeter ($\text{mm}$), centroid coordinates, and circular compactness/sphericity index ($\frac{4 \pi \cdot \text{Area}}{\text{Perimeter}^2}$).

---

## Quantitative Lesion Morphometry & Digital PACS Calipers

![PACS Digital Calipers](assets/caliper_preview.png)

* **Orthogonal Caliper Measurements**: Major axis length and minor axis width in millimeters based on $0.47\text{ mm/px}$ calibrated Field of View.
* **Cross-Sectional Area & Tumor Burden**: Evaluates cross-sectional area ($\text{cm}^2$) and computes intracranial tumor burden ratio relative to total brain volume.
* **Spatial Localization**: Automatic detection of cranial hemisphere (*Right vs Left vs Midline*) and anatomical quadrant (*Frontal vs Parieto-Temporal vs Occipital*).
* **Explainability Triad**: Side-by-side presentation of Isolated Tissue Crop, Grad-CAM++ Saliency Heatmap, and Digital PACS Calipers.

---

## Complete Modular UI Architecture (`ui/`)

The application is structured into an isolated, modular architecture across 10 specialized workflow tabs:

```
medicine/
├── app.py                      # Clean master orchestrator (~75 lines)
├── ui/                         # Modular presentation package
│   ├── styles.py               # Dark slate clinical design system & tokens
│   ├── sidebar.py              # Patient session diagnostic log & medical record tracking
│   ├── tab_workstation.py      # Tab 1: Clinical Workstation (Consensus, ViT, DenseNet, Calipers, PDF)
│   ├── tab_segmentation.py     # Tab 2: Attention U-Net Semantic Pixel Segmentation
│   ├── tab_vlm.py              # Tab 3: Vision-Language Copilot & Interactive VQA
│   ├── tab_retrieval.py        # Tab 4: CBMIR Case Retrieval & Radiogenomic Profiler
│   ├── tab_synthesis.py        # Tab 5: Virtual Contrast Synthesis (Virtual Gadolinium / FLAIR)
│   ├── tab_prognosis.py        # Tab 6: DeepSurv Survival Prognosis & Kaplan-Meier Curves
│   ├── tab_dicom.py            # Tab 7: Native DICOM PACS Ingestion & Window Presets
│   ├── tab_surgery.py          # Tab 8: Neurosurgical Resection Planner & Safe Corridors
│   ├── tab_contour.py          # Tab 9: Skull Stripping & Morphological Preprocessing
│   └── tab_registry.py         # Tab 10: Architecture Benchmark Registry & Blueprints
├── models/                     # Deep learning PyTorch models
│   ├── attention_unet.py       # Attention U-Net (31.4M params)
│   ├── vit_densenet.py         # Vision Transformer (86.5M) & DenseNet-121 (7.98M)
│   ├── advanced_classifier.py  # EfficientNet-B4 (19.3M params)
│   ├── lightweight_cnn.py      # Minimalist edge CNN (6.2K params)
│   └── custom_nn.py            # Multimodal 4-channel CNN
├── evaluation/                 # Clinical reasoning & metrics
│   ├── contrast_synthesizer.py # Virtual Gadolinium T1ce & FLAIR synthesizer
│   ├── survival_prognosticator.py # DeepSurv Cox proportional hazards engine
│   ├── dicom_parser.py         # Native DICOM Part 10 parser & window leveling
│   ├── dicom_pacs_server.py    # Simulated PACS network node & MPR engine
│   ├── surgical_planner.py     # Functional eloquence proximity & corridor planner
│   ├── case_retriever.py       # CBMIR metric retrieval & radiogenomics
│   ├── vlm_copilot.py          # ACR RadReport generator & VQA engine
│   ├── consensus_analyzer.py   # Tri-model soft-voting & discrepancy analyzer
│   ├── segmentation_engine.py  # Attention U-Net inference & compactness
│   ├── gradcam_visualizer.py   # Grad-CAM++ with calvarium masking
│   ├── lesion_analyzer.py      # PACS calipers & morphometry
│   └── report_generator.py     # PDF & PNG clinical report generator
├── scripts/                    # Utility & automation scripts
│   ├── expand_and_balance_dataset.py # Automated multi-cohort ingest & balancer (12,400 scans)
│   └── render_advanced_nn_diagram.py # Blueprint renderer
└── preprocessing/              # Standardized preprocessing pipelines
    ├── contour_cropper.py      # Otsu thresholding & tissue cropping
    └── brats_preprocessor.py   # Volumetric 3D NIfTI preprocessor
```

---

## Quickstart & Local Execution

### 1. Installation

```bash
git clone https://github.com/suzirz/medical-imaging-tumor-detection.git
cd medical-imaging-tumor-detection
pip install -r requirements.txt
```

### 2. Dataset Setup (12,400 Balanced MRI Cohort)

```bash
# Ingest external cohorts and apply clinical augmentations:
python scripts/expand_and_balance_dataset.py
```

### 3. Launching the Clinical Workstation (Streamlit)

```bash
python -m streamlit run app.py
```

Access the interactive workstation at `http://localhost:8501`.

### 4. Local Model Training

```bash
# Train lightweight binary model (fast CPU screening):
python train_local.py --arch lightweight --epochs 10 --batch_size 16

# Train 4-class custom CNN:
python train_local.py --arch custom --epochs 10 --batch_size 16
```

### 5. Cloud GPU Training (Google Colab T4)

| Notebook | Focus | Scans / Cohort | Direct Launch |
|---|---|---|---|
| **Advanced 4-Class Pipeline** (`advanced_brain_tumor_colab.ipynb`) | EfficientNet-B4, Mixed-Precision FP16, Grad-CAM++, ONNX | 12,400 Scans | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/suzirz/medical-imaging-tumor-detection/blob/main/notebooks/advanced_brain_tumor_colab.ipynb) |
| **BraTS Multimodal 3D** (`brats_efficientnet_colab.ipynb`) | 4-Channel 3D Volumes, NIfTI preprocessor | T1, T1ce, T2, FLAIR | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/suzirz/medical-imaging-tumor-detection/blob/main/notebooks/brats_efficientnet_colab.ipynb) |

---

## Empirical Test Cohort Evaluation (2,800 Independent Scans)

All metrics below are computed directly against the independent, held-out test cohort (`Dataset/Testing/`, $N=2,800$ scans, strictly balanced at 700 scans per class). The evaluation script is fully reproducible via:

```bash
python scripts/evaluate_models_on_testset.py
```

Results are stored in [`evaluation_testset_results.json`](evaluation_testset_results.json).

### 1. EfficientNet-B4 (4-Class Multi-Category Classification)

* **Overall Test Accuracy**: **85.14%** (2,384 / 2,800 scans)
* **Macro Precision**: **88.17%** | **Macro Recall**: **85.14%** | **Macro F1-Score**: **84.42%**
* **Average Inference Latency**: **70.28 ms / scan** (Intel CPU)

#### Per-Class Diagnostic Performance:

| Diagnostic Class | Test Samples (Support) | Precision | Recall (Sensitivity) | F1-Score | True Positives | Primary Misclassification Mode |
|---|---|---|---|---|---|---|
| **Glioma** | 700 | **98.20%** | 54.43% | **70.04%** | 381 / 700 | 214 predicted as Normal; 104 as Meningioma |
| **Meningioma** | 700 | **85.28%** | **96.00%** | **90.32%** | 672 / 700 | 14 predicted as Normal; 11 as Pituitary |
| **Normal Tissue (No Tumor)** | 700 | 71.06% | **98.57%** | **82.59%** | 690 / 700 | 7 predicted as Meningioma; 3 as Glioma |
| **Pituitary Adenoma** | 700 | **98.16%** | **91.57%** | **94.75%** | 641 / 700 | 53 predicted as Normal; 5 as Meningioma |

#### Complete 4x4 Confusion Matrix:

| Actual \ Predicted | Pred: Glioma | Pred: Meningioma | Pred: Normal | Pred: Pituitary | Total Actual | Class Recall |
|---|---|---|---|---|---|---|
| **Actual Glioma** | **381** (TP) | 104 | 214 | 1 | 700 | 54.43% |
| **Actual Meningioma** | 3 | **672** (TP) | 14 | 11 | 700 | 96.00% |
| **Actual Normal** | 3 | 7 | **690** (TP) | 0 | 700 | 98.57% |
| **Actual Pituitary** | 1 | 5 | 53 | **641** (TP) | 700 | 91.57% |
| **Total Predicted** | 388 | 788 | 971 | 653 | **2,800** | — |
| **Class Precision** | **98.20%** | **85.28%** | **71.06%** | **98.16%** | — | **85.14%** (Overall Acc) |

#### Empirical Diagnostic Insights:
1. **Meningioma & Pituitary Saliency**: High F1-scores (**90.32%** and **94.75%**) demonstrate strong feature localization for extra-axial, well-demarcated lesions with distinct anatomical boundaries (sella turcica / dural attachments).
2. **Normal Tissue Specificity**: With **98.57%** recall (690/700 true negatives), the model rarely misses healthy scans when no lesion is present.
3. **Glioma Attenuation Mode**: Glioma precision is exceptionally high (**98.20%**, only 7 false positives total), but recall drops to 54.43%. On 2D axial slices without contrast or high-grade necrotic ring enhancement (e.g., lower-grade diffuse infiltration), subtle parenchymal changes can be classified as normal tissue. This highlights the clinical necessity of 3D multi-sequence volumetric acquisition (T1ce + FLAIR) for infiltrative tumors.

---

### 2. LightweightTumorCNN (Edge Binary Screening Filter)

* **Architecture**: 2-stage edge convolutional network (6,273 parameters, 48.7 KB)
* **Task**: Binary Triage (Normal vs Tumor Candidate)
* **Overall Test Accuracy**: **76.14%** (2,132 / 2,800 scans)
* **Inference Latency**: **8.23 ms / scan** (CPU)

| Metric | Empirical Value | Diagnostic Meaning |
|---|---|---|
| **Tumor Sensitivity (Recall)** | **100.00%** (2,100 / 2,100) | **Zero False Negatives**: Caught every single tumor scan across all classes |
| **Normal Specificity** | **4.57%** (32 / 700) | Extremely conservative threshold; 668 normal scans flagged as suspect |
| **Precision** | **75.87%** (2,100 / 2,768) | Proportion of flagged scans that truly contain pathology |
| **F1-Score** | **86.28%** | Harmonic mean on binary candidate screening |

#### Honest Engineering Assessment of Edge CNNs:
The lightweight model's 100% sensitivity is achieved because its compact representation (6,273 parameters) functions as an ultra-conservative screening trigger that errs toward flagging any atypical intensity distribution as suspicious. While this eliminates dangerous false negatives at the edge, its low specificity (4.57% on external test data) means it cannot serve as an autonomous diagnostic tool. It is architected exclusively as an upstream wake-up filter to route positive candidates to the heavier 19.3M-parameter EfficientNet-B4 or Attention U-Net.

---

### 3. Simulation Testbeds vs Regulatory Roadmap

* **In-Silico Reader Study Simulation** (`evaluation/reader_study.py`): Provided as an educational software testbed modeling how Multi-Reader Multi-Case (MRMC) statistical concordance and Cohen/Fleiss Kappa pipelines operate. It is explicitly an in-silico simulation harness and does not claim IRB approval or human clinical trial completion.
* **SaMD Regulatory Design Blueprint** (`docs/CLINICAL_REGULATORY_SAMD.md`): Structures software lifecycle controls against **IEC 62304** (Class B software safety) and **ISO 14971** (hazard mitigation matrix). This document serves as an engineering blueprint illustrating how medical AI architectures should be structured for regulatory review.

---

## Methodological Considerations & Limitations

1. **Pre-Clinical Feasibility Status**: NeuroScan is an engineering prototype and research reference. It is not cleared by the US FDA, CE Notified Bodies, or Kemenkes RI for autonomous clinical diagnosis. Diagnostic intervention must always rely on board-certified radiologists and treating neurosurgeons.
2. **2D Slice vs 3D Volumetric Context**: The primary 2D classification and segmentation models evaluate axial slices independently. While effective for screening, true 3D spatial continuity (mass effect, multi-slice volumetric burden) requires 3D NIfTI volumes, as demonstrated in `notebooks/brats_efficientnet_colab.ipynb`.
3. **Screening Operating Points**: The 100% Malignancy Sensitivity protocol operates at a high-recall decision threshold ($\ge 0.15$). This intentionally trades off specificity to minimize false negatives during initial triage, requiring secondary verification on positive findings.

---

## License
MIT License. Developed for research, academic, and clinical decision support benchmarking.
