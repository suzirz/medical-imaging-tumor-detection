import streamlit as st
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2

from models.custom_nn import BrainTumorCustomCNN
from models.efficientnet_tumor_classifier import BrainTumorClassifier
from evaluation.gradcam_visualizer import GradCAMVisualizer

st.set_page_config(
    page_title="Medical Tumor Detection & Neural Network Inspector",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Medical Imaging: Neural Network Inspector & Pipeline")
st.markdown("""
Aplikasi web untuk **menginspeksi arsitektur Neural Network (CNN)**, melihat diagram lapisan secara visual, dan menguji inferensi deteksi tumor pada citra medis.
""")

tabs = st.tabs(["🔬 1. Visualisasi & Inspeksi Neural Network", "🩺 2. Uji Deteksi Scan Medis"])

# ================= TAB 1: INSPEKSI NEURAL NETWORK =================
with tabs[0]:
    st.subheader("Arsitektur Lapisan Neural Network (CNN)")
    
    col_nn1, col_nn2 = st.columns([1, 1])
    with col_nn1:
        selected_model = st.selectbox(
            "Pilih Neural Network yang Ingin Diinspeksi:",
            ["BrainTumorCustomCNN (4-Layer Pure CNN)", "EfficientNet-B4 + SE Attention"]
        )
    
    if "CustomCNN" in selected_model:
        model = BrainTumorCustomCNN(in_channels=4, num_classes=4)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        st.info(f"**Ringkasan Parameter:** Total Parameter: `{total_params:,}` | Trainable: `{trainable_params:,}`")
        
        st.markdown("""
        ### Diagram Alur Komputasi Layer:
        ```text
        [Input Scan 4-Modalitas] (Batch, 4, 380, 380)
                   │
                   ▼
        [Block 1] Conv2d(4 → 32, k=3, p=1) + BatchNorm2d + ReLU + MaxPool(2x2)
                   │  └─ Output: (32, 190, 190)
                   ▼
        [Block 2] Conv2d(32 → 64, k=3, p=1) + BatchNorm2d + ReLU + MaxPool(2x2)
                   │  └─ Output: (64, 95, 95)
                   ▼
        [Block 3] Conv2d(64 → 128, k=3, p=1) + BatchNorm2d + ReLU + MaxPool(2x2)
                   │  └─ Output: (128, 47, 47)
                   ▼
        [Block 4] Conv2d(128 → 256, k=3, p=1) + BatchNorm2d + ReLU + AdaptiveAvgPool(6x6)
                   │  └─ Output: (256, 6, 6)
                   ▼
        [Flatten]  9,216 Features
                   │
                   ▼
        [Dense FC1] Linear(9216 → 256) + ReLU + Dropout(p=0.3)
                   │
                   ▼
        [Dense FC2] Linear(256 → 4) ──► [Softmax: Normal, Glioma, Meningioma, Pituitary]
        ```
        """)

        with st.expander("🔍 Lihat Definisi Raw PyTorch Module"):
            st.code(str(model), language="text")

    else:
        model = BrainTumorClassifier(num_classes=4, pretrained=False)
        total_params = sum(p.numel() for p in model.parameters())
        st.info(f"**Ringkasan Parameter:** Total Parameter: `{total_params:,}`")
        
        st.markdown("""
        ### Diagram Arsitektur EfficientNet-B4 + SE Attention:
        ```text
        [Input Tensor 4-Ch] (T1, T1ce, T2, FLAIR)
                   │
                   ▼
        [Stem Conv] 4 → 48 Channels (Inverted Residual Blocks: MBConv1 & MBConv6)
                   │
                   ▼
        [SE Attention Block] Squeeze (GlobalPool) → Excitation (FC SiLU → FC Sigmoid)
                   │         └─ Rekalibrasi bobot channel fitur tumor
                   ▼
        [Head Classifier] AdaptiveAvgPool2d(1) -> Flatten(1792)
                   ├── BatchNorm1d(1792) -> Linear(1792 → 512) -> SiLU -> Dropout(0.4)
                   ├── Linear(512 → 256) -> SiLU -> Dropout(0.2)
                   └── Linear(256 → 4 Classes Output)
        ```
        """)
        with st.expander("🔍 Lihat Detail Submodul Backbone & Head"):
            st.code(str(model), language="text")

# ================= TAB 2: UJI DETEKSI SCAN MEDIS =================
with tabs[1]:
    st.subheader("Uji Coba Deteksi Citra Medis")
    
    col_a, col_b = st.columns([1, 1])
    with col_a:
        uploaded_file = st.file_uploader("Upload Scan MRI (.png, .jpg, .jpeg):", type=["png", "jpg", "jpeg"])
        if uploaded_file is None:
            # Synthetic MRI visual
            arr = np.zeros((380, 380), dtype=np.uint8)
            cv2.circle(arr, (190, 190), 120, 160, -1)
            cv2.circle(arr, (230, 160), 30, 240, -1) # simulated tumor area
            demo_img = Image.fromarray(arr)
            st.caption("Menampilkan gambar simulasi MRI bawaan:")
            st.image(demo_img, caption="Simulasi MRI T1ce", use_container_width=True)
            active_img = demo_img
        else:
            active_img = Image.open(uploaded_file).convert("RGB")
            st.image(active_img, caption="Gambar yang diupload", use_container_width=True)

    with col_b:
        st.write("### Evaluasi Model & Heatmap Grad-CAM++")
        if st.button("🚀 Jalankan Inferensi Neural Network"):
            # Siapkan tensor 4 channel
            np_img = np.array(active_img.convert("RGB"))
            resized = cv2.resize(np_img, (380, 380))
            ch4 = np.zeros((1, 4, 380, 380), dtype=np.float32)
            ch4[0, 0] = resized[:, :, 0] / 255.0
            ch4[0, 1] = resized[:, :, 1] / 255.0
            ch4[0, 2] = resized[:, :, 2] / 255.0
            ch4[0, 3] = (ch4[0, 0] + ch4[0, 1]) / 2.0
            t_input = torch.from_numpy(ch4)

            eval_model = BrainTumorCustomCNN(4, 4)
            eval_model.eval()
            with torch.no_grad():
                logits = eval_model(t_input)
                probs = F.softmax(logits, dim=1).numpy()[0]

            labels = ["Normal", "Glioma", "Meningioma", "Tumor Hipofisis"]
            pred_idx = int(np.argmax(probs))
            
            st.success(f"**Prediksi Teratas:** {labels[pred_idx]} (Keyakinan: {probs[pred_idx]*100:.1f}%)")
            
            # Progress bar untuk tiap kelas
            for i, name in enumerate(labels):
                st.write(f"**{name}**: {probs[i]*100:.1f}%")
                st.progress(float(probs[i]))

            # Heatmap Visualizer
            vis = GradCAMVisualizer(eval_model)
            heatmap = vis.generate_heatmap(t_input)
            overlay = vis.overlay_on_mri(resized[:, :, 0], heatmap)

            st.write("#### Peta Aktivasi Spasial (Grad-CAM++ Overlay):")
            st.image(overlay, caption="Area bergradasi merah menunjukkan fokus layer konvolusi tertinggi.", use_container_width=True)
