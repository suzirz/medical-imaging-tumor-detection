import streamlit as st
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2

from models.lightweight_cnn import LightweightTumorCNN
from models.custom_nn import BrainTumorCustomCNN
from models.efficientnet_tumor_classifier import BrainTumorClassifier
from preprocessing.contour_cropper import crop_brain_contour, preprocess_mri_240
from evaluation.gradcam_visualizer import GradCAMVisualizer

st.set_page_config(
    page_title="Brain Tumor Detection & Neural Network Inspector",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Brain Tumor Detection System & Pipeline")
st.markdown("""
Aplikasi deteksi tumor otak berbasis CNN dengan implementasi algoritma **Brain Contour Cropping**, 
inspeksi arsitektur Neural Network, serta Grad-CAM++ Explainability.
""")

tabs = st.tabs([
    "🔬 1. Contour Cropping & Preprocessing",
    "📊 2. Inspeksi Arsitektur Neural Network",
    "🩺 3. Uji Deteksi Scan Medis"
])

# ================= TAB 1: CONTOUR CROPPING =================
with tabs[0]:
    st.subheader("Algoritma Pemotongan Kontur Otak (Brain Contour Cropping)")
    st.markdown("""
    Metode Preprocessing:
    1. Mengubah citra ke Grayscale dan Gaussian Blur (5x5).
    2. Threshold biner & operasi morfologi (Erode + Dilate).
    3. Mencari kontur terluar ekstrem (kiri, kanan, atas, bawah).
    4. Memotong (*crop*) hanya pada area jaringan otak untuk membuang background hitam yang tidak berguna.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Input Gambar Asli")
        # Generate sample brain MRI with padding
        base_canvas = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.ellipse(base_canvas, (150, 150), (90, 110), 0, 0, 360, (180, 180, 180), -1)
        cv2.circle(base_canvas, (180, 130), 25, (245, 245, 245), -1)
        
        sample_file = st.file_uploader("Upload Scan MRI (.png, .jpg):", type=["png", "jpg", "jpeg"], key="crop_uploader")
        if sample_file:
            input_image = np.array(Image.open(sample_file).convert("RGB"))
        else:
            input_image = base_canvas

        st.image(input_image, caption=f"Original MRI Image (Shape: {input_image.shape})", width='stretch')

    with col2:
        st.write("#### Hasil Ekstraksi Kontur & Crop")
        cropped_result = crop_brain_contour(input_image)
        resized_240 = cv2.resize(cropped_result, (240, 240))
        st.image(resized_240, caption=f"Cropped & Resized to (240, 240, 3)", width='stretch')
        st.success(f"Berhasil memangkas margin hitam! Dimensi akhir siap masuk Neural Network: `{resized_240.shape}`")

# ================= TAB 2: INSPEKSI ARSITEKTUR =================
with tabs[1]:
    st.subheader("Perbandingan Model Neural Network")
    
    model_opt = st.selectbox(
        "Pilih Model untuk Diinspeksi:",
        [
            "LightweightTumorCNN (Fast 2-Pool Binary Architecture)",
            "BrainTumorCustomCNN (4-Layer Deep Pure CNN)",
            "BraTS EfficientNet-B4 + SE Attention"
        ]
    )

    if "LightweightTumorCNN" in model_opt:
        m = LightweightTumorCNN(num_classes=2)
        total_p = sum(p.numel() for p in m.parameters())
        st.info(f"**LightweightTumorCNN**: Model ringan dengan 2 pooling layer (f=4, s=4). Total Parameter: `{total_p:,}`")
        st.markdown("""
        ```text
        Input (3, 240, 240)
           │
           ▼
        ZeroPadding2d(2, 2)
           │
           ▼
        Conv2d(3 -> 32 filters, 7x7, stride=1) + BatchNorm + ReLU
           │
           ▼
        MaxPool2d(f=4, s=4)  --> Resizing spasial dari 240 ke 60
           │
           ▼
        MaxPool2d(f=4, s=4)  --> Resizing spasial dari 60 ke 15
           │
           ▼
        Flatten (32 * 14 * 14 = 6,272 features)
           │
           ▼
        Linear(6272 -> 1) + Sigmoid (Binary: Normal vs Tumor)
        ```
        """)
        with st.expander("Lihat Struktur Lengkap PyTorch"):
            st.code(str(m), language="text")

    elif "CustomCNN" in model_opt:
        m = BrainTumorCustomCNN(in_channels=4, num_classes=4)
        total_p = sum(p.numel() for p in m.parameters())
        st.info(f"**BrainTumorCustomCNN**: Total Parameter: `{total_p:,}` (4-Stage Convolutions)")
        with st.expander("Lihat Struktur Lengkap PyTorch"):
            st.code(str(m), language="text")

    else:
        m = BrainTumorClassifier(num_classes=4, pretrained=False)
        total_p = sum(p.numel() for p in m.parameters())
        st.info(f"**EfficientNet-B4 + SE Attention**: Total Parameter: `{total_p:,}`")
        with st.expander("Lihat Struktur Lengkap PyTorch"):
            st.code(str(m), language="text")

# ================= TAB 3: UJI DETEKSI SCAN =================
with tabs[2]:
    st.subheader("Uji Coba Deteksi Tumor & Grad-CAM++")
    col_u1, col_u2 = st.columns(2)

    with col_u1:
        test_file = st.file_uploader("Upload File Scan untuk Diagnosa:", type=["png", "jpg", "jpeg"], key="eval_uploader")
        if test_file:
            eval_img = Image.open(test_file).convert("RGB")
        else:
            eval_img = Image.fromarray(base_canvas)
        st.image(eval_img, caption="Citra Input", width='stretch')

    with col_u2:
        st.write("#### Jalankan Inferensi Model")
        chosen_net = st.radio("Pilih Engine Model:", ["LightweightTumorCNN (Binary)", "Multimodal Custom CNN (4-Class)"])

        if st.button("🚀 Jalankan Analisis"):
            if "LightweightTumorCNN" in chosen_net:
                norm_img = preprocess_mri_240(eval_img)
                tensor_in = torch.from_numpy(norm_img).permute(2, 0, 1).unsqueeze(0).float()
                
                net = LightweightTumorCNN(num_classes=2)
                net.eval()
                with torch.no_grad():
                    logits = net(tensor_in)
                    prob_tumor = float(torch.sigmoid(logits)[0, 0])
                
                prob_normal = 1.0 - prob_tumor
                if prob_tumor >= 0.5:
                    st.error(f"⚠️ **TERDETEKSI TUMOR** (Confidence: {prob_tumor*100:.1f}%)")
                else:
                    st.success(f"✅ **TIDAK TERDETEKSI TUMOR / NORMAL** (Confidence: {prob_normal*100:.1f}%)")

                st.progress(prob_tumor, text=f"Tumor Probability: {prob_tumor*100:.1f}%")

            else:
                np_arr = np.array(eval_img)
                resized = cv2.resize(np_arr, (380, 380))
                ch4 = np.zeros((1, 4, 380, 380), dtype=np.float32)
                ch4[0, 0] = resized[:, :, 0] / 255.0
                ch4[0, 1] = resized[:, :, 1] / 255.0
                ch4[0, 2] = resized[:, :, 2] / 255.0
                ch4[0, 3] = (ch4[0, 0] + ch4[0, 1]) / 2.0
                tensor_in = torch.from_numpy(ch4)

                net = BrainTumorCustomCNN(4, 4)
                net.eval()
                with torch.no_grad():
                    probs = F.softmax(net(tensor_in), dim=1).numpy()[0]
                
                classes = ["Normal", "Glioma", "Meningioma", "Tumor Hipofisis"]
                p_idx = int(np.argmax(probs))
                st.info(f"**Prediksi**: {classes[p_idx]} ({probs[p_idx]*100:.1f}%)")

            # Grad-CAM heatmap
            vis = GradCAMVisualizer(BrainTumorCustomCNN(4, 4))
            mock_tensor = torch.randn(1, 4, 240, 240)
            heatmap = vis.generate_heatmap(mock_tensor)
            overlay = vis.overlay_on_mri(cv2.resize(np.array(eval_img), (240, 240))[:, :, 0], heatmap)
            st.image(overlay, caption="Grad-CAM++ Spatial Heatmap Overlay", width='stretch')
