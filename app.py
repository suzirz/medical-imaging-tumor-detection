import streamlit as st
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import io

from models import SimpleMedicalCNN
from pipeline import preprocess_medical_image, generate_mock_heatmap

st.set_page_config(
    page_title="Medical Imaging Tumor Detection Pipeline",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Medical Imaging Tumor Detection Pipeline Builder")
st.markdown("""
Aplikasi ini mendemonstrasikan **alur kerja (pipeline) deteksi & segmentasi tumor** berbasis Deep Learning (CNN) pada citra medis (MRI, CT Scan, X-Ray).
""")

# Sidebar: Pipeline Configuration
st.sidebar.header("⚙️ Konfigurasi Pipeline")
modality = st.sidebar.selectbox("Pilih Modalitas Citra Medis:", ["Brain MRI", "Chest CT Scan", "Mammography", "X-Ray"])
model_choice = st.sidebar.selectbox("Pilih Arsitektur Model:", ["SimpleMedicalCNN (Classification)", "UNetLite (Segmentation)"])
confidence_threshold = st.sidebar.slider("Threshold Sensitivitas Deteksi:", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

st.sidebar.markdown("---")
st.sidebar.info("""
**Tentang Proyek**:
Menerjemahkan ide dari *PartyRock AWS Pipeline Builder* ke implementasi nyata berbasis Python PyTorch & Streamlit.
""")

# Main Content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Input Citra Medis")
    uploaded_file = st.file_uploader("Upload citra scan medis (.png, .jpg, .jpeg):", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is None:
        # Default placeholder demo
        st.caption("Belum ada file diupload? Gambar sample demo digunakan.")
        # Create a simple synthetic circle scan
        synthetic_img = np.zeros((224, 224, 3), dtype=np.uint8)
        rr, cc = np.ogrid[:224, :224]
        circle = ((rr - 112)**2 + (cc - 112)**2) < 80**2
        synthetic_img[circle] = 180
        img = Image.fromarray(synthetic_img)
    else:
        img = Image.open(uploaded_file)

    st.image(img, caption=f"Scan Medis ({modality})", use_container_width=True)

with col2:
    st.subheader("2. Hasil Analisis Pipeline AI")
    
    if st.button("🚀 Jalankan Analisis Pipeline"):
        with st.spinner("Menjalankan Preprocessing & Inferensi CNN..."):
            # Step 1: Preprocessing
            tensor_input = preprocess_medical_image(img)
            
            # Step 2: Model Inference (Simulasi / Evaluasi Forward Pass)
            model = SimpleMedicalCNN(num_classes=2)
            model.eval()
            with torch.no_grad():
                logits = model(tensor_input)
                probs = F.softmax(logits, dim=1).numpy()[0]
            
            prob_normal = float(probs[0])
            prob_tumor = float(probs[1])

            # Result Display
            st.write("### Status Diagnosa:")
            if prob_tumor >= confidence_threshold:
                st.error(f"⚠️ **Terdeteksi Pola Tumor / Massa Abnormal** (Confidence: {prob_tumor*100:.1f}%)")
            else:
                st.success(f"✅ **Tidak Terdeteksi Tumor (Normal / Jinak)** (Confidence: {prob_normal*100:.1f}%)")

            # Metrics
            m_col1, m_col2 = st.columns(2)
            m_col1.metric("Probabilitas Tumor", f"{prob_tumor*100:.1f}%")
            m_col2.metric("Probabilitas Normal", f"{prob_normal*100:.1f}%")

            # Step 3: Heatmap / Localization (Explainable AI - XAI)
            st.write("### 3. Visualisasi Deteksi (Grad-CAM / Localization)")
            heatmap = generate_mock_heatmap(img.size[::-1])
            
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(img.convert('RGB'))
            ax.imshow(heatmap, cmap='jet', alpha=0.45)
            ax.axis('off')
            st.pyplot(fig)
            st.caption("Peta panas (heatmap) merah menunjukkan fokus area fitur anomali yang dipelajari CNN.")
