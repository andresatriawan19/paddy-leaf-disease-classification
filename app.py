import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import base64
from io import BytesIO

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Deteksi Penyakit Daun Padi",
    page_icon="🌾",
    layout="centered"
)

# --- Load Model ---
model = load_model("mobilenetv2_rice.keras")

# --- Label Kelas ---
class_labels = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy",
    "Narrow Brown Spot",
    "Tungro"
]

# --- Saran Penanganan ---
treatment_suggestions = {
    "Bacterial Leaf Blight": """
**Bacterial Leaf Blight (Hawar Daun Bakteri)**  
🟢 *Gejala:* bercak kekuningan memanjang dari ujung dan tepi daun.  
🛠 **Penanganan:**
- Gunakan varietas tahan (Inpari 32, Ciherang Sub 1, IR64 BLB)
- Hindari pupuk nitrogen berlebih
- Rotasi tanaman
- Jaga kebersihan lahan dan irigasi
- Semprot bakterisida (streptomisin / kasugamisin) bila parah
""",
    "Brown Spot": """
**Brown Spot (Bercak Coklat)**  
🟠 *Gejala:* bercak bulat kecil coklat pada daun, batang, dan malai.  
🛠 **Penanganan:**
- Gunakan benih sehat (Inpari 33)
- Perbaiki drainase
- Semprot fungisida (mankozeb, trifloksistrobin + tebukonazol)
- Tambahkan kalium & fosfor
- Bersihkan sisa jerami terinfeksi
""",
    "Healthy": """
**Daun Sehat**  
🟩 *Tidak ada gejala penyakit.*  
✅ **Rekomendasi:**
- Pertahankan pola tanam dan pemupukan berimbang
- Cek daun berkala
- Jaga kebersihan area sawah
""",
    "Narrow Brown Spot": """
**Narrow Brown Spot (Bercak Coklat Sempit)**  
🟡 *Gejala:* bercak sempit & panjang coklat gelap sejajar tulang daun.  
🛠 **Penanganan:**
- Gunakan varietas tahan (Inpari 42)
- Pupuk nitrogen secukupnya
- Semprot fungisida (propikonazol / azoksistrobin)
- Perbaiki pengairan
""",
    "Tungro": """
**Tungro (Virus oleh Wereng Hijau)**  
🔴 *Gejala:* daun kuning-oranye, tanaman kerdil.  
🛠 **Penanganan:**
- Gunakan varietas tahan (Inpari 36 / Inpari 19)
- Kendalikan wereng (imidakloprid / fipronil)
- Tanam serempak
- Cabut tanaman terinfeksi berat
"""
}

# --- Fungsi Utility ---
def image_to_base64(img: Image.Image) -> str:
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# --- Header ---
st.markdown("<h1 style='text-align: center; color: #2e7d32;'>🌾 Deteksi Penyakit Daun Padi</h1>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align: center;'>
Aplikasi ini menggunakan model <strong>MobileNetV2</strong> untuk mengklasifikasikan penyakit daun padi dan memberikan solusi penanganan.  
Didesain untuk membantu petani melakukan deteksi dini di lapangan.
</p>
""", unsafe_allow_html=True)
st.write("---")

# --- Upload Gambar ---
uploaded_file = st.file_uploader("📤 Unggah gambar daun padi", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')

    # Tampilkan gambar
    img_base64 = image_to_base64(img)
    st.markdown(
        f"""
        <div style="text-align: center;">
            <img src="data:image/png;base64,{img_base64}" width="300" style="border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.2);" />
            <p><em>Pratinjau Gambar</em></p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Preprocessing
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediksi
    prediction = model.predict(img_array)
    predicted_class_index = int(np.argmax(prediction))
    confidence_score = float(prediction[0][predicted_class_index])  # skala 0-1
    predicted_label = class_labels[predicted_class_index]

    # Hasil Prediksi
    st.write("---")
    st.markdown("### 🧠 Hasil Prediksi")
    st.success(f"**Jenis Penyakit: {predicted_label}**")
    st.markdown("#### 🔍 Tingkat Keyakinan Model")
    st.progress(confidence_score)  # skala 0-1
    st.markdown(f"<p style='text-align:center;font-size:20px;'><strong>{confidence_score*100:.2f}%</strong></p>", unsafe_allow_html=True)

    # Saran Penanganan
    st.write("---")
    st.markdown("### 💡 Rekomendasi Penanganan")
    st.markdown(treatment_suggestions[predicted_label])

# --- Footer ---
st.write("---")
st.markdown(
    "<p style='text-align: center; font-size: 13px;'>© 2025 | Deteksi Penyakit Daun Padi • MobileNetV2 | Untuk Petani Indonesia</p>",
    unsafe_allow_html=True
)
