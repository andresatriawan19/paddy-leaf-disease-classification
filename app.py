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

# --- Saran Penanganan Detail ---
treatment_suggestions = {
    "Bacterial Leaf Blight": """
**Bacterial Leaf Blight (Hawar Daun Bakteri)**  
🟢 *Gejala:* bercak kekuningan memanjang dari ujung dan tepi daun, sering menyebabkan pengeringan daun.  
🛠 **Tindakan Penanganan:**
- Gunakan varietas padi tahan hawar seperti Inpari 32, Ciherang Sub 1, atau IR64 BLB.
- Hindari penggunaan pupuk nitrogen secara berlebihan karena dapat memperparah penyakit.
- Lakukan rotasi tanaman untuk memutus siklus bakteri.
- Jaga kebersihan lahan dan saluran irigasi, hindari genangan air.
- Semprotkan bakterisida berbahan aktif streptomisin atau kasugamisin bila serangan berat.
""",
    "Brown Spot": """
**Brown Spot (Bercak Coklat)**  
🟠 *Gejala:* bercak bulat kecil berwarna coklat pada daun, batang, dan malai.  
🛠 **Tindakan Penanganan:**
- Gunakan benih sehat dan tahan penyakit seperti varietas Inpari 33.
- Perbaiki drainase sawah agar air tidak menggenang.
- Semprot fungisida berbahan aktif mankozeb atau trifloksistrobin + tebukonazol.
- Tambahkan pemupukan kalium dan fosfor untuk memperkuat ketahanan tanaman.
- Lakukan sanitasi lahan dari sisa-sisa jerami terinfeksi.
""",
    "Healthy": """
**Daun Sehat**  
🟩 *Tidak ada gejala penyakit.*  
✅ **Rekomendasi:**
- Pertahankan pola tanam yang baik dan pemupukan berimbang.
- Cek daun secara berkala agar deteksi dini tetap dapat dilakukan.
- Jaga kebersihan area persawahan untuk mencegah munculnya penyakit baru.
""",
    "Narrow Brown Spot": """
**Narrow Brown Spot (Bercak Coklat Sempit)**  
🟡 *Gejala:* bercak sempit dan panjang berwarna coklat gelap sejajar dengan tulang daun.  
🛠 **Tindakan Penanganan:**
- Gunakan varietas tahan seperti Inpari 42.
- Lakukan pemupukan nitrogen secukupnya, tidak berlebihan.
- Semprot fungisida seperti propikonazol atau azoksistrobin jika gejala meluas.
- Perbaiki sistem pengairan dan jangan biarkan tanaman terlalu lembab.
""",
    "Tungro": """
**Tungro (Virus yang Ditransmisikan Wereng Hijau)**  
🔴 *Gejala:* daun menjadi kuning-oranye, tanaman kerdil dan pertumbuhan terhambat.  
🛠 **Tindakan Penanganan:**
- Gunakan varietas tahan seperti Inpari 36 atau Inpari 19.
- Kendalikan vektor (wereng hijau) dengan insektisida berbahan imidakloprid atau fipronil.
- Lakukan tanam serempak agar populasi wereng tidak berpindah ke tanaman muda.
- Cabut tanaman yang terinfeksi berat agar tidak menular ke tanaman lain.
"""
}

# --- Fungsi Utility ---
def image_to_base64(img: Image.Image) -> str:
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# --- Tampilan Header ---
st.markdown("<h1 style='text-align: center; color: #2e7d32;'>🌾 Deteksi Penyakit Daun Padi</h1>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align: center;'>
Aplikasi ini menggunakan model <strong>MobileNetV2</strong> untuk mengklasifikasikan jenis penyakit pada daun padi dan memberikan solusi penanganan langsung berdasarkan gejala yang terdeteksi.  
Ditujukan untuk membantu petani melakukan deteksi dini dan penanganan cepat di lapangan.
</p>
""", unsafe_allow_html=True)
st.write("---")

# --- Upload Gambar ---
uploaded_file = st.file_uploader("📤 Silakan unggah gambar daun padi yang ingin didiagnosis", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')

    # Tampilkan gambar
    img_base64 = image_to_base64(img)
    st.markdown(
        f"""
        <div style="text-align: center;">
            <img src="data:image/png;base64,{img_base64}" width="300" style="border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.2);" />
            <p><em>Pratinjau Gambar Daun Padi</em></p>
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
    predicted_class_index = np.argmax(prediction)
    confidence_score = prediction[0][predicted_class_index] * 100
    predicted_label = class_labels[predicted_class_index]

    # Hasil Prediksi
    st.write("---")
    st.markdown("### 🧠 Hasil Prediksi")
    st.success(f"**Jenis Penyakit: {predicted_label}**")
    st.markdown("#### 🔍 Tingkat Keyakinan Model")
    st.progress(float(confidence_score) / 100)
    st.markdown(f"<p style='text-align:center;font-size:20px;'><strong>{confidence_score:.2f}%</strong></p>", unsafe_allow_html=True)

    # Saran Penanganan
    st.write("---")
    st.markdown("### 💡 Rekomendasi Penanganan")
    st.markdown(treatment_suggestions[predicted_label])

# --- Footer ---
st.write("---")
st.markdown(
    "<p style='text-align: center; font-size: 13px;'>© 2025 | Aplikasi Deteksi Penyakit Daun Padi • MobileNetV2 | Dibuat untuk membantu petani Indonesia</p>",
    unsafe_allow_html=True
)

