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
Untuk mencegah penyakit hawar daun bakteri (Bacterial Leaf Blight) pada padi, tanam varietas tahan, jaga sanitasi lahan dengan membersihkan gulma dan sisa tanaman, kelola air dengan baik, lakukan pemupukan berimbang, serta terapkan perlakuan benih dan pemantauan rutin. Jika serangan terjadi, gunakan pestisida bakterisida sesuai anjuran untuk menghentikan penyebarannya. 
**Sumber:**
- [distan.bulelengkab.go.id](https://distan.bulelengkab.go.id/informasi/detail/artikel/55_penyakit-hawar-daun-bakteri-pada-tanaman-padi)
- [eos.com](https://eos.com/blog/leaf-blight/#:~:text=Untuk%20mengendalikan%20penyakit%20hawar%20daun,juga%20dapat%20mengurangi%20keparahan%20penyakit.)
- [pertanian.ngawikab.go.id](https://pertanian.ngawikab.go.id/2023/01/12/hama-penyakit-bakteri-pada-tanaman-padi/#:~:text=Salah%20satu%20cara%20untuk%20menangani,penyakit%20tersebut%20berkembang%20lebih%20lanjut.)
""",

    "Brown Spot": """
Untuk menghindari penyakit Brown Spot (Bercak Daun Coklat) pada padi, lakukan sanitasi lahan, gunakan varietas tahan penyakit, atur jarak tanam agar tidak terlalu rapat dan gemburkan tanah, lakukan pemupukan berimbang, rendam benih dalam air panas sebelum tanam, dan kontrol kelembaban dengan pengaturan pengairan serta pengelolaan gulma, serta gunakan fungisida kontak dan sistemik sebagai tindakan pencegahan dan pengendalian jika diperlukan. 
**Sumber:**
- [plantix.net](https://plantix.net/id/library/plant-diseases/100064/brown-spot-of-rice/)
- [cybex.id](https://cybex.id/artikel/52964/penyakit-bercak-daun-coklat-brown-leaf-spot-pada-tanaman-padi--pengendaliannya/)
- [researchgate.net](https://www.researchgate.net/publication/324778944_Kajian_Intensitas_Penyakit_Bercak_Coklat_Sempit_Cercospora_oryzae_dan_Teknik_Pengendaliannya_pada_Pertanaman_Padi_di_Kecamatan_Tanggul_Kabupaten_Jember/fulltext/5ae1dcc6a6fdcc91399fbfb4/Kajian-Intensitas-Penyakit-Bercak-Coklat-Sempit-Cercospora-oryzae-dan-Teknik-Pengendaliannya-pada-Pertanaman-Padi-di-Kecamatan-Tanggul-Kabupaten-Jember.pdf)
- [gardeningknowhow.com](https://www.gardeningknowhow.com/edible/grains/rice/treating-rice-brown-leaf-spot.htm#:~:text=Mengobati%20Bercak%20Coklat%20pada%20Daun%20Padi&text=Infeksi%20ini%20terjadi%20ketika%20daun,produksi%20dan%20kualitas%20tanaman%20Anda.&text=Daftar%20untuk%20buletin%20Gardening%20Know,Cara%20Menanam%20Tomat%20yang%20Lezat%22.)
""",

    "Healthy": """
Untuk mempertahankan daun padi yang sehat dan menghindarinya dari penyakit, Anda perlu menerapkan praktik pengelolaan yang baik, meliputi penggunaan varietas tahan penyakit, pemupukan berimbang, menjaga sanitasi lahan dari gulma dan sisa tanaman, serta melakukan penyiangan dan pengaturan air secara rutin. Selain itu, penggunaan bakterisida dan fungisida yang tepat dosisnya dapat membantu mengendalikan serangan penyakit secara kuratif. 
**Sumber:**
- [petani-sejahtera.basf.co.id](https://petani-sejahtera.basf.co.id/news/Tanaman-Padi-Bebas-dari-Bakteri-dan-Jamur-Ini-Caranya#:~:text=Menggunakan%20Fungisida%20dan%20Bakterisida,untuk%20mendapatkan%20informasi%20lebih%20lanjut.)
- [gdm.id](https://gdm.id/cara-mengatasi-daun-padi-yang-mengering-2/#:~:text=Cara%20mengatasi%20daun%20padi%20yang%20mengering%20kedua%20adalah%20dengan%20melakukan,dari%20sampah%20atau%20kotoran%20lain.)
- [distan.bulelengkab.go.id](https://distan.bulelengkab.go.id/informasi/detail/artikel/55_penyakit-hawar-daun-bakteri-pada-tanaman-padi)
- [agri.kompas.com](https://agri.kompas.com/read/2022/10/22/111749384/cara-mengatasi-penyakit-hawar-daun-padi-yang-efektif-dan-efisien?page=all)
""",

    "Narrow Brown Spot": """
Untuk mencegah penyakit Narrow Brown Spot pada padi, gunakan varietas tahan penyakit, tanam dengan jarak yang lebih longgar, lakukan pemupukan berimbang (hindari kelebihan Urea), pastikan sanitasi lahan yang baik, dan gunakan fungisida sebagai pencegahan jika diperlukan. Kondisi kelembapan tinggi dan terlalu lama, serta penggunaan pupuk nitrogen berlebih, dapat memperburuk penyakit ini. 
**Sumber:**
- [researchgate.net](https://www.researchgate.net/publication/324778944_Kajian_Intensitas_Penyakit_Bercak_Coklat_Sempit_Cercospora_oryzae_dan_Teknik_Pengendaliannya_pada_Pertanaman_Padi_di_Kecamatan_Tanggul_Kabupaten_Jember/fulltext/5ae1dcc6a6fdcc91399fbfb4/Kajian-Intensitas-Penyakit-Bercak-Coklat-Sempit-Cercospora-oryzae-dan-Teknik-Pengendaliannya-pada-Pertanaman-Padi-di-Kecamatan-Tanggul-Kabupaten-Jember.pdf)
- [cybex.id](https://cybex.id/artikel/52964/penyakit-bercak-daun-coklat-brown-leaf-spot-pada-tanaman-padi--pengendaliannya/)
- [gardeningknowhow.com](https://www.gardeningknowhow.com/edible/grains/rice/treating-rice-brown-leaf-spot.htm#:~:text=Mengobati%20Bercak%20Coklat%20pada%20Daun%20Padi&text=Infeksi%20ini%20terjadi%20ketika%20daun,produksi%20dan%20kualitas%20tanaman%20Anda.&text=Daftar%20untuk%20buletin%20Gardening%20Know,Cara%20Menanam%20Tomat%20yang%20Lezat%22.)
- [plantix.net](https://plantix.net/en/library/plant-diseases/100064/brown-spot-of-rice/)
""",

    "Tungro": """
Untuk mencegah penyakit Tungro, lakukan pengendalian terpadu dengan menanam varietas padi tahan tungro, melakukan tanam serempak di hamparan luas, mengatur waktu tanam agar tidak bersamaan dengan puncak populasi wereng hijau, serta melakukan sanitasi lahan dengan memusnahkan gulma dan tanaman sakit. Selain itu, pelestarian musuh alami wereng dan penggunaan insektisida selektif pada waktu yang tepat juga dapat membantu mengendalikan vektor penyakit. 
**Sumber:**
- [distan.bulelengkab.go.id](https://distan.bulelengkab.go.id/informasi/detail/artikel/cara-pengendalian-penyakit-tungro-pada-padi-36#:~:text=Beberapa%20teknologi%20pengendalian%20tungro%20yang,serempak%20dan%20penanaman%20sepanjang%20tahun.)
- [distan.bulelengkab.go.id](https://distan.bulelengkab.go.id/informasi/detail/artikel/34_mengenal-penyakit-tungro-pada-padi-dan-cara-mengatasinya)
- [distan.bulelengkab.go.id](https://distan.bulelengkab.go.id/informasi/detail/artikel/89_wereng-hijau-nephotettix-virescens-vektor-utama-penyakit-tungro-pada-tanaman-padi#:~:text=Pengendalian%20penyakit%20tungro%20harus%20dilakukan,peningkatan%20populasi%20walaupun%20hanya%20sedikit.)
- [kumparan.com](https://kumparan.com/seputar-hobi/cara-pengendalian-penyakit-tungro-untuk-mengurangi-risiko-serangan-23cwJTAE6kU)
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


