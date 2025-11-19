import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# -----------------------------------------------------------------------------
# 1. PERSIAPAN DATA & PELATIHAN MODEL (Model Linear Regression)
# -----------------------------------------------------------------------------

# Fungsi untuk memuat data, membersihkan, dan melatih model
@st.cache_resource
def load_and_train_model():
    """Memuat data, membersihkan, melakukan OHE, menskalakan, dan melatih model LR."""
    # --- CATATAN PENTING ---
    # Asumsikan file CSV sudah tersedia di lingkungan deployment Streamlit.
    # Jika tidak, data dummy akan digunakan (seperti di bawah).
    try:
        df = pd.read_csv("Food_Delivery_Times.csv")
    except:
        # Membuat data dummy jika file CSV tidak ditemukan (untuk testing)
        data = {
            'Order_ID': range(200),
            'Delivery_Time_min': np.random.randint(15, 70, 200),
            'Distance_km': np.random.uniform(1, 20, 200),
            'Preparation_Time_min': np.random.randint(10, 40, 200),
            'Courier_Experience_yrs': np.random.uniform(0.5, 10, 200),
            'Weather': np.random.choice(['Sunny', 'Rainy', 'Foggy', 'Snowy', 'Windy'], 200),
            'Traffic_Level': np.random.choice(['Low', 'Medium', 'High'], 200),
            'Time_of_Day': np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night'], 200),
            'Vehicle_Type': np.random.choice(['Motorcycle', 'Scooter', 'Car'], 200),
        }
        df = pd.DataFrame(data)

    # Pembersihan Data (sesuai langkah-langkah dalam proyek)
    df.dropna(subset=['Delivery_Time_min'], inplace=True)
    for col in ['Weather', 'Traffic_Level', 'Time_of_Day', 'Vehicle_Type']:
        df[col].fillna(df[col].mode()[0], inplace=True)
    for col in ['Courier_Experience_yrs', 'Distance_km', 'Preparation_Time_min']:
        df[col].fillna(df[col].median(), inplace=True)
    df.drop('Order_ID', axis=1, inplace=True)

    # One-Hot Encoding (OHE)
    categorical_cols = ['Weather', 'Traffic_Level', 'Time_of_Day', 'Vehicle_Type']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    for col in df.columns:
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(int)

    # Persiapan Model
    X = df.drop(columns=['Delivery_Time_min'])
    y = df['Delivery_Time_min']
    X = X.astype(np.float64)

    # Split dan Scaling
    numerical_cols = ['Distance_km', 'Preparation_Time_min', 'Courier_Experience_yrs']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    # Hanya fit pada data training
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])

    # Pelatihan Model Linear Regression (Model terbaik Anda)
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    # Simpan nama-nama fitur untuk memastikan urutan input yang benar
    feature_names = X.columns.tolist()

    return lr_model, scaler, feature_names, numerical_cols

# Panggil fungsi pelatihan model
lr_model, scaler, feature_names, numerical_cols = load_and_train_model()


# -----------------------------------------------------------------------------
# 2. ANTARMUKA STREAMLIT
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Prediksi Waktu Pengiriman",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛵 Sistem Prediksi Waktu Pengiriman Makanan")
st.markdown("Aplikasi ini menggunakan model **Regresi Linear** (Best Model R²≈0.826) untuk memperkirakan waktu pengiriman.")

# Pembagian layout menjadi 2 kolom untuk input
col1, col2 = st.columns(2)

# Input Numerik
with col1:
    st.header("1. Data Pengiriman")
    distance = st.number_input(
        "Jarak Pengiriman (km)",
        min_value=1.0,
        max_value=100.0,
        value=5.0,
        step=0.5,
        help="Jarak dari restoran ke tujuan dalam kilometer."
    )
    preparation_time = st.number_input(
        "Waktu Persiapan Makanan (menit)",
        min_value=5,
        max_value=60,
        value=20,
        step=1,
        help="Perkiraan waktu yang dibutuhkan restoran untuk menyiapkan pesanan."
    )
    courier_experience = st.number_input(
        "Pengalaman Kurir (tahun)",
        min_value=0.0,
        max_value=20.0,
        value=3.0,
        step=0.1,
        help="Lama pengalaman kurir dalam tahun."
    )

# Input Kategorikal
with col2:
    st.header("2. Kondisi Lingkungan")
    weather = st.selectbox(
        "Kondisi Cuaca",
        ['Sunny', 'Rainy', 'Foggy', 'Snowy', 'Windy'],
        index=0
    )
    traffic = st.selectbox(
        "Tingkat Lalu Lintas",
        ['Low', 'Medium', 'High'],
        index=1
    )
    time_of_day = st.selectbox(
        "Waktu Hari",
        ['Morning', 'Afternoon', 'Evening', 'Night'],
        index=2
    )
    vehicle = st.selectbox(
        "Tipe Kendaraan",
        ['Motorcycle', 'Scooter', 'Car'],
        index=0
    )

# -----------------------------------------------------------------------------
# 3. FUNGSI PREDIKSI
# -----------------------------------------------------------------------------

def predict_delivery_time(lr_model, scaler, feature_names, numerical_cols, input_data):
    """Memproses input pengguna dan menghasilkan prediksi."""
    
    # Membuat DataFrame dari input pengguna
    input_df = pd.DataFrame({
        'Distance_km': [input_data['Distance_km']],
        'Preparation_Time_min': [input_data['Preparation_Time_min']],
        'Courier_Experience_yrs': [input_data['Courier_Experience_yrs']],
        'Weather': [input_data['Weather']],
        'Traffic_Level': [input_data['Traffic_Level']],
        'Time_of_Day': [input_data['Time_of_Day']],
        'Vehicle_Type': [input_data['Vehicle_Type']]
    })

    # Melakukan One-Hot Encoding pada data input
    categorical_cols_input = ['Weather', 'Traffic_Level', 'Time_of_Day', 'Vehicle_Type']
    input_encoded = pd.get_dummies(input_df, columns=categorical_cols_input, drop_first=True)
    
    # -------------------------------------------------------------------------
    # Menyamakan Kolom: Langkah Kritis!
    # Tambahkan kolom dummy yang hilang (nilai 0) dan hapus yang tidak perlu
    # Ini memastikan urutan kolom input sama persis dengan saat pelatihan model
    # -------------------------------------------------------------------------
    
    final_input = pd.DataFrame(0, index=[0], columns=feature_names)
    
    for col in final_input.columns:
        if col in input_encoded.columns:
            final_input[col] = input_encoded[col].values

    # -------------------------------------------------------------------------

    # Scaling Fitur Numerik
    # Transformasi menggunakan scaler yang sudah di-fit pada data training
    final_input[numerical_cols] = scaler.transform(final_input[numerical_cols])

    # Prediksi
    prediction = lr_model.predict(final_input)
    
    # Pastikan output prediksi tidak negatif
    return max(0, prediction[0])

# Tombol Prediksi
if st.button("Hitung Waktu Pengiriman"):
    
    input_data = {
        'Distance_km': distance,
        'Preparation_Time_min': preparation_time,
        'Courier_Experience_yrs': courier_experience,
        'Weather': weather,
        'Traffic_Level': traffic,
        'Time_of_Day': time_of_day,
        'Vehicle_Type': vehicle
    }
    
    try:
        predicted_time = predict_delivery_time(lr_model, scaler, feature_names, numerical_cols, input_data)
        
        # Tampilkan Hasil
        st.markdown("---")
        st.subheader("🎉 Hasil Prediksi")
        
        col_res1, col_res2 = st.columns([1, 2])
        
        with col_res1:
             # Menampilkan hasil dengan 2 angka di belakang koma
             st.metric(label="Waktu Pengiriman Diprediksi", value=f"{predicted_time:.2f} Menit")
        
        with col_res2:
            st.info(
                f"Prediksi ini menggunakan model Regresi Linear dengan R2 Score sekitar 82.6% (model terbaik)."
                f" Artinya, rata-rata kesalahan prediksi adalah sekitar {5.9:.2f} menit (MAE)."
            )

    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi. Pastikan data yang dimasukkan valid: {e}")

# -----------------------------------------------------------------------------
# 4. KETERANGAN TAMBAHAN
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption("Catatan: Model ini dilatih berdasarkan asumsi data dari Food_Delivery_Times.csv dan menggunakan Linear Regression karena kinerjanya yang superior.")
