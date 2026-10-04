# Dashboard Prediksi & Simulasi Emisi Karbon Hutan Indonesia

Platform analitik dan simulasi interaktif berbasis *Machine Learning* dan data geospasial hilangnya tutupan pohon di Indonesia (2001–2025). Aplikasi ini dirancang untuk pemantauan dinamika deforestasi, proyeksi emisi jangka menengah (2026–2030), deteksi anomali karhutla, serta perencanaan intervensi mitigasi iklim di tingkat **Nasional**, **Provinsi**, hingga **Kabupaten/Kota**.

---

## Fitur Utama

1. **Tren Historis & Dinamika Pemicu:**
   - Visualisasi dinamika kehilangan tutupan pohon tahunan (2001–2025).
   - Penguraian pemicu emisi: perkebunan/sawit, ladang berpindah, kebakaran hutan, pembalakan, tambang, dan infrastruktur.
   - Analisis proporsi kehilangan hutan alam primer (*carbon-dense*).

2. **Peta & Prioritas Emisi Wilayah:**
   - Pemetaan kloropet spasial dan pemeringkatan wilayah penyumbang emisi terbesar (*Pareto principle*).
   - Filter jendela waktu historis (1 hingga 25 tahun terakhir).

3. **Emisi Bersih (Net Flux) & Hotspot Karbon Tersembunyi:**
   - Evaluasi neraca emisi kotor vs serapan alami vegetasi hutan (*Net Sink* vs *Net Source*).
   - Identifikasi wilayah dengan rasio pelepasan karbon melebihi biomassa tegakan pohon (*deep peatland carbon release*).

4. **Tipologi Wilayah Kabupaten:**
   - Pengelompokan 497 kabupaten/kota ke dalam 5 tipe biofisik dan pola pemicu emisi berbasis PCA dan Klasterisasi.

5. **Proyeksi Emisi 2026–2030:**
   - Proyeksi emisi masa depan menggunakan *Quantile Regression* dengan rentang ketidakpastian P10 hingga P90.
   - Evaluasi keselarasan terhadap target penurunan emisi *Enhanced NDC* (target FOLU Net Sink 2030).

6. **Deteksi Anomali Karhutla (Nasional):**
   - Identifikasi lonjakan deforestasi ekstrem menggunakan pendekatan statistik *Robust Z-Score* (Median Absolute Deviation) terkait peristiwa iklim El Niño.

7. **Tools Simulasi & Mitigasi Kebijakan:**
   - Simulator respon emisi interaktif terhadap modifikasi tutupan lahan per aktivitas pemicu.
   - Estimasi potensi reduksi emisi dan valuasi nilai ekonomi karbon (skema bursa karbon / insentif iklim).

---

## Menjalankan Aplikasi Secara Lokal

### 1. Prasyarat
- Python 3.10, 3.11, atau 3.12
- Git

### 2. Instalasi Dependensi
Pastikan berada di direktori proyek, kemudian pasang dependensi:
```bash
pip install -r requirements.txt
```

### 3. Menjalankan Dashboard
```bash
streamlit run app.py
```
Aplikasi akan otomatis terbuka di browser pada alamat `http://localhost:8501`.

---

## Panduan Deploy ke Streamlit Community Cloud

1. **Push Proyek ke GitHub:**
   - Buat repositori baru di GitHub (publik atau privat).
   - Inisialisasi dan push seluruh berkas proyek ke repositori tersebut:
     ```bash
     git init
     git add .
     git commit -m "Initial commit: Forest Carbon Emission Dashboard"
     git branch -M main
     git remote add origin <URL_REPOSITORY_ANDA>
     git push -u origin main
     ```

2. **Deploy di Streamlit Cloud:**
   - Kunjungi [share.streamlit.io](https://share.streamlit.io) dan masuk dengan akun GitHub Anda.
   - Klik tombol **"New app"**.
   - Pilih repositori, branch (`main`), dan tentukan berkas utama:
     - **Main file path:** `app.py`
   - Klik **"Deploy!"**. Streamlit Cloud akan membaca `requirements.txt` dan menjalankan aplikasi secara otomatis.

---

## Struktur Berkas Repository

```text
├── .streamlit/
│   └── config.toml                  # Konfigurasi server & antarmuka Streamlit Cloud
├── .gitignore                       # Berkas pengabaian git (venv, cache, temporary files)
├── app.py                           # Skrip utama aplikasi dashboard Streamlit
├── requirements.txt                 # Daftar dependensi Python produksi
├── README.md                        # Dokumentasi proyek dan panduan deployment
├── model_emission.joblib            # Artefak model inferensi HistGradientBoosting (mean, q10, q90)
├── metadata_carbon.joblib           # Metadata variabel, lookup, dan state awal tutupan lahan
├── processed_forest_carbon.csv      # Data panel historis emisi dan deforestasi (2001–2025)
├── indonesia_provinces.geojson      # Batas administrasi geospasial provinsi
├── indonesia_kabupaten.geojson      # Batas administrasi geospasial kabupaten/kota
├── cluster_table.csv                # Data pengelompokan tipologi klaster wilayah
├── cluster_summary.csv              # Ringkasan profil tipologi klaster
├── forecast_2026_2030.csv           # Hasil proyeksi emisi jangka menengah
├── forecast_backtest.csv            # Metrik evaluasi akurasi model proyeksi
├── net_table.csv                    # Neraca emisi bersih (gross, removals, net flux)
├── pareto.csv                       # Analisis kontribusi kumulatif emisi per wilayah
├── intensity_hotspots.csv           # Wilayah terindikasi emisi gambut dalam ekstrem
├── anomalies.csv                    # Deteksi kejadian lonjakan emisi anomali historis
├── driver_intensity.csv             # Koefisien intensitas emisi per pemicu
└── shap_driver.csv                  # Nilai kontribusi pemicu (SHAP analysis)
```
