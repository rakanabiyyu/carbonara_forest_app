# 📑 LAPORAN AUDIT & REVIEW METODOLOGI DATA SCIENCE
## Sistem Analisis, Prediksi, dan Simulasi Emisi Karbon Hutan Indonesia (LOBMA & Streamlit DSS)

> **Identitas Dokumen Reviewer**  
> * **Objek Review:** Notebook Analisis ([lobma_rapi.ipynb](file:///C:/Users/rakan/dev%20c/JECT/forest%20usb%202/lobma_rapi.ipynb)) dan Kode Dashboard ([app.py](file:///C:/Users/rakan/dev%20c/JECT/forest%20usb%202/app.py))  
> * **Peran Reviewer:** Senior Machine Learning Auditor & Environmental Data Science Peer Reviewer  
> * **Tujuan:** Membedah secara mendalam integritas metodologis, rekayasa fitur, keabsahan algoritma, manipulasi data di dashboard, serta memberikan rekomendasi tegas: **komponen mana yang metodologinya kuat (wajib dipertahankan)** dan **komponen mana yang rapuh/lemah (perlu dirombak atau disingkirkan)**.  
> * **Pendekatan Bahasa:** Menggunakan standar evaluasi saintifik yang ketat, namun dibawakan dengan bahasa yang membumi, analogi fundamental, dan mudah dipahami orang awam.

---

## DAFTAR ISI REVIEW
1. [EXECUTIVE SUMMARY: Rapor Keseluruhan Proyek](#1-executive-summary-rapor-keseluruhan-proyek)
2. [BEDAH TAHAP 1: Audit Data Mentah & Pembuatan Panel](#2-bedah-tahap-1-audit-data-mentah--pembuatan-panel)
3. [BEDAH TAHAP 2: Rekayasa Fitur & Formulasi Fisika](#3-bedah-tahap-2-rekayasa-fitur--formulasi-fisika)
4. [BEDAH TAHAP 3: Validasi Anti-Bocor (Data Leakage Controls)](#4-bedah-tahap-3-validasi-anti-bocor-data-leakage-controls)
5. [BEDAH TAHAP 4: Pemodelan Machine Learning & Regresi Kuantil](#5-bedah-tahap-4-pemodelan-machine-learning--regresi-kuantil)
6. [BEDAH TAHAP 5: Audit 8 Metode Data Mining Pendukung](#6-bedah-tahap-5-audit-8-metode-data-mining-pendukung)
7. [BEDAH TAHAP 6: Manipulasi & Tampilan Dashboard Streamlit (`app.py`)](#7-bedah-tahap-6-manipulasi--tampilan-dashboard-streamlit-apppy)
8. [VONIS AKHIR REVIEWER: Mana yang Dipertahankan vs Mana yang Disingkirkan](#8-vonis-akhir-reviewer-rekomendasi-pertahankan-vs-singkirkan)

---

# 1. EXECUTIVE SUMMARY: Rapor Keseluruhan Proyek

Secara keseluruhan, proyek data mining ini memiliki **kualitas metodologi di atas rata-rata proyek data science terapan di Indonesia**. Penulis notebook menunjukkan kesadaran tinggi terhadap jebakan-jebakan klasik data kehutanan (seperti kebocoran data spasial dan distorsi lahan gambut).

```
                            RAPOR METODOLOGI PROYEK
┌───────────────────────────────────────┬────────┬───────────────────────────────┐
│ Aspek Metodologi                      │ Nilai  │ Catatan Auditor               │
├───────────────────────────────────────┼────────┼───────────────────────────────┤
│ 1. Integritas Data & Audit            │ A (95) │ Sangat teliti; Left Join rapi │
│ 2. Rekayasa Fitur Fisika (Intensitas) │ A+ (98)│ Formulasi brilian anti-saturasi│
│ 3. Skema Validasi (Anti-Leakage)      │ A (95) │ Split masa depan & GroupKFold │
│ 4. Pemodelan Inti (HistGB WARM)       │ A (92) │ Kuat, cepat, ada rentang P10/90│
│ 5. Peramalan Deret Waktu (Time-Series)│ B+ (85)│ Baik di h=1..3, jujur di h=5  │
│ 6. Atribusi Pemicu (NNLS)             │ C (65) │ Lemah (masalah multikolinear) │
│ 7. Klasterisasi Wilayah (K-Means)     │ C+ (72)│ Silhouette 0,257 terlalu lemah│
│ 8. Arsitektur Deployment (Streamlit)  │ A (94) │ Efisien, modular, zero-train  │
└───────────────────────────────────────┴────────┴───────────────────────────────┘
```

---

# 2. BEDAH TAHAP 1: Audit Data Mentah & Pembuatan Panel

### Metodologi yang Digunakan:
Data diambil dari [IDN.xlsx](file:///C:/Users/rakan/dev%20c/JECT/forest%20usb%202/IDN.xlsx) milik Global Forest Watch (GFW). Penulis menyatukan 5 lembar kerja terpisah menjadi **Data Panel Terstruktur**: 497 kabupaten $\times$ 25 tahun (2001–2025) = **12.425 baris**.

```
[Sheet Karbon: 497 Kab] ─── LEFT JOIN ───► [Sheet Drivers: 494 Kab]
                                            │
                                            ▼
                    (3 Kabupaten kecil tidak hilang, diisi 0)
```

### Analisis Kritis Reviewer:
* **Kekuatan Utama:**  
  Keputusan menggunakan **`LEFT JOIN`** (bukan `INNER JOIN`) adalah langkah penyelamat. Banyak data scientist pemula tanpa sadar menggunakan `INNER JOIN` yang secara otomatis membuang 3 kabupaten/kota kepulauan kecil karena tidak tercatat di data pemicu. Jika dibuang, representasi nasional akan cacat.
* **Penanganan Data Kosong (*Missing Values*):**  
  Terdapat 483 baris tanpa catatan pemicu yang diisi dengan angka `0`. Ini sah dan masuk akal secara fisik, karena ketiadaan catatan satelit untuk pemicu komersial di pulau-pulau kecil umumnya berarti tidak adanya deforestasi industri yang signifikan.
* **Verifikasi Konsistensi ($0{,}998$):**  
  Penulis menguji penjumlahan 7 pemicu terhadap total kehilangan tutupan pohon (`tc_loss_ha`). Rasio kecocokannya mencapai **99,83%**. Ini bukti audit data yang sangat disiplin sebelum melangkah ke pemodelan.

---

# 3. BEDAH TAHAP 2: Rekayasa Fitur & Formulasi Fisika

Tahap ini adalah **mahakarya terbaik** dari notebook ini.

### 1. Formulasi Target Intensitas: $\log((E+1)/(L+1))$
* **Masalah di Lapangan:**  
  Jika model Machine Learning disuruh langsung menebak total emisi ($y = \text{gross\_emissions}$), model pohon seperti Random Forest atau HistGB akan **gagal total saat simulasi skenario ekstrem**. Algoritma pohon tidak bisa memprediksi angka di luar nilai maksimum yang pernah dilihat di data latih (*tree saturation*).
* **Solusi Penulis:**  
  Model dilatih untuk memprediksi **Log-Intensitas Emisi per Hektare**:
  $$\text{Target} = \log\left(\frac{E + 1}{\text{Loss\_ha} + 1}\right)$$
  Saat prediksi, hasilnya dikalikan kembali dengan luas lahan:
  $$\hat{E} = (\text{Loss\_ha} + 1) \cdot e^{\widehat{\text{Target}}} - 1$$
* **Penilaian Reviewer:**  
  **Brilian.** Formulasi ini memastikan prediksi emisi berskala proporsional secara fisika dengan luas hektare bukaan lahan. Mau pengguna memasukkan 100 ha atau 500.000 ha, hasilnya tidak akan mendatar/mentok.

### 2. Fitur Biofisika Karbon: `potential_co2e`
* Rumus: $\text{Luas Hilang} \times \text{Densitas Biomassa 2000} \times \frac{44}{12}$.
* **Penilaian Reviewer:**  
  Sangat tepat. Memasukkan hukum kimia dasar (stoikiometri konversi karbon ke gas $\text{CO}_2$ sebesar $44/12 \approx 3{,}67$) menyuntikkan *domain knowledge* ke dalam model matematika.

### 3. Fitur Riwayat Lokal: `hist_intensity` (Kunci Mode WARM)
* **Masalah:** Dataset satelit GFW ini tidak memiliki peta radar lapisan gambut bawah tanah. Padahal gambut di Riau dan Kalteng menyimpan emisi ribuan ton per hektare.
* **Solusi Penulis:**  
  Membuat fitur `hist_intensity`, yaitu rata-rata emisi per hektare kabupaten tersebut di tahun-tahun **sebelum** tahun berjalan.
* **Penilaian Reviewer:**  
  Ini adalah teknik proksi terselubung (*implicit proxy*) yang sangat cerdas. Tanpa perlu mengunduh peta geologi gambut yang berukuran gigabyte, model otomatis "menghafal" bahwa daerah seperti Pelalawan atau Pulang Pisau memiliki emisi per hektare yang jauh lebih tinggi daripada rata-rata nasional.

---

# 4. BEDAH TAHAP 3: Validasi Anti-Bocor (*Data Leakage Controls*)

### Jebakan Data Panel:
Pada data panel (ruang $\times$ waktu), melakukan pemisahan acak biasa (`train_test_split` acak) adalah **kesalahan fatal**. Jika data Kabupaten Riau tahun 2021 masuk data latih, dan Riau tahun 2022 masuk data uji, model akan tampak memiliki akurasi $99\%$ karena model cuma "menyontek" pola lokal Riau yang nyaris tidak berubah antar-tahun.

```
┌────────────────────────────────────────────────────────────────────────┐
│                      SKEMA VALIDASI ANTI-BOCOR                         │
├───────────────────────────────────┬────────────────────────────────────┤
│ 1. Uji Lintas Waktu (Temporal)    │ Latih data ≤ 2021 (21 tahun),      │
│                                   │ Uji ke masa depan 2022-2025.       │
├───────────────────────────────────┼────────────────────────────────────┤
│ 2. Uji Lintas Wilayah (Spasial)   │ GroupKFold(5) mengunci seluruh     │
│                                   │ baris satu kabupaten di 1 lipatan. │
├───────────────────────────────────┼────────────────────────────────────┤
│ 3. Larangan Fitur Bocor           │ Net flux & rerata emisi DIHAPUS    │
│                                   │ dari daftar fitur input.           │
└───────────────────────────────────┴────────────────────────────────────┘
```

### Penilaian Reviewer:
* **Sangat Disiplin:** Validasi masa depan (*Future Years 2022–2025*) membuktikan apakah model bisa dipakai untuk meramal tahun berikutnya.
* **GroupKFold Spasial:** Menjawab apakah model bisa dipakai di daerah pemekaran baru tanpa riwayat (*Cold Start*).
* **Penghapusan Fitur Terlarang:** Penulis secara sadar melarang *Net Flux* masuk sebagai fitur karena *Net Flux* dihitung dari variabel target (emisi kotor). Jika dimasukkan, itu adalah dosa besar pembocoran target (*target leakage*).

---

# 5. BEDAH TAHAP 4: Pemodelan Machine Learning & Regresi Kuantil

Di Bagian 3 notebook, dilakukan adu performa model pada tahun uji 2022–2025:

| Model | R² Uji (2022-25) | MAE Uji (Mg CO2e) | Status di Proyek |
|---|---|---|---|
| Ridge (Garis Lurus Baseline) | 0,917 | 343.294 | Pembanding dasar |
| Random Forest (Target Mentah) | 0,952 | 180.642 | Pembanding ensemble |
| HistGB (Target Mentah) | 0,939 | 184.962 | Eksperimen |
| HistGB (Log1p Target) | 0,966 | 170.128 | Eksperimen |
| HistGB Intensitas (Mode COLD) | 0,976 | 155.060 | Eksperimen daerah baru |
| **HistGB Intensitas (Mode WARM)** | **0,979** | **137.609** | **PEMENANG FINAL (DIDEPLOY)** |

### Analisis Kritis Reviewer:
1. **Kenapa HistGradientBoosting Menang atas Random Forest?**  
   HistGB membangun pohon secara bertahap (*boosting*), di mana pohon baru fokus memperbaiki kesalahan pohon sebelumnya. Ditambah pengelompokan data ke dalam 256 keranjang (*histogram binning*), model ini jauh lebih tangguh terhadap pencilan (*outliers*) dan sangat cepat dieksekusi.
2. **Penggunaan Regresi Kuantil (P10 dan P90):**  
   Penulis tidak hanya melatih model nilai tengah (*mean*), tetapi melatih dua model identik dengan fungsi kerugian kuantil: `loss="quantile", quantile=0.10` dan `quantile=0.90`.  
   *Penilaian Reviewer:* **Luar biasa.** Ini mengubah kalkulator emisi dari sekadar "tebakan angka tunggal yang sombong" menjadi "alat estimasi risiko dengan interval kepercayaan 80%".

---

# 6. BEDAH TAHAP 5: Audit 8 Metode Data Mining Pendukung

Bagian 4 notebook memuat 8 analisis lanjutan. Mari kita bedah keabsahan metodologi masing-masing:

```
                  EVALUASI METODOLOGI 8 TEKNIK DATA MINING
 ┌───────────────────────────┬──────────────┬───────────────────────────────┐
 │ Teknik Data Mining        │ Status Ilmiah│ Rekomendasi Reviewer          │
 ├───────────────────────────┼──────────────┼───────────────────────────────┤
 │ 1. Analisis Pareto        │ SANGAT KUAT  │ Wajib dipertahankan           │
 │ 2. Net Flux Analysis      │ SANGAT KUAT  │ Wajib dipertahankan           │
 │ 3. Hotspot Gambut         │ KUAT         │ Wajib dipertahankan           │
 │ 4. SHAP Interpretation    │ SANGAT KUAT  │ Wajib dipertahankan           │
 │ 5. Robust Z-Score Anomaly │ SANGAT KUAT  │ Wajib dipertahankan           │
 │ 6. Forecast Multi-Horizon │ CUKUP (h≤3)  │ Pertahankan h=1..3, revisi h=5│
 │ 7. NNLS Driver Intensity  │ LEMAH        │ PERTIMBANGKAN UNTUK DIREVISI  │
 │ 8. K-Means Clustering     │ MERAGUKAN    │ PERTIMBANGKAN UNTUK DIROMBAK  │
└───────────────────────────┴──────────────┴───────────────────────────────┘
```

### 1. Analisis Pareto (86 Kabupaten = 80% Emisi) ➔ **[SANGAT KUAT]**
* Mengurutkan kontribusi kumulatif emisi seluruh kabupaten.
* Memberikan pesan kebijakan yang sangat tajam bagi pemerintah: tidak perlu membagi rata anggaran ke 497 kabupaten; cukup awasi ketat 86 kabupaten teratas.

### 2. Atribusi Pemicu dengan NNLS (*Non-Negative Least Squares*) ➔ **[LEMAH & BERBAHAYA]**
* **Tujuan:** Menghitung emisi per hektare untuk tiap pemicu dengan mengunci nilai $\beta \ge 0$.
* **Temuan:** Di data nasional, koefisien untuk *Hard commodities* (tambang), *Settlements* (infrastruktur), dan *Disturbances* keluar dengan nilai **0,000 Mg CO2e/ha**.
* **Kritik Keras Reviewer:**  
  Ini terjadi karena fenomena **Multikolinearitas Parah**. Di lapangan, pembukaan tambang dan permukiman sering terjadi bersamaan dengan perluasan kebun sawit atau logging. Secara statistik, algoritma NNLS memberikan seluruh nilai koefisien ke pemicu yang paling dominan (sawit/logging), dan men-nol-kan pemicu lainnya!  
  Bagi orang awam atau pejabat daerah, tabel ini bisa disalahartikan: *"Oh, buka tambang batu bara emisinya 0 ton per hektar, berarti aman dong!"* **Ini penyesatan informasi yang tidak disengaja.**

### 3. Net Flux (Emisi Kotor vs Serapan) ➔ **[SANGAT KUAT]**
* Menemukan bahwa 308 dari 497 kabupaten di Indonesia sebenarnya berstatus *Net Sink* (penyerap bersih), tetapi secara nasional tetap menjadi *Net Source* (+349 Jt ton $\text{CO}_2\text{e}$/thn) akibat kerusakan masif di 189 kabupaten lainnya. Ini akurat secara biogeokimia.

### 4. Hotspot Karbon Tersembunyi ➔ **[KUAT]**
* Membandingkan rasio emisi aktual per hektare terhadap ekspektasi biomassa atas tanah ($E / \text{loss} \gg \text{AGB} \times 44/12$). Berhasil memetakan anomali gambut di Riau, Sumsel, dan Kalteng tanpa data spasial geologi tanah.

### 5. SHAP Interpretasi ➔ **[SANGAT KUAT]**
* Penulis secara cerdas membuat model terpisah **tanpa variabel agregat** (`total_loss_ha`). Jika variabel total dimasukkan, SHAP akan menaruh 99% kepentingan ke variabel total tersebut, dan menutupi peran masing-masing dari 7 pemicu.

### 6. Klasterisasi K-Means ($K=5$) ➔ **[MERAGUKAN / LEMAH]**
* **Temuan:** Nilai **Silhouette Score hanya $0{,}257$**.
* **Kritik Keras Reviewer:**  
  Dalam kaidah *Unsupervised Learning*, nilai Silhouette di bawah $0{,}30$ menandakan **struktur klaster yang sangat lemah / nyaris tidak ada pemisahan alami (*no substantial clustering structure*)**.  
  Titik-titik kabupaten sebenarnya menumpuk seperti gumpalan awan tunggal yang menyatu. Memaksa membaginya menjadi 5 kelompok adalah tindakan arbitrer (semu). Label seperti "Klaster C1" dan "Klaster C2" memiliki tumpang tindih (*overlap*) yang sangat besar di dunia nyata.

### 7. Deteksi Anomali dengan Robust Z-Score ➔ **[SANGAT KUAT]**
* Menggunakan Median dan MAD (*Median Absolute Deviation*) alih-alih Mean dan Standar Deviasi biasa.
* Terbukti berhasil mengisolasi 20 kejadian anomali, di mana 11 di antaranya menumpuk di tahun 2015–2016 (musim kebakaran dahsyat El Niño).

### 8. Peramalan Multi-Tahun (2026–2030) & Backtesting ➔ **[CUKUP KUAT DI h≤3, LEMAH DI h=5]**
* **Kelebihan:** Menggunakan *Direct Multi-Horizon* (bukan *recursive*), sehingga kesalahan tebak tahun 2026 tidak berlipat ganda ke tahun 2030.
* **Kejujuran Saintifik:** Mengakui bahwa pada tahun ke-5 ($h=5$), Machine Learning kalah akurat dibanding rumus rata-rata naif 3 tahun, sehingga sistem beralih ke metode naif.
* **Kelemahan:** Mengabaikan variabel cuaca makro (El Niño tidak bisa ditebak dari data lag 3 tahun).

---

# 7. BEDAH TAHAP 6: Manipulasi & Tampilan Dashboard Streamlit (`app.py`)

Aplikasi Streamlit bertindak sebagai *Decision Support System*. Berikut audit terhadap kode antarmuka dan manipulasinya:

```
┌───────────────────────────────────┬───────────────────────────────────┐
│ Modul di app.py                   │ Audit Teknis & Integritas Metoda  │
├───────────────────────────────────┼───────────────────────────────────┤
│ Tab 1: Simulasi What-If           │ SANGAT BAIK: Rekonstruksi baris   │
│                                   │ fitur (build_row) 100% konsisten  │
│                                   │ dengan logika training notebook.  │
├───────────────────────────────────┼───────────────────────────────────┤
│ Tab 2: Skenario Mitigasi & USD    │ PERLU CATATAN: Asumsi harga pasar │
│                                   │ karbon terlalu menyederhanakan    │
│                                   │ mekanisme sertifikasi kredit.     │
├───────────────────────────────────┼───────────────────────────────────┤
│ Tab 3: Rekomendasi Otomatis       │ AGAK KAKU: Berbasis If-Else pada  │
│                                   │ 1 pemicu dominan, mengabaikan     │
│                                   │ status izin tata ruang (APL/KPHP).│
├───────────────────────────────────┼───────────────────────────────────┤
│ Tab 4: Peta Choropleth GeoJSON    │ SANGAT BAIK: Visualisasi spasial  │
│                                   │ jernih, filter waktu dinamis.     │
├───────────────────────────────────┼───────────────────────────────────┤
│ Tab 7: Proyeksi & Confidence Band │ SANGAT BAIK: Menampilkan pita     │
│                                   │ arsiran P10-P90 secara jujur.     │
└───────────────────────────────────┴───────────────────────────────────┘
```

### 1. Konsistensi Rekonstruksi Fitur (`build_row`)
Di baris 72–98 [app.py](file:///C:/Users/rakan/dev%20c/JECT/forest%20usb%202/app.py#L72-L98), fungsi `build_row()` merekonstruksi 20 fitur secara langsung di memori browser. Penulis memastikan urutan kolom `[FEATURES]` sama persis dengan urutan saat training. Ini mencegah bug pergeseran kolom yang mematikan.

### 2. Tab Skenario Mitigasi & Valuasi Ekonomi Karbon
Di baris 221–228 [app.py](file:///C:/Users/rakan/dev%20c/JECT/forest%20usb%202/app.py#L221-L228):
$$\text{Nilai Ekonomi} = \text{Emisi Dicegah (Ton)} \times \text{Harga Karbon (USD)}$$
* **Kritik Reviewer:**  
  Rumus ini sangat memikat bagi bupati atau dinas kehutanan, tetapi **terlalu naif**. Di pasar karbon sukarela (*voluntary carbon market* seperti Verra/VCS), untuk mengubah 1 ton emisi yang dicegah menjadi uang tunai, ada biaya verifikasi yang mahal, potongan risiko kebocoran (*buffer pool* 10–20%), dan syarat *permanence*. Aplikasi harus memberi peringatan tebal bahwa ini adalah *potensi bruto teoretis*, bukan uang tunai yang langsung cair.

### 3. Rekomendasi Kebijakan Otomatis
Di baris 250–257 [app.py](file:///C:/Users/rakan/dev%20c/JECT/forest%20usb%202/app.py#L250-L257), aplikasi otomatis memberikan teks rekomendasi berdasarkan pemicu dominan.
* **Kritik Reviewer:**  
  Rekomendasi ini berbasis aturan kaku (*hardcoded if-else*). Contoh: jika pemicu dominan adalah *Settlements & Infrastructure*, rekomendasinya selalu: *"Arahkan tata ruang ke lahan terdegradasi..."* Ini saran yang bagus, namun belum mempertimbangkan batas legalitas kawasan (apakah lahan tersebut Hutan Lindung atau Area Penggunaan Lain / APL).

---

# 8. VONIS AKHIR REVIEWER: Rekomendasi Pertahankan vs Singkirkan

Berikut adalah panduan tegas bagi pengembang untuk menyempurnakan proyek ini:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        MATRIKS KEPUTUSAN REVIEWER                      │
├───────────────────────────────────┬────────────────────────────────────┤
│   ✅ PERTAHANKAN (CORE STRENGTH)   │   ❌ SINGKIRKAN / ROMBAK (WEAKNESS) │
├───────────────────────────────────┼────────────────────────────────────┤
│ 1. Formulasi Target Log-Intensitas│ 1. Atribusi Pemicu NNLS Mentah     │
│ 2. Validasi Future Years + GroupKF│ 2. Klasterisasi K-Means (Silh 0.25)│
│ 3. HistGB WARM (Proxy Gambut)     │ 3. Peramalan Jauh h=4 dan h=5      │
│ 4. Regresi Kuantil P10 & P90      │ 4. Kalkulator Dolar Karbon Naif    │
│ 5. Analisis Konsentrasi Pareto    │ 5. Rekomendasi Hardcoded Tunggal   │
│ 6. Deteksi Anomali Robust Z-Score │                                    │
│ 7. Arsitektur Batch-Decoupling    │                                    │
└───────────────────────────────────┴────────────────────────────────────┘
```

---

## A. METODE & FITUR YANG SANGAT KUAT (WAJIB DIPERTAHANKAN)

### 1. Formulasi Target Log-Intensitas: $\log((E+1)/(L+1))$
* **Alasan:** Ini adalah fondasi terkuat seluruh proyek. Menghindari model mengalami saturasi saat pengujian skenario tebangan pohon berskala besar di masa depan. Menjamin konsistensi hukum fisika.

### 2. Kerangka Validasi Anti-Bocor (*Future Years Split* + *GroupKFold*)
* **Alasan:** Memastikan model tidak "menyontek" data masa depan maupun menghafal identitas lokal kabupaten. Menghasilkan angka evaluasi ($R^2 = 0{,}979$, $\text{MAE} = 137\text{ ribu}$) yang jujur dan dapat dipertanggungjawabkan di hadapan akademisi internasional.

### 3. Fitur `hist_intensity` (Model Mode WARM)
* **Alasan:** Sukses memecahkan kebuntuan ketiadaan data spasial lahan gambut. Fitur ini secara elegan menangkap emisi bawah tanah yang tidak terlihat oleh satelit optik.

### 4. Rentang Ketidakpastian 80% (Model Kuantil P10 & P90)
* **Alasan:** Menyajikan sains secara dewasa dan bertanggung jawab. Mengakui bahwa alam memiliki faktor ketidakpastian cuaca yang tidak bisa ditebak dengan satu angka mutlak.

### 5. Analisis Pareto & Deteksi Anomali Robust Z-Score
* **Alasan:** Keduanya memberikan nilai terapan praktis yang luar biasa. Pareto membuktikan 86 kabupaten adalah kunci 80% masalah nasional, dan Robust Z-score membuktikan validitas data sejarah kebakaran El Niño 2015–2016.

### 6. Arsitektur *Decoupling* (Pemisahan Notebook dan Streamlit)
* **Alasan:** Keputusan menyimpan model ke `.joblib` dan hasil proyeksi ke `.csv` membuat aplikasi web berjalan instan dalam hitungan milidetik tanpa membebani server dengan proses training ulang.

---

## B. METODE & PROSES YANG PERLU DIPERTIMBANGKAN UNTUK DISINGKIRKAN / DIREVISI

### 1. ❌ SINGKIRKAN / ROMBAK: Atribusi Pemicu dengan NNLS Mentah
* **Masalah Utama:** Mengalami distorsi multikolinearitas. Pemicu penting seperti pertambangan (*Hard commodities*) dan infrastruktur keluar dengan nilai **0,000 Mg CO2e/ha**, yang secara fisika keliru dan menyesatkan persepsi pengguna awam.
* **Saran Perbaikan:**  
  - **Opsi A (Singkirkan):** Hapus grafik NNLS dari Tab 7 dashboard dan gantikan sepenuhnya dengan **Koefisien SHAP Global** yang terbukti jauh lebih adil dalam mendistribusikan kontribusi pemicu.
  - **Opsi B (Revisi):** Terapkan regresi terikat dengan penalti regularisasi (*Constrained ElasticNet / Ridge*) agar pemicu-pemicu yang berkorelasi tidak dipaksa menjadi nol mutlak.

---

### 2. ❌ ROMBAK: Klasterisasi K-Means dengan K=5 (Silhouette 0,257)
* **Masalah Utama:** Nilai Silhouette $0{,}257$ membuktikan bahwa tidak ada batas klaster alami yang tegas. Kabupaten-kabupaten di Indonesia membentuk kontinum gradasi, bukan 5 pulau terpisah. Label klaster ini rentan menjadi klasifikasi semu (*pseudo-science*).
* **Saran Perbaikan:**  
  - Ganti algoritma K-Means tanpa pengawasan (*unsupervised*) dengan **Segmentasi Matriks Berbasis Kebijakan (Policy Matrix Segmentation)**.  
  - Contoh: Bagilah kabupaten berdasarkan 2 kuadran nyata yang sudah jelas batasnya:  
    1. Sumbu X: Dominasi Lahan (Mayoritas Sawit vs Mayoritas Hutan Alami).  
    2. Sumbu Y: Status Karbon (Net Source vs Net Sink).  
  - Klasifikasi berbasis aturan ini jauh lebih kokoh, mudah dipahami kepala daerah, dan tidak bergantung pada angka klaster K-Means yang rapuh.

---

### 3. ⚠️ BATASI / REVISI: Peramalan Jangka Panjang ($h=4$ dan $h=5$)
* **Masalah Utama:** Deret waktu data hanya 25 tahun. Meramal 5 tahun ke depan (2030) tanpa memasukkan variabel iklim global (seperti anomali suhu muka laut Pasifik / siklus El Niño) adalah ekstrapolasi yang terlalu jauh. Buktinya di tahun ke-5 model ML kalah dari rata-rata naif 3 tahun.
* **Saran Perbaikan:**  
  - Batasi peramalan Machine Learning hanya untuk **horizon 1 sampai 3 tahun ke depan (2026–2028)**.
  - Untuk tahun 2029 dan 2030, jangan sebut sebagai "Ramalan Model ML", melainkan beri label transparan: **"Skenario Ekstrapolasi Tren Rata-Rata Historis"**.

---

### 4. ⚠️ PERBAIKI: Penyederhanaan Valuasi Finansial Karbon (Tab Mitigasi)
* **Masalah Utama:** Mengalikan langsung ton emisi yang dicegah dengan harga dollar tanpa memperhitungkan biaya sertifikasi dan risiko pembalikan (*non-permanence risk*) memberikan ekspektasi finansial yang tidak realistis bagi pembuat kebijakan.
* **Saran Perbaikan:**  
  - Tambahkan slider penyesuaian: *Faktor Risiko / Cadangan Pengaman (Buffer Pool)* sebesar 20%, dan biaya transaksi verifikasi.  
  - Tambahkan label peringatan metodologis: *"Nilai ini adalah valuasi indikatif teoritis sebelum dipotong biaya metodologi MRV (Measurement, Reporting, and Verification)."*

---

# 9. KESIMPULAN REVIEWER

Proyek ini dibangun di atas **fondasi pemodelan inti yang sangat solid (Grade A)**. Formulasi target intensitas, validasi bebas bocor, dan integrasi fitur riwayat lokal adalah contoh penerapan data science terapan yang sangat matang.

Kelemahan proyek ini tidak terletak pada model prediksi emisi utamanya, melainkan pada **fitur-fitur sampingannya**:
1. Atribusi NNLS yang terdistorsi korelasi.
2. Klasterisasi K-Means yang memaksakan batas pada data yang menyatu.
3. Upaya meramal masa depan yang terlalu jauh hingga tahun 2030.

Dengan **mempertahankan fondasi inti HistGB WARM** dan **menyingkirkan/merevisi modul sampingan yang rapuh**, sistem ini akan naik kelas dari sekadar prototipe data science menjadi **standar emas instrumen pendukung kebijakan FOLU Net Sink 2030 di Indonesia**.
