# 🌲 BUKU PANDUAN LENGKAP & TEORI DASAR: SISTEM PREDIKSI EMISI KARBON HUTAN INDONESIA (LOBMA & STREAMLIT DASHBOARD)

> **Untuk Siapa Dokumen Ini Ditulis?**
> Dokumen ini dirancang khusus untuk siapa saja—bahkan jika Anda **sama sekali belum pernah belajar Machine Learning, kehutanan, ataupun kimia lingkungan**. Di sini, semua konsep dijelaskan dari nol secara fundamental menggunakan bahasa manusia sehari-hari, analogi sederhana, dan penjelasan langkah demi langkah.

---

## DAFTAR ISI

1. [BAB 1: Gambaran Besar Proyek (Mengapa Proyek Ini Ada?)](#bab-1-gambaran-besar-proyek)
2. [BAB 2: Kamus Lengkap Domain Kehutanan &amp; Emisi Karbon](#bab-2-kamus-lengkap-domain-kehutanan--emisi-karbon)
   - [2.1 Satuan-Satuan Misterius (ha, Mg C, Mg CO2e, Mt, Gt)](#21-satuan-satuan-misterius)
   - [2.2 Rahasia Angka Ajaib 44/12 (Kimia Sederhana Emisi)](#22-rahasia-angka-ajaib-4412)
   - [2.3 Stok Karbon, Densitas, dan Biomassa Atas Tanah](#23-stok-karbon-densitas-dan-biomassa-atas-tanah)
   - [2.4 Emisi Kotor, Serapan, dan Emisi Bersih (Net Flux)](#24-emisi-kotor-serapan-dan-emisi-bersih-net-flux)
   - [2.5 Hutan Primer vs Non-Primer](#25-hutan-primer-vs-non-primer)
   - [2.6 Karbon Tersembunyi &amp; Misteri Lahan Gambut](#26-karbon-tersembunyi--misteri-lahan-gambut)
   - [2.7 Mengenal 7 Pemicu Kehilangan Pohon (Drivers of Loss)](#27-mengenal-7-pemicu-kehilangan-pohon-drivers-of-loss)
3. [BAB 3: Memahami Dataset Mentah GFW (Global Forest Watch)](#bab-3-memahami-dataset-mentah-gfw)
   - [3.1 Struktur Data Panel (497 Kabupaten × 25 Tahun)](#31-struktur-data-panel)
   - [3.2 Audit Data &amp; Mengapa &#34;Left Join&#34; Sangat Penting](#32-audit-data--mengapa-left-join-sangat-penting)
4. [BAB 4: Kamus Machine Learning &amp; Metodologi Anti-Bocor](#bab-4-kamus-machine-learning--metodologi-anti-bocor)
   - [4.1 Apa itu Machine Learning? (Analogi Koki Belajar Resep)](#41-apa-itu-machine-learning)
   - [4.2 Bahaya Mematikan &#34;Kebocoran Data&#34; (Data Leakage)](#42-bahaya-mematikan-kebocoran-data-data-leakage)
   - [4.3 Cara Uji yang Jujur: Future Years &amp; GroupKFold](#43-cara-uji-yang-jujur-future-years--groupkfold)
   - [4.4 Formulasi Cerdas: Memprediksi Intensitas (Bukan Emisi Mentah)](#44-formulasi-cerdas-memprediksi-intensitas)
   - [4.5 Mode COLD vs Mode WARM](#45-mode-cold-vs-mode-warm)
   - [4.6 Algoritma yang Diadu: Ridge, Random Forest, &amp; HistGradientBoosting](#46-algoritma-yang-diadu)
   - [4.7 Kuantil Regresi: Mengapa Satu Angka Tebakan Saja Berbahaya?](#47-kuantil-regresi-p10-dan-p90)
5. [BAB 5: Cara Membaca Metrik Evaluasi Model](#bab-5-cara-membaca-metrik-evaluasi-model)
   - [5.1 R-Squared (R²): Nilai Rapor Model](#51-r-squared-r)
   - [5.2 MAE (Mean Absolute Error): Rata-rata Meleset Berapa?](#52-mae-mean-absolute-error)
   - [5.3 Bedah Hasil Evaluasi: Mengapa HistGB WARM Jadi Juara?](#53-bedah-hasil-evaluasi)
6. [BAB 6: Bedah 8 Teknik Data Mining Lanjutan](#bab-6-bedah-8-teknik-data-mining-lanjutan)
   - [6.1 Analisis Pareto: 17% Wilayah Menghasilkan 80% Masalah](#61-analisis-pareto)
   - [6.2 NNLS (Non-Negative Least Squares): Menghitung Dosa Tiap Pemicu](#62-nnls-non-negative-least-squares)
   - [6.3 Analisis Net Flux: Siapa Pahlawan (Sink) dan Siapa Beban (Source)?](#63-analisis-net-flux)
   - [6.4 Hotspot Karbon Tersembunyi: Mendeteksi Anomali Gambut](#64-hotspot-karbon-tersembunyi)
   - [6.5 SHAP: Membuka Kotak Hitam Model Tanpa Tertipu Fitur Agregat](#65-shap-teori-permainan-untuk-interpretasi)
   - [6.6 Klasterisasi K-Means &amp; PCA: 5 Tipologi Wilayah Indonesia](#66-klasterisasi-k-means--pca)
   - [6.7 Deteksi Anomali: Robust Z-Score &amp; Tragedi El Niño 2015-2016](#67-deteksi-anomali-robust-z-score)
   - [6.8 Proyeksi 5 Tahun (2026-2030) &amp; Backtesting Multi-Horizon](#68-proyeksi-5-tahun--backtesting)
7. [BAB 7: Bedah Arsitektur Deployment &amp; Dashboard Streamlit (`app.py`)](#bab-7-bedah-arsitektur-deployment--dashboard-streamlit)
   - [7.1 Filosofi Tanpa Latih Ulang (Artifact Decoupling)](#71-filosofi-tanpa-latih-ulang)
   - [7.2 Bedah Tab 1: Simulasi Emisi (What-If Engine) &amp; Analisis Sensitivitas](#72-bedah-tab-1-simulasi-emisi)
   - [7.3 Bedah Tab 2: Skenario Mitigasi &amp; Nilai Ekonomi Karbon (USD)](#73-bedah-tab-2-skenario-mitigasi)
   - [7.4 Bedah Tab 3: Tren Historis &amp; Rekomendasi Otomatis Kebijakan](#74-bedah-tab-3-tren-historis)
   - [7.5 Bedah Tab 4: Peta Geografis (Choropleth) &amp; Peringkat Kabupaten](#75-bedah-tab-4-peta-geografis)
   - [7.6 Bedah Tab 5: Kuadran Emisi Bersih &amp; Hotspot Gambut](#76-bedah-tab-5-kuadran-emisi-bersih)
   - [7.7 Bedah Tab 6: Peta Tipologi Wilayah (Posisi Bintang Kabupaten Anda)](#77-bedah-tab-6-peta-tipologi-wilayah)
   - [7.8 Bedah Tab 7: Grafik Proyeksi 2026-2030 &amp; Pita Ketidakpastian](#78-bedah-tab-7-grafik-proyeksi)
   - [7.9 Bedah Tab 8 &amp; 9: Transparansi, Audit, dan Batasan Saintifik](#79-bedah-tab-8--9-transparansi-dan-batasan)
8. [BAB 8: Kesimpulan &amp; Pedoman Pemakaian](#bab-8-kesimpulan--pedoman-pemakaian)

---

# BAB 1: GAMBARAN BESAR PROYEK

### Mengapa Proyek Ini Dibuat?

Indonesia memiliki salah satu hutan hujan tropis terluas di dunia. Hutan ini berfungsi seperti "AC raksasa" dan "penyimpan karbon alami" bumi. Ketika pohon ditebang atau dibakar, karbon yang tadinya tersimpan aman di dalam batang, daun, dan tanah akan lepas ke atmosfer dalam bentuk gas rumah kaca (**Karbon Dioksida / $\text{CO}_2$**), memicu pemanasan global dan krisis iklim.

Pemerintah Indonesia memiliki komitmen global bernama **Indonesia FOLU Net Sink 2030** (*Forestry and Other Land Uses Net Sink*). Artinya: pada tahun 2030, sektor kehutanan Indonesia ditargetkan menyerap lebih banyak gas rumah kaca daripada yang dilepaskannya.

### Tantangan di Lapangan

Indonesia sangat luas, terdiri dari **497 kabupaten/kota** dengan karakteristik yang berbeda drastis:

- Ada daerah yang hutannya hilang karena tambang (misal di Bangka Belitung atau Kalimantan Timur).
- Ada daerah yang terbakar hebat saat musim kemarau panjang (misal lahan gambut di Riau atau Pulang Pisau, Kalteng).
- Ada daerah yang didominasi perkebunan kelapa sawit skala besar, ladang berpindah skala kecil, atau pembalakan kayu.

Jika pengambil kebijakan hanya menebak-nebak tanpa data, kebijakan mitigasi akan salah sasaran. Oleh karena itu, notebook [lobma_rapi.ipynb](<file:///C:/Users/rakan/dev%20c/JECT/forest%20usb%202/lobma_rapi.ipynb>) dan aplikasi [app.py](<file:///C:/Users/rakan/dev%20c/JECT/forest%20usb%202/app.py>) dibangun untuk menjawab pertanyaan krusial:

1. *Berapa banyak emisi karbon yang dilepaskan di setiap kabupaten dari tahun 2001 hingga 2025?*
2. *Pemicu apa yang paling merusak di masing-masing daerah?*
3. *Jika bupati atau menteri ingin mengurangi penebangan sawit sebesar 30% tahun depan, berapa ton emisi karbon yang berhasil dicegah dan berapa nilai ekonominya dalam rupiah/dollar?*
4. *Berapa proyeksi emisi Indonesia hingga tahun 2030 jika pola saat ini terus berlanjut?*

---

# BAB 2: KAMUS LENGKAP DOMAIN KEHUTANAN & EMISI KARBON

Bagian ini membedah semua istilah teknis yang ada di dataset [IDN.xlsx](<file:///C:/Users/rakan/dev%20c/JECT/forest%20usb%202/IDN.xlsx>), notebook, dan aplikasi.

---

### 2.1 Satuan-Satuan Misterius

Di dalam notebook dan aplikasi, Anda akan sering melihat singkatan seperti `ha`, `Mg C`, `Mg CO2e`, `Mt CO2e`, dan `Gt CO2e`. Apa arti semua itu?

| Simbol / Satuan            | Kepanjangan                                 | Arti Bahasa Awam                                                                                                                                                                                                                                       | Analogi Sederhana                                                                                                                                          |
| -------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`ha`**           | Hektare (*Hectare*)                       | Ukuran luas tanah =$10.000\text{ m}^2$                                                                                                                                                                                                               | Sekitar**1,4 kali luas lapangan sepak bola standar internasional**.                                                                                  |
| **`Mg`**           | Megagram                                    | **1 Megagram = 1.000.000 gram = 1.000 kilogram = 1 Ton Metrik**.                                                                                                                                                                                 | Di dunia sains internasional, kata "ton" sering diganti menjadi "Mg" agar tidak tertukar dengan ton Amerika (*short ton*). Jadi: **1 Mg = 1 Ton**. |
| **`Mg C`**         | Megagram Carbon                             | **Ton Karbon Murni**. Bobot atom karbon murni padat yang terkunci di dalam batang, ranting, dan akar kayu pohon yang masih hidup.                                                                                                                | Seperti berat batubara atau arang murni yang ada di dalam kayu pohon sebelum terbakar.                                                                     |
| **`Mg CO2e`**      | Megagram$\text{CO}_2$ equivalent          | **Ton Gas Rumah Kaca Setara $\text{CO}_2$**. Ukuran berat gas pencemar yang membubung ke atmosfer. Huruf "e" (*equivalent*) artinya semua gas (termasuk metana $\text{CH}_4$) sudah dikonversi ke dampak pemanasan setara $\text{CO}_2$. | Asap knalpot raksasa. Jika 1 pohon dibakar habis, karbon padat di batang berubah menjadi gas tak kasat mata yang memenuhi langit.                          |
| **`Mt CO2e`**      | Megaton$\text{CO}_2\text{e}$              | **1 Juta Ton $\text{CO}_2\text{e}$** ($10^6\text{ ton}$).                                                                                                                                                                                    | Skala emisi tingkat provinsi atau nasional tahunan.                                                                                                        |
| **`Gt CO2e`**      | Gigaton$\text{CO}_2\text{e}$              | **1 Miliar Ton $\text{CO}_2\text{e}$** ($10^9\text{ ton}$ atau 1.000 Megaton).                                                                                                                                                               | Skala akumulasi emisi nasional selama puluhan tahun. Total emisi kotor Indonesia 2001-2025 adalah sekitar**23,6 Gt CO2e**.                           |
| **`Mg C / ha`**    | Megagram C per hektare                      | **Densitas (Kerapatan) Karbon**. Berapa ton karbon murni yang tersimpan dalam setiap 1 hektare hutan.                                                                                                                                            | Tebal tipisnya tabungan pohon di lahan tersebut.                                                                                                           |
| **`Mg CO2e / ha`** | Megagram$\text{CO}_2\text{e}$ per hektare | **Intensitas Emisi**. Berapa ton gas polusi yang dilepaskan ketika 1 hektare hutan di daerah itu musnah.                                                                                                                                         | Efek kerusakan per hektare yang ditebang.                                                                                                                  |

---

### 2.2 Rahasia Angka Ajaib 44/12

Di baris 83 [app.py](<file:///C:/Users/rakan/dev%20c/JECT/forest%20usb%202/app.py#L83>) dan berbagai sel notebook, terdapat rumus:

$$
\text{potential\_co2e} = \text{total\_loss\_ha} \times \text{densitas} \times \frac{44}{12}
$$

**Dari mana datangnya angka $44/12$ (atau sekitar $3{,}667$)?**Ini adalah hukum stoikiometri kimia dasar:

1. Kayu pohon tersusun dari atom Karbon ($\text{C}$). Berat atom relatif Karbon adalah **12**.
2. Ketika kayu ditebang dan membusuk atau terbakar, 1 atom Karbon ($\text{C}$) akan mengikat 2 atom Oksigen ($\text{O}$) dari udara bebas untuk membentuk gas Karbon Dioksida ($\text{CO}_2$).
3. Berat atom Oksigen adalah **16**. Maka berat molekul gas $\text{CO}_2$ adalah:
   $$
   \text{Berat } \text{CO}_2 = 12 + (16 \times 2) = 12 + 32 = 44
   $$
4. Artinya, **setiap 12 kilogram karbon padat yang terbakar akan mengikat 32 kilogram oksigen dari atmosfer dan menghasilkan 44 kilogram gas $\text{CO}_2$!**

$$
\text{Rasio Konversi} = \frac{44}{12} \approx 3{,}667
$$

> 💡 **Intisari Awam:**
> Jika 1 ton kayu padat di hutan musnah, gas rumah kaca yang tercipta di langit bukan 1 ton, melainkan **3,67 ton gas $\text{CO}_2$** karena karbon tersebut mengikat oksigen dari udara!

---

### 2.3 Stok Karbon, Densitas, dan Biomassa Atas Tanah

Data dari Global Forest Watch membagi ekosistem hutan ke dalam beberapa konsep:

- **Tutupan Pohon (*Tree Cover Extent 2000*)**: Berapa hektare wilayah kabupaten yang ditumbuhi pohon dengan kanopi (kerapatan daun) di atas 30% pada tahun 2000 (dijadikan garis dasar / *baseline*).
- **Biomassa Atas Tanah (*Aboveground Biomass / AGB*)**: Segala bagian pohon yang hidup di atas permukaan tanah: batang utama, kulit kayu, cabang, ranting, dan dedaunan.
- **Stok Karbon (*Aboveground Carbon Stock*)**: Total berat seluruh karbon murni yang tersimpan di pepohonan tersebut (dalam satuan `Mg C`).
- **Densitas Karbon (*Carbon Density*)**: Rata-rata stok karbon dibagi luas hutan (`Mg C/ha`). Hutan rimba Kalimantan atau Papua bisa memiliki densitas tinggi (> 150 Mg C/ha), sedangkan semak belukar atau sabana memiliki densitas rendah (< 50 Mg C/ha).

---

### 2.4 Emisi Kotor, Serapan, dan Emisi Bersih (*Net Flux*)

Bayangkan hutan sebagai **rekening tabungan karbon** dunia:

```
[ REKENING TABUNGAN KARBON HUTAN ]
       ▲                             │
       │                             ▼
  SERAPAN (Removals)         EMISI KOTOR (Gross Emissions)
 Pohon tumbuh menyerap        Pohon ditebang / terbakar
 gas CO2 dari atmosfer       melepaskan gas CO2 ke udara
       │                             │
       └──────────────┬──────────────┘
                      ▼
               EMISI BERSIH (Net Flux)
           = Emisi Kotor - Serapan
```

1. **Emisi Kotor (*Gross Emissions*)**: Total gas rumah kaca yang dilepaskan ke udara akibat hilangnya pohon karena kebakaran, penebangan, tambang, dll. (Pengeluaran uang tabungan).
2. **Serapan Karbon (*Carbon Removals*)**: Pohon-pohon yang masih hidup dan bertumbuh menyerap gas $\text{CO}_2$ dari udara melalui proses fotosintesis dan menguncinya kembali menjadi kayu. (Pemasukan uang tabungan).
3. **Emisi Bersih (*Net Flux*)**: Selisih antara Emisi Kotor dikurangi Serapan.
   $$
   \text{Net Flux} = \text{Emisi Kotor} - \text{Serapan}
   $$

   - **Net Sink (Penyerap Bersih)**: Jika $\text{Net Flux} < 0$ (bernilai negatif). Artinya hutan menyerap gas lebih banyak daripada yang dilepasnya. Lingkungan **tertolong**. Di Indonesia, ada **308 kabupaten yang berstatus Net Sink**!
   - **Net Source (Sumber Emisi Bersih)**: Jika $\text{Net Flux} > 0$ (bernilai positif). Artinya penebangan dan kebakaran jauh lebih masif daripada kemampuan pohon yang tersisa untuk menyerapnya. Lingkungan **terbebani**.
   - **Secara Nasional**: Rata-rata tahunan Indonesia adalah emisi kotor **943,5 Jt ton $\text{CO}_2\text{e}$**, sedangkan serapannya **594,4 Jt ton $\text{CO}_2$**. Hasilnya, Indonesia masih menjadi **Net Source sebesar +349,1 Jt ton $\text{CO}_2\text{e}$ per tahun**.

---

### 2.5 Hutan Primer vs Non-Primer

- **Hutan Primer (*Primary Forest*)**: Hutan perawan tropis alami yang belum pernah dijamah atau ditebang manusia secara masif selama ratusan tahun. Pohon-pohonnya raksasa, ekosistemnya stabil, dan tanah di bawahnya kaya akan cadangan karbon tua.
- **Hutan Non-Primer (Hutan Sekunder / Lahan Regenerasi)**: Hutan yang pernah ditebang lalu tumbuh kembali, semak belukar lebat, atau tanaman perkebunan kayu monokultur.

> ⚠️ **Fakta Kunci dari Data:**
> Ketika 1 hektare **hutan primer** hilang di Indonesia, rata-rata emisi yang lepas mencapai **1.170 Mg $\text{CO}_2\text{e}$**. Tetapi jika yang hilang adalah **hutan non-primer**, rata-rata emisinya hanya **629 Mg $\text{CO}_2\text{e}$**. Artinya, menebang hutan primer dosanya **hampir 2 kali lipat lebih fatal** bagi atmosfer!

---

### 2.6 Karbon Tersembunyi & Misteri Lahan Gambut (*Peatland*)

Satelit optik di langit hanya bisa memotret apa yang terlihat di permukaan bumi (*Aboveground Biomass*). Satelit bisa menghitung: *"Oh, di atas tanah ada pohon seberat 100 ton per hektare"*.

Secara teori, kalau 100 ton biomassa itu musnah, harusnya emisi yang timbul maksimal:

$$
100 \times 3{,}667 = 367\text{ Mg }\text{CO}_2\text{e/ha}
$$

Namun di beberapa kabupaten di **Riau (seperti Pelalawan, Indragiri Hilir), Kalimantan Tengah (Pulang Pisau), dan Sumatera Selatan (Ogan Komering Ilir)**, data satelit mencatat emisi aktual mencapai **1.500 hingga 2.500 Mg $\text{CO}_2\text{e}$ per hektare yang hilang!**

**Kenapa bisa 5 sampai 7 kali lipat lebih besar dari pohonnya?**
Jawabannya adalah **Lahan Gambut (*Peatland*)**.
Gambut adalah tanah lunak basah yang terbentuk dari tumpukan sisa daun dan pohon mati purba selama ribuan tahun dengan ketebalan bisa mencapai 3 hingga 10 meter ke dalam tanah.
Ketika hutan di atasnya dibabat dan kanal air digali untuk perkebunan, lahan gambut menjadi kering. Ketika gambut kering terbakar, **tanah di bawahnya ikut terbakar membara seperti briket batubara selama berbulan-bulan di bawah tanah!**

Emisi raksasa yang berasal dari bawah tanah inilah yang disebut dalam proyek ini sebagai **"Karbon Tersembunyi" (*Hidden Carbon*)**. Model Machine Learning kita secara cerdik menangkap fenomena ini melalui riwayat intensitas masa lalu (`hist_intensity`).

---

### 2.7 Mengenal 7 Pemicu Kehilangan Pohon (*Drivers of Loss*)

Dataset GFW memecah penyebab hilangnya tutupan pohon di setiap kabupaten menjadi 7 pemicu (*drivers*):

```
                               7 PEMICU KEHILANGAN POHON
 ┌─────────────────────────────────────────┼────────────────────────────────────────┐
 │                                         │                                        │
[Aktivitas Manusia Terencana]     [Aktivitas Manusia Bertahap]             [Bencana & Alam]
- Permanent agriculture (Sawit)   - Logging (Pembalakan kayu)              - Wildfire (Kebakaran)
- Hard commodities (Tambang)      - Shifting cultivation (Ladang rotasi)   - Other natural disturbances
- Settlements & Infrastructure
```

1. **`Permanent agriculture` (Pertanian Menetap / Kelapa Sawit)**: Pembukaan hutan skala besar yang diubah secara permanen menjadi perkebunan komersial (terutama kelapa sawit dan karet). Merupakan penyumbang deforestasi terbesar di Indonesia.
2. **`Shifting cultivation` (Ladang Berpindah)**: Praktik tradisional masyarakat lokal yang membersihkan sepetak kecil lahan untuk bertani pangan selama 1–3 tahun, lalu meninggalkannya agar tanah beristirahat dan pohon tumbuh kembali (rotasi).
3. **`Hard commodities` (Komoditas Keras / Pertambangan)**: Pembukaan hutan untuk tambang terbuka (batu bara, nikel, bauksit, timah) atau tanaman keras industri tertentu.
4. **`Logging` (Pembalakan Kayu Hutan)**: Penebangan pohon berkayu besar komersial untuk industri pulp & kertas atau kayu gergajian, baik yang legal (HPH) maupun ilegal.
5. **`Wildfire` (Kebakaran Hutan dan Lahan / Karhutla)**: Kebakaran hutan tak terkendali, terutama sangat parah di tahun-tahun kemarau ekstrem (El Niño).
6. **`Settlements & Infrastructure` (Permukiman & Infrastruktur)**: Penebangan untuk perluasan kota, desa, jalan raya antarprovinsi, bendungan, transmisi listrik, atau fasilitas publik.
7. **`Other natural disturbances` (Gangguan Alami Lain)**: Bencana alam murni yang menumbangkan pohon, seperti tanah longsor, angin puting beliung (*cyclone*), atau serangan hama tanaman hutan skala luas.

---

# BAB 3: MEMAHAMI DATASET MENTAH GFW

### 3.1 Struktur Data Panel (497 Kabupaten × 25 Tahun)

Data yang digunakan berasal dari berkas [IDN.xlsx](<file:///C:/Users/rakan/dev%20c/JECT/forest%20usb%202/IDN.xlsx>) rilisan Global Forest Watch (versi satelit Hansen 20260427).Data ini berformat **Data Panel**. Artinya:

- Ada dimensi **Ruang (Spasial)**: **497 kabupaten/kota** di 34/38 provinsi Indonesia.
- Ada dimensi **Waktu (Temporal)**: **25 tahun** berturut-turut dari tahun 2001 sampai 2025.

Jika dikalikan:

$$
497\text{ kabupaten} \times 25\text{ tahun} = 12.425\text{ baris data panel}
$$

Setiap 1 baris merepresentasikan: *"Apa yang terjadi di Kabupaten X pada Tahun Y?"* (Berapa hektare sawit dibuka, berapa hektare kebakaran terjadi, berapa hektare hutan primer tumbang, dan berapa ton emisi kotor yang timbul).

---

### 3.2 Audit Data & Mengapa "Left Join" Sangat Penting

Di dalam [lobma_rapi.ipynb](<file:///C:/Users/rakan/dev%20c/JECT/forest%20usb%202/lobma_rapi.ipynb>) Bagian 1, dilakukan **Data Audit** sebelum pelatihan model.

Terdapat temuan kritis:

1. Sheet data karbon mencatat **497 kabupaten**, tetapi sheet data pemicu hanya mencatat **494 kabupaten** (ada 3 kabupaten/kota kepulauan kecil yang tidak ada catatan pemicu deforestasi).
2. Jika seorang data scientist pemula menggabungkan kedua tabel menggunakan teknik **`INNER JOIN`**, maka 3 kabupaten tersebut akan terhapus otomatis dari sistem, dan total emisi nasional akan hilang sebagian.
3. Notebook ini menggunakan **`LEFT JOIN`** dengan tabel karbon sebagai induknya. Kabupaten yang tidak memiliki catatan pemicu diisi nilai 0 (*missing driver handling*), sehingga ke-497 kabupaten tetap utuh terjaga 100%!
4. **Verifikasi Konsistensi ($0{,}998$)**: Jumlah 7 pemicu dijumlahkan dan dibandingkan dengan total kehilangan tutupan pohon (`tc_loss_ha`). Rasio kecocokannya adalah **0,9983** (99,83% cocok sempurna).

---

# BAB 4: KAMUS MACHINE LEARNING & METODOLOGI ANTI-BOCOR

Bagi orang awam, Machine Learning sering terdengar seperti sihir. Padahal intinya sangat sederhana.

---

### 4.1 Apa itu Machine Learning? (Analogi Koki Belajar Resep)

Bayangkan seorang anak magang di restoran masakan Padang:

- **Metode Tradisional (Pemrograman Biasa)**: Sang koki kepala menuliskan buku aturan kaku: *"Jika cabai 5 biji dan santan 200 ml, maka pedasnya level 3"*. Jika ada bahan baru, aturan kaku ini langsung rusak.
- **Machine Learning**: Sang koki kepala memberikan **12.425 piring masakan masa lalu** kepada si anak magang beserta catatan bumbu dan tingkat kepedasannya. Si anak magang mencicipi semuanya berulang-ulang sampai lidahnya sendiri mengenali pola tersembunyi: *"Oh, kalau santannya kental dan ada asam kandis, pedasnya cabai akan teredam sekian persen"*.

Di proyek ini:

- **Bahan Masakan ($X$ / Fitur)**: Luas bukaan sawit, luas tambang, kebakaran, densitas biomassa awal, sisa luas hutan, status hutan primer.
- **Hasil Masakan ($y$ / Target)**: Emisi gas $\text{CO}_2$ yang keluar.
- **Model Machine Learning**: Otak matematika yang mempelajari hubungan rumit antara hilangnya pohon dengan gas emisi yang dihasilkan.

---

### 4.2 Bahaya Mematikan "Kebocoran Data" (*Data Leakage*)

Ini adalah kesalahan paling umum dan paling fatal yang sering dilakukan praktisi AI pemula di data panel kehutanan.

> 🏫 **Analogi Menyontek Ujian Sekolah:**
> Bayangkan guru ingin menguji apakah murid paham Matematika. Guru memecah 100 soal latihan: 80 soal untuk belajar (data latih), 20 soal untuk ujian (data uji).
> Namun, guru memilih soal secara acak (*random split*). Akibatnya, soal nomor 5 di ujian sama persis dengan soal nomor 4 di kelas latihan, cuma diganti nama orangnya dari Budi menjadi Siti.
> Murid mendapat nilai 100! Tapi apakah murid itu pintar? Belum tentu! Dia hanya **menghafal contekan**. Saat menghadapi ujian nasional yang sesungguhnya (soal yang benar-benar baru), nilainya anjlok!

Dalam data panel kehutanan:

- Karakteristik Kabupaten Bogor tahun 2021 sangat mirip dengan Kabupaten Bogor tahun 2022.
- Jika data diacak menggunakan `train_test_split(random_state=42)` biasa, model akan belajar dari Bogor 2021 lalu diuji di Bogor 2022. Model akan tampak sangat jenius di atas kertas ($R^2 > 0{,}99$), padahal **model hanya menghafal karakteristik lokal Bogor!** Saat model dipakai untuk meramal tahun 2026 atau dipakai di kabupaten baru, prediksi model akan hancur lebur!

---

### 4.3 Cara Uji yang Jujur: Future Years & GroupKFold

Untuk mencegah kebocoran data (*data leakage*), notebook menerapkan 2 benteng pengujian yang ketat:

```
                  BENTENG PENGUJIAN ANTI-BOCOR
                
1. Ujian Lintas Waktu (Future Years Split)
   [ Data Latih: Tahun 2001 - 2021 ] ──► Diuji ke ──► [ Data Uji: Tahun 2022 - 2025 ]
   (Model dipaksa meramal masa depan yang belum pernah ia lihat sama sekali!)

2. Ujian Lintas Wilayah (GroupKFold per Kabupaten)
   [ 80% Kabupaten Indonesia ] ────────► Diuji ke ──► [ 20% Kabupaten Terisolasi ]
   (Model dipaksa meramal daerah baru tanpa pernah melihat kabupaten itu sebelumnya!)
```

Selain itu, fitur-fitur seperti *Net Flux* (karena dihitung dari target) **diharamkan masuk menjadi variabel input**, dan riwayat emisi (`hist_intensity`) hanya boleh dihitung dari tahun-tahun **sebelum tahun $t$**, tidak boleh memakai data masa depan!

---

### 4.4 Formulasi Cerdas: Memprediksi Intensitas (Bukan Emisi Mentah)

Mengapa model di proyek ini **tidak langsung memprediksi angka emisi kotor mentah ($y = \text{gross\_emissions}$)**?

Mari kita lihat kegagalan jika memprediksi emisi mentah: 

- Jika model dilatih menebak angka emisi mentah, algoritma pohon (*tree-based models*) memiliki kelemahan bawaan: **tidak bisa melakukan ekstrapolasi di luar angka maksimum yang pernah ia lihat**.
- Jika pengguna di dashboard memasukkan skenario ekstrim: *"Bagaimana jika kebakaran hutan tahun depan melahap 500.000 hektare?"* (padahal di masa lalu kabupaten itu paling parah hanya 50.000 hektare), model emisi mentah akan **jenuh (*saturate*)** dan memberikan tebakan datar yang salah besar!

**Solusi Cerdas:**
Model dipecah mengikuti rumus fisika alam:

$$
\text{Emisi} = \text{Luas Hilang (Hektare)} \times \text{Intensitas Emisi per Hektare}
$$

Model Machine Learning hanya bertugas menebak **Log-Intensitas**:

$$
\text{Target Model} = \log\left( \frac{\text{Emisi} + 1}{\text{Luas Hilang} + 1} \right)
$$

Lalu ketika memprediksi emisi asli ($\hat{E}$), hasilnya dikalikan kembali dengan luas hilang yang dimasukkan pengguna:

$$
\hat{E} = (\text{Luas Hilang} + 1) \times e^{\widehat{\text{intensitas}}} - 1
$$

> 🌟 **Kelebihan Formulasi Ini:**
> Prediksi otomatis berskala (*scale perfectly*) dengan luas lahan. Mau dimasukkan 10 hektare, 1.000 hektare, atau 1.000.000 hektare, hasil prediksi emisi tetap masuk akal dan proporsional secara fisika!

---

### 4.5 Mode COLD vs Mode WARM

Di dalam proyek ini ada istilah fitur **COLD** dan **WARM**:

- **Fitur COLD (Dingin)**: Hanya berisi data biofisik dasar (luas wilayah, stok pohon tahun 2000, 7 pemicu, luas hutan primer). Mode ini dipakai jika kita ingin memprediksi daerah baru yang belum punya rekam jejak emisi satelit sama sekali.
- **Fitur WARM (Hangat - Dipakai di Dashboard `app.py`)**: Mode COLD ditambah satu fitur sakti: `hist_intensity` (riwayat rata-rata intensitas emisi kabupaten tersebut di tahun-tahun sebelumnya). Karena kabupaten tersebut sudah punya rekam jejak, model tahu apakah daerah ini memiliki "karbon tersembunyi" (seperti gambut) atau tidak.

---

### 4.6 Algoritma yang Diadu

Di sel 17 notebook, 3 keluarga algoritma Machine Learning diadu secara adil:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   PERBANDINGAN 3 KELUARGA ALGORITMA                    │
├─────────────────────┬───────────────────┬──────────────────────────────┤
│ 1. Ridge Regression │ Model Garis Lurus │ Cepat, sederhana, tapi gagal │
│    (Linear Model)   │ (Linear)          │ menangkap pola rumit         │
├─────────────────────┼───────────────────┼──────────────────────────────┤
│ 2. Random Forest    │ Hutan Pohon       │ Akurat, tapi lambat dan file │
│                     │ Keputusan         │ ukurannya sangat membengkak  │
├─────────────────────┼───────────────────┼──────────────────────────────┤
│ 3. HistGradient-    │ Pohon Bertahap    │ SANGAT CEPAT, hemat memori,  │
│    Boosting (HGB)   │ Berbasis Binning  │ menangani missing values,    │
│    ★ PEMENANG ★     │ Histogram         │ dan paling akurat!           │
└─────────────────────┴───────────────────┴──────────────────────────────┘
```

1. **Ridge Regression (Baseline Pembanding)**: Algoritma garis lurus sederhana. Dipakai sebagai standar minimal: *"Kalau model canggih tidak bisa mengalahkan Ridge, berarti model canggih itu tidak ada gunanya."*
2. **Random Forest**: Membangun ratusan pohon keputusan independen lalu mengambil suara terbanyak (*voting* / rata-rata).
3. **HistGradientBoostingRegressor (Pilihan Final)**:
   Algoritma canggih modern dari scikit-learn (terinspirasi dari LightGBM). Data numerik dikelompokkan ke dalam 256 keranjang (*bins/histogram*). Algoritma membangun pohon demi pohon secara berurutan, di mana setiap pohon baru bertugas khusus **memperbaiki kesalahan (*residual error*) yang dibuat oleh pohon sebelumnya**. Algoritma ini terbukti paling tangguh dan presisi.

---

### 4.7 Kuantil Regresi (P10 dan P90): Mengapa Satu Angka Tebakan Saja Berbahaya?

Bayangkan seorang peramal cuaca berkata: *"Besok hujan tepat 12,4 milimeter"*. Tentu Anda ragu. Tapi jika ia berkata: *"Besok kemungkinan besar hujan sekitar 10 sampai 15 milimeter"*, Anda bisa bersiap membawa payung yang tepat.

Dalam kebijakan lingkungan hidup, memberikan **hanya 1 angka prediksi (misal: "Emisi akan sebesar 1,2 juta ton") adalah berbahaya**, karena alam memiliki ketidakpastian (cuaca kemarau, arah angin kebakaran, dll.).

Oleh karena itu, proyek ini melatih **3 Model HistGB Sekaligus**:

1. **Model Rata-Rata (`models['mean']`)**: Tebakan nilai tengah paling mungkin.
2. **Model Kuantil 10 (`models['q10']` / P10)**: Skenario optimis / batas bawah (hanya ada peluang 10% emisi jatuh di bawah angka ini).
3. **Model Kuantil 90 (`models['q90']` / P90)**: Skenario pesimis / batas atas (ada peluang 90% emisi berada di bawah angka ini).

Rentang antara P10 sampai P90 ini disebut **Interval Ketidakpastian 80% (*80% Uncertainty Interval*)**. Di aplikasi Streamlit, pengambil kebijakan bisa melihat rentang aman ini secara transparan!

---

# BAB 5: CARA MEMBACA METRIK EVALUASI MODEL

Ketika melihat tabel evaluasi di notebook atau di Tab Kualitas Model pada dashboard, bagaimana cara orang awam memahaminya?

```
+----------------------------------------+-------------------+--------------------+
| Model                                  | R² Uji (2022-25)  | MAE Uji (2022-25)  |
+----------------------------------------+-------------------+--------------------+
| Ridge Regression (Baseline)            | 0,917             | 343.294 Mg CO2e    |
| Random Forest (Target Mentah)          | 0,952             | 180.642 Mg CO2e    |
| HistGB (Target Mentah)                 | 0,939             | 184.962 Mg CO2e    |
| HistGB Log1p Target                    | 0,966             | 170.128 Mg CO2e    |
| HistGB Intensitas (COLD)               | 0,976             | 155.060 Mg CO2e    |
| HistGB Intensitas (WARM - Model Final) | 0,979 ★           | 137.609 Mg CO2e ★  |
+----------------------------------------+-------------------+--------------------+
```

---

### 5.1 R-Squared ($R^2$): Nilai Rapor Model

- **Apa itu?** $R^2$ mengukur **berapa persen variasi naik-turunnya emisi di dunia nyata yang berhasil ditebak dengan benar oleh model**.
- **Skala:** Dari 0 sampai 1 (atau 0% sampai 100%).
  - $R^2 = 0$: Model sama bodohnya dengan menebak angka rata-rata terus menerus.
  - $R^2 = 1{,}0$ (100%): Model sempurna tanpa kesalahan sedikit pun.
- **Hasil Model Final Kita ($0{,}979$)**: Artinya **97,9% variasi emisi di seluruh kabupaten Indonesia pada tahun 2022–2025 berhasil dijelaskan dengan sangat akurat oleh model!**

---

### 5.2 MAE (*Mean Absolute Error*): Rata-rata Meleset Berapa?

- **Apa itu?** $R^2$ terkadang bisa "menipu" karena angka kabupaten raksasa mendominasi statistik. Oleh karena itu, kita wajib membaca **MAE**.
- **Arti Bahasa Awam:** Jika model menebak emisi kabupaten, **rata-rata tebakannya meleset berapa ton dari kenyataan aslinya?**
- **Sifat:** Semakin **KECIL** angka MAE, semakin **HEBAT** modelnya.
- **Hasil Evaluasi:**
  - Model Baseline (Ridge) rata-rata meleset: **343.294 Mg $\text{CO}_2\text{e}$**.
  - Model Final Kita (HistGB WARM) rata-rata hanya meleset: **137.609 Mg $\text{CO}_2\text{e}$**.
  - **Kesimpulan:** Model final berhasil memangkas kesalahan tebak (*error*) hingga **lebih dari 55%** dibanding model standar!

---

# BAB 6: BEDAH 8 TEKNIK DATA MINING LANJUTAN

Notebook [lobma_rapi.ipynb](<file:///C:/Users/rakan/dev%20c/JECT/forest%20usb%202/lobma_rapi.ipynb>) tidak berhenti hanya pada melatih model regresi, melainkan menyajikan 8 analisis data mining mendalam untuk menjawab pertanyaan strategis.

---

### 6.1 Analisis Pareto: 17% Wilayah Menghasilkan 80% Masalah

> 🪙 **Prinsip Pareto (Hukum 80/20):**
> Di dunia bisnis, sering kali 80% pendapatan toko disumbang oleh 20% pelanggan setia.

Ketika seluruh 497 kabupaten diurutkan berdasarkan akumulasi emisi 2001–2025:

- **Ditemukan bahwa hanya 86 dari 497 kabupaten (sekitar 17%) yang menyumbang 80% total emisi kumulatif Indonesia!**
- 10 kabupaten teratas di antaranya: Kutai Timur, Pulang Pisau, Kotawaringin Timur, Ketapang, Paser, Kapuas, Ogan Komering Ilir, Rokan Hilir, Pelalawan, dan Bengkalis.

```
       KURVA KONSENTRASI PARETO EMISI KARBON INDONESIA
100% ┌───────────────────────────────────────────────------
     │                                        ..-''''
 80% │                             ..-''''  ◄── (Cukup tangani 86 kabupaten ini!)
     │                    ..-''''
     │             ..-''''
     │      ..-''''
  0% └──────┴─────────────────────────────────┴─────────────
     0     86 kabupaten                      497 kabupaten
```

> 🎯 **Implikasi Kebijakan:**
> Kementerian Lingkungan Hidup tidak perlu membagi rata anggaran patroli dan mitigasi ke 497 kabupaten secara seragam. **Fokuskan 80% anggaran dan intervensi ke 86 kabupaten prioritas ini**, maka sebagian besar masalah deforestasi nasional terkendali!

---

### 6.2 NNLS (*Non-Negative Least Squares*): Menghitung Dosa Tiap Pemicu

Jika kita menggunakan regresi linear biasa (*Ordinary Least Squares / OLS*) untuk mencari tahu *"Berapa ton emisi per 1 hektare bukaan sawit atau tambang?"*, matematika OLS sering kali menghasilkan koefisien **negatif** (misal: $-120\text{ Mg/ha}$ untuk tambang). Secara fisika kehutanan, **ini mustahil!** Tidak mungkin menebang pohon justru menyerap emisi.

Oleh karena itu, notebook menggunakan **NNLS (*Non-Negative Least Squares*)**, yaitu algoritma optimasi yang mengunci syarat mutlak: $\beta \ge 0$ (koefisien tidak boleh negatif).

**Temuan Lapangan:**

- Hutan Primer rata-rata melepas **1.170 Mg $\text{CO}_2\text{e/ha}$**.
- Hutan Non-Primer rata-rata melepas **629 Mg $\text{CO}_2\text{e/ha}$**.
- Pemicu dengan intensitas terbesar adalah kebakaran di lahan gambut dan pembukaan sawit skala besar di area bertutupan rapat.

---

### 6.3 Analisis Net Flux: Siapa Pahlawan (*Sink*) dan Siapa Beban (*Source*)?

Di sel 37 notebook dan Tab 5 aplikasi, data emisi kotor disandingkan dengan data serapan alami hutan.

```
                 PETA KUADRAN EMISI KOTOR VS SERAPAN
     ▲ Emisi Kotor (Gross)
     │
     │     KUADRAN "NET SOURCE" (Beban)
     │     Emisi kotor > Serapan alami
     │     (189 Kabupaten: Riau, Kalteng, Kalbar, dll.)
     │                     /
     │                   /   Garis Diagonal (Kotor = Serapan)
     │                 /
     │               /     KUADRAN "NET SINK" (Pahlawan)
     │             /       Serapan alami > Emisi kotor
     │           /         (308 Kabupaten: Papua, Maluku, Jawa, dll.)
     │         /
     └────────┴────────────────────────────────────────► Serapan (Removals)
```

**Temuan Menakjubkan:**

- Mayoritas kabupaten di Indonesia (**308 dari 497 kabupaten**) sebenarnya berstatus **Net Sink** (hutan mereka masih menyerap lebih banyak karbon daripada yang ditebang).
- Namun, karena emisi dari **189 kabupaten Net Source** sangat masif (terutama daerah perkebunan gambut dan kebakaran liar), secara nasional Indonesia defisit dan menjadi **Net Source sebesar +349 Jt Ton $\text{CO}_2\text{e}$ per tahun**.

---

### 6.4 Hotspot Karbon Tersembunyi: Mendeteksi Anomali Gambut

Bagian 4.4 notebook menghitung rasio:

$$
\text{Rasio Hotspot} = \frac{\text{Emisi Aktual per Hektare Hilang}}{\text{Ekspektasi Biomassa Atas Tanah}}
$$

Jika rasio mendekati $1{,}0$, artinya emisi murni berasal dari pohon di atas tanah yang roboh.
Tetapi jika rasio mencapai **$3{,}0$ hingga $7{,}0$**, artinya ada sumber karbon raksasa lain di luar pohon yang ikut terbakar.
Daftar 10 kabupaten teratas didominasi mutlak oleh **kabupaten berlahan gambut tebal di Riau, Jambi, Sumsel, dan Kalimantan**. Model data mining ini berhasil membuktikan keberadaan emisi gambut secara matematis tanpa memerlukan peta satelit radar bawah tanah!

---

### 6.5 SHAP: Teori Permainan untuk Interpretasi

Machine Learning sering dituduh sebagai "kotak hitam" (*black box*) yang tidak bisa dijelaskan. Untuk memecahkannya, notebook menggunakan **SHAP (*SHapley Additive exPlanations*)**, sebuah metode penerima hadiah Nobel ekonomi berbasis teori permainan (*Game Theory*).

SHAP menghitung kontribusi adil setiap variabel terhadap tebakan akhir.

- **Trik Cerdas di Notebook:** Jika kita memasukkan variabel agregat seperti `total_loss_ha` ke dalam model interpretasi, maka variabel itu akan "memonopoli" seluruh nilai kepentingan (*feature credit*), sehingga pemicu spesifik seperti sawit atau tambang terlihat tidak penting.
- Oleh karena itu, notebook melatih **Model Interpretasi Khusus** tanpa fitur agregat.
- **Hasil SHAP:** Pemicu paling berpengaruh secara nasional berturut-turut adalah:
  1. **`Permanent agriculture` (Sawit / Pertanian Menetap)** (Paling dominan secara konsisten).
  2. **`umd_tree_cover_extent_2000__ha` (Luas Hutan Awal)**.
  3. **`Shifting cultivation` (Ladang Berpindah)**.
  4. **`Settlements & Infrastructure`**.
  5. **`Logging` (Pembalakan)**.

---

### 6.6 Klasterisasi K-Means & PCA: 5 Tipologi Wilayah Indonesia

Apakah semua kabupaten harus diperlakukan sama? Jelas tidak.Notebook mengelompokkan 497 kabupaten menggunakan algoritma **K-Means Clustering** berdasarkan profil pemicu dan kondisi biofisiknya.

- **Penentuan Jumlah Kelompok ($K=5$):**Notebook menguji $K=3$ sampai $K=7$ menggunakan metrik **Silhouette Score** (derajat pemisahan antar-klaster). Nilai tertinggi dicapai pada **$K=5$ (Silhouette $= 0{,}257$)**.*Catatan Jujur:* Nilai 0,257 menunjukkan struktur klaster sedang (*moderate*), bukan pulau-pulau yang terpisah ekstrem. Ini wajar di alam nyata karena batas antar-ekosistem bersifat gradasi.
- **Visualisasi PCA (*Principal Component Analysis*):** Karena ada belasan fitur, dimensi disederhanakan menjadi 2 sumbu koordinat utama (PC1 dan PC2) agar bisa digambar dalam grafik 2 dimensi.

**5 Tipe Kabupaten Hasil Klasterisasi:**

1. **Klaster C1 (Agrikultur Rendah - 209 Kab)**: Kabupaten pedesaan dengan pembukaan sawit/kebun kecil dan emisi rendah.
2. **Klaster C2 (Agrikultur Masif - 142 Kab)**: Pusat perkebunan sawit raksasa nasional dengan emisi tinggi (100% anggotanya berstatus *Net Source*!).
3. **Klaster C3 (Infrastruktur / Urban - 47 Kab)**: Daerah perkotaan atau padat penduduk di mana kehilangan pohon didorong oleh jalan, perumahan, dan fasilitas publik.
4. **Klaster C4 (Agrikultur + Rawan Karhutla - 28 Kab)**: Daerah gambut rawan bencana kebakaran besar saat kemarau panjang.
5. **Klaster C5 (Hutan Lebat Tradisional - 71 Kab)**: Daerah pedalaman Kalimantan dan Papua dengan tutupan hutan primer tinggi, didominasi ladang berpindah dan konsesi logging.

---

### 6.7 Deteksi Anomali: Robust Z-Score & Tragedi El Niño 2015-2016

Kapan sebuah kabupaten mengalami "kejadian luar biasa" (lonjakan emisi ekstrem yang tidak wajar)?

Jika memakai rumus statistik biasa (Mean dan Standar Deviasi), rumus itu akan rusak jika datanya memiliki angka ekstrem. Oleh karena itu, notebook menggunakan **Robust Z-Score**:

$$
\text{Robust } Z = 0{,}6745 \times \frac{\log(\text{Emisi}) - \text{Median}}{\text{MAD}}
$$

Di mana **MAD (*Median Absolute Deviation*)** adalah jarak rata-rata data terhadap nilai tengahnya. Angka pengali $0{,}6745$ membuat skalanya setara dengan standar deviasi normal.

**Temuan Menakjubkan:**

- Ditemukan **20 kejadian anomali besar** di Indonesia.
- Dari 20 kejadian tersebut, **11 kejadian menumpuk pada tahun 2015 (6 kabupaten) dan 2016 (5 kabupaten)**.
- Seluruh anomali tersebut terbukti disebabkan oleh **Kebakaran Hutan Hebat (*Wildfire*) akibat fenomena iklim El Niño super kering tahun 2015**! Data mining berhasil menemukan bukti sejarah bencana nasional secara mandiri dari angka!

---

### 6.8 Proyeksi 5 Tahun (2026–2030) & Backtesting Multi-Horizon

Bagaimana cara meramal emisi tahun 2026, 2027, hingga 2030?

Ada 2 cara meramal deret waktu:

1. **Peramalan Berantai (*Recursive*)**: Tebak tahun 2026, lalu tebakan 2026 dipakai untuk menebak 2027, dan seterusnya.*Kelemahan:* Jika tebakan 2026 sedikit salah, kesalahan itu akan berlipat ganda (*snowball effect*) di tahun 2030.
2. **Peramalan Langsung (*Direct Multi-Horizon - Dipakai di Proyek Ini*)**:Melatih 5 model independen yang berbeda untuk masing-masing horizon waktu:
   - Model $h=1$ khusus meramal 1 tahun ke depan (2026).
   - Model $h=2$ khusus meramal 2 tahun ke depan (2027), dst.

**Uji Mundur (*Rolling-Origin Backtest*):**Sebelum model dipakai meramal 2026-2030, performanya diuji mundur seolah-olah kita berada di tahun **2012, 2015, 2018, dan 2020**. Model Machine Learning diadu dengan 2 metode sederhana (*Naive Baselines*):

- *Naive 1*: Menebak emisi sama persis dengan tahun lalu.
- *Naive 3*: Menebak emisi sama dengan rata-rata 3 tahun terakhir.

```
+-----------+-------------------+--------------------+----------------------+-----------------+
| Horizon   | HistGB-Lag (ML)   | Naive Tahun Lalu   | Naive Rata-rata 3 Th | Metode Terpilih |
+-----------+-------------------+--------------------+----------------------+-----------------+
| h=1 (1 th)| 1.061.671 Mg ★    | 1.127.628 Mg       | 1.126.261 Mg         | HistGB-Lag (ML) |
| h=2 (2 th)|   729.075 Mg ★    |   919.966 Mg       |   948.756 Mg         | HistGB-Lag (ML) |
| h=3 (3 th)| 1.027.841 Mg ★    | 1.151.516 Mg       | 1.143.748 Mg         | HistGB-Lag (ML) |
| h=4 (4 th)| 1.058.558 Mg ★    | 1.167.083 Mg       | 1.297.061 Mg         | HistGB-Lag (ML) |
| h=5 (5 th)| 1.072.517 Mg      | 1.211.957 Mg       | 1.057.873 Mg ★       | Naive 3 Tahun   |
+-----------+-------------------+--------------------+----------------------+-----------------+
```

> 💡 **Pelajaran Kejujuran Ilmiah yang Luar Biasa:**
> Pada horizon 1 sampai 4 tahun, model Machine Learning mengalahkan tebakan naif dengan selisih yang signifikan. Namun pada horizon 5 tahun ($h=5$), ketidakpastian alamiah membuat **rata-rata naif 3 tahun ternyata sedikit lebih stabil daripada Machine Learning!**
> Sistem di proyek ini dengan jujur memilih metode terbaik di masing-masing horizon, bukan memaksakan Machine Learning di mana ia tidak unggul!

---

# BAB 7: BEDAH ARSITEKTUR DEPLOYMENT & DASHBOARD STREAMLIT (`app.py`)

Aplikasi [app.py](<file:///C:/Users/rakan/dev%20c/JECT/forest%20usb%202/app.py>) adalah antarmuka visual berbasis web (*Streamlit*) yang mengubah model matematika rumit menjadi **Sistem Pendukung Keputusan (*Decision Support System*)** yang ramah pengguna.

---

### 7.1 Filosofi Tanpa Latih Ulang (*Artifact Decoupling*)

Jika aplikasi web harus melatih ulang model Machine Learning setiap kali dibuka oleh pengguna, web akan memakan waktu 10 menit untuk memuat dan server akan langsung *crash*.

Oleh karena itu, arsitektur sistem ini memisahkan secara tegas antara **Pabrik (Notebook)** dan **Toko (Aplikasi)**:

- [lobma_rapi.ipynb](<file:///C:/Users/rakan/dev%20c/JECT/forest%20usb%202/lobma_rapi.ipynb>) bertindak sebagai pabrik yang memproses data mentah, melatih model, dan mengekspor "artefak jadi":
  1. `model_emission.joblib`: Berkas biner model regresi (Model Rata-Rata, P10, dan P90).
  2. `metadata_carbon.joblib`: Berkas biner berisi daftar fitur, status hutan akhir tahun 2025 tiap kabupaten, dan daftar wilayah.
  3. Berkas-berkas ringkasan CSV (`pareto.csv`, `net_table.csv`, `cluster_table.csv`, dll.).
- [app.py](<file:///C:/Users/rakan/dev%20c/JECT/forest%20usb%202/app.py>) hanya bertugas membaca berkas-berkas jadi ini menggunakan fungsi `@st.cache_resource` dan `@st.cache_data`. Hasilnya: **aplikasi berjalan instan hanya dalam hitungan milidetik!**

---

### 7.2 Bedah Tab 1: Simulasi Emisi (*What-If Engine*) & Analisis Sensitivitas

Tab ini adalah kalkulator interaktif utama:

1. **Input Pengguna**: Di sidebar, pengguna memilih Provinsi dan Kabupaten. Di layar utama, pengguna memasukkan rencana bukaan lahan untuk 7 pemicu (dalam hektare) serta porsi hutan primer.
2. **Kalkulasi di Balik Layar (`build_row`)**:Fungsi [build_row()](<file:///C:/Users/rakan/dev%20c/JECT/forest%20usb%202/app.py#L72-L98>) menyusun 1 baris data baru secara real-time:
   - Menghitung `total_loss_ha` = penjumlahan 7 pemicu.
   - Menghitung rasio pertanian, rasio kebakaran, rasio kehilangan tutupan.
   - Menghitung `potential_co2e` fisik ($L \times \text{densitas} \times 44/12$).
   - Mengambil status sisa hutan kabupaten dari `metadata_carbon.joblib`.
3. **Prediksi (`predict`)**:Model menebak intensitas emisi, lalu dikonversi kembali menjadi ton emisi kotor beserta rentang P10 (batas bawah) dan P90 (batas atas).
4. **Grafik Sensitivitas Marginal (+10%)**:
   Sistem secara otomatis menguji: *"Jika salah satu pemicu naik 10%, pemicu mana yang paling mendongkrak emisi di kabupaten ini?"* Ini memberi tahu bupati pemicu mana yang paling berbahaya untuk disentuh.

---

### 7.3 Bedah Tab 2: Skenario Mitigasi & Nilai Ekonomi Karbon (USD)

Tab ini membantu perencana anggaran menghitung manfaat finansial dari pelestarian hutan:

1. **Kondisi Dasar (*Baseline*)**: Menghitung rata-rata bukaan lahan 5 tahun terakhir di kabupaten tersebut.
2. **Slider Pengurangan (%)**: Pengguna bisa menggeser slider: *"Bagaimana jika kebakaran berhasil ditekan 50% dan izin sawit ditekan 20%?"*
3. **Emisi yang Berhasil Dicegah (*Avoided Emissions*)**:
   $$
   \text{Emisi Dicegah} = \text{Emisi Baseline} - \text{Emisi Skenario Baru}
   $$
4. **Potensi Kredit Karbon (*Carbon Pricing*)**:
   Pengguna memasukkan asumsi harga pasar karbon dunia (misal $5 per ton $\text{CO}_2\text{e}$). Sistem langsung menampilkan valuasi uang:
   $$
   \text{Potensi Pendapatan} = \text{Emisi Dicegah (Ton)} \times \text{Harga Karbon (USD)}
   $$

   *Contoh:* Jika sebuah kabupaten di Kalteng berhasil mencegah 2.000.000 ton emisi, daerah tersebut berpotensi meraih dana insentif iklim sebesar **US$ 10.000.000 (sekitar Rp 150 Miliar)**!

---

### 7.4 Bedah Tab 3: Tren Historis & Rekomendasi Otomatis Kebijakan

Tab ini menampilkan grafik garis emisi tahun 2001–2025 dan grafik batang komposisi 7 pemicu di kabupaten terpilih.

**Sistem Rekomendasi Otomatis Berbasis Data:**Aplikasi mendeteksi pemicu terbesar di kabupaten tersebut selama 25 tahun terakhir, lalu memberikan rekomendasi kebijakan spesifik:

- Jika dominan **Sawit / Pertanian**: *"Perketat izin pembukaan lahan baru & dorong intensifikasi lahan eksisting (peningkatan produktivitas tanpa perluasan kebun)."*
- Jika dominan **Kebakaran**: *"Prioritaskan pencegahan karhutla: patroli terpadu, pembasahan kembali lahan gambut (rewetting), dan pembangunan sekat kanal."*
- Jika dominan **Ladang Berpindah**: *"Perkuat pendampingan petani lokal, bantuan pupuk, penerapan agroforestri, dan tata kelola ladang rotasi adat."*
- Jika dominan **Tambang**: *"Wajibkan reklamasi lubang tambang pasca-operasi dan moratorium bukaan tambang di area berhutan primer tinggi."*

---

### 7.5 Bedah Tab 4: Peta Geografis (*Choropleth*) & Peringkat Kabupaten

- **Peta Interaktif Indonesia**: Menggunakan berkas [indonesia_provinces.geojson](<file:///C:/Users/rakan/dev%20c/JECT/forest%20usb%202/indonesia_provinces.geojson>). Setiap provinsi diwarnai gradasi dari kuning muda (emisi rendah) hingga merah pekat (emisi raksasa).
- **Peringkat Top-N Kabupaten**: Pengguna dapat melihat 10 atau 15 kabupaten dengan emisi terbesar dalam rentang waktu yang bisa diatur (1 hingga 10 tahun terakhir).
- **Kurva Pareto**: Menampilkan secara visual kurva akumulasi 86 kabupaten penyumbang 80% emisi nasional.

---

### 7.6 Bedah Tab 5: Kuadran Emisi Bersih & Hotspot Gambut

- **Ringkasan Nasional**: Menampilkan metrik besar bahwa nasional masih defisit +349 Jt Ton $\text{CO}_2\text{e}$ per tahun.
- **Diagram Sebar Kuadran Log-Log**: Titik-titik kabupaten diplot dengan sumbu X = Serapan dan sumbu Y = Emisi Kotor.
  - Titik **HIJAU** (di bawah garis diagonal) = Kabupaten Net Sink (Pahlawan iklim).
  - Titik **MERAH** (di atas garis diagonal) = Kabupaten Net Source (Sumber emisi).
- **Tabel Hotspot Karbon Tersembunyi**: Menampilkan 15 kabupaten dengan emisi per hektare paling ekstrem melampaui biomassa atas tanah (indikator kuat lahan gambut).

---

### 7.7 Bedah Tab 6: Peta Tipologi Wilayah (Posisi Bintang Kabupaten Anda)

- **Diagram Tebaran PCA**: Seluruh 497 kabupaten ditampilkan dalam peta 2 dimensi berdasarkan kemiripan karakternya.
- **Tanda Bintang Hitam Spesial**: Kabupaten yang sedang dipilih pengguna di sidebar akan ditandai dengan **bintang hitam besar** di dalam grafik klaster. Pengguna langsung tahu: *"Kabupaten saya termasuk tipe C2 (Agrikultur Tinggi) bersama kabupaten tetangga mana saja?"*
- **Heatmap Karakteristik**: Menampilkan tabel matriks warna hijau yang menunjukkan rata-rata porsi pemicu untuk masing-masing tipe klaster.

---

### 7.8 Bedah Tab 7: Grafik Proyeksi 2026–2030 & Pita Ketidakpastian

- Menampilkan grafik tren masa lalu (garis biru solid) disambung dengan garis proyeksi masa depan 2026–2030 (garis putus-putus oranye).
- Khusus di tingkat kabupaten, proyeksi dilengkapi dengan **Pita Arsiran Oranye (Interval 80%)** yang menunjukkan batas optimis (P10) dan pesimis (P90).
- Tabel hasil *backtesting* disajikan transparan untuk membuktikan akurasi peramalan.

---

### 7.9 Bedah Tab 8 & 9: Transparansi, Audit, dan Batasan Saintifik

Sains data yang baik tidak boleh menyembunyikan kelemahan. Tab Kualitas menyajikan:

1. **Tabel Metrik Perbandingan Semua Model**.
2. **Audit Data Mentah JSON** (`data_audit.json`).
3. **Pernyataan Batasan Etis & Saintifik**:
   - Model ini bersifat **Korelatif, bukan Kausal Murni**: Jika model menunjukkan sawit berhubungan dengan emisi tinggi, ini adalah hubungan statistik historis, bukan pembuktian hukum sebab-akibat laboratorium.
   - Variabel gambut belum dimasukkan secara eksplisit sebagai peta spasial (baru ditangkap via proksi `hist_intensity`).
   - Model tidak boleh dipakai untuk mengekstrapolasi skenario di luar batas yang pernah terjadi di Indonesia.

---

# BAB 8: KESIMPULAN & PEDOMAN PEMAKAIAN

### Ringkasan Pencapaian Utama Proyek

1. **Akurasi Tinggi Tanpa Menyontek**: Model regresi intensitas (HistGB WARM) mencatatkan akurasi $R^2 = 0{,}979$ pada tahun-tahun uji masa depan (2022–2025) dan memangkas kesalahan tebak (MAE) sebesar 55% dibanding baseline.
2. **Terbukti Secara Spasial**: 17% kabupaten menyumbang 80% emisi nasional, dan 308 kabupaten sudah berstatus net sink.
3. **Terbukti Secara Ekologis**: Hutan primer terbukti melepaskan emisi 1,9 kali lipat lebih besar dibanding hutan non-primer.
4. **Siap Pakai untuk Publik**: Sistem dilengkapi antarmuka web interaktif yang siap dipakai untuk simulasi kebijakan iklim dan evaluasi FOLU Net Sink 2030.

---

### Cara Menjalankan Aplikasi di Komputer Anda

1. Pastikan seluruh sel di notebook [lobma_rapi.ipynb](<file:///C:/Users/rakan/dev%20c/JECT/forest%20usb%202/lobma_rapi.ipynb>) sudah dijalankan sekali (*Run All*) hingga seluruh artefak (`.joblib`, `.json`, `.csv`) tercipta di folder kerja.
2. Buka terminal / PowerShell di direktori proyek ini:
   ```bash
   cd "C:\Users\rakan\dev c\JECT\forest usb 2"
   ```
3. Jalankan aplikasi Streamlit:
   ```bash
   streamlit run app.py
   ```
4. Buka peramban (*browser*) Anda di alamat: `http://localhost:8501`.

---

*Dokumen ini disusun sebagai panduan referensi teoritis komprehensif proyek Analisis dan Pemodelan Emisi Karbon Hutan Indonesia (LOBMA & Streamlit DSS).*
