import json
import joblib
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Prediksi & Simulasi Emisi Karbon Hutan Indonesia", layout="wide")

DENS = "avg_gfw_aboveground_carbon_stocks_2000__Mg_C_ha-1"
EXTENT = "umd_tree_cover_extent_2000__ha"
STOCK = "gfw_aboveground_carbon_stocks_2000__Mg_C"
TARGET = "gross_emissions_Mg_CO2e"
LABELS = {
    "Permanent agriculture": "Pertanian Menetap / Sawit",
    "Shifting cultivation": "Ladang Berpindah",
    "Hard commodities": "Tambang / Komoditas Keras",
    "Settlements & Infrastructure": "Permukiman & Infrastruktur",
    "Logging": "Pembalakan",
    "Wildfire": "Kebakaran Hutan & Lahan",
    "Other natural disturbances": "Gangguan Alami Lain",
}

POLICY_REC = {
    "Permanent agriculture": "Perketat izin pembukaan lahan baru & dorong intensifikasi lahan eksisting.",
    "Shifting cultivation": "Perkuat pendampingan petani, agroforestri, dan tata kelola ladang rotasi.",
    "Wildfire": "Prioritaskan pencegahan karhutla: patroli, pembasahan gambut, sekat kanal.",
    "Logging": "Perketat pengawasan konsesi, terapkan reduced-impact logging.",
    "Hard commodities": "Wajibkan reklamasi dan batasi bukaan tambang di area bertutupan tinggi.",
    "Settlements & Infrastructure": "Arahkan tata ruang ke lahan terdegradasi, hindari koridor hutan.",
    "Other natural disturbances": "Perkuat pemantauan dini dan restorasi pascagangguan.",
}


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_artifacts():
    return (
        joblib.load(os.path.join(BASE_DIR, "model_emission.joblib")),
        joblib.load(os.path.join(BASE_DIR, "metadata_carbon.joblib")),
    )


@st.cache_data
def load_history():
    return pd.read_csv(os.path.join(BASE_DIR, "processed_forest_carbon.csv"))


@st.cache_data
def load_csv(path):
    try:
        full_path = path if os.path.isabs(path) else os.path.join(BASE_DIR, path)
        return pd.read_csv(full_path)
    except Exception:
        return None


@st.cache_data
def load_geojson(mtime=None):
    geo_path = os.path.join(BASE_DIR, "indonesia_provinces.geojson")
    if os.path.exists(geo_path):
        try:
            with open(geo_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


@st.cache_data
def load_kabupaten_geojson(mtime=None):
    geo_path = os.path.join(BASE_DIR, "indonesia_kabupaten.geojson")
    if os.path.exists(geo_path):
        try:
            with open(geo_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


try:
    models, meta = load_artifacts()
    df = load_history()
except Exception as e:
    st.error(f"Gagal memuat artefak model/data. Pastikan file artefak tersedia. Detail: {e}")
    st.stop()

FEATURES, DRIVERS, lookup = meta["feature_names"], meta["driver_list"], meta["lookup_wilayah"]
STATE = meta["state"].set_index(["subnational1", "subnational2"])
LAST_YEAR = meta["last_year"]


def build_row(drv: dict, primary: float, st_state) -> pd.DataFrame:
    """Membangun baris fitur input inferensi model berbasis kondisi sisa tutupan di akhir LAST_YEAR."""
    total = sum(drv.values())
    primary = min(primary, total)
    hi = st_state["hist_intensity"]
    row = {
        **drv,
        "total_loss_ha": total,
        "agri_loss_ratio": (drv["Permanent agriculture"] + drv["Shifting cultivation"]) / (total + 1e-6),
        "wildfire_ratio": drv["Wildfire"] / (total + 1e-6),
        "loss_extent_ratio": total / (st_state[EXTENT] + 1),
        "potential_co2e": total * st_state[DENS] * 44 / 12,
        "primary_loss_ha": primary,
        "primary_share": primary / (total + 1e-6),
        "remaining_extent_ha": st_state["remaining_extent_ha"],
        "remaining_frac": st_state["remaining_frac"],
        "loss_remaining_ratio": total / (st_state["remaining_extent_ha"] + 1),
        "remaining_carbon_Mg_C": st_state["remaining_carbon_Mg_C"],
        "cum_loss_prior_ha": st_state["cum_loss_prior_ha"],
        "area_ha": st_state["area_ha"],
        "forest_cover_frac": st_state["forest_cover_frac"],
        EXTENT: st_state[EXTENT],
        STOCK: st_state[STOCK],
        DENS: st_state[DENS],
        "year": LAST_YEAR + 1,
        "hist_intensity": hi,
        "hist_intensity_x_loss": hi * total if pd.notna(hi) else np.nan,
    }
    return pd.DataFrame([row])[FEATURES]


def predict(drv, primary, st_state):
    """Model memprediksi log-intensitas emisi; emisi kotor = (loss + 1) * exp(pred) - 1."""
    X = build_row(drv, primary, st_state)
    total = sum(drv.values())
    f = lambda m: float(max((total + 1) * np.exp(m.predict(X))[0] - 1, 0))
    mean, lo, hi = f(models["mean"]), f(models["q10"]), f(models["q90"])
    return mean, min(lo, mean), max(hi, mean)


# ------------------------------------------------------------------ FORMATTING HELPERS
def format_ha(val):
    if pd.isna(val) or val is None:
        return "-"
    val = float(val)
    if val >= 100e6:
        return f"{val / 1e6:.1f} Jt ha"
    elif val >= 1e6:
        return f"{val / 1e6:.2f} Jt ha"
    elif val >= 1e3:
        return f"{val / 1e3:.0f} Rb ha"
    return f"{val:,.0f} ha"


def format_carbon(val):
    if pd.isna(val) or val is None:
        return "-"
    val = float(val)
    if val >= 1e9:
        return f"{val / 1e9:.2f} Miliar Mg C"
    elif val >= 100e6:
        return f"{val / 1e6:.0f} Jt Mg C"
    elif val >= 1e6:
        return f"{val / 1e6:.1f} Jt Mg C"
    elif val >= 1e3:
        return f"{val / 1e3:.0f} Rb Mg C"
    return f"{val:,.0f} Mg C"


def format_em(val):
    if pd.isna(val) or val is None:
        return "-"
    val = float(val)
    if val >= 1e9:
        return f"{val / 1e9:.2f} Gt CO₂e"
    elif val >= 100e6:
        return f"{val / 1e6:.0f} Jt CO₂e"
    elif val >= 1e6:
        return f"{val / 1e6:.1f} Jt CO₂e"
    elif val >= 1e3:
        return f"{val / 1e3:.0f} Rb CO₂e"
    return f"{val:,.0f} Mg CO₂e"


# ------------------------------------------------------------------ SIDEBAR DINAMIS
st.sidebar.header("Pemilihan Wilayah")
level = st.sidebar.selectbox(
    "Tingkat Wilayah Analisis:",
    ["Nasional", "Provinsi", "Kabupaten/Kota"],
    index=0,
    help="Pilih cakupan analisis: Agregat Makro Nasional, Tingkat Provinsi, atau Khusus Kabupaten/Kota.",
)

# Agregat Acuan
_nat = df.groupby("year")[TARGET].sum()
_pareto = load_csv("pareto.csv")
_n80 = int((_pareto["cum_share"] < 0.8).sum() + 1) if _pareto is not None else 86
nt = load_csv("net_table.csv")
ct = load_csv("cluster_table.csv")
cs = load_csv("cluster_summary.csv")
fc = load_csv("forecast_2026_2030.csv")
bt = load_csv("forecast_backtest.csv")
sh = load_csv("shap_driver.csv")
di = load_csv("driver_intensity.csv")
an = load_csv("anomalies.csv")
hs = load_csv("intensity_hotspots.csv")
_prov_geo_file = os.path.join(BASE_DIR, "indonesia_provinces.geojson")
_kab_geo_file = os.path.join(BASE_DIR, "indonesia_kabupaten.geojson")
geo = load_geojson(os.path.getmtime(_prov_geo_file) if os.path.exists(_prov_geo_file) else None)
kab_geo = load_kabupaten_geojson(os.path.getmtime(_kab_geo_file) if os.path.exists(_kab_geo_file) else None)

prov = None
kab = None
st_k = None
extent, stock, dens = None, None, None

if level == "Nasional":
    cur_extent = lookup[EXTENT].sum()
    cur_stock = lookup[STOCK].sum()
    cur_dens = cur_stock / (cur_extent + 1e-6)
    cur_rem_ext = STATE["remaining_extent_ha"].sum()
    cur_rem_frac = cur_rem_ext / (cur_extent + 1e-6)

    active_df = df
    em_series = _nat
    em_latest = _nat.iloc[-1]
    peak_yr = _nat.idxmax()
    peak_val = _nat.max()
    cum_em = _nat.sum()
    sub_heading = "Nasional"

    page_title = "Emisi Karbon Hutan Indonesia: Tingkat Nasional"
    page_caption = (
        "Analisis data mining makro, dinamika pemicu, sebaran wilayah, dan proyeksi emisi sektor kehutanan Indonesia "
        "(data GFW 2001-2025) - mendukung evaluasi komitmen FOLU Net Sink 2030."
    )

elif level == "Provinsi":
    prov = st.sidebar.selectbox("Pilih Provinsi:", sorted(lookup["subnational1"].unique()))
    lp = lookup[lookup["subnational1"] == prov]
    sp = STATE.loc[prov]
    cur_extent = lp[EXTENT].sum()
    cur_stock = lp[STOCK].sum()
    cur_dens = cur_stock / (cur_extent + 1e-6)
    cur_rem_ext = sp["remaining_extent_ha"].sum()
    cur_rem_frac = cur_rem_ext / (cur_extent + 1e-6)

    active_df = df[df["subnational1"] == prov]
    em_series = active_df.groupby("year")[TARGET].sum()
    em_latest = em_series.iloc[-1]
    peak_yr = em_series.idxmax()
    peak_val = em_series.max()
    cum_em = em_series.sum()
    sub_heading = f"Provinsi {prov}"

    page_title = f"Emisi Karbon Hutan: Provinsi {prov}"
    page_caption = f"Analisis data historis, pemicu kehilangan tutupan, peringkat kabupaten, dan proyeksi emisi untuk Provinsi {prov} (data GFW 2001-2025)."

else:  # Kabupaten/Kota
    prov = st.sidebar.selectbox("Pilih Provinsi:", sorted(lookup["subnational1"].unique()))
    kabs = sorted(lookup.loc[lookup["subnational1"] == prov, "subnational2"].unique())
    kab = st.sidebar.selectbox("Pilih Kabupaten/Kota:", kabs)

    info = lookup[(lookup["subnational1"] == prov) & (lookup["subnational2"] == kab)].iloc[0]
    st_k = STATE.loc[(prov, kab)]
    extent, stock, dens = info[EXTENT], info[STOCK], info[DENS]
    cur_extent = extent
    cur_stock = stock
    cur_dens = dens
    cur_rem_ext = st_k["remaining_extent_ha"]
    cur_rem_frac = st_k["remaining_frac"]

    active_df = df[(df["subnational1"] == prov) & (df["subnational2"] == kab)].sort_values("year")
    em_series = active_df.set_index("year")[TARGET] if not active_df.empty else pd.Series()
    em_latest = active_df.iloc[-1][TARGET] if not active_df.empty else 0.0
    peak_yr = active_df.loc[active_df[TARGET].idxmax(), "year"] if not active_df.empty else "-"
    peak_val = active_df[TARGET].max() if not active_df.empty else 0.0
    cum_em = active_df[TARGET].sum() if not active_df.empty else 0.0
    sub_heading = f"{kab}, {prov}"

    page_title = f"Emisi Karbon Hutan: {kab}, {prov}"
    page_caption = f"Analisis data historis, status net flux, tipologi, serta simulator prediksi emisi dan mitigasi untuk {kab}."


# ==============================================================================
# HEADER & EXECUTIVE METRICS (FORMAT SERAGAM & VISUAL INTUITIF DI SEMUA TINGKAT)
# ==============================================================================


# ==============================================================================
# HEADER & EXECUTIVE METRICS
# ==============================================================================
st.title(page_title)
st.caption(page_caption)

# Delta emisi tahun terakhir vs tahun sebelumnya
em_delta_pct = None
if len(em_series) >= 2:
    prev_val = em_series.iloc[-2]
    if prev_val > 0:
        em_delta_pct = ((em_latest - prev_val) / prev_val) * 100

col_summary, _ = st.columns([3, 1])

with col_summary:
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            f"Emisi ({LAST_YEAR})",
            format_em(em_latest),
            delta=f"{em_delta_pct:+.1f}% vs {LAST_YEAR - 1}" if em_delta_pct is not None else None,
            delta_color="inverse",
            help=f"Total pelepasan emisi kotor pada tahun terakhir ({LAST_YEAR}). Delta menunjukkan persentase perubahan dibanding tahun {LAST_YEAR - 1}.",
        )

    with c2:
        st.metric(
            "Puncak Emisi",
            f"Tahun {peak_yr}",
            f"{format_em(peak_val)}",
            delta_color="off",
            help=f"Tahun terjadinya pelepasan emisi kotor tertinggi dalam catatan sejarah 2001-{LAST_YEAR} ({format_em(peak_val)}).",
        )

    with c3:
        st.metric(
            "Total Emisi Kumulatif",
            format_em(cum_em),
            f"2001–{LAST_YEAR} (25 thn)",
            delta_color="off",
            help=f"Akumulasi seluruh emisi kotor yang dilepaskan selama 25 tahun masa pencatatan (2001–{LAST_YEAR}).",
        )

    with c4:
        mean_em = em_series.mean() if not em_series.empty else 0.0
        st.metric(
            "Rata-rata Tahunan",
            f"{format_em(mean_em)}/thn",
            f"Rerata 2001–{LAST_YEAR}",
            delta_color="off",
            help=f"Rata-rata volume emisi kotor per tahun selama periode 2001-{LAST_YEAR}. Berfungsi sebagai tolok ukur (baseline) kinerja historis.",
        )

    # Visualisasi Dinamika Tutupan Hutan (Sisa vs Hilang) sejajar dengan 4 angka di atasnya
    lost_ext = max(cur_extent - cur_rem_ext, 0.0)
    rem_pct = (cur_rem_ext / (cur_extent + 1e-6)) * 100
    lost_pct = (lost_ext / (cur_extent + 1e-6)) * 100

    fig_bio = go.Figure()
    fig_bio.add_trace(
        go.Bar(
            y=["Status Tutupan"],
            x=[cur_rem_ext],
            name="Sisa Tutupan",
            orientation="h",
            marker=dict(color="#16a34a"),
            text=f"Sisa Terjaga {rem_pct:.1f}% ({format_ha(cur_rem_ext)})" if rem_pct > 15 else f"{rem_pct:.0f}%",
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="white", size=12),
            hovertemplate=f"<b>Sisa Tutupan Pohon ({LAST_YEAR})</b><br>Luas: {cur_rem_ext:,.0f} ha ({rem_pct:.1f}%)<extra></extra>",
        )
    )
    if lost_ext > 0:
        fig_bio.add_trace(
            go.Bar(
                y=["Status Tutupan"],
                x=[lost_ext],
                name="Tutupan Hilang",
                orientation="h",
                marker=dict(color="#d97706"),
                text=f"Hilang {lost_pct:.1f}% ({format_ha(lost_ext)})" if lost_pct > 15 else (f"{lost_pct:.0f}%" if lost_pct > 6 else ""),
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(color="white", size=12),
                hovertemplate=f"<b>Tutupan Hilang (2001-{LAST_YEAR})</b><br>Luas: {lost_ext:,.0f} ha ({lost_pct:.1f}%)<extra></extra>",
            )
        )

    fig_bio.update_layout(
        barmode="stack",
        height=36,
        margin=dict(l=0, r=0, t=2, b=2),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0, cur_extent * 1.001]),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    st.caption(f"Dinamika Tutupan Hutan ({sub_heading}): Sisa Terjaga vs Hilang (Baseline 2000: {format_ha(cur_extent)})", help="Membandingkan sisa tutupan pohon yang masih bertahan hingga 2025 terhadap baseline tutupan awal tahun 2000 berbasis citra satelit resolusi 30 meter.")
st.plotly_chart(fig_bio, width="stretch", config={"displayModeBar": False})

st.markdown("---")

# ==============================================================================
# TAB UTAMA: DASHBOARD (VISUALISASI) vs TOOLS (PREDIKSI & SIMULASI)
# ==============================================================================
main_dash, main_tools = st.tabs(["Dashboard (Visualisasi)", "Tools (Prediksi & Simulasi)"])

with main_dash:
    if level == "Nasional":
        tab_hist, tab_rank, tab_net, tab_typ, tab_fc, tab_anom = st.tabs(
            [
                "Tren & Pemicu",
                "Peta & Prioritas",
                "Emisi Bersih & Hotspot",
                "Tipologi Wilayah",
                "Proyeksi 2026-2030",
                "Deteksi Anomali",
            ]
        )
    else:
        tab_hist, tab_rank, tab_net, tab_typ, tab_fc = st.tabs(
            [
                "Tren & Pemicu",
                "Peta & Prioritas",
                "Emisi Bersih & Hotspot",
                "Tipologi Wilayah",
                "Proyeksi 2026-2030",
            ]
        )

# --------------------------------------------------------------------------
    # 1. TREN & PEMICU (TERMASUK INSIGHT KARAKTERISTIK PEMICU & PERAN HUTAN PRIMER)
    # --------------------------------------------------------------------------
    with tab_hist:
        sub_heading = "Nasional" if level == "Nasional" else (f"Provinsi {prov}" if level == "Provinsi" else f"{kab}, {prov}")
        st.subheader(f"Tren Historis & Dinamika Pemicu Emisi: {sub_heading}", help="Tren pelepasan emisi kotor tahunan (2001–2025) dalam satuan Juta Megagram (Jt Mg CO₂e = Juta Ton CO₂e) dan pergeseran komposisi pemicu kehilangan tutupan pohon.")

        col_h1, col_h2 = st.columns(2)
        with col_h1:
            df_em_plot = em_series.reset_index()
            df_em_plot.columns = ["year", TARGET]
            df_em_plot["Emisi_Jt"] = df_em_plot[TARGET] / 1e6
            fig_em = px.area(
                df_em_plot,
                x="year",
                y="Emisi_Jt",
                markers=True,
                title=f"Tren Emisi Kotor Tahunan (Jt Mg CO₂e) - {sub_heading}",
                labels={"year": "Tahun", "Emisi_Jt": "Emisi (Juta Mg CO₂e)"},
            )
            fig_em.update_traces(line_color="#ef4444", fillcolor="rgba(239, 68, 68, 0.2)")
            fig_em.update_layout(height=380)
            st.plotly_chart(fig_em, width="stretch")
            st.caption("Puncak lonjakan tajam merefleksikan krisis kebakaran hutan lahan gambut (terutama periode El Niño 2006 dan 2015).")

        with col_h2:
            drv_yearly = active_df.groupby("year")[DRIVERS].sum().reset_index()
            drv_melt = drv_yearly.melt(id_vars="year", var_name="Driver", value_name="ha_lost")
            drv_melt["Pemicu"] = drv_melt["Driver"].map(LABELS)
            fig_drv_yr = px.bar(
                drv_melt,
                x="year",
                y="ha_lost",
                color="Pemicu",
                title=f"Dinamika Luas Kehilangan Tutupan per Pemicu (ha) - {sub_heading}",
                labels={"year": "Tahun", "ha_lost": "Luas Hilang (ha)"},
            )
            fig_drv_yr.update_layout(height=380)
            st.plotly_chart(fig_drv_yr, width="stretch")
            st.caption("Komposisi tahunan pergeseran penyebab kehilangan tutupan pohon sepanjang 2001–2025.")

        # Pemicu dominan & rekomendasi
        drv_tot_sum = active_df[DRIVERS].sum().sort_values(ascending=False)
        if not drv_tot_sum.empty:
            top_drv = drv_tot_sum.index[0]
            top_pct = drv_tot_sum.iloc[0] / (drv_tot_sum.sum() + 1e-6) * 100
            st.markdown(
                f"**Pemicu dominan 2001–2025 di {sub_heading}:** {LABELS[top_drv]} "
                f"({top_pct:.0f}% dari total luas tutupan pohon yang hilang)."
            )
            st.success(f"Rekomendasi kebijakan berbasis data: {POLICY_REC.get(top_drv, 'Perketat pengawasan tutupan hutan.')}")

        # ----------------------------------------------------------------------
        # INSIGHT PEMICU & PERAN HUTAN PRIMER (DENGAN DONUT CHART)
        # ----------------------------------------------------------------------
        st.markdown("---")
        st.subheader(f"Karakteristik Pemicu & Peran Hutan Primer: {sub_heading}", help="Akumulasi luas kehilangan pohon per kategori pemicu serta perbandingan proporsi pembukaan hutan primer berkepadatan karbon tinggi vs hutan non-primer.")
        col_in1, col_in2 = st.columns(2)

        with col_in1:
            st.markdown(f"##### Akumulasi Luas Hutan Hilang per Pemicu - {sub_heading}")
            drv_sum = active_df[DRIVERS].sum().reset_index()
            drv_sum.columns = ["driver", "ha"]
            drv_sum["Pemicu"] = drv_sum["driver"].map(LABELS).fillna(drv_sum["driver"])
            fig_drv_loc = px.bar(
                drv_sum.sort_values("ha", ascending=True),
                x="ha",
                y="Pemicu",
                orientation="h",
                labels={"ha": "Total Luas Hilang (ha)", "Pemicu": ""},
                text_auto=",.0f",
            )
            fig_drv_loc.update_layout(height=420, margin=dict(l=20, r=20, t=10, b=40))
            st.plotly_chart(fig_drv_loc, width="stretch")
            st.caption(f"Total 25 tahun (2001–2025) luas tutupan pohon yang hilang berdasarkan aktivitas penyebab utama di {sub_heading}.")

        with col_in2:
            st.markdown(f"##### Proporsi Kehilangan Hutan Primer vs Non-Primer ({sub_heading})")
            p_loss = active_df["primary_loss_ha"].sum() if "primary_loss_ha" in active_df.columns else 0.0
            tot_loss = active_df["total_loss_ha"].sum() if "total_loss_ha" in active_df.columns else 0.0
            np_loss = max(tot_loss - p_loss, 0.0)
            p_pct = (p_loss / (tot_loss + 1e-6)) * 100
            np_pct = (np_loss / (tot_loss + 1e-6)) * 100

            lbl_prim = f"Hutan Primer ({p_loss/1e6:.1f} Jt ha)" if p_loss >= 1e6 else f"Hutan Primer ({p_loss:,.0f} ha)"
            lbl_nonprim = f"Hutan Non-Primer ({np_loss/1e6:.1f} Jt ha)" if np_loss >= 1e6 else f"Hutan Non-Primer ({np_loss:,.0f} ha)"

            df_prim = pd.DataFrame([
                {"Kategori": "Hutan Primer", "Luas": p_loss, "Label": lbl_prim},
                {"Kategori": "Hutan Non-Primer", "Luas": np_loss, "Label": lbl_nonprim},
            ])
            fig_donut_prim = px.pie(
                df_prim,
                names="Label",
                values="Luas",
                hole=0.45,
                color="Kategori",
                color_discrete_map={"Hutan Primer": "#e11d48", "Hutan Non-Primer": "#0284c7"},
            )
            fig_donut_prim.update_traces(
                textposition="inside",
                textinfo="percent+label",
                insidetextfont=dict(size=14, color="white"),
                hovertemplate="<b>%{label}</b><br>Luas Hilang: %{value:,.0f} ha<br>Porsi: %{percent}<extra></extra>",
            )
            fig_donut_prim.update_layout(
                height=420,
                legend=dict(orientation="h", y=-0.15, x=0.05),
                margin=dict(l=20, r=20, t=10, b=40),
            )
            st.plotly_chart(fig_donut_prim, width="stretch")
            st.caption(
                f"Dari total hutan hilang ({tot_loss:,.0f} ha), {p_pct:.1f}% adalah hutan alam primer berkepadatan karbon tinggi. "
                "Membuka 1 ha hutan primer melepaskan rata-rata ~1.170 Mg CO₂e (hampir dua kali lipat dibanding hutan non-primer ~629 Mg CO₂e)."
            )

        with st.expander(f"Buka / Lihat Data Mentah Tahunan ({sub_heading})", expanded=False):
            df_raw_show = active_df.groupby("year")[[TARGET, "total_loss_ha"] + DRIVERS].sum().reset_index()
            st.dataframe(df_raw_show, width="stretch")

    # --------------------------------------------------------------------------
    # 2. PETA & PRIORITAS
    # --------------------------------------------------------------------------
    with tab_rank:
        st.subheader(f"Peta & Prioritas Emisi: {sub_heading}", help="Peta tematik spasial dan pemeringkatan wilayah penyumbang emisi deforestasi tertinggi untuk menentukan fokus wilayah intervensi mitigasi.")
        yr = st.slider(
            "Periode pengamatan (tahun terakhir):",
            min_value=1,
            max_value=25,
            value=25,
            step=1,
            key="slider_yr_rank",
            help="Pilih rentang jendela waktu pengamatan ke belakang (misal 5, 10, atau 25 tahun terakhir) untuk melihat akumulasi emisi.",
        )
        sub_period = active_df[active_df["year"] > LAST_YEAR - yr]

        if level == "Nasional":
            prov_tot = sub_period.groupby("subnational1")[TARGET].sum().reset_index()
            prov_tot["state"] = prov_tot["subnational1"].replace({"Bangka Belitung": "Bangka-Belitung"})
            if geo is not None:
                figm = px.choropleth(
                    prov_tot,
                    geojson=geo,
                    locations="state",
                    featureidkey="properties.state",
                    color=TARGET,
                    color_continuous_scale="YlOrRd",
                    hover_name="subnational1",
                    labels={TARGET: "Emisi Kotor (Mg CO₂e)"},
                    title=f"Peta Sebaran Emisi Kotor Nasional per Provinsi ({yr} Tahun Terakhir)",
                )
                figm.update_geos(fitbounds="locations", visible=False, bgcolor="rgba(0,0,0,0)")
                figm.update_layout(margin=dict(l=0, r=0, t=35, b=0), height=460, paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(figm, width="stretch")
            else:
                st.plotly_chart(
                    px.bar(
                        prov_tot.sort_values(TARGET, ascending=True).tail(15),
                        x=TARGET,
                        y="subnational1",
                        orientation="h",
                        title=f"Top 15 Provinsi Sumber Emisi Terbesar ({yr} tahun terakhir)",
                        labels={TARGET: "Emisi Kotor (Mg CO₂e)", "subnational1": "Provinsi"},
                    ),
                    width="stretch",
                )

            rank = (
                sub_period.groupby(["subnational1", "subnational2"])[[TARGET, "total_loss_ha"]]
                .sum()
                .reset_index()
                .sort_values(TARGET, ascending=False)
            )
            rank["Label"] = rank["subnational2"] + " (" + rank["subnational1"] + ")"
            n = st.slider("Top-N kabupaten:", 5, 30, 15, key="slider_top_n_rank", help="Pilih jumlah kabupaten dengan emisi tertinggi yang ingin ditampilkan pada peringkat.")
            st.plotly_chart(
                px.bar(
                    rank.head(n).iloc[::-1],
                    x=TARGET,
                    y="Label",
                    orientation="h",
                    title=f"Top {n} kabupaten - emisi {yr} tahun terakhir",
                ),
                width="stretch",
            )

            if _pareto is not None:
                st.plotly_chart(
                    px.line(
                        _pareto,
                        x="rank",
                        y="cum_share",
                        title="Kurva Pareto: kontribusi kumulatif emisi 2001-2025",
                        labels={"rank": "Peringkat kabupaten", "cum_share": "Porsi kumulatif"},
                    ),
                    width="stretch",
                )
                st.caption(f"Hanya {_n80} dari {len(_pareto)} kabupaten menyumbang 80% emisi nasional.")

            with st.expander(f"Buka / Lihat Rincian Tabel Top {n} Kabupaten Sumber Emisi Terbesar", expanded=False):
                st.dataframe(
                    rank.head(n)[["subnational1", "subnational2", TARGET, "total_loss_ha"]].rename(
                        columns={
                            "subnational1": "Provinsi",
                            "subnational2": "Kabupaten/Kota",
                            TARGET: "Emisi Kotor (Mg CO₂e)",
                            "total_loss_ha": "Luas Hilang (ha)",
                        }
                    ).round(1),
                    width="stretch",
                )

        elif level == "Provinsi":
            rank_p = (
                sub_period.groupby("subnational2")[[TARGET, "total_loss_ha"]]
                .sum()
                .reset_index()
                .sort_values(TARGET, ascending=True)
            )
            rank_p["Porsi (%)"] = (rank_p[TARGET] / (rank_p[TARGET].sum() + 1e-6) * 100).round(1)
            rank_p["kab_id"] = prov + " - " + rank_p["subnational2"]

            col_map, col_chart = st.columns([1, 1])

            with col_map:
                geo_kabs_prov = (
                    {
                        "type": "FeatureCollection",
                        "features": [
                            f for f in kab_geo["features"]
                            if f.get("properties", {}).get("subnational1") == prov
                        ],
                    }
                    if kab_geo is not None
                    else None
                )

                if geo_kabs_prov and len(geo_kabs_prov["features"]) > 0:
                    figm_p = px.choropleth(
                        rank_p,
                        geojson=geo_kabs_prov,
                        locations="kab_id",
                        featureidkey="properties.kab_id",
                        color=TARGET,
                        color_continuous_scale="YlOrRd",
                        hover_name="subnational2",
                        hover_data={"kab_id": False, TARGET: ":,.0f", "Porsi (%)": ":.1f%"},
                        labels={TARGET: "Emisi (Mg CO₂e)"},
                        title=f"Gradasi Emisi per Kabupaten/Kota di {prov} ({yr} Thn Terakhir)",
                    )
                    figm_p.update_geos(fitbounds="locations", visible=False, bgcolor="rgba(0,0,0,0)")
                    figm_p.update_traces(marker_line_width=0.8, marker_line_color="#475569")
                    figm_p.update_layout(margin=dict(l=0, r=0, t=35, b=0), height=420, paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(figm_p, width="stretch")
                    st.caption(f"Peta gradasi sebaran emisi per kabupaten/kota di Provinsi {prov}.")
                else:
                    prov_state = "Bangka-Belitung" if prov == "Bangka Belitung" else prov
                    geo_single = (
                        {
                            "type": "FeatureCollection",
                            "features": [f for f in geo["features"] if f.get("properties", {}).get("state") == prov_state],
                        }
                        if geo is not None
                        else None
                    )
                    if geo_single and len(geo_single["features"]) > 0:
                        p_em = sub_period[TARGET].sum() if not sub_period.empty else 0.0
                        df_single = pd.DataFrame([{"state": prov_state, "subnational1": f"Provinsi {prov}", TARGET: p_em}])
                        figm_p = px.choropleth(
                            df_single,
                            geojson=geo_single,
                            locations="state",
                            featureidkey="properties.state",
                            color=TARGET,
                            color_continuous_scale="YlOrRd",
                            hover_name="subnational1",
                            labels={TARGET: "Emisi (Mg CO₂e)"},
                            title=f"Peta Wilayah: Provinsi {prov}",
                        )
                        figm_p.update_geos(fitbounds="locations", visible=False, bgcolor="rgba(0,0,0,0)")
                        figm_p.update_layout(margin=dict(l=0, r=0, t=35, b=0), height=420, paper_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(figm_p, width="stretch")
                        st.caption(f"Fokus batas administrasi geografis Provinsi {prov}.")
                    else:
                        st.info(f"Peta batas wilayah Provinsi {prov} belum tersedia.")

            with col_chart:
                st.plotly_chart(
                    px.bar(
                        rank_p,
                        x=TARGET,
                        y="subnational2",
                        orientation="h",
                        title=f"Peringkat Emisi per Kabupaten/Kota di {prov} ({yr} Thn Terakhir)",
                        labels={TARGET: "Emisi Kotor (Mg CO₂e)", "subnational2": "Kabupaten/Kota"},
                    ),
                    width="stretch",
                    height=420,
                )

            with st.expander(f"Buka / Lihat Rincian Tabel Emisi per Kabupaten di {prov}", expanded=False):
                st.dataframe(rank_p.sort_values(TARGET, ascending=False), width="stretch")

        else:  # Kabupaten
            prov_all = df[(df["subnational1"] == prov) & (df["year"] > LAST_YEAR - yr)]
            rank_k = (
                prov_all.groupby("subnational2")[TARGET]
                .sum()
                .reset_index()
                .sort_values(TARGET, ascending=True)
            )
            rank_k["Status"] = np.where(rank_k["subnational2"] == kab, f"{kab} (Terpilih)", "Kabupaten Lain")
            rank_k["kab_id"] = prov + " - " + rank_k["subnational2"]

            col_kmap, col_kchart = st.columns([1, 1])

            with col_kmap:
                geo_kabs_prov = (
                    {
                        "type": "FeatureCollection",
                        "features": [
                            f for f in kab_geo["features"]
                            if f.get("properties", {}).get("subnational1") == prov
                        ],
                    }
                    if kab_geo is not None
                    else None
                )

                if geo_kabs_prov and len(geo_kabs_prov["features"]) > 0:
                    figm_k = px.choropleth(
                        rank_k,
                        geojson=geo_kabs_prov,
                        locations="kab_id",
                        featureidkey="properties.kab_id",
                        color=TARGET,
                        color_continuous_scale="YlOrRd",
                        hover_name="subnational2",
                        hover_data={"kab_id": False, TARGET: ":,.0f", "Status": True},
                        labels={TARGET: "Emisi (Mg CO₂e)"},
                        title=f"Peta Gradasi Kabupaten di {prov} (Fokus: {kab})",
                    )
                    figm_k.update_geos(fitbounds="locations", visible=False, bgcolor="rgba(0,0,0,0)")
                    figm_k.update_traces(marker_line_width=0.8, marker_line_color="#475569")
                    figm_k.update_layout(margin=dict(l=0, r=0, t=35, b=0), height=420, paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(figm_k, width="stretch")
                    st.caption(f"Peta gradasi emisi kabupaten di Provinsi {prov}. Fokus analisis wilayah: **{kab}**.")
                else:
                    prov_state = "Bangka-Belitung" if prov == "Bangka Belitung" else prov
                    geo_single = (
                        {
                            "type": "FeatureCollection",
                            "features": [f for f in geo["features"] if f.get("properties", {}).get("state") == prov_state],
                        }
                        if geo is not None
                        else None
                    )
                    if geo_single and len(geo_single["features"]) > 0:
                        kab_em = sub_period[TARGET].sum() if not sub_period.empty else 0.0
                        df_kab_map = pd.DataFrame([{"state": prov_state, "subnational1": f"{kab}, {prov}", TARGET: kab_em}])
                        figm_k = px.choropleth(
                            df_kab_map,
                            geojson=geo_single,
                            locations="state",
                            featureidkey="properties.state",
                            color=TARGET,
                            color_continuous_scale="YlOrRd",
                            hover_name="subnational1",
                            labels={TARGET: f"Emisi {kab} (Mg CO₂e)"},
                            title=f"Peta Wilayah: {kab}, Provinsi {prov}",
                        )
                        figm_k.update_geos(fitbounds="locations", visible=False, bgcolor="rgba(0,0,0,0)")
                        figm_k.update_layout(margin=dict(l=0, r=0, t=35, b=0), height=420, paper_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(figm_k, width="stretch")
                        st.caption(f"Fokus geografis Provinsi {prov} untuk konteks wilayah {kab}.")
                    else:
                        st.info(f"Peta batas wilayah Provinsi {prov} belum tersedia.")

            with col_kchart:
                st.plotly_chart(
                    px.bar(
                        rank_k,
                        x=TARGET,
                        y="subnational2",
                        color="Status",
                        color_discrete_map={f"{kab} (Terpilih)": "#d97706", "Kabupaten Lain": "#64748b"},
                        orientation="h",
                        title=f"Posisi Emisi {kab} vs Kabupaten Lain di {prov} ({yr} Thn Terakhir)",
                        labels={TARGET: "Emisi Kotor (Mg CO₂e)", "subnational2": ""},
                    ),
                    width="stretch",
                    height=420,
                )

            with st.expander(f"Buka / Lihat Rincian Tabel Posisi Emisi Kabupaten di {prov}", expanded=False):
                st.dataframe(
                    rank_k.sort_values(TARGET, ascending=False)[["subnational2", TARGET, "Status"]].rename(
                        columns={
                            "subnational2": "Kabupaten/Kota",
                            TARGET: "Emisi Kotor (Mg CO₂e)",
                        }
                    ).round(1),
                    width="stretch",
                )

    # --------------------------------------------------------------------------
    # 3. EMISI BERSIH & HOTSPOT
    # --------------------------------------------------------------------------
    with tab_net:
        st.subheader("Emisi Bersih (Net Flux) = Emisi Kotor − Serapan", help="Emisi Bersih (Net Flux) adalah selisih antara emisi kotor dikurangi serapan alami pertumbuhan hutan. Nilai negatif (-) = Penyerap Karbon (Net Sink), nilai positif (+) = Sumber Emisi (Net Source).")
        if nt is None:
            st.warning("File net_table.csv belum tersedia.")
        else:
            if level == "Nasional":
                nt_scope = nt
                sub_label = "Nasional"
            elif level == "Provinsi":
                nt_scope = nt[nt["subnational1"] == prov]
                sub_label = f"Provinsi {prov}"
            else:
                nt_scope = nt[(nt["subnational1"] == prov) & (nt["subnational2"] == kab)]
                sub_label = f"{kab}"

            g_ = nt_scope["gross_avg_Mg_CO2e_yr"].sum() / 1e6
            r_ = nt_scope["removals_avg_Mg_CO2_yr"].sum() / 1e6
            n_ = nt_scope["net_flux_avg_Mg_CO2e_yr"].sum() / 1e6
            n_sink = int((nt_scope["net_status"] == "Net sink").sum())

            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Emisi kotor (rata-rata/thn)", f"{g_:,.1f} Jt CO₂e", help="Total emisi kotor rata-rata per tahun akibat hilangnya tutupan pohon (2001–2025). 1 Mg = 1 Ton.")
            a2.metric("Serapan (rata-rata/thn)", f"{r_:,.1f} Jt CO₂e", help="Jumlah gas rumah kaca rata-rata per tahun yang diserap kembali oleh pertumbuhan vegetasi hutan.")
            a3.metric(
                "Net flux rata-rata",
                f"{n_:+,.1f} Jt CO₂e/thn",
                delta="Net Sink (Penyerap)" if n_ < 0 else "Net Source (Sumber Emisi)",
                delta_color="normal" if n_ < 0 else "inverse",
                help="Positif = sumber emisi bersih; Negatif = penyerap emisi bersih.",
            )
            a4.metric("Wilayah net sink", f"{n_sink} / {len(nt_scope)}", help="Jumlah kabupaten/wilayah dalam cakupan ini yang berstatus Net Sink.")

            st.caption(
                "Angka GFW adalah rata-rata tahunan 2001-2025. Negatif = penyerap karbon (net sink), positif = sumber emisi (net source)."
            )

            if level == "Nasional":
                pn = (
                    nt.groupby("subnational1")["net_flux_avg_Mg_CO2e_yr"]
                    .sum()
                    .reset_index()
                    .sort_values("net_flux_avg_Mg_CO2e_yr")
                )
                pn["status"] = np.where(pn["net_flux_avg_Mg_CO2e_yr"] < 0, "Net sink", "Net source")
                st.plotly_chart(
                    px.bar(
                        pn,
                        x="net_flux_avg_Mg_CO2e_yr",
                        y="subnational1",
                        color="status",
                        orientation="h",
                        color_discrete_map={"Net sink": "#2e7d32", "Net source": "#c62828"},
                        title="Net flux per provinsi (Mg CO₂e/thn; negatif = penyerap)",
                    ),
                    width="stretch",
                    height=700,
                )
                st.plotly_chart(
                    px.scatter(
                        nt,
                        x="removals_avg_Mg_CO2_yr",
                        y="gross_avg_Mg_CO2e_yr",
                        color="net_status",
                        hover_data=["subnational1", "subnational2"],
                        log_x=True,
                        log_y=True,
                        color_discrete_map={"Net sink": "#2e7d32", "Net source": "#c62828"},
                        title="Emisi kotor vs serapan per kabupaten di Indonesia (skala log)",
                    ),
                    width="stretch",
                )
            elif level == "Provinsi":
                st.plotly_chart(
                    px.bar(
                        nt_scope.sort_values("net_flux_avg_Mg_CO2e_yr"),
                        x="net_flux_avg_Mg_CO2e_yr",
                        y="subnational2",
                        color="net_status",
                        orientation="h",
                        color_discrete_map={"Net sink": "#2e7d32", "Net source": "#c62828"},
                        title=f"Net Flux per Kabupaten/Kota di {prov} (Mg CO₂e/thn)",
                    ),
                    width="stretch",
                    height=max(380, len(nt_scope) * 26),
                )
            else:
                prov_net = nt[nt["subnational1"] == prov].sort_values("net_flux_avg_Mg_CO2e_yr")
                if not prov_net.empty:
                    st.plotly_chart(
                        px.bar(
                            prov_net,
                            x="net_flux_avg_Mg_CO2e_yr",
                            y="subnational2",
                            color="net_status",
                            orientation="h",
                            color_discrete_map={"Net sink": "#2e7d32", "Net source": "#c62828"},
                            title=f"Posisi Net Flux {kab} dibanding Kabupaten Lain di {prov}",
                        ),
                        width="stretch",
                        height=max(350, len(prov_net) * 24),
                    )

            with st.expander(f"Buka / Lihat Rincian Tabel Emisi Bersih (Net Flux) - {sub_heading}", expanded=False):
                st.dataframe(
                    nt_scope.rename(
                        columns={
                            "subnational1": "Provinsi",
                            "subnational2": "Kabupaten/Kota",
                            "gross_avg_Mg_CO2e_yr": "Emisi Kotor (Mg CO₂e/thn)",
                            "removals_avg_Mg_CO2_yr": "Serapan (Mg CO₂/thn)",
                            "net_flux_avg_Mg_CO2e_yr": "Net Flux (Mg CO₂e/thn)",
                            "net_status": "Status",
                        }
                    ).round(1),
                    width="stretch",
                )

            st.markdown("---")
            st.subheader(f"Hotspot Karbon Tersembunyi: Emisi Melampaui Biomassa Pohon ({sub_heading})", help="Analisis rasio intensitas emisi per hektare dibandingkan biomassa tegakan pohon awal di atas tanah. Rasio > 1.0 mengindikasikan pelepasan karbon besar dari tanah gambut dalam yang terbakar.")
            if hs is not None:
                if level == "Nasional":
                    hs_show = hs.head(15).copy()
                elif level == "Provinsi":
                    hs_show = hs[hs["subnational1"] == prov].copy()
                    if hs_show.empty:
                        st.info(f"Tidak ada kabupaten di Provinsi {prov} yang terindikasi memiliki emisi karbon tersembunyi / gambut ekstrem.")
                else:
                    hs_show = hs[(hs["subnational1"] == prov) & (hs["subnational2"] == kab)].copy()
                    if hs_show.empty:
                        st.info(f"Wilayah {kab} tidak terindikasi memiliki emisi karbon tersembunyi / gambut ekstrem.")

                if not hs_show.empty:
                    hs_show["Label"] = hs_show["subnational2"] if level != "Nasional" else (hs_show["subnational2"] + ", " + hs_show["subnational1"])
                    hs_plot = hs_show.head(10).copy()

                    col_hs1, col_hs2 = st.columns(2)
                    with col_hs1:
                        fig_hs1 = px.bar(
                            hs_plot.sort_values("intensity_vs_aboveground_expectation", ascending=True),
                            x="intensity_vs_aboveground_expectation",
                            y="Label",
                            orientation="h",
                            color="intensity_vs_aboveground_expectation",
                            color_continuous_scale="YlOrRd",
                            title="Rasio Pelepasan Emisi vs Ekspektasi Pohon (x lipat)",
                            labels={"intensity_vs_aboveground_expectation": "Rasio vs Ekspektasi (x lipat)", "Label": ""},
                            text_auto=".1f",
                        )
                        fig_hs1.update_layout(height=380, coloraxis_showscale=False)
                        st.plotly_chart(fig_hs1, width="stretch")
                        st.caption("Rasio > 1.0 mengindikasikan emisi karbon yang dilepaskan jauh lebih besar dari kayu/pohon di atas tanah (berasal dari tanah gambut dalam yang terbakar).")

                    with col_hs2:
                        comp_melt = hs_plot.sort_values("Mg_CO2e_per_ha_lost", ascending=True).tail(8).melt(
                            id_vars=["Label"],
                            value_vars=["density_aboveground", "Mg_CO2e_per_ha_lost"],
                            var_name="Tipe",
                            value_name="Nilai",
                        )
                        comp_melt["Tipe"] = comp_melt["Tipe"].replace({
                            "density_aboveground": "Biomassa Pohon (Mg C/ha)",
                            "Mg_CO2e_per_ha_lost": "Emisi Terlepas (Mg CO₂e/ha)",
                        })
                        fig_hs2 = px.bar(
                            comp_melt,
                            x="Nilai",
                            y="Label",
                            color="Tipe",
                            barmode="group",
                            orientation="h",
                            title="Perbandingan Emisi Terlepas vs Cadangan Pohon Awal",
                            labels={"Nilai": "Nilai per Hektare", "Label": "", "Tipe": ""},
                            color_discrete_map={"Biomassa Pohon (Mg C/ha)": "#22c55e", "Emisi Terlepas (Mg CO₂e/ha)": "#ef4444"},
                        )
                        fig_hs2.update_layout(height=380)
                        st.plotly_chart(fig_hs2, width="stretch")
                        st.caption("Membandingkan cadangan awal pohon (hijau) dengan emisi riil yang lepas (merah) saat hutan dibuka.")

                    with st.expander("Buka / Lihat Rincian Tabel Data Hotspot Karbon Tersembunyi", expanded=False):
                        st.dataframe(
                            hs_show.rename(
                                columns={
                                    "subnational1": "Provinsi",
                                    "subnational2": "Kabupaten/Kota",
                                    "cum_L": "Total Luas Hilang (ha)",
                                    "Mg_CO2e_per_ha_lost": "Emisi (Mg CO₂e/ha)",
                                    "density_aboveground": "Biomassa Pohon (Mg C/ha)",
                                    "intensity_vs_aboveground_expectation": "Rasio vs Ekspektasi",
                                }
                            ).round(1),
                            width="stretch",
                        )
# --------------------------------------------------------------------------
    # 4. TIPOLOGI WILAYAH
    # --------------------------------------------------------------------------
    with tab_typ:
        st.subheader("Tipologi Wilayah Kabupaten: Karakteristik Biofisik & Pemicu Emisi", help="Pengelompokan 497 kabupaten/kota ke dalam 5 kelompok tipologi berdasarkan analisis PCA & Klasterisasi terhadap profil biomassa dan pemicu deforestasi.")
        if ct is None or cs is None:
            st.warning("File klaster belum tersedia.")
        else:
            CLUSTER_MAP = {
                "C1: Agrikultur/Sawit (emisi rendah)": "C1: Agrikultur Rendah (209 Kab)",
                "C2: Agrikultur/Sawit (emisi tinggi)": "C2: Agrikultur Masif (142 Kab)",
                "C3: Infrastruktur + Agrikultur/Sawit (emisi rendah)": "C3: Infrastruktur / Urban (47 Kab)",
                "C4: Agrikultur/Sawit + Kebakaran (emisi tinggi)": "C4: Agrikultur + Rawan Karhutla (28 Kab)",
                "C5: Ladang Berpindah + Agrikultur/Sawit + Pembalakan (emisi tinggi)": "C5: Hutan Lebat Tradisional (71 Kab)",
            }
            CLUSTER_DESC = {
                "C1: Agrikultur Rendah (209 Kab)": "Kabupaten pedesaan dengan pembukaan sawit/kebun kecil dan emisi rendah.",
                "C2: Agrikultur Masif (142 Kab)": "Pusat perkebunan sawit raksasa nasional dengan emisi tinggi (100% anggotanya berstatus Net Source!).",
                "C3: Infrastruktur / Urban (47 Kab)": "Daerah perkotaan atau padat penduduk di mana kehilangan pohon didorong oleh jalan, perumahan, dan fasilitas publik.",
                "C4: Agrikultur + Rawan Karhutla (28 Kab)": "Daerah gambut rawan bencana kebakaran besar saat kemarau panjang.",
                "C5: Hutan Lebat Tradisional (71 Kab)": "Daerah pedalaman Kalimantan dan Papua dengan tutupan hutan primer tinggi, didominasi ladang berpindah dan konsesi logging.",
            }
            CLUSTER_COLORS = {
                "C1: Agrikultur Rendah (209 Kab)": "#22c55e",
                "C2: Agrikultur Masif (142 Kab)": "#ef4444",
                "C3: Infrastruktur / Urban (47 Kab)": "#3b82f6",
                "C4: Agrikultur + Rawan Karhutla (28 Kab)": "#f97316",
                "C5: Hutan Lebat Tradisional (71 Kab)": "#15803d",
            }

            ct_plot = ct.copy()
            ct_plot["Tipologi"] = ct_plot["cluster_label"].map(CLUSTER_MAP).fillna(ct_plot["cluster_label"])

            if level == "Kabupaten/Kota":
                mine = ct_plot[(ct_plot["subnational1"] == prov) & (ct_plot["subnational2"] == kab)]
                if not mine.empty:
                    c_name = mine.iloc[0]["Tipologi"]
                    st.success(f"**Tipologi Wilayah {kab}: {c_name}**\n\n{CLUSTER_DESC.get(c_name, '')}")
            elif level == "Provinsi":
                st.info(f"Distribusi tipologi kabupaten/kota di Provinsi {prov} ditandai dengan simbol berlian putih pada peta PCA di bawah.")

            # 1. PETA TIPOLOGI (PROYEKSI PCA 2D) - ATAS (BESAR & PROPORSIONAL)
            figp = px.scatter(
                ct_plot,
                x="pc1",
                y="pc2",
                color="Tipologi",
                hover_data=["subnational1", "subnational2"],
                title="Peta Sebaran Tipologi Kabupaten (Proyeksi PCA 2D)",
                labels={
                    "pc1": "Komponen Utama 1 (PC1: Skala Luas Hutan & Biomassa)",
                    "pc2": "Komponen Utama 2 (PC2: Karakter Emisi & Intensitas Deforestasi)",
                    "Tipologi": "Klaster Tipologi",
                },
                color_discrete_map=CLUSTER_COLORS,
                opacity=0.75 if level != "Nasional" else 0.85,
            )
            if level == "Provinsi":
                ct_p = ct_plot[ct_plot["subnational1"] == prov]
                figp.add_trace(
                    go.Scatter(
                        x=ct_p["pc1"],
                        y=ct_p["pc2"],
                        mode="markers+text",
                        name=f"Kabupaten di {prov}",
                        text=ct_p["subnational2"],
                        textposition="top center",
                        marker=dict(size=13, color="white", line=dict(width=2, color="black"), symbol="diamond"),
                    )
                )
            elif level == "Kabupaten/Kota":
                mine = ct_plot[(ct_plot["subnational1"] == prov) & (ct_plot["subnational2"] == kab)]
                if not mine.empty:
                    figp.add_trace(
                        go.Scatter(
                            x=mine["pc1"],
                            y=mine["pc2"],
                            mode="markers",
                            name=f"{kab} (Terpilih)",
                            marker=dict(size=18, color="#fbbf24", line=dict(width=2.5, color="black"), symbol="star"),
                        )
                    )
            figp.update_layout(
                height=620,
                yaxis=dict(scaleanchor="x", scaleratio=1),
                legend=dict(orientation="h", y=-0.12, x=0),
                margin=dict(l=20, r=20, t=50, b=50),
            )
            figp.update_traces(marker=dict(size=9))
            st.plotly_chart(figp, width="stretch")
            st.caption("Peta proyeksi PCA 2D dibuat besar dan proporsional (skala 1:1) memisahkan kabupaten ke dalam 5 kelompok tipologi berdasarkan karakter biofisik & pemicu deforestasi.")

            st.markdown(
                """
                **5 Tipe Kabupaten Hasil Klasterisasi:**
                1. **Klaster C1 (Agrikultur Rendah - 209 Kab):** Kabupaten pedesaan dengan pembukaan sawit/kebun kecil dan emisi rendah.
                2. **Klaster C2 (Agrikultur Masif - 142 Kab):** Pusat perkebunan sawit raksasa nasional dengan emisi tinggi (100% anggotanya berstatus *Net Source*!).
                3. **Klaster C3 (Infrastruktur / Urban - 47 Kab):** Daerah perkotaan atau padat penduduk di mana kehilangan pohon didorong oleh jalan, perumahan, dan fasilitas publik.
                4. **Klaster C4 (Agrikultur + Rawan Karhutla - 28 Kab):** Daerah gambut rawan bencana kebakaran besar saat kemarau panjang.
                5. **Klaster C5 (Hutan Lebat Tradisional - 71 Kab):** Daerah pedalaman Kalimantan dan Papua dengan tutupan hutan primer tinggi, didominasi ladang berpindah dan konsesi logging.
                """
            )

            # 2. PETA PANAS KOMPOSISI PEMICU (HEATMAP) - BAWAH
            st.markdown("---")
            share_cols = [c for c in cs.columns if c.startswith("share_")]
            cs_heat = cs.copy()
            cs_heat["label_clean"] = cs_heat["label"].map(CLUSTER_MAP).fillna(cs_heat["label"])
            heat = cs_heat.set_index("label_clean")[share_cols]
            heat.columns = [c.replace("share_", "") for c in heat.columns]
            heat.columns = [LABELS.get(c, c) for c in heat.columns]
            fig_heat = px.imshow(
                heat,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="Greens",
                title="Komposisi Pemicu Rata-rata per Tipologi Wilayah",
                labels={"color": "Porsi Rata-rata"},
            )
            fig_heat.update_layout(
                height=360,
                margin=dict(l=20, r=20, t=40, b=40),
            )
            fig_heat.update_xaxes(tickangle=0)
            st.plotly_chart(fig_heat, width="stretch")
            st.caption("Peta panas menunjukkan porsi rata-rata masing-masing aktivitas pemicu deforestasi pada setiap klaster tipologi.")

            with st.expander("Buka / Lihat Rincian Tabel Karakteristik Tipologi", expanded=False):
                cs_show = cs.copy()
                cs_show["Tipologi"] = cs_show["label"].map(CLUSTER_MAP).fillna(cs_show["label"])
                st.dataframe(
                    cs_show[
                        ["Tipologi", "n", "density", "emission_per_ha", "primary_share", "net_source_share", "trend"]
                    ].rename(
                        columns={
                            "n": "Jumlah kab.",
                            "density": "Densitas biomassa (Mg C/ha)",
                            "emission_per_ha": "Intensitas (Mg CO₂e/ha)",
                            "primary_share": "Porsi hutan primer",
                            "net_source_share": "Porsi net source",
                            "trend": "Tren relatif",
                        }
                    ).round(2),
                    width="stretch",
                )

    # --------------------------------------------------------------------------
    # 5. PROYEKSI 2026-2030 (MULTI-HORIZON DENGAN VALIDASI MODEL & SPATIAL BREAKDOWN)
    # --------------------------------------------------------------------------
    with tab_fc:
        st.subheader(f"Proyeksi Emisi 2026–2030: {sub_heading}", help="Prediksi emisi multi-horizon jangka menengah (2026–2030) menggunakan model regresi berbobot waktu dan estimasi pita ketidakpastian probabilistik 80% (P10–P90).")

        if fc is None:
            st.warning("File proyeksi belum tersedia.")
        else:
            if level == "Nasional":
                scope_fc = fc.groupby("year")[["forecast_Mg_CO2e", "p10", "p90"]].sum().reset_index()
                hist_plot = _nat.reset_index().rename(columns={TARGET: "forecast_Mg_CO2e"})
            elif level == "Provinsi":
                scope_fc = fc[fc["subnational1"] == prov].groupby("year")[["forecast_Mg_CO2e", "p10", "p90"]].sum().reset_index()
                hist_plot = active_df.groupby("year")[TARGET].sum().reset_index().rename(columns={TARGET: "forecast_Mg_CO2e"})
            else:
                scope_fc = fc[(fc["subnational1"] == prov) & (fc["subnational2"] == kab)].sort_values("year")
                hist_plot = active_df.reset_index().rename(columns={TARGET: "forecast_Mg_CO2e"})

            if scope_fc.empty:
                st.info(f"Tidak ada data proyeksi untuk {sub_heading}.")
            else:
                # --------------------------------------------------------------
                # METRIC CARDS RINGKASAN PROYEKSI
                # --------------------------------------------------------------
                h_recent = hist_plot[hist_plot["year"] >= LAST_YEAR - 4]["forecast_Mg_CO2e"]
                h_mean = h_recent.mean() if not h_recent.empty else 0.0
                f_mean = scope_fc["forecast_Mg_CO2e"].mean()
                pct_delta = ((f_mean - h_mean) / (h_mean + 1e-6)) * 100

                def _fmt_em(val):
                    if abs(val) >= 1e6:
                        return f"{val/1e6:.2f} Jt Mg CO₂e"
                    return f"{val:,.0f} Mg CO₂e"

                m1, m2, m3 = st.columns(3)
                m1.metric("Rerata Historis (2021–2025)", _fmt_em(h_mean) + "/thn", help="Rata-rata emisi bruto tahunan selama 5 tahun terakhir periode historis.")
                m2.metric("Rerata Proyeksi (2026–2030)", _fmt_em(f_mean) + "/thn", help="Estimasi emisi rata-rata tahunan baseline untuk 5 tahun ke depan.")
                m3.metric(
                    "Tren Perubahan Rata-rata",
                    f"{pct_delta:+.1f}%",
                    delta="Kenaikan Emisi" if pct_delta > 0 else "Penurunan Emisi",
                    delta_color="inverse" if pct_delta > 0 else "normal",
                    help="Perubahan persentase rata-rata emisi proyeksi (2026–2030) dibanding 5 tahun terakhir (2021–2025).",
                )

                # --------------------------------------------------------------
                # GRAFIK UTAMA: LINTASAN HISTORIS & PROYEKSI MULTI-HORIZON
                # --------------------------------------------------------------
                fig_proj = go.Figure()
                fig_proj.add_trace(
                    go.Scatter(
                        x=hist_plot["year"],
                        y=hist_plot["forecast_Mg_CO2e"],
                        name="Realisasi Historis (2001–2025)",
                        mode="lines+markers",
                        line=dict(color="#2563eb", width=2.5),
                        marker=dict(size=5),
                        hovertemplate="<b>Tahun %{x} (Historis)</b><br>Emisi: %{y:,.0f} Mg CO₂e<extra></extra>",
                    )
                )
                fig_proj.add_trace(
                    go.Scatter(
                        x=scope_fc["year"],
                        y=scope_fc["p90"],
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
                fig_proj.add_trace(
                    go.Scatter(
                        x=scope_fc["year"],
                        y=scope_fc["p10"],
                        fill="tonexty",
                        line=dict(width=0),
                        name="Pita Ketidakpastian 80% (P10–P90)",
                        fillcolor="rgba(249, 115, 22, 0.22)",
                        hoverinfo="skip",
                    )
                )
                fig_proj.add_trace(
                    go.Scatter(
                        x=scope_fc["year"],
                        y=scope_fc["forecast_Mg_CO2e"],
                        name="Proyeksi Multi-Horizon (2026–2030)",
                        mode="lines+markers",
                        line=dict(color="#dc2626", width=2.5, dash="dash"),
                        marker=dict(size=6, symbol="diamond"),
                        hovertemplate="<b>Tahun %{x} (Proyeksi)</b><br>Proyeksi: %{y:,.0f} Mg CO₂e<extra></extra>",
                    )
                )
                fig_proj.update_layout(
                    title=f"Lintasan Historis & Proyeksi Emisi Tahunan (Mg CO₂e) - {sub_heading}",
                    height=440,
                    margin=dict(l=20, r=20, t=50, b=40),
                    legend=dict(orientation="h", y=-0.15, x=0),
                    xaxis=dict(dtick=2),
                )
                st.plotly_chart(fig_proj, width="stretch")

                # --------------------------------------------------------------
                # DIAGRAM PRIORITAS & SEBARAN PROYEKSI WILAYAH (KHUSUS NASIONAL & PROVINSI)
                # --------------------------------------------------------------
                if level == "Nasional":
                    st.markdown("---")
                    st.subheader("Top 10 Provinsi Prioritas Proyeksi Emisi (2026–2030)", help="10 provinsi dengan akumulasi volume proyeksi emisi kotor tertinggi periode 2026–2030.")
                    prov_fc = fc.groupby("subnational1")["forecast_Mg_CO2e"].sum().reset_index()
                    prov_fc["Jt_CO2e"] = prov_fc["forecast_Mg_CO2e"] / 1e6
                    top10_fc = prov_fc.sort_values("Jt_CO2e", ascending=True).tail(10)

                    fig_top_prov = px.bar(
                        top10_fc,
                        x="Jt_CO2e",
                        y="subnational1",
                        orientation="h",
                        text_auto=".1f",
                        labels={"Jt_CO2e": "Kumulatif 5-Thn (Jt Mg CO₂e)", "subnational1": ""},
                        color="Jt_CO2e",
                        color_continuous_scale="Reds",
                    )
                    fig_top_prov.update_layout(
                        height=400,
                        margin=dict(l=20, r=20, t=20, b=40),
                        coloraxis_showscale=False,
                    )
                    st.plotly_chart(fig_top_prov, width="stretch")
                    st.caption(
                        "10 provinsi dengan akumulasi proyeksi emisi tertinggi periode 2026–2030."
                    )

                elif level == "Provinsi":
                    st.markdown("---")
                    st.subheader(f"Peringkat Kabupaten/Kota Proyeksi Emisi Terbesar di {prov}", help=f"Peringkat kabupaten/kota di {prov} dengan akumulasi proyeksi emisi tertinggi periode 2026–2030.")
                    kab_fc = fc[fc["subnational1"] == prov].groupby("subnational2")["forecast_Mg_CO2e"].sum().reset_index()
                    kab_fc["Jt_CO2e"] = kab_fc["forecast_Mg_CO2e"] / 1e6
                    top10_k = kab_fc.sort_values("Jt_CO2e", ascending=True).tail(10)

                    fig_top_k = px.bar(
                        top10_k,
                        x="Jt_CO2e",
                        y="subnational2",
                        orientation="h",
                        text_auto=".1f",
                        labels={"Jt_CO2e": "Kumulatif 5-Thn (Jt Mg CO₂e)", "subnational2": ""},
                        color="Jt_CO2e",
                        color_continuous_scale="Oranges",
                    )
                    fig_top_k.update_layout(
                        height=400,
                        margin=dict(l=20, r=20, t=20, b=40),
                        coloraxis_showscale=False,
                    )
                    st.plotly_chart(fig_top_k, width="stretch")
                    st.caption(
                        f"Sebaran kabupaten/kota di {prov} dengan akumulasi proyeksi emisi tertinggi sepanjang 2026–2030."
                    )

                # --------------------------------------------------------------
                # EXPANDER TABEL RINCIAN
                # --------------------------------------------------------------
                with st.expander(f"Buka / Lihat Rincian Angka Proyeksi Tahunan 2026–2030 ({sub_heading})", expanded=False):
                    fc_show = scope_fc[["year", "forecast_Mg_CO2e", "p10", "p90"]].copy()
                    fc_show.columns = ["Tahun", "Proyeksi Dasar (Mg CO₂e)", "Batas Bawah P10 (Mg CO₂e)", "Batas Atas P90 (Mg CO₂e)"]
                    st.dataframe(fc_show.round(1), width="stretch")

    if level == "Nasional":
        with tab_anom:
            st.subheader("Deteksi Anomali: Lonjakan Emisi Ekstrem Historis (Nasional)", help="Identifikasi lonjakan emisi abnormal secara statistik menggunakan Skor-Z Robust (Median Absolute Deviation) dengan ambang batas Z > 3.0 untuk mendeteksi krisis karhutla katastrofik.")
            if an is not None and not an.empty:
                an_sub = an.copy()
                an_sub["Label"] = an_sub["subnational2"] + ", " + an_sub["subnational1"] + " (" + an_sub["year"].astype(str) + ")"
                an_sub["Emisi_Jt"] = an_sub[TARGET] / 1e6

                col_an1, col_an2 = st.columns(2)

                with col_an1:
                    st.markdown("##### Timeline Sebaran Lonjakan Anomali Ekstrem (Nasional)", help="Titik krisis lonjakan emisi ekstrem sepanjang tahun 2001–2025.")
                    fig_bubble = px.scatter(
                        an_sub,
                        x="year",
                        y="Emisi_Jt",
                        size="total_loss_ha",
                        color="likely_cause",
                        hover_name="Label",
                        hover_data={
                            "year": True,
                            "Emisi_Jt": ":.2f",
                            "total_loss_ha": ":,.0f ha",
                            "robust_z": ":.1f",
                            "likely_cause": True,
                        },
                        labels={
                            "year": "Tahun Kejadian",
                            "Emisi_Jt": "Emisi (Juta Mg CO₂e)",
                            "likely_cause": "Penyebab",
                            "total_loss_ha": "Luas Hilang (ha)",
                        },
                        color_discrete_map={
                            "Kebakaran": "#dc2626",
                            "Agrikultur/Sawit": "#2563eb",
                            "Pembalakan": "#059669",
                        },
                    )
                    fig_bubble.update_layout(
                        height=420,
                        xaxis=dict(dtick=2),
                        margin=dict(l=20, r=20, t=10, b=40),
                    )
                    fig_bubble.update_traces(marker=dict(sizemin=8, line=dict(width=1, color="white")))
                    st.plotly_chart(fig_bubble, width="stretch")
                    st.caption("Ukuran lingkaran merefleksikan besaran luas pembukaan/kebakaran hutan (ha) pada titik krisis lonjakan emisi ekstrem.")

                with col_an2:
                    st.markdown("##### Proporsi Emisi Anomali Berdasarkan Penyebab (Nasional)", help="Distribusi penyebab dominan dari total emisi lonjakan anomali.")
                    an_cause = an_sub.groupby("likely_cause")[TARGET].sum().reset_index()
                    an_cause["Emisi_Jt"] = an_cause[TARGET] / 1e6
                    tot_an_em = an_cause["Emisi_Jt"].sum() + 1e-6
                    an_cause["Pct"] = (an_cause["Emisi_Jt"] / tot_an_em * 100).round(1)
                    an_cause["Label"] = (
                        an_cause["likely_cause"]
                        + " ("
                        + an_cause["Emisi_Jt"].round(1).astype(str)
                        + " Jt - "
                        + an_cause["Pct"].astype(str)
                        + "%)"
                    )

                    fig_donut_an = px.pie(
                        an_cause,
                        names="Label",
                        values="Emisi_Jt",
                        hole=0.45,
                        color="likely_cause",
                        color_discrete_map={
                            "Kebakaran": "#dc2626",
                            "Agrikultur/Sawit": "#2563eb",
                            "Pembalakan": "#059669",
                        },
                    )
                    fig_donut_an.update_traces(
                        rotation=90,
                        textposition="outside",
                        textinfo="percent+label",
                        outsidetextfont=dict(size=12),
                        hovertemplate="<b>%{label}</b><br>Emisi: %{value:.2f} Jt Mg CO₂e<br>Porsi: %{percent}<extra></extra>",
                    )
                    fig_donut_an.update_layout(
                        height=420,
                        legend=dict(orientation="h", y=-0.15, x=0),
                        margin=dict(l=30, r=30, t=10, b=40),
                    )
                    st.plotly_chart(fig_donut_an, width="stretch")
                    st.caption("Kebakaran lahan gambut menyumbang >92% emisi krisis anomali nasional saat kemarau ekstrem El Niño (2006, 2014, dan 2015–2016).")

                with st.expander("Buka / Lihat Rincian Tabel Data Anomali Nasional", expanded=False):
                    st.dataframe(
                        an_sub.rename(
                            columns={
                                "subnational1": "Provinsi",
                                "subnational2": "Kabupaten/Kota",
                                "year": "Tahun",
                                TARGET: "Emisi (Mg CO₂e)",
                                "total_loss_ha": "Luas Hilang (ha)",
                                "likely_cause": "Penyebab Dominan",
                                "wildfire_share": "Porsi Kebakaran",
                                "robust_z": "Skor-Z",
                            }
                        ).round(2),
                        width="stretch",
                    )
            else:
                st.info("Data deteksi anomali belum tersedia.")


# ==============================================================================
# TAB UTAMA: TOOLS (PREDIKSI & SIMULASI) - TERSEDIA DI SEMUA TINGKAT
# ==============================================================================
with main_tools:
    tool_sim, tool_scen = st.tabs(["Simulasi Emisi", "Skenario Mitigasi & Nilai Karbon"])

    # --------------------------------------------------------------------------
    # 1. SIMULASI EMISI
    # --------------------------------------------------------------------------
    with tool_sim:
        st.subheader(f"Simulasi Skenario: {sub_heading}", help="Simulator interaktif untuk memperkirakan respon emisi terhadap perubahan luas kehilangan tutupan lahan per aktivitas pemicu.")
        st.info(
            f"Prediksi memakai kondisi hutan di akhir {LAST_YEAR}. "
            "Model memprediksi intensitas emisi per hektare lalu mengalikannya dengan luas hilang, sehingga skenario bukaan tetap rasional."
        )

        recent_scope = active_df[active_df["year"] > LAST_YEAR - 5]
        if level == "Kabupaten/Kota":
            latest_row = active_df.iloc[-1] if not active_df.empty else None
            default_drv = {d: float(latest_row[d]) if latest_row is not None else 0.0 for d in DRIVERS}
            prim_default = float(latest_row["primary_loss_ha"]) if latest_row is not None and pd.notna(latest_row["primary_loss_ha"]) else 0.0
        elif level == "Provinsi":
            prov_yearly_drv = active_df.groupby("year")[DRIVERS].sum()
            default_drv = {d: float(prov_yearly_drv[d].iloc[-1]) if not prov_yearly_drv.empty else 0.0 for d in DRIVERS}
            prov_prim = active_df.groupby("year")["primary_loss_ha"].sum()
            prim_default = float(prov_prim.iloc[-1]) if not prov_prim.empty else 0.0
        else:
            nat_yearly_drv = df.groupby("year")[DRIVERS].sum()
            default_drv = {d: float(nat_yearly_drv[d].iloc[-1]) if not nat_yearly_drv.empty else 0.0 for d in DRIVERS}
            nat_prim = df.groupby("year")["primary_loss_ha"].sum()
            prim_default = float(nat_prim.iloc[-1]) if not nat_prim.empty else 0.0

        cols = st.columns(3)
        drv_in = {}
        for i, d in enumerate(DRIVERS):
            with cols[i % 3]:
                drv_in[d] = st.number_input(
                    f"{LABELS[d]} (ha)",
                    min_value=0.0,
                    value=default_drv[d],
                    step=10.0,
                    key=f"sim_in_{level}_{d}",
                )
        tot_loss_sim = sum(drv_in.values())
        prim_sim = st.number_input(
            "Dari total itu, berapa ha hutan primer? (hutan primer = karbon lebih tinggi)",
            min_value=0.0,
            max_value=max(tot_loss_sim, 1.0),
            value=min(prim_default, max(tot_loss_sim, 1.0)),
            step=10.0,
            key=f"sim_prim_{level}",
        )

        if level == "Kabupaten/Kota":
            mean_p, lo_p, hi_p = predict(drv_in, prim_sim, st_k)
        else:
            avg_int = active_df[TARGET].sum() / (active_df["total_loss_ha"].sum() + 1e-6)
            mean_p = tot_loss_sim * avg_int
            lo_p = mean_p * 0.75
            hi_p = mean_p * 1.35

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Kehilangan Tutupan", f"{tot_loss_sim:,.0f} ha", help="Total luas tutupan pohon yang hilang dari penjumlahan seluruh pemicu.")
        c2.metric("Estimasi Emisi Kotor", f"{mean_p:,.0f} Mg CO₂e", help="Estimasi total emisi kotor berdasarkan model Machine Learning (Mg CO₂e = Ton CO₂e).")
        c3.metric("Intensitas Emisi", f"{mean_p / tot_loss_sim:,.0f} Mg CO₂e/ha" if tot_loss_sim > 0 else "-", help="Rata-rata emisi terlepas per 1 hektare tutupan pohon yang dibuka.")
        st.caption(f"Rentang ketidakpastian 80%: {lo_p:,.0f} - {hi_p:,.0f} Mg CO₂e")

        # Sensitivitas
        sens = []
        for d in DRIVERS:
            alt_drv = dict(drv_in)
            alt_drv[d] = drv_in[d] * 1.10 if drv_in[d] > 0 else 10.0
            alt_tot = sum(alt_drv.values())
            alt_prim = prim_sim * (alt_tot / (tot_loss_sim + 1e-6))
            if level == "Kabupaten/Kota":
                em_alt = predict(alt_drv, alt_prim, st_k)[0]
            else:
                em_alt = alt_tot * avg_int
            sens.append({"Pemicu": LABELS[d], "Δ Emisi (Mg CO₂e)": em_alt - mean_p})

        sens_df = pd.DataFrame(sens).sort_values("Δ Emisi (Mg CO₂e)")
        st.plotly_chart(
            px.bar(
                sens_df,
                x="Δ Emisi (Mg CO₂e)",
                y="Pemicu",
                orientation="h",
                title="Sensitivitas: perubahan emisi jika tiap pemicu naik 10% (atau +10 ha bila 0)",
            ),
            width="stretch",
        )
        st.download_button(
            "Unduh hasil (CSV)",
            pd.DataFrame([{"wilayah": sub_heading, **drv_in, "pred_mean": mean_p, "pred_p10": lo_p, "pred_p90": hi_p}]).to_csv(index=False),
            file_name=f"simulasi_{level}_{prov or 'nasional'}.csv",
        )

    # --------------------------------------------------------------------------
    # 2. SKENARIO MITIGASI & NILAI KARBON
    # --------------------------------------------------------------------------
    with tool_scen:
        st.subheader(f"Skenario Mitigasi: berapa emisi yang bisa dicegah? ({sub_heading})", help="Perhitungan potensi penurunan emisi jika target pencegahan deforestasi tercapai, beserta potensi valuasi kredit pasar karbon.")
        st.markdown("Bandingkan kondisi dasar (rata-rata 5 tahun terakhir) dengan skenario pengurangan per pemicu.")

        if active_df.empty:
            st.warning("Data historis tidak tersedia.")
        else:
            recent_5 = active_df[active_df["year"] > LAST_YEAR - 5]
            if level == "Kabupaten/Kota":
                base_drv = {d: float(recent_5[d].mean()) for d in DRIVERS}
                base_prim = float(recent_5["primary_loss_ha"].mean()) if recent_5["primary_loss_ha"].notna().any() else 0.0
            else:
                yearly_5 = recent_5.groupby("year")[DRIVERS].sum()
                base_drv = {d: float(yearly_5[d].mean()) for d in DRIVERS}
                yearly_prim_5 = recent_5.groupby("year")["primary_loss_ha"].sum()
                base_prim = float(yearly_prim_5.mean()) if not yearly_prim_5.empty else 0.0

            red = {}
            cc = st.columns(3)
            for i, d in enumerate(DRIVERS):
                with cc[i % 3]:
                    red[d] = st.slider(f"Pengurangan {LABELS[d]} (%)", 0, 100, 0, 5, key=f"mit_red_{level}_{d}")

            scen_drv = {d: base_drv[d] * (1 - red[d] / 100) for d in DRIVERS}
            base_tot = sum(base_drv.values())
            scen_tot = sum(scen_drv.values())

            if level == "Kabupaten/Kota":
                b_pred = predict(base_drv, base_prim, st_k)[0]
                s_pred = predict(scen_drv, base_prim * (scen_tot / (base_tot + 1e-6)), st_k)[0]
            else:
                avg_int_5 = recent_5[TARGET].sum() / (recent_5["total_loss_ha"].sum() + 1e-6)
                b_pred = base_tot * avg_int_5
                s_pred = scen_tot * avg_int_5

            avoided = max(0.0, b_pred - s_pred)
            m1, m2, m3 = st.columns(3)
            m1.metric("Emisi Baseline", f"{b_pred:,.0f} Mg CO₂e", help="Emisi acuan tahunan berdasarkan tren aktivitas rata-rata 5 tahun terakhir.")
            m2.metric("Emisi Skenario", f"{s_pred:,.0f} Mg CO₂e", help="Emisi tahunan yang diproyeksikan jika target pengurangan tercapai.")
            m3.metric("Emisi Dicegah", f"{avoided:,.0f} Mg CO₂e", f"{avoided / b_pred * 100:.1f}%" if b_pred else None, help="Total beban emisi gas rumah kaca yang berhasil dicegah berkat aksi mitigasi.")

            price = st.number_input(
                "Asumsi harga karbon (USD per ton CO₂e):",
                min_value=0.0,
                value=5.0,
                step=1.0,
                key=f"price_{level}",
            )
            val_usd = avoided * price
            val_idr = val_usd * 16000
            st.metric(
                "Nilai ekonomi emisi dicegah (potensi kredit karbon)",
                f"US$ {val_usd:,.0f}",
                delta=f"≈ Rp {val_idr / 1e9:,.2f} Miliar (Kurs Rp16.000)",
                delta_color="off",
                help="Potensi nilai ekonomi dari sertifikat penurunan emisi di pasar karbon berdasarkan asumsi harga per ton.",
            )
            figc = go.Figure(go.Bar(x=["Baseline", "Skenario"], y=[b_pred, s_pred], marker_color=["#c62828", "#2e7d32"]))
            figc.update_layout(title="Perbandingan Emisi (Mg CO₂e)")
            st.plotly_chart(figc, width="stretch")
            st.caption("Catatan: model bersifat indikatif berbasis pola historis data GFW.")
