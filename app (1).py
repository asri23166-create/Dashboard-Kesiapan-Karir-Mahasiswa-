
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import json
import time
import os
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Career Learning Path",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Plus Jakarta Sans', sans-serif !important; }

/* Dark dropdown */
[data-baseweb="select"] ul li,
[data-baseweb="popover"] [data-baseweb="menu"],
[role="option"],
[data-baseweb="option"] {
    color: #e6edf3 !important;
    background-color: #1c2128 !important;
}

[role="option"]:hover,
[data-baseweb="option"]:hover {
    background-color: #2d333b !important;
    color: white !important;
}

.stApp {
    background: linear-gradient(135deg,
        #0a0a1a 0%, #0d1b2a 40%, #1a0a2e 100%);
    min-height: 100vh;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg,
        #0d1117 0%, #161b22 100%);
    border-right: 1px solid rgba(88,166,255,0.15);
}

/* Glass Card */
.glass {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 24px;
    margin-bottom: 16px;
    backdrop-filter: blur(20px);
    transition: all 0.3s ease;
}
.glass:hover {
    border-color: rgba(88,166,255,0.3);
    transform: translateY(-2px);
}

/* Gradient title */
.grad-title {
    background: linear-gradient(135deg, #58a6ff, #bc8cff, #ff7eb3);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    line-height: 1.2;
}

/* Metric card */
.metric {
    background: rgba(88,166,255,0.08);
    border: 1px solid rgba(88,166,255,0.2);
    border-radius: 16px;
    padding: 18px;
    text-align: center;
}

/* Verdict cards */
.v-sangat {
    background: linear-gradient(135deg, #0d4f3c, #145a32);
    border: 1px solid #27ae60;
    border-radius: 20px; padding: 24px;
    text-align: center;
    box-shadow: 0 0 30px rgba(39,174,96,0.3);
}
.v-siap {
    background: linear-gradient(135deg, #0a2d5c, #154360);
    border: 1px solid #2e86c1;
    border-radius: 20px; padding: 24px;
    text-align: center;
    box-shadow: 0 0 30px rgba(46,134,193,0.3);
}
.v-hampir {
    background: linear-gradient(135deg, #5c3a0a, #7d5a0f);
    border: 1px solid #d4ac0d;
    border-radius: 20px; padding: 24px;
    text-align: center;
    box-shadow: 0 0 30px rgba(212,172,13,0.3);
}
.v-perlu {
    background: linear-gradient(135deg, #5c0a0a, #7d1212);
    border: 1px solid #e74c3c;
    border-radius: 20px; padding: 24px;
    text-align: center;
    box-shadow: 0 0 30px rgba(231,76,60,0.3);
}

/* Progress */
.prog-wrap {
    background: rgba(255,255,255,0.08);
    border-radius: 99px; height: 8px;
    overflow: hidden; margin: 6px 0;
}
.prog-fill {
    height: 8px; border-radius: 99px;
}

/* Info boxes */
.box-info {
    background: rgba(88,166,255,0.1);
    border-left: 3px solid #58a6ff;
    border-radius: 0 12px 12px 0;
    padding: 12px 16px; margin: 8px 0;
    color: #a5d8ff; font-size: 13px;
}
.box-warn {
    background: rgba(255,193,7,0.1);
    border-left: 3px solid #ffc107;
    border-radius: 0 12px 12px 0;
    padding: 12px 16px; margin: 8px 0;
    color: #ffe082; font-size: 13px;
}
.box-success {
    background: rgba(39,174,96,0.1);
    border-left: 3px solid #27ae60;
    border-radius: 0 12px 12px 0;
    padding: 12px 16px; margin: 8px 0;
    color: #a9dfbf; font-size: 13px;
}
.box-danger {
    background: rgba(231,76,60,0.1);
    border-left: 3px solid #e74c3c;
    border-radius: 0 12px 12px 0;
    padding: 12px 16px; margin: 8px 0;
    color: #f5b7b1; font-size: 13px;
}

/* Tahap card */
.tahap-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px; padding: 18px;
    margin: 10px 0;
    transition: all 0.3s;
}
.tahap-card:hover {
    border-color: rgba(88,166,255,0.3);
    background: rgba(88,166,255,0.05);
}

/* Kursus/sertif badge */
.badge-kursus {
    background: rgba(88,166,255,0.15);
    border: 1px solid rgba(88,166,255,0.3);
    border-radius: 8px; padding: 8px 12px;
    margin: 4px 0; font-size: 12px;
    color: #a5d8ff;
}
.badge-sertif {
    background: rgba(188,140,255,0.15);
    border: 1px solid rgba(188,140,255,0.3);
    border-radius: 8px; padding: 8px 12px;
    margin: 4px 0; font-size: 12px;
    color: #d8b4fe;
}
.badge-kegiatan {
    background: rgba(255,126,179,0.1);
    border: 1px solid rgba(255,126,179,0.3);
    border-radius: 8px; padding: 8px 12px;
    margin: 4px 0; font-size: 12px;
    color: #fbb6ce;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg,
        #58a6ff, #bc8cff) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 28px !important;
    font-weight: 600 !important;
    width: 100% !important;
    transition: all 0.3s !important;
    font-size: 15px !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(88,166,255,0.4) !important;
}

/* Text Input */
.stTextInput input {
    background-color: #1e293b !important;
    color: #ffffff !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
}

/* Number Input */
.stNumberInput input {
    background-color: #1e293b !important;
    color: #ffffff !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
}

/* Placeholder */
.stTextInput input::placeholder,
.stNumberInput input::placeholder {
    color: #94a3b8 !important;
}

/* Selectbox */
.stSelectbox div[data-baseweb="select"] > div {
    background-color: #1e293b !important;
    color: #ffffff !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.04);
    border-radius: 14px; padding: 4px; gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    color: #8b949e !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,
        #58a6ff, #bc8cff) !important;
    color: white !important;
}

/* Divider */
hr { border-color: rgba(255,255,255,0.06) !important; }

/* Hide branding */
#MainMenu, footer, header { visibility: hidden; }

/* Text */
h1,h2,h3,h4,p,span,label,div { color: #e6edf3; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTS
# ============================================================

CATEGORIES = [
    'Perlu Persiapan', 'Hampir Siap',
    'Siap Masuk Industri', 'Sangat Siap'
]

TARGET_SKOR = {
    'Sangat Siap'        : 85,
    'Siap Masuk Industri': 70,
    'Hampir Siap'        : 55,
    'Perlu Persiapan'    : 40
}

MODUL_KEYS = [
    'Skor M1 (Background)','Skor M2 (Skills)',
    'Skor M3 (Industry)', 'Skor M4 (Interest)',
    'Skor M5 (Compass)',  'Skor M6 (Company)',
    'Skor M7 (Branding)', 'Skor M8 (Ambisi)',
    'Skor M9 (Resiliensi)'
]

MODUL_LABELS = {
    'Skor M1 (Background)' : ('M1','Background','🏫'),
    'Skor M2 (Skills)'     : ('M2','Skills','💻'),
    'Skor M3 (Industry)'   : ('M3','Industry','🏭'),
    'Skor M4 (Interest)'   : ('M4','Interest','❤️'),
    'Skor M5 (Compass)'    : ('M5','Compass','🧭'),
    'Skor M6 (Company)'    : ('M6','Company','🏢'),
    'Skor M7 (Branding)'   : ('M7','Branding','✨'),
    'Skor M8 (Ambisi)'     : ('M8','Ambisi','🎯'),
    'Skor M9 (Resiliensi)' : ('M9','Resiliensi','💪')
}

MODUL_KATALOG_MAP = {
    'Skor M1 (Background)' : 'M1_Background',
    'Skor M2 (Skills)'     : 'M2_Skills',
    'Skor M3 (Industry)'   : 'M3_Industry',
    'Skor M4 (Interest)'   : 'M4_Interest',
    'Skor M5 (Compass)'    : 'M5_Compass',
    'Skor M6 (Company)'    : 'M6_Company',
    'Skor M7 (Branding)'   : 'M7_Branding',
    'Skor M8 (Ambisi)'     : 'M8_Ambisi',
    'Skor M9 (Resiliensi)' : 'M9_Resiliensi'
}

FEATURE_IMP = {
    'Skor M2 (Skills)'     : 0.0533,
    'Skor M4 (Interest)'   : 0.0530,
    'Skor M3 (Industry)'   : 0.0516,
    'Skor M7 (Branding)'   : 0.0495,
    'Skor M9 (Resiliensi)' : 0.0475,
    'Skor M8 (Ambisi)'     : 0.0249,
    'Skor M1 (Background)' : 0.0158,
    'Skor M5 (Compass)'    : 0.0132,
    'Skor M6 (Company)'    : 0.0023
}

# ============================================================
# LOAD MODEL & KATALOG
# ============================================================

@st.cache_resource
def load_assets():
    try:
        model    = joblib.load('best_model_clean.pkl')
        le       = joblib.load('label_encoder.pkl')
        scaler   = joblib.load('scaler_clean.pkl')
        features = joblib.load('feature_names_clean.pkl')
        loaded   = True
        print("Model berhasil dimuat!")
    except Exception as e:
        print(f"Model tidak dimuat: {e}")
        model = le = scaler = features = None
        loaded = False
    try:
        with open('katalog_rekomendasi.json',
                  'r', encoding='utf-8') as f:
            katalog = json.load(f)
    except:
        katalog = {}
    return model, le, scaler, features, katalog, loaded

model, le, scaler, feature_names, katalog, model_loaded = load_assets()

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def skor_color(s):
    if s >= 80: return '#27ae60'
    if s >= 65: return '#2e86c1'
    if s >= 50: return '#d4ac0d'
    return '#e74c3c'

def skor_label(s):
    if s >= 80: return '🟢 Sangat Baik'
    if s >= 65: return '🔵 Cukup Baik'
    if s >= 50: return '🟡 Perlu Ditingkatkan'
    return '🔴 Perlu Perhatian'

def get_level(s):
    if s < 50: return 'rendah'
    if s < 75: return 'sedang'
    return 'tinggi'

def predict_verdict(input_dict):

    if model_loaded:

        try:

            df_in = pd.DataFrame([input_dict])

            # Encode fitur kategorikal
            for col, col_enc in lab_encs.items():

                if col in df_in.columns:

                    try:
                        df_in[col] = col_enc.transform(
                            df_in[col].astype(str)
                        )

                    except:
                        df_in[col] = 0

            # Tambah kolom yang hilang
            for col in feature_names:

                if col not in df_in.columns:
                    df_in[col] = 0

            df_in = df_in[feature_names]

            # Boolean → Integer
            for col in df_in.select_dtypes(
                include=["bool"]
            ).columns:

                df_in[col] = df_in[col].astype(int)

            # Scaling
            df_sc = scaler.transform(df_in)

            # Predict
            pred = model.predict(df_sc)[0]

            proba = model.predict_proba(df_sc)[0]

            verdict = le.inverse_transform([pred])[0]

            confidence = proba[pred] * 100

            return verdict, confidence, proba

        except Exception as e:

            print(f"Predict error: {e}")

            return _demo_predict(input_dict)

    else:

        return _demo_predict(input_dict)

def _demo_predict(input_dict):
        """Demo mode prediksi sederhana"""
        skor_list = [input_dict.get(k, 70) for k in MODUL_KEYS]
        avg = np.mean(skor_list)
        ipk = input_dict.get('IPK', 3.0)
        tot = avg * 0.75 + float(str(ipk).replace(',','.') if ipk else 3.0) * 10 * 0.25
        if tot >= 82: v, i = 'Sangat Siap', 3
        elif tot >= 68: v, i = 'Siap Masuk Industri', 2
        elif tot >= 55: v, i = 'Hampir Siap', 1
        else: v, i = 'Perlu Persiapan', 0
        proba = np.zeros(4)
        proba[i] = 0.72
        proba[(i+1)%4] = 0.16
        proba[(i-1)%4] = 0.12
        return v, 72.0, proba

def generate_lp(skor_dict, verdict):
    target = TARGET_SKOR.get(verdict, 70)
    prio   = []
    for modul, skor in skor_dict.items():
        gap = max(0, target - skor)
        if gap > 0:
            fi    = FEATURE_IMP.get(modul, 0.01)
            score = gap * (1 + fi * 10)
            prio.append({
                'modul' : modul,
                'gap'   : gap,
                'skor'  : skor,
                'target': target,
                'fi'    : fi,
                'score' : score
            })
    prio = sorted(prio, key=lambda x: x['score'],
                  reverse=True)[:5]
    lp = []
    for i, item in enumerate(prio):
        mk  = MODUL_KATALOG_MAP.get(item['modul'])
        if not mk or mk not in katalog: continue
        lvl = get_level(item['skor'])
        kat = katalog[mk][lvl]
        lp.append({
            'tahap'    : i+1,
            'modul'    : item['modul'],
            'icon'     : katalog[mk]['icon'],
            'nama'     : katalog[mk]['nama'],
            'skor'     : item['skor'],
            'target'   : item['target'],
            'gap'      : item['gap'],
            'level'    : lvl,
            'kursus'   : kat['kursus'],
            'sertif'   : kat['sertifikasi'],
            'kegiatan' : kat['kegiatan'],
            'timeline' : kat['timeline']
        })
    return lp, prio

def verdict_card_css(v):
    return {
        'Sangat Siap'        : ('v-sangat','🟢','#27ae60'),
        'Siap Masuk Industri': ('v-siap','🔵','#2e86c1'),
        'Hampir Siap'        : ('v-hampir','🟡','#d4ac0d'),
        'Perlu Persiapan'    : ('v-perlu','🔴','#e74c3c')
    }.get(v, ('v-perlu','⚪','#888'))

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:20px 0 12px;">
        <div style="font-size:44px;">🚀</div>
        <div style="font-size:16px; font-weight:700;
             background:linear-gradient(135deg,#58a6ff,#bc8cff);
             -webkit-background-clip:text;
             -webkit-text-fill-color:transparent;
             margin-top:6px;">Career Learning Path</div>
        <div style="font-size:11px; color:#484f58;
             margin-top:2px;">Sistem Rekomendasi Karir</div>
    </div>
    <hr>
    """, unsafe_allow_html=True)

    page = st.radio("Menu", [
        "🏠  Beranda",
        "🎯  Prediksi & Klasifikasi",
        "📊  Gap Analysis",
        "🗺️   Learning Path",
        "📁  Prediksi Massal",
        "📈  Dashboard Data",
        "ℹ️   Panduan"
    ], label_visibility="collapsed")

    st.markdown("<hr>", unsafe_allow_html=True)

    if not model_loaded:
        st.markdown("""
        <div class="box-warn">
            ⚠️ <b>Demo Mode</b><br>
            Upload file .pkl untuk prediksi akurat.
        </div>""", unsafe_allow_html=True)
        up = st.file_uploader("Upload Model",
                              type=['pkl'])
        if up:
            with open('best_model_clean.pkl','wb') as f:
                f.write(up.read())
            st.success("✅ Refresh halaman!")
    else:
        st.markdown("""
        <div class="box-success">
            ✅ <b>Model Aktif</b><br>
            Siap prediksi & rekomendasi.
        </div>""", unsafe_allow_html=True)

    if katalog:
        st.markdown("""
        <div class="box-success">
            ✅ <b>Katalog Aktif</b><br>
            27 set rekomendasi tersedia.
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; font-size:11px;
         color:#484f58; margin-top:20px;">
        Asri Putri Lestari<br>
        Proyek Akhir © 2026
    </div>""", unsafe_allow_html=True)

# ============================================================
# PAGE: BERANDA
# ============================================================

if page == "🏠  Beranda":

    st.markdown("""
    <div style="text-align:center; padding:50px 0 30px;">
        <div style="font-size:56px;">🚀</div>
        <div class="grad-title" style="font-size:38px;">
            Sistem Rekomendasi<br>Learning Path Karir
        </div>
        <p style="color:#8b949e; font-size:16px;
           margin-top:16px; max-width:600px;
           margin-left:auto; margin-right:auto;">
            Dari hasil klasifikasi kesiapan karir, sistem ini
            merekomendasikan learning path yang personal dan
            terstruktur untuk membantu mahasiswa mencapai
            kesiapan karir yang optimal.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    for col,(val,lbl,icon) in zip(
        [c1,c2,c3,c4],
        [('1000+','Data Mahasiswa','👥'),
         ('9','Modul Asesmen','📊'),
         ('27','Set Rekomendasi','📚'),
         ('4','Kategori Verdict','🏆')]
    ):
        with col:
            st.markdown(f"""
            <div class="metric">
                <div style="font-size:26px;">{icon}</div>
                <div style="font-size:24px; font-weight:800;
                     background:linear-gradient(135deg,
                     #58a6ff,#bc8cff);
                     -webkit-background-clip:text;
                     -webkit-text-fill-color:transparent;">
                     {val}</div>
                <div style="font-size:12px; color:#8b949e;">
                     {lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:18px; font-weight:700;
         background:linear-gradient(135deg,#58a6ff,#bc8cff);
         -webkit-background-clip:text;
         -webkit-text-fill-color:transparent;
         margin-bottom:16px;">
        🔄 Alur Sistem
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    for col,(num,icon,title,desc) in zip(
        [c1,c2,c3,c4],
        [('1','📝','Input Data',
          'Isi data diri & skor asesmen M1–M9'),
         ('2','🤖','Klasifikasi',
          'Model ML prediksi verdict kesiapan'),
         ('3','📊','Gap Analysis',
          'Identifikasi modul yang perlu ditingkatkan'),
         ('4','🗺️','Learning Path',
          'Rekomendasi kursus, sertifikasi & kegiatan')]
    ):
        with col:
            st.markdown(f"""
            <div class="glass" style="text-align:center;">
                <div style="background:linear-gradient(135deg,
                     #58a6ff,#bc8cff); color:white;
                     border-radius:50%; width:32px; height:32px;
                     display:inline-flex; align-items:center;
                     justify-content:center; font-weight:700;
                     font-size:14px; margin-bottom:8px;">
                     {num}</div>
                <div style="font-size:24px;">{icon}</div>
                <div style="font-weight:700; color:#e6edf3;
                     font-size:14px; margin:6px 0;">{title}</div>
                <div style="font-size:12px; color:#8b949e;">
                     {desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:18px; font-weight:700;
         background:linear-gradient(135deg,#58a6ff,#bc8cff);
         -webkit-background-clip:text;
         -webkit-text-fill-color:transparent;
         margin-bottom:16px;">
        🏅 Kategori Verdict & Rekomendasi
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    for col,(css,icon,title,clr,target,desc) in zip(
        [c1,c2,c3,c4],
        [('v-perlu','🔴','Perlu Persiapan','#e74c3c',
          'Skor < 40','Butuh banyak pengembangan'),
         ('v-hampir','🟡','Hampir Siap','#d4ac0d',
          'Skor 40–54','Sudah di jalur benar'),
         ('v-siap','🔵','Siap Masuk Industri','#2e86c1',
          'Skor 55–69','Siap mulai apply kerja'),
         ('v-sangat','🟢','Sangat Siap','#27ae60',
          'Skor ≥ 70','Siap penuh & optimal')]
    ):
        with col:
            st.markdown(f"""
            <div class="{css}">
                <div style="font-size:28px;">{icon}</div>
                <div style="font-weight:700; color:white;
                     font-size:13px; margin:8px 0;">{title}</div>
                <div style="font-size:11px;
                     color:rgba(255,255,255,0.7);">{target}</div>
                <div style="font-size:11px;
                     color:rgba(255,255,255,0.6);
                     margin-top:4px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# PAGE: PREDIKSI & KLASIFIKASI
# ============================================================

elif page == "🎯  Prediksi & Klasifikasi":

    st.markdown("""
    <div class="grad-title" style="font-size:26px;">
        🎯 Prediksi Kesiapan Karir
    </div>
    <p style="color:#8b949e; margin-bottom:24px;">
        Isi data diri dan skor asesmen untuk mendapatkan
        prediksi & learning path yang personal.
    </p>
    """, unsafe_allow_html=True)

    # Form data diri
    st.markdown("""
    <div class="glass">
        <div style="font-weight:700; color:#58a6ff;
             margin-bottom:12px; font-size:15px;">
            👤 Data Diri & Akademik
        </div>
        <div class="box-info">
            💡 Isi sesuai kondisi akademik terkini kamu.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    with c1:
        nama  = st.text_input("👤 Nama",
                               placeholder="Nama lengkap")
    with c2:
      prodi = st.selectbox(
        "📚 Program Studi",
        [
            "Pilih...",
            "Data Science",
            "Teknik Informatika",
            "Sistem Informasi",
            "Ilmu Komputer",
            "Statistika",
            "Lainnya"
        ]
    )

    if prodi == "Lainnya":
        prodi = st.text_input(
            "✍️ Masukkan Program Studi",
            placeholder="Contoh: Teknik Elektro"
        )

    with c3:
      univ = st.selectbox(
        "🏫 Universitas",
        [
            "Pilih...",
            "UGM",
            "ITB",
            "UI",
            "ITS",
            "Universitas Brawijaya",
            "Lainnya"
        ]
    )

    if univ == "Lainnya":
        univ = st.text_input(
            "✍️ Masukkan Nama Universitas",
            placeholder="Contoh: Universitas Negeri Malang"
        )

    c1,c2,c3,c4 = st.columns(4)
    with c1:
        semester = st.selectbox("📅 Semester",
                                 ["Pilih...",1,2,3,4,5,6,7,8])
    with c2:
        ipk = st.number_input("🎓 IPK",
                               min_value=0.0, max_value=4.0,
                               value=3.0, step=0.01)
    with c3:
        bahasa = st.selectbox("🌐 Bahasa Inggris",[
            "Pilih...","Basic","Intermediate","Advanced"
        ])
    with c4:
        frekuensi = st.selectbox("📖 Frekuensi Belajar",[
            "Pilih...","< 1 jam/hari","1–3 jam/hari",
            "3–5 jam/hari","> 5 jam/hari"
        ])

    st.markdown("<br>", unsafe_allow_html=True)

    # Skor asesmen
    st.markdown("""
    <div class="glass">
        <div style="font-weight:700; color:#58a6ff;
             margin-bottom:12px; font-size:15px;">
            📊 Skor Asesmen Skill (M1–M9)
        </div>
        <div class="box-info">
            💡 Geser slider sesuai skor asesmen kamu. (0–100)
        </div>
    </div>
    """, unsafe_allow_html=True)

    skor_vals = {}
    row1 = list(MODUL_LABELS.items())[:3]
    row2 = list(MODUL_LABELS.items())[3:6]
    row3 = list(MODUL_LABELS.items())[6:]

    for row in [row1, row2, row3]:
        cols = st.columns(len(row))
        for col,(key,(kode,nm,icon)) in zip(cols,row):
            with col:
                s = st.session_state.get(f's_{kode}',70)
                clr = skor_color(s)
                st.markdown(f"""
                <div class="glass" style="padding:14px;
                     text-align:center; margin-bottom:6px;
                     border-color:{clr}33;">
                    <div style="font-size:18px;">{icon}</div>
                    <div style="font-weight:600; font-size:13px;
                         color:{clr};">{kode} — {nm}</div>
                    <div style="font-size:11px;color:#8b949e;">
                         {skor_label(s)}</div>
                </div>
                """, unsafe_allow_html=True)
                skor_vals[key] = st.slider(
                    kode, 0, 100, 70,
                    key=f's_{kode}',
                    label_visibility="collapsed"
                )

    # Total skor
    avg_s = np.mean(list(skor_vals.values()))
    clr_a = skor_color(avg_s)
    st.markdown(f"""
    <div class="glass" style="text-align:center;
         border-color:{clr_a}55; padding:20px;">
        <div style="font-size:13px; color:#8b949e;">
            Rata-rata Total Skor
        </div>
        <div style="font-size:42px; font-weight:800;
             color:{clr_a};">{avg_s:.1f}
            <span style="font-size:18px;color:#8b949e;">
            /100</span>
        </div>
        <div class="prog-wrap" style="max-width:300px;
             margin:10px auto 0;">
            <div class="prog-fill"
                 style="width:{avg_s}%;background:{clr_a};">
            </div>
        </div>
        <div style="font-size:13px;color:{clr_a};margin-top:6px;">
            {skor_label(avg_s)}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1,c2,c3 = st.columns([1,2,1])
    with c2:
        btn = st.button("🚀 PREDIKSI & GENERATE LEARNING PATH",
                        use_container_width=True)

    if btn:
        if not nama:
            st.error("⚠️ Nama wajib diisi!")
        else:
            input_data = {
                'IPK'     : ipk,
                'Semester': int(semester)
                             if isinstance(semester,int)
                             else 7,
                **skor_vals
            }

            with st.spinner("🤖 Menganalisis profil kamu..."):
                time.sleep(1.5)
                v, conf, proba = predict_verdict(input_data)
                lp, prio       = generate_lp(skor_vals, v)

            st.session_state.update({
                'predicted'  : True,
                'verdict'    : v,
                'confidence' : conf,
                'proba'      : proba,
                'skor_vals'  : skor_vals,
                'learning_path': lp,
                'prioritas'  : prio,
                'nama'       : nama,
                'prodi'      : prodi,
                'ipk'        : ipk,
            })

            st.success("✅ Analisis selesai!")
            st.balloons()

            # Tampil hasil
            css,icon_v,clr_v = verdict_card_css(v)
            st.markdown(f"""
            <div class="{css}" style="margin:20px 0;">
                <div style="font-size:40px;">{icon_v}</div>
                <div style="font-size:26px; font-weight:800;
                     color:white; margin:10px 0;">
                     {v.upper()}</div>
                <div style="font-size:14px;
                     color:rgba(255,255,255,0.8);">
                     Tingkat Kesiapan Karir Kamu
                </div>
                <div style="margin:12px auto 0;
                     background:rgba(255,255,255,0.2);
                     border-radius:99px; height:8px;
                     max-width:260px;">
                    <div style="background:white; height:8px;
                         border-radius:99px;
                         width:{conf}%;"></div>
                </div>
                <div style="font-size:13px;
                     color:rgba(255,255,255,0.7);
                     margin-top:8px;">
                     Confidence: {conf:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="box-success">
                ✅ Learning path berhasil di-generate!
                Lihat di menu <b>🗺️ Learning Path</b>
                dan <b>📊 Gap Analysis</b>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# PAGE: GAP ANALYSIS
# ============================================================

elif page == "📊  Gap Analysis":

    st.markdown("""
    <div class="grad-title" style="font-size:26px;">
        📊 Gap Analysis
    </div>
    <p style="color:#8b949e; margin-bottom:24px;">
        Analisis kesenjangan antara skor kamu saat ini
        dengan target skor yang perlu dicapai.
    </p>
    """, unsafe_allow_html=True)

    if not st.session_state.get('predicted'):
        st.markdown("""
        <div class="glass" style="text-align:center;
             padding:60px 20px;">
            <div style="font-size:48px;">📊</div>
            <div style="font-size:18px; font-weight:700;
                 color:#58a6ff; margin-top:12px;">
                 Belum ada data analisis</div>
            <p style="color:#8b949e; margin-top:8px;">
                Lakukan prediksi di menu
                <b>🎯 Prediksi & Klasifikasi</b> terlebih dahulu.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        v      = st.session_state['verdict']
        sv     = st.session_state['skor_vals']
        prio   = st.session_state['prioritas']
        target = TARGET_SKOR.get(v, 70)
        css,icon_v,clr_v = verdict_card_css(v)

        # Summary
        c1,c2,c3 = st.columns(3)
        n_gap    = sum(1 for s in sv.values() if s < target)
        avg_gap  = np.mean([max(0,target-s)
                            for s in sv.values()])
        with c1:
            st.markdown(f"""
            <div class="metric">
                <div style="font-size:22px;">🎯</div>
                <div style="font-size:22px; font-weight:800;
                     color:{clr_v};">{v}</div>
                <div style="font-size:12px;color:#8b949e;">
                     Verdict Saat Ini</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric">
                <div style="font-size:22px;">⚠️</div>
                <div style="font-size:22px;font-weight:800;
                     color:#d4ac0d;">{n_gap} Modul</div>
                <div style="font-size:12px;color:#8b949e;">
                     Perlu Ditingkatkan</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric">
                <div style="font-size:22px;">📈</div>
                <div style="font-size:22px;font-weight:800;
                     color:#58a6ff;">{avg_gap:.1f} Poin</div>
                <div style="font-size:12px;color:#8b949e;">
                     Rata-rata Gap</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Gap per modul
        st.markdown("""
        <div style="font-weight:700; color:#58a6ff;
             margin-bottom:12px;">
             📋 Detail Gap per Modul
        </div>
        """, unsafe_allow_html=True)

        for key, skor in sv.items():
            kode,nm,icon = MODUL_LABELS[key]
            gap  = max(0, target - skor)
            pct  = min(100, skor)
            clr  = skor_color(skor)
            status = "✅ Tercapai" if gap == 0 else f"⚠️ Kurang {gap} poin"

            st.markdown(f"""
            <div class="tahap-card">
                <div style="display:flex;
                     justify-content:space-between;
                     align-items:center;">
                    <div>
                        <span style="font-size:16px;">{icon}</span>
                        <span style="font-weight:600;
                              color:#e6edf3; font-size:14px;
                              margin-left:8px;">
                              {kode} — {nm}</span>
                    </div>
                    <div style="text-align:right;">
                        <span style="font-size:14px;
                              font-weight:700; color:{clr};">
                              {skor}/100</span>
                        <span style="font-size:12px;
                              color:#8b949e; margin-left:8px;">
                              → Target: {target}</span>
                        <span style="font-size:12px;
                              color:{"#27ae60" if gap==0 else "#e74c3c"};
                              margin-left:8px;">{status}</span>
                    </div>
                </div>
                <div class="prog-wrap">
                    <div class="prog-fill"
                         style="width:{pct}%; background:{clr};">
                    </div>
                </div>
                <div style="display:flex;justify-content:space-between;
                     font-size:11px; color:#8b949e;">
                    <span>0</span>
                    <span style="color:{clr};">Skor: {skor}</span>
                    <span style="color:#58a6ff;">
                          Target: {target}</span>
                    <span>100</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Visualisasi
        c1,c2 = st.columns(2)

        with c1:
            # Bar chart gap
            labels  = [MODUL_LABELS[k][0] for k in sv]
            skors   = list(sv.values())
            gaps    = [max(0,target-s) for s in skors]
            colors  = [skor_color(s) for s in skors]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Skor Saat Ini',
                x=labels, y=skors,
                marker_color=colors,
                text=skors,
                textposition='outside',
                textfont=dict(color='white',size=10)
            ))
            fig.add_trace(go.Bar(
                name='Gap ke Target',
                x=labels, y=gaps,
                marker_color='rgba(231,76,60,0.4)',
                text=[f'+{g}' if g > 0 else '' for g in gaps],
                textposition='outside',
                textfont=dict(color='#e74c3c',size=10)
            ))
            fig.add_hline(y=target,
                          line_dash='dash',
                          line_color='#d4ac0d',
                          annotation_text=f'Target: {target}',
                          annotation_font_color='#d4ac0d')
            fig.update_layout(
                barmode='stack',
                title=dict(text='Skor vs Gap per Modul',
                           font=dict(color='white')),
                xaxis=dict(tickfont=dict(color='#8b949e')),
                yaxis=dict(range=[0,130],
                           tickfont=dict(color='#8b949e'),
                           gridcolor='rgba(255,255,255,0.05)'),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(255,255,255,0.03)',
                legend=dict(font=dict(color='white'),
                            bgcolor='rgba(0,0,0,0)'),
                height=340,
                margin=dict(t=40,b=10,l=10,r=10)
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            # Radar: saat ini vs target
            theta  = [f"{MODUL_LABELS[k][0]}"
                      for k in sv]
            r_now  = list(sv.values())
            r_tgt  = [target] * len(sv)
            theta.append(theta[0])
            r_now.append(r_now[0])
            r_tgt.append(r_tgt[0])

            fig2 = go.Figure()
            fig2.add_trace(go.Scatterpolar(
                r=r_tgt, theta=theta,
                fill='toself',
                fillcolor='rgba(88,166,255,0.1)',
                line=dict(color='#58a6ff',
                          width=2, dash='dash'),
                name=f'Target ({target})'
            ))
            fig2.add_trace(go.Scatterpolar(
                r=r_now, theta=theta,
                fill='toself',
                fillcolor='rgba(188,140,255,0.2)',
                line=dict(color='#bc8cff', width=2),
                name='Skor Kamu'
            ))
            fig2.update_layout(
                title=dict(text='Radar: Kamu vs Target',
                           font=dict(color='white')),
                polar=dict(
                    bgcolor='rgba(255,255,255,0.03)',
                    radialaxis=dict(
                        range=[0,100],
                        tickfont=dict(color='#8b949e',size=8),
                        gridcolor='rgba(255,255,255,0.06)'
                    ),
                    angularaxis=dict(
                        tickfont=dict(color='white',size=10)
                    )
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(font=dict(color='white'),
                            bgcolor='rgba(0,0,0,0)'),
                height=340,
                margin=dict(t=40,b=10,l=30,r=30)
            )
            st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# PAGE: LEARNING PATH
# ============================================================

elif page == "🗺️   Learning Path":

    st.markdown("""
    <div class="grad-title" style="font-size:26px;">
        🗺️ Learning Path Personal
    </div>
    <p style="color:#8b949e; margin-bottom:24px;">
        Rekomendasi kursus, sertifikasi, dan kegiatan
        yang diprioritaskan berdasarkan gap skor kamu.
    </p>
    """, unsafe_allow_html=True)

    if not st.session_state.get('predicted'):
        st.markdown("""
        <div class="glass" style="text-align:center;
             padding:60px 20px;">
            <div style="font-size:48px;">🗺️</div>
            <div style="font-size:18px;font-weight:700;
                 color:#58a6ff; margin-top:12px;">
                 Belum ada learning path</div>
            <p style="color:#8b949e; margin-top:8px;">
                Lakukan prediksi di menu
                <b>🎯 Prediksi & Klasifikasi</b> dulu.
            </p>
        </div>""", unsafe_allow_html=True)
    else:
        v   = st.session_state['verdict']
        lp  = st.session_state['learning_path']
        nm  = st.session_state.get('nama','Mahasiswa')
        ipk = st.session_state.get('ipk', 0)
        css,icon_v,clr_v = verdict_card_css(v)

        # Header
        st.markdown(f"""
        <div class="{css}" style="padding:20px;
             margin-bottom:20px;">
            <div style="font-size:20px; font-weight:700;
                 color:white;">{icon_v} Halo, {nm}!</div>
            <p style="color:rgba(255,255,255,0.8);
               font-size:14px; margin-top:6px;">
               Verdict kamu: <b>{v}</b> |
               IPK: <b>{ipk}</b> |
               {len(lp)} tahap learning path
            </p>
        </div>
        """, unsafe_allow_html=True)

        if not lp:
            st.markdown("""
            <div class="box-success">
                🎉 Semua modul sudah mencapai target!
                Tidak perlu learning path tambahan.
            </div>""", unsafe_allow_html=True)
        else:
            # Timeline visual
            st.markdown("""
            <div style="font-weight:700; color:#58a6ff;
                 margin-bottom:16px; font-size:15px;">
                 📅 Timeline Pengembangan
            </div>""", unsafe_allow_html=True)

            # Tampilkan sebagai timeline
            tl_cols = st.columns(min(len(lp), 5))
            for col, item in zip(tl_cols, lp):
                with col:
                    clr_s = skor_color(item['skor'])
                    st.markdown(f"""
                    <div class="glass" style="text-align:center;
                         padding:14px; border-color:{clr_s}33;">
                        <div style="background:linear-gradient(
                             135deg,#58a6ff,#bc8cff);
                             color:white; border-radius:50%;
                             width:28px; height:28px;
                             display:inline-flex;
                             align-items:center;
                             justify-content:center;
                             font-weight:700; font-size:13px;">
                             {item['tahap']}</div>
                        <div style="font-size:18px;
                             margin:6px 0;">
                             {item['icon']}</div>
                        <div style="font-weight:600;
                             font-size:12px; color:#e6edf3;">
                             {item['nama']}</div>
                        <div style="font-size:11px;
                             color:{clr_s}; margin-top:4px;">
                             {item['skor']} → {item['target']}
                             (+{item['gap']})</div>
                        <div style="font-size:10px;
                             color:#8b949e; margin-top:2px;">
                             ⏱️ {item['timeline']}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Detail tiap tahap
            st.markdown("""
            <div style="font-weight:700; color:#58a6ff;
                 margin-bottom:16px; font-size:15px;">
                 📚 Detail Rekomendasi per Tahap
            </div>""", unsafe_allow_html=True)

            for item in lp:
                clr_s = skor_color(item['skor'])
                with st.expander(
                    f"Tahap {item['tahap']} "
                    f"{item['icon']} {item['nama']} "
                    f"— Gap: {item['gap']} poin "
                    f"| Timeline: {item['timeline']}",
                    expanded=(item['tahap'] == 1)
                ):
                    c1,c2,c3 = st.columns(3)

                    with c1:
                        st.markdown("""
                        <div style="font-weight:600;
                             color:#58a6ff; font-size:13px;
                             margin-bottom:8px;">
                             📖 Kursus Online
                        </div>""", unsafe_allow_html=True)
                        if item['kursus']:
                            for k in item['kursus']:
                                st.markdown(f"""
                                <div class="badge-kursus">
                                    <b>{k['nama']}</b><br>
                                    🌐 {k['platform']} |
                                    ⏱️ {k['durasi']} |
                                    💰 {k['biaya']}
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.markdown(
                                '<div style="color:#8b949e;'
                                'font-size:12px;">'
                                'Tidak diperlukan</div>',
                                unsafe_allow_html=True
                            )

                    with c2:
                        st.markdown("""
                        <div style="font-weight:600;
                             color:#bc8cff; font-size:13px;
                             margin-bottom:8px;">
                             🏆 Sertifikasi
                        </div>""", unsafe_allow_html=True)
                        if item['sertif']:
                            for s in item['sertif']:
                                st.markdown(f"""
                                <div class="badge-sertif">
                                    <b>{s['nama']}</b><br>
                                    🌐 {s['platform']} |
                                    ⏱️ {s['durasi']} |
                                    💰 {s['biaya']}
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.markdown(
                                '<div style="color:#8b949e;'
                                'font-size:12px;">'
                                'Tidak diperlukan</div>',
                                unsafe_allow_html=True
                            )

                    with c3:
                        st.markdown("""
                        <div style="font-weight:600;
                             color:#ff7eb3; font-size:13px;
                             margin-bottom:8px;">
                             🎯 Kegiatan
                        </div>""", unsafe_allow_html=True)
                        for keg in item['kegiatan']:
                            st.markdown(f"""
                            <div class="badge-kegiatan">
                                {keg}
                            </div>""",
                            unsafe_allow_html=True)

            # Download
            st.markdown("<br>", unsafe_allow_html=True)
            report = f"""LEARNING PATH KESIAPAN KARIR
================================
Nama    : {nm}
Prodi   : {st.session_state.get('prodi','-')}
IPK     : {ipk}
Verdict : {v}

TAHAPAN LEARNING PATH
=====================
"""
            for item in lp:
                report += f"""
Tahap {item['tahap']}: {item['nama']} ({item['icon']})
  Skor Sekarang : {item['skor']}/100
  Target Skor   : {item['target']}/100
  Gap           : {item['gap']} poin
  Timeline      : {item['timeline']}
  Kursus        : {", ".join([k['nama'] for k in item['kursus']])}
  Sertifikasi   : {", ".join([s['nama'] for s in item['sertif']])}
  Kegiatan      :
"""
                for keg in item['kegiatan']:
                    report += f"  • {keg}\n"

            c1,c2,c3 = st.columns([1,2,1])
            with c2:
                st.download_button(
                    "📥 Download Learning Path (.txt)",
                    data=report,
                    file_name=f"LearningPath_{nm}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

# ============================================================
# PAGE: PREDIKSI MASSAL
# ============================================================

elif page == "📁  Prediksi Massal":

    st.markdown("""
    <div class="grad-title" style="font-size:26px;">
        📁 Prediksi Massal
    </div>
    <p style="color:#8b949e; margin-bottom:24px;">
        Upload file CSV/Excel berisi data banyak mahasiswa
        untuk prediksi & rekomendasi learning path massal.
    </p>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="box-warn">
        ⚠️ Pastikan file CSV memiliki kolom
        Skor M1–M9, IPK, dan data akademik lainnya.
    </div>""", unsafe_allow_html=True)

    up = st.file_uploader(
        "📤 Upload File CSV/Excel",
        type=['csv','xlsx','xls']
    )

    if up:
        try:
            if up.name.endswith('.csv'):
                df_up = pd.read_csv(up)
            else:
                df_up = pd.read_excel(up)

            st.success(f"✅ {df_up.shape[0]} data dimuat!")
            st.dataframe(df_up.head(), use_container_width=True)

            if st.button("🚀 Proses Semua Data",
                         use_container_width=True):
              
                with st.spinner("⏳ Memproses..."):
                  progress = st.progress(10)
                  df_pred = df_up.copy()
                  progress.progress(25)

                  # Load encoder sekali
                  try:
                      lab_encs = joblib.load(
                          "label_encoder_fitur.pkl"
                      )
                  except:
                      lab_encs = {}

                  # Encode kolom kategorikal
                  for col, col_enc in lab_encs.items():

                      if col in df_pred.columns:

                          try:
                              df_pred[col] = col_enc.transform(
                                  df_pred[col].astype(str)
                              )

                          except:
                              df_pred[col] = 0

                  progress.progress(50)

                  # Tambahkan fitur yang hilang
                  for col in feature_names:

                      if col not in df_pred.columns:
                          df_pred[col] = 0

                  df_pred = df_pred[feature_names]

                  # Boolean → Integer
                  for col in df_pred.select_dtypes(
                      include=["bool"]
                  ).columns:

                      df_pred[col] = df_pred[col].astype(int)

                  progress.progress(70)

                  # Scaling SEKALI
                  X = scaler.transform(df_pred)

                  progress.progress(85)

                  # Predict SEKALI
                  preds = model.predict(X)

                  # Predict proba SEKALI
                  probas = model.predict_proba(X)

                  progress.progress(95)

                  verdicts = le.inverse_transform(preds)

                  confs = (
                      probas.max(axis=1) * 100
                  ).round(2)

                  progress.progress(100)
                  progress.empty()

                df_up['Verdict']        = verdicts
                df_up['Confidence (%)'] = confs
                st.success("✅ Prediksi massal selesai!")

                dist  = pd.Series(verdicts).value_counts()
                clrs  = {
                    'Sangat Siap'        :'#27ae60',
                    'Siap Masuk Industri':'#2e86c1',
                    'Hampir Siap'        :'#d4ac0d',
                    'Perlu Persiapan'    :'#e74c3c'
                }

                c1,c2,c3,c4 = st.columns(4)
                for col,cat in zip(
                    [c1,c2,c3,c4], CATEGORIES[::-1]
                ):
                    cnt = dist.get(cat, 0)
                    pct = cnt/len(verdicts)*100
                    clr = clrs.get(cat,'#888')
                    with col:
                        st.markdown(f"""
                        <div class="metric" style="border-color:{clr}44;">
                            <div style="font-size:24px;
                                 font-weight:800;color:{clr};">
                                 {cnt}</div>
                            <div style="font-size:11px;
                                 color:#8b949e;">{cat}</div>
                            <div style="font-size:13px;
                                 color:{clr};">({pct:.1f}%)</div>
                        </div>""", unsafe_allow_html=True)

                fig = go.Figure(go.Pie(
                    labels=list(dist.index),
                    values=list(dist.values),
                    hole=0.4,
                    marker=dict(colors=[
                        clrs.get(c,'#888')
                        for c in dist.index
                    ]),
                    textinfo='label+percent',
                    textfont=dict(color='white',size=13)
                ))
                fig.update_layout(
                    title=dict(
                        text='Distribusi Verdict Massal',
                        font=dict(color='white')
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    legend=dict(
                        font=dict(color='white'),
                        bgcolor='rgba(0,0,0,0)'
                    ),
                    height=360
                )
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(df_up, use_container_width=True)
                st.download_button(
                    "📥 Download Hasil CSV",
                    df_up.to_csv(index=False).encode('utf-8'),
                    "hasil_prediksi_massal.csv",
                    "text/csv",
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"❌ Error: {e}")

# ============================================================
# PAGE: DASHBOARD DATA
# ============================================================

elif page == "📈  Dashboard Data":

    st.markdown("""
    <div class="grad-title" style="font-size:26px;">
        📈 Dashboard Analisis Data
    </div>
    <p style="color:#8b949e; margin-bottom:24px;">
        Visualisasi insight dari dataset kesiapan karir.
    </p>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="box-info">
        ℹ️ Menampilkan data demonstrasi.
    </div>""", unsafe_allow_html=True)

    np.random.seed(42)
    n = 1000
    cats = ['Perlu Persiapan','Hampir Siap',
            'Siap Masuk Industri','Sangat Siap']
    vd   = np.random.choice(cats, n,
                             p=[0.07,0.22,0.58,0.13])
    ipk  = np.random.normal(3.2,0.4,n).clip(2.0,4.0)
    sk   = {f'M{i}': np.random.normal(70,13,n).clip(0,100)
            for i in range(1,10)}
    df_d = pd.DataFrame({'Verdict':vd,'IPK':ipk,**sk})

    dist = pd.Series(vd).value_counts()
    clrs_v = {
        'Sangat Siap'        :'#27ae60',
        'Siap Masuk Industri':'#2e86c1',
        'Hampir Siap'        :'#d4ac0d',
        'Perlu Persiapan'    :'#e74c3c'
    }

    c1,c2,c3,c4,c5 = st.columns(5)
    for col,(lbl,val,clr) in zip(
        [c1,c2,c3,c4,c5],
        [('Total',n,'#58a6ff'),
         ('Sangat Siap',dist.get('Sangat Siap',0),'#27ae60'),
         ('Siap',dist.get('Siap Masuk Industri',0),'#2e86c1'),
         ('Hampir',dist.get('Hampir Siap',0),'#d4ac0d'),
         ('Perlu',dist.get('Perlu Persiapan',0),'#e74c3c')]
    ):
        with col:
            st.markdown(f"""
            <div class="metric">
                <div style="font-size:20px;font-weight:800;
                     color:{clr};">{val}</div>
                <div style="font-size:11px;color:#8b949e;">
                     {lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1,c2 = st.columns(2)

    with c1:
        fig = go.Figure(go.Bar(
            x=list(dist.index),
            y=list(dist.values),
            marker_color=[clrs_v.get(c,'#888')
                          for c in dist.index],
            text=list(dist.values),
            textposition='outside',
            textfont=dict(color='white',size=11)
        ))
        fig.update_layout(
            title=dict(text='Distribusi Verdict',
                       font=dict(color='white')),
            xaxis=dict(tickfont=dict(color='#8b949e')),
            yaxis=dict(tickfont=dict(color='#8b949e'),
                       gridcolor='rgba(255,255,255,0.05)'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(255,255,255,0.03)',
            height=320,
            margin=dict(t=40,b=10,l=10,r=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = go.Figure()
        for cat,clr in clrs_v.items():
            mask = df_d['Verdict'] == cat
            fig2.add_trace(go.Box(
                y=df_d[mask]['IPK'],
                name=cat,marker_color=clr,
                boxmean=True
            ))
        fig2.update_layout(
            title=dict(text='IPK per Kategori',
                       font=dict(color='white')),
            yaxis=dict(tickfont=dict(color='#8b949e'),
                       gridcolor='rgba(255,255,255,0.05)'),
            xaxis=dict(tickfont=dict(color='#8b949e',size=9)),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(255,255,255,0.03)',
            legend=dict(font=dict(color='white'),
                        bgcolor='rgba(0,0,0,0)'),
            height=320,
            margin=dict(t=40,b=10,l=10,r=10)
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Feature Importance Chart
    fi_data = sorted(FEATURE_IMP.items(),
                     key=lambda x: x[1])
    fig3 = go.Figure(go.Bar(
        x=[v for _,v in fi_data],
        y=[k.replace('Skor ','') for k,_ in fi_data],
        orientation='h',
        marker=dict(
            color=[v for _,v in fi_data],
            colorscale='Viridis'
        ),
        text=[f'{v:.4f}' for _,v in fi_data],
        textposition='outside',
        textfont=dict(color='white',size=10)
    ))
    fig3.update_layout(
        title=dict(
            text='Feature Importance (Tanpa Kolom Bocor)',
            font=dict(color='white',size=14)
        ),
        xaxis=dict(tickfont=dict(color='#8b949e')),
        yaxis=dict(tickfont=dict(color='white')),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.03)',
        height=380,
        margin=dict(t=50,b=10,l=10,r=80)
    )
    st.plotly_chart(fig3, use_container_width=True)

# ============================================================
# PAGE: PANDUAN
# ============================================================

elif page == "ℹ️   Panduan":

    st.markdown("""
    <div class="grad-title" style="font-size:26px;">
        ℹ️ Panduan Penggunaan
    </div><br>
    """, unsafe_allow_html=True)

    faqs = [
        ("🤔 Apa itu sistem ini?",
         "Sistem rekomendasi learning path yang mengklasifikasikan kesiapan karir lalu memberikan rekomendasi kursus, sertifikasi, dan kegiatan yang personal."),
        ("🔄 Apa bedanya dengan klasifikasi biasa?",
         "Sistem ini tidak hanya memberi verdict, tapi juga memberikan learning path lengkap berdasarkan gap analisis profil skor mahasiswa."),
        ("📊 Apa itu Gap Analysis?",
         "Analisis selisih antara skor kamu saat ini dengan target skor yang perlu dicapai untuk naik ke kategori verdict yang lebih tinggi."),
        ("🗺️ Apa itu Learning Path?",
         "Rekomendasi tahapan pengembangan diri berisi kursus online, sertifikasi, dan kegiatan yang diprioritaskan berdasarkan modul dengan gap terbesar."),
        ("💡 Bagaimana prioritas rekomendasi ditentukan?",
         "Berdasarkan kombinasi gap skor (makin besar gap makin prioritas) dan feature importance dari model ML (modul paling berpengaruh diprioritaskan)."),
        ("📱 Bisa diakses dari HP?",
         "Ya! Dashboard ini responsive dan bisa diakses dari semua perangkat termasuk smartphone dan tablet."),
    ]

    c1,c2 = st.columns(2)
    for i,(q,a) in enumerate(faqs):
        col = c1 if i%2==0 else c2
        with col:
            st.markdown(f"""
            <div class="glass">
                <div style="font-weight:700; font-size:14px;
                     color:#58a6ff; margin-bottom:8px;">{q}</div>
                <div style="font-size:13px; color:#8b949e;
                     line-height:1.6;">{a}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-weight:700; color:#58a6ff;
         margin-bottom:16px; font-size:16px;">
         📋 Penjelasan Modul M1–M9
    </div>""", unsafe_allow_html=True)

    moduls = [
        ('M1','Background','🏫','#7c3aed',
         'Latar belakang pendidikan & pengalaman dasar'),
        ('M2','Skills','💻','#0891b2',
         'Kemampuan teknis & hard skills'),
        ('M3','Industry','🏭','#059669',
         'Pemahaman industri & tren kerja'),
        ('M4','Interest','❤️','#dc2626',
         'Kejelasan minat & passion karir'),
        ('M5','Compass','🧭','#d97706',
         'Arah karir & rencana jangka panjang'),
        ('M6','Company','🏢','#7c3aed',
         'Pengetahuan tentang perusahaan & lingkungan kerja'),
        ('M7','Branding','✨','#0891b2',
         'Personal branding & komunikasi profesional'),
        ('M8','Ambisi','🎯','#059669',
         'Tingkat ambisi & motivasi berkembang'),
        ('M9','Resiliensi','💪','#dc2626',
         'Kemampuan menghadapi tekanan & adaptasi'),
    ]

    c1,c2,c3 = st.columns(3)
    for i,(kode,nm,icon,clr,desc) in enumerate(moduls):
        col = [c1,c2,c3][i%3]
        with col:
            st.markdown(f"""
            <div class="glass"
                 style="border-top:3px solid {clr};">
                <div style="font-size:18px;">{icon}</div>
                <div style="font-weight:800; font-size:16px;
                     color:{clr};">{kode}</div>
                <div style="font-weight:600; font-size:13px;
                     color:#e6edf3; margin:4px 0;">{nm}</div>
                <div style="font-size:12px; color:#8b949e;
                     line-height:1.5;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("""
    <hr>
    <div style="text-align:center; color:#484f58;
         font-size:12px; padding:16px 0;">
        🚀 Sistem Rekomendasi Learning Path Kesiapan Karir<br>
        Asri Putri Lestari — Proyek Akhir © 2026<br>
        Powered by Machine Learning + Content-Based Filtering
    </div>""", unsafe_allow_html=True)
