
import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np

import re
import math
import jellyfish

kata_kunci_mencurigakan = [
    'login', 'verify', 'secure', 'account', 'update', 'confirm',
    'signin', 'password', 'bank', 'wallet', 'billing', 'suspend',
    'unlock', 'security', 'support', 'alert', 'validation', 'auth'
]
tld_umum = {'com', 'co.id', 'go.id', 'id', 'net', 'org', 'ac.id'}
skor_abuse_tld = {
    'xyz': 0.9, 'top': 0.85, 'tk': 0.9, 'club': 0.7, 'info': 0.6,
    'online': 0.65, 'site': 0.6, 'cf': 0.85, 'gq': 0.85, 'ml': 0.85,
    'buzz': 0.75, 'work': 0.6, 'icu': 0.8, 'live': 0.5, 'app': 0.3,
}
karakter_homoglyph = {'0', '1', '3', '4', '5', '7', '@'}

daftar_brand_resmi_full = [
    'apple.com', 'pajak.go.id', 'microsoft.com', 'amazon.com', 'bca.co.id',
    'bankmandiri.co.id', 'gojek.com', 'tokopedia.com', 'shopee.co.id',
    'traveloka.com', 'dana.id', 'google.com', 'facebook.com'
]

AMBANG_SIMILARITY = 0.7

def extract_sld_tld(domain: str):
    domain = str(domain).strip().lower().rstrip('.')
    parts = domain.split('.')
    if len(parts) >= 3 and parts[-2] in ('co', 'go', 'ac'):
        sld, tld = parts[-3], f"{parts[-2]}.{parts[-1]}"
    elif len(parts) >= 2:
        sld, tld = parts[-2], parts[-1]
    else:
        sld, tld = parts[0], ''
    return sld, tld

daftar_brand_key = [extract_sld_tld(b)[0] for b in daftar_brand_resmi_full]

def hitung_entropy(s: str) -> float:
    if not s:
        return 0.0
    prob = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in prob)

def hitung_homoglyph(sld: str) -> int:
    return sum(1 for ch in sld if ch in karakter_homoglyph)

def hitung_similarity_ke_brand(sld: str, brand_key: str) -> dict:
    return {
        'levenshtein_distance': jellyfish.levenshtein_distance(sld, brand_key),
        'damerau_levenshtein_distance': jellyfish.damerau_levenshtein_distance(sld, brand_key),
        'jaro_winkler_score': jellyfish.jaro_winkler_similarity(sld, brand_key),
    }

def cari_brand_paling_mirip(sld: str) -> dict:
    hasil_terbaik = {'levenshtein_distance': 99, 'damerau_levenshtein_distance': 99,
                      'jaro_winkler_score': 0.0}
    for brand_key in daftar_brand_key:
        kandidat = hitung_similarity_ke_brand(sld, brand_key)
        if kandidat['jaro_winkler_score'] > hasil_terbaik['jaro_winkler_score']:
            hasil_terbaik = kandidat
    return hasil_terbaik

def extract_features(domain: str) -> dict:
    domain = str(domain).strip().lower()
    sld, tld = extract_sld_tld(domain)

    sim = cari_brand_paling_mirip(sld)

    return {
        'domain': domain,
        'digit_count': sum(c.isdigit() for c in sld),
        'hyphen_count': sld.count('-'),
        'entropy': hitung_entropy(sld),
        'homoglyph_count': hitung_homoglyph(sld),
        'levenshtein_distance': sim['levenshtein_distance'],
        'damerau_levenshtein_distance': sim['damerau_levenshtein_distance'],
        'jaro_winkler_score': sim['jaro_winkler_score'],
        'suspicious_keywords': sum(1 for kw in kata_kunci_mencurigakan if kw in sld),
        'is_common_tld': 1 if tld in tld_umum else 0,
        'tld_abuse_score': skor_abuse_tld.get(tld, 0.2),
    }


# Set Page Config
st.set_page_config(page_title="Typosquatting Detector", layout="wide")

# Load Model
@st.cache_resource
def load_model():
    return joblib.load("best_lightgbm_model_70_30.pkl")

model = load_model()
FEATURE_NAMES = ['digit_count', 'hyphen_count', 'entropy', 'homoglyph_count', 'levenshtein_distance', 'damerau_levenshtein_distance', 'jaro_winkler_score', 'suspicious_keywords', 'is_common_tld', 'tld_abuse_score']

# Sidebar for Documentation
with st.sidebar:
    st.title("ℹ️ Tentang Aplikasi")
    st.info("""
    Aplikasi ini menggunakan **Machine Learning** untuk mendeteksi domain phishing.

    **Apa itu Typosquatting?**
    Teknik penipuan dengan mendaftarkan domain yang mirip dengan brand asli (contoh: `g00gle.com` vs `google.com`).
    """)
    st.subheader("Cara Membaca XAI:")
    st.write("🔴 **Merah**: Mendorong ke arah Phishing")
    st.write("🔵 **Biru**: Mendorong ke arah Aman")

# Main UI
st.title("🛡️ Typosquatting Domain Detector")
st.write("Sistem deteksi cerdas berbasis AI untuk mengidentifikasi domain berbahaya.")

domain_input = st.text_input("Masukkan Nama Domain (Contoh: google-login.xyz)", "google-login.xyz")

if st.button("Cek Keamanan Domain", type="primary"):
    if domain_input:
        features_dict = extract_features(domain_input)
        features_df = pd.DataFrame([{k: features_dict[k] for k in FEATURE_NAMES}])

        prediction = model.predict(features_df)[0]
        prob = model.predict_proba(features_df)[0]

        # Hasil Utama dalam Kolom
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Hasil Prediksi")
            if prediction == 1:
                st.error("### 🚨 TERDETEKSI PHISHING")
            else:
                st.success("### ✅ TERDETEKSI AMAN")

            # Gunakan format string manual untuk menghindari konflik f-string Colab
            conf_score = prob[1] * 100
            st.metric("Skor Kepercayaan Phishing", f"{conf_score:.1f}%")

        with col2:
            st.subheader("🔍 Interpretasi Fitur (XAI)")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(features_df)

            if isinstance(shap_values, list): val = shap_values[1]
            else: val = shap_values

            fig, ax = plt.subplots()
            shap.plots.bar(shap.Explanation(
                values=val[0] if len(val.shape) > 1 else val,
                base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value,
                data=features_df.iloc[0],
                feature_names=FEATURE_NAMES
            ))
            st.pyplot(fig)

        # Detail Tambahan
        with st.expander("📊 Lihat Detail Teknis (Fitur Ter-ekstraksi)"):
            st.dataframe(features_df.T.rename(columns={0: 'Nilai'}), use_container_width=True)
