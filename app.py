# app.py
import streamlit as st
import pandas as pd
import numpy as np
import json, shap, matplotlib.pyplot as plt
import skfuzzy as fuzz
import plotly.express as px
from skfuzzy import control as ctrl
from xgboost import XGBRegressor

# ── evaluasi & split
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

# ─────────────────────────────────────────────
# 1. Konfigurasi halaman
# ─────────────────────────────────────────────
st.set_page_config(page_title="AI Disabilitas Dashboard", layout="wide")

# ─────────────────────────────────────────────
# 2. Input data JSON
# ─────────────────────────────────────────────
st.sidebar.header("Upload / Path")
data_src = st.sidebar.radio("Sumber data", ["Upload JSON", "Path lokal"])
if data_src == "Upload JSON":
    up_file = st.sidebar.file_uploader("disabilitas.json", type="json")
    if up_file:
        raw = json.load(up_file)
else:
    json_path = st.sidebar.text_input("Path file JSON", "disabilitas.json")
    if json_path:
        raw = json.load(open(json_path))

if "raw" not in locals():
    st.info("Unggah / ketik path dulu.")
    st.stop()

# ─────────────────────────────────────────────
# 3. Pre‑processing
# ─────────────────────────────────────────────
df = (
    pd.DataFrame(raw["data"])
    .rename(
        columns={
            "nama_item__kategori_1": "umur",
            "nama_item__kategori_2": "area",
            "nama_item__kategori_3": "gender",
            "nama_item__kategori_4": "jenis_dis",
            "nilai": "nilai",
        }
    )
)
df_tot = df[df["jenis_dis"].str.strip() == "Total"].copy()

# ─────────────────────────────────────────────
# 4. Eksplorasi Data
# ─────────────────────────────────────────────
st.header("📊 Eksplorasi Data")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Treemap Umur × Gender × Area (opsional)")
with col2:
    st.subheader("Korelasi Gender vs Umur")
    corr = df_tot.pivot_table(
        index="umur", columns="gender", values="nilai", aggfunc="sum"
    )
    st.dataframe(corr)

# ─────────────────────────────────────────────
# 5. Fuzzy + Machine Learning
# ─────────────────────────────────────────────
st.header("🤖 Prioritas & Teknologi Bantu")

# 5‑A. Siapkan pivot
pivot = (
    df_tot.groupby(["umur", "gender", "area"])["nilai"]
    .sum()
    .reset_index(name="total_dis")
)
pivot["gender_idx"] = pivot["gender"].map({"Laki-laki": 0, "Perempuan": 1})
pivot = pivot[
    (pivot["umur"] != "Total")
    & (pivot["gender"] != "Total")
    & (pivot["area"] != "Total")
]

# 5‑B. Fuzzy system
low_mid, mid_high = pivot["total_dis"].quantile([0.33, 0.66]).astype(int)
total_max = int(pivot["total_dis"].max() * 1.1)
step = max(100, int(total_max / 10))

total = ctrl.Antecedent(np.arange(0, total_max + 1, step), "total_dis")
mob = ctrl.Antecedent(np.arange(0, 101, 1), "mob")
gender = ctrl.Antecedent(np.arange(0, 2, 1), "gender")
prio = ctrl.Consequent(np.arange(0, 11, 1), "priority")

total["rendah"] = fuzz.trapmf(
    total.universe, [0, 0, 0.8 * low_mid, 1.2 * low_mid]
)
total["sedang"] = fuzz.trimf(
    total.universe, [low_mid, (low_mid + mid_high) // 2, mid_high]
)
total["tinggi"] = fuzz.trapmf(
    total.universe, [0.8 * mid_high, 1.2 * mid_high, total_max, total_max]
)

mob["kecil"] = fuzz.trimf(mob.universe, [0, 0, 30])
mob["sedang"] = fuzz.trimf(mob.universe, [20, 50, 80])
mob["besar"] = fuzz.trimf(mob.universe, [60, 100, 100])

gender["laki"] = fuzz.trapmf(gender.universe, [-0.1, 0, 0, 0.1])
gender["perempuan"] = fuzz.trapmf(gender.universe, [0.9, 1, 1, 1.1])

prio["rendah"] = fuzz.trimf(prio.universe, [0, 0, 4])
prio["sedang"] = fuzz.trimf(prio.universe, [3, 5, 7])
prio["tinggi"] = fuzz.trimf(prio.universe, [6, 10, 10])

rules = [
    ctrl.Rule(total["tinggi"] | mob["besar"], prio["tinggi"]),
    ctrl.Rule(total["sedang"] & mob["sedang"], prio["sedang"]),
    ctrl.Rule(gender["perempuan"] & total["tinggi"], prio["tinggi"]),
    ctrl.Rule(total["rendah"] & mob["kecil"], prio["rendah"]),
]
fis_system = ctrl.ControlSystem(rules)

# 5‑C. Machine‑learning data set
df_ml = pivot.copy()
df_ml["umur_ord"] = pd.Categorical(df_ml["umur"]).codes
df_ml = pd.get_dummies(df_ml, columns=["area"], drop_first=True)

X = (
    df_ml.drop(columns=["total_dis", "umur", "gender"])
    .apply(pd.to_numeric, errors="coerce")
    .fillna(0)
)
y = df_ml["total_dis"]

# Split train / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
ml_model = XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42)
ml_model.fit(X_train, y_train)

# ── evaluasi
y_pred = ml_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
r2 = r2_score(y_test, y_pred)

# ─────────────────────────────────────────────
# 6. Hybrid Prediction
# ─────────────────────────────────────────────
alpha = st.sidebar.slider("Bobot Fuzzy ↔ ML", 0.0, 1.0, 0.6, 0.05)

out_rows = []
for _, r in pivot.iterrows():
    sim = ctrl.ControlSystemSimulation(fis_system)
    sim.input["total_dis"] = r["total_dis"]

    gender_total = pivot[pivot["gender_idx"] == r["gender_idx"]][
        "total_dis"
    ].sum()
    gender_total = max(gender_total, 1)

    sim.input["mob"] = 100 * r["total_dis"] / gender_total
    sim.input["gender"] = r["gender_idx"]
    sim.compute()
    fuzzy_score = sim.output["priority"]

    X_row = df_ml.loc[_].drop(["total_dis", "umur", "gender"])
    X_row = pd.to_numeric(X_row, errors="coerce").fillna(0)
    ml_score = ml_model.predict(X_row.to_frame().T)[0] / total_max * 10

    final_score = alpha * fuzzy_score + (1 - alpha) * ml_score
    tech = (
        "Kursi roda & ramp"
        if final_score >= 7
        else "Ramp manual"
        if final_score >= 4
        else "Tongkat pemandu"
    )
    out_rows.append(
        {
            "Umur": r["umur"],
            "Gender": r["gender"],
            "Area": r["area"],
            "Total_Dis": int(r["total_dis"]),
            "Priority": round(final_score, 2),
            "Rekom_Teknologi": tech,
        }
    )

out_df = pd.DataFrame(out_rows).sort_values("Priority", ascending=False)

# ─────────────────────────────────────────────
# 7. Output Data & Form Uji Manual
# ─────────────────────────────────────────────
st.subheader("📋 Daftar Prioritas (Fuzzy + ML)")
st.dataframe(
    out_df.style.background_gradient(subset=["Priority"], cmap="YlOrRd"),
    height=400,
)

with st.expander("🔧 Input Manual Uji Teknologi Bantu"):
    input_umur = st.selectbox("Umur", sorted(df_ml["umur"].unique()))
    input_gender = st.selectbox("Gender", ["Laki-laki", "Perempuan"])
    input_area = st.selectbox("Area", ["Pedesaan", "Perkotaan"])
    input_total = st.number_input(
        "Total Penyandang Disabilitas",
        min_value=0,
        max_value=total_max,
        step=100,
    )

    if st.button("Jalankan Fuzzy + ML"):
        gender_idx = 0 if input_gender == "Laki-laki" else 1
        # ── Siapkan fitur ML
        input_row = {
            "umur_ord": pd.Categorical(
                [input_umur], categories=df_ml["umur"].unique()
            ).codes[0],
            "gender_idx": gender_idx,
        }
        for col in X.columns:
            if col.startswith("area_"):
                input_row[col] = 1 if col == f"area_{input_area}" else 0
        input_df = pd.DataFrame([input_row], columns=X.columns)

        ml_score = ml_model.predict(input_df)[0] / total_max * 10

        # ── Fuzzy
        sim = ctrl.ControlSystemSimulation(fis_system)
        gender_total = pivot[pivot["gender_idx"] == gender_idx][
            "total_dis"
        ].sum()
        gender_total = max(gender_total, 1)
        sim.input["total_dis"] = input_total
        sim.input["mob"] = 100 * input_total / gender_total
        sim.input["gender"] = gender_idx
        sim.compute()
        fuzzy_score = sim.output["priority"]

        final_score = alpha * fuzzy_score + (1 - alpha) * ml_score
        tech = (
            "Kursi roda & ramp"
            if final_score >= 7
            else "Ramp manual"
            if final_score >= 4
            else "Tongkat pemandu"
        )
        st.success(
            f"🎯 Skor Prioritas: {round(final_score,2)} — Rekomendasi: **{tech}**"
        )

# ─────────────────────────────────────────────
# 8. Tambahan Visual & Interpretasi
# ─────────────────────────────────────────────
if st.checkbox("Tampilkan SHAP Feature Importance"):
    st.write(
        "🔍 SHAP menunjukkan fitur mana yang paling memengaruhi prediksi model ML."
    )
    explainer = shap.Explainer(ml_model)
    shap_vals = explainer(X)
    fig = plt.figure()
    shap.summary_plot(shap_vals, X, plot_type="bar", show=False)
    st.pyplot(fig)

with st.expander("🌲 Treemap Total Disabilitas"):
    fig_tm = px.treemap(
        pivot,
        path=["umur", "gender", "area"],
        values="total_dis",
        color="total_dis",
        color_continuous_scale="RdBu",
        title="Treemap: Umur × Gender × Area",
    )
    st.plotly_chart(fig_tm, use_container_width=True)

with st.expander("🔍 Visualisasi Hasil Fuzzy (Defuzzifikasi)"):
    fuzzy_results = []
    sim = ctrl.ControlSystemSimulation(fis_system)
    for total_ in range(0, total_max + 1, 1000):
        sim.input["total_dis"] = total_
        sim.input["mob"] = 100 * total_ / gender_total
        sim.input["gender"] = 0  # asumsi laki‑laki
        sim.compute()
        fuzzy_results.append((total_, sim.output["priority"]))

    df_fuzzy_result = pd.DataFrame(
        fuzzy_results, columns=["Total_Dis", "Priority_Score"]
    )
    st.line_chart(df_fuzzy_result.set_index("Total_Dis"))

    fig_fuzzy = px.line(
        df_fuzzy_result,
        x="Total_Dis",
        y="Priority_Score",
        title="Defuzzifikasi: Hubungan Total Disabilitas ↔ Skor Prioritas",
        markers=True,
    )
    st.plotly_chart(fig_fuzzy, use_container_width=True)

# ─────────────────────────────────────────────
# 9. Evaluasi Model (sidebar + caption)
# ─────────────────────────────────────────────
st.sidebar.header("📊 Evaluasi Model XGBoost")

st.sidebar.metric("MAPE", f"{mape:.2f}%")



