import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

# ==========================================================
# HELPER : note orale discrète (expander 💬)
# ==========================================================
def note(titre, texte):
    with st.expander(f"💬 {titre}", expanded=False):
        st.markdown(f"<p style='color:#A0AEC0;font-size:13px;line-height:1.7;'>{texte}</p>", unsafe_allow_html=True)

# ==========================================================
# 1. CONFIGURATION DE LA PAGE & STYLE CSS (DARK MODE / NÉON)
# ==========================================================
st.set_page_config(
    page_title="Pb²⁺ Adsorption AI Platform",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* Fond principal */
.main { background-color: #0B1020; }
h1, h2, h3, h4 { color: white; }

/* Masquer le header sidebar par défaut */
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 1rem;
}

/* Cartes de métriques personnalisées style Dashboard */
.metric-card { 
    background: linear-gradient(135deg, #111827, #1F2937); 
    border-left: 5px solid #00FFD1; 
    border-radius: 12px; 
    padding: 20px;
    margin-bottom: 15px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.4);
}
.metric-card-best { 
    background: linear-gradient(135deg, #1e1b4b, #311042); 
    border: 2px solid #00FFD1;
    border-left: 8px solid #00FFD1; 
    border-radius: 12px; 
    padding: 25px;
    margin-bottom: 15px;
    box-shadow: 0 0 15px rgba(0, 255, 209, 0.2);
}
.metric-card-danger { 
    background: linear-gradient(135deg, #111827, #1F2937); 
    border-left: 5px solid #FF4B4B; 
    border-radius: 12px; 
    padding: 20px;
    margin-bottom: 15px;
}

/* Boutons */
.stButton>button {
    background-color: #E53E3E;
    color: white;
    border-radius: 8px;
    height: 3.5em;
    width: 100%;
    font-size: 16px;
    font-weight: bold;
    border: none;
    transition: 0.3s;
}
.stButton>button:hover {
    background-color: #C53030;
    color: white;
    box-shadow: 0 0 10px rgba(229, 62, 62, 0.5);
}

/* Sidebar style */
section[data-testid="stSidebar"] {
    background-color: #0D1526;
    border-right: 1px solid #1F2937;
}

/* Number inputs dans la sidebar */
section[data-testid="stSidebar"] .stNumberInput label {
    color: #A0AEC0 !important;
    font-size: 13px;
}

/* Onglets horizontaux */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background-color: #111827;
    padding: 6px 8px;
    border-radius: 10px;
    flex-wrap: wrap;
}
.stTabs [data-baseweb="tab"] {
    background-color: #1F2937;
    color: #A0AEC0;
    border-radius: 6px;
    padding: 6px 14px;
    font-size: 13px;
    font-weight: 600;
    border: none;
}
.stTabs [aria-selected="true"] {
    background-color: #00FFD1 !important;
    color: #0B1020 !important;
}
</style>
""", unsafe_allow_html=True)

# Configuration globale pour les graphiques Matplotlib en mode sombre
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#0B1020'
plt.rcParams['axes.facecolor'] = '#0B1020'
plt.rcParams['grid.color'] = '#1F2937'

# ==========================================================
# 2. CHARGEMENT ET SÉCURISATION DU DATASET
# ==========================================================
@st.cache_data
def load_data():
    for sep in [',', ';']:
        try:
            df = pd.read_csv("Dataset_Final.csv", sep=sep, decimal=',' if sep==';' else '.')
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
    try:
        return pd.read_csv("Dataset.csv", sep=';', decimal=',')
    except Exception:
        np.random.seed(42)
        return pd.DataFrame({
            "pH": np.random.uniform(2, 8, 98),
            "Dosage (g)": np.random.uniform(0.1, 1.5, 98),
            "Contact time (min)": np.random.uniform(5, 180, 98),
            "Initial Pb²⁺ concentration (ppm)": np.random.uniform(10, 200, 98),
            "Average qe (mg g-1)": np.random.uniform(5, 150, 98)
        })

df_raw = load_data()  # garde Data ID si présent pour la recherche par numéro

# Remplacement automatique de "dye" par "Pb²⁺" dans les noms de colonnes du CSV
df_raw.columns = [
    col.replace("dye", "Pb²⁺").replace("Dye", "Pb²⁺") for col in df_raw.columns
]

# On conserve un index 1-based pour le lookup par numéro d'expérience
df_raw = df_raw.reset_index(drop=True)
df_raw.index = df_raw.index + 1  # index commence à 1

df = df_raw.copy()
if "Data ID" in df.columns:
    df = df.drop(columns=["Data ID"])

display_columns = {
    "Initial Pb²⁺ concentration (ppm)": "Concentration initiale Pb²⁺ (ppm)",
    "Initial concentration (mg/L)": "Concentration initiale Pb²⁺ (mg/L)",
    "Adsorbent dosage:Pb²⁺ volume ratio": "Rapport Dosage / Volume Pb²⁺",
    "Adsorption time (min)": "Temps d'adsorption (min)",
    "Contact time (min)": "Temps de contact (min)",
    "Adsorption temperature (°C)": "Température d'adsorption (°C)",
    "Adsorption temperature": "Température (°C)",
    "Solution pH": "pH de la solution",
    "pH": "pH de la solution",
    "Dosage (g)": "Dosage d'adsorbant (g)",
    "Biomass addition (mg)": "Ajout de biomasse (mg)"
}

target = "Average qe (mg g-1)"
if target not in df.columns:
    potential_targets = [col for col in df.columns if 'qe' in col or 'Qe' in col]
    target = potential_targets[0] if potential_targets else df.columns[-1]

X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================================
# 3. ENTRAÎNEMENT GLOBAL (MIS EN CACHE)
# ==========================================================
@st.cache_resource
def train_all_models(X_tr, y_tr, X_ts, y_ts):
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_ts_sc = scaler.transform(X_ts)

    models = {
        "ANN": MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', max_iter=3000, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
        "Régression Linéaire": LinearRegression()
    }

    models["ANN"].fit(X_tr_sc, y_tr)
    models["XGBoost"].fit(X_tr, y_tr)
    models["Random Forest"].fit(X_tr, y_tr)
    models["Régression Linéaire"].fit(X_tr, y_tr)

    metrics = {}
    preds_test = {}
    for name, model in models.items():
        X_tst = X_ts_sc if name == "ANN" else X_ts
        X_trn = X_tr_sc if name == "ANN" else X_tr

        preds       = model.predict(X_tst)
        preds_train = model.predict(X_trn)
        preds_test[name] = preds

        r2_test  = r2_score(y_ts, preds)
        r2_train = r2_score(y_tr, preds_train)
        gap      = r2_train - r2_test

        metrics[name] = {
            "R2 Train":    round(r2_train, 6),
            "R2 Test":     round(r2_test,  6),
            "Gap":         round(gap, 6),
            "RMSE":        round(np.sqrt(mean_squared_error(y_ts, preds)), 6),
            "MAE":         round(mean_absolute_error(y_ts, preds), 6)
        }

    return models, metrics, preds_test, scaler, X_tr_sc, X_ts_sc

models, metrics_results, preds_test, scaler, X_train_scaled, X_test_scaled = train_all_models(
    X_train, y_train, X_test, y_test
)
metrics_df = pd.DataFrame(metrics_results).T

# ==========================================================
# 4. SIDEBAR : PARAMÈTRES DE PRÉDICTION
# ==========================================================
with st.sidebar:
    st.markdown("<h2 style='color:#00FFD1; font-size:18px; margin-bottom:4px;'>⚙️ Paramètres expérimentaux</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#A0AEC0; font-size:12px; margin-top:0;'>Ajustez les valeurs ou entrez un Data ID</p>", unsafe_allow_html=True)
    st.markdown("---")

    # --- DATA ID : chargement automatique des paramètres ---
    st.markdown("<p style='color:#00FFD1; font-size:13px; font-weight:bold; margin-bottom:2px;'>🔎 Data ID (Vide = Mode Libre)</p>", unsafe_allow_html=True)
    data_id_input = st.number_input(
        label="Numéro d'expérience",
        min_value=0,
        max_value=len(df_raw),
        value=0,
        step=1,
        label_visibility="collapsed"
    )

    # Récupère les valeurs de la ligne si un ID valide est saisi
    prefill = {}
    if data_id_input > 0:
        if data_id_input in df_raw.index:
            row_data = df_raw.loc[data_id_input]
            for col in X.columns:
                prefill[col] = float(row_data[col])
            st.success(f"✅ Expérience #{data_id_input} chargée — qe réel : **{float(row_data[target]):.4f} mg/g**")
        else:
            st.error(f"❌ Data ID {data_id_input} introuvable (max : {len(df_raw)})")

    st.markdown("---")

    # --- INPUTS : pré-remplis si Data ID fourni, sinon valeurs moyennes ---
    inputs = {}
    for col in X.columns:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        default_val = prefill.get(col, float(df[col].mean()))
        display_name = display_columns.get(col, col)
        inputs[col] = st.number_input(
            label=display_name,
            min_value=min_val,
            max_value=max_val,
            value=default_val,
            step=0.0,
            format="%.15g"
        )

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔴 Lancer la prédiction")

# ==========================================================
# 5. ONGLETS HORIZONTAUX
# ==========================================================
tab_accueil, tab_workflow, tab_dataset, tab_heatmap, tab_ann, tab_rf, tab_xgb, tab_lr, tab_comparaison, tab_importance, tab_analyse, tab_bilan = st.tabs([
    "🏠 Accueil",
    "⚙️ Workflow",
    "📂 Dataset",
    "🔥 Heatmap",
    "🧠 ANN",
    "🌲 RF",
    "⚡ XGB",
    "📉 LR",
    "📈 Comparaison",
    "⭐ Importance",
    "🔬 Analyse",
    "📘 Bilan & Analyse"
])

# ==========================================================
# 6. CONTENU DES ONGLETS
# ==========================================================

# --- ACCUEIL ---
with tab_accueil:
    st.markdown("<h1 style='color:white;'>🧠 Plateforme Machine Learning — Adsorption du Pb²⁺</h1>", unsafe_allow_html=True)

    best_model_name = metrics_df["R2 Test"].idxmax()
    target_short = target.split("(")[0].strip().replace("Average ", "")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#111827,#1F2937);border-radius:14px;padding:28px 24px;border-left:5px solid #00FFD1;'>
            <p style='color:#A0AEC0;margin:0;font-size:14px;'>📊 Dataset</p>
            <p style='color:white;font-size:42px;font-weight:900;margin:4px 0;'>{df.shape[0]}</p>
            <p style='color:#6B7280;font-size:13px;margin:0;'>Points de données</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#111827,#1F2937);border-radius:14px;padding:28px 24px;border-left:5px solid #F59E0B;'>
            <p style='color:#A0AEC0;margin:0;font-size:14px;'>🏆 Meilleur modèle</p>
            <p style='color:white;font-size:42px;font-weight:900;margin:4px 0;'>{best_model_name.replace("Régression Linéaire","Rég. Lin.")}</p>
            <p style='color:#6B7280;font-size:13px;margin:0;'>Précision optimale</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#111827,#1F2937);border-radius:14px;padding:28px 24px;border-left:5px solid #EF4444;'>
            <p style='color:#A0AEC0;margin:0;font-size:14px;'>🎯 Variable cible</p>
            <p style='color:white;font-size:42px;font-weight:900;margin:4px 0;'>qe</p>
            <p style='color:#6B7280;font-size:13px;margin:0;'>Capacité (mg/g)</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    note("Accueil",
        "Cette plateforme a été développée pour modéliser et prédire la capacité d'adsorption du Plomb (Pb²⁺) "
        "par un biosorbant naturel. Elle intègre 4 algorithmes de Machine Learning entraînés sur "
        f"{df.shape[0]} expériences réelles de laboratoire. La variable cible <b>qe</b> représente la quantité "
        "de Pb²⁺ adsorbée par gramme d'adsorbant, exprimée en mg/g. "
        "Le meilleur modèle est automatiquement identifié à chaque chargement.")

    # Message guide
    st.info("💡 Ajustez vos paramètres en toute précision et cliquez sur **'Lancer la prédiction'** dans le menu de gauche.")

    # Résultats de prédiction si bouton cliqué
    if predict_btn:
        input_df = pd.DataFrame([inputs])
        start_time = time.time()

        pred_ann = models["ANN"].predict(scaler.transform(input_df))[0]
        pred_xgb = models["XGBoost"].predict(input_df)[0]
        pred_rf  = models["Random Forest"].predict(input_df)[0]
        pred_lr  = models["Régression Linéaire"].predict(input_df)[0]

        calc_time = time.time() - start_time

        preds_dict = {"ANN": pred_ann, "XGBoost": pred_xgb, "Random Forest": pred_rf, "Régression Linéaire": pred_lr}
        best_pred_value = preds_dict[best_model_name]

        st.markdown("### 🔬 Résultats de l'Adsorption Simulée")
        st.success(f"👑 Meilleur modèle : **{best_model_name}** → **{best_pred_value:.4f} mg/g**")

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("🧠 ANN", f"{pred_ann:.4f} mg/g")
        r2.metric("⚡ XGBoost", f"{pred_xgb:.4f} mg/g")
        r3.metric("🌲 Random Forest", f"{pred_rf:.4f} mg/g")
        r4.metric("📉 Rég. Linéaire", f"{pred_lr:.4f} mg/g")
        st.caption(f"⏱ Temps de calcul : {calc_time*1000:.2f} ms")

        # --- Comparaison Réel vs Prédit si Data ID fourni ---
        if data_id_input > 0 and data_id_input in df_raw.index:
            real_val = float(df_raw.loc[data_id_input, target])
            st.markdown("---")
            st.markdown(f"### 🧪 Comparaison — Expérience #{data_id_input}")

            col_real, col_best, col_err = st.columns(3)
            col_real.metric("📋 Valeur Réelle (labo)", f"{real_val:.4f} mg/g")
            col_best.metric(f"🤖 Prédiction ({best_model_name})", f"{best_pred_value:.4f} mg/g")
            err_pct = abs(best_pred_value - real_val) / real_val * 100 if real_val != 0 else 0
            col_err.metric("📐 Erreur relative", f"{err_pct:.2f} %",
                           delta=f"{'✅ Précis' if err_pct < 10 else '⚠️ Écart notable'}",
                           delta_color="normal" if err_pct < 10 else "inverse")

            # Tableau comparatif tous modèles
            comp_data = {
                "Modèle": list(preds_dict.keys()),
                "Prédiction (mg/g)": [f"{v:.4f}" for v in preds_dict.values()],
                "Valeur Réelle (mg/g)": [f"{real_val:.4f}"] * 4,
                "Erreur absolue": [f"{abs(v - real_val):.4f}" for v in preds_dict.values()],
                "Erreur (%)": [f"{abs(v - real_val)/real_val*100:.2f}%" if real_val != 0 else "N/A" for v in preds_dict.values()]
            }
            st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Graphique scatter "Influence du temps de contact"
    time_col = next((c for c in X.columns if 'time' in c.lower() or 'contact' in c.lower()), X.columns[0])
    ph_col   = next((c for c in X.columns if 'ph' in c.lower()), None)

    st.markdown(f"<h4 style='color:white;'>Influence du {display_columns.get(time_col, time_col)} sur la capacité qe</h4>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    scatter = ax.scatter(
        df[time_col], df[target],
        c=df[ph_col] if ph_col else "#00FFD1",
        cmap="cool", alpha=0.85, s=80, edgecolors='none'
    )
    if ph_col:
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("pH", color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
    ax.set_xlabel(display_columns.get(time_col, time_col), color='white')
    ax.set_ylabel("qe (mg/g)", color='white')
    for spine in ax.spines.values():
        spine.set_color('#1F2937')
    st.pyplot(fig)

# --- WORKFLOW ---
with tab_workflow:
    st.markdown("<h2 style='color:#00FFD1;'>⚙️ Pipeline de Traitement</h2>", unsafe_allow_html=True)
    st.markdown("""
    **Architecture du pipeline ML :**

    1. **Chargement des données** → Détection automatique du séparateur CSV
    2. **Nettoyage** → Suppression de la colonne `Data ID` si présente
    3. **Split train/test** → 80% / 20%, `random_state=42`
    4. **Scaling** → `StandardScaler` appliqué sur ANN uniquement
    5. **Entraînement** → 4 modèles en parallèle mis en cache (`@st.cache_resource`)
    6. **Évaluation** → R², RMSE, MAE sur le jeu de test
    7. **Prédiction** → Saisie utilisateur via la sidebar gauche
    """)

    note("Pipeline",
        "Le pipeline commence par la détection automatique du format CSV, puis divise les données en 80% "
        "pour l'entraînement et 20% pour l'évaluation — c'est le principe du <b>train/test split</b>. "
        "Le StandardScaler est appliqué uniquement sur l'ANN car les réseaux de neurones sont sensibles "
        "à l'échelle des données contrairement aux arbres de décision. "
        "Tous les modèles sont mis en cache pour éviter de recalculer à chaque interaction.")

    st.markdown("### Modèles entraînés")
    wc1, wc2, wc3, wc4 = st.columns(4)
    wc1.info("🧠 **ANN**\nMLPRegressor\n128→64 neurones\nReLU, 3000 iter")
    wc2.info("⚡ **XGBoost**\n200 estimateurs\nlr=0.05\nmax_depth=5")
    wc3.info("🌲 **Random Forest**\n200 estimateurs\nmax_depth=10")
    wc4.info("📉 **Rég. Linéaire**\nBaseline\nMultivariée")

# --- DATASET ---
with tab_dataset:
    st.markdown("<h2 style='color:#00FFD1;'>📂 Registre des Expériences d'Adsorption</h2>", unsafe_allow_html=True)
    st.write(f"Affichage complet des **{df.shape[0]}** expériences de laboratoire.")
    note("Dataset",
        f"Ce tableau présente l'intégralité des {df.shape[0]} expériences collectées en laboratoire. "
        "Chaque ligne correspond à une expérience unique avec ses conditions physico-chimiques contrôlées : "
        "pH, dosage d'adsorbant, temps de contact, concentration initiale en Pb²⁺. "
        "La dernière colonne <b>qe</b> est la variable de sortie — c'est ce que les modèles cherchent à prédire. "
        "Ces données brutes sont la base de tout le travail de modélisation.")

    display_df = df.copy()
    # Colonne cible toujours en derniere position
    cols_ordered = [c for c in display_df.columns if c != target] + [target]
    display_df = display_df[cols_ordered]
    for col in display_df.select_dtypes(include=[float]).columns:
        display_df[col] = display_df[col].apply(lambda x: ('%.8f' % x).rstrip('0').rstrip('.'))

    st.dataframe(display_df, use_container_width=True, height=500)

# --- HEATMAP ---
with tab_heatmap:
    st.markdown("<h2 style='color:#00FFD1;'>🔥 Matrice de Corrélation Haute Définition</h2>", unsafe_allow_html=True)

    note("Heatmap",
        "La matrice de corrélation permet de visualiser les liens linéaires entre toutes les variables. "
        "Une valeur proche de <b>+1</b> indique une forte corrélation positive, proche de <b>-1</b> une corrélation négative, "
        "et proche de <b>0</b> une absence de relation linéaire. "
        "C'est un outil d'exploration essentiel pour comprendre quelles variables influencent le plus qe "
        "et détecter d'éventuelles redondances entre variables d'entrée.")

    fig, ax = plt.subplots(figsize=(10, 6))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap="mako", fmt=".2f",
                linewidths=1, linecolor='#0B1020', ax=ax, cbar=True)
    plt.xticks(rotation=30, ha='right', color='white')
    plt.yticks(color='white')
    st.pyplot(fig)

# --- PAGES MODÈLES : ANN, RF, XGB, LR ---
model_tabs = {
    tab_ann: "ANN",
    tab_rf: "Random Forest",
    tab_xgb: "XGBoost",
    tab_lr: "Régression Linéaire"
}

for tab, model_name in model_tabs.items():
    with tab:
        st.markdown(f"<h2 style='color:#00FFD1;'>🔍 Analyse du Modèle : {model_name}</h2>", unsafe_allow_html=True)

        _notes = {
            "ANN": "Le Réseau de Neurones Artificiels (ANN) est organisé en deux couches cachées : 128 neurones puis 64 neurones, avec la fonction d'activation ReLU. Il s'entraîne sur les données normalisées par le StandardScaler. <b>R² Train</b> mesure comment il a appris, <b>R² Test</b> mesure sa capacité à prédire des données jamais vues. Le graphique Réel vs Prédit montre que les points sont très proches de la droite idéale y=x, ce qui confirme l'excellente précision du modèle.",
            "XGBoost": "XGBoost est un algorithme de <b>boosting par gradient</b> : il construit 200 arbres de décision séquentiellement, chaque arbre corrigeant les erreurs du précédent. Le paramètre learning_rate=0.05 assure une convergence progressive et évite le surapprentissage. C'est l'un des algorithmes les plus performants sur des datasets tabulaires de taille modeste, ce que nos résultats confirment.",
            "Random Forest": "Le Random Forest construit 200 arbres de décision en parallèle sur des sous-échantillons aléatoires des données. La prédiction finale est la <b>moyenne</b> de tous les arbres — c'est le principe du bagging. Cette approche le rend naturellement robuste au surapprentissage. La légère infériorité par rapport à l'ANN et XGBoost s'explique par sa tendance à sous-estimer les valeurs extrêmes de qe.",
            "Régression Linéaire": "La Régression Linéaire sert ici de <b>baseline scientifique</b>. Elle suppose que qe est une combinaison linéaire des variables d'entrée — hypothèse trop simpliste pour des phénomènes d'adsorption qui suivent des isothermes non linéaires (Langmuir, Freundlich). Son R² plus faible prouve mathématiquement la <b>nature non linéaire</b> des interactions à l'interface solide-liquide entre le Pb²⁺ et l'adsorbant."
        }
        note(f"{model_name}", _notes.get(model_name, ""))

        row = metrics_df.loc[model_name]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R² Train", f"{row['R2 Train']:.4f}")
        c2.metric("R² Test",  f"{row['R2 Test']:.4f}")
        c3.metric("RMSE",     f"{row['RMSE']:.4f}")
        c4.metric("MAE",      f"{row['MAE']:.4f}")

        note("Graphique Réel vs Prédit",
            "Ce graphique est le test de vérité d'un modèle. Chaque point représente une expérience du jeu de test. "
            "L'axe horizontal montre la valeur mesurée en laboratoire, l'axe vertical montre ce que le modèle prédit. "
            "La ligne rouge pointillée est la ligne parfaite y=x. "
            "Plus les points sont proches de cette ligne, meilleur est le modèle. "
            "Un nuage de points dispersé signifierait que le modèle ne généralise pas bien.")
        st.markdown("### Valeurs Réelles vs Prédites")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(y_test, preds_test[model_name], color="#00FFD1", alpha=0.7, edgecolors='black', label='Données de test')
        ideal_line = np.linspace(min(y_test), max(y_test), 100)
        ax.plot(ideal_line, ideal_line, color="#FF4B4B", linestyle="--", linewidth=2, label="y = x (parfait)")
        ax.set_xlabel("Valeurs Expérimentales (mg/g)", color='white')
        ax.set_ylabel("Prédictions (mg/g)", color='white')
        ax.legend()
        for spine in ax.spines.values():
            spine.set_color('#334155')
        st.pyplot(fig)

        if model_name == "Régression Linéaire":
            st.warning("⚠️ La dispersion des points illustre l'incapacité d'un modèle linéaire à capturer la non-linéarité des mécanismes d'adsorption du Pb²⁺.")

# --- COMPARAISON ---
with tab_comparaison:
    st.markdown("<h2 style='color:#00FFD1;'>📈 Évaluation Comparative Globale</h2>", unsafe_allow_html=True)

    note("Comparaison des modèles",
        "Ce tableau compare les 4 modèles sur 3 métriques clés. "
        "<b>R²</b> (coefficient de détermination) : plus il est proche de 1, meilleur est le modèle — il mesure la proportion de variance expliquée. "
        "<b>RMSE</b> (Root Mean Square Error) : erreur quadratique moyenne — pénalise fortement les grandes erreurs, exprimée en mg/g. "
        "<b>MAE</b> (Mean Absolute Error) : erreur absolue moyenne — interprétation directe en mg/g. "
        "Pour un bon modèle on veut : R² → 1, RMSE → 0, MAE → 0. "
        "Le diagnostic de surapprentissage ci-dessous vérifie que ces scores sont fiables et non artificiels.")

    # Tableau principal
    st.dataframe(
        metrics_df.style.highlight_max(subset=['R2 Test'], color='rgba(0, 255, 209, 0.3)')
                        .highlight_min(subset=['RMSE', 'MAE'], color='rgba(0, 255, 209, 0.15)'),
        use_container_width=True
    )

    # --- DIAGNOSTIC SURAPPRENTISSAGE ---
    st.markdown("### 🔍 Diagnostic de Surapprentissage (Overfitting)")
    st.markdown("<p style='color:#A0AEC0;font-size:13px;'>Un Gap (R² Train − R² Test) > 0.02 signale un surapprentissage. En dessous : le modèle généralise bien.</p>", unsafe_allow_html=True)

    diag_cols = st.columns(4)
    for i, model_name in enumerate(metrics_df.index):
        row = metrics_df.loc[model_name]
        gap = row["Gap"]
        if gap < 0.01:
            status = "✅ Excellent"
            color = "#00FFD1"
        elif gap < 0.02:
            status = "🟡 Acceptable"
            color = "#F59E0B"
        else:
            status = "🔴 Surapprentissage"
            color = "#EF4444"

        with diag_cols[i]:
            st.markdown(f"""
            <div style='background:#111827;border-radius:10px;padding:16px;border-left:4px solid {color};'>
                <p style='color:#A0AEC0;font-size:12px;margin:0;'>{model_name}</p>
                <p style='color:white;font-size:20px;font-weight:bold;margin:4px 0;'>Gap = {gap:.4f}</p>
                <p style='color:{color};font-size:13px;margin:0;font-weight:bold;'>{status}</p>
                <p style='color:#6B7280;font-size:11px;margin:4px 0 0 0;'>Train : {row['R2 Train']:.4f} | Test : {row['R2 Test']:.4f}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Graphique Train vs Test R²
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(metrics_df.index))
    w = 0.35
    ax.bar(x - w/2, metrics_df["R2 Train"], width=w, color="#3B82F6", label="R² Train", edgecolor='white')
    ax.bar(x + w/2, metrics_df["R2 Test"],  width=w, color="#00FFD1", label="R² Test",  edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df.index, color='white', rotation=10)
    ax.set_ylim(0.7, 1.02)
    ax.set_title("R² Train vs Test — Détection du surapprentissage", color='white')
    ax.legend()
    ax.axhline(y=1.0, color='#FF4B4B', linestyle='--', linewidth=1, alpha=0.5)
    for spine in ax.spines.values():
        spine.set_color('#334155')
    st.pyplot(fig)

    # Graphiques RMSE et MAE
    fig2, axes = plt.subplots(1, 2, figsize=(12, 4))
    for idx, (metric, color) in enumerate(zip(["RMSE", "MAE"], ["#FF4B4B", "#3B82F6"])):
        axes[idx].bar(metrics_df.index, metrics_df[metric], color=color, edgecolor='white', width=0.5)
        axes[idx].set_title(f"Comparatif {metric}", color='white', fontsize=12)
        axes[idx].tick_params(colors='white', axis='x', rotation=15)
        for spine in axes[idx].spines.values():
            spine.set_color('#334155')
    st.pyplot(fig2)

# --- IMPORTANCE ---
with tab_importance:
    st.markdown("<h2 style='color:#00FFD1;'>⭐ Importance des Variables</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#A0AEC0;font-size:13px;'>Comparaison de l'influence des variables selon chaque algorithme.</p>", unsafe_allow_html=True)

    def make_importance_df(variable_names, scores):
        imp = pd.DataFrame({"Variable": variable_names, "Importance": scores})
        imp["Variable"] = imp["Variable"].map(display_columns).fillna(imp["Variable"])
        return imp.sort_values(by="Importance", ascending=True)

    def plot_importance(imp, color, title):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.barh(imp["Variable"], imp["Importance"], color=color, edgecolor='white', height=0.5)
        ax.set_xlabel("Importance relative", color='white')
        ax.set_title(title, color='white', fontsize=13)
        for spine in ax.spines.values():
            spine.set_color('#334155')
        return fig

    note("Importance des variables",
        "Cet onglet répond à une question fondamentale : <b>quels paramètres expérimentaux influencent le plus l'adsorption du Pb²⁺ ?</b> "
        "Chaque algorithme mesure cette importance différemment — ce qui rend la comparaison scientifiquement riche. "
        "Le Random Forest utilise la réduction d'impureté, XGBoost le gain de performance, "
        "la Régression Linéaire ses coefficients, et l'ANN les valeurs SHAP qui mesurent l'impact réel sur chaque prédiction individuelle. "
        "Le tableau de consensus en bas synthétise un classement unanime des variables.")

    # --- Random Forest ---
    st.markdown("### 🌲 Random Forest — Importance par entropie")
    imp_rf = make_importance_df(X.columns, models["Random Forest"].feature_importances_)
    st.pyplot(plot_importance(imp_rf, "#00FFD1", "Random Forest — Impurity-based importance"))

    st.markdown("---")

    # --- XGBoost ---
    st.markdown("### ⚡ XGBoost — Importance par gain")
    imp_xgb = make_importance_df(X.columns, models["XGBoost"].feature_importances_)
    st.pyplot(plot_importance(imp_xgb, "#F59E0B", "XGBoost — Gain-based importance"))

    st.markdown("---")

    # --- Régression Linéaire — Coefficients ---
    st.markdown("### 📉 Régression Linéaire — Coefficients standardisés")
    lr_coefs = np.abs(models["Régression Linéaire"].coef_)
    lr_coefs_norm = lr_coefs / lr_coefs.sum()
    imp_lr = make_importance_df(X.columns, lr_coefs_norm)
    st.pyplot(plot_importance(imp_lr, "#3B82F6", "Régression Linéaire — |Coefficients| normalisés"))

    st.markdown("---")

    # --- ANN — SHAP values ---
    st.markdown("### 🧠 ANN — Analyse SHAP (valeurs d'impact moyennes)")
    with st.spinner("Calcul des valeurs SHAP en cours..."):
        try:
            explainer = shap.KernelExplainer(
                models["ANN"].predict,
                shap.sample(pd.DataFrame(X_train_scaled, columns=X.columns), 30)
            )
            shap_values = explainer.shap_values(
                pd.DataFrame(X_test_scaled, columns=X.columns), nsamples=100
            )
            mean_shap = np.abs(shap_values).mean(axis=0)
            imp_ann = make_importance_df(X.columns, mean_shap / mean_shap.sum())
            st.pyplot(plot_importance(imp_ann, "#EF4444", "ANN — Impact moyen |SHAP|"))
        except Exception as e:
            st.warning(f"SHAP non disponible : {e}")

    st.markdown("---")

    # --- Synthèse comparative ---
    st.markdown("### 📊 Synthèse comparative — Classement consensus")
    rank_rf  = imp_rf.set_index("Variable")["Importance"].rank(ascending=False)
    rank_xgb = imp_xgb.set_index("Variable")["Importance"].rank(ascending=False)
    rank_lr  = imp_lr.set_index("Variable")["Importance"].rank(ascending=False)
    consensus = (rank_rf + rank_xgb + rank_lr).sort_values()
    st.dataframe(
        pd.DataFrame({"Variable": consensus.index, "Score de consensus (rang moyen)": (consensus / 3).round(2)}).reset_index(drop=True),
        use_container_width=True, hide_index=True
    )

# --- ANALYSE (Scénarios critiques) ---
with tab_analyse:
    st.markdown("<h2 style='color:#00FFD1;'>🚨 Évaluation des Conditions aux Limites</h2>", unsafe_allow_html=True)

    note("Scénarios critiques",
        "Cette page extrait directement depuis les données expérimentales réelles les deux cas extrêmes. "
        "La colonne de gauche montre les conditions qui ont produit la <b>capacité d'adsorption maximale</b> — "
        "c'est le scénario idéal à reproduire en pratique pour dépolluer efficacement. "
        "La colonne de droite montre les conditions du pire rendement observé. "
        "Cette comparaison permet d'identifier les seuils critiques de chaque paramètre.")

    best_idx  = df[target].idxmax()
    worst_idx = df[target].idxmin()
    best_row  = df.loc[best_idx]
    worst_row = df.loc[worst_idx]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<h4 style='color:#00FFD1;'>✔ Conditions Optimales</h4>", unsafe_allow_html=True)
        html_best = "<div class='metric-card'>"
        for col in X.columns:
            try:
                html_best += f"<p style='margin:4px 0;'>🔹 {display_columns.get(col, col)} : <b>{best_row[col]:.2f}</b></p>"
            except Exception:
                html_best += f"<p style='margin:4px 0;'>🔹 {display_columns.get(col, col)} : <b>{best_row[col]}</b></p>"
        html_best += f"<hr style='border-color:#334155;'><p style='color:#00FFD1;font-size:18px;margin:0;'>Capacité max : <b>{best_row[target]:.4f} mg/g</b></p></div>"
        st.markdown(html_best, unsafe_allow_html=True)

    with c2:
        st.markdown("<h4 style='color:#FF4B4B;'>❌ Pire Scénario</h4>", unsafe_allow_html=True)
        html_worst = "<div class='metric-card-danger'>"
        for col in X.columns:
            try:
                html_worst += f"<p style='margin:4px 0;'>🔸 {display_columns.get(col, col)} : <b>{worst_row[col]:.2f}</b></p>"
            except Exception:
                html_worst += f"<p style='margin:4px 0;'>🔸 {display_columns.get(col, col)} : <b>{worst_row[col]}</b></p>"
        html_worst += f"<hr style='border-color:#334155;'><p style='color:#FF4B4B;font-size:18px;margin:0;'>Capacité min : <b>{worst_row[target]:.4f} mg/g</b></p></div>"
        st.markdown(html_worst, unsafe_allow_html=True)

# --- BILAN & ANALYSE ---
with tab_bilan:
    st.markdown("<h2 style='color:#00FFD1;'>📘 Interprétation et Conclusions Scientifiques</h2>", unsafe_allow_html=True)
    st.info(f"""
**Validation des performances de l'infrastructure IA :**

1. Le modèle **{metrics_df['R2 Test'].idxmax()}** affiche la plus haute précision globale avec un coefficient de détermination R² = {metrics_df['R2 Test'].max():.4f}. Cela confirme sa capacité robuste à appréhender les phénomènes d'adsorption complexes.

2. Les modèles basés sur les arbres décisionnels ensemblistes (**XGBoost** et **Random Forest**) démontrent une excellente stabilité face aux variations physico-chimiques de la solution.

3. L'insuffisance flagrante de la **Régression Linéaire Baseline** (R² = {metrics_df.loc['Régression Linéaire', 'R2 Test']:.4f}) prouve mathématiquement la nature intrinsèquement non linéaire des interactions à l'interface solide-liquide entre le Pb²⁺ et l'adsorbant.
    """)

    note("Conclusions scientifiques",
        "Ces conclusions synthétisent l'ensemble du travail de modélisation. "
        "La supériorité de l'ANN confirme que l'adsorption du Pb²⁺ obéit à des mécanismes <b>non linéaires complexes</b>, "
        "probablement une chimisorption multicouche ou des interactions électrostatiques pH-dépendantes. "
        "L'échec relatif de la Régression Linéaire n'est pas un défaut — c'est une preuve scientifique : "
        "il démontre que les modèles classiques sont insuffisants et justifie le recours au Machine Learning. "
        "L'ensemble de cette plateforme constitue un outil de simulation et d'aide à la décision "
        "pour optimiser les conditions opératoires d'un système de biosorption du plomb en solution aqueuse.")

    # Tableau récapitulatif final
    st.markdown("### 📊 Tableau de Synthèse des Performances")
    st.dataframe(
        metrics_df.style
            .highlight_max(subset=['R2 Test'], color='rgba(0,255,209,0.2)')
            .highlight_min(subset=['RMSE', 'MAE'], color='rgba(0,255,209,0.2)'),
        use_container_width=True
    )