import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import plotly.express as px

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="PCA Dashboard",
    layout="wide"
)

st.title("📊 Principal Component Analysis (PCA) Dashboard")

# --------------------------------------------------
# CREATE FOLDERS
# --------------------------------------------------

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# --------------------------------------------------
# DEFAULT DATASET
# --------------------------------------------------

@st.cache_data
def load_default_data():

    file_path = "data/sample_dataset.csv"

    if os.path.exists(file_path):
        return pd.read_csv(file_path)

    np.random.seed(42)

    n = 300

    df = pd.DataFrame({
        "Age": np.clip(np.random.normal(40, 12, n).astype(int), 18, 70),
        "Income": np.clip(np.random.normal(70, 25, n).astype(int), 15, 150),
        "SpendingScore": np.clip(np.random.normal(50, 20, n).astype(int), 1, 100),
        "Savings": np.clip(np.random.normal(30, 15, n).astype(int), 0, 100),
        "Transactions": np.clip(np.random.normal(20, 8, n).astype(int), 1, 50)
    })

    df.to_csv(file_path, index=False)

    return df

# --------------------------------------------------
# DATASET UPLOAD
# --------------------------------------------------

st.sidebar.header("📂 Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    dataset_path = os.path.join(
        "data",
        uploaded_file.name
    )

    df.to_csv(
        dataset_path,
        index=False
    )

    st.sidebar.success(
        f"Saved: {uploaded_file.name}"
    )

else:

    df = load_default_data()

    dataset_path = "data/sample_dataset.csv"

# --------------------------------------------------
# DOWNLOAD SAMPLE DATA
# --------------------------------------------------

sample_csv = load_default_data().to_csv(index=False)

st.sidebar.download_button(
    "Download Sample Dataset",
    sample_csv,
    "sample_dataset.csv",
    "text/csv"
)

# --------------------------------------------------
# DATA OVERVIEW
# --------------------------------------------------

st.subheader("📁 Dataset Overview")

col1, col2 = st.columns(2)

col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])

st.dataframe(df.head())

# --------------------------------------------------
# HANDLE MISSING VALUES
# --------------------------------------------------

numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].mean())

# --------------------------------------------------
# FEATURE SELECTION
# --------------------------------------------------

st.sidebar.header("🎯 Feature Selection")

numeric_features = df.select_dtypes(
    include=np.number
).columns.tolist()

if len(numeric_features) < 2:
    st.error("Dataset must contain at least 2 numeric columns.")
    st.stop()

selected_features = st.sidebar.multiselect(
    "Select Features",
    numeric_features,
    default=numeric_features
)

if len(selected_features) < 2:
    st.warning("Please select at least 2 features.")
    st.stop()

X = df[selected_features]

# --------------------------------------------------
# SCALING
# --------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# --------------------------------------------------
# PCA PARAMETERS
# --------------------------------------------------

st.sidebar.header("⚙ PCA Parameters")

max_components = min(
    len(selected_features),
    X.shape[0]
)

n_components = st.sidebar.slider(
    "Number of Principal Components",
    min_value=2,
    max_value=max_components,
    value=min(2, max_components)
)

# --------------------------------------------------
# PCA MODEL
# --------------------------------------------------

pca = PCA(
    n_components=n_components
)

X_pca = pca.fit_transform(X_scaled)

# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

model_path = "models/pca_model.pkl"

with open(model_path, "wb") as f:
    pickle.dump(pca, f)

# --------------------------------------------------
# CREATE PCA DATAFRAME
# --------------------------------------------------

pca_columns = [
    f"PC{i+1}"
    for i in range(n_components)
]

pca_df = pd.DataFrame(
    X_pca,
    columns=pca_columns
)

# --------------------------------------------------
# SAVE OUTPUT
# --------------------------------------------------

output_path = "outputs/pca_transformed_dataset.csv"

pca_df.to_csv(
    output_path,
    index=False
)

# --------------------------------------------------
# EXPLAINED VARIANCE
# --------------------------------------------------

st.subheader("📈 Explained Variance Analysis")

variance_ratio = pca.explained_variance_ratio_

variance_df = pd.DataFrame({
    "Component": pca_columns,
    "Explained Variance": variance_ratio
})

variance_df["Cumulative Variance"] = (
    variance_df["Explained Variance"]
    .cumsum()
)

total_variance = variance_ratio.sum()

st.metric(
    "Total Variance Retained",
    f"{total_variance:.2%}"
)

st.dataframe(variance_df)

# --------------------------------------------------
# EXPLAINED VARIANCE CHART
# --------------------------------------------------

fig1 = px.bar(
    variance_df,
    x="Component",
    y="Explained Variance",
    title="Explained Variance by Principal Component"
)

st.plotly_chart(
    fig1,
    use_container_width=True
)

# --------------------------------------------------
# CUMULATIVE VARIANCE
# --------------------------------------------------

fig2 = px.line(
    variance_df,
    x="Component",
    y="Cumulative Variance",
    markers=True,
    title="Cumulative Variance Explained"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# --------------------------------------------------
# PCA VISUALIZATION
# --------------------------------------------------

if n_components >= 2:

    st.subheader("🎯 PCA 2D Visualization")

    fig3 = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        title="PCA Projection"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

# --------------------------------------------------
# PCA 3D VISUALIZATION
# --------------------------------------------------

if n_components >= 3:

    st.subheader("🌍 PCA 3D Visualization")

    fig4 = px.scatter_3d(
        pca_df,
        x="PC1",
        y="PC2",
        z="PC3",
        title="3D PCA Projection"
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )

# --------------------------------------------------
# FEATURE LOADINGS
# --------------------------------------------------

st.subheader("🧠 Feature Loadings")

loadings = pd.DataFrame(
    pca.components_.T,
    columns=pca_columns,
    index=selected_features
)

st.dataframe(loadings)

# --------------------------------------------------
# LOADINGS HEATMAP
# --------------------------------------------------

fig5 = px.imshow(
    loadings,
    text_auto=True,
    aspect="auto",
    title="Feature Contribution Heatmap"
)

st.plotly_chart(
    fig5,
    use_container_width=True
)

# --------------------------------------------------
# TRANSFORMED DATA
# --------------------------------------------------

st.subheader("📄 PCA Transformed Dataset")

st.dataframe(
    pca_df.head(20)
)

# --------------------------------------------------
# DOWNLOAD RESULTS
# --------------------------------------------------

st.subheader("⬇ Download PCA Dataset")

csv = pca_df.to_csv(index=False)

st.download_button(
    "Download PCA Results",
    csv,
    "pca_transformed_dataset.csv",
    "text/csv"
)

# --------------------------------------------------
# STATUS
# --------------------------------------------------

st.success(
    f"Dataset stored in: {dataset_path}"
)

st.success(
    f"Model stored in: {model_path}"
)

st.success(
    f"Results stored in: {output_path}"
)

st.success(
    "PCA analysis completed successfully 🚀"
)

