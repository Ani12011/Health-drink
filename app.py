import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, ConfusionMatrixDisplay
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# --------------------------------------------------
st.set_page_config(page_title="Health‑Drink Consumer Dashboard", layout="wide")

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv("health_drink_survey_synthetic.csv")
    # Classification target label
    df["Willingness_Label"] = pd.cut(
        df["Willingness_Score"],
        bins=[-1, 3, 6, 10],
        labels=["Unwilling", "Neutral", "Willing"]
    )
    return df

df = load_data()

# --------------------------------------------------
# Sidebar filters
with st.sidebar:
    st.header("Filters")
    age_range = st.slider("Age Range", 5, 80, (18, 65))
    cities = st.multiselect(
        "Cities",
        sorted(df["City"].unique()),
        default=list(df["City"].unique())
    )
    apply_filters = st.button("Apply")

def filter_df(_df):
    return _df[
        (_df["Age"].between(*age_range)) &
        (_df["City"].isin(cities))
    ]

# --------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Visualisation",
    "🔎 Classification",
    "🔗 Clustering",
    "🛒 Association Rules",
    "📈 Regression"
])

# ==================================================
# 1) DATA VISUALISATION
# --------------------------------------------------
with tab1:
    st.subheader("Descriptive Insights")

    dff = filter_df(df) if apply_filters else df

    # --- Chart 1 & 2
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**1. Age Distribution**")
        fig = px.histogram(dff, x="Age", nbins=30)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**2. Income Distribution (AED)**")
        fig2 = px.histogram(dff, x="Monthly_Household_Income_AED", nbins=30)
        st.plotly_chart(fig2, use_container_width=True)

    # --- Chart 3 & 4
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**3. Health Consciousness by City**")
        fig3 = px.box(dff, x="City", y="Health_Consciousness", color="City")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("**4. Monthly Spend vs Income**")
        fig4 = px.scatter(
            dff,
            x="Monthly_Household_Income_AED",
            y="Monthly_Spend_AED",
            opacity=0.7
        )
        st.plotly_chart(fig4, use_container_width=True)

    # --- Chart 5 & 6
    col5, col6 = st.columns(2)

    with col5:
        st.markdown("**5. Top Purchase Channels**")
        chan_counts = dff["Purchase_Channels"].str.get_dummies(sep=",").sum().sort_values()
        fig5 = px.bar(chan_counts, orientation="h")
        st.plotly_chart(fig5, use_container_width=True)

    with col6:
        st.markdown("**6. Flavor Preferences**")
        flavor_counts = dff["Flavor_Preferences"].str.get_dummies(sep=",").sum()
        fig6 = px.pie(values=flavor_counts, names=flavor_counts.index)
        st.plotly_chart(fig6, use_container_width=True)

    # --- Heatmap
    st.markdown("**7. Correlation Heatmap (numeric)**")
    corr = dff.select_dtypes("number").corr()
    fig7 = px.imshow(corr, text_auto=True, aspect="auto")
    st.plotly_chart(fig7, use_container_width=True)

    # --- Boxplot & Scatter
    col7, col8 = st.columns(2)
    with col7:
        st.markdown("**8. Eco‑Friendly Score vs Willingness**")
        fig8 = px.box(
            dff,
            x="Eco_Friendly_Packaging_Score",
            y="Willingness_Score",
            points="all"
        )
        st.plotly_chart(fig8, use_container_width=True)

    with col8:
        st.markdown("**9. Price Sensitivity**")
        fig9 = px.scatter(
            dff,
            x="Price_Discount_Influence",
            y="Max_Price_WTP_500ml_AED",
            color="Willingness_Label"
        )
        st.plotly_chart(fig9, use_container_width=True)

    # --- Bar
    st.markdown("**10. Subscription Plan Preference**")
    subs = dff["Subscription_Plan"].value_counts()
    fig10 = px.bar(subs, color=subs.index, labels={"value": "Count", "index": "Plan"})
    st.plotly_chart(fig10, use_container_width=True)

    st.info("Use the sidebar filters to focus on specific segments.")

# ==================================================
# 2) CLASSIFICATION
# --------------------------------------------------
with tab2:
    st.subheader("Classification Models")

    X = df.drop(columns=["Willingness_Label", "Willingness_Score"])
    y = df["Willingness_Label"]

    num_cols = X.select_dtypes("number").columns
    cat_cols = X.select_dtypes(exclude="number").columns

    preproc = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    models_dict = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    metrics = {}
    probas = {}

    for name, mdl in models_dict.items():
        pipe = Pipeline([("prep", preproc), ("model", mdl)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        metrics[name] = {
            "Train Acc": accuracy_score(y_train, pipe.predict(X_train)),
            "Test Acc": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "F1": f1_score(y_test, y_pred, average="weighted", zero_division=0)
        }

        if hasattr(mdl, "predict_proba"):
            probas[name] = pipe.predict_proba(X_test)

        models_dict[name] = pipe  # store pipeline

    st.dataframe(
        pd.DataFrame(metrics).T.style.background_gradient(cmap="Blues"),
        use_container_width=True
    )

    # Confusion matrix
    st.markdown("### Confusion Matrix")
    selected_algo = st.selectbox("Select model", list(models_dict.keys()))
    cm_fig = ConfusionMatrixDisplay.from_estimator(
        models_dict[selected_algo], X_test, y_test,
        display_labels=models_dict[selected_algo].classes_
    ).figure_
    st.pyplot(cm_fig)

    # ROC curves (one‑vs‑rest on "Willing")
    st.markdown("### ROC Curve")
    plt.figure(figsize=(8, 6))
    for n, pr in probas.items():
        fpr, tpr, _ = roc_curve(y_test.map({"Unwilling": 0, "Neutral": 1, "Willing": 2}), pr[:, 2])
        plt.plot(fpr, tpr, label=f"{n} (AUC={auc(fpr, tpr):.2f})")
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Positive Class: 'Willing'")
    plt.legend()
    st.pyplot(plt.gcf())

    # Batch prediction
    st.markdown("### Batch Prediction")
    upl = st.file_uploader("Upload new CSV (same schema, no label)", type="csv")
    if upl:
        new_df = pd.read_csv(upl)
        mdl_choice = st.selectbox(
            "Model for prediction",
            list(models_dict.keys()),
            key="pred_model"
        )
        preds = models_dict[mdl_choice].predict(new_df)
        new_df["Predicted_Willingness_Label"] = preds
        st.write(new_df.head())

        st.download_button(
            "Download Predictions",
            new_df.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv"
        )

# ==================================================
# 3) CLUSTERING
# --------------------------------------------------
with tab3:
    st.subheader("K‑Means Clustering")

    num_cols_all = df.select_dtypes("number").columns
    k_val = st.slider("Number of clusters (k)", 2, 10, 4)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[num_cols_all])

    kmeans = KMeans(n_clusters=k_val, random_state=42, n_init="auto")
    clusters = kmeans.fit_predict(X_scaled)

    dfc = df.copy()
    dfc["Cluster"] = clusters

    # Elbow chart
    inertias = [
        KMeans(n_clusters=k, n_init="auto", random_state=42).fit(X_scaled).inertia_
        for k in range(2, 11)
    ]
    fig_elbow = px.line(
        x=range(2, 11),
        y=inertias,
        markers=True,
        labels={"x": "k", "y": "Inertia"},
        title="Elbow Chart"
    )
    st.plotly_chart(fig_elbow, use_container_width=True)

    # Persona table
    persona = dfc.groupby("Cluster").agg({
        "Age": "median",
        "Monthly_Household_Income_AED": "median",
        "Health_Consciousness": "median",
        "Monthly_Spend_AED": "median",
        "Willingness_Score": "median"
    }).rename(columns={
        "Age": "Median Age",
        "Monthly_Household_Income_AED": "Median Income",
        "Health_Consciousness": "Median HealthConsc.",
        "Monthly_Spend_AED": "Median Spend",
        "Willingness_Score": "Median Willingness"
    })
    st.dataframe(persona)

    # Download
    st.download_button(
        "Download Data with Clusters",
        dfc.to_csv(index=False).encode("utf-8"),
        file_name="clustered_data.csv",
        mime="text/csv"
    )

# ==================================================
# 4) ASSOCIATION RULES
# --------------------------------------------------
with tab4:
    st.subheader("Association Rule Mining")

    multiselect_cols = [
        "Fitness_Activities", "Motivations", "Purchase_Channels",
        "Flavor_Preferences", "Brand_Descriptors"
    ]

    col_a, col_b = st.columns(2)
    with col_a:
        colA = st.selectbox("Select first column", multiselect_cols)
    with col_b:
        colB = st.selectbox(
            "Select second column",
            [c for c in multiselect_cols if c != colA]
        )

    min_sup = st.slider("Min Support", 0.01, 0.5, 0.05, 0.01)
    min_conf = st.slider("Min Confidence", 0.1, 1.0, 0.3, 0.05)

    # Prepare transactions
    transactions = df[[colA, colB]].astype(str).apply(lambda x: ",".join(x), axis=1)
    transactions = transactions.str.split(",").tolist()

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    trans_df = pd.DataFrame(te_ary, columns=te.columns_)

    freq = apriori(trans_df, min_support=min_sup, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
    rules = rules.sort_values("confidence", ascending=False).head(10)

    st.dataframe(rules[["antecedents", "consequents", "support", "confidence", "lift"]])

# ==================================================
# 5) REGRESSION
# --------------------------------------------------
with tab5:
    st.subheader("Regression Insights")

    target = st.selectbox(
        "Target variable",
        ["Max_Price_WTP_500ml_AED", "Monthly_Spend_AED"]
    )

    Xr = df.drop(columns=[target])
    yr = df[target]

    num_cols_r = Xr.select_dtypes("number").columns
    cat_cols_r = Xr.select_dtypes(exclude="number").columns

    prep_r = ColumnTransformer([
        ("num", StandardScaler(), num_cols_r),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols_r)
    ])

    regressors = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1),
        "Decision Tree": DecisionTreeRegressor(max_depth=5)
    }

    Xtr, Xts, ytr, yts = train_test_split(Xr, yr, test_size=0.25, random_state=42)

    reg_metrics = {}
    for n, reg in regressors.items():
        p = Pipeline([("prep", prep_r), ("reg", reg)])
        p.fit(Xtr, ytr)
        y_pred = p.predict(Xts)
        reg_metrics[n] = {
            "Train R²": p.score(Xtr, ytr),
            "Test R²": p.score(Xts, yts),
            "MAE": np.abs(yts - y_pred).mean()
        }

    st.dataframe(
        pd.DataFrame(reg_metrics).T.style.background_gradient(cmap="Greens"),
        use_container_width=True
    )

    # Feature importances (Decision Tree)
    tree_pipe = Pipeline([("prep", prep_r), ("reg", regressors["Decision Tree"])])
    tree_pipe.fit(Xtr, ytr)

    # Extract feature names
    num_feats = tree_pipe.named_steps["prep"].transformers_[0][2]
    cat_feats_names = tree_pipe.named_steps["prep"].transformers_[1][1]\
        .get_feature_names_out(cat_cols_r)
    feat_names = list(num_feats) + list(cat_feats_names)

    importances = tree_pipe.named_steps["reg"].feature_importances_
    imp_series = pd.Series(importances, index=feat_names).sort_values(ascending=False).head(10)

    fig_imp = px.bar(
        imp_series,
        orientation="h",
        title="Top 10 Feature Importances (Decision Tree)"
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    st.info("Compare models quickly and inspect key drivers behind the target variable.")

# --------------------------------------------------
st.success("Dashboard ready! Explore insights across the tabs.")
