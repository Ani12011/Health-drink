import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
import io

st.set_page_config(page_title="Health‑Drink Consumer Dashboard", layout="wide")

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv("health_drink_survey_synthetic.csv")
    # Engineer classification label
    df["Willingness_Label"] = pd.cut(df["Willingness_Score"],
                                     bins=[-1,3,6,10],
                                     labels=["Unwilling","Neutral","Willing"])
    return df

df = load_data()

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    age_range = st.slider("Age Range", 5, 80, (18,65))
    cities = st.multiselect("Cities", sorted(df["City"].unique()), default=list(df["City"].unique()))
    apply_filters = st.button("Apply")

def filter_df(df):
    sub = df[(df["Age"].between(*age_range)) & (df["City"].isin(cities))]
    return sub

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data Visualisation",
                                        "🔎 Classification",
                                        "🔗 Clustering",
                                        "🛒 Association Rules",
                                        "📈 Regression"])

# ------------------------------------------------------------------
with tab1:
    st.subheader("Descriptive Insights")
    dff = filter_df(df) if 'apply_filters' in globals() and apply_filters else df
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**1. Age Distribution**")
        fig = px.histogram(dff, x="Age", nbins=30)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("**2. Income Distribution (AED)**")
        fig2 = px.histogram(dff, x="Monthly_Household_Income_AED", nbins=30)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**3. Health Consciousness by City**")
        fig3 = px.box(dff, x="City", y="Health_Consciousness", color="City")
        st.plotly_chart(fig3, use_container_width=True)
    with col4:
        st.markdown("**4. Monthly Spend vs Income**")
        fig4 = px.scatter(dff, x="Monthly_Household_Income_AED", y="Monthly_Spend_AED",
                          trendline="ols")
        st.plotly_chart(fig4, use_container_width=True)

    col5, col6 = st.columns(2)
    with col5:
        st.markdown("**5. Top Purchase Channels**")
        channels_expanded = dff["Purchase_Channels"].str.get_dummies(sep=",")
        top_channels = channels_expanded.sum().sort_values(ascending=False).head(10)
        fig5 = px.bar(top_channels, orientation="h")
        st.plotly_chart(fig5, use_container_width=True)
    with col6:
        st.markdown("**6. Flavor Preferences**")
        flavors_expanded = dff["Flavor_Preferences"].str.get_dummies(sep=",")
        top_flavors = flavors_expanded.sum().sort_values(ascending=False)
        fig6 = px.pie(values=top_flavors, names=top_flavors.index)
        st.plotly_chart(fig6, use_container_width=True)

    st.markdown("**7. Correlation Heatmap (numeric cols)**")
    num_cols = dff.select_dtypes(include="number")
    corr = num_cols.corr()
    fig7 = px.imshow(corr, text_auto=True, aspect="auto")
    st.plotly_chart(fig7, use_container_width=True)

    st.markdown("**8. Eco‑friendly Attitude vs Willingness**")
    fig8 = px.box(dff, x="Eco_Friendly_Packaging_Score", y="Willingness_Score",
                  points="all")
    st.plotly_chart(fig8, use_container_width=True)

    st.markdown("**9. Price Sensitivity vs Discounts Influence**")
    fig9 = px.scatter(dff, x="Price_Discount_Influence", y="Max_Price_WTP_500ml_AED",
                      color="Willingness_Label")
    st.plotly_chart(fig9, use_container_width=True)

    st.markdown("**10. Subscription Plan Preference**")
    counts = dff["Subscription_Plan"].value_counts()
    fig10 = px.bar(counts, color=counts.index)
    st.plotly_chart(fig10, use_container_width=True)

    st.info("Each chart includes subtle filters from the sidebar so you can focus on segments of interest.")

# ------------------------------------------------------------------
with tab2:
    st.subheader("Classification Models")

    features = df.drop(columns=["Willingness_Label","Willingness_Score"])
    target = df["Willingness_Label"]

    num_feats = features.select_dtypes(include="number").columns
    cat_feats = features.select_dtypes(exclude="number").columns

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_feats),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats)
    ])

    models = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=42, stratify=target)

    results = {}
    y_prob_dict = {}
    for name, model in models.items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        results[name] = {
            "Train Acc": accuracy_score(y_train, pipe.predict(X_train)),
            "Test Acc": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "F1": f1_score(y_test, y_pred, average="weighted", zero_division=0)
        }
        # Probas for ROC
        if hasattr(pipe.named_steps["model"], "predict_proba"):
            y_prob_dict[name] = pipe.predict_proba(X_test)
        else:
            # Use decision_function if available
            if hasattr(pipe.named_steps["model"], "decision_function"):
                dec = pipe.decision_function(X_test)
                # Convert to prob-like array
                y_prob_dict[name] = np.exp(dec) / np.sum(np.exp(dec), axis=1, keepdims=True)

        models[name] = pipe  # store pipeline

    metrics_df = pd.DataFrame(results).T
    st.dataframe(metrics_df.style.background_gradient(cmap="Blues"), use_container_width=True)

    st.markdown("### Confusion Matrix")
    algo_choice = st.selectbox("Select model for confusion matrix", list(models.keys()))
    with st.spinner("Computing..."):
        model = models[algo_choice]
        cm_disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=model.classes_)
        fig_cm = cm_disp.figure_
        st.pyplot(fig_cm)

    st.markdown("### ROC Curve")
    plt.figure(figsize=(8,6))
    for name, probs in y_prob_dict.items():
        fpr, tpr, _ = roc_curve(y_test.map({"Unwilling":0,"Neutral":1,"Willing":2}), probs[:,2])  # one‑vs‑rest on positive class
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")
    plt.plot([0,1],[0,1],"--", color="grey")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves"); plt.legend()
    st.pyplot(plt.gcf())

    st.markdown("### Batch Prediction")
    uploaded = st.file_uploader("Upload new CSV (same schema but without 'Willingness_Label' / 'Willingness_Score')", type="csv")
    if uploaded:
        new_df = pd.read_csv(uploaded)
        sel_model = st.selectbox("Pick model for prediction", list(models.keys()), key="pred_model")
        preds = models[sel_model].predict(new_df)
        new_df["Predicted_Willingness_Label"] = preds
        st.write(new_df.head())
        csv_data = new_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions", csv_data, file_name="predictions.csv", mime="text/csv")

# ------------------------------------------------------------------
with tab3:
    st.subheader("K‑Means Clustering")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    k = st.slider("Number of clusters (k)", 2, 10, 4)
    scaler = StandardScaler()
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    scaled = scaler.fit_transform(df[numeric_cols])
    clusters = km.fit_predict(scaled)
    df_clustered = df.copy()
    df_clustered["Cluster"] = clusters
    st.markdown("#### Elbow Method")
    inertias = []
    for kk in range(2,11):
        inertias.append(KMeans(n_clusters=kk, n_init="auto", random_state=42).fit(scaled).inertia_)
    fig_elbow = px.line(x=list(range(2,11)), y=inertias, markers=True, labels={"x":"k", "y":"Inertia"})
    st.plotly_chart(fig_elbow, use_container_width=True)

    st.markdown("#### Cluster Persona Summary")
    persona_table = df_clustered.groupby("Cluster").agg({
        "Age":"median",
        "Monthly_Household_Income_AED":"median",
        "Health_Consciousness":"median",
        "Monthly_Spend_AED":"median",
        "Willingness_Score":"median"
    }).rename(columns={
        "Age":"Median Age",
        "Monthly_Household_Income_AED":"Median Income",
        "Health_Consciousness":"Median Health Consciousness",
        "Monthly_Spend_AED":"Median Spend",
        "Willingness_Score":"Median Willingness"
    })
    st.dataframe(persona_table)

    csv_clusters = df_clustered.to_csv(index=False).encode("utf-8")
    st.download_button("Download Data with Cluster Labels", csv_clusters, file_name="clustered_data.csv", mime="text/csv")

# ------------------------------------------------------------------
with tab4:
    st.subheader("Association Rule Mining")
    multi_cols = ["Fitness_Activities","Motivations","Purchase_Channels","Flavor_Preferences","Brand_Descriptors"]
    col1, col2 = st.columns(2)
    with col1:
        colA = st.selectbox("Select first column", multi_cols)
    with col2:
        colB = st.selectbox("Select second column", [c for c in multi_cols if c!=colA])
    min_sup = st.slider("Minimum support", 0.01, 0.5, 0.05, 0.01)
    min_conf = st.slider("Minimum confidence", 0.1, 1.0, 0.3, 0.05)

    def explode_transactions(series):
        return series.str.split(",").explode()

    te_df = pd.DataFrame({
        colA: explode_transactions(df[colA]),
        colB: explode_transactions(df[colB])
    }).dropna()

    # Build transaction list
    transactions = [row.dropna().tolist() for _, row in df[[colA,colB]].iterrows()]
    from mlxtend.preprocessing import TransactionEncoder
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    trans_df = pd.DataFrame(te_ary, columns=te.columns_)
    freq_items = apriori(trans_df, min_support=min_sup, use_colnames=True)
    rules = association_rules(freq_items, metric="confidence", min_threshold=min_conf)
    top_rules = rules.sort_values("confidence", ascending=False).head(10)
    st.dataframe(top_rules[["antecedents","consequents","support","confidence","lift"]])

# ------------------------------------------------------------------
with tab5:
    st.subheader("Regression Insights")
    target_var = st.selectbox("Select target variable", ["Max_Price_WTP_500ml_AED","Monthly_Spend_AED"])
    X = df.drop(columns=[target_var])
    y = df[target_var]

    num_feats = X.select_dtypes(include="number").columns
    cat_feats = X.select_dtypes(exclude="number").columns

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_feats),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats)
    ])

    regressors = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1),
        "Decision Tree": DecisionTreeRegressor(max_depth=5)
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    metrics = {}
    for name, reg in regressors.items():
        pipe = Pipeline([("prep", preprocessor), ("reg", reg)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        metrics[name] = {
            "Train R2": pipe.score(X_train, y_train),
            "Test R2": pipe.score(X_test, y_test),
            "MAE": np.mean(np.abs(y_test - y_pred))
        }

    st.dataframe(pd.DataFrame(metrics).T.style.background_gradient(cmap="Greens"))

    # Quick feature importance for tree model
    tree_model = Pipeline([("prep", preprocessor), ("reg", regressors["Decision Tree"])])
    tree_model.fit(X_train, y_train)
    importances = tree_model.named_steps["reg"].feature_importances_
    feature_names = (tree_model.named_steps["prep"]
                     .transformers_[0][2].tolist() +
                     list(tree_model.named_steps["prep"]
                          .transformers_[1][1].get_feature_names_out(cat_feats)))
    imp_df = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(10)
    fig_imp = px.bar(imp_df, orientation="h", title="Top 10 Feature Importances (Decision Tree)")
    st.plotly_chart(fig_imp, use_container_width=True)

    st.info("Regression tab lets you compare model performances and inspect key drivers quickly.")

st.success("Dashboard ready! Use the sidebar to filter data and explore insights across tabs.")