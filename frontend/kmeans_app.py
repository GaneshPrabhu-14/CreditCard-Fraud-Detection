from io import BytesIO

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8001"
FEATURE_COLUMNS = [
    "BALANCE",
    "PURCHASES",
    "INSTALLMENTS_PURCHASES",
    "CASH_ADVANCE",
    "CREDIT_LIMIT",
    "PAYMENTS",
    "MINIMUM_PAYMENTS",
    "TENURE",
]

st.set_page_config(page_title="Credit card fraud detection", layout="wide")

brand_column, _, appearance_column = st.columns([3, 5, 2])
with brand_column:
    st.markdown(
        """<div class="brand-lockup">
            <div class="brand-icon">CC</div>
            <div>
                <div class="brand-name">Credit card fraud detection</div>
                <div class="brand-meta">Risk intelligence workspace</div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )
with appearance_column:
    appearance = st.radio(
        "Appearance",
        ["Light", "Dark"],
        horizontal=True,
        label_visibility="collapsed",
        key="kmeans_appearance",
    )

if "kmeans_results" not in st.session_state:
    st.session_state["kmeans_results"] = None
dark_mode = appearance == "Dark"
theme = {
    "ink": "#e8efeb" if dark_mode else "#202b28",
    "muted": "#a4b2ac" if dark_mode else "#68736f",
    "line": "#394942" if dark_mode else "#dce3df",
    "paper": "#131a18" if dark_mode else "#f5f7f4",
    "surface": "#1c2521" if dark_mode else "#ffffff",
    "input": "#222d28" if dark_mode else "#ffffff",
    "green": "#70c4a1" if dark_mode else "#176b52",
    "green_soft": "#24372e" if dark_mode else "#e7f1ec",
    "amber": "#f0a46e" if dark_mode else "#c47c47",
    "normal": "#80b69a" if dark_mode else "#6c8f7d",
    "high_risk": "#e87979" if dark_mode else "#b6483e",
    "grid": "#34423b" if dark_mode else "#e4e9e6",
}

st.markdown(
    f"""
    <style>
    :root {{
        --ink: {theme['ink']}; --muted: {theme['muted']}; --line: {theme['line']};
        --paper: {theme['paper']}; --white: {theme['surface']}; --input: {theme['input']};
        --green: {theme['green']}; --green-soft: {theme['green_soft']};
        --amber: {theme['amber']}; --primary-color: {theme['green']};
    }}
    .stApp {{ background: var(--paper); color: var(--ink); }}
    [data-testid="stHeader"] {{ background: transparent; }}
    [data-testid="stAppViewContainer"] > .main {{ background: var(--paper); }}
    .block-container {{ max-width: 1440px; padding: 4.5rem 1.5rem 3rem; }}
    h1, h2, h3, p, label {{ color: var(--ink); }}
    .brand-lockup {{ display:flex; align-items:center; gap:.75rem; min-height:2.8rem; }}
    .brand-icon {{ display:grid; place-items:center; width:2.5rem; height:2.5rem;
        background:var(--green); color:var(--white); border-radius:8px;
        font:700 .78rem 'IBM Plex Mono',monospace; letter-spacing:-.04em; }}
    .brand-name {{ color:var(--ink); font-size:.95rem; font-weight:700; letter-spacing:-.02em; }}
    .brand-meta {{ color:var(--muted); font-size:.72rem; margin-top:.15rem; }}
    .masthead {{ display:flex; justify-content:space-between; align-items:flex-end; gap:1rem;
        padding:1.25rem 0 1.4rem; margin-bottom:1.2rem; border-bottom:1px solid var(--line); }}
    .eyebrow {{ color:var(--green); font:500 .72rem 'IBM Plex Mono',monospace;
        text-transform:uppercase; margin:0 0 .55rem; }}
    .masthead h1 {{ font-size:2.5rem; line-height:1.1; letter-spacing:-.04em;
        margin:0; font-weight:700; }}
    .masthead p {{ color:var(--muted); margin:.55rem 0 0; }}
    .service-tag {{ border:1px solid var(--line); background:var(--white); color:var(--muted);
        padding:.45rem .7rem; font: .72rem 'IBM Plex Mono',monospace; white-space:nowrap; }}
    [data-testid="stTabs"] [role="tablist"] {{ gap:1.25rem; border-bottom:1px solid var(--line); }}
    [data-testid="stTabs"] button[role="tab"] {{ color:var(--muted); padding:.75rem .15rem; }}
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
        color:var(--green); border-bottom-color:var(--green); }}
    [data-testid="stRadio"] div[role="radiogroup"] {{ gap:0!important; }}
    [data-testid="stRadio"] label[data-baseweb="radio"] {{
        min-height:2.35rem; padding:.35rem .8rem; margin:0!important;
        background:var(--white); border:1px solid var(--line); border-radius:0;
    }}
    [data-testid="stRadio"] label[data-baseweb="radio"]:first-of-type {{ border-radius:3px 0 0 3px; }}
    [data-testid="stRadio"] label[data-baseweb="radio"]:last-of-type {{ border-radius:0 3px 3px 0; }}
    [data-testid="stRadio"] label[data-baseweb="radio"] + label[data-baseweb="radio"] {{ margin-left:-1px!important; }}
    [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {{
        position:relative; z-index:1; background:var(--green-soft); border-color:var(--green);
    }}
    [data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {{ display:none; }}
    [data-testid="stMetric"] {{ background:var(--white); border:1px solid var(--line); padding:1rem; }}
    [data-testid="stMetricLabel"] {{ color:var(--muted); }}
    [data-testid="stMetricValue"] {{ color:var(--ink); font-family:'IBM Plex Mono',monospace; }}
    [data-testid="stForm"] {{ background:var(--white); border:1px solid var(--line); padding:1.2rem; }}
    [data-testid="stNumberInput"] input, [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stFileUploaderDropzone"] {{ background:var(--input)!important; color:var(--ink)!important;
        border-color:var(--line)!important; }}
    [data-testid="stFileUploaderDropzone"] > div, [data-testid="stFileUploaderDropzone"] > div * {{
        color:var(--muted)!important; }}
    [data-testid="stNumberInput"] button, [data-testid="stFileUploaderDropzone"] button {{
        background:var(--green-soft)!important; color:var(--ink)!important; border-color:var(--line)!important; }}
    [data-testid="stFileUploaderDropzone"] button * {{ color:var(--ink)!important; }}
    [data-testid="stSlider"] {{ accent-color:var(--green); }}
    [data-testid="stSlider"] [role="slider"] {{ background:var(--green)!important; border-color:var(--green)!important; }}
    [data-testid="stSlider"] [data-baseweb="slider"] > div > div {{ background:var(--green)!important; }}
    [data-testid="stSlider"] [data-baseweb="slider"] div[style*="height: 0.25rem"] {{
        background-image:none!important; background-color:var(--line)!important;
    }}
    [data-testid="stSliderThumbValue"], [data-testid="stSliderTickBar"],
    [data-testid="stSliderTickBar"] span {{ color:var(--muted)!important; }}
    .stButton button, [data-testid="stFormSubmitButton"] button {{ border-radius:3px; font-weight:600; min-height:2.7rem; }}
    .stButton button[kind="primary"], [data-testid="stFormSubmitButton"] button {{
        background:var(--green)!important; border-color:var(--green)!important; color:var(--white)!important; }}
    .report-table {{ width:100%; border-collapse:collapse; background:var(--white); color:var(--ink); font-size:.86rem; }}
    .report-table th, .report-table td {{ padding:.5rem; border-bottom:1px solid var(--line); color:var(--ink); text-align:left; }}
    .report-table thead th {{ background:var(--green-soft); }}
    @media(max-width:700px) {{ .block-container {{ padding:3.5rem 1rem 2rem; }}
        .masthead {{ align-items:flex-start; flex-direction:column; }} .masthead h1 {{ font-size:1.8rem; }} }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
        f"""<header class="masthead"><div>
            <div class="eyebrow">Risk intelligence / transaction monitoring</div>
            <h1>Credit card fraud detection</h1>
            <p>K-Means anomaly review · accounts farther from a cluster centroid receive higher risk scores.</p>
        </div><div class="service-tag">ACTIVE · K-MEANS</div></header>""",
        unsafe_allow_html=True,
)


def style_chart(figure):
    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color=theme["ink"]),
        title_font_size=16,
        margin=dict(l=55, r=20, t=55, b=45),
    )


def show_classification_report(data):
    report = data["classification_report"]
    frame = pd.DataFrame(
        [report[name] for name in ("Normal", "Anomaly")],
        index=["Normal", "Anomaly"],
    )
    frame.index.name = "Class"
    metrics = frame.loc[["Normal", "Anomaly"], ["precision", "recall", "f1-score"]].reset_index()
    metrics = metrics.melt(id_vars="Class", var_name="Metric", value_name="Score")

    rows = [{"Class": name, **report[name]} for name in ("Normal", "Anomaly")]
    rows.append({
        "Class": "accuracy", "precision": None, "recall": None,
        "f1-score": report["accuracy"], "support": data["test_samples"],
    })
    rows.extend({"Class": name, **report[name]} for name in ("macro avg", "weighted avg"))
    display = pd.DataFrame(rows)
    for column in ("precision", "recall", "f1-score"):
        display[column] = display[column].map(
            lambda value: f"{value:.3f}" if pd.notna(value) else ""
        )
    display["support"] = display["support"].map(lambda value: f"{int(value):,}")

    left, right = st.columns([1.15, 1])
    with left:
        figure = px.bar(
            metrics, x="Class", y="Score", color="Metric", barmode="group", range_y=[0, 1],
            title="Precision, recall and F1 by class",
            color_discrete_sequence=[theme["green"], theme["amber"], theme["normal"]],
        )
        style_chart(figure)
        figure.update_yaxes(gridcolor=theme["grid"], zeroline=False)
        st.plotly_chart(figure, use_container_width=True, theme=None)
    with right:
        st.markdown(display.to_html(index=False, classes="report-table", border=0), unsafe_allow_html=True)


def train_model(test_size, n_clusters, upload_bytes, dataset_name):
    config = {"test_size": test_size}
    parameters = {**config, "filename": dataset_name}
    config["n_clusters"] = n_clusters
    parameters["n_clusters"] = n_clusters

    if upload_bytes is None:
        response = requests.post(f"{API_URL}/train", json=config, timeout=120)
    else:
        response = requests.post(
            f"{API_URL}/train/upload",
            params=parameters,
            data=upload_bytes,
            headers={"Content-Type": "text/csv"},
            timeout=120,
        )
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise ValueError(f"K-Means Clustering: {detail}")

    result = response.json()
    result["dataset_name"] = dataset_name
    st.session_state["kmeans_results"] = result
    return result


tab_train, tab_predict = st.tabs(["Train and evaluate", "Assess an account"])
with tab_train:
    st.header("Training data")
    uploaded_file = st.file_uploader(
        "Upload a CSV dataset",
        type=["csv"],
        help="Maximum 25 MB. Required fields are listed below; additional columns are ignored.",
        key="kmeans_upload",
    )
    upload_bytes = None
    uploaded_df = None
    upload_is_valid = False
    if uploaded_file is not None:
        upload_bytes = uploaded_file.getvalue()
        if len(upload_bytes) > 25 * 1024 * 1024:
            st.error("This file exceeds the 25 MB upload limit.")
        else:
            try:
                uploaded_df = pd.read_csv(BytesIO(upload_bytes))
                missing = [column for column in FEATURE_COLUMNS if column not in uploaded_df.columns]
                if missing:
                    st.error(f"Missing required columns: {', '.join(missing)}")
                elif len(uploaded_df) < 10:
                    st.error("Add at least 10 data rows before training.")
                else:
                    upload_is_valid = True
                    st.caption(f"{uploaded_file.name} · {len(uploaded_df):,} rows · {len(uploaded_df.columns)} columns")
                    st.dataframe(uploaded_df.head(5), use_container_width=True, hide_index=True)
            except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
                st.error(f"Could not read this CSV: {error}")

    st.caption("Required numeric columns: " + " · ".join(FEATURE_COLUMNS))
    st.header("Detector settings")
    split_col, clusters_col = st.columns(2)
    with split_col:
        test_size = st.slider("Test split", 0.10, 0.40, 0.20, step=0.05)
    training_rows = len(uploaded_df) if uploaded_df is not None else 8950
    max_clusters = max(1, min(50, int(training_rows * (1 - test_size))))
    with clusters_col:
        n_clusters = st.number_input(
            "Number of clusters", min_value=1, max_value=max_clusters, value=min(3, max_clusters), step=1
        )

    dataset_name = uploaded_file.name if upload_is_valid and uploaded_file is not None else "Bundled credit-card sample"
    train_clicked = st.button(
        "Train K-Means Clustering",
        type="primary",
        disabled=uploaded_file is not None and not upload_is_valid,
        use_container_width=True,
    )

    if train_clicked:
        with st.spinner("Training K-Means Clustering..."):
            try:
                train_model(
                    test_size,
                    int(n_clusters),
                    upload_bytes if upload_is_valid else None,
                    dataset_name,
                )
                st.success("K-Means Clustering training complete.")
            except requests.RequestException as error:
                st.error(f"Could not connect to the K-Means API: {error}")
            except ValueError as error:
                st.error(str(error))

    data = st.session_state["kmeans_results"]
    if data is not None:
        st.caption(
            f"Source: {data.get('dataset_name', dataset_name)} · clusters = {data['n_clusters']} "
            f"· suspicious threshold = {data['suspicious_threshold']:.4f} "
            f"· high-risk threshold = {data['anomaly_threshold']:.4f}"
        )
        st.subheader("Run summary")
        metric_columns = st.columns(3)
        metric_columns[0].metric("Training rows", f"{data['train_samples']:,}")
        metric_columns[1].metric("Test rows", f"{data['test_samples']:,}")
        metric_columns[2].metric("ROC-AUC", f"{data['roc_curve']['auc']:.4f}")

        st.subheader("Evaluation")
        confusion_col, roc_col = st.columns(2)
        with confusion_col:
            figure = px.imshow(
                data["confusion_matrix"], text_auto=True,
                labels={"x": "Predicted", "y": "Heuristic label", "color": "Count"},
                x=["Normal", "Anomaly"], y=["Normal", "Anomaly"],
                color_continuous_scale=[[0, theme["green_soft"]], [1, theme["green"]]],
                title="Confusion matrix",
            )
            style_chart(figure)
            st.plotly_chart(figure, use_container_width=True, theme=None)
        with roc_col:
            roc = data["roc_curve"]
            figure = go.Figure()
            figure.add_trace(go.Scatter(
                x=roc["fpr"], y=roc["tpr"], mode="lines",
                name=f"K-means centroid distance (AUC={roc['auc']:.3f})",
                line={"color": theme["green"], "width": 3},
            ))
            figure.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="Baseline",
                line={"color": theme["muted"], "dash": "dash"},
            ))
            figure.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate")
            style_chart(figure)
            figure.update_xaxes(gridcolor=theme["grid"], zeroline=False)
            figure.update_yaxes(gridcolor=theme["grid"], zeroline=False)
            st.plotly_chart(figure, use_container_width=True, theme=None)

        st.subheader("Classification report")
        st.caption("Reference labels use high balance or cash-advance thresholds; they are heuristics, not verified fraud outcomes.")
        show_classification_report(data)

        score_frame = pd.DataFrame(data["score_distribution"])
        score_column = "anomaly_score"
        score_title = "Centroid-distance distribution"
        score_description = "Greater distance from the nearest centroid indicates higher anomaly risk"
        score_frame["classification"] = score_frame["is_anomaly"].map({0: "Normal", 1: "Anomaly"})
        st.subheader(score_title)
        score_figure = px.histogram(
            score_frame, x=score_column, color="classification", barmode="overlay",
            opacity=0.72, nbins=40, color_discrete_map={"Normal": theme["normal"], "Anomaly": theme["amber"]},
            labels={score_column: "Nearest-centroid distance", "classification": "Prediction"},
            title=score_description,
        )
        threshold = data["anomaly_threshold"]
        score_figure.add_vline(
            x=threshold,
            line_dash="dash",
            line_color=theme["ink"],
            annotation_text="Threshold",
        )
        style_chart(score_figure)
        score_figure.update_xaxes(gridcolor=theme["grid"], zeroline=False)
        score_figure.update_yaxes(gridcolor=theme["grid"], zeroline=False)
        st.plotly_chart(score_figure, use_container_width=True, theme=None)

        pca_frame = pd.DataFrame(data["pca_scatter"])
        if {"risk_level", "anomaly_score", "cluster_id"}.issubset(pca_frame.columns):
            st.subheader("K-Means clusters")
            st.caption(
                f"Each point is assigned to one of {data['n_clusters']} learned clusters. "
                "PCA projects the account features into two dimensions for display."
            )
            cluster_frame = pca_frame.copy()
            cluster_frame["cluster"] = cluster_frame["cluster_id"].map(
                lambda cluster_id: f"Cluster {cluster_id + 1}"
            )
            cluster_order = [
                f"Cluster {cluster_id + 1}" for cluster_id in range(data["n_clusters"])
            ]
            cluster_figure = px.scatter(
                cluster_frame,
                x="pca1",
                y="pca2",
                color="cluster",
                category_orders={"cluster": cluster_order},
                color_discrete_sequence=px.colors.qualitative.Plotly,
                hover_data={
                    "cluster": True,
                    "anomaly_score": ":.3f",
                    "pca1": ":.3f",
                    "pca2": ":.3f",
                },
                labels={
                    "cluster": "K-Means group",
                    "pca1": "PCA component 1",
                    "pca2": "PCA component 2",
                    "anomaly_score": "Centroid distance",
                },
                title="Account groups found by K-Means",
            )
            cluster_figure.update_traces(marker={"size": 8, "opacity": 0.85})
            style_chart(cluster_figure)
            cluster_figure.update_layout(height=500)
            cluster_figure.update_xaxes(gridcolor=theme["grid"], zeroline=False)
            cluster_figure.update_yaxes(gridcolor=theme["grid"], zeroline=False)
            st.plotly_chart(cluster_figure, use_container_width=True, theme=None)

            st.subheader("K-Means risk clusters")
            st.caption(
                "Risk groups use the training-set centroid-distance thresholds; they are not verified fraud labels."
            )
            risk_order = ["NORMAL", "SUSPICIOUS", "HIGH RISK"]
            risk_frame = pca_frame.copy()
            risk_frame["risk_level"] = pd.Categorical(
                risk_frame["risk_level"],
                categories=risk_order,
                ordered=True,
            )
            risk_figure = px.scatter(
                risk_frame,
                x="pca1",
                y="pca2",
                color="risk_level",
                category_orders={"risk_level": risk_order},
                color_discrete_map={
                    "NORMAL": theme["normal"],
                    "SUSPICIOUS": theme["amber"],
                    "HIGH RISK": theme["high_risk"],
                },
                hover_data={
                    "risk_level": True,
                    "cluster_id": True,
                    "anomaly_score": ":.3f",
                    "pca1": ":.3f",
                    "pca2": ":.3f",
                },
                labels={
                    "risk_level": "Risk group",
                    "cluster_id": "K-Means cluster",
                    "anomaly_score": "Centroid distance",
                    "pca1": "PCA component 1",
                    "pca2": "PCA component 2",
                },
                title="Normal, Suspicious, and High Risk accounts",
            )
            style_chart(risk_figure)
            risk_figure.update_xaxes(gridcolor=theme["grid"], zeroline=False)
            risk_figure.update_yaxes(gridcolor=theme["grid"], zeroline=False)
            st.plotly_chart(risk_figure, use_container_width=True, theme=None)
        else:
            st.info("Retrain K-Means to generate the three risk groups for this chart.")

        if "risk_level" in pca_frame:
            pca_frame["classification"] = pca_frame["risk_level"].map(
                {"NORMAL": "Normal", "SUSPICIOUS": "Suspicious", "HIGH RISK": "Anomaly"}
            )
        else:
            pca_frame["classification"] = pca_frame["is_anomaly"].map(
                {0: "Normal", 1: "Anomaly"}
            )
        show_records = st.selectbox("PCA records", ["All records", "Anomalies only", "Normal only"])
        if show_records == "Anomalies only":
            pca_frame = pca_frame[pca_frame["is_anomaly"] == 1]
        elif show_records == "Normal only":
            pca_frame = pca_frame[pca_frame["is_anomaly"] == 0]
        st.caption(f"Showing {len(pca_frame):,} test rows")
        record_columns = [
            column
            for column in (
                "classification",
                "risk_level",
                "cluster_id",
                "anomaly_score",
                "pca1",
                "pca2",
            )
            if column in pca_frame.columns
        ]
        pca_records = pca_frame[record_columns].rename(
            columns={
                "classification": "Prediction",
                "risk_level": "Risk level",
                "cluster_id": "K-Means cluster",
                "anomaly_score": "Centroid distance",
                "pca1": "PCA component 1",
                "pca2": "PCA component 2",
            }
        )
        if "K-Means cluster" in pca_records:
            pca_records["K-Means cluster"] = pca_records["K-Means cluster"].map(
                lambda cluster_id: f"Cluster {cluster_id + 1}"
            )
        st.dataframe(pca_records, hide_index=True, width="stretch")

with tab_predict:
    st.header("Assess an account")
    st.markdown("Accounts farther from their nearest cluster centroid receive higher anomaly scores.")
    data = st.session_state["kmeans_results"]
    example_options = ["Manual entry"]
    if data is not None:
        example_options.extend(data["dataset_examples"])

    def load_dataset_example():
        example_name = st.session_state["account_example"]
        if example_name != "Manual entry":
            for column, value in data["dataset_examples"][example_name].items():
                st.session_state[f"account_{column}"] = value

    st.selectbox(
        "Quick-load an account",
        example_options,
        help="Load a representative account from the dataset used for the latest training run.",
        key="account_example",
        on_change=load_dataset_example if data is not None else None,
    )
    if data is None:
        st.caption("Train the model first to load suspicious and high-risk examples from the selected dataset.")

    default_values = {
        "BALANCE": 40.90,
        "PURCHASES": 95.40,
        "INSTALLMENTS_PURCHASES": 95.40,
        "CASH_ADVANCE": 0.0,
        "CREDIT_LIMIT": 1000.0,
        "PAYMENTS": 201.80,
        "MINIMUM_PAYMENTS": 139.51,
    }
    for column, value in default_values.items():
        st.session_state.setdefault(f"account_{column}", value)
    st.session_state.setdefault("account_TENURE", 12)

    with st.form("kmeans_prediction_form"):
        left, right = st.columns(2)
        with left:
            balance = st.number_input("Account balance ($)", step=50.0, key="account_BALANCE")
            purchases = st.number_input("Total purchases ($)", step=50.0, key="account_PURCHASES")
            installments = st.number_input("Installment purchases ($)", step=50.0, key="account_INSTALLMENTS_PURCHASES")
            cash_advance = st.number_input("Cash advance ($)", step=100.0, key="account_CASH_ADVANCE")
        with right:
            credit_limit = st.number_input("Credit limit ($)", step=500.0, key="account_CREDIT_LIMIT")
            payments = st.number_input("Total payments ($)", step=50.0, key="account_PAYMENTS")
            min_payments = st.number_input("Minimum payments ($)", step=20.0, key="account_MINIMUM_PAYMENTS")
            tenure = st.number_input("Tenure (months)", min_value=0, step=1, key="account_TENURE")
        predict_clicked = st.form_submit_button("Evaluate account", type="primary")

    if predict_clicked:
        payload = {
            "BALANCE": balance,
            "PURCHASES": purchases,
            "INSTALLMENTS_PURCHASES": installments,
            "CASH_ADVANCE": cash_advance,
            "CREDIT_LIMIT": credit_limit,
            "PAYMENTS": payments,
            "MINIMUM_PAYMENTS": min_payments,
            "TENURE": tenure,
        }
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
            if response.ok:
                result = response.json()
                if result["risk_level"] == "HIGH RISK":
                    st.error(f"Status: {result['status']}")
                elif result["risk_level"] == "SUSPICIOUS":
                    st.warning(f"Status: {result['status']}")
                else:
                    st.success(f"Status: {result['status']}")
                st.metric("Nearest-centroid distance", result["decision_score"], help=result["score_direction"])
                st.caption(f"Training threshold: {result['threshold']:.4f}")
            else:
                st.error(response.json().get("detail", response.text))
        except requests.RequestException as error:
            st.error(f"Cannot connect to the K-Means API at {API_URL}: {error}")
