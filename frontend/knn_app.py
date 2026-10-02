from io import BytesIO
import hashlib

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

MODEL_APIS = {
    "Isolation Forest": "http://127.0.0.1:8000",
    "K-Nearest Neighbors": "http://127.0.0.1:8001",
}
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

st.set_page_config(page_title="KNN Card Anomaly Review", layout="wide")

_, model_column, appearance_column = st.columns([2, 5, 2])
with model_column:
    selected_model = st.radio(
        "Model",
        list(MODEL_APIS),
        horizontal=True,
        label_visibility="collapsed",
        key="selected_anomaly_model",
    )

with appearance_column:
    appearance = st.radio(
        "Appearance",
        ["Light", "Dark"],
        horizontal=True,
        label_visibility="collapsed",
        key="knn_appearance",
    )

is_knn = selected_model == "K-Nearest Neighbors"
API_URL = MODEL_APIS[selected_model]
if "model_results" not in st.session_state:
    st.session_state["model_results"] = {}
if "knn_eval_data" in st.session_state and "K-Nearest Neighbors" not in st.session_state["model_results"]:
    st.session_state["model_results"]["K-Nearest Neighbors"] = st.session_state.pop("knn_eval_data")

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
    .masthead {{ display:flex; justify-content:space-between; align-items:flex-end; gap:1rem;
        padding:1.25rem 0 1.4rem; margin-bottom:1.2rem; border-bottom:1px solid var(--line); }}
    .eyebrow {{ color:var(--green); font:500 .72rem 'IBM Plex Mono',monospace;
        text-transform:uppercase; margin:0 0 .55rem; }}
    .masthead h1 {{ font-size:2rem; margin:0; font-weight:600; }}
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
        .masthead {{ align-items:flex-start; flex-direction:column; }} .masthead h1 {{ font-size:1.55rem; }} }}
    </style>
    """,
    unsafe_allow_html=True,
)

model_summary = (
        "Accounts with unusually distant neighbors receive higher anomaly scores."
        if is_knn
        else "Isolation Forest separates accounts by how isolated their feature patterns are."
)
st.markdown(
        f"""<header class="masthead"><div>
            <div class="eyebrow">Risk intelligence / transaction monitoring</div>
            <h1>{selected_model} anomaly review</h1>
            <p>{model_summary}</p>
        </div><div class="service-tag">ACTIVE · {selected_model.upper()}</div></header>""",
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


def train_model(model_name, contamination, test_size, n_neighbors, upload_bytes, dataset_name):
    api_url = MODEL_APIS[model_name]
    config = {"contamination": contamination, "test_size": test_size}
    parameters = {**config, "filename": dataset_name}
    if model_name == "K-Nearest Neighbors":
        config["n_neighbors"] = n_neighbors
        parameters["n_neighbors"] = n_neighbors

    if upload_bytes is None:
        response = requests.post(f"{api_url}/train", json=config, timeout=120)
    else:
        response = requests.post(
            f"{api_url}/train/upload",
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
        raise ValueError(f"{model_name}: {detail}")

    result = response.json()
    result["comparison_signature"] = (
        hashlib.sha256(upload_bytes).hexdigest() if upload_bytes is not None else "bundled-credit-card.csv",
        contamination,
        test_size,
    )
    result["dataset_name"] = dataset_name
    st.session_state["model_results"][model_name] = result
    return result


tab_train, tab_predict = st.tabs(["Train and evaluate", "Assess an account"])
with tab_train:
    st.header("Training data")
    uploaded_file = st.file_uploader(
        "Upload a CSV dataset",
        type=["csv"],
        help="Maximum 25 MB. Required fields are listed below; additional columns are ignored.",
        key="knn_upload",
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
    contamination_col, split_col, k_col = st.columns(3)
    with contamination_col:
        contamination = st.slider("Expected anomaly rate", 0.01, 0.10, 0.03, step=0.01)
    with split_col:
        test_size = st.slider("Test split", 0.10, 0.40, 0.20, step=0.05)
    training_rows = len(uploaded_df) if uploaded_df is not None else 8950
    max_k = max(1, min(50, int(training_rows * (1 - test_size)) - 1))
    with k_col:
        n_neighbors = st.number_input(
            "Neighbors (k, KNN)", min_value=1, max_value=max_k, value=min(10, max_k), step=1
        )

    dataset_name = uploaded_file.name if upload_is_valid and uploaded_file is not None else "Bundled credit-card sample"
    train_col, compare_col = st.columns(2)
    with train_col:
        train_clicked = st.button(
            f"Train {selected_model}",
            type="primary",
            disabled=uploaded_file is not None and not upload_is_valid,
            use_container_width=True,
        )
    with compare_col:
        compare_clicked = st.button(
            "Train and compare both",
            disabled=uploaded_file is not None and not upload_is_valid,
            use_container_width=True,
        )

    if train_clicked or compare_clicked:
        models_to_train = list(MODEL_APIS) if compare_clicked else [selected_model]
        with st.spinner("Training model(s) on the same dataset and holdout split..."):
            try:
                for model_name in models_to_train:
                    train_model(
                        model_name,
                        contamination,
                        test_size,
                        int(n_neighbors),
                        upload_bytes if upload_is_valid else None,
                        dataset_name,
                    )
                st.success("Training complete for " + " and ".join(models_to_train) + ".")
            except requests.RequestException as error:
                st.error(f"Could not connect to a model API: {error}")
            except ValueError as error:
                st.error(str(error))

    model_results = st.session_state["model_results"]
    if all(model in model_results for model in MODEL_APIS):
        isolation_result = model_results["Isolation Forest"]
        knn_result = model_results["K-Nearest Neighbors"]
        if isolation_result.get("comparison_signature") == knn_result.get("comparison_signature"):
            isolation_auc = isolation_result["roc_curve"]["auc"]
            knn_auc = knn_result["roc_curve"]["auc"]
            isolation_f1 = isolation_result["classification_report"]["Anomaly"]["f1-score"]
            knn_f1 = knn_result["classification_report"]["Anomaly"]["f1-score"]
            st.subheader("Which model performed better?")
            if abs(isolation_auc - knn_auc) < 0.01:
                st.info(
                    f"The models are close on ROC-AUC ({isolation_auc:.3f} vs {knn_auc:.3f}). "
                    f"Anomaly F1 is {isolation_f1:.3f} for Isolation Forest and {knn_f1:.3f} for KNN; "
                    "neither is a clear overall winner on this holdout."
                )
            else:
                winner = "Isolation Forest" if isolation_auc > knn_auc else "K-Nearest Neighbors"
                winner_auc = max(isolation_auc, knn_auc)
                winner_f1 = isolation_f1 if winner == "Isolation Forest" else knn_f1
                st.info(
                    f"{winner} performed better on this holdout: ROC-AUC {winner_auc:.3f}. "
                    f"Its anomaly-class F1 is {winner_f1:.3f}. ROC-AUC measures how well risk scores "
                    "rank the heuristic high-risk examples across thresholds."
                )
            comparison = pd.DataFrame([
                {"Model": "Isolation Forest", "ROC-AUC": isolation_auc, "Anomaly F1": isolation_f1},
                {"Model": "K-Nearest Neighbors", "ROC-AUC": knn_auc, "Anomaly F1": knn_f1},
            ])
            st.dataframe(comparison, use_container_width=True, hide_index=True)
            st.caption("Comparison uses the same dataset and split. Evaluation labels are balance/cash-advance heuristics, not verified fraud outcomes.")
        else:
            st.warning("These results use different data or split settings. Select ‘Train and compare both’ to compare them fairly.")

    if selected_model in model_results:
        data = model_results[selected_model]
        if is_knn:
            st.caption(f"Source: {data.get('dataset_name', dataset_name)} · k = {data['n_neighbors']} · anomaly threshold = {data['anomaly_threshold']:.4f}")
        else:
            st.caption(f"Source: {data.get('dataset_name', dataset_name)} · Isolation Forest")
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
                name=f"{'KNN distance' if is_knn else 'Isolation Forest'} (AUC={roc['auc']:.3f})",
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
        score_column = "anomaly_score" if is_knn else "decision_score"
        score_title = (
            "Neighbor-distance distribution"
            if is_knn
            else "Isolation Forest decision-score distribution"
        )
        score_description = (
            "Higher distance indicates higher anomaly risk"
            if is_knn
            else "More negative scores indicate higher anomaly risk"
        )
        score_frame["classification"] = score_frame["is_anomaly"].map({0: "Normal", 1: "Anomaly"})
        st.subheader(score_title)
        score_figure = px.histogram(
            score_frame, x=score_column, color="classification", barmode="overlay",
            opacity=0.72, nbins=40, color_discrete_map={"Normal": theme["normal"], "Anomaly": theme["amber"]},
            labels={score_column: "Mean neighbor distance" if is_knn else "Decision score", "classification": "Prediction"},
            title=score_description,
        )
        threshold = data["anomaly_threshold"] if is_knn else 0
        score_figure.add_vline(
            x=threshold,
            line_dash="dash",
            line_color=theme["ink"],
            annotation_text="Threshold" if is_knn else "Zero score",
        )
        style_chart(score_figure)
        score_figure.update_xaxes(gridcolor=theme["grid"], zeroline=False)
        score_figure.update_yaxes(gridcolor=theme["grid"], zeroline=False)
        st.plotly_chart(score_figure, use_container_width=True, theme=None)

        pca_frame = pd.DataFrame(data["pca_scatter"])
        pca_frame["classification"] = pca_frame["is_anomaly"].map({0: "Normal", 1: "Anomaly"})
        show_records = st.selectbox("PCA records", ["All records", "Anomalies only", "Normal only"])
        if show_records == "Anomalies only":
            pca_frame = pca_frame[pca_frame["is_anomaly"] == 1]
        elif show_records == "Normal only":
            pca_frame = pca_frame[pca_frame["is_anomaly"] == 0]
        st.caption(f"Showing {len(pca_frame):,} test rows")
        pca_figure = px.scatter(
            pca_frame, x="pca1", y="pca2", color="classification",
            color_discrete_map={"Normal": theme["normal"], "Anomaly": theme["amber"]},
            hover_data={"classification": True, "pca1": ":.3f", "pca2": ":.3f"},
            title="PCA projection of detected anomalies",
        )
        style_chart(pca_figure)
        pca_figure.update_xaxes(gridcolor=theme["grid"], zeroline=False)
        pca_figure.update_yaxes(gridcolor=theme["grid"], zeroline=False)
        st.plotly_chart(pca_figure, use_container_width=True, theme=None)

with tab_predict:
    st.header("Assess an account")
    prediction_guidance = (
        "Accounts farther from their nearest training examples receive higher anomaly scores."
        if is_knn
        else "Isolation Forest assigns lower decision scores to more isolated account patterns."
    )
    st.markdown(prediction_guidance)
    with st.form("knn_prediction_form"):
        left, right = st.columns(2)
        with left:
            balance = st.number_input("Account balance ($)", value=40.90, step=50.0)
            purchases = st.number_input("Total purchases ($)", value=95.40, step=50.0)
            installments = st.number_input("Installment purchases ($)", value=95.40, step=50.0)
            cash_advance = st.number_input("Cash advance ($)", value=0.0, step=100.0)
        with right:
            credit_limit = st.number_input("Credit limit ($)", value=1000.0, step=500.0)
            payments = st.number_input("Total payments ($)", value=201.80, step=50.0)
            min_payments = st.number_input("Minimum payments ($)", value=139.51, step=20.0)
            tenure = st.slider("Tenure (months)", 6, 12, 12)
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
                if result["is_anomaly"]:
                    st.error(f"Status: {result['status']}")
                else:
                    st.success(f"Status: {result['status']}")
                if is_knn:
                    st.metric("Mean neighbor distance", result["decision_score"], help=result["score_direction"])
                    st.caption(f"Training threshold: {result['threshold']:.4f}")
                else:
                    st.metric("Isolation Forest decision score", result["decision_score"])
                    st.caption("More negative decision scores indicate higher anomaly risk.")
            else:
                st.error(response.json().get("detail", response.text))
        except requests.RequestException as error:
            st.error(f"Cannot connect to KNN API at {API_URL}: {error}")
