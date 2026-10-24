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

st.set_page_config(page_title="Credit card KNN anomaly detection", layout="wide")

_, appearance_column = st.columns([7, 2])
with appearance_column:
    appearance = st.radio(
        "Appearance",
        ["Light", "Dark"],
        horizontal=True,
        label_visibility="collapsed",
        key="knn_appearance",
    )

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
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
    .stApp {{ background: var(--paper); color: var(--ink); font-family: 'DM Sans', sans-serif; }}
    [data-testid="stHeader"], [data-testid="stAppViewContainer"] > .main {{ background: transparent; }}
    .block-container {{ max-width: 1440px; padding-top: 4.5rem; padding-bottom: 3rem; }}
    h1, h2, h3, p, label {{ color: var(--ink); font-family: 'DM Sans', sans-serif; }}
    .masthead {{ display:flex; justify-content:space-between; align-items:flex-end; gap:1rem;
        padding:1.25rem 0 1.4rem; margin-bottom:1.2rem; border-bottom:1px solid var(--line); }}
    .eyebrow {{ color:var(--green); font: .72rem 'IBM Plex Mono',monospace; letter-spacing:.08em;
        text-transform:uppercase; margin:0 0 .55rem; }}
    .masthead h1 {{ font-size:2rem; line-height:1.2; margin:0; font-weight:600; }}
    .masthead p {{ color:var(--muted); margin:.55rem 0 0; font-size:.95rem; }}
    .service-tag {{ border:1px solid var(--line); background:var(--white); color:var(--muted);
        padding:.45rem .7rem; font:.72rem 'IBM Plex Mono',monospace; white-space:nowrap; }}
    [data-testid="stTabs"] [role="tablist"] {{ gap:1.25rem; border-bottom:1px solid var(--line); }}
    [data-testid="stTabs"] button[role="tab"] {{ color:var(--muted); padding:.75rem .15rem; }}
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
        color:var(--green); border-bottom-color:var(--green); }}
    [data-testid="stTabs"] button[role="tab"] p {{ font-weight:600; }}
    h2 {{ font-size:1.25rem !important; margin-top:1.4rem !important; }}
    [data-testid="stMetric"] {{ background:var(--white); border:1px solid var(--line); padding:1rem 1.1rem; }}
    [data-testid="stMetricLabel"] {{ color:var(--muted); }}
    [data-testid="stMetricValue"] {{ color:var(--ink); font-family:'IBM Plex Mono',monospace; }}
    [data-testid="stForm"] {{ background:var(--white); border:1px solid var(--line); padding:1.2rem 1.35rem; }}
    [data-testid="stNumberInput"] input, [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stFileUploaderDropzone"] {{ background:var(--input)!important; color:var(--ink)!important;
        border-color:var(--line)!important; }}
    [data-testid="stFileUploaderDropzone"] > div, [data-testid="stFileUploaderDropzone"] > div * {{ color:var(--muted)!important; }}
    [data-testid="stNumberInput"] button, [data-testid="stFileUploaderDropzone"] button {{
        background:var(--green-soft)!important; color:var(--ink)!important; border-color:var(--line)!important; }}
    [data-testid="stFileUploaderDropzone"] button * {{ color:var(--ink)!important; }}
    [data-testid="stSlider"] {{ accent-color:var(--green); }}
    [data-testid="stSlider"] [role="slider"] {{ background:var(--green)!important; border-color:var(--green)!important; }}
    [data-testid="stSlider"] [data-baseweb="slider"] > div > div {{ background:var(--green)!important; }}
    .stButton button, [data-testid="stFormSubmitButton"] button {{ border-radius:3px; font-weight:600; min-height:2.7rem; }}
    .stButton button[kind="primary"], [data-testid="stFormSubmitButton"] button {{
        background:var(--green)!important; border-color:var(--green)!important; color:var(--white)!important; }}
    .report-table {{ width:100%; border-collapse:collapse; background:var(--white); color:var(--ink); font-size:.88rem; }}
    .report-table th, .report-table td {{ padding:.55rem .65rem; border-bottom:1px solid var(--line); color:var(--ink); text-align:left; }}
    .report-table thead th {{ background:var(--green-soft); }}
    [data-testid="stDataFrame"] {{ border:1px solid var(--line); }}
    @media(max-width:700px) {{ .block-container {{ padding:3.5rem 1rem 2rem; }}
        .masthead {{ align-items:flex-start; flex-direction:column; }} .masthead h1 {{ font-size:1.55rem; }} }}
    </style>
    <header class="masthead">
        <div>
            <div class="eyebrow">Risk intelligence / transaction monitoring</div>
            <h1>Card anomaly review</h1>
            <p>Train the KNN detector, inspect model performance, and assess account activity.</p>
        </div>
        <div class="service-tag">MODEL · KNN</div>
    </header>
    """,
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


def train_model(test_size, n_neighbors, upload_bytes, dataset_name):
    config = {"test_size": test_size, "n_neighbors": n_neighbors}
    if upload_bytes is None:
        response = requests.post(f"{API_URL}/train", json=config, timeout=120)
    else:
        response = requests.post(
            f"{API_URL}/train/upload",
            params={**config, "filename": dataset_name},
            data=upload_bytes,
            headers={"Content-Type": "text/csv"},
            timeout=120,
        )
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise ValueError(f"KNN training failed: {detail}")
    result = response.json()
    result["dataset_name"] = dataset_name
    st.session_state["knn_results"] = result
    return result


def show_classification_report(data):
    report = data["classification_report"]
    rows = [{"Class": name, **report[name]} for name in ("Normal", "Anomaly")]
    rows.append(
        {
            "Class": "accuracy",
            "precision": None,
            "recall": None,
            "f1-score": report["accuracy"],
            "support": data["test_samples"],
        }
    )
    rows.extend({"Class": name, **report[name]} for name in ("macro avg", "weighted avg"))
    frame = pd.DataFrame(rows)
    for column in ("precision", "recall", "f1-score"):
        frame[column] = frame[column].map(
            lambda value: f"{value:.3f}" if pd.notna(value) else ""
        )
    frame["support"] = frame["support"].map(lambda value: f"{int(value):,}")
    st.dataframe(frame, hide_index=True, use_container_width=True)


if "knn_results" not in st.session_state:
    st.session_state["knn_results"] = None

train_tab, predict_tab = st.tabs(["Train and evaluate", "Assess an account"])

with train_tab:
    uploaded_file = st.file_uploader(
        "Upload a CSV dataset",
        type=["csv"],
        help="Maximum 25 MB. Required columns are listed below.",
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
                    st.caption(f"{uploaded_file.name} · {len(uploaded_df):,} rows")
                    st.dataframe(uploaded_df.head(), hide_index=True, use_container_width=True)
            except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
                st.error(f"Could not read this CSV: {error}")

    st.caption("Required numeric columns: " + " · ".join(FEATURE_COLUMNS))
    split_column, neighbor_column = st.columns(2)
    with split_column:
        test_size = st.slider("Test split", 0.10, 0.40, 0.20, step=0.05)
    training_rows = len(uploaded_df) if uploaded_df is not None else 8950
    max_neighbors = max(1, min(50, int(training_rows * (1 - test_size))))
    with neighbor_column:
        n_neighbors = st.number_input(
            "Number of nearest neighbors",
            min_value=1,
            max_value=max_neighbors,
            value=min(5, max_neighbors),
            step=1,
        )

    dataset_name = (
        uploaded_file.name
        if upload_is_valid and uploaded_file is not None
        else "Bundled credit-card sample"
    )
    train_clicked = st.button(
        "Train KNN",
        type="primary",
        disabled=uploaded_file is not None and not upload_is_valid,
        use_container_width=True,
    )
    if train_clicked:
        with st.spinner("Training KNN..."):
            try:
                train_model(
                    test_size,
                    int(n_neighbors),
                    upload_bytes if upload_is_valid else None,
                    dataset_name,
                )
                st.success("KNN training complete.")
            except requests.RequestException as error:
                st.error(f"Could not connect to the KNN API: {error}")
            except ValueError as error:
                st.error(str(error))

    data = st.session_state["knn_results"]
    if data is not None:
        st.caption(
            f"Source: {data['dataset_name']} · neighbors = {data['n_neighbors']} · "
            f"suspicious threshold = {data['suspicious_threshold']:.4f} · "
            f"high-risk threshold = {data['anomaly_threshold']:.4f}"
        )
        summary = st.columns(3)
        summary[0].metric("Training rows", f"{data['train_samples']:,}")
        summary[1].metric("Test rows", f"{data['test_samples']:,}")
        summary[2].metric("ROC-AUC", f"{data['roc_curve']['auc']:.4f}")

        confusion, roc = st.columns(2)
        with confusion:
            figure = px.imshow(
                data["confusion_matrix"],
                text_auto=True,
                x=["Normal", "Anomaly"],
                y=["Normal", "Anomaly"],
                labels={"x": "Predicted", "y": "Heuristic label", "color": "Count"},
                title="Confusion matrix",
            )
            style_chart(figure)
            st.plotly_chart(figure, use_container_width=True, theme=None)
        with roc:
            curve = data["roc_curve"]
            figure = go.Figure()
            figure.add_trace(
                go.Scatter(
                    x=curve["fpr"],
                    y=curve["tpr"],
                    mode="lines",
                    name=f"KNN (AUC={curve['auc']:.3f})",
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Baseline",
                    line={"dash": "dash"},
                )
            )
            figure.update_layout(
                title="ROC curve",
                xaxis_title="False positive rate",
                yaxis_title="True positive rate",
            )
            style_chart(figure)
            figure.update_xaxes(gridcolor=theme["grid"], zeroline=False)
            figure.update_yaxes(gridcolor=theme["grid"], zeroline=False)
            st.plotly_chart(figure, use_container_width=True, theme=None)

        st.subheader("Classification report")
        st.caption(
            "Reference labels are heuristics based on balance and cash advance, "
            "not verified fraud outcomes."
        )
        show_classification_report(data)

        scores = pd.DataFrame(data["score_distribution"])
        scores["classification"] = scores["is_anomaly"].map({0: "Normal", 1: "Anomaly"})
        figure = px.histogram(
            scores,
            x="anomaly_score",
            color="classification",
            barmode="overlay",
            nbins=40,
            title="Average nearest-neighbor distance",
            labels={"anomaly_score": "Average neighbor distance"},
        )
        figure.add_vline(x=data["anomaly_threshold"], line_dash="dash", annotation_text="Threshold")
        style_chart(figure)
        figure.update_xaxes(gridcolor=theme["grid"], zeroline=False)
        figure.update_yaxes(gridcolor=theme["grid"], zeroline=False)
        st.plotly_chart(figure, use_container_width=True, theme=None)

        pca_frame = pd.DataFrame(data["pca_scatter"])
        figure = px.scatter(
            pca_frame,
            x="pca1",
            y="pca2",
            color="risk_level",
            hover_data=["anomaly_score"],
            title="PCA projection of test accounts",
            labels={"pca1": "PCA component 1", "pca2": "PCA component 2"},
        )
        style_chart(figure)
        figure.update_xaxes(gridcolor=theme["grid"], zeroline=False)
        figure.update_yaxes(gridcolor=theme["grid"], zeroline=False)
        st.plotly_chart(figure, use_container_width=True, theme=None)

with predict_tab:
    st.markdown(
        "Enter an account's feature values. The saved KNN model compares it with "
        "nearby training accounts."
    )
    data = st.session_state["knn_results"]
    examples = ["Manual entry"]
    if data is not None:
        examples.extend(data["dataset_examples"])
    selected_example = st.selectbox("Example", examples)

    defaults = {
        "BALANCE": 40.90,
        "PURCHASES": 95.40,
        "INSTALLMENTS_PURCHASES": 95.40,
        "CASH_ADVANCE": 0.0,
        "CREDIT_LIMIT": 1000.0,
        "PAYMENTS": 201.80,
        "MINIMUM_PAYMENTS": 139.51,
        "TENURE": 12,
    }
    if selected_example != "Manual entry" and data is not None:
        defaults.update(data["dataset_examples"][selected_example])

    with st.form("knn_prediction_form"):
        left, right = st.columns(2)
        with left:
            balance = st.number_input("Account balance ($)", value=float(defaults["BALANCE"]), step=50.0)
            purchases = st.number_input("Total purchases ($)", value=float(defaults["PURCHASES"]), step=50.0)
            installments = st.number_input(
                "Installment purchases ($)",
                value=float(defaults["INSTALLMENTS_PURCHASES"]),
                step=50.0,
            )
            cash_advance = st.number_input(
                "Cash advance ($)", value=float(defaults["CASH_ADVANCE"]), step=100.0
            )
        with right:
            credit_limit = st.number_input("Credit limit ($)", value=float(defaults["CREDIT_LIMIT"]), step=500.0)
            payments = st.number_input("Total payments ($)", value=float(defaults["PAYMENTS"]), step=50.0)
            min_payments = st.number_input(
                "Minimum payments ($)", value=float(defaults["MINIMUM_PAYMENTS"]), step=20.0
            )
            tenure = st.number_input("Tenure (months)", value=int(defaults["TENURE"]), min_value=0, step=1)
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
            if not response.ok:
                st.error(response.json().get("detail", response.text))
            else:
                result = response.json()
                message = f"Status: {result['status']}"
                if result["risk_level"] == "HIGH RISK":
                    st.error(message)
                elif result["risk_level"] == "SUSPICIOUS":
                    st.warning(message)
                else:
                    st.success(message)
                st.metric(
                    "Average neighbor distance",
                    result["decision_score"],
                    help=result["score_direction"],
                )
                st.caption(
                    f"Suspicious threshold: {result['suspicious_threshold']:.4f} · "
                    f"High-risk threshold: {result['threshold']:.4f}"
                )
        except requests.RequestException as error:
            st.error(f"Cannot connect to the KNN API at {API_URL}: {error}")
