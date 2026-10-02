import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from io import BytesIO

API_URL = "http://127.0.0.1:8000"
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

st.set_page_config(page_title="Credit Card Anomaly Detector", layout="wide")

_, appearance_column = st.columns([7, 2])
with appearance_column:
    appearance = st.radio(
        "Appearance",
        ["Light", "Dark"],
        horizontal=True,
        label_visibility="collapsed",
        key="appearance",
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
        --ink: {theme['ink']};
        --muted: {theme['muted']};
        --line: {theme['line']};
        --paper: {theme['paper']};
        --white: {theme['surface']};
        --input: {theme['input']};
        --green: {theme['green']};
        --green-soft: {theme['green_soft']};
        --amber: {theme['amber']};
        --primary-color: {theme['green']};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

    .stApp {
        background: var(--paper);
        color: var(--ink);
        font-family: 'DM Sans', sans-serif;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stAppViewContainer"] > .main { background: var(--paper); }
    .block-container { max-width: 1440px; padding-top: 4.5rem; padding-bottom: 3rem; }
    h1, h2, h3, p, label { color: var(--ink); font-family: 'DM Sans', sans-serif; }
    .masthead {
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        gap: 1rem;
        padding: 1.25rem 0 1.4rem;
        margin-bottom: 1.2rem;
        border-bottom: 1px solid var(--line);
    }
    .eyebrow {
        color: var(--green);
        font-family: 'IBM Plex Mono', monospace;
        font-size: .72rem;
        letter-spacing: .08em;
        text-transform: uppercase;
        margin: 0 0 .55rem;
    }
    .masthead h1 { font-size: 2rem; line-height: 1.2; margin: 0; font-weight: 600; }
    .masthead p { color: var(--muted); margin: .55rem 0 0; font-size: .95rem; }
    .service-tag {
        border: 1px solid var(--line);
        background: var(--white);
        color: var(--muted);
        padding: .45rem .7rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: .72rem;
        white-space: nowrap;
    }
    [data-testid="stTabs"] [role="tablist"] { gap: 1.25rem; border-bottom: 1px solid var(--line); }
    [data-testid="stTabs"] button[role="tab"] { color: var(--muted); padding: .75rem .15rem; }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: var(--green);
        border-bottom-color: var(--green);
    }
    [data-testid="stTabs"] button[role="tab"] p { font-weight: 600; }
    h2 { font-size: 1.25rem !important; margin-top: 1.4rem !important; }
    h3 { font-size: 1rem !important; }
    [data-testid="stMetric"] {
        background: var(--white);
        border: 1px solid var(--line);
        padding: 1rem 1.1rem;
    }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { color: var(--ink); font-family: 'IBM Plex Mono', monospace; }
    [data-testid="stForm"] {
        background: var(--white);
        border: 1px solid var(--line);
        padding: 1.2rem 1.35rem;
    }
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stFileUploaderDropzone"] {
        background: var(--input) !important;
        color: var(--ink) !important;
        border-color: var(--line) !important;
    }
    [data-testid="stFileUploaderDropzone"] > div,
    [data-testid="stFileUploaderDropzone"] > div * {
        color: var(--muted) !important;
    }
    [data-testid="stNumberInput"] button,
    [data-testid="stFileUploaderDropzone"] button {
        background: var(--green-soft) !important;
        color: var(--ink) !important;
        border-color: var(--line) !important;
    }
    [data-testid="stFileUploaderDropzone"] button * { color: var(--ink) !important; }
    [data-testid="stSlider"] { accent-color: var(--green); }
    [data-testid="stSlider"] [role="slider"] {
        background-color: var(--green) !important;
        border-color: var(--green) !important;
    }
    [data-testid="stSlider"] [data-baseweb="slider"] > div > div {
        background-color: var(--green) !important;
    }
    [data-testid="stRadio"] input[type="radio"] { accent-color: var(--green); }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); }
    .report-table {
        width: 100%;
        border-collapse: collapse;
        background: var(--white);
        color: var(--ink);
        font-size: .88rem;
    }
    .report-table th, .report-table td {
        padding: .55rem .65rem;
        border-bottom: 1px solid var(--line);
        color: var(--ink);
        text-align: left;
    }
    .report-table thead th { background: var(--green-soft); }
    .report-table tbody th { background: var(--white); }
    .stButton button, [data-testid="stFormSubmitButton"] button {
        border-radius: 3px;
        font-weight: 600;
        min-height: 2.7rem;
    }
    .stButton button[kind="primary"], [data-testid="stFormSubmitButton"] button {
        background: var(--green) !important;
        border-color: var(--green) !important;
        color: var(--white) !important;
    }
    [data-testid="stAlert"] { border-radius: 3px; }
    [data-testid="stRadio"] label { color: var(--ink); }
    @media (max-width: 700px) {
        .block-container { padding: 3.5rem 1rem 2rem; }
        .masthead { align-items: flex-start; flex-direction: column; }
        .masthead h1 { font-size: 1.55rem; }
    }
    </style>
    <header class="masthead">
        <div>
            <div class="eyebrow">Risk intelligence / transaction monitoring</div>
            <h1>Card anomaly review</h1>
            <p>Train the detector, inspect model performance, and assess account activity.</p>
        </div>
        <div class="service-tag">MODEL · ISOLATION FOREST</div>
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


tab1, tab2 = st.tabs(["Model evaluation", "Account assessment"])

with tab1:
    st.header("Training data")
    uploaded_file = st.file_uploader(
        "Upload a CSV dataset",
        type=["csv"],
        help="Maximum 25 MB. Use the eight account fields listed below; CUST_ID and other columns are ignored.",
    )
    upload_is_valid = False
    upload_bytes = None
    if uploaded_file is not None:
        upload_bytes = uploaded_file.getvalue()
        if len(upload_bytes) > 25 * 1024 * 1024:
            st.error("This file exceeds the 25 MB upload limit.")
        else:
            try:
                uploaded_df = pd.read_csv(BytesIO(upload_bytes))
                missing_columns = [column for column in FEATURE_COLUMNS if column not in uploaded_df.columns]
                if missing_columns:
                    st.error(f"Missing required columns: {', '.join(missing_columns)}")
                elif len(uploaded_df) < 10:
                    st.error("Add at least 10 data rows before training.")
                else:
                    upload_is_valid = True
                    st.caption(f"{uploaded_file.name} · {len(uploaded_df):,} rows · {len(uploaded_df.columns)} columns")
                    st.dataframe(uploaded_df.head(5), use_container_width=True, hide_index=True)
            except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as e:
                st.error(f"Could not read this CSV: {e}")

    st.caption("Required numeric columns: " + " · ".join(FEATURE_COLUMNS))
    st.header("Training controls")
    
    col_param1, col_param2, col_btn = st.columns([2, 2, 1.5])
    with col_param1:
        contamination = st.slider("Contamination Rate (Expected Anomaly %)", 0.01, 0.10, 0.03, step=0.01)
    with col_param2:
        test_size = st.slider("Test Set Split Ratio", 0.10, 0.40, 0.20, step=0.05)
    with col_btn:
        st.write("")
        st.write("")
        train_clicked = st.button(
            "Train uploaded data" if uploaded_file is not None else "Train sample data",
            type="primary",
            disabled=uploaded_file is not None and not upload_is_valid,
            use_container_width=True,
        )

    if train_clicked:
        with st.spinner("Training Isolation Forest and processing metrics..."):
            try:
                if upload_is_valid and upload_bytes is not None and uploaded_file is not None:
                    res = requests.post(
                        f"{API_URL}/train/upload",
                        params={
                            "contamination": contamination,
                            "test_size": test_size,
                            "filename": uploaded_file.name,
                        },
                        data=upload_bytes,
                        headers={"Content-Type": "text/csv"},
                        timeout=120,
                    )
                else:
                    res = requests.post(
                        f"{API_URL}/train",
                        json={"contamination": contamination, "test_size": test_size},
                        timeout=120,
                    )
                if res.status_code == 200:
                    st.session_state["eval_data"] = res.json()
                    st.success(f"Model trained on {res.json().get('dataset_name', 'the selected data')}.")
                else:
                    detail = res.json().get("detail", res.text)
                    st.error(f"Training failed: {detail}")
            except Exception as e:
                st.error(f"Cannot connect to FastAPI backend: {e}")

    if "eval_data" in st.session_state:
        data = st.session_state["eval_data"]
        st.caption(f"Current model source: {data.get('dataset_name', 'Bundled credit-card sample')}")
        
        st.subheader("Run summary")
        m1, m2, m3 = st.columns(3)
        m1.metric("Training Samples", data["train_samples"])
        m2.metric("Testing Samples", data["test_samples"])
        m3.metric("ROC-AUC Score", data["roc_curve"]["auc"])
        
        st.subheader("Evaluation")
        
        g_col1, g_col2 = st.columns(2)
        
        with g_col1:
            cm = data["confusion_matrix"]
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                labels=dict(x="Predicted Label", y="Actual Label (Heuristic)", color="Count"),
                x=['Normal (0)', 'Anomaly (1)'],
                y=['Normal (0)', 'Anomaly (1)'],
                title="Confusion matrix",
                color_continuous_scale=[[0, theme["green_soft"]], [1, theme["green"]]],
            )
            style_chart(fig_cm)
            st.plotly_chart(fig_cm, use_container_width=True, theme=None)
            
        with g_col2:
            roc = data["roc_curve"]
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode='lines', name=f'Isolation Forest (AUC={roc["auc"]})', line=dict(color=theme["green"], width=3)))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Baseline', line=dict(color=theme["muted"], dash='dash')))
            fig_roc.update_layout(
                title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate",
            )
            style_chart(fig_roc)
            fig_roc.update_xaxes(gridcolor=theme["grid"], zeroline=False)
            fig_roc.update_yaxes(gridcolor=theme["grid"], zeroline=False)
            st.plotly_chart(fig_roc, use_container_width=True, theme=None)
            

        st.subheader("Classification report")
        if "classification_report" in data:
            st.caption("Reference labels use high balance or cash advance thresholds; they are evaluation heuristics, not verified fraud outcomes.")
            report_values = data["classification_report"]
            report_df = pd.DataFrame(
                [report_values[class_name] for class_name in ("Normal", "Anomaly")],
                index=["Normal", "Anomaly"],
            )
            report_df.index.name = "Class"
            class_metrics = report_df.loc[["Normal", "Anomaly"], ["precision", "recall", "f1-score"]].reset_index()
            class_metrics = class_metrics.melt(id_vars="Class", var_name="Metric", value_name="Score")
            report_rows = [
                {"Class": class_name, **report_values[class_name]}
                for class_name in ("Normal", "Anomaly")
            ]
            report_rows.append({
                "Class": "accuracy",
                "precision": None,
                "recall": None,
                "f1-score": report_values["accuracy"],
                "support": data["test_samples"],
            })
            report_rows.extend(
                {"Class": average, **report_values[average]}
                for average in ("macro avg", "weighted avg")
            )
            report_display = pd.DataFrame(report_rows)
            for metric in ("precision", "recall", "f1-score"):
                report_display[metric] = report_display[metric].map(
                    lambda value: f"{value:.3f}" if pd.notna(value) else ""
                )
            report_display["support"] = report_display["support"].map(lambda value: f"{int(value):,}")

            report_col1, report_col2 = st.columns([1.15, 1])
            with report_col1:
                fig_report = px.bar(
                    class_metrics,
                    x="Class",
                    y="Score",
                    color="Metric",
                    barmode="group",
                    range_y=[0, 1],
                    title="Precision, recall and F1 by class",
                    color_discrete_sequence=[theme["green"], theme["amber"], theme["normal"]],
                )
                style_chart(fig_report)
                fig_report.update_yaxes(gridcolor=theme["grid"], zeroline=False)
                st.plotly_chart(fig_report, use_container_width=True, theme=None)
            with report_col2:
                st.markdown(report_display.to_html(index=False, classes="report-table", border=0), unsafe_allow_html=True)
        else:
            st.info("Train the model again to generate its classification report.")

        if "score_distribution" in data:
            st.subheader("Decision score distribution")
            score_df = pd.DataFrame(data["score_distribution"])
            score_df["classification"] = score_df["is_anomaly"].map({0: "Normal", 1: "Anomaly"})
            fig_scores = px.histogram(
                score_df,
                x="decision_score",
                color="classification",
                barmode="overlay",
                opacity=0.72,
                nbins=40,
                color_discrete_map={"Normal": theme["normal"], "Anomaly": theme["amber"]},
                labels={"decision_score": "Decision score", "classification": "Prediction"},
                title="More negative scores indicate higher anomaly risk",
            )
            fig_scores.add_vline(x=0, line_dash="dash", line_color=theme["muted"])
            style_chart(fig_scores)
            fig_scores.update_xaxes(gridcolor=theme["grid"], zeroline=False)
            fig_scores.update_yaxes(gridcolor=theme["grid"], zeroline=False)
            st.plotly_chart(fig_scores, use_container_width=True, theme=None)


        st.subheader("Anomaly distribution")
        pca_df = pd.DataFrame(data["pca_scatter"])
        pca_df["classification"] = pca_df["is_anomaly"].map({0: "Normal", 1: "Anomaly"})
        record_filter = st.selectbox(
            "Show records",
            ["All records", "Anomalies only", "Normal only"],
            label_visibility="collapsed",
        )
        if record_filter == "Anomalies only":
            pca_df = pca_df[pca_df["is_anomaly"] == 1]
        elif record_filter == "Normal only":
            pca_df = pca_df[pca_df["is_anomaly"] == 0]
        st.caption(f"Showing {len(pca_df):,} test records")
        fig_pca = px.scatter(
            pca_df,
            x="pca1",
            y="pca2",
            color="classification",
            color_discrete_map={"Normal": theme["normal"], "Anomaly": theme["amber"]},
            title="PCA projection of normal accounts and detected anomalies",
            hover_data={"classification": True, "pca1": ":.3f", "pca2": ":.3f"},
        )
        style_chart(fig_pca)
        fig_pca.update_xaxes(gridcolor=theme["grid"], zeroline=False)
        fig_pca.update_yaxes(gridcolor=theme["grid"], zeroline=False)
        st.plotly_chart(fig_pca, use_container_width=True, theme=None)

with tab2:
    st.header("Assess an account")
    st.markdown("Enter account activity to calculate its anomaly risk.")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            balance = st.number_input("ACCOUNT BALANCE ($)", value=40.90, step=50.0)
            purchases = st.number_input("TOTAL PURCHASES ($)", value=95.40, step=50.0)
            installments = st.number_input("INSTALLMENT PURCHASES ($)", value=95.40, step=50.0)
            cash_advance = st.number_input("CASH ADVANCE ($)", value=0.00, step=100.0)
        with col2:
            credit_limit = st.number_input("CREDIT LIMIT ($)", value=1000.00, step=500.0)
            payments = st.number_input("TOTAL PAYMENTS ($)", value=201.80, step=50.0)
            min_payments = st.number_input("MINIMUM PAYMENTS ($)", value=139.51, step=20.0)
            tenure = st.slider("TENURE (Months)", 6, 12, 12)
            
        submit_pred = st.form_submit_button("Evaluate Account Risk", type="primary")

    if submit_pred:
        payload = {
            "BALANCE": balance,
            "PURCHASES": purchases,
            "INSTALLMENTS_PURCHASES": installments,
            "CASH_ADVANCE": cash_advance,
            "CREDIT_LIMIT": credit_limit,
            "PAYMENTS": payments,
            "MINIMUM_PAYMENTS": min_payments,
            "TENURE": tenure
        }
        
        try:
            res = requests.post(f"{API_URL}/predict", json=payload)
            if res.status_code == 200:
                result = res.json()
                st.subheader("Prediction Result")
                
                if result["is_anomaly"]:
                    st.error(f"Status: {result['status']}")
                else:
                    st.success(f"Status: {result['status']}")
                    
                st.info(f"**Anomaly Decision Score:** `{result['decision_score']}` (Negative = Higher Risk)")
            else:
                st.error(f"Prediction Error: {res.text}")
        except Exception as e:
            st.error(f"Connection to backend failed: {e}")