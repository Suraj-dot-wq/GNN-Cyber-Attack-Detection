"""
GNN Cybersecurity Dashboard
Stage 1 + Stage 2 UI
--------------------------------
Stage 1: Traffic intake, dataset validation and security overview.
Stage 2: GNN inference, threat analysis, results and report export.

The trained GNN, scaler, feature preparation, k-NN graph construction and
prediction logic are intentionally preserved from the working dashboard.
"""

import os
import pickle
import hashlib
import io
import joblib
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import plotly.graph_objects as go

from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch_geometric.nn import GCNConv


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="GNN Cybersecurity SOC",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")

MODEL_PATH = os.path.join(DATA_DIR, "gnn_model.pt")
SCALER_PATH = os.path.join(DATA_DIR, "scaler.pkl")


# ============================================================
# MODEL CONFIGURATION
# ============================================================

EXPECTED_FEATURE_COUNT = 39
HIDDEN_FEATURES = 64
NUM_CLASSES = 2
K_NEIGHBORS = 5

TEST_ACCURACY = 97.68
TEST_F1 = 98.79


# ============================================================
# SESSION STATE
# ============================================================

if "results" not in st.session_state:
    st.session_state.results = None

if "detection_stats" not in st.session_state:
    st.session_state.detection_stats = None

if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = None

if "uploaded_signature" not in st.session_state:
    st.session_state.uploaded_signature = None

if "graph_edges" not in st.session_state:
    st.session_state.graph_edges = None

if "scaled_features" not in st.session_state:
    st.session_state.scaled_features = None

if "feature_columns" not in st.session_state:
    st.session_state.feature_columns = None

if "blocked_records" not in st.session_state:
    st.session_state.blocked_records = set()

if "isolated_records" not in st.session_state:
    st.session_state.isolated_records = set()


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 0%, rgba(37,99,235,.11), transparent 35%),
        radial-gradient(circle at 90% 10%, rgba(14,165,233,.06), transparent 35%),
        #050912;
    color: #e5e7eb;
}

.block-container {
    max-width: 1500px;
    padding-top: 1.2rem;
    padding-bottom: 3rem;
}

::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-track { background: #070c16; }
::-webkit-scrollbar-thumb { background: #243247; border-radius: 10px; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #05070d 0%, #060b15 100%);
    border-right: 1px solid #182337;
}

[data-testid="stSidebar"] .block-container {
    padding-top: 1.4rem;
}

.brand {
    display:flex;
    align-items:center;
    gap:10px;
    color:#f8fafc;
    font-size:24px;
    font-weight:850;
    letter-spacing:-.02em;
}

.brand-icon {
    width:34px;
    height:34px;
    border-radius:10px;
    display:flex;
    align-items:center;
    justify-content:center;
    background:linear-gradient(135deg,#2563eb,#38bdf8);
    box-shadow:0 8px 22px rgba(37,99,235,.25);
}

.brand-sub {
    color:#64748b;
    font-size:11px;
    margin:5px 0 18px 44px;
}

.sidebar-heading {
    color:#f1f5f9;
    font-size:14px;
    font-weight:750;
    margin:20px 0 8px;
}

.sidebar-status {
    border:1px solid #1e293b;
    background:#0b1321;
    border-radius:12px;
    padding:11px 12px;
    margin-top:10px;
}

.sidebar-status-row {
    display:flex;
    justify-content:space-between;
    align-items:center;
    color:#94a3b8;
    font-size:11px;
    padding:5px 0;
}

.sidebar-status-value {
    color:#e2e8f0;
    font-weight:700;
}

.status-pill {
    display:inline-flex;
    align-items:center;
    gap:7px;
    padding:6px 10px;
    border-radius:999px;
    font-size:10px;
    font-weight:800;
    letter-spacing:.04em;
}

.status-pill.online {
    color:#4ade80;
    background:rgba(34,197,94,.10);
    border:1px solid rgba(34,197,94,.25);
}

.status-dot {
    width:6px;
    height:6px;
    border-radius:50%;
    background:#4ade80;
    box-shadow:0 0 8px rgba(74,222,128,.9);
}

.sidebar-metric {
    border:1px solid #1d2a3e;
    background:linear-gradient(150deg,#0d1728,#09111f);
    border-radius:13px;
    padding:13px;
    margin-top:8px;
}

.sidebar-metric-label {
    color:#64748b;
    font-size:9px;
    font-weight:750;
    text-transform:uppercase;
    letter-spacing:.08em;
}

.sidebar-metric-value {
    color:#f8fafc;
    font-size:22px;
    font-weight:850;
    margin-top:3px;
}

.hero {
    position:relative;
    overflow:hidden;
    border:1px solid #20304a;
    border-radius:20px;
    padding:30px 32px;
    margin-bottom:22px;
    background:
        radial-gradient(circle at 90% 20%, rgba(37,99,235,.22), transparent 28%),
        linear-gradient(125deg,#0d1729,#101c31 58%,#0b1828);
    box-shadow:0 20px 45px rgba(0,0,0,.32);
}

.hero::after {
    content:"";
    position:absolute;
    right:-75px;
    bottom:-95px;
    width:250px;
    height:250px;
    border:1px solid rgba(96,165,250,.12);
    border-radius:50%;
    box-shadow:0 0 0 25px rgba(96,165,250,.025),
               0 0 0 55px rgba(96,165,250,.02);
}

.eyebrow {
    color:#60a5fa;
    font-size:11px;
    font-weight:800;
    letter-spacing:.14em;
    text-transform:uppercase;
    margin-bottom:8px;
}

.hero-title {
    color:#f8fafc;
    font-size:36px;
    line-height:1.15;
    font-weight:900;
    letter-spacing:-.035em;
    margin-bottom:9px;
}

.hero-subtitle {
    color:#94a3b8;
    font-size:14px;
    line-height:1.6;
    max-width:720px;
}

.hero-status {
    display:inline-flex;
    align-items:center;
    gap:8px;
    color:#4ade80;
    border:1px solid rgba(34,197,94,.25);
    background:rgba(34,197,94,.08);
    padding:7px 12px;
    border-radius:999px;
    margin-top:15px;
    font-size:10px;
    font-weight:800;
}

.section-header {
    color:#f1f5f9;
    font-size:20px;
    font-weight:820;
    margin-top:25px;
    margin-bottom:3px;
}

.section-description {
    color:#64748b;
    font-size:12px;
    margin-bottom:14px;
}

.stage-badge {
    display:inline-flex;
    align-items:center;
    gap:7px;
    color:#93c5fd;
    background:rgba(37,99,235,.10);
    border:1px solid rgba(96,165,250,.25);
    border-radius:999px;
    padding:5px 10px;
    font-size:10px;
    font-weight:800;
    letter-spacing:.06em;
    text-transform:uppercase;
}

.dashboard-card {
    background:linear-gradient(155deg,#0e192b,#09111f);
    border:1px solid #1d2a3e;
    border-radius:16px;
    padding:20px;
    box-shadow:0 10px 28px rgba(0,0,0,.22);
}

.card-title {
    color:#f8fafc;
    font-size:15px;
    font-weight:760;
    margin-bottom:13px;
}

.card-subtitle {
    color:#64748b;
    font-size:11px;
    line-height:1.55;
}

.model-row {
    display:flex;
    justify-content:space-between;
    gap:15px;
    padding:9px 0;
    border-bottom:1px solid #172234;
    font-size:12px;
}

.model-row:last-child { border-bottom:0; }
.model-label { color:#64748b; }
.model-value { color:#e2e8f0; font-weight:700; }

[data-testid="stMetric"] {
    background:linear-gradient(155deg,#0f1a2c,#09111f);
    border:1px solid #1d2a3e;
    border-radius:15px;
    padding:17px 18px;
    min-height:105px;
    box-shadow:0 10px 25px rgba(0,0,0,.20);
}

[data-testid="stMetricLabel"] {
    color:#94a3b8 !important;
    font-size:11px !important;
    font-weight:700 !important;
    text-transform:uppercase;
    letter-spacing:.045em;
}

[data-testid="stMetricValue"] {
    color:#f8fafc !important;
    font-size:27px !important;
    font-weight:850 !important;
}

[data-testid="stMetricDelta"] {
    font-size:10px !important;
}

[data-testid="stFileUploader"] {
    background:#0a1322;
    border:1.5px dashed #2c3e5f;
    border-radius:15px;
    padding:16px;
}

.stButton > button {
    width:100%;
    min-height:50px;
    border-radius:12px;
    border:1px solid #2563eb;
    background:linear-gradient(135deg,#2563eb,#1d4ed8);
    color:#fff;
    font-weight:800;
    font-size:14px;
    box-shadow:0 8px 20px rgba(37,99,235,.18);
}

.stButton > button:hover {
    border-color:#60a5fa;
    background:linear-gradient(135deg,#3b82f6,#2563eb);
}

.stDownloadButton > button {
    width:100%;
    min-height:48px;
    border-radius:12px;
    background:#0e192b;
    border:1px solid #334155;
    color:#e2e8f0;
    font-weight:700;
}

.workflow {
    display:grid;
    grid-template-columns:repeat(5,1fr);
    gap:8px;
}

.workflow-item {
    background:#0b1423;
    border:1px solid #1b2a40;
    border-radius:12px;
    padding:13px 10px;
    text-align:center;
}

.workflow-num {
    width:28px;
    height:28px;
    margin:0 auto 8px;
    border-radius:9px;
    display:flex;
    align-items:center;
    justify-content:center;
    background:rgba(37,99,235,.14);
    border:1px solid rgba(96,165,250,.25);
    color:#60a5fa;
    font-weight:800;
    font-size:11px;
}

.workflow-text {
    color:#cbd5e1;
    font-size:10px;
    font-weight:650;
    line-height:1.35;
}

.attack-alert {
    display:flex;
    justify-content:space-between;
    align-items:center;
    gap:10px;
    flex-wrap:wrap;
    padding:12px 14px;
    margin-bottom:8px;
    border-radius:11px;
    background:rgba(127,29,29,.14);
    border:1px solid rgba(239,68,68,.26);
    border-left:4px solid #ef4444;
}

.attack-title {
    color:#f87171;
    font-size:12px;
    font-weight:800;
}

.attack-details {
    color:#64748b;
    font-size:10px;
    margin-top:2px;
}

.confidence-badge {
    color:#93c5fd;
    background:rgba(37,99,235,.11);
    border:1px solid rgba(96,165,250,.24);
    padding:4px 9px;
    border-radius:999px;
    font-size:9px;
    font-weight:800;
}

.safe-alert {
    padding:14px;
    border-radius:11px;
    color:#4ade80;
    background:rgba(20,83,45,.12);
    border:1px solid rgba(34,197,94,.23);
    border-left:4px solid #22c55e;
    font-size:12px;
    font-weight:700;
}

.threat-box {
    text-align:center;
    border:1px solid #1d2a3e;
    background:#0a1321;
    border-radius:14px;
    padding:17px 10px;
}

.threat-number {
    color:#f8fafc;
    font-size:26px;
    font-weight:850;
}

.threat-label {
    color:#64748b;
    font-size:9px;
    text-transform:uppercase;
    letter-spacing:.06em;
    font-weight:750;
    margin-top:3px;
}

[data-testid="stExpander"] {
    background:#09111e;
    border:1px solid #1c2a3e;
    border-radius:12px;
}

[data-testid="stDataFrame"] {
    border:1px solid #1c2a3e;
    border-radius:12px;
    overflow:hidden;
}

.footer {
    text-align:center;
    color:#475569;
    font-size:10px;
    margin-top:42px;
    padding-top:20px;
    border-top:1px solid #1b2536;
}

@media (max-width: 900px) {
    .workflow { grid-template-columns:1fr; }
    .hero-title { font-size:29px; }
}

/* =========================================================
   STAGE 3 + STAGE 4 — INTELLIGENCE / RESPONSE UI
   ========================================================= */

.intel-card {
    background:linear-gradient(155deg,#0e192b,#09111f);
    border:1px solid #1d2a3e;
    border-radius:16px;
    padding:20px;
    box-shadow:0 10px 28px rgba(0,0,0,.22);
    height:100%;
}

.intel-label {
    color:#64748b;
    font-size:9px;
    font-weight:800;
    text-transform:uppercase;
    letter-spacing:.08em;
}

.intel-value {
    color:#f8fafc;
    font-size:28px;
    font-weight:900;
    margin-top:5px;
}

.risk-critical {
    color:#f87171;
}

.risk-high {
    color:#fb923c;
}

.risk-medium {
    color:#facc15;
}

.risk-low {
    color:#4ade80;
}

.risk-pill {
    display:inline-flex;
    align-items:center;
    padding:5px 10px;
    border-radius:999px;
    font-size:10px;
    font-weight:850;
    letter-spacing:.05em;
}

.risk-pill.critical {
    color:#fca5a5;
    background:rgba(239,68,68,.12);
    border:1px solid rgba(239,68,68,.28);
}

.risk-pill.high {
    color:#fdba74;
    background:rgba(249,115,22,.12);
    border:1px solid rgba(249,115,22,.28);
}

.risk-pill.medium {
    color:#fde68a;
    background:rgba(234,179,8,.12);
    border:1px solid rgba(234,179,8,.25);
}

.risk-pill.low {
    color:#86efac;
    background:rgba(34,197,94,.10);
    border:1px solid rgba(34,197,94,.24);
}

.response-card {
    background:linear-gradient(155deg,#111c2d,#0a1320);
    border:1px solid #25344c;
    border-radius:16px;
    padding:20px;
}

.response-item {
    color:#cbd5e1;
    font-size:12px;
    padding:9px 0;
    border-bottom:1px solid #172234;
}

.response-item:last-child {
    border-bottom:0;
}

.action-status {
    margin-top:12px;
    padding:10px 12px;
    border-radius:10px;
    background:rgba(34,197,94,.08);
    border:1px solid rgba(34,197,94,.20);
    color:#4ade80;
    font-size:11px;
    font-weight:700;
}

.incident-card {
    background:
        radial-gradient(circle at 100% 0%, rgba(239,68,68,.10), transparent 35%),
        linear-gradient(155deg,#111b2c,#09111f);
    border:1px solid #29384f;
    border-radius:16px;
    padding:22px;
}

.small-note {
    color:#64748b;
    font-size:10px;
    line-height:1.55;
}

.graph-legend {
    display:flex;
    gap:14px;
    flex-wrap:wrap;
    color:#94a3b8;
    font-size:10px;
    margin-top:4px;
}

.graph-dot {
    display:inline-block;
    width:8px;
    height:8px;
    border-radius:50%;
    margin-right:5px;
}

</style>
""",
    unsafe_allow_html=True,
)



# ============================================================
# STAGE 3 + STAGE 4 — HELPER FUNCTIONS
# ============================================================

def detect_attack_name_column(df):
    """Return a genuine attack-family column when the uploaded CSV has one.

    The trained GNN is binary (Benign/Attack), so the dashboard must never
    invent DDoS/Mirai/etc. from the binary prediction. Only an explicit
    attack-category/type/family/name column from the uploaded file is used.
    """
    preferred = [
        "attack_type",
        "attack_category",
        "attack_family",
        "attack_name",
        "Attack Type",
        "Attack Category",
        "Attack Family",
        "Attack Name",
    ]
    for column in preferred:
        if column in df.columns:
            return column

    for column in df.columns:
        normalized = str(column).strip().lower().replace("-", "_").replace(" ", "_")
        if "attack" in normalized and any(
            token in normalized for token in ("type", "category", "family", "name")
        ):
            return column

    return None


def calculate_risk_scores(predictions, confidence):
    """
    Convert GNN confidence into a practical SOC triage risk score.

    Attack:
        High confidence -> high risk

    Benign:
        Low risk by default, but unusual confidence can still be surfaced.

    This is a triage score, NOT a probability of compromise.
    """

    predictions = np.asarray(predictions)
    confidence = np.asarray(confidence, dtype=float)

    risk_scores = np.zeros(len(predictions), dtype=float)

    # -----------------------------
    # ATTACK RISK
    # -----------------------------
    attack_mask = predictions == 1

    # Confidence is already 0-100.
    # Slightly amplify very confident attacks.
    attack_confidence = confidence[attack_mask]

    risk_scores[attack_mask] = np.where(
        attack_confidence >= 95,
        90 + ((attack_confidence - 95) / 5) * 10,
        np.where(
            attack_confidence >= 80,
            70 + ((attack_confidence - 80) / 15) * 20,
            attack_confidence * 0.875,
        ),
    )

    # -----------------------------
    # BENIGN RISK
    # -----------------------------
    benign_mask = predictions == 0

    benign_confidence = confidence[benign_mask]

    # A benign prediction with low confidence gets a small
    # triage score so unusual traffic can still be noticed.
    risk_scores[benign_mask] = np.clip(
        (100 - benign_confidence) * 0.30,
        0,
        30,
    )

    return np.clip(risk_scores, 0, 100)


def get_risk_level(risk_score):
    if risk_score >= 80:
        return "CRITICAL"
    elif risk_score >= 60:
        return "HIGH"
    elif risk_score >= 30:
        return "MEDIUM"
    else:
        return "LOW"


def get_risk_css_class(level):
    return {
        "CRITICAL": "critical",
        "HIGH": "high",
        "MEDIUM": "medium",
        "LOW": "low",
    }.get(level, "low")


def get_recommended_response(score):
    if score >= 80:
        return [
            "🚫 Block suspicious traffic",
            "🔒 Isolate the affected node",
            "🔔 Generate a critical security alert",
            "📝 Create an incident for investigation",
        ]
    if score >= 60:
        return [
            "⚠️ Increase traffic monitoring",
            "🔍 Investigate the suspicious source",
            "📝 Create a security incident",
        ]
    if score >= 30:
        return [
            "👀 Continue monitoring the traffic",
            "📊 Review surrounding graph activity",
        ]
    return [
        "🟢 No immediate response required",
        "👀 Continue normal monitoring",
    ]


def get_top_anomalous_features(
    X_scaled,
    predictions,
    feature_columns,
    row_index,
    top_n=7,
):
    """
    Explain an attack by identifying features that are most unusual
    compared with the benign predictions in the current uploaded dataset.
    This is anomaly-based explanation, not causal feature attribution.
    """
    benign_mask = predictions == 0

    if np.sum(benign_mask) < 2:
        reference = X_scaled
    else:
        reference = X_scaled[benign_mask]

    reference_mean = np.mean(reference, axis=0)
    reference_std = np.std(reference, axis=0)
    reference_std = np.where(reference_std < 1e-8, 1.0, reference_std)

    z_scores = np.abs(
        (X_scaled[row_index] - reference_mean)
        / reference_std
    )

    top_indices = np.argsort(z_scores)[::-1][:top_n]

    return [
        (str(feature_columns[i]), float(z_scores[i]))
        for i in top_indices
    ]


def build_graph_visualization(edge_index, predictions, risk_scores, max_nodes=100):
    """
    Create a browser-friendly Plotly view of the GNN graph.
    For large datasets, visualize a focused subgraph instead of thousands
    of nodes/edges.
    """
    if edge_index is None or edge_index.numel() == 0:
        return None

    edges = edge_index.cpu().numpy().T
    total_nodes = len(predictions)

    attack_nodes = np.where(predictions == 1)[0]
    if len(attack_nodes) > 0:
        ranked_attacks = attack_nodes[
            np.argsort(risk_scores[attack_nodes])[::-1]
        ]
        seed_nodes = ranked_attacks[:min(25, len(ranked_attacks))]
    else:
        seed_nodes = np.arange(min(25, total_nodes))

    selected = set(int(x) for x in seed_nodes)

    # Add graph neighbors of the most suspicious nodes.
    for u, v in edges:
        if int(u) in selected or int(v) in selected:
            selected.add(int(u))
            selected.add(int(v))
            if len(selected) >= max_nodes:
                break

    selected = sorted(selected)[:max_nodes]
    selected_set = set(selected)

    if not selected:
        return None

    # Deterministic circular layout keeps the visualization lightweight.
    angles = np.linspace(
        0,
        2 * np.pi,
        len(selected),
        endpoint=False,
    )

    positions = {
        node: (
            float(np.cos(angle)),
            float(np.sin(angle)),
        )
        for node, angle in zip(selected, angles)
    }

    edge_x = []
    edge_y = []

    for u, v in edges:
        u = int(u)
        v = int(v)
        if u in selected_set and v in selected_set:
            edge_x.extend([
                positions[u][0],
                positions[v][0],
                None,
            ])
            edge_y.extend([
                positions[u][1],
                positions[v][1],
                None,
            ])

    fig = go.Figure()

    if edge_x:
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(
                    color="rgba(100,116,139,.28)",
                    width=1,
                ),
                hoverinfo="none",
                showlegend=False,
            )
        )

    benign_nodes = [
        node for node in selected
        if predictions[node] == 0
    ]
    attack_nodes_selected = [
        node for node in selected
        if predictions[node] == 1
    ]

    def node_trace(nodes, label, marker_color):
        if not nodes:
            return None

        return go.Scatter(
            x=[positions[n][0] for n in nodes],
            y=[positions[n][1] for n in nodes],
            mode="markers",
            name=label,
            customdata=[
                [
                    int(n + 1),
                    float(risk_scores[n]),
                    "Attack" if predictions[n] == 1 else "Benign",
                ]
                for n in nodes
            ],
            hovertemplate=(
                "Record #%{customdata[0]}<br>"
                "Prediction: %{customdata[2]}<br>"
                "Risk Score: %{customdata[1]:.2f}"
                "<extra></extra>"
            ),
            marker=dict(
                size=[
                    13 + min(12, risk_scores[n] / 10)
                    if predictions[n] == 1 else 10
                    for n in nodes
                ],
                color=marker_color,
                line=dict(
                    color="rgba(255,255,255,.45)",
                    width=1,
                ),
            ),
        )

    benign_trace = node_trace(
        benign_nodes,
        "Benign",
        "#22c55e",
    )
    attack_trace = node_trace(
        attack_nodes_selected,
        "Attack",
        "#ef4444",
    )

    if benign_trace:
        fig.add_trace(benign_trace)
    if attack_trace:
        fig.add_trace(attack_trace)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color="#94a3b8",
            family="Inter",
        ),
        height=520,
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
            font=dict(size=10),
        ),
        xaxis=dict(
            visible=False,
            fixedrange=True,
        ),
        yaxis=dict(
            visible=False,
            fixedrange=True,
        ),
        hovermode="closest",
    )

    return fig


def build_incident_report(
    record_number,
    prediction,
    confidence,
    risk_score,
    risk_level,
    attack_name,
    recommendations,
    anomalous_features,
    graph_edges,
):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    feature_lines = "\n".join(
        f"  - {name}: anomaly score {score:.2f}"
        for name, score in anomalous_features
    )

    response_lines = "\n".join(
        f"  - {item}"
        for item in recommendations
    )

    return f"""GNN CYBERSECURITY INCIDENT REPORT
=====================================

Incident ID:
INC-{datetime.now().strftime("%Y%m%d%H%M%S")}-{record_number}

Generated:
{timestamp}

DETECTION
---------
Detection Engine: GNN Cybersecurity Platform
Model: Graph Convolutional Network (GCN)
Input Features: {EXPECTED_FEATURE_COUNT}
k-NN Neighbors: {K_NEIGHBORS}

THREAT
------
Record Number: #{record_number}
Prediction: {prediction}
Threat Category: {attack_name}
Confidence: {confidence:.2f}%
Risk Score: {risk_score:.2f}/100
Risk Level: {risk_level}

GRAPH CONTEXT
-------------
Current Graph Edges: {graph_edges:,}

TOP ANOMALOUS FEATURES
----------------------
{feature_lines if feature_lines else "  - Not available"}

RECOMMENDED RESPONSE
--------------------
{response_lines}

ACTION STATUS
-------------
Blocking: SIMULATION ONLY
Isolation: SIMULATION ONLY
Incident Status: OPEN

NOTE
----
The risk score is a dashboard triage score derived from model confidence.
The anomalous-feature section compares the selected traffic record with
benign traffic in the uploaded dataset. It is not a causal explanation.
"""


# ============================================================
# GCN MODEL — MATCHES THE TRAINED CHECKPOINT
# ============================================================

class GCN(nn.Module):
    """GCN architecture used during training.

    The saved checkpoint contains conv1/conv2 GCNConv weights, so the
    dashboard must instantiate the same layer types and names.
    """

    def __init__(self, input_features, hidden_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(input_features, hidden_features)
        self.conv2 = GCNConv(hidden_features, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x


# LOAD MODEL — PRESERVED
# ============================================================

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, "Model file not found."

    try:
        model = GCN(
            input_features=EXPECTED_FEATURE_COUNT,
            hidden_features=HIDDEN_FEATURES,
            num_classes=NUM_CLASSES,
        )

        checkpoint = torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=False,
        )

        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=True)
        model.eval()

        return model, None

    except Exception as e:
        return None, str(e)


# ============================================================
# LOAD SCALER — PRESERVED
# ============================================================

@st.cache_resource
def load_scaler():
    if not os.path.exists(SCALER_PATH):
        return None

    try:
        scaler = joblib.load(SCALER_PATH)
        if not hasattr(scaler, "transform"):
            raise ValueError("The saved scaler does not provide transform().")
        if getattr(scaler, "n_features_in_", EXPECTED_FEATURE_COUNT) != EXPECTED_FEATURE_COUNT:
            raise ValueError(
                f"The trained scaler expects {getattr(scaler, 'n_features_in_', 'unknown')} features; "
                f"the GNN requires {EXPECTED_FEATURE_COUNT}."
            )
        return scaler
    except Exception as e:
        st.error(f"Failed to load the trained scaler: {e}")
        return None


model, model_error = load_model()
trained_scaler = load_scaler()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">🛡️</div>
            <span>GNN SOC</span>
        </div>
        <div class="brand-sub">Security Operations Center</div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown('<div class="sidebar-heading">🧠 Model Information</div>',
                unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="sidebar-status">
            <div class="sidebar-status-row">
                <span>Architecture</span><span class="sidebar-status-value">GCN</span>
            </div>
            <div class="sidebar-status-row">
                <span>Input Features</span><span class="sidebar-status-value">{EXPECTED_FEATURE_COUNT}</span>
            </div>
            <div class="sidebar-status-row">
                <span>Hidden Features</span><span class="sidebar-status-value">{HIDDEN_FEATURES}</span>
            </div>
            <div class="sidebar-status-row">
                <span>Classes</span><span class="sidebar-status-value">{NUM_CLASSES}</span>
            </div>
            <div class="sidebar-status-row">
                <span>k-NN Neighbors</span><span class="sidebar-status-value">{K_NEIGHBORS}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-heading">📊 Test Performance</div>',
                unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="sidebar-metric">
            <div class="sidebar-metric-label">Accuracy</div>
            <div class="sidebar-metric-value">{TEST_ACCURACY:.2f}%</div>
        </div>
        <div class="sidebar-metric">
            <div class="sidebar-metric-label">F1 Score</div>
            <div class="sidebar-metric-value">{TEST_F1:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-heading">⚙️ Engine Status</div>',
                unsafe_allow_html=True)

    if model is not None:
        st.markdown(
            """
            <div class="status-pill online">
                <span class="status-dot"></span> GNN MODEL READY
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.error("GNN Model Unavailable")
        if model_error:
            st.caption(model_error)

    st.markdown(
        """
        <div style="color:#475569;font-size:10px;margin-top:18px;line-height:1.5;">
            Detection uses the trained GCN model, trained scaler and
            k-NN graph construction. No model training occurs in the dashboard.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">⚡ REAL-TIME THREAT ENGINE</div>
        <div class="hero-title">🛡️ GNN Cybersecurity</div>
        <div class="hero-subtitle">
            AI-powered network traffic analysis using a Graph Convolutional
            Network. Upload traffic data, construct a k-NN graph and identify
            malicious activity with confidence scoring.
        </div>
        <div class="hero-status">
            <span class="status-dot"></span>
            SYSTEM ONLINE
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# STAGE 1 — TRAFFIC INTAKE
# ============================================================

st.markdown(
    '<span class="stage-badge">STAGE 1 • TRAFFIC INTAKE</span>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-header">📊 Security Overview</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    'Upload and validate network traffic before sending it through the GNN threat engine.'
    '</div>',
    unsafe_allow_html=True,
)

current_results = st.session_state.results
stats = st.session_state.detection_stats

if stats is None:
    overview_total = overview_benign = overview_attack = 0
    overview_rate = 0.0
else:
    overview_total = stats["total"]
    overview_benign = stats["benign"]
    overview_attack = stats["attack"]
    overview_rate = stats["attack_rate"]

m1, m2, m3, m4 = st.columns(4)
m1.metric("📊 Total Traffic", f"{overview_total:,}")
m2.metric("🟢 Benign Traffic", f"{overview_benign:,}")
m3.metric("🔴 Attack Traffic", f"{overview_attack:,}")
m4.metric("📈 Attack Rate", f"{overview_rate:.2f}%")


st.markdown(
    '<div class="section-header">📁 Upload Network Traffic</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    'CSV files are processed locally by the dashboard. The trained model requires 39 numeric features.'
    '</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload Network Traffic CSV",
    type=["csv"],
    help="Upload a CSV containing the network traffic features.",
)

if uploaded_file is not None:

    file_signature = (
        uploaded_file.name,
        uploaded_file.size
    )

    if st.session_state.get("rt_file_signature") != file_signature:

        st.session_state.rt_file_signature = file_signature
        st.session_state.rt_running = False
        st.session_state.rt_index = 0
        st.session_state.rt_events = []


if uploaded_file is None:
    st.markdown(
        """
        <div class="dashboard-card">
            <div class="card-title">🔍 Detection Workflow</div>
            <div class="workflow">
                <div class="workflow-item">
                    <div class="workflow-num">1</div>
                    <div class="workflow-text">Upload<br>traffic CSV</div>
                </div>
                <div class="workflow-item">
                    <div class="workflow-num">2</div>
                    <div class="workflow-text">Validate<br>39 features</div>
                </div>
                <div class="workflow-item">
                    <div class="workflow-num">3</div>
                    <div class="workflow-text">Build<br>k-NN graph</div>
                </div>
                <div class="workflow-item">
                    <div class="workflow-num">4</div>
                    <div class="workflow-text">Run<br>GNN inference</div>
                </div>
                <div class="workflow-item">
                    <div class="workflow-num">5</div>
                    <div class="workflow-text">Analyze<br>threats</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info("📄 Upload a CSV file to start attack detection.")

else:
    # Reset old detection when the uploaded file changes.
    # Use content signature as well as filename so replacing test.csv with
    # another test.csv cannot display stale predictions.
    uploaded_bytes = uploaded_file.getvalue()
    uploaded_signature = hashlib.sha256(uploaded_bytes).hexdigest()

    if st.session_state.uploaded_signature != uploaded_signature:
        st.session_state.results = None
        st.session_state.detection_stats = None
        st.session_state.graph_edges = None
        st.session_state.scaled_features = None
        st.session_state.feature_columns = None
        st.session_state.blocked_records = set()
        st.session_state.isolated_records = set()
        st.session_state.uploaded_name = uploaded_file.name
        st.session_state.uploaded_signature = uploaded_signature

    try:
        df = pd.read_csv(io.BytesIO(uploaded_bytes))
    except Exception as e:
        st.error(f"Unable to read CSV: {e}")
        st.stop()

    if df.empty:
        st.error("The uploaded CSV contains no rows.")
        st.stop()

    attack_name_column = detect_attack_name_column(df)

    st.success(
        f"✓ Dataset loaded — {len(df):,} rows × {len(df.columns):,} columns"
    )

    info1, info2, info3, info4 = st.columns(4)

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    info1.metric("Rows", f"{len(df):,}")
    info2.metric("Columns", f"{len(df.columns):,}")
    info3.metric("Numeric Columns", f"{len(numeric_columns):,}")
    info4.metric(
        "Model Features",
        f"{EXPECTED_FEATURE_COUNT}",
        "Required",
    )

    with st.expander("👀 Preview Dataset", expanded=False):
        st.dataframe(
            df.head(10),
            use_container_width=True,
            height=280,
        )

    # ========================================================
    # FEATURE VALIDATION — PRESERVED LOGIC
    # ========================================================

    st.markdown(
        '<div class="section-header">🔧 Feature Validation</div>',
        unsafe_allow_html=True,
    )

    REQUIRED_FEATURE_COLUMNS = [
        "Header_Length", "Protocol Type", "Time_To_Live", "Rate",
        "fin_flag_number", "syn_flag_number", "rst_flag_number",
        "psh_flag_number", "ack_flag_number", "ece_flag_number",
        "cwr_flag_number", "ack_count", "syn_count", "fin_count",
        "rst_count", "HTTP", "HTTPS", "DNS", "Telnet", "SMTP",
        "SSH", "IRC", "TCP", "UDP", "DHCP", "ARP", "ICMP",
        "IGMP", "IPv", "LLC", "Tot sum", "Min", "Max", "AVG",
        "Std", "Tot size", "IAT", "Number", "Variance"
    ]

    missing_features = [
        column for column in REQUIRED_FEATURE_COLUMNS
        if column not in df.columns
    ]

    if missing_features:
        st.error(
            "The uploaded CSV is missing required GNN features: "
            + ", ".join(missing_features)
        )
        st.stop()

    feature_columns = REQUIRED_FEATURE_COLUMNS

    st.markdown(
        f"""
        <div class="dashboard-card">
            <div class="card-title">✓ Feature Set Ready</div>
            <div class="card-subtitle">
                The first {EXPECTED_FEATURE_COUNT} numeric columns are used,
                matching the current dashboard's trained-model input contract.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("View Features Used", expanded=False):
        feature_df = pd.DataFrame(
            {
                "Position": range(1, len(feature_columns) + 1),
                "Feature": feature_columns,
            }
        )
        st.dataframe(
            feature_df,
            use_container_width=True,
            hide_index=True,
            height=260,
        )

    # ========================================================
    # PREPARE FEATURES — PRESERVED
    # ========================================================

    X_df = df[feature_columns].copy()

    X_df = X_df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    for column in feature_columns:
        X_df[column] = pd.to_numeric(
            X_df[column],
            errors="coerce",
        )

    X_df = X_df.fillna(X_df.median())
    X_df = X_df.fillna(0)

    X = X_df.values.astype(np.float32)

    # ========================================================
    # NORMALIZATION — USE THE EXACT TRAINED SCALER
    # ========================================================

    if trained_scaler is None:
        st.error(
            "The trained scaler could not be loaded. "
            "Detection has been stopped to prevent incorrect predictions."
        )
        st.stop()

    try:
        X_scaled = trained_scaler.transform(X)
        scaler_status = "Trained StandardScaler"
    except Exception as e:
        st.error(f"Failed to normalize the uploaded features: {e}")
        st.stop()

    X_tensor = torch.tensor(
        X_scaled,
        dtype=torch.float32,
    )

    # ========================================================
    # STAGE 2 — THREAT DETECTION
    # ========================================================

    st.markdown(
        '<span class="stage-badge">STAGE 2 • THREAT ANALYSIS</span>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-header">🚨 Attack Detection</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-description">'
        'Run the trained GNN model against the uploaded network traffic.'
        '</div>',
        unsafe_allow_html=True,
    )

    engine_col1, engine_col2, engine_col3 = st.columns(3)

    with engine_col1:
        st.markdown(
            f"""
            <div class="threat-box">
                <div class="threat-number">{EXPECTED_FEATURE_COUNT}</div>
                <div class="threat-label">Input Features</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with engine_col2:
        st.markdown(
            f"""
            <div class="threat-box">
                <div class="threat-number">{K_NEIGHBORS}</div>
                <div class="threat-label">k-NN Neighbors</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with engine_col3:
        st.markdown(
            f"""
            <div class="threat-box">
                <div class="threat-number">{scaler_status}</div>
                <div class="threat-label">Normalization</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ========================================================
    # GRAPH CREATION — PRESERVED
    # ========================================================

    def build_graph(X_input):
        number_of_nodes = len(X_input)

        if number_of_nodes < 2:
            return torch.empty(
                (2, 0),
                dtype=torch.long,
            )

        n_neighbors = min(
            K_NEIGHBORS,
            number_of_nodes - 1,
        )

        adjacency = kneighbors_graph(
            X_input,
            n_neighbors=n_neighbors,
            mode="connectivity",
            include_self=False,
        )

        rows, cols = adjacency.nonzero()

        edge_index = torch.tensor(
            np.vstack([rows, cols]),
            dtype=torch.long,
        )

        return edge_index

    detect_button = st.button(
        "🔍  DETECT ATTACKS",
        type="primary",
        use_container_width=True,
    )

    if detect_button:
        if model is None:
            st.error("GNN model could not be loaded.")
            st.stop()

        if len(X_tensor) == 0:
            st.error("The uploaded dataset contains no rows.")
            st.stop()

        try:
            with st.spinner(
                "Building k-NN graph and running GNN inference..."
            ):
                edge_index = build_graph(X_scaled)

                with torch.no_grad():
                    logits = model(
                        X_tensor,
                        edge_index,
                    )

                    probabilities = torch.softmax(
                        logits,
                        dim=1,
                    )

                    predictions = torch.argmax(
                        probabilities,
                        dim=1,
                    )

                    confidence = torch.max(
                        probabilities,
                        dim=1,
                    ).values

                predictions = predictions.numpy()

                confidence = confidence.numpy() * 100

                labels = np.where(
                    predictions == 0,
                    "Benign",
                    "Attack",
                )

                # ====================================================
                # STAGE 3 — RISK INTELLIGENCE
                # ====================================================

                risk_scores = calculate_risk_scores(
                    predictions,
                    confidence,
                )

                risk_levels = np.array(
                    [
                        get_risk_level(score)
                        for score in risk_scores
                    ]
                )

                results = df.copy()

                results["Prediction"] = labels

                results["Confidence (%)"] = np.round(
                    confidence,
                    2,
                )

                results["Risk Score"] = np.round(
                    risk_scores,
                    2,
                )

                results["Risk Level"] = risk_levels

                # Ground truth is displayed/evaluated only when the uploaded
                # CSV contains a label column. It is NEVER used for inference.
                ground_truth_available = "label" in df.columns
                if ground_truth_available:
                    truth_numeric = pd.to_numeric(df["label"], errors="coerce")
                    valid_truth = truth_numeric.isin([0, 1])
                    results["Ground Truth"] = np.where(
                        valid_truth,
                        np.where(truth_numeric == 1, "Attack", "Benign"),
                        "Unknown",
                    )

                total_records = len(results)

                attack_count = int(
                    np.sum(predictions == 1)
                )

                benign_count = int(
                    np.sum(predictions == 0)
                )

                attack_rate = (
                    attack_count / total_records * 100
                )

                benign_rate = (
                    benign_count / total_records * 100
                )

                st.session_state.results = results
                st.session_state.graph_edges = edge_index
                st.session_state.scaled_features = X_scaled
                st.session_state.feature_columns = feature_columns

                st.session_state.detection_stats = {
                    "total": total_records,
                    "benign": benign_count,
                    "attack": attack_count,
                    "attack_rate": attack_rate,
                    "benign_rate": benign_rate,
                    "edges": int(edge_index.shape[1]),
                    "ground_truth_available": bool(ground_truth_available),
                }

            st.success(
                "✓ Attack detection completed successfully!"
            )

            # Rerun so the top Security Overview cards update.
            st.rerun()

        except Exception as e:
            st.error(f"An error occurred during detection: {e}")
            st.exception(e)

    # ========================================================
    # DISPLAY SAVED RESULTS
    # ========================================================

    results = st.session_state.results
    stats = st.session_state.detection_stats

    if results is not None and stats is not None:

        st.markdown(
            '<div class="section-header">📊 Detection Summary</div>',
            unsafe_allow_html=True,
        )

        s1, s2, s3, s4 = st.columns(4)

        s1.metric(
            "📊 Total Traffic",
            f"{stats['total']:,}",
        )

        s2.metric(
            "🟢 Benign",
            f"{stats['benign']:,}",
            f"{stats['benign_rate']:.2f}%",
        )

        s3.metric(
            "🔴 Attacks",
            f"{stats['attack']:,}",
            f"{stats['attack_rate']:.2f}%",
        )

        s4.metric(
            "📈 Attack Rate",
            f"{stats['attack_rate']:.2f}%",
        )

        # ----------------------------------------------------
        # OPTIONAL GROUND-TRUTH VALIDATION
        # ----------------------------------------------------
        if stats.get("ground_truth_available", False):
            truth = pd.to_numeric(df["label"], errors="coerce")
            valid_mask = truth.isin([0, 1]).to_numpy()
            if np.any(valid_mask):
                y_true = truth.to_numpy()[valid_mask].astype(int)
                y_pred = np.where(results["Prediction"].to_numpy()[valid_mask] == "Attack", 1, 0)
                gt_acc = accuracy_score(y_true, y_pred) * 100
                gt_precision = precision_score(y_true, y_pred, zero_division=0) * 100
                gt_recall = recall_score(y_true, y_pred, zero_division=0) * 100
                gt_f1 = f1_score(y_true, y_pred, zero_division=0) * 100
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

                st.markdown(
                    '<div class="section-header">🎯 Ground-Truth Validation</div>',
                    unsafe_allow_html=True,
                )
                st.caption(
                    "The uploaded label column is used only to evaluate the GNN after inference. "
                    "It does not influence the predictions."
                )
                v1, v2, v3, v4 = st.columns(4)
                v1.metric("Accuracy", f"{gt_acc:.2f}%")
                v2.metric("Precision", f"{gt_precision:.2f}%")
                v3.metric("Recall", f"{gt_recall:.2f}%")
                v4.metric("F1 Score", f"{gt_f1:.2f}%")
                with st.expander("View confusion matrix", expanded=False):
                    st.dataframe(
                        pd.DataFrame(
                            cm,
                            index=["Actual Benign", "Actual Attack"],
                            columns=["Predicted Benign", "Predicted Attack"],
                        ),
                        use_container_width=True,
                    )

        # ====================================================
        # CHART + MODEL CARD
        # ====================================================

        chart_col, model_col = st.columns([1.65, 1])

        with chart_col:
            st.markdown(
                '<div class="section-header">📈 Traffic Classification</div>',
                unsafe_allow_html=True,
            )

            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    x=["Benign", "Attack"],
                    y=[stats["benign"], stats["attack"]],
                    marker=dict(
                        color=["#22c55e", "#ef4444"],
                        line=dict(
                            color="rgba(255,255,255,0.08)",
                            width=1,
                        ),
                    ),
                    text=[
                        f"{stats['benign']:,}",
                        f"{stats['attack']:,}",
                    ],
                    textposition="outside",
                    textfont=dict(
                        color="#e5e7eb",
                        size=13,
                    ),
                    width=0.42,
                    hovertemplate="%{x}: %{y:,}<extra></extra>",
                )
            )

            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(
                    color="#94a3b8",
                    family="Inter",
                ),
                margin=dict(
                    l=10,
                    r=10,
                    t=25,
                    b=10,
                ),
                height=315,
                showlegend=False,
                xaxis=dict(
                    showgrid=False,
                    tickfont=dict(
                        color="#cbd5e1",
                        size=12,
                    ),
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor="rgba(148,163,184,0.10)",
                    tickfont=dict(
                        color="#94a3b8",
                        size=10,
                    ),
                    title=dict(
                        text="Records",
                        font=dict(
                            color="#64748b",
                            size=11,
                        ),
                    ),
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        with model_col:
            st.markdown(
                """
                <div class="section-header">🧠 Model Information</div>
                <div class="dashboard-card">
                    <div class="card-title">
                        <span class="status-pill online">
                            <span class="status-dot"></span> MODEL ONLINE
                        </span>
                    </div>
                    <div class="model-row">
                        <span class="model-label">Architecture</span>
                        <span class="model-value">GCN</span>
                    </div>
                    <div class="model-row">
                        <span class="model-label">Input Features</span>
                        <span class="model-value">39</span>
                    </div>
                    <div class="model-row">
                        <span class="model-label">Hidden Features</span>
                        <span class="model-value">64</span>
                    </div>
                    <div class="model-row">
                        <span class="model-label">k-NN</span>
                        <span class="model-value">5</span>
                    </div>
                    <div class="model-row">
                        <span class="model-label">Graph Edges</span>
                        <span class="model-value">
                """
                + f"{stats['edges']:,}"
                + """
                        </span>
                    </div>
                    <div class="model-row">
                        <span class="model-label">Accuracy</span>
                        <span class="model-value">97.68%</span>
                    </div>
                    <div class="model-row">
                        <span class="model-label">F1 Score</span>
                        <span class="model-value">98.79%</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ----------------------------------------------------
        # ATTACK INTELLIGENCE
        # ----------------------------------------------------
        st.markdown(
            '<div class="section-header">🎯 Attack Intelligence</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="dashboard-card">
                <div class="card-title">🔎 Network Attack Profile</div>
                <div class="card-subtitle">
                    Analyze the detected traffic using the GNN predictions and
                    network-flow characteristics.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ----------------------------------------------------
        # ATTACK / BENIGN DISTRIBUTION
        # ----------------------------------------------------

        # Calculate intelligence statistics directly from results
        intel_attack_count = int(
            (results["Prediction"] == "Attack").sum()
        )

        intel_benign_count = int(
            (results["Prediction"] == "Benign").sum()
        )

        intel_total_count = len(results)

        intel_attack_rate = (
            intel_attack_count / intel_total_count * 100
            if intel_total_count > 0
            else 0
        )

        intel_col1, intel_col2, intel_col3 = st.columns(3)

        with intel_col1:
            st.metric(
                "🔴 Detected Attacks",
                f"{intel_attack_count:,}",
            )

        with intel_col2:
            st.metric(
                "🟢 Benign Traffic",
                f"{intel_benign_count:,}",
            )

        with intel_col3:
            st.metric(
                "📈 Attack Rate",
                f"{intel_attack_rate:.2f}%",
            )

        # ----------------------------------------------------
        # ATTACK PATTERN ANALYSIS
        # ----------------------------------------------------

        st.markdown(
            "### 🧠 Attack Pattern Analysis"
        )

        attack_rows = results[results["Prediction"] == "Attack"]
        benign_rows = results[results["Prediction"] == "Benign"]

        numeric_features = [
            col for col in feature_columns
            if col in results.columns
        ]

        if len(attack_rows) > 0 and len(benign_rows) > 0:

            attack_means = attack_rows[numeric_features].mean()
            benign_means = benign_rows[numeric_features].mean()

            comparison = pd.DataFrame({
                "Feature": numeric_features,
                "Attack Average": [
                    attack_means[col] for col in numeric_features
                ],
                "Benign Average": [
                    benign_means[col] for col in numeric_features
                ],
            })

            comparison["Difference (%)"] = np.where(
                comparison["Benign Average"].abs() > 1e-8,
                (
                    (
                        comparison["Attack Average"]
                        - comparison["Benign Average"]
                    ).abs()
                    / comparison["Benign Average"].abs()
                ) * 100,
                0,
            )

            comparison = comparison.sort_values(
                "Difference (%)",
                ascending=False,
            ).head(10)

            st.dataframe(
                comparison.round(3),
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "The table highlights features whose average values differ most "
                "between GNN-detected attacks and benign traffic. "
                "This is statistical pattern analysis, not causal explanation."
            )

        else:

            st.info(
                "Attack-pattern comparison requires both attack and benign "
                "detections in the uploaded dataset."
            )

        # ----------------------------------------------------
        # TRAFFIC BEHAVIOR SUMMARY
        # ----------------------------------------------------

        st.markdown(
            "### 📊 Traffic Behavior Summary"
        )

        if len(attack_rows) > 0:

            behavior_features = [
                col for col in [
                    "Rate",
                    "Time_To_Live",
                    "Tot size",
                    "IAT",
                    "Number",
                    "Variance",
                    "Std",
                    "AVG",
                ]
                if col in attack_rows.columns
            ]

            if behavior_features:

                behavior_data = pd.DataFrame({
                    "Feature": behavior_features,
                    "Attack Mean": [
                        attack_rows[col].mean()
                        for col in behavior_features
                    ],
                    "Attack Max": [
                        attack_rows[col].max()
                        for col in behavior_features
                    ],
                    "Attack Min": [
                        attack_rows[col].min()
                        for col in behavior_features
                    ],
                })

                st.dataframe(
                    behavior_data.round(3),
                    use_container_width=True,
                    hide_index=True,
                )

        # ============================================================
        # REAL-TIME THREAT MONITORING
        # ============================================================

        st.markdown(
            '<div class="section-header">⚡ Real-Time Threat Monitoring</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="dashboard-card">
                <div class="card-title">⚡ LIVE NETWORK THREAT SIMULATION</div>
                <div class="card-subtitle">
                    Replay the analyzed network traffic sequentially to simulate
                    real-time threat monitoring using the GNN detection results.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ------------------------------------------------------------
        # Initialize simulation state
        # ------------------------------------------------------------

        if "rt_running" not in st.session_state:
            st.session_state.rt_running = False

        if "rt_index" not in st.session_state:
            st.session_state.rt_index = 0

        if "rt_events" not in st.session_state:
            st.session_state.rt_events = []

        if "rt_file_signature" not in st.session_state:
            st.session_state.rt_file_signature = None

        # ------------------------------------------------------------
        # Simulation controls
        # ------------------------------------------------------------

        col1, col2, col3 = st.columns(3)

        with col1:
            start_simulation = st.button(
                "▶ Start Simulation",
                use_container_width=True,
                type="primary",
            )

        with col2:
            pause_simulation = st.button(
                "⏸ Pause",
                use_container_width=True,
            )

        with col3:
            reset_simulation = st.button(
                "🔄 Reset",
                use_container_width=True,
            )

        # ------------------------------------------------------------
        # Button actions
        # ------------------------------------------------------------

        if start_simulation:
            st.session_state.rt_running = True

        if pause_simulation:
            st.session_state.rt_running = False

        if reset_simulation:
            st.session_state.rt_running = False
            st.session_state.rt_index = 0
            st.session_state.rt_events = []

        # ------------------------------------------------------------
        # Make sure detection results exist
        # ------------------------------------------------------------

        if "results" in locals() and results is not None and len(results) > 0:

            total_rows = len(results)

            # Find security columns safely
            prediction_column = None
            confidence_column = None
            risk_column = None

            for col in results.columns:

                col_lower = str(col).lower()

                if prediction_column is None and (
                    "prediction" in col_lower
                    or "classification" in col_lower
                ):
                    prediction_column = col

                if confidence_column is None and "confidence" in col_lower:
                    confidence_column = col

                if risk_column is None and "risk score" in col_lower:
                    risk_column = col

            # --------------------------------------------------------
            # Current simulation position
            # --------------------------------------------------------

            current_index = min(
                st.session_state.rt_index,
                total_rows - 1,
            )

            current_row = results.iloc[current_index]

            # --------------------------------------------------------
            # Determine prediction
            # --------------------------------------------------------

            prediction_value = None

            if prediction_column is not None:
                prediction_value = current_row[prediction_column]

            prediction_text = str(prediction_value).upper()

            if (
                prediction_value == 1
                or prediction_text == "1"
                or "ATTACK" in prediction_text
                or "MALICIOUS" in prediction_text
            ):
                is_attack = True
            else:
                is_attack = False

            # --------------------------------------------------------
            # Confidence
            # --------------------------------------------------------

            confidence_value = 0.0

            if confidence_column is not None:
                try:
                    confidence_value = float(current_row[confidence_column])
                except Exception:
                    confidence_value = 0.0

            # --------------------------------------------------------
            # Risk
            # --------------------------------------------------------

            risk_value = 0.0

            if risk_column is not None:
                try:
                    risk_value = float(current_row[risk_column])
                except Exception:
                    risk_value = confidence_value

            # --------------------------------------------------------
            # Live counters
            # --------------------------------------------------------

            processed_count = st.session_state.rt_index + 1

            attack_count_rt = 0
            benign_count_rt = 0

            for event in st.session_state.rt_events:

                if event["Status"] == "ATTACK":
                    attack_count_rt += 1
                else:
                    benign_count_rt += 1

            # Include current event
            if is_attack:
                attack_count_rt += 1
            else:
                benign_count_rt += 1

            threat_rate_rt = (
                attack_count_rt / processed_count * 100
                if processed_count > 0
                else 0
            )

            # --------------------------------------------------------
            # System status
            # --------------------------------------------------------

            if st.session_state.rt_running:

                st.success("🟢 SYSTEM ONLINE • LIVE MONITORING")

            else:

                st.info("⏸ SYSTEM PAUSED • READY FOR SIMULATION")

            # --------------------------------------------------------
            # Live metrics
            # --------------------------------------------------------

            m1, m2, m3, m4 = st.columns(4)

            with m1:
                st.metric(
                    "Packets Analyzed",
                    f"{processed_count:,}",
                )

            with m2:
                st.metric(
                    "Threats Detected",
                    f"{attack_count_rt:,}",
                )

            with m3:
                st.metric(
                    "Benign Traffic",
                    f"{benign_count_rt:,}",
                )

            with m4:
                st.metric(
                    "Threat Rate",
                    f"{threat_rate_rt:.2f}%",
                )

            # --------------------------------------------------------
            # Current event
            # --------------------------------------------------------

            st.markdown("### 🚨 Latest Network Event")

            if is_attack:

                st.error(
                    f"🔴 ATTACK DETECTED  •  "
                    f"Risk: {risk_value:.2f}/100  •  "
                    f"Confidence: {confidence_value:.2f}%"
                )

            else:

                st.success(
                    f"🟢 BENIGN TRAFFIC  •  "
                    f"Risk: {risk_value:.2f}/100  •  "
                    f"Confidence: {confidence_value:.2f}%"
                )

            # --------------------------------------------------------
            # Event history
            # --------------------------------------------------------

            st.markdown("### 📡 Latest Events")

            if st.session_state.rt_events:

                event_df = pd.DataFrame(
                    st.session_state.rt_events[-10:]
                )

                st.dataframe(
                    event_df,
                    use_container_width=True,
                    hide_index=True,
                )

            # --------------------------------------------------------
            # Automatically process next row
            # --------------------------------------------------------

            if st.session_state.rt_running:

                import time

                # Save current event
                event = {
                    "Event": f"#{processed_count}",
                    "Status": "ATTACK" if is_attack else "BENIGN",
                    "Risk": round(risk_value, 2),
                    "Confidence": round(confidence_value, 2),
                }

                st.session_state.rt_events.append(event)

                # Move to next traffic record
                st.session_state.rt_index += 1

                # Stop at end
                if st.session_state.rt_index >= total_rows:

                    st.session_state.rt_index = total_rows - 1
                    st.session_state.rt_running = False

                    st.success(
                        "✅ Real-time simulation completed."
                    )

                else:

                    time.sleep(0.7)
                    st.rerun()

        else:

            st.warning(
                "Upload a CSV and run GNN detection before starting "
                "Real-Time Threat Monitoring."
            )

# ====================================================
# STAGE 3 — THREAT INTELLIGENCE
# ====================================================
        # ====================================================
        # STAGE 3 — THREAT INTELLIGENCE
        # ====================================================

        st.markdown(
            '<span class="stage-badge">STAGE 3 • THREAT INTELLIGENCE</span>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-header">🧠 Threat Intelligence</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-description">'
            'Turn GNN predictions into severity scores, anomaly explanations and graph context.'
            '</div>',
            unsafe_allow_html=True,
        )

        risk_values = results["Risk Score"].astype(float).values
        critical_count = int(np.sum(risk_values >= 80))
        high_count = int(np.sum((risk_values >= 60) & (risk_values < 80)))
        medium_count = int(np.sum((risk_values >= 30) & (risk_values < 60)))
        max_risk = float(np.max(risk_values)) if len(risk_values) else 0.0

        intelligence_cols = st.columns(4)

        intelligence_cols[0].metric(
            "☢️ Critical",
            f"{critical_count:,}",
        )
        intelligence_cols[1].metric(
            "🔶 High",
            f"{high_count:,}",
        )
        intelligence_cols[2].metric(
            "🟡 Medium",
            f"{medium_count:,}",
        )
        intelligence_cols[3].metric(
            "🎯 Max Risk",
            f"{max_risk:.2f}",
        )

        # ----------------------------------------------------
        # RISK DISTRIBUTION
        # ----------------------------------------------------

        risk_col, explain_col = st.columns([1.1, 1])

        with risk_col:
            st.markdown(
                '<div class="section-header">📊 Risk Distribution</div>',
                unsafe_allow_html=True,
            )

            risk_fig = go.Figure()

            risk_fig.add_trace(
                go.Bar(
                    x=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                    y=[
                        int(np.sum(risk_values < 30)),
                        medium_count,
                        high_count,
                        critical_count,
                    ],
                    marker=dict(
                        color=[
                            "#22c55e",
                            "#eab308",
                            "#f97316",
                            "#ef4444",
                        ],
                    ),
                    text=[
                        int(np.sum(risk_values < 30)),
                        medium_count,
                        high_count,
                        critical_count,
                    ],
                    textposition="outside",
                    hovertemplate="%{x}: %{y:,}<extra></extra>",
                )
            )

            risk_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(
                    color="#94a3b8",
                    family="Inter",
                ),
                height=310,
                margin=dict(l=10, r=10, t=20, b=10),
                showlegend=False,
                xaxis=dict(
                    showgrid=False,
                    tickfont=dict(
                        color="#cbd5e1",
                        size=11,
                    ),
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor="rgba(148,163,184,.10)",
                    title=dict(
                        text="Records",
                        font=dict(
                            color="#64748b",
                            size=10,
                        ),
                    ),
                ),
            )

            st.plotly_chart(
                risk_fig,
                use_container_width=True,
            )

            st.markdown(
                '<div class="small-note">'
                'Risk Score is a triage score derived from model confidence. '
                'It is not a calibrated probability of compromise.'
                '</div>',
                unsafe_allow_html=True,
            )

        # ----------------------------------------------------
        # EXPLAINABILITY
        # ----------------------------------------------------

        with explain_col:
            st.markdown(
                '<div class="section-header">🧠 Why Was It Flagged?</div>',
                unsafe_allow_html=True,
            )

            attack_indices = np.where(
                results["Prediction"].values == "Attack"
            )[0]

            if len(attack_indices) == 0:
                st.markdown(
                    """
                    <div class="safe-alert">
                        🟢 No attack records are available for explainability.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                attack_options = [
                    int(index + 1)
                    for index in attack_indices
                ]

                selected_record = st.selectbox(
                    "Select an attack record",
                    attack_options,
                    key="explain_record",
                )

                selected_index = selected_record - 1

                selected_anomalies = get_top_anomalous_features(
                    st.session_state.scaled_features,
                    results["Prediction"].values == "Attack",
                    st.session_state.feature_columns,
                    selected_index,
                    top_n=7,
                )

                explain_df = pd.DataFrame(
                    selected_anomalies,
                    columns=[
                        "Feature",
                        "Anomaly Score",
                    ],
                ).sort_values(
                    "Anomaly Score",
                    ascending=True,
                )

                explain_fig = go.Figure(
                    go.Bar(
                        x=explain_df["Anomaly Score"],
                        y=explain_df["Feature"],
                        orientation="h",
                        marker=dict(
                            color="#ef4444",
                        ),
                        hovertemplate=(
                            "%{y}<br>"
                            "Anomaly score: %{x:.2f}"
                            "<extra></extra>"
                        ),
                    )
                )

                explain_fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(
                        color="#94a3b8",
                        family="Inter",
                    ),
                    height=310,
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis=dict(
                        showgrid=True,
                        gridcolor="rgba(148,163,184,.10)",
                        title="Anomaly Score",
                    ),
                    yaxis=dict(
                        showgrid=False,
                        tickfont=dict(size=9),
                    ),
                )

                st.plotly_chart(
                    explain_fig,
                    use_container_width=True,
                )

                st.markdown(
                    '<div class="small-note">'
                    'These are the most unusual features compared with benign '
                    'traffic in the uploaded dataset. This is an anomaly-based '
                    'explanation, not causal feature attribution.'
                    '</div>',
                    unsafe_allow_html=True,
                )

        # ====================================================
        # GNN GRAPH VISUALIZATION
        # ====================================================

        st.markdown(
            '<div class="section-header">🕸️ Interactive GNN Graph</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-description">'
            'Visualize a focused subgraph around the most suspicious traffic nodes.'
            '</div>',
            unsafe_allow_html=True,
        )

        graph_fig = build_graph_visualization(
            st.session_state.graph_edges,
            results["Prediction"].values == "Attack",
            results["Risk Score"].values.astype(float),
            max_nodes=100,
        )

        if graph_fig is not None:
            st.plotly_chart(
                graph_fig,
                use_container_width=True,
            )

            st.markdown(
                """
                <div class="graph-legend">
                    <span><span class="graph-dot" style="background:#22c55e;"></span>Benign node</span>
                    <span><span class="graph-dot" style="background:#ef4444;"></span>Attack node</span>
                    <span>Node size increases with attack risk</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="small-note">'
                f'Focused visualization: up to 100 nodes from the current '
                f'{len(results):,}-node traffic graph. The underlying GNN '
                f'inference still uses the complete graph.'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("Graph visualization is unavailable for this dataset.")

        # ====================================================
        # STAGE 4 — RESPONSE CENTER
        # ====================================================

        st.markdown(
            '<span class="stage-badge">STAGE 4 • RESPONSE CENTER</span>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-header">🛡️ Response Center</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-description">'
            'Translate high-risk detections into recommended and simulated SOC actions.'
            '</div>',
            unsafe_allow_html=True,
        )

        attack_indices = np.where(
            results["Prediction"].values == "Attack"
        )[0]

        if len(attack_indices) == 0:
            st.markdown(
                """
                <div class="safe-alert">
                    🟢 No active threats require a response.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            ranked_attack_indices = attack_indices[
                np.argsort(
                    results["Risk Score"].values[attack_indices]
                )[::-1]
            ]

            response_options = [
                int(index + 1)
                for index in ranked_attack_indices
            ]

            response_record = st.selectbox(
                "Select threat for response",
                response_options,
                key="response_record",
            )

            response_index = response_record - 1
            response_row = results.iloc[response_index]

            response_risk = float(
                response_row["Risk Score"]
            )
            response_confidence = float(
                response_row["Confidence (%)"]
            )
            response_level = str(
                response_row["Risk Level"]
            )

            response_attack_name = "ATTACK DETECTED"

            if attack_name_column is not None:
                raw_name = str(
                    response_row[attack_name_column]
                ).strip()

                if raw_name and raw_name.lower() not in (
                    "nan",
                    "benign",
                    "0",
                    "1",
                ):
                    response_attack_name = raw_name.upper()

            response_col1, response_col2 = st.columns([1, 1.25])

            with response_col1:
                risk_class = get_risk_css_class(
                    response_level
                )

                st.markdown(
                    f"""
                    <div class="intel-card">
                        <div class="intel-label">Selected Threat</div>
                        <div style="color:#f87171;font-size:19px;font-weight:850;margin-top:5px;">
                            🔴 {response_attack_name}
                        </div>
                        <div style="margin-top:16px;">
                            <span class="risk-pill {risk_class}">
                                {response_level}
                            </span>
                        </div>
                        <div style="margin-top:18px;">
                            <div class="model-row">
                                <span class="model-label">Record</span>
                                <span class="model-value">#{response_record}</span>
                            </div>
                            <div class="model-row">
                                <span class="model-label">Risk Score</span>
                                <span class="model-value">{response_risk:.2f}/100</span>
                            </div>
                            <div class="model-row">
                                <span class="model-label">Confidence</span>
                                <span class="model-value">{response_confidence:.2f}%</span>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with response_col2:
                recommendations = get_recommended_response(
                    response_risk
                )

                response_html = "".join(
                    f'<div class="response-item">{item}</div>'
                    for item in recommendations
                )

                st.markdown(
                    f"""
                    <div class="response-card">
                        <div class="card-title">🛡️ Recommended Response</div>
                        {response_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # ------------------------------------------------
            # SIMULATED SECURITY ACTIONS
            # ------------------------------------------------

            action1, action2 = st.columns(2)

            with action1:
                if st.button(
                    "🚫 SIMULATE BLOCK",
                    key="simulate_block",
                ):
                    st.session_state.blocked_records.add(
                        response_record
                    )

                if response_record in st.session_state.blocked_records:
                    st.markdown(
                        """
                        <div class="action-status">
                            ✓ BLOCK SIMULATION ACTIVE — no real network rule was changed.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with action2:
                if st.button(
                    "🔒 SIMULATE ISOLATION",
                    key="simulate_isolation",
                ):
                    st.session_state.isolated_records.add(
                        response_record
                    )

                if response_record in st.session_state.isolated_records:
                    st.markdown(
                        """
                        <div class="action-status">
                            ✓ ISOLATION SIMULATION ACTIVE — no real host/network was isolated.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # ------------------------------------------------
            # INCIDENT REPORT
            # ------------------------------------------------

            anomalous_features = get_top_anomalous_features(
                st.session_state.scaled_features,
                results["Prediction"].values == "Attack",
                st.session_state.feature_columns,
                response_index,
                top_n=7,
            )

            incident_report = build_incident_report(
                record_number=response_record,
                prediction=str(
                    response_row["Prediction"]
                ),
                confidence=response_confidence,
                risk_score=response_risk,
                risk_level=response_level,
                attack_name=response_attack_name,
                recommendations=recommendations,
                anomalous_features=anomalous_features,
                graph_edges=stats["edges"],
            )

            st.markdown(
                '<div class="section-header">📝 Incident Report</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="incident-card">
                    <div class="card-title">🚨 INC-{datetime.now().strftime("%Y%m%d")}-{response_record:04d}</div>
                    <div class="card-subtitle">
                        Security incident prepared from the selected GNN detection.
                        The response actions remain simulation-only.
                    </div>
                    <div style="margin-top:14px;">
                        <div class="model-row">
                            <span class="model-label">Threat</span>
                            <span class="model-value">{response_attack_name}</span>
                        </div>
                        <div class="model-row">
                            <span class="model-label">Risk</span>
                            <span class="model-value">{response_risk:.2f}/100 • {response_level}</span>
                        </div>
                        <div class="model-row">
                            <span class="model-label">Confidence</span>
                            <span class="model-value">{response_confidence:.2f}%</span>
                        </div>
                        <div class="model-row">
                            <span class="model-label">Status</span>
                            <span class="model-value">OPEN</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.download_button(
                label="📝 Download Incident Report",
                data=incident_report.encode("utf-8"),
                file_name=f"gnn_incident_report_record_{response_record}.txt",
                mime="text/plain",
                use_container_width=True,
                key="download_incident_report",
            )


        # ====================================================
        # RESULTS TABLE
        # ====================================================

        st.markdown(
            '<div class="section-header">📋 Detection Results</div>',
            unsafe_allow_html=True,
        )

        # Keep the primary security fields visible and avoid the
        # extremely wide table shown by the previous dashboard.
        result_priority = [
            "Prediction",
            "Confidence (%)",
            "Risk Score",
            "Risk Level",
        ]

        if "Ground Truth" in results.columns:
            result_priority.insert(1, "Ground Truth")

        if attack_name_column is not None and attack_name_column not in result_priority:
            result_priority.append(attack_name_column)

        other_columns = [
            column
            for column in results.columns
            if column not in result_priority
        ]

        compact_columns = result_priority + other_columns[:8]
        compact_results = results[compact_columns]

        st.dataframe(
            compact_results,
            use_container_width=True,
            height=390,
        )

        with st.expander(
            f"🔎 View all original traffic features ({len(results.columns)} columns)"
        ):
            st.dataframe(
                results,
                use_container_width=True,
                height=450,
            )

        # ====================================================
        # EXPORT
        # ====================================================

        st.markdown(
            '<div class="section-header">⬇️ Export Detection Report</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="dashboard-card">
                <div class="card-title">📄 Detection Report</div>
                <div class="card-subtitle">
                    Export the complete uploaded dataset with GNN predictions
                    confidence, risk score and risk level. The downloaded CSV
                    contains every original column plus the GNN security fields.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        csv_data = results.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="⬇️ Download Complete Detection Results CSV",
            data=csv_data,
            file_name="gnn_attack_detection_results.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🛡️ GNN Cybersecurity Detection
        &nbsp; • &nbsp;
        Graph Neural Network Security Platform
        &nbsp; • &nbsp;
        Stage 1 + Stage 2 + Stage 3 + Stage 4
    </div>
    """,
    unsafe_allow_html=True,
)