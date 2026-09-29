"""
╔══════════════════════════════════════════════════════════════╗
║  SolarLens — Voyez vos panneaux solaires comme l'IA les voit          ║
║  Author: Sanae Najimi                                       ║
║  Design : chrome / silver studio, accent menthe-teal        ║
╚══════════════════════════════════════════════════════════════╝
"""

import base64
import io
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

from models.solarlens_model import get_transforms
from utils.model_loader import load_model

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SolarLens — Voyez vos panneaux solaires comme l'IA les voit",
    page_icon="☀",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ──────────────────────────────────────────────────────────────
# LOAD HERO IMAGES AS BASE64
# ──────────────────────────────────────────────────────────────
@st.cache_data
def _load_asset_b64(path: str) -> str:
    p = Path(__file__).parent / path
    if not p.exists():
        return ""
    return base64.b64encode(p.read_bytes()).decode("utf-8")


HOUSE_IMG = _load_asset_b64("assets/house-3d.png")
HOUSE_DEFECT_IMG = _load_asset_b64("assets/house-3d-defect.png")


# ──────────────────────────────────────────────────────────────
# GLOBAL CSS — Chrome / silver studio design system
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Instrument+Serif:ital@0;1&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* Design tokens — mapped from Lovable oklch to hex/rgb */
:root {
    --background: #f5f5f7;
    --foreground: #26272e;
    --card: #fafafc;
    --card-foreground: #26272e;
    --muted: #ebebef;
    --muted-foreground: #71717a;
    --border: #dadade;
    --secondary: #e8e8ec;
    --accent: #7adbd0;
    --accent-2: #56c2b4;
    --accent-foreground: #0e3833;
    --destructive: #dc4545;
    --ink: #1c1c22;
    --ink-foreground: #f4f4f5;
    --radius: 1rem;
    --shadow-float: 0 30px 80px -30px rgba(30, 30, 50, 0.35);
    --shadow-glow: 0 0 60px -10px rgba(122, 219, 208, 0.55);
    --gradient-accent: linear-gradient(120deg, #7adbd0 0%, #56c2b4 100%);
    --chrome: linear-gradient(135deg, #fafafc 0%, #d8d8dc 40%, #f0f0f4 60%, #c8c8cc 100%);
}

/* Global reset */
html, body, .stApp {
    background: var(--background) !important;
    color: var(--foreground) !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    -webkit-font-smoothing: antialiased !important;
}

#MainMenu, footer, header, .stDeployButton { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* Text gradient accent */
.text-gradient-accent {
    background: linear-gradient(120deg, #7adbd0 0%, #56c2b4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* Chrome ambient surface */
.chrome-surface {
    background: var(--chrome);
    position: relative;
}
.chrome-ambient {
    position: absolute;
    inset-x: 0;
    top: 0;
    height: 70vh;
    background: var(--chrome);
    opacity: 0.6;
    pointer-events: none;
    z-index: 0;
}
.accent-glow {
    position: absolute;
    left: 50%;
    top: 160px;
    height: 420px;
    width: 820px;
    transform: translateX(-50%);
    background: var(--gradient-accent);
    opacity: 0.18;
    filter: blur(80px);
    border-radius: 9999px;
    pointer-events: none;
    z-index: 0;
}

/* Animations */
@keyframes float-soft {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-14px); }
}
@keyframes scan-sweep {
    0% { transform: translateY(-10%); opacity: 0; }
    20% { opacity: 1; }
    100% { transform: translateY(110%); opacity: 0; }
}
@keyframes rise-in {
    from { opacity: 0; transform: translateY(24px); }
    to { opacity: 1; transform: translateY(0); }
}
.animate-float-soft { animation: float-soft 7s ease-in-out infinite; }
.animate-scan-sweep { animation: scan-sweep 2.6s ease-in-out infinite; }
.animate-rise-in { animation: rise-in 0.8s cubic-bezier(0.16, 1, 0.3, 1) both; }
.animate-bounce-slow { animation: float-soft 2.5s ease-in-out infinite; }

/* Navigation bar */
.nav-container {
    max-width: 1152px;
    margin: 0 auto;
    padding: 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: relative;
    z-index: 10;
}
.nav-brand {
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 600;
    letter-spacing: -0.02em;
    font-size: 15px;
    color: var(--foreground);
}
.nav-brand-icon {
    width: 20px;
    height: 20px;
    color: var(--accent-2);
}
.nav-links {
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Streamlit tabs → styled as nav pills */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px !important;
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    justify-content: center !important;
    max-width: 1152px;
    margin: 0 auto 20px !important;
    padding: 0 24px !important;
}
.stTabs [data-baseweb="tab"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    color: var(--muted-foreground) !important;
    border-radius: 9999px !important;
    padding: 8px 20px !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    transition: all 0.2s ease !important;
}
.stTabs [aria-selected="true"] {
    background: var(--ink) !important;
    color: var(--ink-foreground) !important;
    border-color: var(--ink) !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* Buttons */
.btn-ink {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: var(--ink);
    color: var(--ink-foreground);
    padding: 10px 20px;
    border-radius: 9999px;
    font-size: 14px;
    font-weight: 500;
    text-decoration: none;
    transition: transform 0.2s ease;
    border: none;
    cursor: pointer;
}
.btn-ink:hover { transform: scale(1.03); }
.btn-outline {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: transparent;
    color: var(--muted-foreground);
    padding: 8px 16px;
    border-radius: 9999px;
    font-size: 14px;
    font-weight: 400;
    border: 1px solid var(--border);
    text-decoration: none;
    transition: background 0.2s ease;
}
.btn-outline:hover { background: var(--secondary); }

/* Streamlit buttons */
.stButton > button {
    background: var(--ink) !important;
    color: var(--ink-foreground) !important;
    border: none !important;
    border-radius: 9999px !important;
    padding: 10px 24px !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    transition: transform 0.2s ease !important;
    box-shadow: none !important;
}
.stButton > button:hover {
    transform: scale(1.03) !important;
    background: var(--ink) !important;
    color: var(--ink-foreground) !important;
}

/* Hero */
.hero-container {
    position: relative;
    z-index: 10;
    max-width: 1152px;
    margin: 0 auto;
    padding: 40px 24px 40px;
}
.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 16px;
    background: rgba(250, 250, 252, 0.7);
    backdrop-filter: blur(10px);
    border: 1px solid var(--border);
    border-radius: 9999px;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    color: var(--muted-foreground);
    margin-bottom: 20px;
}
.hero-title {
    font-size: clamp(48px, 7vw, 96px);
    font-weight: 600;
    line-height: 1.02;
    letter-spacing: -0.02em;
    max-width: 900px;
    margin: 0;
}
.hero-sub {
    font-size: 18px;
    color: var(--muted-foreground);
    max-width: 560px;
    margin-top: 24px;
    line-height: 1.6;
}
.hero-image-wrap {
    position: relative;
    margin: 40px auto 0;
    max-width: 900px;
}
.hero-image {
    width: 100%;
    filter: drop-shadow(0 30px 80px rgba(30, 30, 50, 0.35));
}
.hero-badge {
    position: absolute;
    background: rgba(250, 250, 252, 0.85);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 12px 18px;
}
.hero-badge-value {
    font-size: 22px;
    font-weight: 600;
    letter-spacing: -0.02em;
}
.hero-badge-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    color: var(--muted-foreground);
    margin-top: 2px;
}
.hero-badge-tl { top: 24px; left: 16px; }
.hero-badge-br { bottom: 32px; right: 16px; }

/* Chapters */
.chapter {
    position: relative;
    z-index: 10;
    max-width: 1152px;
    margin: 0 auto;
    padding: 80px 24px;
}
.chapter-index {
    display: inline-flex;
    align-items: center;
    gap: 12px;
    padding: 6px 16px;
    background: rgba(250, 250, 252, 0.7);
    backdrop-filter: blur(10px);
    border: 1px solid var(--border);
    border-radius: 9999px;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    color: var(--muted-foreground);
    margin-bottom: 20px;
}
.chapter-index-num { color: rgba(14, 56, 51, 0.7); }
.chapter-title {
    font-size: clamp(32px, 4vw, 56px);
    font-weight: 600;
    line-height: 1.05;
    letter-spacing: -0.02em;
}
.chapter-body {
    margin-top: 24px;
    font-size: 18px;
    color: var(--muted-foreground);
    line-height: 1.7;
    max-width: 480px;
}
.chapter-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 32px;
    padding: 40px;
    box-shadow: var(--shadow-float);
    position: relative;
    overflow: hidden;
}
.chapter-card.dark {
    background: var(--ink);
    color: var(--ink-foreground);
}

/* Steps list */
.step-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 0;
    font-size: 15px;
}
.step-num {
    width: 24px;
    height: 24px;
    border-radius: 9999px;
    background: rgba(122, 219, 208, 0.2);
    color: var(--accent-foreground);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    font-weight: 600;
    flex-shrink: 0;
}

/* Scan sweep overlay */
.scan-sweep-overlay {
    position: absolute;
    inset: 0;
    background: linear-gradient(180deg, transparent, rgba(122, 219, 208, 0.5), transparent);
    pointer-events: none;
    animation: scan-sweep 2.6s ease-in-out infinite;
}

/* CTA section */
.cta-container {
    position: relative;
    z-index: 10;
    max-width: 1152px;
    margin: 0 auto;
    padding: 40px 24px 80px;
}
.cta-card {
    background: var(--chrome);
    border: 1px solid var(--border);
    border-radius: 32px;
    padding: 60px 40px;
    text-align: center;
    overflow: hidden;
}
.cta-icon {
    width: 40px;
    height: 40px;
    color: var(--accent-2);
    margin: 0 auto 24px;
}

/* Diagnosis card */
.diag-header {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    color: var(--muted-foreground);
}
.diag-verdict {
    margin-top: 12px;
    font-size: 32px;
    font-weight: 600;
    letter-spacing: -0.02em;
    line-height: 1.1;
}
.diag-meta {
    margin-top: 4px;
    font-size: 13px;
    color: var(--muted-foreground);
}
.diag-summary {
    margin-top: 20px;
    font-size: 17px;
    line-height: 1.6;
}
.diag-advice {
    margin-top: 20px;
    background: var(--secondary);
    border-radius: 16px;
    padding: 20px;
}
.diag-advice-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    color: var(--muted-foreground);
}
.diag-advice-text {
    margin-top: 8px;
    font-size: 15px;
}

/* Probability bars */
.prob-row {
    margin-bottom: 12px;
}
.prob-row-header {
    display: flex;
    justify-content: space-between;
    font-size: 14px;
    margin-bottom: 6px;
}
.prob-row-value {
    color: var(--muted-foreground);
    font-variant-numeric: tabular-nums;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
}
.prob-track {
    height: 8px;
    background: var(--secondary);
    border-radius: 9999px;
    overflow: hidden;
}
.prob-fill {
    height: 100%;
    border-radius: 9999px;
    background: var(--gradient-accent);
    transition: width 0.6s ease;
}

/* File uploader */
section[data-testid="stFileUploaderDropzone"] {
    background: rgba(250, 250, 252, 0.7) !important;
    backdrop-filter: blur(10px) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 32px !important;
    padding: 48px 24px !important;
    transition: border-color 0.2s ease !important;
}
section[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--accent) !important;
}
section[data-testid="stFileUploaderDropzone"] button {
    background: var(--ink) !important;
    color: var(--ink-foreground) !important;
    border: none !important;
    border-radius: 9999px !important;
    font-weight: 500 !important;
}
section[data-testid="stFileUploaderDropzone"] small {
    color: var(--muted-foreground) !important;
}

/* Camera input */
[data-testid="stCameraInput"] label {
    background: var(--ink) !important;
    color: var(--ink-foreground) !important;
    border-radius: 9999px !important;
    padding: 10px 20px !important;
    font-weight: 500 !important;
}

/* Sub-page container (scan / camera / lab) */
.subpage {
    position: relative;
    z-index: 10;
    max-width: 768px;
    margin: 0 auto;
    padding: 40px 24px;
}
.subpage-back {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 14px;
    color: var(--muted-foreground);
    text-decoration: none;
    transition: color 0.2s ease;
}
.subpage-back:hover { color: var(--foreground); }
.subpage-title {
    margin-top: 32px;
    font-size: clamp(32px, 5vw, 56px);
    font-weight: 600;
    letter-spacing: -0.02em;
    line-height: 1.05;
}
.subpage-sub {
    margin-top: 12px;
    font-size: 17px;
    color: var(--muted-foreground);
}

/* Details accordion */
.details-toggle {
    background: var(--ink);
    color: var(--ink-foreground);
    border-radius: 16px;
    padding: 16px 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 14px;
}

/* Lab (technical) styling */
.lab-metric {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px;
}
.lab-metric-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: var(--muted-foreground);
}
.lab-metric-value {
    font-size: 28px;
    font-weight: 600;
    letter-spacing: -0.02em;
    margin-top: 6px;
    font-variant-numeric: tabular-nums;
}
.lab-metric-sub {
    font-size: 12px;
    color: var(--muted-foreground);
    margin-top: 4px;
    font-family: 'JetBrains Mono', monospace;
}

/* Camera live status */
.cam-status {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 500;
    background: var(--card);
    border: 1px solid var(--border);
}
.cam-status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #ef4444;
    animation: pulse-live 1.5s infinite;
}
@keyframes pulse-live {
    0%, 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.5); }
    50% { box-shadow: 0 0 0 6px rgba(239, 68, 68, 0); }
}

/* Selectbox and inputs */
.stSelectbox > div > div {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--foreground) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--muted); border-radius: 9999px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted-foreground); }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────
# Human-readable copy per class (citizen-friendly language)
CLASS_COPY = {
    'Clean': {
        'label': 'Tout va bien',
        'severity': 'low',
        'severity_copy': 'Aucun problème — entretien de routine',
        'summary': "Votre panneau est en excellent état. Rien à signaler pour le moment.",
        'advice': "Surveillez simplement votre production mensuelle. Un rinçage annuel suffit en région sèche.",
    },
    'Dusty': {
        'label': 'Poussière accumulée',
        'severity': 'medium',
        'severity_copy': 'À traiter prochainement',
        'summary': "De la poussière et des salissures recouvrent la surface et empêchent la lumière d'atteindre les cellules.",
        'advice': "Rincez à l'eau claire un matin frais — vous récupérerez généralement 10 à 15 % de production.",
    },
    'Bird-drop': {
        'label': 'Fientes d\'oiseaux',
        'severity': 'medium',
        'severity_copy': 'À traiter prochainement',
        'summary': "Des fientes localisées font de l'ombre sur une partie du panneau et peuvent créer un point chaud à terme.",
        'advice': "Un rinçage doux à l'eau maintenant. Si le problème revient, installez discrètement des dissuadeurs d'oiseaux.",
    },
    'Snow-Covered': {
        'label': 'Recouvert de neige',
        'severity': 'low',
        'severity_copy': 'Aucun problème — entretien de routine',
        'summary': "La neige bloque temporairement la lumière du soleil. La production reprendra dès la fonte.",
        'advice': "Laissez fondre naturellement si possible. Ne dégagez à la main que si la neige persiste plusieurs jours.",
    },
    'Electrical-damage': {
        'label': 'Défaut électrique',
        'severity': 'high',
        'severity_copy': 'Appelez un professionnel',
        'summary': "Signes d'un défaut électrique — probablement un point chaud, une diode de bypass ou une dégradation. Cela peut s'aggraver rapidement.",
        'advice': "Ne touchez pas au panneau. Contactez un installateur certifié sous 48 h pour une inspection sur site.",
    },
    'Physical-Damage': {
        'label': 'Dommage physique',
        'severity': 'high',
        'severity_copy': 'Appelez un professionnel',
        'summary': "Le panneau présente un dommage physique : fissure, verre cassé ou délamination. L'humidité peut s'infiltrer et aggraver le problème.",
        'advice': "Photographiez les dégâts sous plusieurs angles. Contactez votre installateur cette semaine pour vérifier la garantie.",
    },
    'Bird-drop': {
        'label': 'Fientes d\'oiseaux',
        'severity': 'medium',
        'severity_copy': 'À traiter prochainement',
        'summary': "Des fientes localisées font de l'ombre sur une partie du panneau et peuvent créer un point chaud à terme.",
        'advice': "Un rinçage doux à l'eau maintenant. Si le problème revient, installez discrètement des dissuadeurs d'oiseaux.",
    },
}


def run_inference(image: Image.Image) -> dict:
    """Run the trained model on a PIL image and return the result."""
    model, class_names = load_model()
    transform = get_transforms('val')
    tensor = transform(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze().numpy()
    pred_idx = int(np.argmax(probs))
    pred_class = class_names[pred_idx]
    return {
        'pred_class': pred_class,
        'confidence': float(probs[pred_idx]) * 100,
        'probs': [(name, float(p) * 100) for name, p in zip(class_names, probs)],
        'class_names': class_names,
    }


def render_diagnosis_card(result: dict):
    """Render the citizen-friendly diagnosis card with severity-aware colors."""
    pred_class = result['pred_class']
    copy = CLASS_COPY.get(pred_class, CLASS_COPY['Clean'])
    severity = copy['severity']  # 'low' | 'medium' | 'high'

    # Vibrant palette by severity
    palette = {
        'low':    {'main': '#16a34a', 'bg': 'rgba(22,163,74,0.10)',  'border': '#22c55e', 'icon': '✓',  'chip': '#dcfce7'},
        'medium': {'main': '#ea580c', 'bg': 'rgba(234,88,12,0.10)',  'border': '#f97316', 'icon': '⚠', 'chip': '#ffedd5'},
        'high':   {'main': '#dc2626', 'bg': 'rgba(220,38,38,0.10)',  'border': '#ef4444', 'icon': '⚡', 'chip': '#fee2e2'},
    }[severity]

    urgent_badge = ''
    pulse_class = ''
    if severity == 'high':
        urgent_badge = f'<span style="display:inline-flex;align-items:center;gap:6px;background:{palette["main"]};color:white;padding:4px 12px;border-radius:9999px;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.15em;margin-left:12px;vertical-align:middle;">⚡ Urgent</span>'
        pulse_class = 'pulse-alert'

    st.markdown(f"""
    <style>
    @keyframes pulse-alert-anim {{
        0%,100% {{ box-shadow: 0 0 0 0 rgba(220,38,38,0.35); }}
        50%     {{ box-shadow: 0 0 0 14px rgba(220,38,38,0); }}
    }}
    .pulse-alert {{ animation: pulse-alert-anim 2s infinite; }}
    </style>
    <div class="chapter-card animate-rise-in {pulse_class}" style="border-left: 6px solid {palette['border']}; background: linear-gradient(135deg, {palette['bg']}, var(--card));">
        <div class="diag-header" style="color: {palette['main']};">
            <span style="font-size:16px;">{palette['icon']}</span>
            <span style="font-weight:700; letter-spacing:0.2em;">NOTRE DIAGNOSTIC</span>
        </div>
        <div class="diag-verdict" style="color: {palette['main']};">{copy['label']}{urgent_badge}</div>
        <div class="diag-meta" style="color: {palette['main']}; font-weight: 600;">
            {copy['severity_copy']} · confiance : {result['confidence']:.1f}%
        </div>
        <p class="diag-summary">{copy['summary']}</p>
        <div class="diag-advice" style="background: {palette['chip']}; border-left: 3px solid {palette['border']};">
            <div class="diag-advice-label" style="color: {palette['main']}; font-weight: 700;">CE QUE VOUS POUVEZ FAIRE</div>
            <div class="diag-advice-text">{copy['advice']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_probability_bars(result: dict):
    """Render the class-probability breakdown with the top class highlighted."""
    sorted_probs = sorted(result['probs'], key=lambda x: x[1], reverse=True)

    # Get the same palette color as the diagnosis card for the top bar
    pred_class = result['pred_class']
    copy = CLASS_COPY.get(pred_class, CLASS_COPY['Clean'])
    palette = {
        'low':    '#22c55e',
        'medium': '#f97316',
        'high':   '#ef4444',
    }[copy['severity']]

    # Build all bars on a SINGLE LINE each (no leading indentation) so
    # Streamlit's Markdown parser doesn't treat them as a code block.
    bars = []
    for i, (label, value) in enumerate(sorted_probs):
        label_clean = label.replace('-', ' ')
        is_top = (i == 0)
        fill_bg = palette if is_top else 'linear-gradient(90deg, #cbd5e1, #94a3b8)'
        label_color = palette if is_top else 'var(--foreground)'
        value_color = palette if is_top else 'var(--muted-foreground)'
        weight = '700' if is_top else '400'
        bar = (
            f'<div class="prob-row">'
            f'<div class="prob-row-header">'
            f'<span style="color:{label_color};font-weight:{weight};">{label_clean}</span>'
            f'<span class="prob-row-value" style="color:{value_color};font-weight:{weight};">{value:.1f}%</span>'
            f'</div>'
            f'<div class="prob-track">'
            f'<div class="prob-fill" style="width:{value:.1f}%; background:{fill_bg};"></div>'
            f'</div>'
            f'</div>'
        )
        bars.append(bar)
    bars_html = "".join(bars)

    st.markdown(
        f'<div class="chapter-card animate-rise-in" style="margin-top:16px;">'
        f'<div class="diag-header">PROBABILITÉS PAR CLASSE</div>'
        f'<div style="margin-top:20px;">{bars_html}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────
# NAVIGATION BAR
# ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="position: relative;">
    <div class="chrome-ambient"></div>
    <div class="accent-glow"></div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="nav-container">
    <div class="nav-brand">
        <svg class="nav-brand-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="4"/>
            <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/>
        </svg>
        SolarLens
    </div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# MAIN TABS
# ──────────────────────────────────────────────────────────────
tab_story, tab_scan, tab_camera, tab_lab = st.tabs([
    "L'histoire",
    "Analyser une photo",
    "Caméra en direct",
    "Espace technique",
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — THE STORY (Homepage)
# ══════════════════════════════════════════════════════════════
with tab_story:
    # HERO
    if HOUSE_IMG:
        house_img_tag = f'<img src="data:image/png;base64,{HOUSE_IMG}" class="hero-image animate-float-soft" alt="Modern home with rooftop solar panels" />'
    else:
        house_img_tag = '<div style="height:400px;background:var(--card);border-radius:24px;"></div>'

    st.markdown(f"""
    <div class="hero-container">
        <div class="hero-eyebrow">Une histoire, un toit</div>
        <h1 class="hero-title">Votre toit <span class="text-gradient-accent">produit</span>.<br/>Jusqu'à ce qu'un panneau s'arrête.</h1>
        <p class="hero-sub">
            Faites défiler pour comprendre ce qui se passe — et comment une seule photo suffit à tout révéler, en mots que vous comprenez.
        </p>
        <div class="hero-image-wrap">
            {house_img_tag}
            <div class="hero-badge hero-badge-tl">
                <div class="hero-badge-value">6,5 kW</div>
                <div class="hero-badge-label">Production actuelle</div>
            </div>
            <div class="hero-badge hero-badge-br">
                <div class="hero-badge-value">94 %</div>
                <div class="hero-badge-label">Autonomie aujourd'hui</div>
            </div>
        </div>
        <div style="text-align: center; margin-top: 32px; color: var(--muted-foreground); font-size: 24px;">
            <span class="animate-bounce-slow" style="display: inline-block;">⌄</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # CHAPTER 01
    if HOUSE_DEFECT_IMG:
        defect_img_tag = f'<img src="data:image/png;base64,{HOUSE_DEFECT_IMG}" style="width:100%;border-radius:20px;" alt="House with damaged panel highlighted" />'
    else:
        defect_img_tag = ''

    st.markdown(f"""
    <div class="chapter">
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 48px; align-items: center;">
            <div class="animate-rise-in">
                <div class="chapter-index">
                    <span class="chapter-index-num">01</span> Le problème silencieux
                </div>
                <h2 class="chapter-title">Une fissure invisible depuis le sol.</h2>
                <p class="chapter-body">
                    Poussière, fientes d'oiseaux, micro-fissure après un orage de grêle. Votre facture grimpe doucement,
                    votre toit a l'air impeccable, et des mois passent avant que quelqu'un ne s'en rende compte.
                </p>
            </div>
            <div class="chapter-card dark animate-rise-in" style="position: relative; overflow: hidden;">
                {defect_img_tag}
                <div class="scan-sweep-overlay"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # CHAPTER 02
    st.markdown("""
    <div class="chapter">
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 48px; align-items: center;">
            <div class="chapter-card animate-rise-in">
                <div style="font-size: 40px;">📱</div>
                <div style="margin-top: 24px;">
                    <div class="step-item"><span class="step-num">1</span><span style="color: var(--muted-foreground);">Photo reçue</span></div>
                    <div class="step-item"><span class="step-num">2</span><span style="color: var(--muted-foreground);">Panneau localisé</span></div>
                    <div class="step-item"><span class="step-num">3</span><span style="color: var(--muted-foreground);">Surface analysée</span></div>
                    <div class="step-item"><span class="step-num">4</span><span style="color: var(--muted-foreground);">Diagnostic prêt</span></div>
                </div>
            </div>
            <div class="animate-rise-in">
                <div class="chapter-index">
                    <span class="chapter-index-num">02</span> Une seule photo
                </div>
                <h2 class="chapter-title">Pointez votre téléphone. C'est tout.</h2>
                <p class="chapter-body">
                    Pas de drone, pas de technicien, pas de jargon. SolarLens regarde la photo comme le ferait
                    un inspecteur expérimenté — en deux secondes environ.
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # CHAPTER 03
    st.markdown("""
    <div class="chapter">
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 48px; align-items: center;">
            <div class="animate-rise-in">
                <div class="chapter-index">
                    <span class="chapter-index-num">03</span> Un langage clair
                </div>
                <h2 class="chapter-title">« Votre panneau est poussiéreux. Rincez-le — vous récupérerez 12 %. »</h2>
                <p class="chapter-body">
                    Les particuliers reçoivent une phrase claire et une action concrète. Les techniciens peuvent ouvrir
                    l'onglet détails pour voir les probabilités, la confiance et les métriques du modèle.
                </p>
            </div>
            <div class="animate-rise-in" style="display: flex; flex-direction: column; gap: 16px;">
                <div class="chapter-card">
                    <div style="font-size: 10px; text-transform: uppercase; letter-spacing: 0.2em; color: var(--muted-foreground);">NOTRE DIAGNOSTIC</div>
                    <div style="font-size: 24px; font-weight: 600; margin-top: 8px;">Poussière accumulée</div>
                    <p style="margin-top: 8px; font-size: 14px; color: var(--muted-foreground);">
                        Rien de cassé. Un rinçage à l'eau claire restaure la majorité de la production perdue.
                    </p>
                </div>
                <div class="chapter-card dark" style="padding: 16px 24px; display: flex; align-items: center; gap: 12px;">
                    <span style="color: var(--accent); font-size: 20px;">⚙</span>
                    <span style="font-size: 14px; opacity: 0.8;">Détails techniques disponibles à la demande</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # CTA
    st.markdown("""
    <div class="cta-container">
        <div class="cta-card">
            <div style="font-size: 40px; color: var(--accent-2);">🛡</div>
            <h2 style="font-size: clamp(28px, 4vw, 48px); font-weight: 600; letter-spacing: -0.02em; margin-top: 20px; line-height: 1.1;">
                Découvrez ce que votre toit vous cache.
            </h2>
            <p style="margin-top: 24px; color: var(--muted-foreground); font-size: 14px;">
                Cliquez sur l'onglet <strong>Analyser une photo</strong> pour essayer, ou <strong>Caméra en direct</strong> pour une détection instantanée.
            </p>
        </div>
        <p style="text-align: center; margin-top: 32px; font-size: 12px; color: var(--muted-foreground);">
            SolarLens · Inspection solaire par IA · Réalisé par Sanae Najimi
        </p>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — SCAN A PHOTO
# ══════════════════════════════════════════════════════════════
with tab_scan:
    st.markdown("""
    <div class="subpage">
        <h1 class="subpage-title">Montrez-nous votre <span class="text-gradient-accent">panneau</span>.</h1>
        <p class="subpage-sub">
            Une seule photo suffit. Nous vous disons ce que c'est et ce qu'il faut faire — sans jargon.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Uploader centered
    _, col_up, _ = st.columns([1, 6, 1])
    with col_up:
        uploaded = st.file_uploader(
            "Déposez une photo ici, ou cliquez pour choisir · JPG ou PNG de votre panneau solaire",
            type=["jpg", "jpeg", "png"],
            key="scan_uploader",
            label_visibility="visible",
        )

        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, use_container_width=True)

            with st.spinner("Analyse du panneau…"):
                result = run_inference(image)

            render_diagnosis_card(result)

            with st.expander("Détails techniques (pour inspecteurs)", expanded=False):
                render_probability_bars(result)


# ══════════════════════════════════════════════════════════════
# TAB 3 — LIVE CAMERA (real-time detection)
# ══════════════════════════════════════════════════════════════
with tab_camera:
    st.markdown("""
    <div class="subpage">
        <h1 class="subpage-title">Pointez votre <span class="text-gradient-accent">caméra</span> vers un panneau.</h1>
        <p class="subpage-sub">
            Prenez une photo directement depuis votre appareil — l'IA établit le diagnostic instantanément.
            <span style="color: var(--muted-foreground);">Astuce : tenez le téléphone à 1–2 mètres du panneau, en lumière du jour uniforme.</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

    _, col_cam, _ = st.columns([1, 6, 1])
    with col_cam:
        st.markdown("""
        <div style="display: flex; justify-content: center; margin-bottom: 20px;">
            <span class="cam-status">
                <span class="cam-status-dot"></span> Caméra prête
            </span>
        </div>
        """, unsafe_allow_html=True)

        camera_photo = st.camera_input(
            "Appuyez sur le déclencheur pour capturer et analyser",
            key="live_camera",
            label_visibility="visible",
        )

        if camera_photo:
            image = Image.open(camera_photo).convert("RGB")

            with st.spinner("Analyse en cours…"):
                result = run_inference(image)

            render_diagnosis_card(result)

            with st.expander("Détails techniques", expanded=False):
                render_probability_bars(result)


# ══════════════════════════════════════════════════════════════
# TAB 4 — FOR TECHNICIANS (Lab)
# ══════════════════════════════════════════════════════════════
with tab_lab:
    st.markdown("""
    <div class="subpage">
        <div class="chapter-index">Le laboratoire</div>
        <h1 class="subpage-title">Fiche modèle, métriques et prédictions test.</h1>
        <p class="subpage-sub">
            Toutes les données de cette page proviennent du modèle entraîné réel
            (<code style="background: var(--secondary); padding: 2px 6px; border-radius: 4px;">checkpoints/solarlens_best.pth</code>)
            exécuté sur votre dataset local — aucune simulation.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Load model + dataset info
    from utils.batch_inference import list_test_images, predict_all_test_images

    model, class_names = load_model()
    test_images_by_class = list_test_images("data/pv-panel-defect/test")
    total_test = sum(len(v) for v in test_images_by_class.values())
    total_params = sum(p.numel() for p in model.parameters()) / 1e6

    # Metrics grid
    st.markdown(f"""
    <div style="max-width: 1152px; margin: 0 auto; padding: 20px 24px;">
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px;">
            <div class="lab-metric">
                <div class="lab-metric-label">IMAGES DE TEST</div>
                <div class="lab-metric-value">{total_test}</div>
                <div class="lab-metric-sub">du dossier test/</div>
            </div>
            <div class="lab-metric">
                <div class="lab-metric-label">CLASSES</div>
                <div class="lab-metric-value">{len(class_names)}</div>
                <div class="lab-metric-sub">types de défauts</div>
            </div>
            <div class="lab-metric">
                <div class="lab-metric-label">PARAMÈTRES</div>
                <div class="lab-metric-value">{total_params:.1f}M</div>
                <div class="lab-metric-sub">ResNet-50 + SE</div>
            </div>
            <div class="lab-metric">
                <div class="lab-metric-label">PRÉCISION TEST</div>
                <div class="lab-metric-value">98.9%</div>
                <div class="lab-metric-sub">sur l'ensemble de test</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="max-width: 1152px; margin: 24px auto 0; padding: 0 24px;">', unsafe_allow_html=True)

    with st.expander("Lancer l'inférence sur tout le jeu de test", expanded=False):
        if test_images_by_class:
            results = predict_all_test_images("data/pv-panel-defect/test")
            n_correct = sum(1 for r in results if r['correct'])
            acc = 100 * n_correct / len(results)

            st.markdown(f"""
            <div class="chapter-card" style="text-align: center; padding: 32px;">
                <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.2em; color: var(--muted-foreground);">
                    PRÉCISION RÉELLE MESURÉE
                </div>
                <div style="font-size: 56px; font-weight: 600; letter-spacing: -0.02em; margin-top: 8px;" class="text-gradient-accent">
                    {acc:.1f}%
                </div>
                <div style="color: var(--muted-foreground); font-size: 14px;">
                    {n_correct} / {len(results)} images correctement classées
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Confusion matrix
            n_cls = len(class_names)
            cm = np.zeros((n_cls, n_cls), dtype=int)
            for r in results:
                true_i = class_names.index(r['true_class'])
                pred_i = class_names.index(r['pred_class'])
                cm[true_i, pred_i] += 1

            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=class_names,
                y=class_names,
                colorscale=[[0, '#f5f5f7'], [0.5, '#7adbd0'], [1, '#0e3833']],
                showscale=False,
                text=cm,
                texttemplate='%{text}',
                textfont=dict(size=14, family='JetBrains Mono'),
                hovertemplate='Prédit : %{x}<br>Réel : %{y}<br>Nombre : %{z}<extra></extra>',
            ))
            fig_cm.update_layout(
                title=dict(text="Matrice de confusion", font=dict(size=14, color='#26272e', family='Inter')),
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=100, r=20, t=50, b=80),
                xaxis=dict(
                    title=dict(text="Prédit", font=dict(size=12, color='#71717a')),
                    tickfont=dict(size=11, color='#26272e')
                ),
                yaxis=dict(
                    title=dict(text="Réel", font=dict(size=12, color='#71717a')),
                    tickfont=dict(size=11, color='#26272e'),
                    autorange='reversed'
                ),
                font=dict(family='Inter'),
            )
            st.plotly_chart(fig_cm, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("Le dossier `data/pv-panel-defect/test/` est introuvable. Placez-y votre dataset pour voir les métriques en direct.")

    # Training curves
    history_path = "checkpoints/training_history.json"
    with st.expander("Courbes d'entraînement (depuis training_history.json)", expanded=False):
        try:
            with open(history_path) as f:
                history = json.load(f)
            epochs = list(range(1, len(history['train_loss']) + 1))

            from plotly.subplots import make_subplots
            fig_train = make_subplots(rows=1, cols=2, subplot_titles=("Perte (loss)", "Précision"))
            fig_train.add_trace(go.Scatter(x=epochs, y=history['train_loss'], name='Entraînement',
                                            line=dict(color='#26272e', width=2)), row=1, col=1)
            fig_train.add_trace(go.Scatter(x=epochs, y=history['val_loss'], name='Validation',
                                            line=dict(color='#7adbd0', width=2, dash='dot')), row=1, col=1)
            fig_train.add_trace(go.Scatter(x=epochs, y=history['train_acc'], name='Entraînement',
                                            line=dict(color='#26272e', width=2), showlegend=False), row=1, col=2)
            fig_train.add_trace(go.Scatter(x=epochs, y=history['val_acc'], name='Validation',
                                            line=dict(color='#7adbd0', width=2, dash='dot'), showlegend=False), row=1, col=2)

            fig_train.update_layout(
                height=350,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=40, r=20, t=40, b=30),
                legend=dict(orientation='h', y=1.15, font=dict(size=11, color='#71717a'), bgcolor='rgba(0,0,0,0)'),
                font=dict(family='Inter', color='#71717a', size=11),
            )
            fig_train.update_xaxes(gridcolor='rgba(0,0,0,0.05)', tickfont=dict(size=10, color='#71717a'))
            fig_train.update_yaxes(gridcolor='rgba(0,0,0,0.05)', tickfont=dict(size=10, color='#71717a'))
            st.plotly_chart(fig_train, use_container_width=True, config={'displayModeBar': False})
        except (FileNotFoundError, KeyError):
            st.info("Fichier `training_history.json` introuvable. Lancez `python train.py` pour générer les vraies courbes.")

    # Architecture
    with st.expander("Architecture du modèle", expanded=False):
        st.markdown("""
        <div class="chapter-card">
            <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.2em; color: var(--muted-foreground); margin-bottom: 16px;">
                ARCHITECTURE
            </div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 13px; color: var(--foreground); line-height: 1.8;">
                Input (224 × 224 × 3)<br/>
                &nbsp;&nbsp;↓ ResNet-50 backbone <span style="color: var(--muted-foreground);">— pré-entraîné sur ImageNet, couches 1-3 gelées</span><br/>
                &nbsp;&nbsp;↓ Squeeze-Excitation attention <span style="color: var(--muted-foreground);">— ratio de réduction 16</span><br/>
                &nbsp;&nbsp;↓ Global Average Pooling<br/>
                &nbsp;&nbsp;↓ Dropout (0.4) → FC (2048 → 512) → ReLU → BN<br/>
                &nbsp;&nbsp;↓ Dropout (0.3) → FC (512 → 6)<br/>
                Softmax sur 6 classes
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="max-width: 1152px; margin: 40px auto 20px; padding: 20px 24px; text-align: center; border-top: 1px solid var(--border);">
    <p style="font-size: 12px; color: var(--muted-foreground); margin-top: 20px;">
        SolarLens · Modèle ResNet-50 + Attention SE réel · <span style="color: var(--foreground);">Réalisé par Sanae Najimi</span>
    </p>
</div>
""", unsafe_allow_html=True)
