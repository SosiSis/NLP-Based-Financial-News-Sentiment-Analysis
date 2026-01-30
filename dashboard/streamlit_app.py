import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="News → Price Predictor", layout="wide")

# ────────────────────────────────────────
# Modern clean CSS + horizontal ticker
# ────────────────────────────────────────
st.markdown("""
<style>
    /* Light clean background */
    .stApp {
        background: #f8f9fc;
    }

    /* Better looking title */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1e3a8a, #3b82f6, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin: 0.4em 0 0.8em 0;
        letter-spacing: -0.5px;
    }

    /* Horizontal scrolling ticker */
    .ticker-wrap {
        width: 100%;
        overflow: hidden;
        background: rgba(30, 58, 138, 0.06);
        padding: 8px 0;
        margin-bottom: 1.5rem;
        border-radius: 8px;
    }

    .ticker {
        display: inline-block;
        padding: 0 2rem;
        font-weight: 700;
        font-size: 1.18rem;
        color: rgba(30, 58, 138, 0.7);
        white-space: nowrap;
        animation: ticker-slide 60s linear infinite;
    }

    .ticker span {
        margin-right: 4rem;
    }

    @keyframes ticker-slide {
        0%   { transform: translateX(100%); }
        100% { transform: translateX(-100%); }
    }

    /* Improve input/button appearance */
    .stTextArea > div > div > textarea,
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        border-radius: 10px !important;
        border: 1px solid #d1d5db !important;
        background-color: white !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
    }

    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.65rem 1.5rem !important;
        background: linear-gradient(90deg, #2563eb, #3b82f6) !important;
        color: white !important;
        border: none !important;
        transition: all 0.2s;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(59,130,246,0.35) !important;
    }

    /* Better spacing for columns */
    .input-row {
        gap: 1.2rem !important;
    }
</style>

<div class="ticker-wrap">
   <div class="ticker">
    <span>NVIDIA • AAPL • GOOGL • MSFT • AMZN • META • AVGO • TSLA • LLY • BRK.B • WMT • JPM • V • UNH • XOM • MA • COST • HD • PG • JNJ</span>
    <span>NVIDIA • AAPL • GOOGL • MSFT • AMZN • META • AVGO • TSLA • LLY • BRK.B • WMT • JPM • V • UNH • XOM • MA • COST • HD • PG • JNJ</span>
    <span>NVIDIA • AAPL • GOOGL • MSFT • AMZN • META • AVGO • TSLA • LLY • BRK.B • WMT • JPM • V • UNH • XOM • MA • COST • HD • PG • JNJ</span>
    <span>NVIDIA • AAPL • GOOGL • MSFT • AMZN • META • AVGO • TSLA • LLY • BRK.B • WMT • JPM • V • UNH • XOM • MA • COST • HD • PG • JNJ</span>
</div>
</div>
""", unsafe_allow_html=True)

# ────────────────────────────────────────
# Header
# ────────────────────────────────────────
st.markdown('<div class="main-title">Financial News → Next-day Price Direction</div>', unsafe_allow_html=True)

# ────────────────────────────────────────
# Form-like inputs
# ────────────────────────────────────────
st.markdown("### Enter News & Market Data")

headline = st.text_area("News Headline / Summary", height=140, 
                       placeholder="Paste the headline or first paragraph here...")

ticker = st.text_input("Ticker Symbol (optional)", placeholder="e.g. AAPL, TSLA, NVDA")

st.markdown('<div class="input-row">', unsafe_allow_html=True)
col1, col2, col3 = st.columns([1,1,1.3])
with col1:
    open_p = st.number_input("Open Price", value=0.00, format="%.4f", step=0.01)
with col2:
    close_p = st.number_input("Close Price", value=0.00, format="%.4f", step=0.01)
with col3:
    volume = st.number_input("Volume", value=0, format="%d", step=1000)
st.markdown('</div>', unsafe_allow_html=True)

# ────────────────────────────────────────
# Prediction button & logic
# ────────────────────────────────────────
if st.button("Predict Direction", type="primary", use_container_width=True):
    if not headline.strip():
        st.warning("Please enter at least a headline.")
        st.stop()

    payload = {
        "headline": headline.strip(),
        "ticker": ticker.strip() or None,
        "open": float(open_p),
        "close": float(close_p),
        "volume": float(volume),
    }

    with st.spinner("Predicting..."):
        try:
            # Change URL if your backend is running elsewhere
            resp = requests.post("http://localhost:8000/api/v1/predict", 
                               json=payload, 
                               timeout=30)
            resp.raise_for_status()
            data = resp.json()

            # ── Main result ───────────────────────────────────
            prob = data.get("prob_up", 0.5)
            direction = "UP" if data.get("label_up", prob >= 0.5) else "DOWN"
            
            delta_color = "normal" if abs(prob - 0.5) < 0.1 else ("normal" if prob > 0.5 else "off")
            
            st.metric(
                label="Probability next-day **UP**",
                value=f"{prob:.1%}",
                delta=direction,
                delta_color=delta_color
            )

            st.caption(f"Predicted direction: **{direction}**")

            # ── Sentiment breakdown ───────────────────────────
            if "finbert_positive" in data:
                st.subheader("Sentiment Breakdown")
                df_sent = pd.DataFrame({
                    "Model": ["FinBERT", "FinBERT", "FinBERT", "VADER"],
                    "Class/Score": ["Positive", "Negative", "Neutral", "Compound"],
                    "Value": [
                        f"{data['finbert_positive']:.3f}",
                        f"{data['finbert_negative']:.3f}",
                        f"{data['finbert_neutral']:.3f}",
                        f"{data.get('vader_compound', '—')}"
                    ]
                })
                st.dataframe(df_sent, hide_index=True, use_container_width=True)

            if msg := data.get("message"):
                st.info(msg)

        except requests.exceptions.RequestException as e:
            st.error(f"Cannot reach prediction API\n\n{str(e)}")
        except Exception as e:
            st.error(f"Unexpected error\n\n{str(e)}")