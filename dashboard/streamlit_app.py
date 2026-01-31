import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="News → Price Predictor", layout="wide")


st.markdown("""
<style>
    .stApp { background: #f8f9fc; }

    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1e3a8a, #3b82f6, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin: 0.4em 0 0.8em 0;
    }

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
        font-size: 1.1rem;
        color: rgba(30, 58, 138, 0.7);
        white-space: nowrap;
        animation: ticker-slide 60s linear infinite;
    }

    @keyframes ticker-slide {
        0% { transform: translateX(100%); }
        100% { transform: translateX(-100%); }
    }

    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.65rem 1.5rem !important;
        background: linear-gradient(90deg, #2563eb, #3b82f6) !important;
        color: white !important;
        border: none !important;
    }
</style>

<div class="ticker-wrap">
  <div class="ticker">
    NVIDIA • AAPL • GOOGL • MSFT • AMZN • META • TSLA • JPM • V • WMT • COST
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Financial News → Next-Day Price Direction</div>', unsafe_allow_html=True)


st.markdown("### 📰 News Input")
headline = st.text_area(
    "News Headline / Summary",
    height=140,
    placeholder="Paste the headline or first paragraph here..."
)

ticker = st.text_input("Ticker Symbol (optional)", placeholder="e.g. AAPL, TSLA")

st.markdown("### 📈 Market Data (Required for LSTM)")

# Row 1
col1, col2, col3 = st.columns(3)
with col1:
    open_p = st.number_input("Open Price", value=0.0, format="%.4f")
with col2:
    high_p = st.number_input("High Price", value=0.0, format="%.4f")
with col3:
    low_p = st.number_input("Low Price", value=0.0, format="%.4f")

# Row 2
col4, col5, col6 = st.columns(3)
with col4:
    close_p = st.number_input("Close Price", value=0.0, format="%.4f")
with col5:
    volume = st.number_input("Volume", value=0, step=1000)
with col6:
    sma_5 = st.number_input("SMA 5", value=0.0, format="%.4f")


if st.button("🚀 Predict Direction", use_container_width=True):

    if not headline.strip():
        st.warning("Please enter a news headline.")
        st.stop()

    payload = {
        "headline": headline.strip(),
        "Ticker": ticker.strip() or None,
        "Open": float(open_p),
        "High": float(high_p),
        "Low": float(low_p),
        "Close": float(close_p),
        "Volume": float(volume),
        "SMA_5": float(sma_5),
    }

    with st.spinner("Predicting using FinBERT + LSTM..."):
        try:
            resp = requests.post(
                "http://localhost:8000/api/v1/predict",
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            prob = data.get("prob_up", 0.5)
            direction = "UP 📈" if data.get("label_up", prob >= 0.5) else "DOWN 📉"

            st.metric(
                label="Probability of Next-Day UP",
                value=f"{prob:.1%}",
                delta=direction
            )

            st.caption(f"Predicted direction: **{direction}**")

            
            st.subheader("🧠 Sentiment Analysis")
            df_sent = pd.DataFrame({
                "Source": ["FinBERT", "FinBERT", "FinBERT", "VADER"],
                "Metric": ["Positive", "Negative", "Neutral", "Compound"],
                "Value": [
                    f"{data.get('finbert_positive', 0):.3f}",
                    f"{data.get('finbert_negative', 0):.3f}",
                    f"{data.get('finbert_neutral', 0):.3f}",
                    f"{data.get('vader_compound', 0):.3f}",
                ],
            })
            st.dataframe(df_sent, hide_index=True, use_container_width=True)

            if msg := data.get("message"):
                st.info(msg)

        except requests.exceptions.RequestException as e:
            st.error(f"API connection failed:\n{e}")
        except Exception as e:
            st.error(f"Unexpected error:\n{e}")
