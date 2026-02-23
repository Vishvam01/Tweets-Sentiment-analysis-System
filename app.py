import streamlit as st
import pandas as pd
import re
import plotly.express as px
from textblob import TextBlob

# ----------------------------
# Page Setup
# ----------------------------
st.set_page_config(
    page_title="Twitter Sentiment Analyzer",
    page_icon="📊",
    layout="wide"
)

# ----------------------------
# Custom Header
# ----------------------------
st.markdown("""
<style>
.header {
    background: linear-gradient(90deg, #1f77b4, #17becf);
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    color: white;
    font-size: 32px;
    font-weight: bold;
}
</style>
<div class="header">🚀 Twitter Sentiment Analysis Dashboard</div>
""", unsafe_allow_html=True)

st.write("Analyze tweets individually or explore dataset insights.")

# ----------------------------
# Clean Text
# ----------------------------
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^A-Za-z ]+", "", text)
    return text.lower()

# ----------------------------
# Sentiment + Confidence
# ----------------------------
def analyze_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity

    if polarity > 0:
        label = "😊 Positive"
    elif polarity < 0:
        label = "😡 Negative"
    else:
        label = "😐 Neutral"

    confidence = abs(polarity) * 100
    return label, round(confidence, 2)

# ----------------------------
# Tabs Layout
# ----------------------------
tab1, tab2 = st.tabs(["🔎 Single Tweet", "📊 Dataset Analysis"])

# ----------------------------
# TAB 1 — Single Tweet
# ----------------------------
with tab1:

    st.subheader("Analyze a Tweet")

    user_input = st.text_area("Paste tweet:", height=120)

    if st.button("Analyze Tweet"):

        if user_input.strip() == "":
            st.warning("Please enter text")

        else:
            clean = clean_text(user_input)
            result, confidence = analyze_sentiment(clean)

            st.success("Result")
            st.markdown(f"### {result}")
            st.info(f"Confidence Score: {confidence:.1f}%")

# ----------------------------
# TAB 2 — Dataset
# ----------------------------
with tab2:

    st.subheader("Analyze Offline Dataset")

    @st.cache_data
    def load_data():
        df = pd.read_csv(
            "training.1600000.processed.noemoticon.csv",
            encoding="latin-1",
            header=None
        )

        df.columns = ["Target", "ID", "Date", "Flag", "User", "Tweet"]
        df = df.sample(5000)

        df["Clean"] = df["Tweet"].apply(clean_text)
        df["Sentiment"] = df["Clean"].apply(lambda x: analyze_sentiment(x)[0])

        return df

    if st.button("Run Dataset Analysis"):

        with st.spinner("Analyzing dataset..."):
            df = load_data()

        pos = len(df[df["Sentiment"] == "😊 Positive"])
        neg = len(df[df["Sentiment"] == "😡 Negative"])
        neu = len(df[df["Sentiment"] == "😐 Neutral"])

        # Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("😊 Positive", pos)
        c2.metric("😡 Negative", neg)
        c3.metric("😐 Neutral", neu)

        # Plotly Pie Chart
        st.subheader("Sentiment Distribution")

        fig = px.pie(
            names=["Positive", "Negative", "Neutral"],
            values=[pos, neg, neu],
            title="Sentiment Breakdown"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Table
        st.subheader("Sample Tweets")
        st.dataframe(df[["Tweet", "Sentiment"]].head(20),
                     use_container_width=True)

# Footer
st.markdown("---")
st.caption("Built by Nitesh 🚀 | AI Project | Streamlit Dashboard")
