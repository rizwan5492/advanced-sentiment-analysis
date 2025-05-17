import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import emoji
import re
import io
import json
import time
from streamlit.components.v1 import html

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Clean and preprocess text
def clean_text(text):
    text = emoji.demojize(text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.strip()

# Analyze sentiment for a single text
def analyze_sentiment(text):
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    sentiment = "Positive" if compound >= 0.05 else "Negative" if compound <= -0.05 else "Neutral"
    return {
        'Sentiment': sentiment,
        'Compound': round(compound, 3),
        'Positive': round(scores['pos'], 3),
        'Negative': round(scores['neg'], 3),
        'Neutral': round(scores['neu'], 3)
    }

# Analyze sentiment by sentence
def analyze_sentences(text):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return [analyze_sentiment(s) for s in sentences]

# Streamlit app configuration
st.set_page_config(page_title="Advanced Sentiment Analysis", layout="wide", initial_sidebar_state="collapsed")

# Custom Tailwind CSS via CDN
html("""
<link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
<style>
body {font-family: 'Inter', sans-serif;}
.stButton>button {background-color: #10B981; color: white; border-radius: 8px; padding: 0.5rem 1rem;}
.stTextArea textarea {border-radius: 8px; border: 1px solid #D1D5DB;}
</style>
""")

# Dark mode toggle
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

def toggle_dark_mode():
    st.session_state.dark_mode = not st.session_state.dark_mode
    update_theme()

def update_theme():
    theme = "dark" if st.session_state.dark_mode else "light"
    bg_color = "#1F2937" if theme == "dark" else "#F3F4F6"
    text_color = "#F9FAFB" if theme == "dark" else "#111827"
    st.markdown(f"""
    <style>
    .main {{background-color: {bg_color}; color: {text_color}; padding: 1rem;}}
    .tab {{background-color: {'#374151' if theme == 'dark' else '#FFFFFF'}; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;}}
    </style>
    """, unsafe_allow_html=True)

# Initial theme setup
update_theme()

# Title and dark mode toggle
theme = "dark" if st.session_state.dark_mode else "light"
bg_color = "#1F2937" if theme == "dark" else "#F3F4F6"
text_color = "#F9FAFB" if theme == "dark" else "#111827"
st.markdown(f"""
<div class="flex justify-between items-center mb-6">
    <h1 class="text-3xl font-bold {text_color}">Advanced Sentiment Analysis</h1>
    <button class="bg-gray-500 text-white px-4 py-2 rounded-lg" onclick="window.parent.parent.parent.postMessage({{type: 'DARK_MODE_TOGGLE'}}, '*')">
        {'Light Mode' if st.session_state.dark_mode else 'Dark Mode'}
    </button>
</div>
""", unsafe_allow_html=True)

# JavaScript to handle dark mode toggle
html("""
<script>
window.addEventListener('message', (event) => {
    if (event.data.type === 'DARK_MODE_TOGGLE') {
        parent.window.location.reload(); // Temporary reload to trigger state update
    }
});
</script>
""", height=0)

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

# Tabs for navigation
tab1, tab2, tab3 = st.tabs(["Analyze", "History", "Insights"])

with tab1:
    st.markdown('<div class="tab">', unsafe_allow_html=True)
    st.subheader("Analyze Text")
    
    # Text input with character count
    user_input = st.text_area("Enter text (supports emojis 😊👍):", height=150, key="text_input")
    char_count = len(user_input)
    st.caption(f"Characters: {char_count}/1000")
    
    # Real-time analysis (debounced)
    if user_input.strip() and char_count <= 1000:
        with st.spinner("Analyzing..."):
            time.sleep(0.5)  # Debounce
            result = analyze_sentiment(user_input)
            
            # Display results
            col1, col2 = st.columns([1, 1])
            with col1:
                st.metric("Sentiment", result['Sentiment'])
                st.metric("Compound Score", result['Compound'])
            with col2:
                st.metric("Positive", result['Positive'])
                st.metric("Negative", result['Negative'])
                st.metric("Neutral", result['Neutral'])
            
            # Sentence-level analysis
            st.subheader("Sentence Breakdown")
            sentence_results = analyze_sentences(user_input)
            if sentence_results:
                sentence_df = pd.DataFrame(sentence_results)
                st.dataframe(sentence_df, use_container_width=True)
            
            # Add to history
            st.session_state.history.append({
                'Text': user_input[:50] + "..." if len(user_input) > 50 else user_input,
                'Sentiment': result['Sentiment'],
                'Compound': result['Compound'],
                'Timestamp': pd.Timestamp.now()
            })
    elif char_count > 1000:
        st.error("Input exceeds 1000 characters. Please shorten your text.")
    elif not user_input.strip():
        st.info("Enter text to analyze.")
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="tab">', unsafe_allow_html=True)
    st.subheader("Analysis History")
    
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)
        
        # Download options
        col1, col2 = st.columns(2)
        with col1:
            csv = history_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="sentiment_history.csv",
                mime="text/csv"
            )
        with col2:
            json_data = history_df.to_json(orient="records", date_format="iso")
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name="sentiment_history.json",
                mime="application/json"
            )
    else:
        st.info("No analysis history yet. Analyze some text to see results here.")
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="tab">', unsafe_allow_html=True)
    st.subheader("Sentiment Trends")
    
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        
        # Interactive trend chart
        fig = px.line(
            history_df,
            x=history_df.index,
            y="Compound",
            markers=True,
            title="Sentiment Trend Over Time",
            labels={"index": "Analysis Number", "Compound": "Compound Score"}
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color=text_color
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Sentiment distribution
        sentiment_counts = history_df['Sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        fig2 = px.pie(
            sentiment_counts,
            names='Sentiment',
            values='Count',
            title="Sentiment Distribution",
            color_discrete_sequence=['#10B981', '#EF4444', '#F59E0B']
        )
        fig2.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color=text_color
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No data for trends yet. Analyze some text to see insights.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown(f"""
<div class="text-center mt-6 text-{text_color}">
    Built with Streamlit & VADER | © 2025 Advanced Sentiment Analysis
</div>
""", unsafe_allow_html=True)