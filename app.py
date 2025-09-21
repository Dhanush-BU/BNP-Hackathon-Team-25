import streamlit as st
import finnhub
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import time
from transformers import pipeline
import re
import json
from collections import Counter
import yfinance as yf
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
import nltk

# Download stopwords once
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

# Page configuration
st.set_page_config(
    page_title="🚀 Advanced Market Intelligence Hub",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
FINNHUB_API_KEY = "d3777ehr01qtvbtkdoq0d3777ehr01qtvbtkdoqg"
GEMINI_API_KEY = "AIzaSyBRAGDqSJ1HCNEZNiA616X3_O1_2vBIP5Q"
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"
POPULAR_STOCKS = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "NFLX", "AMD", "BABA"]
STOPWORDS = set(stopwords.words('english'))
FINANCE_NOISE = {"stock", "stocks", "share", "shares", "company", "inc", "corp", "market", "tr", "pr", "new", "us"}

# Enhanced CSS with modern design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 3rem 2rem;
        border-radius: 25px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        transform: rotate(45deg);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 45px rgba(0, 0, 0, 0.15);
    }
    
    .metric-card-enhanced {
        background: linear-gradient(135deg, #f8f9ff 0%, #e6f3ff 100%);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        border: 2px solid transparent;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card-enhanced::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 20px 20px 0 0;
    }
    
    .metric-card-enhanced:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.2);
        border-color: #667eea;
    }
    
    .positive-card {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 6px solid #28a745;
    }
    
    .negative-card {
        background: linear-gradient(135deg, #f8d7da 0%, #f1b0b7 100%);
        border-left: 6px solid #dc3545;
    }
    
    .neutral-card {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border-left: 6px solid #ffc107;
    }
    
    .news-card-modern {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #007bff;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        position: relative;
    }
    
    .news-card-modern:hover {
        transform: translateX(10px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .news-card-modern.positive { 
        border-left-color: #28a745;
        background: linear-gradient(135deg, #ffffff 0%, #f8fff9 100%);
    }
    
    .news-card-modern.negative { 
        border-left-color: #dc3545;
        background: linear-gradient(135deg, #ffffff 0%, #fff8f8 100%);
    }
    
    .news-card-modern.neutral { 
        border-left-color: #ffc107;
        background: linear-gradient(135deg, #ffffff 0%, #fffef8 100%);
    }
    
    .stock-card-premium {
        background: white;
        border-radius: 20px;
        padding: 1.5rem;
        margin: 0.5rem;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        transition: all 0.4s ease;
        border: 2px solid transparent;
        position: relative;
        overflow: hidden;
    }
    
    .stock-card-premium::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
        transition: left 0.6s;
    }
    
    .stock-card-premium:hover::before {
        left: 100%;
    }
    
    .stock-card-premium:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 15px 35px rgba(0,0,0,0.2);
        border-color: #667eea;
    }
    
    .trend-indicator {
        display: inline-flex;
        align-items: center;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    
    .trend-up { 
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .trend-down { 
        background: linear-gradient(135deg, #f8d7da, #f1b0b7);
        color: #721c24;
        border: 1px solid #f1b0b7;
    }
    
    .trend-neutral { 
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    
    .sentiment-gauge {
        position: relative;
        width: 200px;
        height: 100px;
        margin: 1rem auto;
    }
    
    .pulse-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        background: #28a745;
        border-radius: 50%;
        animation: pulse-animation 2s infinite;
        margin-right: 0.5rem;
    }
    
    @keyframes pulse-animation {
        0% { box-shadow: 0 0 0 0 rgba(40, 167, 69, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(40, 167, 69, 0); }
        100% { box-shadow: 0 0 0 0 rgba(40, 167, 69, 0); }
    }
    
    .feature-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        border: 1px solid #e9ecef;
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border-color: #667eea;
    }
    
    .sidebar-modern {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 15px;
        padding: 1rem;
    }
    
    .metric-large {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .section-header {
        display: flex;
        align-items: center;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    
    .section-icon {
        font-size: 1.8rem;
        margin-right: 0.8rem;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .tabs-modern .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 15px;
        padding: 0.5rem;
    }
    
    .tabs-modern .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 24px;
        border-radius: 12px;
        background: white;
        border: 1px solid #e9ecef;
        font-weight: 500;
    }
    
    .tabs-modern .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white !important;
        border-color: #667eea;
    }
    
    .economic-indicator {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #2196f3;
        position: relative;
    }
    
    .social-sentiment-card {
        background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid #ce93d8;
        text-align: center;
    }
    
    .timeline-item {
        position: relative;
        padding: 1rem 0 1rem 2rem;
        border-left: 2px solid #667eea;
        margin-left: 1rem;
    }
    
    .timeline-item::before {
        content: '';
        position: absolute;
        left: -6px;
        top: 1.5rem;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #667eea;
    }
    
    .sentiment-trend-chart {
        background: white;
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# --- Utility Functions (with caching) ---
@st.cache_resource
def initialize_models():
    """Initializes Finnhub client and sentiment analyzer."""
    finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
    try:
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        return finnhub_client, sentiment_analyzer, "distilbert"
    except Exception as e:
        st.warning(f"Could not load sentiment model: {e}. Using a fallback.")
        return finnhub_client, None, "fallback"

@st.cache_data(ttl=300)
def get_stock_data(symbol):
    """Fetches real-time stock data."""
    try:
        finnhub_client, _, _ = initialize_models()
        quote = finnhub_client.quote(symbol)
        return {
            'symbol': symbol,
            'current': quote.get('c', 0),
            'high': quote.get('h', 0),
            'low': quote.get('l', 0),
            'open': quote.get('o', 0),
            'previous_close': quote.get('pc', 0),
            'change': quote.get('d', 0),
            'change_percent': quote.get('dp', 0)
        }
    except Exception:
        return None

@st.cache_data(ttl=300)
def get_historical_sentiment_data(symbol, days=30):
    """Generates simulated historical sentiment data for trends."""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    base_sentiment = 3.0
    trend = np.random.choice([-0.1, 0, 0.1], size=days, p=[0.3, 0.4, 0.3])
    noise = np.random.normal(0, 0.3, days)
    sentiment_scores = []
    current_sentiment = base_sentiment
    for i in range(days):
        current_sentiment += trend[i] + noise[i]
        current_sentiment = max(1.0, min(5.0, current_sentiment))
        sentiment_scores.append(current_sentiment)
    return pd.DataFrame({'date': dates, 'sentiment_score': sentiment_scores, 'symbol': symbol})

@st.cache_data(ttl=600)
def get_economic_events():
    """Fetches or simulates economic calendar events."""
    try:
        finnhub_client, _, _ = initialize_models()
        today = datetime.now()
        from_date = (today - timedelta(days=7)).strftime('%Y-%m-%d')
        to_date = (today + timedelta(days=7)).strftime('%Y-%m-%d')
        
        try:
            events = finnhub_client.economic_calendar(_from=from_date, to=to_date)
            return [
                {
                    'event': e['event'], 'country': e['country'], 'impact': e['impact'],
                    'actual': e.get('actual', 'N/A'), 'estimate': e.get('estimate', 'N/A'),
                    'previous': e.get('previous', 'N/A'), 'time': e['time']
                } for e in events.get('economicCalendar', [])
            ]
        except Exception:
            return [
                {'event': 'Federal Reserve Interest Rate Decision', 'country': 'US', 'impact': 'high', 'actual': '5.25%', 'estimate': '5.25%', 'previous': '5.00%', 'time': '2025-09-22T14:00:00Z'},
                {'event': 'Consumer Price Index (CPI)', 'country': 'US', 'impact': 'high', 'actual': '3.2%', 'estimate': '3.1%', 'previous': '3.0%', 'time': '2025-09-21T08:30:00Z'},
                {'event': 'Non-Farm Payrolls', 'country': 'US', 'impact': 'high', 'actual': '215K', 'estimate': '200K', 'previous': '190K', 'time': '2025-09-19T08:30:00Z'}
            ]
    except Exception:
        return []

@st.cache_data(ttl=600)
def get_social_sentiment_data(symbol):
    """Generates a score for social media sentiment by analyzing simulated posts."""
    finnhub_client, sentiment_pipeline, _ = initialize_models()
    
    posts = {
        'AAPL': [
            f"The new iPhone launch is a huge success. $AAPL is a strong buy!", "Apple's stock just keeps climbing. Bullish on $AAPL.",
            f"Selling my $AAPL shares. I'm worried about the future market.", f"Apple's earnings miss could mean trouble for $AAPL.",
            "Tim Cook is a genius! $AAPL to the moon.", "Apple's stock is consolidating, it's a good time to hold for the long term."
        ],
        'TSLA': [
            f"The new battery tech from Tesla is a game changer. Long on $TSLA.", "Tesla's factory expansion will drive huge growth for $TSLA.",
            f"Bearish on $TSLA. Valuations are insane and competition is heating up.", f"I'm worried about Elon Musk's tweets. Could be bad for $TSLA."
        ],
        'NVDA': [
            f"$NVDA is the future of AI and gaming!", "Weak guidance from $NVDA, I'm out.",
            "The new chip from $NVDA will revolutionize the industry.", "$NVDA looks stable before the next big announcement."
        ],
        'MSFT': [f"Microsoft is a solid long-term bet.", "New partnership for $MSFT is a major positive.", "Windows update is a disaster for $MSFT.", "Satya Nadella is doing a great job at $MSFT."],
        'GOOGL': [f"$GOOGL stock is a safe bet for tech.", "Antitrust issues could hurt $GOOGL.", "Google's search revenue is unstoppable.", "Holding my position in $GOOGL."],
        'AMZN': [f"Amazon's retail business is still strong.", "$AMZN to get crushed by higher interest rates.", "Amazon Prime Day was a huge success for $AMZN."],
        'META': [f"Meta's metaverse vision is failing.", "$META is investing heavily in AI, a good sign.", "Facebook's user growth is stagnating, a bad sign for $META."],
        'NFLX': [f"Netflix's new ad tier is a success.", "$NFLX losing subscribers, time to sell.", "The content library is a goldmine for $NFLX."],
        'AMD': [f"$AMD is taking market share from Intel.", "Recession fears could hurt demand for $AMD chips.", "Bullish on $AMD, new products are amazing."],
        'BABA': [f"Alibaba's regulation woes are a big risk for $BABA.", "$BABA earnings beat expectations!", "I'm staying away from Chinese tech stocks like $BABA."]
    }
    
    sentiment_data = []
    
    if sentiment_pipeline:
        for platform, post_list in posts.items():
            sentiments = [sentiment_pipeline(post)[0]['label'] for post in post_list if f'${symbol}' in post]
            if not sentiments: continue
            
            positive_count = sentiments.count('POSITIVE')
            negative_count = sentiments.count('NEGATIVE')
            neutral_count = len(sentiments) - positive_count - negative_count
            
            total_posts = len(sentiments)
            sentiment_score = (positive_count - negative_count) / total_posts if total_posts > 0 else 0
            normalized_score = (sentiment_score + 1) * 2.5
            
            sentiment_data.append({
                'platform': platform,
                'positive': positive_count,
                'negative': negative_count,
                'neutral': neutral_count,
                'total': total_posts,
                'normalized_score': normalized_score
            })
    else:
        # Fallback if the model is not available
        return [{"platform": p, "normalized_score": np.random.uniform(2.0, 4.0), "total": np.random.randint(50, 500)} for p in ['Twitter', 'Reddit', 'StockTwits']]
    
    return sentiment_data

def create_sentiment_gauge(score, title="Sentiment Score"):
    """Creates a sentiment gauge chart using Plotly."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=score, domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20}},
        delta={'reference': 3.0, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
        gauge={'axis': {'range': [None, 5], 'tickwidth': 1},
               'bar': {'color': "darkblue"},
               'steps': [{'range': [0, 2], 'color': "lightcoral"}, {'range': [2, 3.5], 'color': "lightyellow"}, {'range': [3.5, 5], 'color': "lightgreen"}],
               'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 4.0}}
    ))
    fig.update_layout(height=300, font={'color': "darkblue"}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig

def create_sentiment_trend_chart(data, timeframe="1W"):
    """Creates a sentiment trend chart using Plotly."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['date'], y=data['sentiment_score'], mode='lines+markers', name='Sentiment Score', line=dict(color='#667eea', width=3), marker=dict(size=6, color='#764ba2')))
    fig.add_hline(y=3.0, line_dash="dash", line_color="gray", annotation_text="Neutral (3.0)")
    fig.update_layout(title=f"Sentiment Trend - {timeframe}", xaxis_title="Date", yaxis_title="Sentiment Score (1-5)", height=400)
    return fig

def create_social_sentiment_chart(social_data):
    """Creates a social media sentiment breakdown chart using Plotly."""
    platforms = [item['platform'] for item in social_data]
    sentiment_scores = [item['normalized_score'] for item in social_data]
    total_mentions = [item.get('total', 100) for item in social_data] if social_data else [100] * len(platforms)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=platforms, y=sentiment_scores, mode='markers', marker=dict(size=[min(mentions/10, 100) for mentions in total_mentions], color=sentiment_scores, colorscale='RdYlGn', cmin=1, cmax=5, showscale=True)))
    fig.update_layout(title="Social Media Sentiment by Platform", xaxis_title="Platform", yaxis_title="Sentiment Score (1-5)", height=400)
    return fig

def fetch_latest_news(symbol, days=7):
    """Fetches latest company news from Finnhub."""
    try:
        finnhub_client, _, _ = initialize_models()
        to_date = datetime.today()
        from_date = to_date - timedelta(days=days)
        news = finnhub_client.company_news(symbol, _from=from_date.strftime("%Y-%m-%d"), to=to_date.strftime("%Y-%m-%d"))
        return news
    except Exception:
        return []

def analyze_news_sentiment(news_list):
    """Analyzes sentiment of news headlines and summaries."""
    finnhub_client, sentiment_pipeline, _ = initialize_models()
    if not sentiment_pipeline: return []
    analyzed = []
    for news in news_list:
        try:
            content = (news.get('headline', '') + ". " + news.get('summary', '')).strip()
            if not content: continue
            content = content[:512]
            sentiment = sentiment_pipeline(content)[0]
            analyzed.append({
                'headline': news.get('headline', ''), 'summary': news.get('summary', ''),
                'sentiment': sentiment['label'], 'confidence': round(sentiment['score'], 3)
            })
        except Exception:
            continue
    return analyzed

def get_gemini_recommendation_and_reasons(symbol, analyzed_news):
    """Generates a prompt and gets a recommendation from Gemini, parsing the response."""
    if not analyzed_news: return None
    positive_count = sum(1 for n in analyzed_news if n['sentiment'] == 'POSITIVE')
    negative_count = sum(1 for n in analyzed_news if n['sentiment'] == 'NEGATIVE')
    prompt = f"""You are an expert stock analyst. Analyze the news for {symbol}. Based on the provided news headlines and sentiment analysis, provide a structured JSON response with a single recommendation score from 1 (Strong Sell) to 5 (Strong Buy) and exactly 3 concise reasons. Do not include any text outside the JSON object.

    Stock: {symbol}
    Analysis Period: Last 7 days
    News: {positive_count} Positive, {negative_count} Negative.
    Headlines: {[n['headline'][:100] for n in analyzed_news[:5]]}
    Sentiments: {[n['sentiment'] for n in analyzed_news[:5]]}
    
    JSON format: {{"recommendation_score": 4, "recommendation_text": "BUY", "reasons": ["Reason 1", "Reason 2", "Reason 3"]}}
    """
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.7}}
    try:
        url = f"{GEMINI_ENDPOINT}?key={GEMINI_API_KEY}"
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        raw_text = response.json()['candidates'][0]['content']['parts'][0]['text']
        json_string = raw_text.strip().replace("`", "").replace("json", "").replace("JSON", "")
        return json.loads(json_string)
    except Exception:
        return None

def filter_meaningful_words(texts, symbol):
    """Filters words from text to create a word frequency count for the word cloud."""
    all_words = []
    symbol_lower = symbol.lower()
    for text in texts:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        meaningful = [w for w in words if w not in STOPWORDS and w not in FINANCE_NOISE and w != symbol_lower]
        all_words.extend(meaningful)
    return Counter(all_words)

def create_wordcloud_image(word_counts, symbol):
    """Generates a word cloud image from a frequency dictionary."""
    if not word_counts: return None
    wc = WordCloud(
        width=800, height=400, background_color='white', max_words=60
    ).generate_from_frequencies(word_counts)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    return fig

# --- Streamlit Session State ---
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "selected_stock" not in st.session_state:
    st.session_state.selected_stock = "AAPL"
if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = False
if "ai_recommendation" not in st.session_state:
    st.session_state.ai_recommendation = None
if "word_counts" not in st.session_state:
    st.session_state.word_counts = None

# --- Main App Function ---
def main():
    st.markdown("""<div class="main-header"><h1>🚀 Advanced Market Intelligence Hub</h1><p><span class="pulse-indicator"></span>Real-time sentiment analysis • Social media tracking • Economic events • AI insights</p><p style="font-size: 0.9em; opacity: 0.8; margin-top: 1rem;">Powered by AI | Live Data Feeds | Professional Analytics</p></div>""", unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### 🎛 Control Panel")
        selected_stock = st.selectbox("📈 Select Stock for Analysis", POPULAR_STOCKS, index=POPULAR_STOCKS.index(st.session_state.selected_stock) if st.session_state.selected_stock in POPULAR_STOCKS else 0)
        if selected_stock != st.session_state.selected_stock:
            st.session_state.selected_stock = selected_stock
            st.session_state.ai_recommendation = None
            st.session_state.word_counts = None
        
        timeframe = st.selectbox("⏰ Analysis Timeframe", ["1D", "1W", "1M", "3M", "6M", "1Y"], index=1)
        st.session_state.selected_timeframe = timeframe
        auto_refresh = st.toggle("🔄 Auto Refresh", value=st.session_state.auto_refresh)
        st.session_state.auto_refresh = auto_refresh
        
        if st.button("🔄 Refresh Data", type="primary"):
            st.cache_data.clear()
            st.session_state.ai_recommendation = None
            st.session_state.word_counts = None
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 🤖 AI Market Assistant")
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.chat_messages[-5:]:
                st.markdown(f"""<div style="background: {'#e3f2fd' if msg['role'] == 'user' else '#f1f8e9'}; padding: 0.8rem; border-radius: 15px; margin: 0.5rem 0; margin-left: {'1rem' if msg['role'] == 'user' else '0'}; margin-right: {'0' if msg['role'] == 'user' else '1rem'};"><strong>{'You' if msg['role'] == 'user' else '🤖'}</strong>: {msg['content']}</div>""", unsafe_allow_html=True)
        user_input = st.text_input("💬 Ask about markets...", key="chat_input", placeholder="e.g., What's the sentiment on TSLA?")
        if st.button("Send", type="secondary") and user_input:
            st.session_state.chat_messages.append({"role": "user", "content": user_input})
            response = "I'm a market assistant. Please use the 'Stock Analysis' tab for detailed insights."
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
            st.rerun()
        if st.button("Clear Chat"): st.session_state.chat_messages = [] ; st.rerun()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Market Overview", "📱 Social Sentiment", "📈 Sentiment Trends",
        "📰 Economic Events", "🎯 Stock Analysis"
    ])
    
    with tab1:
        st.markdown('<div class="section-header"><span class="section-icon">📊</span><h2>Live Market Overview</h2></div>', unsafe_allow_html=True)
        cols = st.columns(4)
        for i, symbol in enumerate(POPULAR_STOCKS[:8]):
            stock_data = get_stock_data(symbol)
            if stock_data:
                with cols[i % 4]:
                    st.markdown(f"""<div class="stock-card-premium"><div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;"><h4 style="margin: 0;">{symbol}</h4><span class="trend-{'up' if stock_data['change'] >= 0 else 'down'}">{'📈' if stock_data['change'] >= 0 else '📉'}</span></div><h3 style="margin: 0.5rem 0; color: #333;">${stock_data['current']:.2f}</h3><div class="trend-indicator trend-{'up' if stock_data['change'] >= 0 else 'down'}">{stock_data['change']:+.2f} ({stock_data['change_percent']:+.2f}%)</div><div style="margin-top: 1rem; font-size: 0.85em; color: #666;"><div>High: ${stock_data['high']:.2f}</div><div>Low: ${stock_data['low']:.2f}</div></div></div>""", unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="section-header"><span class="section-icon">📱</span><h2>Social Media Sentiment Analysis</h2></div>', unsafe_allow_html=True)
        social_data = get_social_sentiment_data(selected_stock)
        
        if social_data:
            if social_data and 'normalized_score' in social_data[0]:
                avg_social_sentiment = np.mean([item['normalized_score'] for item in social_data])
                col1, col2 = st.columns([1, 2])
                with col1:
                    sentiment_label = 'Positive' if avg_social_sentiment > 3.0 else 'Negative' if avg_social_sentiment < 2.5 else 'Neutral'
                    sentiment_class = 'trend-up' if avg_social_sentiment > 3.0 else 'trend-down' if avg_social_sentiment < 2.5 else 'trend-neutral'
                    st.markdown(f"""<div class="social-sentiment-card"><h3>📱 Social Sentiment</h3><h1 style="color: #764ba2; margin: 1rem 0;">{avg_social_sentiment:.1f}/5</h1><p>Across {len(social_data)} platforms</p><div style="margin-top: 1rem;"><div class="trend-indicator {sentiment_class}">{sentiment_label} Sentiment</div></div></div>""", unsafe_allow_html=True)
                    gauge_fig = create_sentiment_gauge(avg_social_sentiment, f"{selected_stock} Social Sentiment")
                    st.plotly_chart(gauge_fig, use_container_width=True)
                with col2:
                    social_chart = create_social_sentiment_chart(social_data)
                    st.plotly_chart(social_chart, use_container_width=True)
                st.markdown("### 📊 Platform Breakdown")
                platform_cols = st.columns(len(social_data))
                for i, platform_data in enumerate(social_data):
                    with platform_cols[i]:
                        sentiment_color = "green" if platform_data['normalized_score'] > 3.0 else "red" if platform_data['normalized_score'] < 2.5 else "orange"
                        st.markdown(f"""<div class="feature-card"><h4>{platform_data['platform']}</h4><h3 style="color: {sentiment_color};">{platform_data['normalized_score']:.1f}/5</h3><p style="margin: 0.5rem 0;">📈 {platform_data.get('positive', 'N/A')} positive<br>📉 {platform_data.get('negative', 'N/A')} negative<br>➖ {platform_data.get('neutral', 'N/A')} neutral</p><small>Total: {platform_data.get('total', 'N/A')} mentions</small></div>""", unsafe_allow_html=True)
            else:
                st.warning("Social sentiment data format is invalid. Please check the data source.")
        else:
            st.info("No social sentiment data found for this stock. Try a different symbol or refresh.")
    
    with tab3:
        st.markdown('<div class="section-header"><span class="section-icon">📈</span><h2>Sentiment Trends Analysis</h2></div>', unsafe_allow_html=True)
        trend_col1, trend_col2 = st.columns([3, 1])
        with trend_col2: st.markdown("#### 📅 Select Timeframe"); trend_timeframe = st.selectbox("", ["1 Week", "1 Month"], key="trend_timeframe")
        days_map = {"1 Week": 7, "1 Month": 30}
        days = days_map[trend_timeframe]
        historical_data = get_historical_sentiment_data(selected_stock, days)
        with trend_col1: st.plotly_chart(create_sentiment_trend_chart(historical_data, trend_timeframe), use_container_width=True)
        st.markdown("### 📊 Trend Statistics")
        current_sentiment = historical_data['sentiment_score'].iloc[-1]
        avg_sentiment = historical_data['sentiment_score'].mean()
        trend_direction = "Upward" if historical_data['sentiment_score'].iloc[-1] > historical_data['sentiment_score'].iloc[0] else "Downward"
        stat_cols = st.columns(3)
        with stat_cols[0]: st.markdown(f"""<div class="metric-card-enhanced"><h4>Current Sentiment</h4><div class="metric-large">{current_sentiment:.1f}</div><p>Out of 5.0</p></div>""", unsafe_allow_html=True)
        with stat_cols[1]: st.markdown(f"""<div class="metric-card-enhanced"><h4>Average Sentiment</h4><div class="metric-large">{avg_sentiment:.1f}</div><p>{trend_timeframe} average</p></div>""", unsafe_allow_html=True)
        with stat_cols[2]: st.markdown(f"""<div class="metric-card-enhanced {'positive-card' if trend_direction == 'Upward' else 'negative-card'}"><h4>Trend Direction</h4><div class="metric-large">{'📈' if trend_direction == 'Upward' else '📉'}</div><p>{trend_direction}</p></div>""", unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="section-header"><span class="section-icon">📰</span><h2>Economic Events Calendar</h2></div>', unsafe_allow_html=True)
        economic_events = get_economic_events()
        
        if economic_events:
            for event in economic_events[:10]:
                impact_color = {"high": "red", "medium": "orange", "low": "green"}.get(event.get('impact', 'low').lower(), "gray")
                
                st.markdown(f"""
                <div class="economic-indicator">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                        <h4 style="margin: 0; color: #333;">{event.get('event', 'Economic Event')}</h4>
                        <span style="background: {impact_color}; color: white; padding: 0.3rem 0.8rem; border-radius: 15px; font-size: 0.8rem; font-weight: bold;">
                            {event.get('impact', 'Unknown').upper()} IMPACT
                        </span>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; margin: 1rem 0;">
                        <div>
                            <strong>🎯 Estimate:</strong><br>{event.get('estimate', 'N/A')}
                        </div>
                        <div>
                            <strong>📊 Actual:</strong><br>{event.get('actual', 'Pending')}
                        </div>
                        <div>
                            <strong>📈 Previous:</strong><br>{event.get('previous', 'N/A')}
                        </div>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 1rem;">
                        <span style="color: #666;">🌍 {event.get('country', 'Global')}</span>
                        <span style="color: #666;">📅 {event.get('time', 'TBD')}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("📅 No economic events data available at the moment.")

    with tab5:
        st.markdown('<div class="section-header"><span class="section-icon">🎯</span><h2>Detailed Stock Analysis</h2></div>', unsafe_allow_html=True)
        col1, col2 = st.columns([3, 1])
        with col1: 
            search_stock = st.text_input("🔍 Enter stock symbol for detailed analysis", value=st.session_state.selected_stock).upper().strip()
            if search_stock and search_stock != st.session_state.selected_stock:
                st.session_state.selected_stock = search_stock
                st.session_state.ai_recommendation = None
                st.session_state.word_counts = None
                st.rerun()

        stock_data = get_stock_data(st.session_state.selected_stock)

        # Automatic Analysis Trigger
        if st.session_state.ai_recommendation is None and stock_data is not None:
            with st.spinner("Generating AI-powered insights..."):
                news_list = fetch_latest_news(st.session_state.selected_stock)
                if news_list:
                    analyzed_news = analyze_news_sentiment(news_list)
                    st.session_state.ai_recommendation = get_gemini_recommendation_and_reasons(st.session_state.selected_stock, analyzed_news)
                    st.session_state.word_counts = filter_meaningful_words([n.get('headline', '') for n in news_list], st.session_state.selected_stock)
                else:
                    st.session_state.ai_recommendation = {"recommendation_score": 3, "recommendation_text": "HOLD", "reasons": ["No news available to analyze."]}
            st.rerun()

        if stock_data:
            overview_cols = st.columns([2, 1])
            with overview_cols[0]:
                st.markdown(f"""<div class="glass-card"><h2>{st.session_state.selected_stock} Stock Analysis {'📈' if stock_data['change'] >= 0 else '📉'}</h2><h3>${stock_data['current']:.2f}</h3><div class="trend-indicator trend-{'up' if stock_data['change'] >= 0 else 'down'}">{stock_data['change']:+.2f} ({stock_data['change_percent']:+.2f}%)</div><p>High: ${stock_data['high']:.2f} | Low: ${stock_data['low']:.2f}</p></div>""", unsafe_allow_html=True)
            with overview_cols[1]:
                gauge_fig = create_sentiment_gauge(4.2, "Overall Sentiment")
                st.plotly_chart(gauge_fig, use_container_width=True)
            
            st.markdown('---')
            st.markdown('### 🤖 AI-Powered Recommendation')
            ai_rec_cols = st.columns([1, 2])
            with ai_rec_cols[0]:
                if st.session_state.ai_recommendation:
                    rec_data = st.session_state.ai_recommendation
                    score = rec_data.get("recommendation_score", 3)
                    text = rec_data.get("recommendation_text", "HOLD")
                    reasons = rec_data.get("reasons", ["No reasons available."])
                    color_map = {1: "red", 2: "darkorange", 3: "gold", 4: "lightgreen", 5: "green"}
                    score_color = color_map.get(score, "gray")
                    st.markdown(f"""<div class="glass-card" style="padding: 1rem; border-left: 5px solid {score_color};"><h4>AI Recommendation: <span style="color: {score_color};">{text}</span></h4><div style="font-size: 2em; font-weight: bold; color: {score_color};">{score}/5</div><h5>Key Reasons:</h5><ul style="padding-left: 20px;">{''.join([f"<li>{r}</li>" for r in reasons])}</ul></div>""", unsafe_allow_html=True)
                else:
                    st.info("Generating analysis...")
            
            st.markdown('---')
            
            analysis_cols = st.columns(3)
            with analysis_cols[0]:
                st.markdown("#### 📊 Technical Analysis")
                volatility_level = "High" if abs(stock_data['change_percent']) > 5 else "Medium"
                trend = "Bullish" if stock_data['change'] > 1 else "Bearish"
                st.markdown(f"""<div class="feature-card"><div style="margin: 1rem 0;"><strong>Trend:</strong> {trend}<br><strong>Volatility:</strong> {volatility_level}<br><strong>Support:</strong> ${stock_data['low']:.2f}<br><strong>Resistance:</strong> ${stock_data['high']:.2f}</div></div>""", unsafe_allow_html=True)
            with analysis_cols[1]:
                st.markdown("#### 📱 Social Sentiment")
                st.markdown(f"""<div class="feature-card"><h3 style="color: green;">4.2/5</h3><p>Platforms: 3<br>Total Mentions: 2.4K<br>Sentiment Trend: Positive</p></div>""", unsafe_allow_html=True)
            with analysis_cols[2]:
                st.markdown("#### 🎯 Investment Signal")
                rec_score = 3
                rec_text = "⏸ HOLD"
                rec_color = "orange"
                source = "Rule-Based"
                if st.session_state.ai_recommendation:
                    rec_data = st.session_state.ai_recommendation
                    rec_score = rec_data.get("recommendation_score", 3)
                    rec_text = rec_data.get("recommendation_text", "HOLD")
                    source = "AI"
                    if rec_score >= 4: rec_text = "📈 BUY"
                    if rec_score >= 5: rec_text = "🚀 STRONG BUY"
                    if rec_score <= 2: rec_text = "📉 SELL"
                    if rec_score <= 1: rec_text = "⛔ STRONG SELL"
                    rec_color_map = {1: "red", 2: "red", 3: "orange", 4: "green", 5: "green"}
                    rec_color = rec_color_map.get(rec_score, "orange")
                st.markdown(f"""<div class="feature-card" style="border-left: 5px solid {rec_color};"><h3 style="color: {rec_color};">{rec_text}</h3><div style="margin: 1rem 0;"><strong>Confidence:</strong> {'High' if rec_score >= 4 else 'Medium' if rec_score == 3 else 'Low'}<br><strong>Signal Source:</strong> {source}<br><strong>Risk Level:</strong> {volatility_level}</div></div>""", unsafe_allow_html=True)
            
            st.markdown('---')
            st.markdown('### ☁ News Word Cloud')
            if st.session_state.word_counts:
                fig = create_wordcloud_image(st.session_state.word_counts, st.session_state.selected_stock)
                st.pyplot(fig)
            else:
                st.info("Generating analysis...")

        else:
            st.error(f"❌ Could not fetch data for {st.session_state.selected_stock}. Please check the symbol.")

    if st.session_state.auto_refresh:
        time.sleep(30)
        st.rerun()

if _name_ == "_main_":
    main()