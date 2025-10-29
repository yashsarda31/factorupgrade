# app.py - Cloud-Optimized Version
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import google.generativeai as genai
from typing import Dict, List, Tuple
import json
import time
import warnings
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

warnings.filterwarnings('ignore')

# Check if matplotlib is available for styling
try:
    import matplotlib
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Page configuration
st.set_page_config(
    page_title="Indian Equity Research Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'stock_data_cache' not in st.session_state:
    st.session_state.stock_data_cache = {}
if 'screening_results' not in st.session_state:
    st.session_state.screening_results = None

class IndianStockAnalyzer:
    """Main class for Indian stock analysis - Optimized for Cloud"""
    
    # Limited stock lists for cloud deployment
    NIFTY_50_STOCKS = [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
        'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
        'LT.NS', 'HCLTECH.NS', 'AXISBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS',
        'SUNPHARMA.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'BAJFINANCE.NS', 'WIPRO.NS',
        'NESTLEIND.NS', 'ADANIENT.NS', 'POWERGRID.NS', 'M&M.NS', 'NTPC.NS',
        'TATAMOTORS.NS', 'ONGC.NS', 'JSWSTEEL.NS', 'TATASTEEL.NS', 'HDFC.NS',
        'BAJAJFINSV.NS', 'COALINDIA.NS', 'GRASIM.NS', 'TECHM.NS', 'INDUSINDBK.NS',
        'HINDALCO.NS', 'DRREDDY.NS', 'DIVISLAB.NS', 'CIPLA.NS', 'SBILIFE.NS',
        'BRITANNIA.NS', 'EICHERMOT.NS', 'BPCL.NS', 'SHREECEM.NS', 'HEROMOTOCO.NS',
        'UPL.NS', 'TATACONSUM.NS', 'APOLLOHOSP.NS', 'ADANIPORTS.NS', 'BAJAJ-AUTO.NS'
    ]
    
    # Smaller sector lists for cloud
    BANKING_STOCKS = [
        'HDFCBANK.NS', 'ICICIBANK.NS', 'AXISBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS',
        'INDUSINDBK.NS', 'BANDHANBNK.NS', 'BANKBARODA.NS', 'PNB.NS', 'FEDERALBNK.NS'
    ]
    
    IT_STOCKS = [
        'TCS.NS', 'INFY.NS', 'HCLTECH.NS', 'WIPRO.NS', 'TECHM.NS',
        'LTI.NS', 'MINDTREE.NS', 'MPHASIS.NS', 'COFORGE.NS', 'PERSISTENT.NS'
    ]
    
    PHARMA_STOCKS = [
        'SUNPHARMA.NS', 'DRREDDY.NS', 'DIVISLAB.NS', 'CIPLA.NS', 'AUROPHARMA.NS',
        'LUPIN.NS', 'TORNTPHARM.NS', 'ALKEM.NS', 'BIOCON.NS', 'GLENMARK.NS'
    ]
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.request_count = 0
        self.last_request_time = time.time()
        if api_key:
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-2.5-flash')
            except Exception as e:
                st.error(f"Error configuring Gemini API: {str(e)}")
    
    def rate_limit(self):
        """Implement rate limiting for API calls"""
        self.request_count += 1
        current_time = time.time()
        
        # Reset counter every minute
        if current_time - self.last_request_time > 60:
            self.request_count = 0
            self.last_request_time = current_time
        
        # If too many requests, wait
        if self.request_count > 30:  # Max 30 requests per minute
            time.sleep(2)
            self.request_count = 0
            self.last_request_time = time.time()
        else:
            time.sleep(0.2)  # Small delay between requests
    
    def get_stock_universe(self, universe_type: str, limit: int = None) -> List[str]:
        """Get stock list based on universe type with optional limit"""
        stocks = []
        
        if universe_type == "NIFTY 50":
            stocks = self.NIFTY_50_STOCKS
        elif universe_type == "NIFTY Top 20":
            stocks = self.NIFTY_50_STOCKS[:20]
        elif universe_type == "NIFTY Top 10":
            stocks = self.NIFTY_50_STOCKS[:10]
        elif universe_type == "Banking":
            stocks = self.BANKING_STOCKS
        elif universe_type == "IT":
            stocks = self.IT_STOCKS
        elif universe_type == "Pharma":
            stocks = self.PHARMA_STOCKS
        else:
            stocks = self.NIFTY_50_STOCKS[:20]
        
        if limit:
            stocks = stocks[:limit]
        
        return stocks
    
    @st.cache_data(ttl=1800, show_spinner=False)  # 30 min cache
    def fetch_stock_data(_self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Fetch stock data with caching and error handling"""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                stock = yf.Ticker(symbol)
                data = stock.history(period=period)
                if not data.empty:
                    return data
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    st.warning(f"Failed to fetch {symbol} after {max_retries} attempts")
        
        return pd.DataFrame()
    
    @st.cache_data(ttl=1800, show_spinner=False)  # 30 min cache
    def get_fundamental_data(_self, symbol: str) -> Dict:
        """Get fundamental data with retry logic"""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                stock = yf.Ticker(symbol)
                info = stock.info
                
                if info:
                    fundamentals = {
                        'Market Cap': info.get('marketCap', 'N/A'),
                        'PE Ratio': info.get('trailingPE', 'N/A'),
                        'Forward PE': info.get('forwardPE', 'N/A'),
                        'PB Ratio': info.get('priceToBook', 'N/A'),
                        'Dividend Yield': info.get('dividendYield', 'N/A'),
                        'ROE': info.get('returnOnEquity', 'N/A'),
                        'ROA': info.get('returnOnAssets', 'N/A'),
                        'Debt to Equity': info.get('debtToEquity', 'N/A'),
                        'Current Ratio': info.get('currentRatio', 'N/A'),
                        'Quick Ratio': info.get('quickRatio', 'N/A'),
                        'Revenue': info.get('totalRevenue', 'N/A'),
                        'Profit Margin': info.get('profitMargins', 'N/A'),
                        'Operating Margin': info.get('operatingMargins', 'N/A'),
                        'EPS': info.get('trailingEps', 'N/A'),
                        'Beta': info.get('beta', 'N/A'),
                        '52 Week High': info.get('fiftyTwoWeekHigh', 'N/A'),
                        '52 Week Low': info.get('fiftyTwoWeekLow', 'N/A'),
                        'Average Volume': info.get('averageVolume', 'N/A'),
                        'Company Name': info.get('longName', symbol),
                        'Sector': info.get('sector', 'N/A'),
                        'Industry': info.get('industry', 'N/A')
                    }
                    return fundamentals
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
        
        return {}
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators - simplified for cloud"""
        if df.empty or len(df) < 20:
            return df
        
        try:
            # Simple moving averages only
            df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
            df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
            
            # Simple RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / (loss + 0.001)  # Avoid division by zero
            df['RSI'] = 100 - (100 / (1 + rs))
        except:
            pass
        
        return df
    
    def quality_score(self, fundamentals: Dict) -> float:
        """Simplified quality score for cloud"""
        score = 0
        max_score = 0
        
        try:
            # ROE Score
            if fundamentals.get('ROE') not in ['N/A', None]:
                roe = float(fundamentals['ROE'])
                if roe > 0.15:
                    score += 30
                elif roe > 0.10:
                    score += 20
                elif roe > 0:
                    score += 10
                max_score += 30
            
            # Debt to Equity
            if fundamentals.get('Debt to Equity') not in ['N/A', None]:
                de = float(fundamentals['Debt to Equity'])
                if de < 1:
                    score += 30
                elif de < 2:
                    score += 20
                elif de < 3:
                    score += 10
                max_score += 30
            
            # Profit Margin
            if fundamentals.get('Profit Margin') not in ['N/A', None]:
                pm = float(fundamentals['Profit Margin'])
                if pm > 0.10:
                    score += 40
                elif pm > 0.05:
                    score += 25
                elif pm > 0:
                    score += 10
                max_score += 40
        except:
            pass
        
        return (score / max_score * 100) if max_score > 0 else 50
    
    def value_score(self, fundamentals: Dict, current_price: float = None) -> float:
        """Simplified value score for cloud"""
        score = 0
        max_score = 0
        
        try:
            # PE Ratio
            if fundamentals.get('PE Ratio') not in ['N/A', None]:
                pe = float(fundamentals['PE Ratio'])
                if 0 < pe < 20:
                    score += 50
                elif 20 <= pe < 30:
                    score += 30
                elif 30 <= pe < 40:
                    score += 10
                max_score += 50
            
            # PB Ratio
            if fundamentals.get('PB Ratio') not in ['N/A', None]:
                pb = float(fundamentals['PB Ratio'])
                if pb < 2:
                    score += 50
                elif pb < 3:
                    score += 30
                elif pb < 5:
                    score += 10
                max_score += 50
        except:
            pass
        
        return (score / max_score * 100) if max_score > 0 else 50
    
    def momentum_score(self, df: pd.DataFrame) -> float:
        """Simplified momentum score for cloud"""
        if df.empty or len(df) < 20:
            return 50
        
        score = 0
        max_score = 100
        
        try:
            latest = df.iloc[-1]
            
            # Price above SMA20
            if 'SMA_20' in df.columns and not pd.isna(latest['SMA_20']):
                if latest['Close'] > latest['SMA_20']:
                    score += 50
            
            # Positive returns
            returns_1m = (latest['Close'] - df.iloc[-22]['Close']) / df.iloc[-22]['Close'] if len(df) > 22 else 0
            if returns_1m > 0:
                score += 50
        except:
            score = 50
        
        return score

class CloudOptimizedScreener:
    """Screener optimized for Streamlit Cloud deployment"""
    
    @staticmethod
    def process_single_stock(symbol: str, analyzer: IndianStockAnalyzer) -> Dict:
        """Process a single stock with all error handling"""
        try:
            # Add rate limiting
            analyzer.rate_limit()
            
            # Fetch data with timeout
            fundamentals = analyzer.get_fundamental_data(symbol)
            if not fundamentals:
                return None
            
            df = analyzer.fetch_stock_data(symbol, period="6mo")  # Reduced period
            if df.empty:
                return None
            
            # Calculate indicators
            df = analyzer.calculate_technical_indicators(df)
            
            # Calculate scores
            quality = analyzer.quality_score(fundamentals)
            value = analyzer.value_score(fundamentals, df.iloc[-1]['Close'] if not df.empty else None)
            momentum = analyzer.momentum_score(df)
            
            # Current metrics
            current_price = df.iloc[-1]['Close'] if not df.empty else 'N/A'
            change_1d = ((df.iloc[-1]['Close'] - df.iloc[-2]['Close']) / 
                        df.iloc[-2]['Close'] * 100) if len(df) > 1 else 0
            
            return {
                'Symbol': symbol.replace('.NS', '').replace('.BO', ''),
                'Company': str(fundamentals.get('Company Name', symbol))[:30],
                'Sector': fundamentals.get('Sector', 'N/A'),
                'Price': round(float(current_price), 2) if current_price != 'N/A' else 'N/A',
                'Change %': round(change_1d, 2),
                'PE': round(float(fundamentals.get('PE Ratio', 0)), 1) if fundamentals.get('PE Ratio') not in ['N/A', None] else 'N/A',
                'Quality': round(quality, 1),
                'Value': round(value, 1),
                'Momentum': round(momentum, 1),
                'Overall': round((quality + value + momentum) / 3, 1)
            }
        except Exception as e:
            return None
    
    @staticmethod
    def screen_stocks_batch(stocks: List[str], analyzer: IndianStockAnalyzer, 
                          screen_type: str, min_score: float = 0,
                          progress_bar=None, status_text=None,
                          batch_size: int = 5) -> pd.DataFrame:
        """Screen stocks in batches to avoid timeouts"""
        results = []
        total_stocks = len(stocks)
        
        # Process in batches
        for i in range(0, total_stocks, batch_size):
            batch = stocks[i:i+batch_size]
            batch_results = []
            
            for j, symbol in enumerate(batch):
                if progress_bar:
                    progress_bar.progress((i + j + 1) / total_stocks)
                if status_text:
                    status_text.text(f"Processing {symbol} ({i+j+1}/{total_stocks})...")
                
                result = CloudOptimizedScreener.process_single_stock(symbol, analyzer)
                if result:
                    # Apply filter
                    score_value = result.get(screen_type.replace(' Score', ''), result.get('Overall', 0))
                    if score_value >= min_score:
                        batch_results.append(result)
            
            results.extend(batch_results)
            
            # Small delay between batches
            time.sleep(0.5)
        
        if not results:
            return pd.DataFrame()
        
        df_results = pd.DataFrame(results)
        
        # Sort by selected screen type
        if screen_type == "Quality":
            df_results = df_results.sort_values('Quality', ascending=False)
        elif screen_type == "Value":
            df_results = df_results.sort_values('Value', ascending=False)
        elif screen_type == "Momentum":
            df_results = df_results.sort_values('Momentum', ascending=False)
        else:
            df_results = df_results.sort_values('Overall', ascending=False)
        
        return df_results

def main():
    st.markdown('<h1 class="main-header">🏛️ Indian Equity Research Platform</h1>', 
                unsafe_allow_html=True)
    
    # Check if running on cloud
    IS_CLOUD = st.secrets.get("IS_CLOUD", False) if hasattr(st, 'secrets') else False
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        api_key = st.text_input("Gemini API Key", 
                               type="password", 
                               value=st.session_state.api_key,
                               help="Enter your Google Gemini API key")
        
        if api_key:
            st.session_state.api_key = api_key
            st.success("✅ API Key configured")
        
        st.markdown("---")
        
        # Deployment info
        if IS_CLOUD:
            st.info("☁️ Running on Streamlit Cloud")
        else:
            st.info("💻 Running Locally")
        
        st.markdown("---")
        st.markdown("### 📊 Quick Links")
        st.markdown("- [NSE India](https://www.nseindia.com)")
        st.markdown("- [BSE India](https://www.bseindia.com)")
    
    # Initialize analyzer
    analyzer = IndianStockAnalyzer(api_key=st.session_state.api_key)
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📊 Stock Analysis", 
                                "🔍 Stock Screener", 
                                "📈 Quick Analysis"])
    
    with tab1:
        st.header("Individual Stock Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            stock_input = st.text_input("Enter Stock Symbol (e.g., RELIANCE, TCS, INFY)", 
                                       value="RELIANCE")
            symbol = f"{stock_input}.NS"
        
        with col2:
            period = st.selectbox("Time Period", 
                                ["1mo", "3mo", "6mo", "1y"],
                                index=2)
            analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)
        
        if analyze_btn:
            with st.spinner(f"Analyzing {symbol}..."):
                # Fetch data
                fundamentals = analyzer.get_fundamental_data(symbol)
                df = analyzer.fetch_stock_data(symbol, period=period)
                
                if not df.empty and fundamentals:
                    # Calculate indicators
                    df = analyzer.calculate_technical_indicators(df)
                    
                    # Display metrics
                    st.subheader(f"📊 {fundamentals.get('Company Name', symbol)}")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Current Price", f"₹{df.iloc[-1]['Close']:.2f}")
                    with col2:
                        change = ((df.iloc[-1]['Close'] - df.iloc[-2]['Close']) / 
                                 df.iloc[-2]['Close'] * 100) if len(df) > 1 else 0
                        st.metric("Change", f"{change:.2f}%")
                    with col3:
                        st.metric("PE Ratio", fundamentals.get('PE Ratio', 'N/A'))
                    with col4:
                        st.metric("PB Ratio", fundamentals.get('PB Ratio', 'N/A'))
                    
                    # Price chart
                    st.subheader("Price Chart")
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=df.index,
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'],
                        name='Price'
                    ))
                    
                    if 'SMA_20' in df.columns:
                        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], 
                                                name='SMA 20', line=dict(color='orange')))
                    
                    fig.update_layout(height=500, xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Scores
                    st.subheader("Stock Scores")
                    quality = analyzer.quality_score(fundamentals)
                    value = analyzer.value_score(fundamentals, df.iloc[-1]['Close'])
                    momentum = analyzer.momentum_score(df)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Quality Score", f"{quality:.1f}")
                        st.progress(quality/100)
                    with col2:
                        st.metric("Value Score", f"{value:.1f}")
                        st.progress(value/100)
                    with col3:
                        st.metric("Momentum Score", f"{momentum:.1f}")
                        st.progress(momentum/100)
                else:
                    st.error("Unable to fetch data. Please try again.")
    
    with tab2:
        st.header("Stock Screener")
        
        # Screener settings
        col1, col2, col3 = st.columns(3)
        
        with col1:
            screen_type = st.selectbox("Strategy", 
                                      ["Overall", "Quality", "Value", "Momentum"])
        
        with col2:
            # Limited options for cloud
            stock_universe = st.selectbox("Universe", 
                                         ["NIFTY Top 10", "NIFTY Top 20", 
                                          "Banking", "IT", "Pharma", "Custom"])
        
        with col3:
            min_score = st.slider("Min Score", 0, 100, 40, 10)
        
        # Custom stocks input
        if stock_universe == "Custom":
            custom_stocks = st.text_input("Symbols (comma-separated)", 
                                         "RELIANCE,TCS,INFY,HDFCBANK,ICICIBANK")
            stocks = [f"{s.strip()}.NS" for s in custom_stocks.split(',')]
        else:
            # Get limited stock list for cloud
            if stock_universe == "NIFTY Top 10":
                limit = 10
            elif stock_universe == "NIFTY Top 20":
                limit = 20
            else:
                limit = 10
            
            stocks = analyzer.get_stock_universe(stock_universe, limit=limit)
        
        if st.button("🔍 Run Screener", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text(f"Screening {len(stocks)} stocks...")
            
            # Use cloud-optimized screener
            screener = CloudOptimizedScreener()
            
            # Screen with smaller batch size for cloud
            results = screener.screen_stocks_batch(
                stocks, analyzer, screen_type, min_score, 
                progress_bar, status_text, batch_size=3
            )
            
            progress_bar.empty()
            status_text.empty()
            
            if not results.empty:
                st.success(f"✅ Found {len(results)} stocks")
                
                # Store in session state
                st.session_state.screening_results = results
                
                # Display results
                st.dataframe(results, use_container_width=True, height=400)
                
                # Download option
                csv = results.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"screening_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
                
                # Top picks
                if len(results) > 0:
                    st.subheader("🏆 Top 5 Picks")
                    st.dataframe(results.head(5))
            else:
                st.warning("No stocks found. Try lowering the minimum score.")
    
    with tab3:
        st.header("Quick Analysis")
        
        # Popular stocks for quick access
        st.subheader("Popular Stocks")
        
        popular = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS']
        
        cols = st.columns(5)
        for i, stock in enumerate(popular):
            with cols[i]:
                if st.button(stock.replace('.NS', ''), use_container_width=True):
                    with st.spinner(f"Fetching {stock}..."):
                        df = analyzer.fetch_stock_data(stock, period="1mo")
                        if not df.empty:
                            current = df.iloc[-1]['Close']
                            prev = df.iloc[-2]['Close']
                            change = ((current - prev) / prev) * 100
                            
                            st.metric(stock.replace('.NS', ''), 
                                     f"₹{current:.2f}", 
                                     f"{change:.2f}%")
        
        # Market summary
        st.subheader("Market Indices")
        
        indices = {
            '^NSEI': 'NIFTY 50',
            '^BSESN': 'SENSEX',
        }
        
        cols = st.columns(len(indices))
        for i, (symbol, name) in enumerate(indices.items()):
            with cols[i]:
                try:
                    index_data = yf.Ticker(symbol).history(period="1d")
                    if not index_data.empty:
                        current = index_data.iloc[-1]['Close']
                        prev = index_data.iloc[0]['Open']
                        change = ((current - prev) / prev) * 100
                        st.metric(name, f"{current:,.2f}", delta=f"{change:.2f}%")
                except:
                    st.metric(name, "N/A")

if __name__ == "__main__":
    main()
