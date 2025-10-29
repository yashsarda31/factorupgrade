# app.py - Complete Version with Cloud Optimization
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
import os

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

# Detect if running on cloud
IS_CLOUD = os.environ.get('STREAMLIT_SHARING', False) or \
           os.environ.get('IS_CLOUD', False) or \
           (hasattr(st, 'secrets') and st.secrets.get("IS_CLOUD", False))

class IndianStockAnalyzer:
    """Main class for Indian stock analysis with cloud optimization"""
    
    # NIFTY 50 stocks
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
    
    # NIFTY NEXT 50 stocks
    NIFTY_NEXT_50_STOCKS = [
        'ADANIGREEN.NS', 'ADANITRANS.NS', 'AMBUJACEM.NS', 'ATGL.NS', 'AUROPHARMA.NS',
        'BANDHANBNK.NS', 'BANKBARODA.NS', 'BERGEPAINT.NS', 'BIOCON.NS', 'BOSCHLTD.NS',
        'CANBK.NS', 'CHOLAFIN.NS', 'COLPAL.NS', 'DLF.NS', 'DABUR.NS',
        'DMART.NS', 'GAIL.NS', 'GODREJCP.NS', 'HAVELLS.NS', 'HDFCLIFE.NS',
        'ICICIGI.NS', 'ICICIPRULI.NS', 'IGL.NS', 'INDIGO.NS', 'INDUSTOWER.NS',
        'JINDALSTEL.NS', 'JUBLFOOD.NS', 'LTI.NS', 'LUPIN.NS', 'MARICO.NS',
        'MCDOWELL-N.NS', 'MUTHOOTFIN.NS', 'NAUKRI.NS', 'PEL.NS', 'PIDILITIND.NS',
        'PNB.NS', 'SBICARD.NS', 'SIEMENS.NS', 'TATAPOWER.NS', 'TORNTPHARM.NS',
        'TRENT.NS', 'TVSMOTOR.NS', 'VEDL.NS', 'ZEEL.NS', 'GODREJPROP.NS',
        'ICICIPRU.NS', 'INDIANB.NS', 'YESBANK.NS', 'ZOMATO.NS', 'PAYTM.NS'
    ]
    
    # NIFTY MIDCAP 100 (Sample)
    NIFTY_MIDCAP_100_STOCKS = [
        'AARTIIND.NS', 'ABB.NS', 'ABBOTINDIA.NS', 'ABCAPITAL.NS', 'ABFRL.NS',
        'ACC.NS', 'ADANIPOWER.NS', 'ALKEM.NS', 'APLLTD.NS', 'ASHOKLEY.NS',
        'ASTRAL.NS', 'ATUL.NS', 'AUBANK.NS', 'BAJAJHLDNG.NS', 'BALKRISIND.NS',
        'BALRAMCHIN.NS', 'BATAINDIA.NS', 'BEL.NS', 'BHARATFORG.NS', 'BHEL.NS',
        'BIRLACORPN.NS', 'BLUEDART.NS', 'CAMS.NS', 'CANFINHOME.NS', 'CASTROLIND.NS',
        'CDSL.NS', 'CENTRALBK.NS', 'CENTURYPLY.NS', 'CHAMBLFERT.NS', 'CHOLAHLDNG.NS'
    ]
    
    # Sector-wise stocks
    BANKING_STOCKS = [
        'HDFCBANK.NS', 'ICICIBANK.NS', 'AXISBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS',
        'INDUSINDBK.NS', 'BANDHANBNK.NS', 'BANKBARODA.NS', 'PNB.NS', 'CANBK.NS',
        'FEDERALBNK.NS', 'AUBANK.NS', 'IDFCFIRSTB.NS', 'RBLBANK.NS', 'YESBANK.NS'
    ]
    
    IT_STOCKS = [
        'TCS.NS', 'INFY.NS', 'HCLTECH.NS', 'WIPRO.NS', 'TECHM.NS',
        'LTI.NS', 'MINDTREE.NS', 'MPHASIS.NS', 'COFORGE.NS', 'PERSISTENT.NS',
        'LTTS.NS', 'ROUTE.NS', 'CYIENT.NS', 'ECLERX.NS', 'LATENTVIEW.NS'
    ]
    
    PHARMA_STOCKS = [
        'SUNPHARMA.NS', 'DRREDDY.NS', 'DIVISLAB.NS', 'CIPLA.NS', 'AUROPHARMA.NS',
        'LUPIN.NS', 'TORNTPHARM.NS', 'ALKEM.NS', 'BIOCON.NS', 'GLENMARK.NS',
        'CADILAHC.NS', 'ABBOTINDIA.NS', 'SANOFI.NS', 'PFIZER.NS', 'GLAXO.NS'
    ]
    
    AUTO_STOCKS = [
        'MARUTI.NS', 'TATAMOTORS.NS', 'M&M.NS', 'BAJAJ-AUTO.NS', 'EICHERMOT.NS',
        'HEROMOTOCO.NS', 'TVSMOTOR.NS', 'ASHOKLEY.NS', 'BHARATFORG.NS', 'ESCORTS.NS',
        'MOTHERSON.NS', 'BOSCHLTD.NS', 'MRF.NS', 'APOLLOTYRE.NS', 'CEAT.NS'
    ]
    
    FMCG_STOCKS = [
        'HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS', 'DABUR.NS',
        'MARICO.NS', 'GODREJCP.NS', 'TATACONSUM.NS', 'COLPAL.NS', 'GILLETTE.NS',
        'PGHH.NS', 'VBL.NS', 'RADICO.NS', 'EMAMILTD.NS', 'BAJAJCON.NS'
    ]
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.request_count = 0
        self.last_request_time = time.time()
        self.is_cloud = IS_CLOUD
        
        if api_key:
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-pro')
            except Exception as e:
                st.error(f"Error configuring Gemini API: {str(e)}")
    
    def get_delay(self):
        """Get appropriate delay based on environment"""
        if self.is_cloud:
            return 0.3  # Longer delay for cloud
        else:
            return 0.1  # Shorter delay for local
    
    def rate_limit(self):
        """Implement rate limiting for API calls"""
        delay = self.get_delay()
        time.sleep(delay)
        
        self.request_count += 1
        current_time = time.time()
        
        # Reset counter every minute
        if current_time - self.last_request_time > 60:
            self.request_count = 0
            self.last_request_time = current_time
        
        # If too many requests, wait more
        if self.is_cloud and self.request_count > 20:  # Stricter for cloud
            time.sleep(2)
            self.request_count = 0
            self.last_request_time = time.time()
        elif not self.is_cloud and self.request_count > 50:  # More lenient locally
            time.sleep(1)
            self.request_count = 0
            self.last_request_time = time.time()
    
    def get_stock_universe(self, universe_type: str, limit: int = None) -> List[str]:
        """Get stock list based on universe type"""
        stocks = []
        
        if universe_type == "NIFTY 50":
            stocks = self.NIFTY_50_STOCKS
        elif universe_type == "NIFTY NEXT 50":
            stocks = self.NIFTY_NEXT_50_STOCKS
        elif universe_type == "NIFTY 100":
            stocks = self.NIFTY_50_STOCKS + self.NIFTY_NEXT_50_STOCKS
        elif universe_type == "NIFTY MIDCAP 100":
            stocks = self.NIFTY_MIDCAP_100_STOCKS
        elif universe_type == "Banking":
            stocks = self.BANKING_STOCKS
        elif universe_type == "IT":
            stocks = self.IT_STOCKS
        elif universe_type == "Pharma":
            stocks = self.PHARMA_STOCKS
        elif universe_type == "Auto":
            stocks = self.AUTO_STOCKS
        elif universe_type == "FMCG":
            stocks = self.FMCG_STOCKS
        elif universe_type == "All Sectors":
            stocks = list(set(self.BANKING_STOCKS + self.IT_STOCKS + self.PHARMA_STOCKS + 
                          self.AUTO_STOCKS + self.FMCG_STOCKS))
        else:
            stocks = self.NIFTY_50_STOCKS
        
        # Apply limit if specified (for cloud optimization)
        if limit and self.is_cloud:
            stocks = stocks[:limit]
        
        return stocks
    
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_stock_data(_self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Fetch stock data with caching"""
        max_retries = 3 if not _self.is_cloud else 2
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
        
        return pd.DataFrame()
    
    @st.cache_data(ttl=3600, show_spinner=False)
    def get_fundamental_data(_self, symbol: str) -> Dict:
        """Get fundamental data for a stock"""
        max_retries = 3 if not _self.is_cloud else 2
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
        """Calculate technical indicators"""
        if df.empty:
            return df
        
        try:
            # Moving averages
            df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
            df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
            df['SMA_200'] = df['Close'].rolling(window=200, min_periods=1).mean()
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / (loss + 0.0001)  # Avoid division by zero
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            
            # Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(window=20, min_periods=1).mean()
            bb_std = df['Close'].rolling(window=20, min_periods=1).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        except:
            pass
        
        return df
    
    def quality_score(self, fundamentals: Dict) -> float:
        """Calculate quality score based on fundamentals"""
        score = 0
        max_score = 0
        
        try:
            # ROE Score (higher is better)
            if fundamentals.get('ROE') not in ['N/A', None]:
                roe = float(fundamentals['ROE'])
                if roe > 0.20:
                    score += 20
                elif roe > 0.15:
                    score += 15
                elif roe > 0.10:
                    score += 10
                max_score += 20
            
            # Debt to Equity (lower is better)
            if fundamentals.get('Debt to Equity') not in ['N/A', None]:
                de = float(fundamentals['Debt to Equity'])
                if de < 0.5:
                    score += 20
                elif de < 1:
                    score += 15
                elif de < 1.5:
                    score += 10
                max_score += 20
            
            # Current Ratio (between 1.5 and 3 is ideal)
            if fundamentals.get('Current Ratio') not in ['N/A', None]:
                cr = float(fundamentals['Current Ratio'])
                if 1.5 <= cr <= 3:
                    score += 20
                elif 1 <= cr < 1.5:
                    score += 10
                max_score += 20
            
            # Profit Margin (higher is better)
            if fundamentals.get('Profit Margin') not in ['N/A', None]:
                pm = float(fundamentals['Profit Margin'])
                if pm > 0.15:
                    score += 20
                elif pm > 0.10:
                    score += 15
                elif pm > 0.05:
                    score += 10
                max_score += 20
            
            # Operating Margin (higher is better)
            if fundamentals.get('Operating Margin') not in ['N/A', None]:
                om = float(fundamentals['Operating Margin'])
                if om > 0.20:
                    score += 20
                elif om > 0.15:
                    score += 15
                elif om > 0.10:
                    score += 10
                max_score += 20
        except:
            pass
        
        return (score / max_score * 100) if max_score > 0 else 50
    
    def value_score(self, fundamentals: Dict, current_price: float = None) -> float:
        """Calculate value score based on fundamentals"""
        score = 0
        max_score = 0
        
        try:
            # PE Ratio (lower is better, but positive)
            if fundamentals.get('PE Ratio') not in ['N/A', None]:
                pe = float(fundamentals['PE Ratio'])
                if 0 < pe < 15:
                    score += 25
                elif 15 <= pe < 20:
                    score += 20
                elif 20 <= pe < 25:
                    score += 10
                max_score += 25
            
            # PB Ratio (lower is better)
            if fundamentals.get('PB Ratio') not in ['N/A', None]:
                pb = float(fundamentals['PB Ratio'])
                if pb < 1:
                    score += 25
                elif pb < 2:
                    score += 20
                elif pb < 3:
                    score += 10
                max_score += 25
            
            # Dividend Yield (higher is better)
            if fundamentals.get('Dividend Yield') not in ['N/A', None]:
                dy = float(fundamentals['Dividend Yield'])
                if dy > 0.03:
                    score += 25
                elif dy > 0.02:
                    score += 20
                elif dy > 0.01:
                    score += 10
                max_score += 25
            
            # Price to 52-week range
            if (fundamentals.get('52 Week High') not in ['N/A', None] and 
                fundamentals.get('52 Week Low') not in ['N/A', None] and 
                current_price):
                high = float(fundamentals['52 Week High'])
                low = float(fundamentals['52 Week Low'])
                price_position = (current_price - low) / (high - low) if high != low else 0.5
                if price_position < 0.3:
                    score += 25
                elif price_position < 0.5:
                    score += 20
                elif price_position < 0.7:
                    score += 10
                max_score += 25
        except:
            pass
        
        return (score / max_score * 100) if max_score > 0 else 50
    
    def momentum_score(self, df: pd.DataFrame) -> float:
        """Calculate momentum score based on price action"""
        if df.empty or len(df) < 20:
            return 50
        
        score = 0
        max_score = 0
        
        try:
            latest = df.iloc[-1]
            
            # Price above moving averages
            if 'SMA_20' in df.columns and not pd.isna(latest['SMA_20']):
                if latest['Close'] > latest['SMA_20']:
                    score += 20
                max_score += 20
            
            if 'SMA_50' in df.columns and not pd.isna(latest['SMA_50']):
                if latest['Close'] > latest['SMA_50']:
                    score += 20
                max_score += 20
            
            if 'SMA_200' in df.columns and not pd.isna(latest['SMA_200']) and len(df) >= 200:
                if latest['Close'] > latest['SMA_200']:
                    score += 20
                max_score += 20
            
            # RSI (between 40 and 70 is good for momentum)
            if 'RSI' in df.columns and not pd.isna(latest['RSI']):
                if 40 <= latest['RSI'] <= 70:
                    score += 20
                max_score += 20
            
            # Price performance
            if len(df) > 22:
                returns_1m = (latest['Close'] - df.iloc[-22]['Close']) / df.iloc[-22]['Close']
                if returns_1m > 0.05:
                    score += 10
                max_score += 10
            
            if len(df) > 66:
                returns_3m = (latest['Close'] - df.iloc[-66]['Close']) / df.iloc[-66]['Close']
                if returns_3m > 0.15:
                    score += 10
                max_score += 10
        except:
            pass
        
        return (score / max_score * 100) if max_score > 0 else 50
    
    def generate_ai_report(self, symbol: str, fundamentals: Dict, 
                          quality: float, value: float, momentum: float) -> str:
        """Generate AI-powered analysis report using Gemini"""
        if not self.api_key:
            return "Please configure Gemini API key to generate AI reports."
        
        try:
            prompt = f"""
            Generate a comprehensive equity research report for {fundamentals.get('Company Name', symbol)} ({symbol}) - an Indian stock.
            
            Fundamental Data:
            - Sector: {fundamentals.get('Sector', 'N/A')}
            - Industry: {fundamentals.get('Industry', 'N/A')}
            - Market Cap: {fundamentals.get('Market Cap', 'N/A')}
            - PE Ratio: {fundamentals.get('PE Ratio', 'N/A')}
            - PB Ratio: {fundamentals.get('PB Ratio', 'N/A')}
            - ROE: {fundamentals.get('ROE', 'N/A')}
            - Debt to Equity: {fundamentals.get('Debt to Equity', 'N/A')}
            - Profit Margin: {fundamentals.get('Profit Margin', 'N/A')}
            - Dividend Yield: {fundamentals.get('Dividend Yield', 'N/A')}
            
            Scores:
            - Quality Score: {quality:.2f}/100
            - Value Score: {value:.2f}/100
            - Momentum Score: {momentum:.2f}/100
            
            Please provide:
            1. Executive Summary (2-3 sentences)
            2. Business Overview and Competitive Position
            3. Financial Health Analysis
            4. Valuation Assessment
            5. Technical Analysis Summary
            6. Key Risks and Concerns
            7. Investment Recommendation (Buy/Hold/Sell with reasoning)
            8. Price Target Indication (if possible)
            
            Make it professional, data-driven, and specific to Indian market context.
            """
            
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating AI report: {str(e)}"

class StockScreener:
    """Stock screening functionality optimized for cloud"""
    
    @staticmethod
    def screen_stocks(stocks: List[str], analyzer: IndianStockAnalyzer, 
                     screen_type: str, min_score: float = 0,
                     progress_bar=None, status_text=None) -> pd.DataFrame:
        """Screen stocks based on criteria"""
        results = []
        failed_stocks = []
        
        # Determine batch size based on environment
        batch_size = 5 if analyzer.is_cloud else 10
        
        for i, symbol in enumerate(stocks):
            if progress_bar:
                progress_bar.progress((i + 1) / len(stocks))
            if status_text:
                status_text.text(f"Analyzing {symbol} ({i+1}/{len(stocks)})...")
            
            try:
                # Rate limiting
                analyzer.rate_limit()
                
                # Fetch data with appropriate period
                period = "6mo" if analyzer.is_cloud else "1y"
                fundamentals = analyzer.get_fundamental_data(symbol)
                df = analyzer.fetch_stock_data(symbol, period=period)
                
                if not fundamentals or df.empty:
                    failed_stocks.append(symbol)
                    continue
                
                # Calculate indicators
                df = analyzer.calculate_technical_indicators(df)
                
                # Calculate scores
                quality = analyzer.quality_score(fundamentals)
                value = analyzer.value_score(fundamentals, 
                                           df.iloc[-1]['Close'] if not df.empty else None)
                momentum = analyzer.momentum_score(df)
                overall = (quality + value + momentum) / 3
                
                # Apply minimum score filter
                score_to_check = {
                    "Quality": quality,
                    "Value": value,
                    "Momentum": momentum,
                    "Overall": overall
                }.get(screen_type, overall)
                
                if score_to_check < min_score:
                    continue
                
                # Current metrics
                current_price = df.iloc[-1]['Close'] if not df.empty else 'N/A'
                change_1d = ((df.iloc[-1]['Close'] - df.iloc[-2]['Close']) / 
                            df.iloc[-2]['Close'] * 100) if len(df) > 1 else 0
                
                results.append({
                    'Symbol': symbol.replace('.NS', '').replace('.BO', ''),
                    'Company': str(fundamentals.get('Company Name', symbol))[:30],
                    'Sector': fundamentals.get('Sector', 'N/A'),
                    'Price': round(float(current_price), 2) if current_price != 'N/A' else 'N/A',
                    'Change %': round(change_1d, 2),
                    'PE Ratio': round(float(fundamentals.get('PE Ratio', 0)), 2) if fundamentals.get('PE Ratio') not in ['N/A', None] else 'N/A',
                    'PB Ratio': round(float(fundamentals.get('PB Ratio', 0)), 2) if fundamentals.get('PB Ratio') not in ['N/A', None] else 'N/A',
                    'ROE %': round(float(fundamentals.get('ROE', 0)) * 100, 2) if fundamentals.get('ROE') not in ['N/A', None] else 'N/A',
                    'D/E': round(float(fundamentals.get('Debt to Equity', 0)), 2) if fundamentals.get('Debt to Equity') not in ['N/A', None] else 'N/A',
                    'Quality': round(quality, 1),
                    'Value': round(value, 1),
                    'Momentum': round(momentum, 1),
                    'Overall': round(overall, 1)
                })
                
            except Exception as e:
                failed_stocks.append(symbol)
                continue
        
        if failed_stocks and status_text:
            status_text.warning(f"Failed to analyze {len(failed_stocks)} stocks")
        
        df_results = pd.DataFrame(results)
        
        if df_results.empty:
            return df_results
        
        # Sort by selected screen type
        sort_column = screen_type if screen_type in df_results.columns else 'Overall'
        df_results = df_results.sort_values(sort_column, ascending=False)
        
        return df_results

def style_dataframe(df, columns_to_style):
    """Apply custom styling to dataframe"""
    def color_negative_red(val):
        """Color negative values red and positive values green"""
        try:
            if isinstance(val, str):
                return ''
            color = 'red' if val < 0 else 'green' if val > 0 else 'black'
            return f'color: {color}'
        except:
            return ''
    
    def background_gradient_custom(val, vmin, vmax):
        """Custom background gradient"""
        try:
            if isinstance(val, str) or val == 'N/A':
                return ''
            # Normalize value between 0 and 1
            norm_val = (val - vmin) / (vmax - vmin) if vmax != vmin else 0.5
            # Create color from red to yellow to green
            if norm_val < 0.5:
                # Red to yellow
                r = 255
                g = int(255 * (norm_val * 2))
                b = 0
            else:
                # Yellow to green
                r = int(255 * (2 - norm_val * 2))
                g = 255
                b = 0
            return f'background-color: rgba({r}, {g}, {b}, 0.3)'
        except:
            return ''
    
    styled_df = df.style
    
    # Apply color to change columns
    if 'Change %' in df.columns:
        styled_df = styled_df.applymap(color_negative_red, subset=['Change %'])
    
    # Apply gradient to score columns
    for col in columns_to_style:
        if col in df.columns:
            if HAS_MATPLOTLIB:
                try:
                    styled_df = styled_df.background_gradient(subset=[col], cmap='RdYlGn', vmin=0, vmax=100)
                except:
                    styled_df = styled_df.applymap(
                        lambda x: background_gradient_custom(x, 0, 100), 
                        subset=[col]
                    )
            else:
                styled_df = styled_df.applymap(
                    lambda x: background_gradient_custom(x, 0, 100), 
                    subset=[col]
                )
    
    return styled_df

def main():
    st.markdown('<h1 class="main-header">🏛️ Indian Equity Research Platform</h1>', 
                unsafe_allow_html=True)
    
    # Display environment info
    if IS_CLOUD:
        st.info("☁️ Running on Streamlit Cloud - Optimized performance mode")
    else:
        st.info("💻 Running Locally - Full feature mode")
    
    # Sidebar for API configuration
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
        st.markdown("### 📊 Quick Links")
        st.markdown("- [NSE India](https://www.nseindia.com)")
        st.markdown("- [BSE India](https://www.bseindia.com)")
        st.markdown("- [Moneycontrol](https://www.moneycontrol.com)")
    
    # Initialize analyzer
    analyzer = IndianStockAnalyzer(api_key=st.session_state.api_key)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Stock Analysis", 
                                            "🔍 Stock Screener", 
                                            "📈 Portfolio Analysis",
                                            "📑 Bulk Reports",
                                            "📉 Market Overview"])
    
    with tab1:
        st.header("Individual Stock Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            stock_input = st.text_input("Enter Stock Symbol (e.g., RELIANCE, TCS, INFY)", 
                                       value="RELIANCE")
            exchange = st.selectbox("Exchange", ["NSE", "BSE"])
            symbol = f"{stock_input}.{'NS' if exchange == 'NSE' else 'BO'}"
        
        with col2:
            period = st.selectbox("Time Period", 
                                ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
                                index=3)
            analyze_btn = st.button("🔍 Analyze Stock", type="primary", use_container_width=True)
        
        if analyze_btn:
            with st.spinner(f"Analyzing {symbol}..."):
                # Fetch data
                fundamentals = analyzer.get_fundamental_data(symbol)
                df = analyzer.fetch_stock_data(symbol, period=period)
                
                if not df.empty and fundamentals:
                    # Calculate technical indicators
                    df = analyzer.calculate_technical_indicators(df)
                    
                    # Display company info
                    st.subheader(f"📊 {fundamentals.get('Company Name', symbol)}")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Sector", fundamentals.get('Sector', 'N/A'))
                    with col2:
                        st.metric("Industry", fundamentals.get('Industry', 'N/A')[:20])
                    with col3:
                        st.metric("Current Price", f"₹{df.iloc[-1]['Close']:.2f}")
                    with col4:
                        change = ((df.iloc[-1]['Close'] - df.iloc[-2]['Close']) / 
                                 df.iloc[-2]['Close'] * 100)
                        st.metric("Change", f"{change:.2f}%")
                    
                    # Tabs for different analyses
                    analysis_tab1, analysis_tab2, analysis_tab3, analysis_tab4 = st.tabs(
                        ["Price Chart", "Fundamentals", "Technical", "AI Report"])
                    
                    with analysis_tab1:
                        # Price chart with technical indicators
                        fig = go.Figure()
                        
                        # Candlestick
                        fig.add_trace(go.Candlestick(
                            x=df.index,
                            open=df['Open'],
                            high=df['High'],
                            low=df['Low'],
                            close=df['Close'],
                            name='Price'
                        ))
                        
                        # Moving averages
                        if 'SMA_20' in df.columns:
                            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], 
                                                    name='SMA 20', line=dict(color='orange')))
                        if 'SMA_50' in df.columns:
                            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], 
                                                    name='SMA 50', line=dict(color='blue')))
                        if 'SMA_200' in df.columns:
                            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], 
                                                    name='SMA 200', line=dict(color='red')))
                        
                        fig.update_layout(
                            title=f"{symbol} Price Chart",
                            yaxis_title="Price (₹)",
                            xaxis_title="Date",
                            height=600,
                            xaxis_rangeslider_visible=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Volume chart
                        fig_vol = go.Figure()
                        fig_vol.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume'))
                        fig_vol.update_layout(
                            title="Volume",
                            height=200,
                            xaxis_title="Date",
                            yaxis_title="Volume"
                        )
                        st.plotly_chart(fig_vol, use_container_width=True)
                    
                    with analysis_tab2:
                        st.subheader("Fundamental Metrics")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("**Valuation Metrics**")
                            st.write(f"PE Ratio: {fundamentals.get('PE Ratio', 'N/A')}")
                            st.write(f"PB Ratio: {fundamentals.get('PB Ratio', 'N/A')}")
                            st.write(f"Market Cap: {fundamentals.get('Market Cap', 'N/A')}")
                            st.write(f"EPS: {fundamentals.get('EPS', 'N/A')}")
                        
                        with col2:
                            st.markdown("**Profitability Metrics**")
                            st.write(f"ROE: {fundamentals.get('ROE', 'N/A')}")
                            st.write(f"ROA: {fundamentals.get('ROA', 'N/A')}")
                            st.write(f"Profit Margin: {fundamentals.get('Profit Margin', 'N/A')}")
                            st.write(f"Operating Margin: {fundamentals.get('Operating Margin', 'N/A')}")
                        
                        with col3:
                            st.markdown("**Financial Health**")
                            st.write(f"Debt to Equity: {fundamentals.get('Debt to Equity', 'N/A')}")
                            st.write(f"Current Ratio: {fundamentals.get('Current Ratio', 'N/A')}")
                            st.write(f"Quick Ratio: {fundamentals.get('Quick Ratio', 'N/A')}")
                            st.write(f"Dividend Yield: {fundamentals.get('Dividend Yield', 'N/A')}")
                        
                        # Scoring
                        st.markdown("---")
                        st.subheader("Stock Scores")
                        
                        quality = analyzer.quality_score(fundamentals)
                        value = analyzer.value_score(fundamentals, df.iloc[-1]['Close'])
                        momentum = analyzer.momentum_score(df)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Quality Score", f"{quality:.1f}/100")
                            st.progress(quality/100)
                        with col2:
                            st.metric("Value Score", f"{value:.1f}/100")
                            st.progress(value/100)
                        with col3:
                            st.metric("Momentum Score", f"{momentum:.1f}/100")
                            st.progress(momentum/100)
                    
                    with analysis_tab3:
                        st.subheader("Technical Indicators")
                        
                        if 'RSI' in df.columns:
                            # RSI Chart
                            fig_rsi = go.Figure()
                            fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], 
                                                         name='RSI', line=dict(color='purple')))
                            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", 
                                             annotation_text="Overbought")
                            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", 
                                             annotation_text="Oversold")
                            fig_rsi.update_layout(
                                title="RSI (14)",
                                height=300,
                                yaxis_title="RSI",
                                xaxis_title="Date"
                            )
                            st.plotly_chart(fig_rsi, use_container_width=True)
                        
                        if 'MACD' in df.columns and 'Signal' in df.columns:
                            # MACD Chart
                            fig_macd = go.Figure()
                            fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], 
                                                          name='MACD', line=dict(color='blue')))
                            fig_macd.add_trace(go.Scatter(x=df.index, y=df['Signal'], 
                                                          name='Signal', line=dict(color='red')))
                            fig_macd.update_layout(
                                title="MACD",
                                height=300,
                                yaxis_title="MACD",
                                xaxis_title="Date"
                            )
                            st.plotly_chart(fig_macd, use_container_width=True)
                        
                        # Latest indicator values
                        st.markdown("---")
                        st.subheader("Current Technical Readings")
                        
                        latest = df.iloc[-1]
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            if 'RSI' in df.columns:
                                st.metric("RSI", f"{latest['RSI']:.2f}")
                        with col2:
                            if 'MACD' in df.columns:
                                st.metric("MACD", f"{latest['MACD']:.2f}")
                        with col3:
                            if 'Signal' in df.columns:
                                st.metric("Signal", f"{latest['Signal']:.2f}")
                        with col4:
                            if 'MACD' in df.columns and 'Signal' in df.columns:
                                macd_signal = "Bullish" if latest['MACD'] > latest['Signal'] else "Bearish"
                                st.metric("MACD Signal", macd_signal)
                    
                    with analysis_tab4:
                        st.subheader("AI-Generated Equity Research Report")
                        
                        if st.session_state.api_key:
                            with st.spinner("Generating AI report..."):
                                quality = analyzer.quality_score(fundamentals)
                                value = analyzer.value_score(fundamentals, df.iloc[-1]['Close'])
                                momentum = analyzer.momentum_score(df)
                                
                                report = analyzer.generate_ai_report(
                                    symbol, fundamentals, quality, value, momentum)
                                st.markdown(report)
                                
                                # Download button for report
                                st.download_button(
                                    label="📥 Download Report",
                                    data=report,
                                    file_name=f"{symbol}_research_report_{datetime.now().strftime('%Y%m%d')}.txt",
                                    mime="text/plain"
                                )
                        else:
                            st.warning("Please configure Gemini API key in the sidebar to generate AI reports.")
                else:
                    st.error("Unable to fetch data for this symbol. Please check and try again.")
    
    with tab2:
        st.header("Stock Screener")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            col1_1, col1_2, col1_3 = st.columns(3)
            
            with col1_1:
                screen_type = st.selectbox("Screening Strategy", 
                                          ["Quality", "Value", "Momentum", "Overall"])
            
            with col1_2:
                # Adjust options based on environment
                if IS_CLOUD:
                    universe_options = ["NIFTY 50", "Banking", "IT", "Pharma", "Auto", "FMCG", "Custom List"]
                    default_limit = 30
                else:
                    universe_options = ["NIFTY 50", "NIFTY NEXT 50", "NIFTY 100", 
                                      "NIFTY MIDCAP 100", "Banking", "IT", 
                                      "Pharma", "Auto", "FMCG", "All Sectors", "Custom List"]
                    default_limit = 50
                
                stock_universe = st.selectbox("Stock Universe", universe_options)
            
            with col1_3:
                min_score = st.slider("Minimum Score Filter", 0, 100, 50, 10)
        
        with col2:
            st.markdown("### Quick Stats")
            if stock_universe != "Custom List":
                # Get stock count with cloud limits
                if IS_CLOUD and stock_universe in ["NIFTY 100", "All Sectors"]:
                    limit = 50
                else:
                    limit = None
                stocks_count = len(analyzer.get_stock_universe(stock_universe, limit=limit))
                st.info(f"📊 {stocks_count} stocks in {stock_universe}")
        
        # Custom stock input
        if stock_universe == "Custom List":
            custom_stocks = st.text_area("Enter symbols (comma-separated)", 
                                        "RELIANCE,TCS,INFY,HDFCBANK,ICICIBANK,WIPRO,BHARTIARTL,SBIN",
                                        height=100)
        
        # Advanced filters
        with st.expander("Advanced Filters"):
            col1, col2, col3 = st.columns(3)
            with col1:
                max_results = st.number_input("Max Results", 
                                             min_value=10, 
                                             max_value=200 if not IS_CLOUD else 100, 
                                             value=default_limit)
            with col2:
                sort_by = st.selectbox("Sort By", ["Score", "PE Ratio", "PB Ratio", "ROE %"])
            with col3:
                ascending = st.checkbox("Ascending Order", value=False)
        
        if st.button("🔍 Run Screener", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Prepare stock list
            if stock_universe == "Custom List":
                stocks = [f"{s.strip()}.NS" for s in custom_stocks.split(',')]
            else:
                # Apply cloud limits if necessary
                limit = max_results if IS_CLOUD else None
                stocks = analyzer.get_stock_universe(stock_universe, limit=limit)
            
            status_text.text(f"Screening {len(stocks)} stocks...")
            
            # Run screener
            screener = StockScreener()
            results = screener.screen_stocks(
                stocks, analyzer, screen_type, min_score, 
                progress_bar, status_text
            )
            
            progress_bar.empty()
            status_text.empty()
            
            if not results.empty:
                # Apply additional sorting if requested
                if sort_by != "Score":
                    sort_column = {
                        "PE Ratio": "PE Ratio",
                        "PB Ratio": "PB Ratio",
                        "ROE %": "ROE %"
                    }[sort_by]
                    if sort_column in results.columns:
                        # Filter out 'N/A' values for sorting
                        results_clean = results[results[sort_column] != 'N/A'].copy()
                        results_na = results[results[sort_column] == 'N/A'].copy()
                        results_clean = results_clean.sort_values(sort_column, ascending=ascending)
                        results = pd.concat([results_clean, results_na])
                
                # Limit results
                results = results.head(max_results)
                
                st.success(f"✅ Found {len(results)} stocks matching criteria (Score ≥ {min_score})")
                
                # Display summary statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    avg_quality = results['Quality'].mean()
                    st.metric("Avg Quality Score", f"{avg_quality:.1f}")
                with col2:
                    avg_value = results['Value'].mean()
                    st.metric("Avg Value Score", f"{avg_value:.1f}")
                with col3:
                    avg_momentum = results['Momentum'].mean()
                    st.metric("Avg Momentum Score", f"{avg_momentum:.1f}")
                with col4:
                    avg_overall = results['Overall'].mean()
                    st.metric("Avg Overall Score", f"{avg_overall:.1f}")
                
                # Display results with proper styling
                st.subheader(f"📊 Screening Results - Top {len(results)} Stocks")
                columns_to_style = ['Quality', 'Value', 'Momentum', 'Overall']
                styled_df = style_dataframe(results, columns_to_style)
                st.dataframe(styled_df, use_container_width=True, height=600)
                
                # Download results
                csv = results.to_csv(index=False)
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv,
                        file_name=f"{stock_universe}_{screen_type}_screening_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                
                # Top picks visualization
                st.subheader("🏆 Top 10 Picks")
                top_10 = results.head(10)
                
                # Create a bar chart for top 10
                fig = px.bar(top_10, x='Symbol', y=screen_type,
                           color=screen_type,
                           color_continuous_scale='RdYlGn',
                           title=f'Top 10 Stocks by {screen_type} Score')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No stocks found matching criteria (Minimum Score: {min_score}). Try adjusting your filters.")
    
    with tab3:
        st.header("Portfolio Analysis")
        
        st.info("Portfolio analysis feature allows you to track multiple stocks and analyze overall performance.")
        
        # Portfolio input
        portfolio_text = st.text_area(
            "Enter your portfolio (Symbol, Quantity, Buy Price)",
            "RELIANCE, 100, 2400\nTCS, 50, 3500\nINFY, 75, 1400\nHDFCBANK, 25, 1600\nICICIBANK, 100, 950"
        )
        
        if st.button("Analyze Portfolio"):
            # Parse portfolio
            portfolio = []
            for line in portfolio_text.strip().split('\n'):
                parts = line.split(',')
                if len(parts) == 3:
                    portfolio.append({
                        'Symbol': parts[0].strip() + '.NS',
                        'Quantity': int(parts[1].strip()),
                        'Buy Price': float(parts[2].strip())
                    })
            
            if portfolio:
                results = []
                total_invested = 0
                total_current = 0
                
                for stock in portfolio:
                    symbol = stock['Symbol']
                    df = analyzer.fetch_stock_data(symbol, period="1d")
                    
                    if not df.empty:
                        current_price = df.iloc[-1]['Close']
                        invested = stock['Quantity'] * stock['Buy Price']
                        current_value = stock['Quantity'] * current_price
                        pnl = current_value - invested
                        pnl_percent = (pnl / invested) * 100
                        
                        results.append({
                            'Symbol': symbol.replace('.NS', ''),
                            'Quantity': stock['Quantity'],
                            'Buy Price': stock['Buy Price'],
                            'Current Price': round(current_price, 2),
                            'Invested': round(invested, 2),
                            'Current Value': round(current_value, 2),
                            'P&L': round(pnl, 2),
                            'P&L %': round(pnl_percent, 2)
                        })
                        
                        total_invested += invested
                        total_current += current_value
                
                # Display portfolio summary
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Invested", f"₹{total_invested:,.2f}")
                with col2:
                    st.metric("Current Value", f"₹{total_current:,.2f}")
                with col3:
                    total_pnl = total_current - total_invested
                    st.metric("Total P&L", f"₹{total_pnl:,.2f}", 
                             delta=f"{(total_pnl/total_invested)*100:.2f}%")
                with col4:
                    st.metric("Portfolio Return", f"{(total_pnl/total_invested)*100:.2f}%")
                
                # Display holdings
                st.subheader("Holdings")
                df_portfolio = pd.DataFrame(results)
                
                # Apply custom styling
                def style_pnl(val):
                    color = 'green' if val > 0 else 'red' if val < 0 else 'black'
                    return f'color: {color}'
                
                styled_portfolio = df_portfolio.style.applymap(
                    style_pnl, subset=['P&L', 'P&L %']
                )
                
                st.dataframe(styled_portfolio, use_container_width=True)
                
                # Portfolio composition
                st.subheader("Portfolio Composition")
                fig = px.pie(df_portfolio, values='Current Value', names='Symbol',
                           title='Portfolio Allocation')
                st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("Bulk Report Generation")
        
        st.info("Generate comprehensive research reports for multiple stocks at once.")
        
        # Adjust default stocks based on environment
        default_stocks = "RELIANCE,TCS,INFY,HDFCBANK,ICICIBANK" if IS_CLOUD else \
                        "RELIANCE,TCS,INFY,HDFCBANK,ICICIBANK,WIPRO,BHARTIARTL,SBIN,KOTAKBANK,AXISBANK"
        
        report_stocks = st.text_area(
            "Enter stock symbols (comma-separated)",
            default_stocks
        )
        
        if st.button("Generate Bulk Reports"):
            stocks = [f"{s.strip()}.NS" for s in report_stocks.split(',')]
            
            # Limit stocks for cloud
            if IS_CLOUD and len(stocks) > 10:
                st.warning("Limiting to 10 stocks for cloud deployment")
                stocks = stocks[:10]
            
            progress_bar = st.progress(0)
            reports = {}
            
            for i, symbol in enumerate(stocks):
                progress_bar.progress((i + 1) / len(stocks))
                
                with st.spinner(f"Analyzing {symbol}..."):
                    # Rate limiting
                    analyzer.rate_limit()
                    
                    period = "6mo" if IS_CLOUD else "1y"
                    fundamentals = analyzer.get_fundamental_data(symbol)
                    df = analyzer.fetch_stock_data(symbol, period=period)
                    
                    if fundamentals and not df.empty:
                        df = analyzer.calculate_technical_indicators(df)
                        quality = analyzer.quality_score(fundamentals)
                        value = analyzer.value_score(fundamentals, df.iloc[-1]['Close'])
                        momentum = analyzer.momentum_score(df)
                        
                        if st.session_state.api_key:
                            report = analyzer.generate_ai_report(
                                symbol, fundamentals, quality, value, momentum)
                        else:
                            report = f"""
EQUITY RESEARCH REPORT
======================
Symbol: {symbol}
Company: {fundamentals.get('Company Name', 'N/A')}
Sector: {fundamentals.get('Sector', 'N/A')}
Date: {datetime.now().strftime('%Y-%m-%d')}

SCORES
------
Quality Score: {quality:.2f}/100
Value Score: {value:.2f}/100
Momentum Score: {momentum:.2f}/100
Overall Score: {(quality+value+momentum)/3:.2f}/100

FUNDAMENTAL METRICS
-------------------
PE Ratio: {fundamentals.get('PE Ratio', 'N/A')}
PB Ratio: {fundamentals.get('PB Ratio', 'N/A')}
ROE: {fundamentals.get('ROE', 'N/A')}
Debt to Equity: {fundamentals.get('Debt to Equity', 'N/A')}
Profit Margin: {fundamentals.get('Profit Margin', 'N/A')}

RECOMMENDATION
--------------
Based on the scores, this stock shows:
- Quality: {'Strong' if quality > 70 else 'Moderate' if quality > 40 else 'Weak'}
- Value: {'Attractive' if value > 70 else 'Fair' if value > 40 else 'Expensive'}
- Momentum: {'Positive' if momentum > 70 else 'Neutral' if momentum > 40 else 'Negative'}

Note: This is an automated analysis. Please conduct your own research before making investment decisions.
"""
                        
                        reports[symbol] = report
            
            progress_bar.empty()
            
            # Display reports
            for symbol, report in reports.items():
                with st.expander(f"Report: {symbol.replace('.NS', '')}"):
                    st.text(report)
            
            # Download all reports
            all_reports = "\n\n" + "="*80 + "\n\n".join(
                [f"REPORT FOR {symbol}\n{report}" for symbol, report in reports.items()]
            )
            
            st.download_button(
                label="📥 Download All Reports",
                data=all_reports,
                file_name=f"bulk_reports_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )
    
    with tab5:
        st.header("Market Overview")
        
        # Market indices
        st.subheader("📊 Major Indices")
        
        indices = {
            '^NSEI': 'NIFTY 50',
            '^BSESN': 'SENSEX',
            '^NSEBANK': 'BANK NIFTY',
            '^NSMIDCP': 'NIFTY MIDCAP',
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
        
        # Sector performance
        st.subheader("📈 Sector Performance")
        
        sector_stocks = {
            'Banking': ['HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'AXISBANK.NS'],
            'IT': ['TCS.NS', 'INFY.NS', 'HCLTECH.NS', 'WIPRO.NS'],
            'Pharma': ['SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS'],
            'Auto': ['MARUTI.NS', 'TATAMOTORS.NS', 'M&M.NS', 'BAJAJ-AUTO.NS'],
            'FMCG': ['HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS']
        }
        
        sector_performance = []
        for sector, stocks in sector_stocks.items():
            total_change = 0
            count = 0
            for stock in stocks:
                try:
                    df = yf.Ticker(stock).history(period="1d")
                    if not df.empty and len(df) > 1:
                        change = ((df.iloc[-1]['Close'] - df.iloc[0]['Open']) / 
                                 df.iloc[0]['Open']) * 100
                        total_change += change
                        count += 1
                except:
                    pass
            
            if count > 0:
                avg_change = total_change / count
                sector_performance.append({
                    'Sector': sector,
                    'Performance': avg_change
                })
        
        if sector_performance:
            df_sectors = pd.DataFrame(sector_performance)
            fig = px.bar(df_sectors, x='Sector', y='Performance',
                        color='Performance',
                        color_continuous_scale=['red', 'yellow', 'green'],
                        title='Sector Performance (Day)')
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
