# app.py - Indian Equity Research Platform - Complete Fixed Version
"""
Indian Equity Research Platform
A Streamlit-based application for analyzing Indian stocks with technical 
and fundamental analysis capabilities, powered by AI insights.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import google.generativeai as genai
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import time
import warnings
import os
import logging
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

# Check if matplotlib is available for styling
try:
    import matplotlib
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# Utility Functions - Define BEFORE they are used
# =============================================================================

def is_cloud_environment() -> bool:
    """
    Detect if running on cloud environment.
    """
    # Check environment variables first (most reliable)
    if os.environ.get('STREAMLIT_SHARING'):
        return True
    if os.environ.get('IS_CLOUD'):
        return True
    if os.environ.get('STREAMLIT_SERVER_HEADLESS'):
        return True
    
    # Safely check Streamlit secrets without causing errors
    try:
        if hasattr(st, 'secrets'):
            try:
                # Check if secrets file exists and has content
                if len(st.secrets) > 0:
                    return bool(st.secrets.get("IS_CLOUD", False))
            except Exception:
                pass
    except Exception:
        pass
    
    return False


def format_value(value: Any, format_type: str = "float", decimals: int = 2) -> str:
    """
    Safely format a numeric value for display.
    
    Args:
        value: The value to format
        format_type: One of "float", "percent", "currency"
        decimals: Number of decimal places
    
    Returns:
        Formatted string or "N/A" if value is invalid
    """
    # Check for invalid values
    if value is None:
        return "N/A"
    
    if isinstance(value, str):
        return value if value else "N/A"
    
    try:
        float_val = float(value)
        
        # Check for NaN or zero (if zero should show as N/A)
        if pd.isna(float_val):
            return "N/A"
        
        if float_val == 0:
            if format_type == "percent":
                return "0.00%"
            return "0.00"
        
        if format_type == "percent":
            return f"{float_val * 100:.{decimals}f}%"
        elif format_type == "currency":
            return format_currency(float_val)
        else:
            return f"{float_val:.{decimals}f}"
            
    except (ValueError, TypeError):
        return "N/A"


def format_currency(value: float, currency: str = "₹") -> str:
    """Format number as currency."""
    if value is None:
        return "N/A"
    
    try:
        float_val = float(value)
        if pd.isna(float_val) or float_val == 0:
            return "N/A"
        
        if abs(float_val) >= 1e12:
            return f"{currency}{float_val/1e12:.2f}T"
        elif abs(float_val) >= 1e9:
            return f"{currency}{float_val/1e9:.2f}B"
        elif abs(float_val) >= 1e6:
            return f"{currency}{float_val/1e6:.2f}M"
        elif abs(float_val) >= 1e3:
            return f"{currency}{float_val/1e3:.2f}K"
        else:
            return f"{currency}{float_val:,.2f}"
    except (ValueError, TypeError):
        return "N/A"


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float."""
    if value is None or value == 'N/A' or value == '':
        return default
    try:
        result = float(value)
        return default if pd.isna(result) else result
    except (ValueError, TypeError):
        return default


def validate_stock_symbol(symbol: str) -> bool:
    """Validate stock symbol format."""
    if not symbol or not isinstance(symbol, str):
        return False
    
    import re
    pattern = r'^[A-Za-z0-9&\-]+\.(NS|BO)$'
    return bool(re.match(pattern, symbol))


def sanitize_symbol_input(user_input: str) -> str:
    """Sanitize user-provided stock symbol."""
    if not user_input:
        return ""
    sanitized = ''.join(c for c in user_input.upper().strip() 
                       if c.isalnum() or c in '&-')
    return sanitized


# Now we can safely use the function
IS_CLOUD = is_cloud_environment()


# =============================================================================
# Page Configuration - Must be first Streamlit command
# =============================================================================

st.set_page_config(
    page_title="Indian Equity Research Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# Constants and Configuration
# =============================================================================

class Config:
    """Application configuration constants."""
    CACHE_TTL_SECONDS = 3600
    CLOUD_DELAY = 0.3
    LOCAL_DELAY = 0.1
    CLOUD_REQUEST_LIMIT = 20
    LOCAL_REQUEST_LIMIT = 50
    RATE_LIMIT_WINDOW = 60
    MAX_RETRIES_CLOUD = 2
    MAX_RETRIES_LOCAL = 3
    INITIAL_RETRY_DELAY = 1
    DEFAULT_CLOUD_LIMIT = 30
    DEFAULT_LOCAL_LIMIT = 50
    MAX_BULK_REPORTS_CLOUD = 10
    CLOUD_DEFAULT_PERIOD = "6mo"
    LOCAL_DEFAULT_PERIOD = "1y"
    SCORE_HIGH = 70
    SCORE_MEDIUM = 40


class StockUniverse(Enum):
    """Enumeration of available stock universes."""
    NIFTY_50 = "NIFTY 50"
    NIFTY_NEXT_50 = "NIFTY NEXT 50"
    NIFTY_100 = "NIFTY 100"
    NIFTY_MIDCAP_100 = "NIFTY MIDCAP 100"
    BANKING = "Banking"
    IT = "IT"
    PHARMA = "Pharma"
    AUTO = "Auto"
    FMCG = "FMCG"
    ALL_SECTORS = "All Sectors"
    CUSTOM = "Custom List"


class ScreenType(Enum):
    """Screening strategy types."""
    QUALITY = "Quality"
    VALUE = "Value"
    MOMENTUM = "Momentum"
    OVERALL = "Overall"


# =============================================================================
# Stock Lists
# =============================================================================

STOCK_LISTS = {
    StockUniverse.NIFTY_50: [
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
    ],
    StockUniverse.NIFTY_NEXT_50: [
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
    ],
    StockUniverse.NIFTY_MIDCAP_100: [
        'AARTIIND.NS', 'ABB.NS', 'ABBOTINDIA.NS', 'ABCAPITAL.NS', 'ABFRL.NS',
        'ACC.NS', 'ADANIPOWER.NS', 'ALKEM.NS', 'APLLTD.NS', 'ASHOKLEY.NS',
        'ASTRAL.NS', 'ATUL.NS', 'AUBANK.NS', 'BAJAJHLDNG.NS', 'BALKRISIND.NS',
        'BALRAMCHIN.NS', 'BATAINDIA.NS', 'BEL.NS', 'BHARATFORG.NS', 'BHEL.NS',
        'BIRLACORPN.NS', 'BLUEDART.NS', 'CAMS.NS', 'CANFINHOME.NS', 'CASTROLIND.NS',
        'CDSL.NS', 'CENTRALBK.NS', 'CENTURYPLY.NS', 'CHAMBLFERT.NS', 'CHOLAHLDNG.NS'
    ],
    StockUniverse.BANKING: [
        'HDFCBANK.NS', 'ICICIBANK.NS', 'AXISBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS',
        'INDUSINDBK.NS', 'BANDHANBNK.NS', 'BANKBARODA.NS', 'PNB.NS', 'CANBK.NS',
        'FEDERALBNK.NS', 'AUBANK.NS', 'IDFCFIRSTB.NS', 'RBLBANK.NS', 'YESBANK.NS'
    ],
    StockUniverse.IT: [
        'TCS.NS', 'INFY.NS', 'HCLTECH.NS', 'WIPRO.NS', 'TECHM.NS',
        'LTI.NS', 'MINDTREE.NS', 'MPHASIS.NS', 'COFORGE.NS', 'PERSISTENT.NS',
        'LTTS.NS', 'ROUTE.NS', 'CYIENT.NS', 'ECLERX.NS', 'LATENTVIEW.NS'
    ],
    StockUniverse.PHARMA: [
        'SUNPHARMA.NS', 'DRREDDY.NS', 'DIVISLAB.NS', 'CIPLA.NS', 'AUROPHARMA.NS',
        'LUPIN.NS', 'TORNTPHARM.NS', 'ALKEM.NS', 'BIOCON.NS', 'GLENMARK.NS',
        'CADILAHC.NS', 'ABBOTINDIA.NS', 'SANOFI.NS', 'PFIZER.NS', 'GLAXO.NS'
    ],
    StockUniverse.AUTO: [
        'MARUTI.NS', 'TATAMOTORS.NS', 'M&M.NS', 'BAJAJ-AUTO.NS', 'EICHERMOT.NS',
        'HEROMOTOCO.NS', 'TVSMOTOR.NS', 'ASHOKLEY.NS', 'BHARATFORG.NS', 'ESCORTS.NS',
        'MOTHERSON.NS', 'BOSCHLTD.NS', 'MRF.NS', 'APOLLOTYRE.NS', 'CEAT.NS'
    ],
    StockUniverse.FMCG: [
        'HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS', 'DABUR.NS',
        'MARICO.NS', 'GODREJCP.NS', 'TATACONSUM.NS', 'COLPAL.NS', 'GILLETTE.NS',
        'PGHH.NS', 'VBL.NS', 'RADICO.NS', 'EMAMILTD.NS', 'BAJAJCON.NS'
    ]
}

MARKET_INDICES = {
    '^NSEI': 'NIFTY 50',
    '^BSESN': 'SENSEX',
    '^NSEBANK': 'BANK NIFTY',
    '^NSMIDCP': 'NIFTY MIDCAP',
}

SECTOR_REPRESENTATIVE_STOCKS = {
    'Banking': ['HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'AXISBANK.NS'],
    'IT': ['TCS.NS', 'INFY.NS', 'HCLTECH.NS', 'WIPRO.NS'],
    'Pharma': ['SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS'],
    'Auto': ['MARUTI.NS', 'TATAMOTORS.NS', 'M&M.NS', 'BAJAJ-AUTO.NS'],
    'FMCG': ['HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS']
}


# =============================================================================
# Custom CSS
# =============================================================================

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


# =============================================================================
# Initialize Session State
# =============================================================================

def initialize_session_state():
    """Initialize session state variables."""
    defaults = {
        'api_key': "",
        'stock_data_cache': {},
        'screening_results': None,
        'last_analysis': None
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


initialize_session_state()


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class FundamentalData:
    """Data class for stock fundamental metrics."""
    market_cap: float = 0.0
    pe_ratio: float = 0.0
    forward_pe: float = 0.0
    pb_ratio: float = 0.0
    dividend_yield: float = 0.0
    roe: float = 0.0
    roa: float = 0.0
    debt_to_equity: float = 0.0
    current_ratio: float = 0.0
    quick_ratio: float = 0.0
    revenue: float = 0.0
    profit_margin: float = 0.0
    operating_margin: float = 0.0
    eps: float = 0.0
    beta: float = 0.0
    week_52_high: float = 0.0
    week_52_low: float = 0.0
    average_volume: float = 0.0
    company_name: str = "N/A"
    sector: str = "N/A"
    industry: str = "N/A"
    
    @classmethod
    def from_yfinance_info(cls, info: Dict) -> 'FundamentalData':
        """Create FundamentalData from yfinance info dict."""
        return cls(
            market_cap=safe_float(info.get('marketCap')),
            pe_ratio=safe_float(info.get('trailingPE')),
            forward_pe=safe_float(info.get('forwardPE')),
            pb_ratio=safe_float(info.get('priceToBook')),
            dividend_yield=safe_float(info.get('dividendYield')),
            roe=safe_float(info.get('returnOnEquity')),
            roa=safe_float(info.get('returnOnAssets')),
            debt_to_equity=safe_float(info.get('debtToEquity')),
            current_ratio=safe_float(info.get('currentRatio')),
            quick_ratio=safe_float(info.get('quickRatio')),
            revenue=safe_float(info.get('totalRevenue')),
            profit_margin=safe_float(info.get('profitMargins')),
            operating_margin=safe_float(info.get('operatingMargins')),
            eps=safe_float(info.get('trailingEps')),
            beta=safe_float(info.get('beta')),
            week_52_high=safe_float(info.get('fiftyTwoWeekHigh')),
            week_52_low=safe_float(info.get('fiftyTwoWeekLow')),
            average_volume=safe_float(info.get('averageVolume')),
            company_name=info.get('longName') or info.get('shortName') or 'N/A',
            sector=info.get('sector') or 'N/A',
            industry=info.get('industry') or 'N/A'
        )
    
    def get_pe_display(self) -> str:
        return format_value(self.pe_ratio)
    
    def get_pb_display(self) -> str:
        return format_value(self.pb_ratio)
    
    def get_roe_display(self) -> str:
        return format_value(self.roe, "percent")
    
    def get_roa_display(self) -> str:
        return format_value(self.roa, "percent")
    
    def get_profit_margin_display(self) -> str:
        return format_value(self.profit_margin, "percent")
    
    def get_operating_margin_display(self) -> str:
        return format_value(self.operating_margin, "percent")
    
    def get_debt_to_equity_display(self) -> str:
        return format_value(self.debt_to_equity)
    
    def get_current_ratio_display(self) -> str:
        return format_value(self.current_ratio)
    
    def get_quick_ratio_display(self) -> str:
        return format_value(self.quick_ratio)
    
    def get_dividend_yield_display(self) -> str:
        return format_value(self.dividend_yield, "percent")
    
    def get_eps_display(self) -> str:
        return format_value(self.eps)
    
    def get_market_cap_display(self) -> str:
        return format_currency(self.market_cap)


@dataclass
class StockScores:
    """Data class for stock scoring metrics."""
    quality: float = 50.0
    value: float = 50.0
    momentum: float = 50.0
    
    @property
    def overall(self) -> float:
        """Calculate overall score."""
        return (self.quality + self.value + self.momentum) / 3


# =============================================================================
# Rate Limiter
# =============================================================================

class RateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(self, is_cloud: bool = False):
        self.is_cloud = is_cloud
        self.request_count = 0
        self.window_start = time.time()
        
    @property
    def delay(self) -> float:
        return Config.CLOUD_DELAY if self.is_cloud else Config.LOCAL_DELAY
    
    @property
    def request_limit(self) -> int:
        return Config.CLOUD_REQUEST_LIMIT if self.is_cloud else Config.LOCAL_REQUEST_LIMIT
    
    def wait(self) -> None:
        """Wait according to rate limiting rules."""
        time.sleep(self.delay)
        
        current_time = time.time()
        
        if current_time - self.window_start > Config.RATE_LIMIT_WINDOW:
            self.request_count = 0
            self.window_start = current_time
        
        self.request_count += 1
        
        if self.request_count > self.request_limit:
            extra_delay = 2 if self.is_cloud else 1
            time.sleep(extra_delay)
            self.request_count = 0
            self.window_start = time.time()


# =============================================================================
# Stock Analyzer
# =============================================================================

class IndianStockAnalyzer:
    """Main class for Indian stock analysis."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.is_cloud = IS_CLOUD
        self.rate_limiter = RateLimiter(self.is_cloud)
        self.model = None
        
        if api_key:
            self._configure_genai(api_key)
    
    def _configure_genai(self, api_key: str) -> None:
        """Configure Google Generative AI."""
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash')
            logger.info("Gemini API configured successfully")
        except Exception as e:
            logger.error(f"Error configuring Gemini API: {e}")
            st.error(f"Error configuring Gemini API: {str(e)}")
    
    @property
    def max_retries(self) -> int:
        return Config.MAX_RETRIES_CLOUD if self.is_cloud else Config.MAX_RETRIES_LOCAL
    
    @property
    def default_period(self) -> str:
        return Config.CLOUD_DEFAULT_PERIOD if self.is_cloud else Config.LOCAL_DEFAULT_PERIOD
    
    def get_stock_universe(self, universe: StockUniverse, limit: Optional[int] = None) -> List[str]:
        """Get stock list based on universe type."""
        if universe == StockUniverse.NIFTY_100:
            stocks = (STOCK_LISTS[StockUniverse.NIFTY_50] + 
                     STOCK_LISTS[StockUniverse.NIFTY_NEXT_50])
        elif universe == StockUniverse.ALL_SECTORS:
            all_sectors = [
                STOCK_LISTS[StockUniverse.BANKING],
                STOCK_LISTS[StockUniverse.IT],
                STOCK_LISTS[StockUniverse.PHARMA],
                STOCK_LISTS[StockUniverse.AUTO],
                STOCK_LISTS[StockUniverse.FMCG]
            ]
            stocks = list(set(stock for sector in all_sectors for stock in sector))
        elif universe in STOCK_LISTS:
            stocks = STOCK_LISTS[universe]
        else:
            stocks = STOCK_LISTS[StockUniverse.NIFTY_50]
        
        if limit and self.is_cloud:
            stocks = stocks[:limit]
        
        return stocks
    
    def fetch_stock_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Fetch stock data with retry logic."""
        if not validate_stock_symbol(symbol):
            logger.warning(f"Invalid symbol format: {symbol}")
            return pd.DataFrame()
        
        for attempt in range(self.max_retries):
            try:
                self.rate_limiter.wait()
                stock = yf.Ticker(symbol)
                data = stock.history(period=period)
                
                if not data.empty:
                    return data
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(Config.INITIAL_RETRY_DELAY * (2 ** attempt))
        
        return pd.DataFrame()
    
    def get_fundamental_data(self, symbol: str) -> Optional[FundamentalData]:
        """Get fundamental data for a stock."""
        if not validate_stock_symbol(symbol):
            return None
        
        for attempt in range(self.max_retries):
            try:
                self.rate_limiter.wait()
                stock = yf.Ticker(symbol)
                info = stock.info
                
                if info:
                    return FundamentalData.from_yfinance_info(info)
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(Config.INITIAL_RETRY_DELAY * (2 ** attempt))
        
        return None
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators on price data."""
        if df.empty:
            return df
        
        df = df.copy()
        
        try:
            # Moving averages
            df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
            df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
            df['SMA_200'] = df['Close'].rolling(window=200, min_periods=1).mean()
            
            # RSI
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / (loss + 1e-10)
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
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
        
        return df
    
    def calculate_quality_score(self, fundamentals: FundamentalData) -> float:
        """Calculate quality score based on fundamentals."""
        score = 0
        max_score = 0
        
        # ROE Score
        if fundamentals.roe > 0:
            max_score += 20
            if fundamentals.roe > 0.20:
                score += 20
            elif fundamentals.roe > 0.15:
                score += 15
            elif fundamentals.roe > 0.10:
                score += 10
        
        # Debt to Equity
        if fundamentals.debt_to_equity >= 0:
            max_score += 20
            if fundamentals.debt_to_equity < 0.5:
                score += 20
            elif fundamentals.debt_to_equity < 1:
                score += 15
            elif fundamentals.debt_to_equity < 1.5:
                score += 10
        
        # Current Ratio
        if fundamentals.current_ratio > 0:
            max_score += 20
            if 1.5 <= fundamentals.current_ratio <= 3:
                score += 20
            elif 1 <= fundamentals.current_ratio < 1.5:
                score += 10
        
        # Profit Margin
        if fundamentals.profit_margin > 0:
            max_score += 20
            if fundamentals.profit_margin > 0.15:
                score += 20
            elif fundamentals.profit_margin > 0.10:
                score += 15
            elif fundamentals.profit_margin > 0.05:
                score += 10
        
        # Operating Margin
        if fundamentals.operating_margin > 0:
            max_score += 20
            if fundamentals.operating_margin > 0.20:
                score += 20
            elif fundamentals.operating_margin > 0.15:
                score += 15
            elif fundamentals.operating_margin > 0.10:
                score += 10
        
        return (score / max_score * 100) if max_score > 0 else 50.0
    
    def calculate_value_score(self, fundamentals: FundamentalData, 
                             current_price: Optional[float] = None) -> float:
        """Calculate value score based on fundamentals."""
        score = 0
        max_score = 0
        
        # PE Ratio
        if fundamentals.pe_ratio > 0:
            max_score += 25
            if fundamentals.pe_ratio < 15:
                score += 25
            elif fundamentals.pe_ratio < 20:
                score += 20
            elif fundamentals.pe_ratio < 25:
                score += 10
        
        # PB Ratio
        if fundamentals.pb_ratio > 0:
            max_score += 25
            if fundamentals.pb_ratio < 1:
                score += 25
            elif fundamentals.pb_ratio < 2:
                score += 20
            elif fundamentals.pb_ratio < 3:
                score += 10
        
        # Dividend Yield
        if fundamentals.dividend_yield > 0:
            max_score += 25
            if fundamentals.dividend_yield > 0.03:
                score += 25
            elif fundamentals.dividend_yield > 0.02:
                score += 20
            elif fundamentals.dividend_yield > 0.01:
                score += 10
        
        # Price position in 52-week range
        if (current_price and fundamentals.week_52_high > 0 and 
            fundamentals.week_52_low > 0 and 
            fundamentals.week_52_high != fundamentals.week_52_low):
            
            max_score += 25
            price_range = fundamentals.week_52_high - fundamentals.week_52_low
            price_position = (current_price - fundamentals.week_52_low) / price_range
            
            if price_position < 0.3:
                score += 25
            elif price_position < 0.5:
                score += 20
            elif price_position < 0.7:
                score += 10
        
        return (score / max_score * 100) if max_score > 0 else 50.0
    
    def calculate_momentum_score(self, df: pd.DataFrame) -> float:
        """Calculate momentum score based on price action."""
        if df.empty or len(df) < 20:
            return 50.0
        
        score = 0
        max_score = 0
        
        try:
            latest = df.iloc[-1]
            
            # Price above moving averages
            if 'SMA_20' in df.columns and pd.notna(latest.get('SMA_20')):
                max_score += 20
                if latest['Close'] > latest['SMA_20']:
                    score += 20
            
            if 'SMA_50' in df.columns and pd.notna(latest.get('SMA_50')):
                max_score += 20
                if latest['Close'] > latest['SMA_50']:
                    score += 20
            
            if 'SMA_200' in df.columns and pd.notna(latest.get('SMA_200')) and len(df) >= 200:
                max_score += 20
                if latest['Close'] > latest['SMA_200']:
                    score += 20
            
            # RSI
            if 'RSI' in df.columns and pd.notna(latest.get('RSI')):
                max_score += 20
                if 40 <= latest['RSI'] <= 70:
                    score += 20
            
            # 1-month returns
            if len(df) > 22:
                max_score += 10
                prev_close = df.iloc[-22]['Close']
                if prev_close > 0:
                    returns_1m = (latest['Close'] - prev_close) / prev_close
                    if returns_1m > 0.05:
                        score += 10
            
            # 3-month returns
            if len(df) > 66:
                max_score += 10
                prev_close = df.iloc[-66]['Close']
                if prev_close > 0:
                    returns_3m = (latest['Close'] - prev_close) / prev_close
                    if returns_3m > 0.15:
                        score += 10
                    
        except Exception as e:
            logger.error(f"Error calculating momentum score: {e}")
        
        return (score / max_score * 100) if max_score > 0 else 50.0
    
    def calculate_all_scores(self, fundamentals: FundamentalData, 
                            df: pd.DataFrame) -> StockScores:
        """Calculate all stock scores."""
        current_price = df.iloc[-1]['Close'] if not df.empty else None
        
        return StockScores(
            quality=self.calculate_quality_score(fundamentals),
            value=self.calculate_value_score(fundamentals, current_price),
            momentum=self.calculate_momentum_score(df)
        )
    
    def generate_ai_report(self, symbol: str, fundamentals: FundamentalData,
                          scores: StockScores) -> str:
        """Generate AI-powered analysis report using Gemini."""
        if not self.model:
            return "Please configure Gemini API key to generate AI reports."
        
        try:
            # Build the prompt with pre-formatted values
            prompt = f"""
Generate a comprehensive equity research report for {fundamentals.company_name} ({symbol}) - an Indian stock.

Fundamental Data:
- Sector: {fundamentals.sector}
- Industry: {fundamentals.industry}
- Market Cap: {fundamentals.get_market_cap_display()}
- PE Ratio: {fundamentals.get_pe_display()}
- PB Ratio: {fundamentals.get_pb_display()}
- ROE: {fundamentals.get_roe_display()}
- Debt to Equity: {fundamentals.get_debt_to_equity_display()}
- Profit Margin: {fundamentals.get_profit_margin_display()}
- Dividend Yield: {fundamentals.get_dividend_yield_display()}

Scores:
- Quality Score: {scores.quality:.2f}/100
- Value Score: {scores.value:.2f}/100
- Momentum Score: {scores.momentum:.2f}/100
- Overall Score: {scores.overall:.2f}/100

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
            logger.error(f"Error generating AI report: {e}")
            return f"Error generating AI report: {str(e)}"
    
    def generate_basic_report(self, symbol: str, fundamentals: FundamentalData,
                             scores: StockScores) -> str:
        """Generate basic report without AI."""
        quality_rating = 'Strong' if scores.quality > Config.SCORE_HIGH else \
                        'Moderate' if scores.quality > Config.SCORE_MEDIUM else 'Weak'
        value_rating = 'Attractive' if scores.value > Config.SCORE_HIGH else \
                      'Fair' if scores.value > Config.SCORE_MEDIUM else 'Expensive'
        momentum_rating = 'Positive' if scores.momentum > Config.SCORE_HIGH else \
                         'Neutral' if scores.momentum > Config.SCORE_MEDIUM else 'Negative'
        
        report = f"""
EQUITY RESEARCH REPORT
======================
Symbol: {symbol}
Company: {fundamentals.company_name}
Sector: {fundamentals.sector}
Date: {datetime.now().strftime('%Y-%m-%d')}

SCORES
------
Quality Score: {scores.quality:.2f}/100
Value Score: {scores.value:.2f}/100
Momentum Score: {scores.momentum:.2f}/100
Overall Score: {scores.overall:.2f}/100

FUNDAMENTAL METRICS
-------------------
PE Ratio: {fundamentals.get_pe_display()}
PB Ratio: {fundamentals.get_pb_display()}
ROE: {fundamentals.get_roe_display()}
Debt to Equity: {fundamentals.get_debt_to_equity_display()}
Profit Margin: {fundamentals.get_profit_margin_display()}

RECOMMENDATION
--------------
Based on the scores, this stock shows:
- Quality: {quality_rating}
- Value: {value_rating}
- Momentum: {momentum_rating}

Note: This is an automated analysis. Please conduct your own research before making investment decisions.
"""
        return report


# =============================================================================
# Stock Screener
# =============================================================================

class StockScreener:
    """Stock screening functionality."""
    
    @staticmethod
    def screen_stocks(
        stocks: List[str],
        analyzer: IndianStockAnalyzer,
        screen_type: ScreenType,
        min_score: float = 0,
        progress_callback: Optional[callable] = None,
        status_callback: Optional[callable] = None
    ) -> pd.DataFrame:
        """Screen stocks based on criteria."""
        results = []
        failed_stocks = []
        total = len(stocks)
        
        if total == 0:
            return pd.DataFrame()
        
        for i, symbol in enumerate(stocks):
            if progress_callback:
                progress_callback((i + 1) / total)
            if status_callback:
                status_callback(f"Analyzing {symbol} ({i+1}/{total})...")
            
            try:
                fundamentals = analyzer.get_fundamental_data(symbol)
                df = analyzer.fetch_stock_data(symbol, period=analyzer.default_period)
                
                if not fundamentals or df.empty:
                    failed_stocks.append(symbol)
                    continue
                
                df = analyzer.calculate_technical_indicators(df)
                scores = analyzer.calculate_all_scores(fundamentals, df)
                
                score_map = {
                    ScreenType.QUALITY: scores.quality,
                    ScreenType.VALUE: scores.value,
                    ScreenType.MOMENTUM: scores.momentum,
                    ScreenType.OVERALL: scores.overall
                }
                score_to_check = score_map.get(screen_type, scores.overall)
                
                if score_to_check < min_score:
                    continue
                
                current_price = df.iloc[-1]['Close']
                change_1d = 0.0
                if len(df) >= 2:
                    prev_price = df.iloc[-2]['Close']
                    if prev_price != 0:
                        change_1d = ((current_price - prev_price) / prev_price) * 100
                
                results.append({
                    'Symbol': symbol.replace('.NS', '').replace('.BO', ''),
                    'Company': (fundamentals.company_name[:30] 
                               if fundamentals.company_name else 'N/A'),
                    'Sector': fundamentals.sector,
                    'Price': round(current_price, 2),
                    'Change %': round(change_1d, 2),
                    'PE Ratio': fundamentals.get_pe_display(),
                    'PB Ratio': fundamentals.get_pb_display(),
                    'ROE %': fundamentals.get_roe_display(),
                    'D/E': fundamentals.get_debt_to_equity_display(),
                    'Quality': round(scores.quality, 1),
                    'Value': round(scores.value, 1),
                    'Momentum': round(scores.momentum, 1),
                    'Overall': round(scores.overall, 1)
                })
                
            except Exception as e:
                logger.error(f"Error screening {symbol}: {e}")
                failed_stocks.append(symbol)
        
        df_results = pd.DataFrame(results)
        
        if df_results.empty:
            return df_results
        
        sort_column = screen_type.value if screen_type.value in df_results.columns else 'Overall'
        df_results = df_results.sort_values(sort_column, ascending=False)
        
        return df_results


# =============================================================================
# UI Components
# =============================================================================

def render_sidebar():
    """Render sidebar configuration."""
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            value=st.session_state.api_key,
            help="Enter your Google Gemini API key for AI-powered reports"
        )
        
        if api_key:
            st.session_state.api_key = api_key
            st.success("✅ API Key configured")
        
        st.markdown("---")
        st.markdown("### 📊 Quick Links")
        st.markdown("- [NSE India](https://www.nseindia.com)")
        st.markdown("- [BSE India](https://www.bseindia.com)")
        st.markdown("- [Moneycontrol](https://www.moneycontrol.com)")


def style_dataframe(df: pd.DataFrame, score_columns: List[str]) -> Any:
    """Apply custom styling to dataframe."""
    
    def color_change(val):
        try:
            if isinstance(val, str) or pd.isna(val):
                return ''
            return 'color: red' if val < 0 else 'color: green' if val > 0 else ''
        except (TypeError, ValueError):
            return ''
    
    def background_gradient(val):
        try:
            if isinstance(val, str) or pd.isna(val):
                return ''
            
            norm_val = max(0, min(1, val / 100))
            
            if norm_val < 0.5:
                r, g = 255, int(255 * (norm_val * 2))
            else:
                r, g = int(255 * (2 - norm_val * 2)), 255
            
            return f'background-color: rgba({r}, {g}, 0, 0.3)'
        except (TypeError, ValueError):
            return ''
    
    styled = df.style
    
    if 'Change %' in df.columns:
        styled = styled.map(color_change, subset=['Change %'])
    
    for col in score_columns:
        if col in df.columns:
            if HAS_MATPLOTLIB:
                try:
                    styled = styled.background_gradient(
                        subset=[col], cmap='RdYlGn', vmin=0, vmax=100
                    )
                except Exception:
                    styled = styled.map(background_gradient, subset=[col])
            else:
                styled = styled.map(background_gradient, subset=[col])
    
    return styled


def create_price_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    """Create price chart with technical indicators."""
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ))
    
    ma_configs = [
        ('SMA_20', 'orange', 'SMA 20'),
        ('SMA_50', 'blue', 'SMA 50'),
        ('SMA_200', 'red', 'SMA 200')
    ]
    
    for col, color, name in ma_configs:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col],
                name=name, line=dict(color=color)
            ))
    
    fig.update_layout(
        title=f"{symbol} Price Chart",
        yaxis_title="Price (₹)",
        xaxis_title="Date",
        height=600,
        xaxis_rangeslider_visible=False
    )
    
    return fig


def create_volume_chart(df: pd.DataFrame) -> go.Figure:
    """Create volume chart."""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume'))
    fig.update_layout(title="Volume", height=200, xaxis_title="Date", yaxis_title="Volume")
    return fig


def create_rsi_chart(df: pd.DataFrame) -> go.Figure:
    """Create RSI chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')))
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    fig.update_layout(title="RSI (14)", height=300, yaxis_title="RSI", xaxis_title="Date")
    return fig


def create_macd_chart(df: pd.DataFrame) -> go.Figure:
    """Create MACD chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], name='Signal', line=dict(color='red')))
    fig.update_layout(title="MACD", height=300, yaxis_title="MACD", xaxis_title="Date")
    return fig


# =============================================================================
# Tab Implementations
# =============================================================================

def render_stock_analysis_tab(analyzer: IndianStockAnalyzer):
    """Render the stock analysis tab."""
    st.header("Individual Stock Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        stock_input = st.text_input(
            "Enter Stock Symbol (e.g., RELIANCE, TCS, INFY)",
            value="RELIANCE"
        )
        exchange = st.selectbox("Exchange", ["NSE", "BSE"])
        
        sanitized_symbol = sanitize_symbol_input(stock_input)
        symbol = f"{sanitized_symbol}.{'NS' if exchange == 'NSE' else 'BO'}"
    
    with col2:
        period = st.selectbox(
            "Time Period",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
            index=3
        )
        analyze_btn = st.button("🔍 Analyze Stock", type="primary", use_container_width=True)
    
    if analyze_btn:
        if not validate_stock_symbol(symbol):
            st.error("Invalid stock symbol format. Please check your input.")
            return
            
        with st.spinner(f"Analyzing {symbol}..."):
            fundamentals = analyzer.get_fundamental_data(symbol)
            df = analyzer.fetch_stock_data(symbol, period=period)
            
            if fundamentals and not df.empty:
                df = analyzer.calculate_technical_indicators(df)
                scores = analyzer.calculate_all_scores(fundamentals, df)
                
                st.subheader(f"📊 {fundamentals.company_name}")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Sector", fundamentals.sector)
                with col2:
                    industry_display = fundamentals.industry[:20] if fundamentals.industry else "N/A"
                    st.metric("Industry", industry_display)
                with col3:
                    current_price = df.iloc[-1]['Close']
                    st.metric("Current Price", f"₹{current_price:.2f}")
                with col4:
                    if len(df) >= 2:
                        prev_price = df.iloc[-2]['Close']
                        if prev_price > 0:
                            change = ((current_price - prev_price) / prev_price) * 100
                            st.metric("Change", f"{change:.2f}%")
                        else:
                            st.metric("Change", "N/A")
                    else:
                        st.metric("Change", "N/A")
                
                analysis_tabs = st.tabs(["Price Chart", "Fundamentals", "Technical", "AI Report"])
                
                with analysis_tabs[0]:
                    st.plotly_chart(create_price_chart(df, symbol), use_container_width=True)
                    st.plotly_chart(create_volume_chart(df), use_container_width=True)
                
                with analysis_tabs[1]:
                    render_fundamentals_tab(fundamentals, scores)
                
                with analysis_tabs[2]:
                    render_technical_tab(df)
                
                with analysis_tabs[3]:
                    render_ai_report_tab(analyzer, symbol, fundamentals, scores)
            else:
                st.error("Unable to fetch data for this symbol. Please check and try again.")


def render_fundamentals_tab(fundamentals: FundamentalData, scores: StockScores):
    """Render fundamentals analysis tab."""
    st.subheader("Fundamental Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Valuation Metrics**")
        st.write(f"PE Ratio: {fundamentals.get_pe_display()}")
        st.write(f"PB Ratio: {fundamentals.get_pb_display()}")
        st.write(f"Market Cap: {fundamentals.get_market_cap_display()}")
        st.write(f"EPS: {fundamentals.get_eps_display()}")
    
    with col2:
        st.markdown("**Profitability Metrics**")
        st.write(f"ROE: {fundamentals.get_roe_display()}")
        st.write(f"ROA: {fundamentals.get_roa_display()}")
        st.write(f"Profit Margin: {fundamentals.get_profit_margin_display()}")
        st.write(f"Operating Margin: {fundamentals.get_operating_margin_display()}")
    
    with col3:
        st.markdown("**Financial Health**")
        st.write(f"Debt to Equity: {fundamentals.get_debt_to_equity_display()}")
        st.write(f"Current Ratio: {fundamentals.get_current_ratio_display()}")
        st.write(f"Quick Ratio: {fundamentals.get_quick_ratio_display()}")
        st.write(f"Dividend Yield: {fundamentals.get_dividend_yield_display()}")
    
    st.markdown("---")
    st.subheader("Stock Scores")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Quality Score", f"{scores.quality:.1f}/100")
        st.progress(scores.quality / 100)
    with col2:
        st.metric("Value Score", f"{scores.value:.1f}/100")
        st.progress(scores.value / 100)
    with col3:
        st.metric("Momentum Score", f"{scores.momentum:.1f}/100")
        st.progress(scores.momentum / 100)


def render_technical_tab(df: pd.DataFrame):
    """Render technical analysis tab."""
    st.subheader("Technical Indicators")
    
    if 'RSI' in df.columns:
        st.plotly_chart(create_rsi_chart(df), use_container_width=True)
    
    if 'MACD' in df.columns and 'Signal' in df.columns:
        st.plotly_chart(create_macd_chart(df), use_container_width=True)
    
    st.markdown("---")
    st.subheader("Current Technical Readings")
    
    if df.empty:
        st.warning("No data available for technical analysis")
        return
        
    latest = df.iloc[-1]
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        rsi_val = latest.get('RSI')
        if pd.notna(rsi_val):
            st.metric("RSI", f"{rsi_val:.2f}")
    with col2:
        macd_val = latest.get('MACD')
        if pd.notna(macd_val):
            st.metric("MACD", f"{macd_val:.2f}")
    with col3:
        signal_val = latest.get('Signal')
        if pd.notna(signal_val):
            st.metric("Signal", f"{signal_val:.2f}")
    with col4:
        macd_val = latest.get('MACD')
        signal_val = latest.get('Signal')
        if pd.notna(macd_val) and pd.notna(signal_val):
            macd_signal = "Bullish" if macd_val > signal_val else "Bearish"
            st.metric("MACD Signal", macd_signal)


def render_ai_report_tab(analyzer: IndianStockAnalyzer, symbol: str,
                        fundamentals: FundamentalData, scores: StockScores):
    """Render AI report tab."""
    st.subheader("AI-Generated Equity Research Report")
    
    if st.session_state.api_key and analyzer.model:
        with st.spinner("Generating AI report..."):
            report = analyzer.generate_ai_report(symbol, fundamentals, scores)
            st.markdown(report)
            
            st.download_button(
                label="📥 Download Report",
                data=report,
                file_name=f"{symbol}_research_report_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )
    else:
        st.warning("Please configure Gemini API key in the sidebar to generate AI reports.")
        st.info("Generating basic report instead...")
        basic_report = analyzer.generate_basic_report(symbol, fundamentals, scores)
        st.text(basic_report)


def render_screener_tab(analyzer: IndianStockAnalyzer):
    """Render stock screener tab."""
    st.header("Stock Screener")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        col1_1, col1_2, col1_3 = st.columns(3)
        
        with col1_1:
            screen_type_options = [st.value for st in ScreenType]
            screen_type = st.selectbox("Screening Strategy", screen_type_options)
            screen_type_enum = ScreenType(screen_type)
        
        with col1_2:
            if IS_CLOUD:
                universe_options = [
                    StockUniverse.NIFTY_50.value,
                    StockUniverse.BANKING.value,
                    StockUniverse.IT.value,
                    StockUniverse.PHARMA.value,
                    StockUniverse.AUTO.value,
                    StockUniverse.FMCG.value,
                    StockUniverse.CUSTOM.value
                ]
                default_limit = Config.DEFAULT_CLOUD_LIMIT
            else:
                universe_options = [u.value for u in StockUniverse]
                default_limit = Config.DEFAULT_LOCAL_LIMIT
            
            stock_universe = st.selectbox("Stock Universe", universe_options)
        
        with col1_3:
            min_score = st.slider("Minimum Score Filter", 0, 100, 50, 10)
    
    with col2:
        st.markdown("### Quick Stats")
        if stock_universe != StockUniverse.CUSTOM.value:
            universe_enum = StockUniverse(stock_universe)
            limit = 50 if IS_CLOUD and stock_universe in ["NIFTY 100", "All Sectors"] else None
            stocks_count = len(analyzer.get_stock_universe(universe_enum, limit=limit))
            st.info(f"📊 {stocks_count} stocks in {stock_universe}")
    
    custom_stocks = ""
    if stock_universe == StockUniverse.CUSTOM.value:
        custom_stocks = st.text_area(
            "Enter symbols (comma-separated)",
            "RELIANCE,TCS,INFY,HDFCBANK,ICICIBANK,WIPRO,BHARTIARTL,SBIN",
            height=100
        )
    
    with st.expander("Advanced Filters"):
        col1, col2, col3 = st.columns(3)
        with col1:
            max_results = st.number_input(
                "Max Results",
                min_value=10,
                max_value=100 if IS_CLOUD else 200,
                value=default_limit
            )
        with col2:
            sort_by = st.selectbox("Sort By", ["Score", "PE Ratio", "PB Ratio", "ROE %"])
        with col3:
            ascending = st.checkbox("Ascending Order", value=False)
    
    if st.button("🔍 Run Screener", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        if stock_universe == StockUniverse.CUSTOM.value:
            stocks = [f"{sanitize_symbol_input(s)}.NS" 
                     for s in custom_stocks.split(',') if s.strip()]
        else:
            universe_enum = StockUniverse(stock_universe)
            limit = max_results if IS_CLOUD else None
            stocks = analyzer.get_stock_universe(universe_enum, limit=limit)
        
        if not stocks:
            st.warning("No stocks to screen. Please check your input.")
            return
            
        status_text.text(f"Screening {len(stocks)} stocks...")
        
        results = StockScreener.screen_stocks(
            stocks, analyzer, screen_type_enum, min_score,
            progress_callback=progress_bar.progress,
            status_callback=status_text.text
        )
        
        progress_bar.empty()
        status_text.empty()
        
        if not results.empty:
            results = results.head(max_results)
            
            st.success(f"✅ Found {len(results)} stocks matching criteria (Score ≥ {min_score})")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Avg Quality Score", f"{results['Quality'].mean():.1f}")
            with col2:
                st.metric("Avg Value Score", f"{results['Value'].mean():.1f}")
            with col3:
                st.metric("Avg Momentum Score", f"{results['Momentum'].mean():.1f}")
            with col4:
                st.metric("Avg Overall Score", f"{results['Overall'].mean():.1f}")
            
            st.subheader(f"📊 Screening Results - Top {len(results)} Stocks")
            styled_df = style_dataframe(results, ['Quality', 'Value', 'Momentum', 'Overall'])
            st.dataframe(styled_df, use_container_width=True, height=600)
            
            st.download_button(
                label="📥 Download Results (CSV)",
                data=results.to_csv(index=False),
                file_name=f"{stock_universe}_{screen_type}_screening_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            
            st.subheader("🏆 Top 10 Picks")
            top_10 = results.head(10)
            fig = px.bar(
                top_10, x='Symbol', y=screen_type,
                color=screen_type,
                color_continuous_scale='RdYlGn',
                title=f'Top 10 Stocks by {screen_type} Score'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No stocks found matching criteria. Try adjusting your filters.")


def render_portfolio_tab(analyzer: IndianStockAnalyzer):
    """Render portfolio analysis tab."""
    st.header("Portfolio Analysis")
    
    st.info("Portfolio analysis allows you to track multiple stocks and analyze overall performance.")
    
    portfolio_text = st.text_area(
        "Enter your portfolio (Symbol, Quantity, Buy Price)",
        "RELIANCE, 100, 2400\nTCS, 50, 3500\nINFY, 75, 1400\nHDFCBANK, 25, 1600\nICICIBANK, 100, 950"
    )
    
    if st.button("Analyze Portfolio"):
        portfolio = []
        errors = []
        
        for line_num, line in enumerate(portfolio_text.strip().split('\n'), 1):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) != 3:
                errors.append(f"Line {line_num}: Invalid format")
                continue
            
            try:
                symbol = sanitize_symbol_input(parts[0])
                quantity = int(parts[1])
                buy_price = float(parts[2])
                
                if quantity <= 0 or buy_price <= 0:
                    errors.append(f"Line {line_num}: Values must be positive")
                    continue
                    
                portfolio.append({
                    'Symbol': f"{symbol}.NS",
                    'Quantity': quantity,
                    'Buy Price': buy_price
                })
            except ValueError:
                errors.append(f"Line {line_num}: Invalid number format")
        
        for error in errors:
            st.warning(error)
        
        if portfolio:
            results = []
            total_invested = 0
            total_current = 0
            
            with st.spinner("Analyzing portfolio..."):
                for stock in portfolio:
                    df = analyzer.fetch_stock_data(stock['Symbol'], period="1d")
                    
                    if not df.empty:
                        current_price = df.iloc[-1]['Close']
                        invested = stock['Quantity'] * stock['Buy Price']
                        current_value = stock['Quantity'] * current_price
                        pnl = current_value - invested
                        pnl_percent = (pnl / invested) * 100 if invested != 0 else 0
                        
                        results.append({
                            'Symbol': stock['Symbol'].replace('.NS', ''),
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
            
            if results:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Invested", f"₹{total_invested:,.2f}")
                with col2:
                    st.metric("Current Value", f"₹{total_current:,.2f}")
                with col3:
                    total_pnl = total_current - total_invested
                    pnl_pct = (total_pnl / total_invested * 100) if total_invested != 0 else 0
                    st.metric("Total P&L", f"₹{total_pnl:,.2f}", delta=f"{pnl_pct:.2f}%")
                with col4:
                    st.metric("Portfolio Return", f"{pnl_pct:.2f}%")
                
                st.subheader("Holdings")
                df_portfolio = pd.DataFrame(results)
                
                def style_pnl(val):
                    try:
                        if isinstance(val, str) or pd.isna(val):
                            return ''
                        return 'color: green' if val > 0 else 'color: red' if val < 0 else ''
                    except Exception:
                        return ''
                
                styled = df_portfolio.style.map(style_pnl, subset=['P&L', 'P&L %'])
                st.dataframe(styled, use_container_width=True)
                
                st.subheader("Portfolio Composition")
                fig = px.pie(df_portfolio, values='Current Value', names='Symbol',
                           title='Portfolio Allocation')
                st.plotly_chart(fig, use_container_width=True)


def render_bulk_reports_tab(analyzer: IndianStockAnalyzer):
    """Render bulk reports tab."""
    st.header("Bulk Report Generation")
    
    st.info("Generate comprehensive research reports for multiple stocks at once.")
    
    default_stocks = "RELIANCE,TCS,INFY,HDFCBANK,ICICIBANK" if IS_CLOUD else \
                    "RELIANCE,TCS,INFY,HDFCBANK,ICICIBANK,WIPRO,BHARTIARTL,SBIN,KOTAKBANK,AXISBANK"
    
    report_stocks = st.text_area("Enter stock symbols (comma-separated)", default_stocks)
    
    if st.button("Generate Bulk Reports"):
        stocks = [f"{sanitize_symbol_input(s)}.NS" 
                 for s in report_stocks.split(',') if s.strip()]
        
        if IS_CLOUD and len(stocks) > Config.MAX_BULK_REPORTS_CLOUD:
            st.warning(f"Limiting to {Config.MAX_BULK_REPORTS_CLOUD} stocks for cloud deployment")
            stocks = stocks[:Config.MAX_BULK_REPORTS_CLOUD]
        
        if not stocks:
            st.warning("No valid stock symbols provided.")
            return
        
        progress_bar = st.progress(0)
        reports = {}
        
        for i, symbol in enumerate(stocks):
            progress_bar.progress((i + 1) / len(stocks))
            
            with st.spinner(f"Analyzing {symbol}..."):
                fundamentals = analyzer.get_fundamental_data(symbol)
                df = analyzer.fetch_stock_data(symbol, period=analyzer.default_period)
                
                if fundamentals and not df.empty:
                    df = analyzer.calculate_technical_indicators(df)
                    scores = analyzer.calculate_all_scores(fundamentals, df)
                    
                    if st.session_state.api_key and analyzer.model:
                        report = analyzer.generate_ai_report(symbol, fundamentals, scores)
                    else:
                        report = analyzer.generate_basic_report(symbol, fundamentals, scores)
                    
                    reports[symbol] = report
        
        progress_bar.empty()
        
        if reports:
            for symbol, report in reports.items():
                with st.expander(f"Report: {symbol.replace('.NS', '')}"):
                    st.text(report)
            
            all_reports = "\n\n" + "="*80 + "\n\n".join(
                [f"REPORT FOR {symbol}\n{report}" for symbol, report in reports.items()]
            )
            
            st.download_button(
                label="📥 Download All Reports",
                data=all_reports,
                file_name=f"bulk_reports_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )
        else:
            st.warning("No reports could be generated.")


def render_market_overview_tab(analyzer: IndianStockAnalyzer):
    """Render market overview tab."""
    st.header("Market Overview")
    
    st.subheader("📊 Major Indices")
    
    cols = st.columns(len(MARKET_INDICES))
    for i, (symbol, name) in enumerate(MARKET_INDICES.items()):
        with cols[i]:
            try:
                index_data = yf.Ticker(symbol).history(period="2d")
                if not index_data.empty and len(index_data) >= 2:
                    current = index_data.iloc[-1]['Close']
                    prev = index_data.iloc[-2]['Close']
                    change = ((current - prev) / prev) * 100 if prev != 0 else 0
                    st.metric(name, f"{current:,.2f}", delta=f"{change:.2f}%")
                else:
                    st.metric(name, "N/A")
            except Exception as e:
                logger.warning(f"Failed to fetch index {symbol}: {e}")
                st.metric(name, "N/A")
    
    st.subheader("📈 Sector Performance")
    
    sector_performance = []
    
    with st.spinner("Loading sector data..."):
        for sector, stocks in SECTOR_REPRESENTATIVE_STOCKS.items():
            total_change = 0
            count = 0
            
            for stock in stocks:
                try:
                    df = yf.Ticker(stock).history(period="2d")
                    if not df.empty and len(df) >= 2:
                        prev_close = df.iloc[-2]['Close']
                        if prev_close > 0:
                            change = ((df.iloc[-1]['Close'] - prev_close) / prev_close) * 100
                            total_change += change
                            count += 1
                except Exception as e:
                    logger.debug(f"Failed to fetch {stock}: {e}")
            
            if count > 0:
                sector_performance.append({
                    'Sector': sector,
                    'Performance': total_change / count
                })
    
    if sector_performance:
        df_sectors = pd.DataFrame(sector_performance)
        fig = px.bar(
            df_sectors, x='Sector', y='Performance',
            color='Performance',
            color_continuous_scale=['red', 'yellow', 'green'],
            title='Sector Performance (Day)'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Unable to load sector performance data")


# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main application entry point."""
    st.markdown(
        '<h1 class="main-header">🏛️ Indian Equity Research Platform</h1>',
        unsafe_allow_html=True
    )
    
    if IS_CLOUD:
        st.info("☁️ Running on Streamlit Cloud - Optimized performance mode")
    else:
        st.info("💻 Running Locally - Full feature mode")
    
    render_sidebar()
    
    analyzer = IndianStockAnalyzer(api_key=st.session_state.api_key)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Stock Analysis",
        "🔍 Stock Screener",
        "📈 Portfolio Analysis",
        "📑 Bulk Reports",
        "📉 Market Overview"
    ])
    
    with tab1:
        render_stock_analysis_tab(analyzer)
    
    with tab2:
        render_screener_tab(analyzer)
    
    with tab3:
        render_portfolio_tab(analyzer)
    
    with tab4:
        render_bulk_reports_tab(analyzer)
    
    with tab5:
        render_market_overview_tab(analyzer)


if __name__ == "__main__":
    main()
