#!/usr/bin/env python3
"""
Indian Equity Research Platform - Production Grade
===================================================
A Streamlit-based application for analyzing Indian stocks with technical
and fundamental analysis capabilities, powered by AI insights.

Architecture:
    - Unified data fetching (single Ticker per stock)
    - TTL-based caching layer
    - Thread-safe rate limiting
    - Proper RSI (Wilder's smoothing)
    - Fixed-denominator scoring with data coverage penalties
    - Portfolio risk analytics (VaR, CVaR, Sharpe, Max Drawdown)
    - Comprehensive input validation and data cleaning
    - Result type pattern for error handling

Author: Refactored for production use
"""

# =============================================================================
# Imports
# =============================================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import google.generativeai as genai
from typing import (
    Dict, List, Tuple, Optional, Any, Union, Callable, Generic, TypeVar
)
from dataclasses import dataclass, field
from enum import Enum
import time
import threading
import warnings
import os
import re
import html
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Logging & warnings
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Optional matplotlib check (used for dataframe gradient styling)
# ---------------------------------------------------------------------------
try:
    import matplotlib  # noqa: F401
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# Result Type – explicit success / failure wrapper
# =============================================================================

T = TypeVar("T")


@dataclass
class Result(Generic[T]):
    """Monadic result wrapper to unify error handling."""

    value: Optional[T] = None
    error: Optional[str] = None

    @property
    def is_ok(self) -> bool:
        return self.error is None and self.value is not None

    @classmethod
    def ok(cls, value: T) -> "Result[T]":
        return cls(value=value)

    @classmethod
    def fail(cls, error: str) -> "Result[T]":
        return cls(error=error)


# =============================================================================
# Environment detection (must come before any st.* calls except set_page_config)
# =============================================================================

def _is_cloud_environment() -> bool:
    """Detect if running on Streamlit Cloud or similar hosted environment."""
    for key in ("STREAMLIT_SHARING", "IS_CLOUD", "STREAMLIT_SERVER_HEADLESS"):
        if os.environ.get(key):
            return True
    try:
        if hasattr(st, "secrets") and len(st.secrets) > 0:
            return bool(st.secrets.get("IS_CLOUD", False))
    except Exception:
        pass
    return False


IS_CLOUD: bool = _is_cloud_environment()


# =============================================================================
# Page configuration – MUST be the first Streamlit command
# =============================================================================

st.set_page_config(
    page_title="Indian Equity Research Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# Configuration constants
# =============================================================================

class Config:
    """Immutable application-wide configuration."""

    # Cache
    CACHE_TTL_SECONDS: int = 3600

    # Rate limiting
    CLOUD_DELAY: float = 0.3
    LOCAL_DELAY: float = 0.1
    CLOUD_REQUEST_LIMIT: int = 20
    LOCAL_REQUEST_LIMIT: int = 50
    RATE_LIMIT_WINDOW: int = 60

    # Retries
    MAX_RETRIES_CLOUD: int = 2
    MAX_RETRIES_LOCAL: int = 3
    INITIAL_RETRY_DELAY: float = 1.0

    # Screener defaults
    DEFAULT_CLOUD_LIMIT: int = 30
    DEFAULT_LOCAL_LIMIT: int = 50
    MAX_BULK_REPORTS_CLOUD: int = 10
    CLOUD_DEFAULT_PERIOD: str = "6mo"
    LOCAL_DEFAULT_PERIOD: str = "1y"

    # Scoring display thresholds
    SCORE_HIGH: float = 70.0
    SCORE_MEDIUM: float = 40.0

    # Threading
    MAX_WORKERS_LOCAL: int = 5
    MAX_WORKERS_CLOUD: int = 1

    # Portfolio validation
    MAX_QUANTITY: int = 10_000_000
    MAX_PRICE: float = 1_000_000.0
    MIN_PRICE: float = 0.01

    # Risk-free rate (India – approx RBI repo)
    RISK_FREE_RATE: float = 0.065
    TRADING_DAYS: int = 252


# =============================================================================
# Scoring thresholds – centralised for easy calibration
# =============================================================================

class Thresholds:
    """Named thresholds so every magic number has a home."""

    # Quality
    ROE_EXCELLENT = 0.20
    ROE_GOOD = 0.15
    ROE_ACCEPTABLE = 0.10

    DEBT_EQUITY_LOW = 0.5
    DEBT_EQUITY_MODERATE = 1.0
    DEBT_EQUITY_HIGH = 1.5

    CURRENT_RATIO_IDEAL_LOW = 1.5
    CURRENT_RATIO_IDEAL_HIGH = 3.0
    CURRENT_RATIO_ACCEPTABLE = 1.0

    PROFIT_MARGIN_EXCELLENT = 0.15
    PROFIT_MARGIN_GOOD = 0.10
    PROFIT_MARGIN_ACCEPTABLE = 0.05

    OPERATING_MARGIN_EXCELLENT = 0.20
    OPERATING_MARGIN_GOOD = 0.15
    OPERATING_MARGIN_ACCEPTABLE = 0.10

    # Value
    PE_CHEAP = 15
    PE_FAIR = 20
    PE_MODERATE = 25

    PB_CHEAP = 1.0
    PB_FAIR = 2.0
    PB_MODERATE = 3.0

    DIVIDEND_YIELD_HIGH = 0.03
    DIVIDEND_YIELD_MODERATE = 0.02
    DIVIDEND_YIELD_LOW = 0.01

    PRICE_POSITION_LOW = 0.3
    PRICE_POSITION_MID = 0.5
    PRICE_POSITION_HIGH = 0.7

    # Momentum
    RSI_OVERSOLD = 30
    RSI_SWEET_LOW = 40
    RSI_SWEET_HIGH = 65
    RSI_OVERBOUGHT = 75

    RETURN_1M_STRONG = 0.05
    RETURN_3M_STRONG = 0.10
    RETURN_6M_STRONG = 0.15

    # Outlier detection
    EXTREME_DAILY_RETURN = 0.50  # 50 %


# =============================================================================
# Enums
# =============================================================================

class StockUniverse(Enum):
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
    QUALITY = "Quality"
    VALUE = "Value"
    MOMENTUM = "Momentum"
    OVERALL = "Overall"


# =============================================================================
# Stock lists
# =============================================================================

STOCK_LISTS: Dict[StockUniverse, List[str]] = {
    StockUniverse.NIFTY_50: [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
        "LT.NS", "HCLTECH.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS",
        "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "BAJFINANCE.NS", "WIPRO.NS",
        "NESTLEIND.NS", "ADANIENT.NS", "POWERGRID.NS", "M&M.NS", "NTPC.NS",
        "TATAMOTORS.NS", "ONGC.NS", "JSWSTEEL.NS", "TATASTEEL.NS", "HDFC.NS",
        "BAJAJFINSV.NS", "COALINDIA.NS", "GRASIM.NS", "TECHM.NS", "INDUSINDBK.NS",
        "HINDALCO.NS", "DRREDDY.NS", "DIVISLAB.NS", "CIPLA.NS", "SBILIFE.NS",
        "BRITANNIA.NS", "EICHERMOT.NS", "BPCL.NS", "SHREECEM.NS", "HEROMOTOCO.NS",
        "UPL.NS", "TATACONSUM.NS", "APOLLOHOSP.NS", "ADANIPORTS.NS", "BAJAJ-AUTO.NS",
    ],
    StockUniverse.NIFTY_NEXT_50: [
        "ADANIGREEN.NS", "ADANITRANS.NS", "AMBUJACEM.NS", "ATGL.NS", "AUROPHARMA.NS",
        "BANDHANBNK.NS", "BANKBARODA.NS", "BERGEPAINT.NS", "BIOCON.NS", "BOSCHLTD.NS",
        "CANBK.NS", "CHOLAFIN.NS", "COLPAL.NS", "DLF.NS", "DABUR.NS",
        "DMART.NS", "GAIL.NS", "GODREJCP.NS", "HAVELLS.NS", "HDFCLIFE.NS",
        "ICICIGI.NS", "ICICIPRULI.NS", "IGL.NS", "INDIGO.NS", "INDUSTOWER.NS",
        "JINDALSTEL.NS", "JUBLFOOD.NS", "LTI.NS", "LUPIN.NS", "MARICO.NS",
        "MCDOWELL-N.NS", "MUTHOOTFIN.NS", "NAUKRI.NS", "PEL.NS", "PIDILITIND.NS",
        "PNB.NS", "SBICARD.NS", "SIEMENS.NS", "TATAPOWER.NS", "TORNTPHARM.NS",
        "TRENT.NS", "TVSMOTOR.NS", "VEDL.NS", "ZEEL.NS", "GODREJPROP.NS",
        "ICICIPRU.NS", "INDIANB.NS", "YESBANK.NS", "ZOMATO.NS", "PAYTM.NS",
    ],
    StockUniverse.NIFTY_MIDCAP_100: [
        "AARTIIND.NS", "ABB.NS", "ABBOTINDIA.NS", "ABCAPITAL.NS", "ABFRL.NS",
        "ACC.NS", "ADANIPOWER.NS", "ALKEM.NS", "APLLTD.NS", "ASHOKLEY.NS",
        "ASTRAL.NS", "ATUL.NS", "AUBANK.NS", "BAJAJHLDNG.NS", "BALKRISIND.NS",
        "BALRAMCHIN.NS", "BATAINDIA.NS", "BEL.NS", "BHARATFORG.NS", "BHEL.NS",
        "BIRLACORPN.NS", "BLUEDART.NS", "CAMS.NS", "CANFINHOME.NS", "CASTROLIND.NS",
        "CDSL.NS", "CENTRALBK.NS", "CENTURYPLY.NS", "CHAMBLFERT.NS", "CHOLAHLDNG.NS",
    ],
    StockUniverse.BANKING: [
        "HDFCBANK.NS", "ICICIBANK.NS", "AXISBANK.NS", "SBIN.NS", "KOTAKBANK.NS",
        "INDUSINDBK.NS", "BANDHANBNK.NS", "BANKBARODA.NS", "PNB.NS", "CANBK.NS",
        "FEDERALBNK.NS", "AUBANK.NS", "IDFCFIRSTB.NS", "RBLBANK.NS", "YESBANK.NS",
    ],
    StockUniverse.IT: [
        "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS",
        "LTI.NS", "MINDTREE.NS", "MPHASIS.NS", "COFORGE.NS", "PERSISTENT.NS",
        "LTTS.NS", "ROUTE.NS", "CYIENT.NS", "ECLERX.NS", "LATENTVIEW.NS",
    ],
    StockUniverse.PHARMA: [
        "SUNPHARMA.NS", "DRREDDY.NS", "DIVISLAB.NS", "CIPLA.NS", "AUROPHARMA.NS",
        "LUPIN.NS", "TORNTPHARM.NS", "ALKEM.NS", "BIOCON.NS", "GLENMARK.NS",
        "CADILAHC.NS", "ABBOTINDIA.NS", "SANOFI.NS", "PFIZER.NS", "GLAXO.NS",
    ],
    StockUniverse.AUTO: [
        "MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS",
        "HEROMOTOCO.NS", "TVSMOTOR.NS", "ASHOKLEY.NS", "BHARATFORG.NS", "ESCORTS.NS",
        "MOTHERSON.NS", "BOSCHLTD.NS", "MRF.NS", "APOLLOTYRE.NS", "CEAT.NS",
    ],
    StockUniverse.FMCG: [
        "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS",
        "MARICO.NS", "GODREJCP.NS", "TATACONSUM.NS", "COLPAL.NS", "GILLETTE.NS",
        "PGHH.NS", "VBL.NS", "RADICO.NS", "EMAMILTD.NS", "BAJAJCON.NS",
    ],
}

MARKET_INDICES: Dict[str, str] = {
    "^NSEI": "NIFTY 50",
    "^BSESN": "SENSEX",
    "^NSEBANK": "BANK NIFTY",
    "^NSMIDCP": "NIFTY MIDCAP",
}

SECTOR_REPRESENTATIVE_STOCKS: Dict[str, List[str]] = {
    "Banking": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS"],
    "IT": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS"],
    "Pharma": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS"],
    "Auto": ["MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS"],
    "FMCG": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS"],
}


# =============================================================================
# Utility functions
# =============================================================================

def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert *value* to float, returning *default* on failure."""
    if value is None or value == "N/A" or value == "":
        return default
    try:
        result = float(value)
        return default if pd.isna(result) else result
    except (ValueError, TypeError):
        return default


def safe_optional_float(value: Any) -> Optional[float]:
    """Return ``None`` when data is genuinely missing vs zero when it is zero."""
    if value is None:
        return None
    try:
        fval = float(value)
        return None if pd.isna(fval) else fval
    except (ValueError, TypeError):
        return None


def format_value(
    value: Any,
    format_type: str = "float",
    decimals: int = 2,
) -> str:
    """Human-readable formatting with N/A fallback."""
    if value is None:
        return "N/A"
    if isinstance(value, str):
        return value if value else "N/A"
    try:
        fv = float(value)
        if pd.isna(fv):
            return "N/A"
        if format_type == "percent":
            return f"{fv * 100:.{decimals}f}%"
        if format_type == "currency":
            return format_currency(fv)
        return f"{fv:.{decimals}f}"
    except (ValueError, TypeError):
        return "N/A"


def format_currency(value: Optional[float], currency: str = "₹") -> str:
    """Format number as abbreviated currency string."""
    if value is None:
        return "N/A"
    try:
        fv = float(value)
        if pd.isna(fv):
            return "N/A"
        abs_fv = abs(fv)
        if abs_fv >= 1e12:
            return f"{currency}{fv / 1e12:.2f}T"
        if abs_fv >= 1e9:
            return f"{currency}{fv / 1e9:.2f}B"
        if abs_fv >= 1e6:
            return f"{currency}{fv / 1e6:.2f}M"
        if abs_fv >= 1e3:
            return f"{currency}{fv / 1e3:.2f}K"
        return f"{currency}{fv:,.2f}"
    except (ValueError, TypeError):
        return "N/A"


_SYMBOL_RE = re.compile(r"^[A-Za-z0-9&\-]+\.(NS|BO)$")


def validate_stock_symbol(symbol: str) -> bool:
    if not symbol or not isinstance(symbol, str):
        return False
    return bool(_SYMBOL_RE.match(symbol))


def sanitize_symbol_input(user_input: str) -> str:
    if not user_input:
        return ""
    return "".join(c for c in user_input.upper().strip() if c.isalnum() or c in "&-")


def sanitize_ai_output(text: str) -> str:
    """Escape dangerous HTML while preserving markdown formatting."""
    sanitized = html.escape(text)
    # Restore markdown-safe characters
    for escaped, original in (("&amp;", "&"), ("&#x27;", "'")):
        sanitized = sanitized.replace(escaped, original)
    return sanitized


def _display(val: Optional[float], fmt: str = "float", decimals: int = 2) -> str:
    """Shorthand for format_value on Optional floats."""
    if val is None:
        return "N/A"
    return format_value(val, fmt, decimals)


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class FundamentalData:
    """Stock fundamental metrics – uses ``None`` for genuinely missing data."""

    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    pb_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    revenue: Optional[float] = None
    profit_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    eps: Optional[float] = None
    beta: Optional[float] = None
    week_52_high: Optional[float] = None
    week_52_low: Optional[float] = None
    average_volume: Optional[float] = None
    company_name: str = "N/A"
    sector: str = "N/A"
    industry: str = "N/A"

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def from_yfinance_info(cls, info: Dict) -> "FundamentalData":
        _e = safe_optional_float
        return cls(
            market_cap=_e(info.get("marketCap")),
            pe_ratio=_e(info.get("trailingPE")),
            forward_pe=_e(info.get("forwardPE")),
            pb_ratio=_e(info.get("priceToBook")),
            dividend_yield=_e(info.get("dividendYield")),
            roe=_e(info.get("returnOnEquity")),
            roa=_e(info.get("returnOnAssets")),
            debt_to_equity=_e(info.get("debtToEquity")),
            current_ratio=_e(info.get("currentRatio")),
            quick_ratio=_e(info.get("quickRatio")),
            revenue=_e(info.get("totalRevenue")),
            profit_margin=_e(info.get("profitMargins")),
            operating_margin=_e(info.get("operatingMargins")),
            eps=_e(info.get("trailingEps")),
            beta=_e(info.get("beta")),
            week_52_high=_e(info.get("fiftyTwoWeekHigh")),
            week_52_low=_e(info.get("fiftyTwoWeekLow")),
            average_volume=_e(info.get("averageVolume")),
            company_name=info.get("longName") or info.get("shortName") or "N/A",
            sector=info.get("sector") or "N/A",
            industry=info.get("industry") or "N/A",
        )

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------
    def get_pe_display(self) -> str:
        return _display(self.pe_ratio)

    def get_pb_display(self) -> str:
        return _display(self.pb_ratio)

    def get_roe_display(self) -> str:
        return _display(self.roe, "percent")

    def get_roa_display(self) -> str:
        return _display(self.roa, "percent")

    def get_profit_margin_display(self) -> str:
        return _display(self.profit_margin, "percent")

    def get_operating_margin_display(self) -> str:
        return _display(self.operating_margin, "percent")

    def get_debt_to_equity_display(self) -> str:
        return _display(self.debt_to_equity)

    def get_current_ratio_display(self) -> str:
        return _display(self.current_ratio)

    def get_quick_ratio_display(self) -> str:
        return _display(self.quick_ratio)

    def get_dividend_yield_display(self) -> str:
        return _display(self.dividend_yield, "percent")

    def get_eps_display(self) -> str:
        return _display(self.eps)

    def get_market_cap_display(self) -> str:
        return format_currency(self.market_cap)


@dataclass
class StockScores:
    """Composite stock score card."""

    quality: float = 50.0
    value: float = 50.0
    momentum: float = 50.0
    data_coverage: float = 1.0  # fraction of metrics that had data

    @property
    def overall(self) -> float:
        return (self.quality + self.value + self.momentum) / 3.0


@dataclass
class StockDataBundle:
    """Single-fetch container: one Ticker call → info + history."""

    symbol: str
    info: Dict
    history: pd.DataFrame
    fundamentals: Optional[FundamentalData] = None

    @property
    def is_valid(self) -> bool:
        return bool(self.info) and not self.history.empty


@dataclass
class PortfolioRiskMetrics:
    """Portfolio-level risk analytics output."""

    annual_return: float = 0.0
    annual_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    var_95_daily: float = 0.0
    cvar_95_daily: float = 0.0
    max_drawdown: float = 0.0
    hhi_concentration: float = 0.0
    avg_correlation: float = 0.0
    daily_returns: Optional[pd.Series] = None
    correlation_matrix: Optional[pd.DataFrame] = None


# =============================================================================
# Thread-safe rate limiter
# =============================================================================

class RateLimiter:
    """Thread-safe token-bucket-style rate limiter for API calls."""

    def __init__(self, is_cloud: bool = False):
        self.is_cloud = is_cloud
        self._request_count = 0
        self._window_start = time.time()
        self._lock = threading.Lock()

    @property
    def delay(self) -> float:
        return Config.CLOUD_DELAY if self.is_cloud else Config.LOCAL_DELAY

    @property
    def request_limit(self) -> int:
        return Config.CLOUD_REQUEST_LIMIT if self.is_cloud else Config.LOCAL_REQUEST_LIMIT

    def wait(self) -> None:
        with self._lock:
            time.sleep(self.delay)
            now = time.time()
            if now - self._window_start > Config.RATE_LIMIT_WINDOW:
                self._request_count = 0
                self._window_start = now
            self._request_count += 1
            if self._request_count > self.request_limit:
                time.sleep(2 if self.is_cloud else 1)
                self._request_count = 0
                self._window_start = time.time()


# =============================================================================
# TTL cache backed by session state
# =============================================================================

class StockCache:
    """TTL-based in-memory cache stored in ``st.session_state``."""

    def __init__(self, ttl: int = Config.CACHE_TTL_SECONDS):
        self._ttl = ttl
        if "_stock_cache" not in st.session_state:
            st.session_state["_stock_cache"] = {}

    @property
    def _store(self) -> Dict:
        return st.session_state["_stock_cache"]

    @staticmethod
    def _key(symbol: str, period: str) -> str:
        return f"{symbol}|{period}"

    def get(self, symbol: str, period: str) -> Optional[StockDataBundle]:
        entry = self._store.get(self._key(symbol, period))
        if entry is None:
            return None
        ts, bundle = entry
        if time.time() - ts > self._ttl:
            self._store.pop(self._key(symbol, period), None)
            return None
        return bundle

    def put(self, symbol: str, period: str, bundle: StockDataBundle) -> None:
        self._store[self._key(symbol, period)] = (time.time(), bundle)

    def invalidate(self, symbol: Optional[str] = None) -> None:
        if symbol is None:
            self._store.clear()
        else:
            to_del = [k for k in self._store if k.startswith(f"{symbol}|")]
            for k in to_del:
                del self._store[k]


# =============================================================================
# Core Analyzer
# =============================================================================

class IndianStockAnalyzer:
    """Central analysis engine."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.is_cloud = IS_CLOUD
        self.rate_limiter = RateLimiter(self.is_cloud)
        self.cache = StockCache()
        self.model = None
        if api_key:
            self._configure_genai(api_key)

    # ------------------------------------------------------------------
    # Gemini setup
    # ------------------------------------------------------------------
    def _configure_genai(self, api_key: str) -> None:
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel("gemini-2.0-flash")
            logger.info("Gemini API configured successfully")
        except Exception as exc:
            logger.error("Error configuring Gemini API: %s", exc)
            st.error(f"Error configuring Gemini API: {exc}")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def max_retries(self) -> int:
        return Config.MAX_RETRIES_CLOUD if self.is_cloud else Config.MAX_RETRIES_LOCAL

    @property
    def default_period(self) -> str:
        return Config.CLOUD_DEFAULT_PERIOD if self.is_cloud else Config.LOCAL_DEFAULT_PERIOD

    # ------------------------------------------------------------------
    # Universe helpers
    # ------------------------------------------------------------------
    def get_stock_universe(
        self, universe: StockUniverse, limit: Optional[int] = None
    ) -> List[str]:
        if universe == StockUniverse.NIFTY_100:
            stocks = STOCK_LISTS[StockUniverse.NIFTY_50] + STOCK_LISTS[StockUniverse.NIFTY_NEXT_50]
        elif universe == StockUniverse.ALL_SECTORS:
            sector_keys = [
                StockUniverse.BANKING, StockUniverse.IT,
                StockUniverse.PHARMA, StockUniverse.AUTO, StockUniverse.FMCG,
            ]
            stocks = list({s for k in sector_keys for s in STOCK_LISTS[k]})
        elif universe in STOCK_LISTS:
            stocks = STOCK_LISTS[universe]
        else:
            stocks = STOCK_LISTS[StockUniverse.NIFTY_50]

        if limit and self.is_cloud:
            stocks = stocks[:limit]
        return stocks

    # ------------------------------------------------------------------
    # Unified data fetching
    # ------------------------------------------------------------------
    def fetch_stock_bundle(
        self, symbol: str, period: str = "1y"
    ) -> Result[StockDataBundle]:
        """One Ticker instantiation → info + history, with cache."""
        if not validate_stock_symbol(symbol):
            return Result.fail(f"Invalid symbol format: {symbol}")

        cached = self.cache.get(symbol, period)
        if cached is not None:
            logger.debug("Cache hit: %s", symbol)
            return Result.ok(cached)

        for attempt in range(self.max_retries):
            try:
                self.rate_limiter.wait()
                ticker = yf.Ticker(symbol)
                info = ticker.info or {}
                history = ticker.history(period=period)

                if not info and history.empty:
                    continue

                history = self._clean_stock_data(history, symbol)
                fundamentals = (
                    FundamentalData.from_yfinance_info(info) if info else None
                )
                bundle = StockDataBundle(
                    symbol=symbol,
                    info=info,
                    history=history,
                    fundamentals=fundamentals,
                )
                if bundle.is_valid:
                    self.cache.put(symbol, period, bundle)
                return Result.ok(bundle)

            except Exception as exc:
                logger.warning("Attempt %d for %s: %s", attempt + 1, symbol, exc)
                if attempt < self.max_retries - 1:
                    time.sleep(Config.INITIAL_RETRY_DELAY * (2 ** attempt))

        return Result.fail(f"Failed to fetch data for {symbol} after {self.max_retries} retries")

    # ------------------------------------------------------------------
    # Data cleaning pipeline
    # ------------------------------------------------------------------
    @staticmethod
    def _clean_stock_data(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if df.empty:
            return df

        original_len = len(df)

        # 1. Deduplicate index
        df = df[~df.index.duplicated(keep="last")]

        # 2. Drop all-NaN / all-zero OHLC rows
        ohlc = [c for c in ("Open", "High", "Low", "Close") if c in df.columns]
        if ohlc:
            df = df.dropna(subset=ohlc, how="all")
            df = df[~(df[ohlc] == 0).all(axis=1)]

        # 3. Fix High < Low
        if {"High", "Low"}.issubset(df.columns):
            bad = df["High"] < df["Low"]
            if bad.any():
                logger.warning("%s: %d rows High < Low – swapping", symbol, bad.sum())
                df.loc[bad, ["High", "Low"]] = df.loc[bad, ["Low", "High"]].values

        # 4. Interpolate extreme single-day outliers
        if "Close" in df.columns and len(df) > 1:
            rets = df["Close"].pct_change().abs()
            extreme = rets > Thresholds.EXTREME_DAILY_RETURN
            extreme.iloc[0] = False
            if extreme.any():
                logger.warning("%s: %d extreme return rows interpolated", symbol, extreme.sum())
                df.loc[extreme, "Close"] = np.nan
                df["Close"] = df["Close"].interpolate(method="linear")

        # 5. Forward-fill small gaps
        df = df.ffill(limit=3)

        # 6. Strip timezone
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        if len(df) < original_len * 0.5:
            logger.warning(
                "%s: Lost >50%% of data during cleaning (%d → %d)",
                symbol, original_len, len(df),
            )
        return df

    # ------------------------------------------------------------------
    # Technical indicators
    # ------------------------------------------------------------------
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute SMA, RSI (Wilder), MACD, Bollinger with %B."""
        if df.empty:
            return df

        required = {"SMA_20", "SMA_50", "SMA_200", "RSI", "MACD", "Signal",
                     "BB_Upper", "BB_Lower", "BB_PctB"}
        if required.issubset(df.columns):
            return df  # idempotent

        df = df.copy()

        close = df["Close"]

        # Moving averages
        for window, col in ((20, "SMA_20"), (50, "SMA_50"), (200, "SMA_200")):
            df[col] = close.rolling(window=window, min_periods=1).mean()

        # RSI – Wilder's exponential smoothing
        df["RSI"] = self._calculate_rsi(close, 14)

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df["MACD"] = ema12 - ema26
        df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_Histogram"] = df["MACD"] - df["Signal"]

        # Bollinger Bands
        bb_mid = close.rolling(window=20, min_periods=1).mean()
        bb_std = close.rolling(window=20, min_periods=1).std()
        df["BB_Middle"] = bb_mid
        df["BB_Upper"] = bb_mid + 2 * bb_std
        df["BB_Lower"] = bb_mid - 2 * bb_std
        bb_width = df["BB_Upper"] - df["BB_Lower"]
        df["BB_Width"] = bb_width / (bb_mid + 1e-10)
        df["BB_PctB"] = (close - df["BB_Lower"]) / (bb_width + 1e-10)

        return df

    @staticmethod
    def _calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """RSI via Wilder's exponential smoothing (vectorised approximation)."""
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))

    # ------------------------------------------------------------------
    # Scoring – fixed denominator with data coverage penalty
    # ------------------------------------------------------------------
    def calculate_quality_score(self, fd: FundamentalData) -> Tuple[float, int, int]:
        """Returns (score, metrics_available, total_metrics)."""
        score = 0.0
        available = 0
        total = 5
        TH = Thresholds

        # ROE (0-20)
        if fd.roe is not None:
            available += 1
            if fd.roe > TH.ROE_EXCELLENT:
                score += 20
            elif fd.roe > TH.ROE_GOOD:
                score += 15
            elif fd.roe > TH.ROE_ACCEPTABLE:
                score += 10
            elif fd.roe > 0:
                score += 5

        # Debt/Equity (0-20)
        if fd.debt_to_equity is not None:
            available += 1
            if fd.debt_to_equity < TH.DEBT_EQUITY_LOW:
                score += 20
            elif fd.debt_to_equity < TH.DEBT_EQUITY_MODERATE:
                score += 15
            elif fd.debt_to_equity < TH.DEBT_EQUITY_HIGH:
                score += 10
            else:
                score += 3

        # Current Ratio (0-20)
        if fd.current_ratio is not None:
            available += 1
            if TH.CURRENT_RATIO_IDEAL_LOW <= fd.current_ratio <= TH.CURRENT_RATIO_IDEAL_HIGH:
                score += 20
            elif fd.current_ratio >= TH.CURRENT_RATIO_ACCEPTABLE:
                score += 12
            elif fd.current_ratio > TH.CURRENT_RATIO_IDEAL_HIGH:
                score += 8
            else:
                score += 3

        # Profit Margin (0-20)
        if fd.profit_margin is not None:
            available += 1
            if fd.profit_margin > TH.PROFIT_MARGIN_EXCELLENT:
                score += 20
            elif fd.profit_margin > TH.PROFIT_MARGIN_GOOD:
                score += 15
            elif fd.profit_margin > TH.PROFIT_MARGIN_ACCEPTABLE:
                score += 10
            elif fd.profit_margin > 0:
                score += 5

        # Operating Margin (0-20)
        if fd.operating_margin is not None:
            available += 1
            if fd.operating_margin > TH.OPERATING_MARGIN_EXCELLENT:
                score += 20
            elif fd.operating_margin > TH.OPERATING_MARGIN_GOOD:
                score += 15
            elif fd.operating_margin > TH.OPERATING_MARGIN_ACCEPTABLE:
                score += 10
            elif fd.operating_margin > 0:
                score += 5

        # Data coverage penalty
        coverage = available / total if total > 0 else 0
        if coverage < 0.6:
            score *= coverage

        return min(score, 100.0), available, total

    def calculate_value_score(
        self, fd: FundamentalData, current_price: Optional[float] = None
    ) -> Tuple[float, int, int]:
        score = 0.0
        available = 0
        total = 4
        TH = Thresholds

        # PE
        if fd.pe_ratio is not None and fd.pe_ratio > 0:
            available += 1
            if fd.pe_ratio < TH.PE_CHEAP:
                score += 25
            elif fd.pe_ratio < TH.PE_FAIR:
                score += 20
            elif fd.pe_ratio < TH.PE_MODERATE:
                score += 10
            else:
                score += 3

        # PB
        if fd.pb_ratio is not None and fd.pb_ratio > 0:
            available += 1
            if fd.pb_ratio < TH.PB_CHEAP:
                score += 25
            elif fd.pb_ratio < TH.PB_FAIR:
                score += 20
            elif fd.pb_ratio < TH.PB_MODERATE:
                score += 10
            else:
                score += 3

        # Dividend Yield
        if fd.dividend_yield is not None and fd.dividend_yield > 0:
            available += 1
            if fd.dividend_yield > TH.DIVIDEND_YIELD_HIGH:
                score += 25
            elif fd.dividend_yield > TH.DIVIDEND_YIELD_MODERATE:
                score += 20
            elif fd.dividend_yield > TH.DIVIDEND_YIELD_LOW:
                score += 10
            else:
                score += 3

        # 52-week price position
        if (
            current_price
            and fd.week_52_high is not None
            and fd.week_52_low is not None
            and fd.week_52_high > fd.week_52_low
        ):
            available += 1
            rng = fd.week_52_high - fd.week_52_low
            pos = (current_price - fd.week_52_low) / rng
            if pos < TH.PRICE_POSITION_LOW:
                score += 25
            elif pos < TH.PRICE_POSITION_MID:
                score += 20
            elif pos < TH.PRICE_POSITION_HIGH:
                score += 10
            else:
                score += 3

        coverage = available / total if total > 0 else 0
        if coverage < 0.6:
            score *= coverage

        return min(score, 100.0), available, total

    def calculate_momentum_score(self, df: pd.DataFrame) -> Tuple[float, int, int]:
        if df.empty or len(df) < 20:
            return 50.0, 0, 1

        score = 0.0
        available = 0
        total = 5
        latest = df.iloc[-1]
        TH = Thresholds

        # MA signals (combined – 0-30)
        ma_hit = 0
        ma_count = 0
        for col in ("SMA_20", "SMA_50", "SMA_200"):
            if col in df.columns and pd.notna(latest.get(col)):
                ma_count += 1
                if latest["Close"] > latest[col]:
                    ma_hit += 1
        if ma_count > 0:
            available += 1
            score += (ma_hit / ma_count) * 30

        # RSI zone (0-20)
        rsi = latest.get("RSI")
        if pd.notna(rsi):
            available += 1
            if TH.RSI_SWEET_LOW <= rsi <= TH.RSI_SWEET_HIGH:
                score += 20
            elif TH.RSI_OVERSOLD <= rsi < TH.RSI_SWEET_LOW:
                score += 15
            elif TH.RSI_SWEET_HIGH < rsi <= TH.RSI_OVERBOUGHT:
                score += 10

        # Bollinger %B (0-10)
        pctb = latest.get("BB_PctB")
        if pd.notna(pctb):
            available += 1
            if 0.2 <= pctb <= 0.8:
                score += 10
            elif 0.0 <= pctb < 0.2:
                score += 5  # near lower band – possible bounce

        # 1-month return (0-20)
        if len(df) > 22:
            available += 1
            prev = df.iloc[-22]["Close"]
            if prev > 0:
                ret = (latest["Close"] - prev) / prev
                if ret > TH.RETURN_1M_STRONG:
                    score += 20
                elif ret > 0:
                    score += 10

        # 3-month return (0-20)
        if len(df) > 66:
            available += 1
            prev = df.iloc[-66]["Close"]
            if prev > 0:
                ret = (latest["Close"] - prev) / prev
                if ret > TH.RETURN_3M_STRONG:
                    score += 20
                elif ret > 0:
                    score += 10

        coverage = available / total if total > 0 else 0
        if coverage < 0.6:
            score *= coverage

        return min(score, 100.0), available, total

    def calculate_all_scores(
        self, fundamentals: FundamentalData, df: pd.DataFrame
    ) -> StockScores:
        current_price = df.iloc[-1]["Close"] if not df.empty else None

        q_score, q_avail, q_total = self.calculate_quality_score(fundamentals)
        v_score, v_avail, v_total = self.calculate_value_score(fundamentals, current_price)
        m_score, m_avail, m_total = self.calculate_momentum_score(df)

        total_avail = q_avail + v_avail + m_avail
        total_metrics = q_total + v_total + m_total
        coverage = total_avail / total_metrics if total_metrics > 0 else 0

        return StockScores(
            quality=q_score,
            value=v_score,
            momentum=m_score,
            data_coverage=coverage,
        )

    # ------------------------------------------------------------------
    # AI report generation
    # ------------------------------------------------------------------
    def generate_ai_report(
        self,
        symbol: str,
        fundamentals: FundamentalData,
        scores: StockScores,
    ) -> str:
        if not self.model:
            return "Please configure Gemini API key to generate AI reports."
        try:
            prompt = f"""
Generate a comprehensive equity research report for {fundamentals.company_name} ({symbol}) – an Indian stock.

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

Scores (data coverage {scores.data_coverage:.0%}):
- Quality Score: {scores.quality:.1f}/100
- Value Score: {scores.value:.1f}/100
- Momentum Score: {scores.momentum:.1f}/100
- Overall Score: {scores.overall:.1f}/100

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
            return sanitize_ai_output(response.text)
        except Exception as exc:
            logger.error("Error generating AI report: %s", exc)
            return f"Error generating AI report: {exc}"

    def generate_basic_report(
        self,
        symbol: str,
        fundamentals: FundamentalData,
        scores: StockScores,
    ) -> str:
        def _rating(s: float, labels: Tuple[str, str, str]) -> str:
            if s > Config.SCORE_HIGH:
                return labels[0]
            if s > Config.SCORE_MEDIUM:
                return labels[1]
            return labels[2]

        q_r = _rating(scores.quality, ("Strong", "Moderate", "Weak"))
        v_r = _rating(scores.value, ("Attractive", "Fair", "Expensive"))
        m_r = _rating(scores.momentum, ("Positive", "Neutral", "Negative"))

        return f"""
EQUITY RESEARCH REPORT
======================
Symbol: {symbol}
Company: {fundamentals.company_name}
Sector: {fundamentals.sector}
Date: {datetime.now().strftime('%Y-%m-%d')}

SCORES (Data Coverage: {scores.data_coverage:.0%})
------
Quality Score: {scores.quality:.1f}/100
Value Score: {scores.value:.1f}/100
Momentum Score: {scores.momentum:.1f}/100
Overall Score: {scores.overall:.1f}/100

FUNDAMENTAL METRICS
-------------------
PE Ratio: {fundamentals.get_pe_display()}
PB Ratio: {fundamentals.get_pb_display()}
ROE: {fundamentals.get_roe_display()}
Debt to Equity: {fundamentals.get_debt_to_equity_display()}
Profit Margin: {fundamentals.get_profit_margin_display()}

ASSESSMENT
----------
Quality: {q_r}
Value: {v_r}
Momentum: {m_r}

NOTE: Automated analysis. Conduct independent research before investing.
"""


# =============================================================================
# Portfolio Risk Analytics
# =============================================================================

class PortfolioRiskAnalyzer:
    """Calculate portfolio-level risk metrics."""

    @staticmethod
    def calculate(
        holdings: List[Dict],
        analyzer: IndianStockAnalyzer,
        lookback: str = "1y",
    ) -> Result[PortfolioRiskMetrics]:
        returns_map: Dict[str, pd.Series] = {}
        weights: Dict[str, float] = {}
        total_value = sum(h.get("current_value", 0) for h in holdings)

        if total_value <= 0:
            return Result.fail("Total portfolio value must be positive")

        for h in holdings:
            sym = h["symbol"]
            w = h.get("current_value", 0) / total_value
            weights[sym] = w

            res = analyzer.fetch_stock_bundle(sym, period=lookback)
            if res.is_ok and not res.value.history.empty:
                ret = res.value.history["Close"].pct_change().dropna()
                if not ret.empty:
                    returns_map[sym] = ret

        if not returns_map:
            return Result.fail("No return data available for any holding")

        returns_df = pd.DataFrame(returns_map).dropna()
        if returns_df.empty or len(returns_df) < 5:
            return Result.fail("Insufficient overlapping return data")

        wvec = np.array([weights.get(c, 0) for c in returns_df.columns])
        wvec = wvec / wvec.sum()

        port_ret = returns_df.dot(wvec)
        td = Config.TRADING_DAYS
        daily_vol = port_ret.std()
        ann_vol = daily_vol * np.sqrt(td)
        ann_ret = port_ret.mean() * td

        # VaR / CVaR (95 %)
        var95 = np.percentile(port_ret, 5) * total_value
        tail = port_ret[port_ret <= np.percentile(port_ret, 5)]
        cvar95 = tail.mean() * total_value if not tail.empty else var95

        # Max drawdown
        cum = (1 + port_ret).cumprod()
        running_max = cum.expanding().max()
        dd = (cum - running_max) / running_max
        max_dd = dd.min()

        # Sharpe
        rf_daily = Config.RISK_FREE_RATE / td
        sharpe = ((port_ret.mean() - rf_daily) / daily_vol * np.sqrt(td)) if daily_vol > 0 else 0

        # Concentration (HHI)
        hhi = float((wvec ** 2).sum())

        # Average pairwise correlation
        corr = returns_df.corr()
        upper = corr.values[np.triu_indices_from(corr.values, k=1)]
        avg_corr = float(upper.mean()) if len(upper) > 0 else 0.0

        return Result.ok(PortfolioRiskMetrics(
            annual_return=ann_ret,
            annual_volatility=ann_vol,
            sharpe_ratio=sharpe,
            var_95_daily=var95,
            cvar_95_daily=cvar95,
            max_drawdown=max_dd,
            hhi_concentration=hhi,
            avg_correlation=avg_corr,
            daily_returns=port_ret,
            correlation_matrix=corr,
        ))


# =============================================================================
# Stock Screener
# =============================================================================

class StockScreener:
    """Threaded stock screening."""

    @staticmethod
    def screen_stocks(
        stocks: List[str],
        analyzer: IndianStockAnalyzer,
        screen_type: ScreenType,
        min_score: float = 0,
        progress_callback: Optional[Callable[[float], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> pd.DataFrame:
        results: List[Dict] = []
        total = len(stocks)
        if total == 0:
            return pd.DataFrame()

        completed = 0
        lock = threading.Lock()

        def _analyze(symbol: str) -> Optional[Dict]:
            nonlocal completed
            try:
                res = analyzer.fetch_stock_bundle(symbol, period=analyzer.default_period)
                if not res.is_ok or not res.value.is_valid or res.value.fundamentals is None:
                    return None

                bundle = res.value
                df = analyzer.calculate_technical_indicators(bundle.history)
                scores = analyzer.calculate_all_scores(bundle.fundamentals, df)

                score_map = {
                    ScreenType.QUALITY: scores.quality,
                    ScreenType.VALUE: scores.value,
                    ScreenType.MOMENTUM: scores.momentum,
                    ScreenType.OVERALL: scores.overall,
                }
                if score_map.get(screen_type, scores.overall) < min_score:
                    return None

                cp = df.iloc[-1]["Close"]
                change_1d = 0.0
                if len(df) >= 2:
                    prev = df.iloc[-2]["Close"]
                    if prev != 0:
                        change_1d = ((cp - prev) / prev) * 100

                fd = bundle.fundamentals
                return {
                    "Symbol": symbol.replace(".NS", "").replace(".BO", ""),
                    "Company": (fd.company_name[:30] if fd.company_name else "N/A"),
                    "Sector": fd.sector,
                    "Price": round(cp, 2),
                    "Change %": round(change_1d, 2),
                    "PE Ratio": fd.get_pe_display(),
                    "PB Ratio": fd.get_pb_display(),
                    "ROE %": fd.get_roe_display(),
                    "D/E": fd.get_debt_to_equity_display(),
                    "Quality": round(scores.quality, 1),
                    "Value": round(scores.value, 1),
                    "Momentum": round(scores.momentum, 1),
                    "Overall": round(scores.overall, 1),
                    "Coverage": f"{scores.data_coverage:.0%}",
                }
            except Exception as exc:
                logger.error("Error screening %s: %s", symbol, exc)
                return None
            finally:
                with lock:
                    completed += 1
                    if progress_callback:
                        progress_callback(completed / total)
                    if status_callback:
                        status_callback(f"Completed {symbol} ({completed}/{total})")

        max_workers = (
            Config.MAX_WORKERS_CLOUD if analyzer.is_cloud else min(Config.MAX_WORKERS_LOCAL, total)
        )

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_analyze, s): s for s in stocks}
            for future in as_completed(futures):
                row = future.result()
                if row is not None:
                    results.append(row)

        df_res = pd.DataFrame(results)
        if not df_res.empty:
            sort_col = screen_type.value if screen_type.value in df_res.columns else "Overall"
            df_res.sort_values(sort_col, ascending=False, inplace=True)
            df_res.reset_index(drop=True, inplace=True)
        return df_res


# =============================================================================
# Input validation helpers
# =============================================================================

def validate_portfolio_entry(
    parts: List[str], line_num: int
) -> Tuple[Optional[Dict], Optional[str]]:
    """Validate a single portfolio line with bounds checking."""
    if len(parts) != 3:
        return None, f"Line {line_num}: Expected Symbol, Quantity, Price"

    sym = sanitize_symbol_input(parts[0])
    if not sym:
        return None, f"Line {line_num}: Invalid symbol"

    try:
        qty = int(parts[1])
    except ValueError:
        return None, f"Line {line_num}: Quantity must be a whole number"

    try:
        price = float(parts[2])
    except ValueError:
        return None, f"Line {line_num}: Price must be a number"

    if qty <= 0 or qty > Config.MAX_QUANTITY:
        return None, f"Line {line_num}: Quantity must be 1–{Config.MAX_QUANTITY:,}"
    if price < Config.MIN_PRICE or price > Config.MAX_PRICE:
        return None, f"Line {line_num}: Price must be {Config.MIN_PRICE}–{Config.MAX_PRICE:,}"

    return {"Symbol": f"{sym}.NS", "Quantity": qty, "Buy Price": price}, None


# =============================================================================
# CSS
# =============================================================================

_CUSTOM_CSS = """
<style>
.main-header {font-size:3rem;color:#1f77b4;text-align:center;margin-bottom:2rem;}
.sub-header  {font-size:1.5rem;color:#ff7f0e;margin-bottom:1rem;}
.metric-card {background-color:#f0f2f6;padding:1rem;border-radius:.5rem;margin-bottom:1rem;}
.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {font-size:1.2rem;}
</style>
"""


# =============================================================================
# Session state
# =============================================================================

def _init_session_state() -> None:
    for key, default in (
        ("api_key", ""),
        ("_stock_cache", {}),
        ("screening_results", None),
        ("last_analysis", None),
    ):
        if key not in st.session_state:
            st.session_state[key] = default


# =============================================================================
# Chart builders
# =============================================================================

def create_price_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Price",
    ))
    for col, color, name in (
        ("SMA_20", "orange", "SMA 20"),
        ("SMA_50", "blue", "SMA 50"),
        ("SMA_200", "red", "SMA 200"),
    ):
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col], name=name, line=dict(color=color),
            ))

    # Bollinger bands (shaded)
    if "BB_Upper" in df.columns and "BB_Lower" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Upper"], name="BB Upper",
            line=dict(color="rgba(128,128,128,0.3)", dash="dot"),
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Lower"], name="BB Lower",
            line=dict(color="rgba(128,128,128,0.3)", dash="dot"),
            fill="tonexty", fillcolor="rgba(128,128,128,0.08)",
        ))

    fig.update_layout(
        title=f"{symbol} Price Chart", yaxis_title="Price (₹)",
        xaxis_title="Date", height=600, xaxis_rangeslider_visible=False,
    )
    return fig


def create_volume_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    colors = [
        "green" if df["Close"].iloc[i] >= df["Open"].iloc[i] else "red"
        for i in range(len(df))
    ]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                         marker_color=colors))
    fig.update_layout(title="Volume", height=200, xaxis_title="Date", yaxis_title="Volume")
    return fig


def create_rsi_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
                             line=dict(color="purple")))
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    fig.add_hrect(y0=30, y1=70, fillcolor="rgba(0,128,0,0.05)", line_width=0)
    fig.update_layout(title="RSI (14 – Wilder)", height=300,
                      yaxis_title="RSI", xaxis_title="Date")
    return fig


def create_macd_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD",
                             line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=df.index, y=df["Signal"], name="Signal",
                             line=dict(color="red")))
    if "MACD_Histogram" in df.columns:
        colors = ["green" if v >= 0 else "red" for v in df["MACD_Histogram"]]
        fig.add_trace(go.Bar(x=df.index, y=df["MACD_Histogram"],
                             name="Histogram", marker_color=colors))
    fig.update_layout(title="MACD", height=300, yaxis_title="MACD", xaxis_title="Date")
    return fig


def create_bb_pctb_chart(df: pd.DataFrame) -> go.Figure:
    """Bollinger %B chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_PctB"], name="%B",
                             line=dict(color="teal")))
    fig.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Upper Band")
    fig.add_hline(y=0.0, line_dash="dash", line_color="green", annotation_text="Lower Band")
    fig.add_hline(y=0.5, line_dash="dot", line_color="gray", annotation_text="Middle")
    fig.update_layout(title="Bollinger %B", height=250, yaxis_title="%B", xaxis_title="Date")
    return fig


# =============================================================================
# Dataframe styling
# =============================================================================

def style_dataframe(df: pd.DataFrame, score_columns: List[str]) -> Any:
    def _color_change(val: Any) -> str:
        try:
            if isinstance(val, str) or pd.isna(val):
                return ""
            return "color: red" if val < 0 else ("color: green" if val > 0 else "")
        except (TypeError, ValueError):
            return ""

    def _bg_gradient(val: Any) -> str:
        try:
            if isinstance(val, str) or pd.isna(val):
                return ""
            n = max(0.0, min(1.0, val / 100))
            r = int(255 * (1 - n))
            g = int(255 * n)
            return f"background-color: rgba({r},{g},0,0.3)"
        except (TypeError, ValueError):
            return ""

    styled = df.style
    if "Change %" in df.columns:
        styled = styled.map(_color_change, subset=["Change %"])

    for col in score_columns:
        if col in df.columns:
            if HAS_MATPLOTLIB:
                try:
                    styled = styled.background_gradient(
                        subset=[col], cmap="RdYlGn", vmin=0, vmax=100,
                    )
                except Exception:
                    styled = styled.map(_bg_gradient, subset=[col])
            else:
                styled = styled.map(_bg_gradient, subset=[col])
    return styled


# =============================================================================
# Sidebar
# =============================================================================

def render_sidebar() -> None:
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Try secrets / env first
        default_key = ""
        try:
            default_key = st.secrets.get("GEMINI_API_KEY", "")
        except Exception:
            default_key = os.environ.get("GEMINI_API_KEY", "")

        api_key = st.text_input(
            "Gemini API Key", type="password",
            value=st.session_state.api_key or default_key,
            help="Enter your Google Gemini API key for AI-powered reports",
        )
        if api_key:
            st.session_state.api_key = api_key
            st.success("✅ API Key configured")

        st.markdown("---")
        st.markdown("### 📊 Quick Links")
        st.markdown("- [NSE India](https://www.nseindia.com)")
        st.markdown("- [BSE India](https://www.bseindia.com)")
        st.markdown("- [Moneycontrol](https://www.moneycontrol.com)")

        st.markdown("---")
        if st.button("🗑️ Clear Cache"):
            StockCache().invalidate()
            st.success("Cache cleared")


# =============================================================================
# Tab: Stock Analysis
# =============================================================================

def render_stock_analysis_tab(analyzer: IndianStockAnalyzer) -> None:
    st.header("Individual Stock Analysis")

    col1, col2 = st.columns([2, 1])
    with col1:
        stock_input = st.text_input(
            "Enter Stock Symbol (e.g., RELIANCE, TCS, INFY)", value="RELIANCE",
        )
        exchange = st.selectbox("Exchange", ["NSE", "BSE"])
        sanitized = sanitize_symbol_input(stock_input)
        symbol = f"{sanitized}.{'NS' if exchange == 'NSE' else 'BO'}"

    with col2:
        period = st.selectbox(
            "Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=3,
        )
        analyze_btn = st.button("🔍 Analyze Stock", type="primary", use_container_width=True)

    if not analyze_btn:
        return

    if not validate_stock_symbol(symbol):
        st.error("Invalid stock symbol format. Please check your input.")
        return

    with st.spinner(f"Analyzing {symbol}…"):
        res = analyzer.fetch_stock_bundle(symbol, period=period)

    if not res.is_ok or res.value.fundamentals is None:
        st.error(f"Unable to fetch data for {symbol}. {res.error or ''}")
        return

    bundle = res.value
    df = analyzer.calculate_technical_indicators(bundle.history)
    fundamentals = bundle.fundamentals
    scores = analyzer.calculate_all_scores(fundamentals, df)

    # Header metrics
    st.subheader(f"📊 {fundamentals.company_name}")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Sector", fundamentals.sector)
    mc2.metric("Industry", (fundamentals.industry[:20] if fundamentals.industry else "N/A"))

    cp = df.iloc[-1]["Close"]
    mc3.metric("Current Price", f"₹{cp:.2f}")

    if len(df) >= 2:
        prev = df.iloc[-2]["Close"]
        ch = ((cp - prev) / prev * 100) if prev > 0 else 0
        mc4.metric("Change", f"{ch:.2f}%")
    else:
        mc4.metric("Change", "N/A")

    tabs = st.tabs(["Price Chart", "Fundamentals", "Technical", "AI Report"])

    with tabs[0]:
        st.plotly_chart(create_price_chart(df, symbol), use_container_width=True)
        st.plotly_chart(create_volume_chart(df), use_container_width=True)

    with tabs[1]:
        _render_fundamentals(fundamentals, scores)

    with tabs[2]:
        _render_technical(df)

    with tabs[3]:
        _render_ai_report(analyzer, symbol, fundamentals, scores)


def _render_fundamentals(fd: FundamentalData, scores: StockScores) -> None:
    st.subheader("Fundamental Metrics")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Valuation**")
        st.write(f"PE Ratio: {fd.get_pe_display()}")
        st.write(f"PB Ratio: {fd.get_pb_display()}")
        st.write(f"Market Cap: {fd.get_market_cap_display()}")
        st.write(f"EPS: {fd.get_eps_display()}")

    with c2:
        st.markdown("**Profitability**")
        st.write(f"ROE: {fd.get_roe_display()}")
        st.write(f"ROA: {fd.get_roa_display()}")
        st.write(f"Profit Margin: {fd.get_profit_margin_display()}")
        st.write(f"Operating Margin: {fd.get_operating_margin_display()}")

    with c3:
        st.markdown("**Financial Health**")
        st.write(f"Debt/Equity: {fd.get_debt_to_equity_display()}")
        st.write(f"Current Ratio: {fd.get_current_ratio_display()}")
        st.write(f"Quick Ratio: {fd.get_quick_ratio_display()}")
        st.write(f"Dividend Yield: {fd.get_dividend_yield_display()}")

    st.markdown("---")
    st.subheader("Stock Scores")
    st.caption(f"Data coverage: {scores.data_coverage:.0%}")

    sc1, sc2, sc3, sc4 = st.columns(4)
    for col_ui, label, val in (
        (sc1, "Quality", scores.quality),
        (sc2, "Value", scores.value),
        (sc3, "Momentum", scores.momentum),
        (sc4, "Overall", scores.overall),
    ):
        with col_ui:
            st.metric(label, f"{val:.1f}/100")
            st.progress(min(val / 100, 1.0))


def _render_technical(df: pd.DataFrame) -> None:
    st.subheader("Technical Indicators")

    if "RSI" in df.columns:
        st.plotly_chart(create_rsi_chart(df), use_container_width=True)
    if "MACD" in df.columns:
        st.plotly_chart(create_macd_chart(df), use_container_width=True)
    if "BB_PctB" in df.columns:
        st.plotly_chart(create_bb_pctb_chart(df), use_container_width=True)

    st.markdown("---")
    st.subheader("Current Technical Readings")

    if df.empty:
        st.warning("No data available")
        return

    latest = df.iloc[-1]
    tc1, tc2, tc3, tc4, tc5 = st.columns(5)

    rsi = latest.get("RSI")
    if pd.notna(rsi):
        tc1.metric("RSI", f"{rsi:.2f}")

    macd_v = latest.get("MACD")
    if pd.notna(macd_v):
        tc2.metric("MACD", f"{macd_v:.4f}")

    sig_v = latest.get("Signal")
    if pd.notna(sig_v):
        tc3.metric("Signal", f"{sig_v:.4f}")

    if pd.notna(macd_v) and pd.notna(sig_v):
        tc4.metric("MACD Signal", "Bullish" if macd_v > sig_v else "Bearish")

    pctb = latest.get("BB_PctB")
    if pd.notna(pctb):
        tc5.metric("BB %B", f"{pctb:.2f}")


def _render_ai_report(
    analyzer: IndianStockAnalyzer,
    symbol: str,
    fd: FundamentalData,
    scores: StockScores,
) -> None:
    st.subheader("AI-Generated Equity Research Report")

    if st.session_state.api_key and analyzer.model:
        with st.spinner("Generating AI report…"):
            report = analyzer.generate_ai_report(symbol, fd, scores)
        st.markdown(report)
        st.download_button(
            "📥 Download Report", data=report,
            file_name=f"{symbol}_report_{datetime.now():%Y%m%d}.txt",
            mime="text/plain",
        )
    else:
        st.warning("Configure Gemini API key in sidebar for AI reports.")
        st.info("Showing basic report instead.")
        basic = analyzer.generate_basic_report(symbol, fd, scores)
        st.text(basic)


# =============================================================================
# Tab: Screener
# =============================================================================

def render_screener_tab(analyzer: IndianStockAnalyzer) -> None:
    st.header("Stock Screener")

    col_left, col_right = st.columns([3, 1])

    with col_left:
        sc1, sc2, sc3 = st.columns(3)

        with sc1:
            screen_type_options = [member.value for member in ScreenType]
            screen_type_val = st.selectbox("Screening Strategy", screen_type_options)
            screen_type_enum = ScreenType(screen_type_val)

        with sc2:
            if IS_CLOUD:
                uni_opts = [
                    StockUniverse.NIFTY_50.value,
                    StockUniverse.BANKING.value,
                    StockUniverse.IT.value,
                    StockUniverse.PHARMA.value,
                    StockUniverse.AUTO.value,
                    StockUniverse.FMCG.value,
                    StockUniverse.CUSTOM.value,
                ]
                default_limit = Config.DEFAULT_CLOUD_LIMIT
            else:
                uni_opts = [u.value for u in StockUniverse]
                default_limit = Config.DEFAULT_LOCAL_LIMIT
            stock_universe_val = st.selectbox("Stock Universe", uni_opts)

        with sc3:
            min_score = st.slider("Minimum Score Filter", 0, 100, 50, 10)

    with col_right:
        st.markdown("### Quick Stats")
        if stock_universe_val != StockUniverse.CUSTOM.value:
            uni_enum = StockUniverse(stock_universe_val)
            lim = 50 if IS_CLOUD and stock_universe_val in ("NIFTY 100", "All Sectors") else None
            n = len(analyzer.get_stock_universe(uni_enum, limit=lim))
            st.info(f"📊 {n} stocks in {stock_universe_val}")

    custom_stocks = ""
    if stock_universe_val == StockUniverse.CUSTOM.value:
        custom_stocks = st.text_area(
            "Enter symbols (comma-separated)",
            "RELIANCE,TCS,INFY,HDFCBANK,ICICIBANK,WIPRO,BHARTIARTL,SBIN",
            height=100,
        )

    with st.expander("Advanced Filters"):
        af1, af2, af3 = st.columns(3)
        with af1:
            max_results = st.number_input(
                "Max Results", min_value=10,
                max_value=100 if IS_CLOUD else 200,
                value=default_limit,
            )
        with af2:
            sort_by = st.selectbox("Sort By", ["Score", "PE Ratio", "PB Ratio", "ROE %"])
        with af3:
            ascending = st.checkbox("Ascending Order", value=False)

    if not st.button("🔍 Run Screener", type="primary", use_container_width=True):
        return

    progress_bar = st.progress(0)
    status_text = st.empty()

    if stock_universe_val == StockUniverse.CUSTOM.value:
        stocks = [
            f"{sanitize_symbol_input(s)}.NS"
            for s in custom_stocks.split(",") if s.strip()
        ]
    else:
        uni_enum = StockUniverse(stock_universe_val)
        lim = max_results if IS_CLOUD else None
        stocks = analyzer.get_stock_universe(uni_enum, limit=lim)

    if not stocks:
        st.warning("No stocks to screen.")
        return

    status_text.text(f"Screening {len(stocks)} stocks…")

    results = StockScreener.screen_stocks(
        stocks, analyzer, screen_type_enum, min_score,
        progress_callback=progress_bar.progress,
        status_callback=status_text.text,
    )

    progress_bar.empty()
    status_text.empty()

    if results.empty:
        st.warning("No stocks found. Try adjusting filters.")
        return

    results = results.head(int(max_results))
    st.success(f"✅ Found {len(results)} stocks (Score ≥ {min_score})")

    # Summary metrics
    sm1, sm2, sm3, sm4 = st.columns(4)
    sm1.metric("Avg Quality", f"{results['Quality'].mean():.1f}")
    sm2.metric("Avg Value", f"{results['Value'].mean():.1f}")
    sm3.metric("Avg Momentum", f"{results['Momentum'].mean():.1f}")
    sm4.metric("Avg Overall", f"{results['Overall'].mean():.1f}")

    st.subheader(f"📊 Top {len(results)} Stocks")
    styled = style_dataframe(results, ["Quality", "Value", "Momentum", "Overall"])
    st.dataframe(styled, use_container_width=True, height=600)

    st.download_button(
        "📥 Download CSV",
        data=results.to_csv(index=False),
        file_name=f"{stock_universe_val}_{screen_type_val}_{datetime.now():%Y%m%d}.csv",
        mime="text/csv",
    )

    st.subheader("🏆 Top 10 Picks")
    top10 = results.head(10)
    fig = px.bar(
        top10, x="Symbol", y=screen_type_val,
        color=screen_type_val, color_continuous_scale="RdYlGn",
        title=f"Top 10 by {screen_type_val} Score",
    )
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# Tab: Portfolio
# =============================================================================

def render_portfolio_tab(analyzer: IndianStockAnalyzer) -> None:
    st.header("Portfolio Analysis")
    st.info("Track holdings, P&L, and portfolio risk metrics (VaR, Sharpe, Max Drawdown).")

    portfolio_text = st.text_area(
        "Enter portfolio (Symbol, Quantity, Buy Price – one per line)",
        "RELIANCE, 100, 2400\nTCS, 50, 3500\nINFY, 75, 1400\nHDFCBANK, 25, 1600\nICICIBANK, 100, 950",
    )

    if not st.button("Analyze Portfolio", type="primary"):
        return

    portfolio: List[Dict] = []
    for line_num, line in enumerate(portfolio_text.strip().split("\n"), 1):
        parts = [p.strip() for p in line.split(",")]
        entry, err = validate_portfolio_entry(parts, line_num)
        if err:
            st.warning(err)
        elif entry:
            portfolio.append(entry)

    if not portfolio:
        st.error("No valid portfolio entries.")
        return

    results: List[Dict] = []
    holdings_for_risk: List[Dict] = []
    total_invested = 0.0
    total_current = 0.0

    with st.spinner("Analyzing portfolio…"):
        for stock in portfolio:
            res = analyzer.fetch_stock_bundle(stock["Symbol"], period="1mo")
            if not res.is_ok or res.value.history.empty:
                st.warning(f"Could not fetch {stock['Symbol']}")
                continue

            cp = res.value.history.iloc[-1]["Close"]
            invested = stock["Quantity"] * stock["Buy Price"]
            current_val = stock["Quantity"] * cp
            pnl = current_val - invested
            pnl_pct = (pnl / invested * 100) if invested != 0 else 0

            results.append({
                "Symbol": stock["Symbol"].replace(".NS", ""),
                "Quantity": stock["Quantity"],
                "Buy Price": stock["Buy Price"],
                "Current Price": round(cp, 2),
                "Invested": round(invested, 2),
                "Current Value": round(current_val, 2),
                "P&L": round(pnl, 2),
                "P&L %": round(pnl_pct, 2),
            })
            holdings_for_risk.append({
                "symbol": stock["Symbol"],
                "current_value": current_val,
            })
            total_invested += invested
            total_current += current_val

    if not results:
        st.error("No data retrieved for any holding.")
        return

    # Summary
    hc1, hc2, hc3, hc4 = st.columns(4)
    hc1.metric("Total Invested", f"₹{total_invested:,.2f}")
    hc2.metric("Current Value", f"₹{total_current:,.2f}")
    total_pnl = total_current - total_invested
    pnl_pct = (total_pnl / total_invested * 100) if total_invested != 0 else 0
    hc3.metric("Total P&L", f"₹{total_pnl:,.2f}", delta=f"{pnl_pct:.2f}%")
    hc4.metric("Portfolio Return", f"{pnl_pct:.2f}%")

    # Holdings table
    st.subheader("Holdings")
    df_port = pd.DataFrame(results)

    def _pnl_style(val: Any) -> str:
        try:
            if isinstance(val, str) or pd.isna(val):
                return ""
            return "color: green" if val > 0 else ("color: red" if val < 0 else "")
        except Exception:
            return ""

    styled_port = df_port.style.map(_pnl_style, subset=["P&L", "P&L %"])
    st.dataframe(styled_port, use_container_width=True)

    # Allocation pie
    st.subheader("Portfolio Composition")
    fig_pie = px.pie(df_port, values="Current Value", names="Symbol",
                     title="Allocation by Market Value")
    st.plotly_chart(fig_pie, use_container_width=True)

    # Risk analytics
    st.subheader("📉 Risk Analytics")
    with st.spinner("Calculating risk metrics…"):
        risk_res = PortfolioRiskAnalyzer.calculate(holdings_for_risk, analyzer)

    if not risk_res.is_ok:
        st.warning(f"Could not compute risk metrics: {risk_res.error}")
        return

    rm = risk_res.value
    rc1, rc2, rc3, rc4 = st.columns(4)
    rc1.metric("Annual Return", f"{rm.annual_return:.2%}")
    rc2.metric("Annual Volatility", f"{rm.annual_volatility:.2%}")
    rc3.metric("Sharpe Ratio", f"{rm.sharpe_ratio:.2f}")
    rc4.metric("Max Drawdown", f"{rm.max_drawdown:.2%}")

    rc5, rc6, rc7, rc8 = st.columns(4)
    rc5.metric("Daily VaR (95%)", f"₹{rm.var_95_daily:,.0f}")
    rc6.metric("Daily CVaR (95%)", f"₹{rm.cvar_95_daily:,.0f}")
    rc7.metric("HHI Concentration", f"{rm.hhi_concentration:.3f}")
    rc8.metric("Avg Correlation", f"{rm.avg_correlation:.2f}")

    # Correlation heatmap
    if rm.correlation_matrix is not None and len(rm.correlation_matrix) > 1:
        st.subheader("Correlation Matrix")
        corr = rm.correlation_matrix.copy()
        corr.columns = [c.replace(".NS", "") for c in corr.columns]
        corr.index = [i.replace(".NS", "") for i in corr.index]
        fig_corr = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1, title="Return Correlations",
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    # Return distribution
    if rm.daily_returns is not None and not rm.daily_returns.empty:
        st.subheader("Return Distribution")
        fig_dist = px.histogram(
            rm.daily_returns, nbins=50,
            title="Portfolio Daily Return Distribution",
            labels={"value": "Daily Return", "count": "Frequency"},
        )
        fig_dist.add_vline(
            x=np.percentile(rm.daily_returns, 5),
            line_dash="dash", line_color="red",
            annotation_text="VaR 95%",
        )
        st.plotly_chart(fig_dist, use_container_width=True)


# =============================================================================
# Tab: Bulk Reports
# =============================================================================

def render_bulk_reports_tab(analyzer: IndianStockAnalyzer) -> None:
    st.header("Bulk Report Generation")
    st.info("Generate research reports for multiple stocks at once.")

    default_stocks = (
        "RELIANCE,TCS,INFY,HDFCBANK,ICICIBANK" if IS_CLOUD
        else "RELIANCE,TCS,INFY,HDFCBANK,ICICIBANK,WIPRO,BHARTIARTL,SBIN,KOTAKBANK,AXISBANK"
    )
    report_stocks = st.text_area("Enter stock symbols (comma-separated)", default_stocks)

    if not st.button("Generate Bulk Reports", type="primary"):
        return

    stocks = [
        f"{sanitize_symbol_input(s)}.NS"
        for s in report_stocks.split(",") if s.strip()
    ]
    if IS_CLOUD and len(stocks) > Config.MAX_BULK_REPORTS_CLOUD:
        st.warning(f"Limiting to {Config.MAX_BULK_REPORTS_CLOUD} stocks for cloud")
        stocks = stocks[: Config.MAX_BULK_REPORTS_CLOUD]

    if not stocks:
        st.warning("No valid symbols.")
        return

    progress_bar = st.progress(0)
    reports: Dict[str, str] = {}

    for i, symbol in enumerate(stocks):
        progress_bar.progress((i + 1) / len(stocks))

        with st.spinner(f"Analyzing {symbol}…"):
            res = analyzer.fetch_stock_bundle(symbol, period=analyzer.default_period)
            if not res.is_ok or res.value.fundamentals is None:
                continue

            bundle = res.value
            df = analyzer.calculate_technical_indicators(bundle.history)
            scores = analyzer.calculate_all_scores(bundle.fundamentals, df)

            if st.session_state.api_key and analyzer.model:
                report = analyzer.generate_ai_report(symbol, bundle.fundamentals, scores)
            else:
                report = analyzer.generate_basic_report(symbol, bundle.fundamentals, scores)
            reports[symbol] = report

    progress_bar.empty()

    if not reports:
        st.warning("No reports generated.")
        return

    for sym, rpt in reports.items():
        with st.expander(f"Report: {sym.replace('.NS', '')}"):
            st.text(rpt)

    combined = "\n\n" + ("=" * 80 + "\n\n").join(
        f"REPORT FOR {s}\n{r}" for s, r in reports.items()
    )
    st.download_button(
        "📥 Download All Reports", data=combined,
        file_name=f"bulk_reports_{datetime.now():%Y%m%d}.txt",
        mime="text/plain",
    )


# =============================================================================
# Tab: Market Overview
# =============================================================================

def render_market_overview_tab(analyzer: IndianStockAnalyzer) -> None:
    st.header("Market Overview")

    # Indices
    st.subheader("📊 Major Indices")
    idx_cols = st.columns(len(MARKET_INDICES))
    for i, (sym, name) in enumerate(MARKET_INDICES.items()):
        with idx_cols[i]:
            try:
                idx_data = yf.Ticker(sym).history(period="2d")
                if not idx_data.empty and len(idx_data) >= 2:
                    cur = idx_data.iloc[-1]["Close"]
                    prev = idx_data.iloc[-2]["Close"]
                    ch = ((cur - prev) / prev * 100) if prev != 0 else 0
                    st.metric(name, f"{cur:,.2f}", delta=f"{ch:.2f}%")
                else:
                    st.metric(name, "N/A")
            except Exception as exc:
                logger.warning("Index %s: %s", sym, exc)
                st.metric(name, "N/A")

    # Sector heat map
    st.subheader("📈 Sector Performance")
    sector_perf: List[Dict] = []

    with st.spinner("Loading sector data…"):
        for sector, stocks in SECTOR_REPRESENTATIVE_STOCKS.items():
            changes: List[float] = []
            for stk in stocks:
                try:
                    d = yf.Ticker(stk).history(period="2d")
                    if not d.empty and len(d) >= 2:
                        prev_c = d.iloc[-2]["Close"]
                        if prev_c > 0:
                            changes.append(
                                (d.iloc[-1]["Close"] - prev_c) / prev_c * 100
                            )
                except Exception:
                    pass
            if changes:
                sector_perf.append({
                    "Sector": sector,
                    "Performance": sum(changes) / len(changes),
                })

    if sector_perf:
        df_sec = pd.DataFrame(sector_perf)
        fig = px.bar(
            df_sec, x="Sector", y="Performance",
            color="Performance", color_continuous_scale=["red", "yellow", "green"],
            title="Sector Performance (Day)",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Unable to load sector data")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    _init_session_state()

    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(
        '<h1 class="main-header">🏛️ Indian Equity Research Platform</h1>',
        unsafe_allow_html=True,
    )

    if IS_CLOUD:
        st.info("☁️ Running on Streamlit Cloud – optimised performance mode")
    else:
        st.info("💻 Running locally – full feature mode")

    render_sidebar()

    analyzer = IndianStockAnalyzer(api_key=st.session_state.api_key)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Stock Analysis",
        "🔍 Stock Screener",
        "📈 Portfolio Analysis",
        "📑 Bulk Reports",
        "📉 Market Overview",
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
