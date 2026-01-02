"""
Blue chip stock lists per market (USA, India, China, Singapore) with Yahoo Finance tickers.

This module provides curated lists of 10 blue chip tickers for each market using
Yahoo Finance ticker conventions:
- USA: standard tickers (e.g., AAPL, MSFT).
- India (NSE): suffix `.NS` (e.g., RELIANCE.NS, TCS.NS).
- China: Shanghai `.SS`, Shenzhen `.SZ`, and Hong Kong `.HK` tickers.
- Singapore (SGX): suffix `.SI` (e.g., D05.SI for DBS).

These lists are intended for UI selection, portfolio construction, and driving
model forecasts in the dashboard.

Note:
- Tickers are representative and stable as of common usage on Yahoo Finance.
- If a ticker changes or is unavailable in your region, you can adjust the lists here.
"""

# USA blue chips (10)
USA_BLUECHIPS: list[str] = [
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "AMZN",  # Amazon
    "GOOGL",  # Alphabet (Class A)
    "META",  # Meta Platforms
    "NVDA",  # NVIDIA
    "JPM",  # JPMorgan Chase
    "JNJ",  # Johnson & Johnson
    "XOM",  # Exxon Mobil
    "BRK-B",  # Berkshire Hathaway Class B
]

# India (NSE) blue chips (10)
INDIA_BLUECHIPS: list[str] = [
    "RELIANCE.NS",  # Reliance Industries
    "TCS.NS",  # Tata Consultancy Services
    "HDFCBANK.NS",  # HDFC Bank
    "ICICIBANK.NS",  # ICICI Bank
    "INFY.NS",  # Infosys
    "LT.NS",  # Larsen & Toubro
    "SBIN.NS",  # State Bank of India
    "HINDUNILVR.NS",  # Hindustan Unilever
    "BHARTIARTL.NS",  # Bharti Airtel
    "TITAN.NS",  # Titan Company
]

# China blue chips (mixed SSE/SEHK; 10)
CHINA_BLUECHIPS: list[str] = [
    "601398.SS",  # Industrial & Commercial Bank of China (ICBC) - Shanghai
    "601939.SS",  # China Construction Bank - Shanghai
    "601988.SS",  # Bank of China - Shanghai
    "601318.SS",  # Ping An Insurance - Shanghai
    "601857.SS",  # PetroChina - Shanghai
    "600519.SS",  # Kweichow Moutai - Shanghai
    "0700.HK",  # Tencent - Hong Kong
    "0941.HK",  # China Mobile - Hong Kong
    "1211.HK",  # BYD Company - Hong Kong
    "0390.HK",  # China Railway Group - Hong Kong
]

# Singapore (SGX) blue chips (10)
SINGAPORE_BLUECHIPS: list[str] = [
    "D05.SI",  # DBS Group
    "O39.SI",  # OCBC Bank
    "U11.SI",  # UOB
    "Z74.SI",  # Singtel
    "C6L.SI",  # Singapore Airlines
    "BN4.SI",  # Keppel Corp
    "S63.SI",  # ST Engineering
    "9CI.SI",  # CapitaLand Investment
    "F34.SI",  # Wilmar International
    "U96.SI",  # Sembcorp Industries
]

# Mapping for convenience
BLUECHIP_TICKERS: dict[str, list[str]] = {
    "usa": USA_BLUECHIPS,
    "india": INDIA_BLUECHIPS,
    "china": CHINA_BLUECHIPS,
    "singapore": SINGAPORE_BLUECHIPS,
}


def get_markets() -> list[str]:
    """
    Return the list of supported markets.

    Returns
    -------
    list[str]
        Markets: ["usa", "india", "china", "singapore"].
    """
    return list(BLUECHIP_TICKERS.keys())


def get_bluechips(market: str) -> list[str]:
    """
    Get the list of blue chip tickers for a given market key.

    Parameters
    ----------
    market : str
        One of "usa", "india", "china", or "singapore" (case-insensitive).

    Returns
    -------
    list[str]
        List of 10 blue chip tickers for the selected market.

    Raises
    ------
    KeyError
        If the market is not supported.
    """
    key = market.lower()
    if key not in BLUECHIP_TICKERS:
        raise KeyError(f"Unsupported market '{market}'. Supported: {', '.join(get_markets())}")
    return BLUECHIP_TICKERS[key]


def all_bluechips() -> dict[str, list[str]]:
    """
    Return the full mapping of markets to blue chip tickers.

    Returns
    -------
    dict[str, list[str]]
        A dictionary mapping market keys to their 10-ticker lists.
    """
    return BLUECHIP_TICKERS.copy()


def validate_market(market: str) -> bool:
    """
    Validate whether a market key is supported.

    Parameters
    ----------
    market : str
        Market identifier to validate.

    Returns
    -------
    bool
        True if the market is supported; otherwise False.
    """
    return market.lower() in BLUECHIP_TICKERS
