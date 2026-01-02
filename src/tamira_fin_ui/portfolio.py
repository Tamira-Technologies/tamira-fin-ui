"""
Portfolio models and storage helpers for the fuzzy LSTM dashboard.

This module defines:
- Pydantic models to represent user portfolios containing up to 6 tickers.
- Validation ensuring selected tickers belong to curated blue chip lists across markets.
- Storage helpers to persist portfolios in a JSON file (configurable via environment).

Usage overview:
- Portfolios can mix tickers across USA, India, China, and Singapore markets.
- The default storage location is "~/.tamira_fin_ui/portfolios.json", configurable via
  the environment variable TAMIRA_STORAGE_PATH.
- Helper functions are provided to create, update, delete, and list portfolios.

Environment variables:
- TAMIRA_STORAGE_PATH: Override path to the portfolios JSON file.

Notes:
- This code uses Pydantic (v2) and Pydantic Settings for configuration management.
- All functions include explicit type annotations and docstrings per project guidelines.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings

from .bluechips import BLUECHIP_TICKERS, all_bluechips, get_markets


class Market(str, Enum):
    """
    Supported equity markets for curated blue chip tickers.

    Attributes
    ----------
    usa : str
        United States market identifier.
    india : str
        India (NSE) market identifier.
    china : str
        China (SSE/SZSE/SEHK) market identifier.
    singapore : str
        Singapore (SGX) market identifier.
    """

    usa = "usa"
    india = "india"
    china = "china"
    singapore = "singapore"


class Settings(BaseSettings):
    """
    Dashboard and storage configuration settings.

    Attributes
    ----------
    storage_path : str
        Filesystem path where portfolios are persisted in JSON format.
        Defaults to "~/.tamira_fin_ui/portfolios.json". Can be overridden by
        the environment variable TAMIRA_STORAGE_PATH.

    Methods
    -------
    json_path() -> Path
        Return a `Path` to the resolved storage file location, expanding user.
    """

    storage_path: str = "~/.tamira_fin_ui/portfolios.json"

    def json_path(self) -> Path:
        """
        Resolve and return the storage JSON path as a `Path`.

        Returns
        -------
        Path
            The portfolio storage file path with the user directory expanded.
        """
        return Path(self.storage_path).expanduser()


def is_valid_ticker(ticker: str) -> bool:
    """
    Check whether a ticker is present in the curated blue chip lists.

    Parameters
    ----------
    ticker : str
        Ticker symbol (Yahoo Finance format).

    Returns
    -------
    bool
        True if the ticker is in any supported market's blue chip list; otherwise False.
    """
    ticker_clean = ticker.strip()
    for _market, tickers in BLUECHIP_TICKERS.items():
        if ticker_clean in tickers:
            return True
    return False


def normalize_ticker(ticker: str) -> str:
    """
    Normalize a ticker string for consistent storage.

    Parameters
    ----------
    ticker : str
        Raw ticker symbol string.

    Returns
    -------
    str
        Trimmed ticker preserving case (suffixes like `.NS`/`.SI` are case-sensitive).
    """
    return ticker.strip()


class Portfolio(BaseModel):
    """
    A user portfolio consisting of up to six blue chip tickers across markets.

    Attributes
    ----------
    name : str
        User-defined portfolio name. Must be unique within the store.
    tickers : list[str]
        List of selected tickers (max size 6). Tickers must belong to curated lists.
    created_at : datetime
        UTC timestamp marking when the portfolio was first created.
    notes : str | None
        Optional notes for the portfolio.

    Validation
    ----------
    - Removes duplicate tickers while preserving input order.
    - Enforces maximum size of 6 tickers.
    - Ensures each ticker is available in curated blue chip lists.
    """

    name: str = Field(..., min_length=1, description="Unique portfolio name.")
    tickers: list[str] = Field(default_factory=list, description="Up to six curated tickers.")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    notes: str | None = Field(default=None, description="Optional user notes.")

    @field_validator("tickers")
    @classmethod
    def _validate_tickers(cls, values: list[str]) -> list[str]:
        """
        Validate and normalize tickers:
        - Strip whitespace.
        - Remove duplicates, preserve original order.
        - Enforce maximum size of six tickers.
        - Ensure each ticker exists within curated blue chip lists.

        Parameters
        ----------
        values : list[str]
            Input tickers from the user.

        Returns
        -------
        list[str]
            Normalized, validated tickers.

        Raises
        ------
        ValueError
            If any ticker is invalid or the portfolio exceeds six tickers.
        """
        seen: set[str] = set()
        normalized: list[str] = []
        for raw in values:
            t = normalize_ticker(raw)
            if t not in seen:
                seen.add(t)
                normalized.append(t)

        if len(normalized) == 0:
            raise ValueError("Portfolio must contain at least one ticker.")

        if len(normalized) > 6:
            raise ValueError("Portfolio cannot contain more than 6 tickers.")

        invalid: list[str] = [t for t in normalized if not is_valid_ticker(t)]
        if invalid:
            markets = ", ".join(get_markets())
            raise ValueError(
                f"Invalid tickers: {', '.join(invalid)}. Select from curated blue chips across markets: {markets}."
            )

        return normalized

    @model_validator(mode="after")
    def _validate_model(self) -> Portfolio:
        """
        Model-level validation hook to ensure name and tickers constraints are satisfied.

        Returns
        -------
        Portfolio
            The validated portfolio instance.

        Raises
        ------
        ValueError
            If the name is empty or tickers exceed the maximum allowed size.
        """
        if not self.name.strip():
            raise ValueError("Portfolio name cannot be empty.")
        if len(self.tickers) > 6:
            raise ValueError("Portfolio cannot contain more than 6 tickers.")
        return self


class PortfolioStore(BaseModel):
    """
    An in-memory collection of portfolios and serialization helpers.

    Attributes
    ----------
    portfolios : dict[str, Portfolio]
        Mapping from portfolio name to `Portfolio`.

    Methods
    -------
    add(portfolio: Portfolio) -> None
        Add a new portfolio (name must be unique).
    upsert(portfolio: Portfolio) -> None
        Insert or overwrite a portfolio by name.
    remove(name: str) -> None
        Remove a portfolio by name.
    get(name: str) -> Portfolio | None
        Fetch a portfolio by name.
    list_names() -> list[str]
        Return all portfolio names sorted lexicographically.
    list() -> list[Portfolio]
        Return all portfolios sorted by name.
    to_dict() -> dict[str, list[dict[str, object]]]
        Serialize the store to a JSON-compatible dictionary.
    from_dict(data: dict[str, object]) -> PortfolioStore
        Construct a store from a JSON-compatible dictionary.
    """

    portfolios: dict[str, Portfolio] = Field(default_factory=dict)

    def add(self, portfolio: Portfolio) -> None:
        """
        Add a new portfolio. Raises if the name already exists.

        Parameters
        ----------
        portfolio : Portfolio
            The portfolio to add.

        Raises
        ------
        ValueError
            If a portfolio with the same name already exists.
        """
        key = portfolio.name
        if key in self.portfolios:
            raise ValueError(f"A portfolio named '{key}' already exists.")
        self.portfolios[key] = portfolio

    def upsert(self, portfolio: Portfolio) -> None:
        """
        Insert or overwrite a portfolio by name.

        Parameters
        ----------
        portfolio : Portfolio
            The portfolio to upsert.

        Returns
        -------
        None
        """
        self.portfolios[portfolio.name] = portfolio

    def remove(self, name: str) -> None:
        """
        Remove a portfolio by name. No-op if not present.

        Parameters
        ----------
        name : str
            Portfolio name to remove.

        Returns
        -------
        None
        """
        self.portfolios.pop(name, None)

    def get(self, name: str) -> Portfolio | None:
        """
        Fetch a portfolio by name.

        Parameters
        ----------
        name : str
            Portfolio name.

        Returns
        -------
        Portfolio | None
            The portfolio if found; otherwise None.
        """
        return self.portfolios.get(name)

    def list_names(self) -> list[str]:
        """
        Return all portfolio names sorted lexicographically.

        Returns
        -------
        list[str]
            Sorted portfolio names.
        """
        return sorted(self.portfolios.keys())

    def list(self) -> list[Portfolio]:
        """
        Return all portfolios sorted by name.

        Returns
        -------
        list[Portfolio]
            List of portfolios sorted by name.
        """
        return [self.portfolios[name] for name in self.list_names()]

    def to_dict(self) -> dict[str, list[dict[str, object]]]:
        """
        Serialize the store to a JSON-compatible dictionary.

        Returns
        -------
        dict[str, list[dict[str, object]]]
            A dictionary with a single key "portfolios" whose value is a list of
            serialized portfolio dictionaries.
        """
        return {"portfolios": [p.model_dump(mode="python") for p in self.list()]}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> PortfolioStore:
        """
        Construct a `PortfolioStore` from a JSON-compatible dictionary.

        Parameters
        ----------
        data : dict[str, object]
            A dictionary potentially containing a "portfolios" key with a list
            of serialized portfolio dictionaries.

        Returns
        -------
        PortfolioStore
            The reconstructed `PortfolioStore` instance.
        """
        raw_list = data.get("portfolios") if isinstance(data, dict) else None
        portfolios: dict[str, Portfolio] = {}
        if isinstance(raw_list, list):
            for item in raw_list:
                if isinstance(item, dict):
                    p = Portfolio(**item)
                    portfolios[p.name] = p
        return cls(portfolios=portfolios)


def ensure_storage_dir(path: Path) -> None:
    """
    Ensure the parent directory for the given path exists.

    Parameters
    ----------
    path : Path
        Target file path for portfolio storage.

    Returns
    -------
    None
    """
    parent = path.parent
    if not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)


def load_store(settings: Settings | None = None) -> PortfolioStore:
    """
    Load the `PortfolioStore` from disk. Returns an empty store if the file does not exist.

    Parameters
    ----------
    settings : Settings | None
        Optional settings instance. If None, defaults are used.

    Returns
    -------
    PortfolioStore
        The loaded or newly initialized store.
    """
    cfg = settings or Settings()
    path = cfg.json_path()
    if not path.exists():
        return PortfolioStore()
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return PortfolioStore.from_dict(data)
    except Exception:
        # Robustness: on any parsing error, fall back to an empty store.
        # For observability, consider logging the exception at the UI layer.
        return PortfolioStore()


def save_store(store: PortfolioStore, settings: Settings | None = None) -> None:
    """
    Persist the `PortfolioStore` to disk using an atomic write.

    Parameters
    ----------
    store : PortfolioStore
        The store to persist.
    settings : Settings | None
        Optional settings instance. If None, defaults are used.

    Returns
    -------
    None
    """
    cfg = settings or Settings()
    path = cfg.json_path()
    ensure_storage_dir(path)
    tmp = path.with_suffix(".tmp")

    payload = store.to_dict()
    content = json.dumps(payload, ensure_ascii=False, indent=2, default=str)

    with tmp.open("w", encoding="utf-8") as f:
        f.write(content)
        f.flush()

    tmp.replace(path)


def add_portfolio(
    name: str,
    tickers: list[str],
    notes: str | None = None,
    settings: Settings | None = None,
) -> Portfolio:
    """
    Create and persist a new portfolio.

    Parameters
    ----------
    name : str
        Portfolio name (unique).
    tickers : list[str]
        Up to six curated tickers.
    notes : str | None
        Optional notes.
    settings : Settings | None
        Optional settings instance.

    Returns
    -------
    Portfolio
        The newly created portfolio.

    Raises
    ------
    ValueError
        If a portfolio with the same name already exists.
    """
    store = load_store(settings)
    portfolio = Portfolio(name=name, tickers=tickers, notes=notes)
    store.add(portfolio)
    save_store(store, settings)
    return portfolio


def update_portfolio(
    name: str,
    tickers: list[str] | None = None,
    notes: str | None = None,
    settings: Settings | None = None,
) -> Portfolio:
    """
    Update and persist an existing portfolio.

    Parameters
    ----------
    name : str
        Portfolio name to update.
    tickers : list[str] | None
        New tickers to set. If None, tickers remain unchanged.
    notes : str | None
        New notes to set. If None, notes remain unchanged.
    settings : Settings | None
        Optional settings instance.

    Returns
    -------
    Portfolio
        The updated portfolio.

    Raises
    ------
    ValueError
        If the portfolio does not exist.
    """
    store = load_store(settings)
    existing = store.get(name)
    if existing is None:
        raise ValueError(f"No portfolio named '{name}' exists.")

    new_tickers = existing.tickers if tickers is None else tickers
    new_notes = existing.notes if notes is None else notes

    updated = Portfolio(
        name=existing.name,
        tickers=new_tickers,
        created_at=existing.created_at,
        notes=new_notes,
    )
    store.upsert(updated)
    save_store(store, settings)
    return updated


def remove_portfolio(name: str, settings: Settings | None = None) -> None:
    """
    Delete a portfolio by name.

    Parameters
    ----------
    name : str
        Portfolio name to remove.
    settings : Settings | None
        Optional settings instance.

    Returns
    -------
    None
    """
    store = load_store(settings)
    store.remove(name)
    save_store(store, settings)


def get_portfolio(name: str, settings: Settings | None = None) -> Portfolio | None:
    """
    Fetch a portfolio by name.

    Parameters
    ----------
    name : str
        Portfolio name.
    settings : Settings | None
        Optional settings instance.

    Returns
    -------
    Portfolio | None
        The portfolio if found; otherwise None.
    """
    store = load_store(settings)
    return store.get(name)


def list_portfolios(settings: Settings | None = None) -> list[Portfolio]:
    """
    List all portfolios sorted by name.

    Parameters
    ----------
    settings : Settings | None
        Optional settings instance.

    Returns
    -------
    list[Portfolio]
        All portfolios sorted by name.
    """
    store = load_store(settings)
    return store.list()


def available_bluechips() -> dict[str, list[str]]:
    """
    Return the full mapping of markets to curated blue chip tickers.

    Returns
    -------
    dict[str, list[str]]
        A dictionary mapping market keys to their 10-ticker lists.
    """
    return all_bluechips()
