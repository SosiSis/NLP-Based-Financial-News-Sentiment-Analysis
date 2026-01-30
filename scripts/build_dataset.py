from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - only needed when downloading
    yf = None

# Paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Data"
NEWS_PATH = DATA_DIR / "raw_analyst_ratings.csv" / "raw_analyst_ratings.csv"
PRICES_DIR = DATA_DIR / "yfinance_data"
OUTPUT_PATH = DATA_DIR / "merged_news_prices.csv"

# Map common legacy tickers to current symbols so we do not lose rows when merging
TICKER_ALIASES: dict[str, str] = {
    "GOOGL": "GOOG",
    "FB": "META",
}


def normalize_ticker(value: str) -> str:
    """Uppercase and map legacy symbols to current tickers."""
    if not isinstance(value, str):
        return ""
    value = value.upper().strip()
    return TICKER_ALIASES.get(value, value)


def load_news(path: Path) -> pd.DataFrame:
    """Load and aggregate FNSPID news by date and ticker."""
    news = pd.read_csv(
        path,
        usecols=["headline", "date", "stock"],
        dtype={"stock": "string"},
    ).rename(columns={"stock": "Ticker"})

    news = news.dropna(subset=["headline", "date", "Ticker"])
    news["Ticker"] = news["Ticker"].apply(normalize_ticker)

    # Parse timezone-aware timestamps and extract local trading date.
    # The date column has timestamps like "2020-06-05 10:30:54-04:00"
    news["date_parsed"] = pd.to_datetime(news["date"], utc=True, errors="coerce")
    news = news.dropna(subset=["date_parsed"])
    news["Date"] = news["date_parsed"].dt.tz_convert("America/New_York").dt.date
    news = news.drop(columns=["date", "date_parsed"])

    aggregated = (
        news.groupby(["Date", "Ticker"], as_index=False)["headline"]
        .agg(" ".join)
        .rename(columns={"headline": "Headlines"})
    )
    return aggregated


def load_prices(directory: Path, tickers: Iterable[str] | None = None) -> pd.DataFrame:
    """Load price CSVs from disk and optionally filter to a ticker subset.

    Handles both yfinance-style files (with "Adj Close", dividends, splits) and
    simpler OHLCV files lacking "Adj Close" by filling missing columns with NaN.
    """
    tickers_set = set(t.upper() for t in tickers) if tickers else None
    frames: list[pd.DataFrame] = []
    base_cols = ["Date", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]

    for csv_path in directory.glob("*_historical_data.csv"):
        ticker = csv_path.name.split("_")[0].upper()
        if tickers_set and ticker not in tickers_set:
            continue

        df = pd.read_csv(csv_path, parse_dates=["Date"])
        # Build a unified frame with expected columns; fill missing ones with NaN
        out = pd.DataFrame()
        out["Date"] = df["Date"].dt.date
        out["Ticker"] = ticker
        for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
            if col in df.columns:
                out[col] = df[col]
            elif col == "Adj Close" and "Close" in df.columns:
                # Backfill adjusted close with close when not provided
                out[col] = df["Close"]
            else:
                out[col] = pd.NA

        frames.append(out[base_cols])

    if not frames:
        raise FileNotFoundError(f"No price files found under {directory}")

    return pd.concat(frames, ignore_index=True)


def download_prices(
    tickers: Sequence[str],
    start: date,
    end: date,
    directory: Path,
) -> None:
    """Download missing ticker price history with yfinance."""
    if yf is None:
        raise RuntimeError("yfinance is required to download prices. Install it with `pip install yfinance`.")

    directory.mkdir(parents=True, exist_ok=True)
    for ticker in tickers:
        out_path = directory / f"{ticker}_historical_data.csv"
        if out_path.exists():
            continue
        try:
            hist = yf.download(ticker, start=start, end=end, progress=False)
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"Failed to download {ticker}: {exc}")
            continue
        if hist.empty:
            print(f"No price data returned for {ticker}; skipping")
            continue
        hist = hist.reset_index().rename(columns={"Adj Close": "Adj Close"})
        hist.to_csv(out_path, index=False)
        print(f"Saved prices for {ticker} to {out_path}")


def label_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add next-day direction label per ticker."""
    df = df.sort_values(["Ticker", "Date"]).copy()
    df["next_close"] = df.groupby("Ticker")["Close"].shift(-1)
    df["Target"] = (df["next_close"] > df["Close"]).astype("Int64")
    df = df.dropna(subset=["next_close"])
    return df.drop(columns=["next_close"])


def main(
    target_tickers: Sequence[str] | None = None,
    download_missing: bool = False,
    max_tickers: int | None = None,
    download_limit: int | None = None,
) -> None:
    news = load_news(NEWS_PATH)

    # If no explicit tickers are provided, keep everything from the news file.
    if target_tickers is None:
        target_tickers = sorted(news["Ticker"].unique())
    else:
        target_tickers = [normalize_ticker(t) for t in target_tickers]

    # Optionally limit to the most frequent tickers, preferring tickers with price files on disk
    if max_tickers:
        counts = news["Ticker"].value_counts()
        freq_order = list(counts.index)
        available = {p.name.split("_")[0].upper() for p in PRICES_DIR.glob("*_historical_data.csv")}
        first = [t for t in freq_order if t in available][:max_tickers]
        remaining = max_tickers - len(first)
        second = [t for t in freq_order if t not in available][:max(0, remaining)]
        target_tickers = first + second

    news = news[news["Ticker"].isin(target_tickers)]

    if news.empty:
        raise ValueError("No news rows left after filtering; check ticker list")

    # Download any missing price files to cover the news date range.
    if download_missing:
        start_date = news["Date"].min()
        end_date = news["Date"].max() + timedelta(days=1)
        missing = [t for t in target_tickers if not (PRICES_DIR / f"{t}_historical_data.csv").exists()]
        if missing:
            # Prioritize tickers by news frequency, then cap to download_limit if provided
            counts = news["Ticker"].value_counts()
            missing_sorted = sorted(missing, key=lambda t: counts.get(t, 0), reverse=True)
            if download_limit is not None:
                missing_sorted = missing_sorted[:download_limit]
            print(
                f"Downloading {len(missing_sorted)} of {len(missing)} missing tickers: "
                f"{missing_sorted[:10]}{'...' if len(missing_sorted) > 10 else ''}"
            )
            download_prices(missing_sorted, start=start_date, end=end_date, directory=PRICES_DIR)
        else:
            print("All requested tickers already have price files on disk.")

    prices = load_prices(PRICES_DIR, tickers=target_tickers)
    # Keep only rows that have news and prices to avoid NaN headlines
    merged = prices.merge(news, on=["Date", "Ticker"], how="inner")
    labeled = label_targets(merged)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(OUTPUT_PATH, index=False)
    print(f"Merged dataset written to {OUTPUT_PATH} with {len(labeled)} rows")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build merged news/price dataset")
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help="Optional list of tickers to include. Defaults to all tickers found in the news file.",
    )
    parser.add_argument(
        "--download-missing",
        action="store_true",
        help="Download missing price history with yfinance for the requested tickers.",
    )
    parser.add_argument(
        "--max-tickers",
        type=int,
        default=None,
        help="Keep only the top N tickers by news count (useful to avoid thousands of downloads).",
    )
    parser.add_argument(
        "--download-limit",
        type=int,
        default=None,
        help="Maximum number of missing tickers to download (prioritized by news frequency).",
    )

    args = parser.parse_args()
    main(
        target_tickers=args.tickers,
        download_missing=args.download_missing,
        max_tickers=args.max_tickers,
        download_limit=args.download_limit,
    )
