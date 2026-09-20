"""NBP front-month price ingestion, cleaning, target construction, and summary.

This module turns a raw daily price CSV (exported from TradingView or
MarketWatch - see README / project plan) into a clean, analysis-ready table
stored as Parquet, and prints a summary plus saves a diagnostic plot.

Design notes
------------
* We only need DAILY data (one settlement per trading day); the prediction
  target is a daily directional move, so intraday ticks add nothing.
* The loader is deliberately tolerant of column-name and formatting variants
  because TradingView and MarketWatch export slightly different CSV shapes.
* A "continuous"/front-month series can contain artificial jumps on contract
  roll days. Those are NOT genuine one-day price moves, so we FLAG large
  daily returns for inspection rather than silently trusting them as labels.

Run it (from the repo root, env `nbp` active):

    python -m src.data_ingestion.fetch_nbp --input data/raw/nbp_history.csv

Outputs:
    data/processed/nbp_prices.parquet   cleaned table with target column
    output/nbp_summary.png              price + returns diagnostic plot
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

# Use a non-interactive backend so the script works headless (no GUI needed).
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402

# ---------------------------------------------------------------------------
# Column-name aliases. Different exporters label the same quantity differently;
# we normalise everything to a canonical lower-case set. `close` is the only
# strictly required column - open/high/low/volume are kept if present.
# ---------------------------------------------------------------------------
COLUMN_ALIASES = {
    "date": {"date", "time", "timestamp", "datetime"},
    "open": {"open"},
    "high": {"high"},
    "low": {"low"},
    "close": {"close", "price", "last", "settle", "settlement", "adj close", "adj_close"},
    "volume": {"volume", "vol"},
}

# A daily return larger in magnitude than this is FLAGGED for manual review as
# a possible contract-roll artefact or bad tick. It is not auto-deleted: gas
# genuinely moves double digits in crises, so a human should judge borderline
# cases. 0.25 = 25% one-day move.
ROLL_FLAG_RETURN_THRESHOLD = 0.25


def _canonicalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename incoming columns to our canonical names using COLUMN_ALIASES.

    Why: downstream code should never have to care whether the source called
    the price column "Close", "Price", or "Settle". We resolve that here once.
    """
    lower_map = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=lower_map)

    rename = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for col in df.columns:
            if col in aliases:
                rename[col] = canonical
                break
    df = df.rename(columns=rename)

    if "date" not in df.columns or "close" not in df.columns:
        raise ValueError(
            "Input CSV must contain a date column and a close/price column. "
            f"Found columns after normalisation: {list(df.columns)}"
        )
    return df


def _to_numeric(series: pd.Series) -> pd.Series:
    """Coerce a price column to float, tolerating thousands separators.

    Why: MarketWatch exports can contain values like "1,234.5" (string with a
    comma). We strip commas before converting so we don't lose rows.
    """
    if series.dtype == object:
        series = series.str.replace(",", "", regex=False)
    return pd.to_numeric(series, errors="coerce")


def load_raw_csv(path: Path) -> pd.DataFrame:
    """Load and normalise a raw price CSV into canonical columns.

    Returns a frame with at least [date, close], plus any of
    [open, high, low, volume] that were present, sorted by date ascending.
    """
    df = pd.read_csv(path)
    df = _canonicalise_columns(df)

    # Parse dates. `dayfirst` is left to pandas inference on ISO dates, but we
    # try a second pass with dayfirst=True if the first pass produced NaT (some
    # UK exporters use DD/MM/YYYY).
    dates = pd.to_datetime(df["date"], errors="coerce")
    if dates.isna().any():
        dates_alt = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
        dates = dates.fillna(dates_alt)
    df["date"] = dates.dt.normalize()  # drop any time-of-day component

    # Coerce all present numeric price columns.
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = _to_numeric(df[col])

    df = df.dropna(subset=["date", "close"])
    df = df.sort_values("date").drop_duplicates(subset="date", keep="last")
    return df.reset_index(drop=True)


def clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the price series and add diagnostic/derived columns.

    Adds:
        prev_close   previous trading-day close
        ret          simple daily return (close / prev_close - 1)
        gap_days     calendar days since the previous observation (reveals
                     weekends = 3 on Mondays, and holidays = larger gaps)
        weekday      0=Mon .. 6=Sun (expected to be 0-4 only for a futures series)
        roll_flag    True if |ret| exceeds the roll/outlier threshold
    """
    df = df.copy()

    # Previous close and daily return. The return is the raw material for the
    # directional target; alignment is one row = one trading day.
    df["prev_close"] = df["close"].shift(1)
    df["ret"] = df["close"] / df["prev_close"] - 1.0

    # Calendar-day gap since the previous row. A normal Mon-Fri series shows
    # gap_days = 1 on Tue-Fri and 3 on Mondays; anything larger signals a
    # holiday or a missing stretch of data.
    df["gap_days"] = df["date"].diff().dt.days
    df["weekday"] = df["date"].dt.weekday

    # Flag suspiciously large moves for manual review (possible roll artefact
    # or bad tick). Not dropped automatically - see module docstring.
    df["roll_flag"] = df["ret"].abs() > ROLL_FLAG_RETURN_THRESHOLD

    return df.reset_index(drop=True)


# -------------------------------------------------
# CONCEPT NOTE: Binary classification target (label)
# -------------------------------------------------
# What it is: In supervised ML, the "target" (or "label"), usually called y,
#   is the answer we want the model to predict. Here it is a single 0/1 value
#   per day: 1 if the price went up, 0 if it went down (or was unchanged).
# Why we use it here: Our question is purely directional ("up or down?"), which
#   is a binary classification problem - the simplest, most robust framing for
#   a small, noisy dataset.
# Common pitfall: Look-ahead leakage. The label for day t uses that day's own
#   close, so any FEATURE must be knowable BEFORE the close (our weather comes
#   from the 06:30 00z run - safe). Never build features from the same close
#   that defines the label.
# -------------------------------------------------
def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add the binary directional target.

    target = 1 if today's close > previous close, else 0.

    This matches the same-day operational setup: today's 00z weather (known at
    06:30) predicts the sign of today's settlement change. The first row has no
    previous close, so its target is left as NaN and dropped by callers that
    need a complete label column.
    """
    df = df.copy()
    df["target"] = (df["ret"] > 0).astype("float")
    df.loc[df["ret"].isna(), "target"] = np.nan
    return df


def summarise(df: pd.DataFrame) -> None:
    """Print a plain-text summary: coverage, gaps, weekends, flags, balance."""
    print("=" * 62)
    print("NBP price data - summary")
    print("=" * 62)

    print(f"Rows (trading days) : {len(df)}")
    print(f"Date range          : {df['date'].min().date()} -> {df['date'].max().date()}")
    span_years = (df["date"].max() - df["date"].min()).days / 365.25
    print(f"Span                : {span_years:.1f} years")

    # Weekend contamination check: a clean futures series should be Mon-Fri.
    n_weekend = int((df["weekday"] >= 5).sum())
    print(f"Weekend rows        : {n_weekend} (expected 0 for a futures series)")

    # Gaps: holidays and missing stretches show up as gap_days > 3.
    big_gaps = df[df["gap_days"] > 3]
    print(f"Gaps > 3 calendar d : {len(big_gaps)} (holidays / missing data)")
    if len(big_gaps):
        worst = big_gaps.nlargest(5, "gap_days")[["date", "gap_days"]]
        print("  largest gaps:")
        for _, r in worst.iterrows():
            print(f"    {r['date'].date()}  ({int(r['gap_days'])} days since prev)")

    # Roll / outlier flags for inspection.
    flagged = df[df["roll_flag"]]
    print(f"Flagged moves (|ret|>{ROLL_FLAG_RETURN_THRESHOLD:.0%}): {len(flagged)}")
    if len(flagged):
        for _, r in flagged.nlargest(5, "ret", keep="all").iterrows():
            print(f"    {r['date'].date()}  ret={r['ret']:+.1%}  close={r['close']:.2f}")

    # Class balance overall and in the heating season (the modelled window).
    valid = df.dropna(subset=["target"])
    up_all = valid["target"].mean()
    heating = valid[valid["date"].dt.month.isin(config.HEATING_SEASON_MONTHS)]
    up_heat = heating["target"].mean() if len(heating) else float("nan")
    print(f"Up-day fraction     : all={up_all:.1%}  heating-season={up_heat:.1%}")
    print(f"Heating-season rows : {len(heating)}  (this is the modelled sample)")

    # Flag the critical sample-size risk from the plan (<~500 labelled examples).
    if len(heating) < 500:
        print("  ** WARNING: fewer than 500 heating-season labels - model "
              "reliability is at risk (see plan). **")
    print("=" * 62)


def save_plot(df: pd.DataFrame, path: Path) -> None:
    """Save a two-panel diagnostic: price history and daily-return distribution.

    Flagged (possible roll) points are marked on the price line so the user can
    eyeball whether they are genuine crisis moves or contract-roll artefacts.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7))

    ax1.plot(df["date"], df["close"], lw=0.8, color="tab:blue")
    flagged = df[df["roll_flag"]]
    if len(flagged):
        ax1.scatter(flagged["date"], flagged["close"], s=18, color="tab:red",
                    zorder=5, label=f"flagged |ret|>{ROLL_FLAG_RETURN_THRESHOLD:.0%}")
        ax1.legend(loc="upper left", fontsize=8)
    ax1.set_title("NBP front-month close")
    ax1.set_ylabel("price")

    ax2.hist(df["ret"].dropna(), bins=80, color="tab:gray")
    ax2.set_title("Daily return distribution")
    ax2.set_xlabel("daily return")
    ax2.set_ylabel("count")

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"Saved diagnostic plot -> {path}")


def process(input_path: Path, output_parquet: Path, plot_path: Path) -> pd.DataFrame:
    """Full pipeline: load -> clean -> add target -> summarise -> save."""
    df = load_raw_csv(input_path)
    df = clean_prices(df)
    df = add_target(df)

    summarise(df)
    save_plot(df, plot_path)

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    print(f"Saved cleaned prices  -> {output_parquet}  ({len(df)} rows)")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest and clean NBP price data.")
    parser.add_argument(
        "--input",
        type=Path,
        default=config.RAW_NBP_FILE,
        help="Path to the raw price CSV exported from TradingView / MarketWatch.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.PROCESSED_NBP_FILE,
        help="Where to write the cleaned Parquet table.",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=config.OUTPUT_DIR / "nbp_summary.png",
        help="Where to write the diagnostic plot.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(
            f"Input file not found: {args.input}\n"
            "Export NBP daily history to that path first (see README, Phase 2)."
        )

    process(args.input, args.output, args.plot)


if __name__ == "__main__":
    main()
