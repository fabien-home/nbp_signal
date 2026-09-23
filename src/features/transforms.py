"""Physical and statistical transforms for the weather indices.

This module holds *pure functions* only - each takes arrays/Series in and
returns arrays/Series out, with no file or network access. Keeping them pure
means we can unit-test them with tiny synthetic inputs, and (crucially) reuse
the EXACT same code in offline training (ERA5) and live operation (IFS), so a
training feature and its live counterpart are computed identically.

Contents
--------
Meteorological transforms:
    kelvin_to_celsius   unit conversion
    wind_speed          100 m wind speed from U/V components
    wind_power_fraction wind speed -> normalised turbine generation [0, 1]
    heating_degree_days temperature -> heating demand proxy (HDD)

Feature-engineering transforms:
    day_of_year_encoding   cyclical seasonal position (sin/cos)
    climatology_by_doy     day-of-year mean/std over a reference period
    standardized_anomaly   (value - climatology mean) / climatology std
    innovation             day-over-day change ("forecast surprise" proxy)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Meteorological transforms
# ---------------------------------------------------------------------------


def kelvin_to_celsius(temp_k):
    """Convert temperature from kelvin (ERA5/IFS native) to degrees Celsius."""
    return np.asarray(temp_k, dtype=float) - 273.15


def wind_speed(u, v):
    """Scalar wind speed from orthogonal wind components.

    ERA5/IFS give wind as U (eastward) and V (northward) components. The speed
    that a turbine experiences is the vector magnitude sqrt(U^2 + V^2); the
    direction is irrelevant for generation, so we collapse to speed here.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    return np.sqrt(u * u + v * v)


def wind_power_fraction(speed, cut_in, rated, cut_out):
    """Convert wind speed (m/s) to normalised turbine generation in [0, 1].

    Why not use raw wind speed as the feature? Because generation is a strongly
    NONLINEAR function of speed, and that nonlinearity is exactly what matters
    for gas-for-power demand:

        speed < cut_in           -> 0     (too weak to spin)
        cut_in <= speed < rated  -> rises ~cubically (power ~ speed^3)
        rated <= speed <= cut_out-> 1     (capped at rated output)
        speed > cut_out          -> 0     (turbine shuts down for safety)

    Feeding this generation fraction to the model, rather than raw speed, means
    the model sees the quantity that actually drives the electricity market.
    Vectorised so it works on scalars, arrays, or gridded fields.
    """
    speed = np.asarray(speed, dtype=float)
    power = np.zeros_like(speed)

    # Ramp region: normalise speed^3 between cut-in and rated so output goes
    # smoothly from 0 at cut_in to 1 at rated.
    ramp = (speed >= cut_in) & (speed < rated)
    power[ramp] = (speed[ramp] ** 3 - cut_in ** 3) / (rated ** 3 - cut_in ** 3)

    # Rated plateau: full output between rated and cut-out.
    plateau = (speed >= rated) & (speed <= cut_out)
    power[plateau] = 1.0

    # Below cut-in and above cut-out remain 0 (already initialised).
    return power


def heating_degree_days(temp_c, base):
    """Heating Degree Days: a demand-weighted temperature transform.

    HDD = max(base - T, 0). Below the base temperature, buildings need heating
    and demand scales with how far below the base it is; at or above the base,
    HDD is 0 (no heating demand). This threshold-and-clip shape captures the
    real, nonlinear temperature->gas-demand relationship far better than a raw
    temperature anomaly, which would (wrongly) treat a warm summer deviation as
    just as meaningful as a cold winter one.
    """
    temp_c = np.asarray(temp_c, dtype=float)
    return np.maximum(base - temp_c, 0.0)


# ---------------------------------------------------------------------------
# Feature-engineering transforms
# ---------------------------------------------------------------------------

# -------------------------------------------------
# CONCEPT NOTE: Cyclical (sin/cos) encoding
# -------------------------------------------------
# What it is: A way to give a model a repeating quantity (like day-of-year)
#   without a false "jump". Day 365 and day 1 are one day apart, but as plain
#   numbers they look 364 apart. Encoding the angle as (sin, cos) puts the year
#   on a circle, so late December sits right next to early January.
# Why we use it here: Gas demand is seasonal; the model should know WHERE in the
#   season we are, with December and January treated as neighbours.
# Common pitfall: Using raw day-of-year (1..365) as a feature - the model then
#   sees an artificial discontinuity at the year boundary.
# -------------------------------------------------
def day_of_year_encoding(dates: pd.DatetimeIndex):
    """Return (sin, cos) of the day-of-year angle for each date.

    Uses 365.25 to keep leap years consistent. Returns two numpy arrays aligned
    with `dates`.
    """
    doy = np.asarray(dates.dayofyear, dtype=float)
    angle = 2.0 * np.pi * doy / 365.25
    return np.sin(angle), np.cos(angle)


def _circular_smooth(values: pd.Series, window_days: int) -> pd.Series:
    """Smooth a day-of-year series with a centred window, wrapping year-end.

    Day-of-year climatology must wrap: 1 January's neighbours include late
    December. We pad the series circularly, apply a centred rolling mean, then
    trim back. `values` is indexed 1..366.
    """
    if window_days <= 1:
        return values
    pad = window_days
    # Circular pad: take the tail before the head and the head after the tail.
    padded = pd.concat([values.iloc[-pad:], values, values.iloc[:pad]])
    smoothed = padded.rolling(window=window_days, center=True, min_periods=1).mean()
    # Trim the padding, restoring the original 1..366 index.
    trimmed = smoothed.iloc[pad:pad + len(values)]
    trimmed.index = values.index
    return trimmed


# -------------------------------------------------
# CONCEPT NOTE: Climatology and standardized anomaly
# -------------------------------------------------
# What it is: A "climatology" is the typical value for each calendar day,
#   averaged over a long reference period (here 1991-2020). An "anomaly" is how
#   far a given day departs from that typical value. A STANDARDIZED anomaly
#   divides that departure by the day's typical spread, giving a value in
#   "sigma" (standard-deviation) units.
# Why we use it here: The market reacts to weather being unusually cold/calm FOR
#   THE TIME OF YEAR, not to the absolute temperature. Anomalies remove the
#   seasonal cycle so the model sees the surprise. Sigma units also make
#   temperature and wind indices directly comparable (the daily signal prints
#   both in sigma).
# Common pitfall: Computing the climatology over ALL years including the test
#   period leaks information. We fix the reference period (1991-2020) up front.
# -------------------------------------------------
def climatology_by_doy(
    daily: pd.Series,
    ref_start_year: int,
    ref_end_year: int,
    smooth_days: int,
):
    """Compute smoothed day-of-year mean and std over a reference period.

    Parameters
    ----------
    daily : pd.Series
        Daily values indexed by a DatetimeIndex.
    ref_start_year, ref_end_year : int
        Inclusive reference window for the climatology (e.g. 1991..2020).
    smooth_days : int
        Centred smoothing window applied circularly across the year boundary.

    Returns
    -------
    (clim_mean, clim_std) : tuple of pd.Series
        Each indexed 1..366 (day-of-year).
    """
    ref = daily[(daily.index.year >= ref_start_year) & (daily.index.year <= ref_end_year)]
    if ref.empty:
        raise ValueError(
            f"No data in the reference period {ref_start_year}-{ref_end_year}; "
            "cannot build a climatology."
        )

    doy = ref.index.dayofyear
    clim_mean = ref.groupby(doy).mean()
    clim_std = ref.groupby(doy).std()

    # Ensure a complete 1..366 index, filling any missing day (e.g. sparse
    # Feb 29) by interpolation, then smooth circularly.
    full = pd.RangeIndex(1, 367)
    clim_mean = clim_mean.reindex(full).interpolate().bfill().ffill()
    clim_std = clim_std.reindex(full).interpolate().bfill().ffill()

    clim_mean = _circular_smooth(clim_mean, smooth_days)
    clim_std = _circular_smooth(clim_std, smooth_days)
    return clim_mean, clim_std


def standardized_anomaly(
    daily: pd.Series,
    clim_mean: pd.Series,
    clim_std: pd.Series,
) -> pd.Series:
    """Standardized anomaly in sigma units: (value - clim_mean) / clim_std.

    Each date is matched to its day-of-year climatology. A zero or missing std
    is treated as NaN to avoid divide-by-zero (it would only happen for a
    degenerate constant day, which we do not want to trust anyway).
    """
    doy = daily.index.dayofyear
    mean = clim_mean.reindex(doy).to_numpy()
    std = clim_std.reindex(doy).to_numpy()
    std = np.where(std > 0, std, np.nan)
    return pd.Series((daily.to_numpy() - mean) / std, index=daily.index)


# -------------------------------------------------
# CONCEPT NOTE: Innovation (the "forecast surprise")
# -------------------------------------------------
# What it is: The day-over-day CHANGE in an index, x(t) - x(t-1). We call it an
#   "innovation" because it is the new information that arrived since yesterday.
# Why we use it here: The market already prices in the weather everyone expected.
#   Prices move on the SURPRISE - when this morning's forecast is colder/calmer
#   than yesterday's. So the change in an index often carries more directional
#   signal than its level. A constant model/source bias also cancels in a
#   difference, which helps when training on ERA5 but serving on IFS.
# Common pitfall: Forgetting that the first row has no previous day, so its
#   innovation is undefined (NaN) and must be dropped by the caller.
# -------------------------------------------------
def innovation(daily: pd.Series, periods: int = 1) -> pd.Series:
    """Day-over-day change of a daily index (default 1-day difference)."""
    return daily.diff(periods=periods)
