"""Download ERA5 single-level fields from MARS, one GRIB file per year.

Why ERA5 for training (and IFS only for live serving)?
    ERA5 is a single, homogeneous reanalysis covering every year we train on,
    so there is no discontinuity in the training data. The live IFS analysis is
    only used at serve time (ERA5 has a multi-day latency). Because our features
    are anomalies and day-over-day innovations, the small ERA5-vs-IFS offset
    largely cancels out.

Data model
----------
We retrieve INSTANTANEOUS ANALYSIS fields (type=an) at 00/06/12/18 UTC:
    2t    2 metre temperature      -> HDD / temperature anomaly
    100u  100 m U wind component   -> wind speed / generation index
    100v  100 m V wind component   -> wind speed / generation index
    tcc   total cloud cover        -> cloud index
Four times a day is a deliberate, RAM-friendly compromise: enough to build a
sensible daily index without pulling hourly ERA5 (~4x the data) on an 8 GB Mac.

SOLAR (ssrd) IS DEFERRED. Surface solar radiation is an ACCUMULATED FORECAST
field in ERA5 (type=fc, not type=an), so it cannot ride along in this clean
analysis request - it needs a separate forecast stream and de-accumulation.
Given solar is a secondary driver for gas (cloud cover already captures the
blocking-high signal), v1 uses the four analysis fields and leaves ssrd for v2.
If config.ERA5_VARIABLES still lists "ssrd", we simply warn and skip it.

Usage (run on the Mac, inside the `nbp` conda env, from the repo root):
    # 1) validate credentials + request with a tiny one-month sample first:
    python -m src.data_ingestion.fetch_ecmwf --start-year 2020 --end-year 2020 --sample
    # 2) then the full historical pull (large; resumes, skips existing years):
    python -m src.data_ingestion.fetch_ecmwf --start-year 1997 --end-year 2025
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime

import config

# ---------------------------------------------------------------------------
# MARS parameter IDs (GRIB paramId). These are the ECMWF numeric codes.
#
# !! PLEASE VERIFY against your MARS catalogue before the full download. The
# 100 m wind IDs in particular are worth a double-check on your system:
#     2t   = 167          (2 metre temperature)
#     100u = 228246       (100 m U wind component)
#     100v = 228247       (100 m V wind component)
#     tcc  = 164          (total cloud cover)
#     ssrd = 169          (surface solar radiation downwards - deferred, fc field)
# ---------------------------------------------------------------------------
MARS_PARAM_IDS = {
    "2t": "167",
    "100u": "228246",
    "100v": "228247",
    "tcc": "164",
    "ssrd": "169",
}

# Instantaneous analysis fields we retrieve in the type=an request.
ANALYSIS_VARS = ["2t", "100u", "100v", "tcc"]

# Accumulated forecast fields we deliberately skip in v1 (see module docstring).
DEFERRED_VARS = ["ssrd"]

# Analysis times (UTC) used to build the daily index.
ANALYSIS_TIMES = "00:00:00/06:00:00/12:00:00/18:00:00"


def _year_target_path(year: int, sample: bool):
    """File path for a year's GRIB download."""
    suffix = "_sample" if sample else ""
    return config.RAW_DIR / f"era5_{year}{suffix}.grib"


def build_era5_request(year: int, sample: bool) -> dict:
    """Build the MARS request dict for one year of ERA5 analysis fields.

    When `sample` is True we fetch only January of that year - a quick, cheap
    way to prove the request and credentials work before the full pull.
    """
    params = "/".join(MARS_PARAM_IDS[v] for v in ANALYSIS_VARS)

    if sample:
        date = f"{year}-01-01/to/{year}-01-31"
    else:
        date = f"{year}-01-01/to/{year}-12-31"

    area = "/".join(str(x) for x in config.MARS_AREA)      # N/W/S/E
    grid = "/".join(str(x) for x in config.MARS_GRID)      # 0.25/0.25

    return {
        "class": "ea",          # ERA5
        "expver": "1",
        "stream": "oper",       # atmospheric model, hourly analysis
        "type": "an",           # analysis (instantaneous) fields only
        "levtype": "sfc",       # single (surface) levels
        "param": params,
        "date": date,
        "time": ANALYSIS_TIMES,
        "grid": grid,
        "area": area,
        "format": "grib",
    }


def fetch_era5_year(server, year: int, sample: bool, overwrite: bool) -> bool:
    """Download one year (or one sample month) of ERA5. Returns True if fetched.

    Skips the download if the target file already exists (unless `overwrite`),
    so the full multi-year pull can be safely resumed after an interruption.
    """
    target = _year_target_path(year, sample)
    if target.exists() and not overwrite:
        print(f"  [skip] {target.name} already exists ({target.stat().st_size / 1e6:.0f} MB)")
        return False

    request = build_era5_request(year, sample)
    print(f"  [get ] {target.name}  ({request['date']})")
    server.execute(request, str(target))
    print(f"  [done] {target.name}  ({target.stat().st_size / 1e6:.0f} MB)")
    return True


def fetch_era5_range(start_year: int, end_year: int, sample: bool, overwrite: bool) -> None:
    """Download ERA5 for an inclusive range of years, one file per year."""
    # Import here so the module can be imported (e.g. by the smoke test or unit
    # tests of build_era5_request) without the ecmwf-api-client being present.
    try:
        from ecmwfapi import ECMWFService
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "ecmwf-api-client is not installed in this environment. Activate the "
            "`nbp` conda env (see environment.yml) before running the download."
        ) from exc

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Warn once if the config still lists deferred (accumulated) variables.
    deferred_in_config = [v for v in config.ERA5_VARIABLES if v in DEFERRED_VARS]
    if deferred_in_config:
        print(
            f"NOTE: {deferred_in_config} is an accumulated forecast field and is "
            "deferred to v2; it will NOT be downloaded. See module docstring.\n"
        )

    server = ECMWFService("mars")

    print(f"Fetching ERA5 {start_year}-{end_year} "
          f"({'SAMPLE (Jan only)' if sample else 'full years'}) "
          f"for {ANALYSIS_VARS} over area {config.MARS_AREA}:")
    fetched = 0
    for year in range(start_year, end_year + 1):
        if fetch_era5_year(server, year, sample, overwrite):
            fetched += 1
    print(f"\nDone. {fetched} file(s) downloaded to {config.RAW_DIR}.")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download ERA5 single-level fields from MARS.")
    current_year = datetime.utcnow().year
    parser.add_argument("--start-year", type=int, default=1997,
                        help="First year to download (inclusive). Default 1997.")
    parser.add_argument("--end-year", type=int, default=current_year,
                        help="Last year to download (inclusive). Default current year.")
    parser.add_argument("--sample", action="store_true",
                        help="Fetch only January of each year - a quick validation pull.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-download even if the target GRIB already exists.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.end_year < args.start_year:
        print("end-year must be >= start-year", file=sys.stderr)
        return 2
    fetch_era5_range(args.start_year, args.end_year, args.sample, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
