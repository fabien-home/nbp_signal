# NBP Weather Signal

A short-term **directional trading signal** for ICE **NBP front-month** natural
gas futures, driven by **ECMWF weather data** over NW Europe. Each morning a
manually-run script ingests the latest **00z run**, computes a small set of
daily weather indices, and prints an actionable UP/DOWN signal with a
calibrated confidence for the **same trading day**.

> **Status:** v1 (weather-only, same-day, NW-Europe domain). Fundamentals
> (gas storage, LNG/pipeline flows), the 2-4 day horizon, and ENS-spread as a
> trained feature are *planned in the pipeline* but activated in v2.

---

## What it predicts

- **Target:** will today's NBP front-month settlement be **higher or lower**
  than the previous settlement, predicted from **this morning's 00z run**
  (available ~06:30, before the 07:00 open).
- **Active season:** September-April (heating season) only. Outside this
  window the script reports "no signal - outside heating season".
- **Output:** direction + calibrated probability + key weather drivers +
  a forecast-reliability (ENS spread) note.

---

## Repository layout

```
nbp_signal/
├── environment.yml            # conda environment (GRIB stack lives here)
├── config.py                  # domain, paths, thresholds, feature flags
├── smoke_test.py              # Phase 1: verify the environment works
├── data/{raw,processed}/      # gitignored
├── src/
│   ├── data_ingestion/        # fetch_ecmwf.py, fetch_nbp.py
│   ├── features/              # build_features.py, transforms.py (HDD, power curve)
│   ├── regime/                # ens_spread.py, price_vol.py, storage.py (v2)
│   ├── models/                # train.py, evaluate.py, predict.py
│   └── utils/                 # helpers.py
├── scripts/                   # run_daily_signal.py, retrain_model.py
├── models_saved/              # gitignored
└── output/                    # signal_log.csv (gitignored)
```

---

## Setup (macOS Monterey, Intel)

These steps assume a clean machine. Run them once.

### 1. Install Homebrew (if not present)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. Install Miniconda (Intel build)

```bash
brew install --cask miniconda
# Initialise conda for zsh, then restart the terminal:
source /usr/local/Caskroom/miniconda/base/etc/profile.d/conda.sh
conda init zsh
```

We use **Miniconda**, not pip/pyenv, because the ECMWF GRIB stack
(`eccodes` / `cfgrib`) installs cleanly from conda-forge but is painful to
build under pip on Intel macOS.

### 3. Create the project environment

```bash
cd nbp_signal
conda env create -f environment.yml
conda activate nbp
```

### 4. Verify the environment

```bash
python smoke_test.py
```

You want `RESULT: PASS`. The credentials check will report MARS/CDS as
`MISSING` for now - that is expected and set up next.

### 5. ECMWF credentials (needed from Phase 3 onward)

Create `~/.ecmwfapirc` for MARS access (institutional login). Example shape
(fill in your own values):

```
{
    "url"   : "https://api.ecmwf.int/v1",
    "key"   : "YOUR-KEY",
    "email" : "you@metservice.example"
}
```

(If we also use the CDS fallback for ERA5, a `~/.cdsapirc` is created similarly.)

---

## Running (later phases)

```bash
# Daily operational signal (Phase 5)
python scripts/run_daily_signal.py

# Periodic retraining (Phase 6)
python scripts/retrain_model.py
```

---

## Roadmap (v2 and beyond)

- Fundamentals features: gas storage (GIE AGSI+, free), ENTSOG flows,
  Norwegian pipeline outages.
- 2-4 day horizon + ENS spread as a *trained* feature (via ENS reforecasts).
- Probability calibration (Platt / isotonic).
- TTF-NBP spread and price-momentum features.
- Overnight automation (launchd/cron) once manual running is trusted.

See the project plan for full detail.
