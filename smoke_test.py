"""Phase 1 smoke test.

Purpose: confirm the conda environment is healthy BEFORE we write any
pipeline code. It checks three things:

  1. Every third-party dependency imports and reports its version.
  2. The GRIB stack (eccodes / cfgrib) is actually functional, not just
     importable - this is the component most likely to break on Intel macOS.
  3. The project's own `config.py` loads and its paths are sane.

It also *warns* (does not fail) if the ECMWF/CDS credential files are absent,
since those are set up separately and are not needed just to import things.

Run from the repo root:  python smoke_test.py
Exit code 0 = all good; 1 = something is wrong.
"""

import importlib
import sys
from pathlib import Path

# Packages we depend on, in (import_name, human_label) form. Some packages
# import under a different name than their pip/conda name, hence the mapping.
REQUIRED = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("pyarrow", "pyarrow"),
    ("scipy", "scipy"),
    ("cfgrib", "cfgrib"),
    ("xarray", "xarray"),
    ("netCDF4", "netcdf4"),
    ("ecmwfapi", "ecmwf-api-client"),
    ("cdsapi", "cdsapi"),
    ("sklearn", "scikit-learn"),
    ("xgboost", "xgboost"),
    ("lightgbm", "lightgbm"),
    ("matplotlib", "matplotlib"),
    ("yaml", "pyyaml"),
    ("joblib", "joblib"),
]


def check_imports() -> bool:
    """Import each required package and print its version.

    Returns True only if every package imported successfully. We collect all
    failures rather than stopping at the first, so the user sees the complete
    picture in one run.
    """
    all_ok = True
    print("-- Dependency imports " + "-" * 40)
    for module_name, label in REQUIRED:
        try:
            mod = importlib.import_module(module_name)
            version = getattr(mod, "__version__", "unknown")
            print(f"  [ ok ] {label:<20} {version}")
        except Exception as exc:  # noqa: BLE001 - we want to report any failure
            all_ok = False
            print(f"  [FAIL] {label:<20} {exc}")
    return all_ok


def check_grib_engine() -> bool:
    """Confirm the eccodes C library underneath cfgrib is operational.

    cfgrib can import even when the eccodes binary is misconfigured, so we
    probe eccodes directly. A working version string means GRIB decoding
    will work when we start pulling ERA5 data in Phase 3.
    """
    print("-- GRIB engine (eccodes) " + "-" * 37)
    try:
        import cfgrib  # noqa: F401  (import validates the backend is present)
        from cfgrib import messages  # noqa: F401
        import eccodes

        version = eccodes.codes_get_api_version()
        print(f"  [ ok ] eccodes API version {version}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] eccodes not functional: {exc}")
        return False


def check_config() -> bool:
    """Load config.py and sanity-check the domain and paths."""
    print("-- Project config " + "-" * 44)
    try:
        import config

        # A minimal sanity check: the bounding box must be well-formed.
        assert config.DOMAIN_NORTH > config.DOMAIN_SOUTH, "domain N/S inverted"
        assert config.DOMAIN_EAST > config.DOMAIN_WEST, "domain E/W inverted"
        print(f"  [ ok ] domain N{config.DOMAIN_NORTH}/S{config.DOMAIN_SOUTH} "
              f"W{config.DOMAIN_WEST}/E{config.DOMAIN_EAST} @ {config.GRID_RESOLUTION} deg")
        print(f"  [ ok ] project root: {config.PROJECT_ROOT}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] config.py problem: {exc}")
        return False


def check_credentials() -> None:
    """Warn (do not fail) if MARS/CDS credential files are missing.

    These are set up separately (see README). We surface their status now so
    the user knows what remains before Phase 3 data downloads.
    """
    print("-- Credentials (informational) " + "-" * 31)
    for path, label in [
        (Path.home() / ".ecmwfapirc", "MARS (~/.ecmwfapirc)"),
        (Path.home() / ".cdsapirc", "CDS  (~/.cdsapirc)"),
    ]:
        status = "found" if path.exists() else "MISSING (set up before Phase 3)"
        print(f"  [info] {label:<22} {status}")


def main() -> int:
    print("=" * 62)
    print("NBP signal - environment smoke test")
    print("=" * 62)

    imports_ok = check_imports()
    grib_ok = check_grib_engine()
    config_ok = check_config()
    check_credentials()

    print("=" * 62)
    if imports_ok and grib_ok and config_ok:
        print("RESULT: PASS - environment is ready.")
        return 0
    print("RESULT: FAIL - fix the [FAIL] items above before continuing.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
