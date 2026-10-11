from pathlib import Path

SEED = 20260601
ALPHA = 0.05
N_BOOT = 10_000
N_PERM = 10_000
SEASON = 7
HORIZON = 30
TEST_DAYS = 7

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "daily-demand-series.csv"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
FIGS = ROOT / "reports" / "figures"
TABLES = ROOT / "reports" / "tables"
