# add_trends.py
from pytrends.request import TrendReq
import pandas as pd
import time
import re
import unicodedata

import os
import argparse

# ---------------- CONFIG ----------------
# Auto-detect latest merged file (prefer merged_movies5.csv)
DEFAULT_CANDIDATES = [
    r"C:\Users\Basyl\OneDrive - Queen Mary, University of London\project\merged_movies5.csv",
    r"C:\Users\Basyl\OneDrive - Queen Mary, University of London\project\merged_movies4.csv",
]

parser = argparse.ArgumentParser(description="Add Google Trends pre-release stats to merged movies")
parser.add_argument("--input", help="Input merged CSV file (overrides auto-detect)")
parser.add_argument("--output", help="Output CSV path")
parser.add_argument("--geo", default="US", help="Default geo to query (e.g., US)")
parser.add_argument("--verbose", action="store_true", help="Verbose logging")
args = parser.parse_args()

INPUT_PATH = args.input or next((p for p in DEFAULT_CANDIDATES if os.path.exists(p)), DEFAULT_CANDIDATES[-1])
OUTPUT_PATH = args.output or r"C:\Users\Basyl\OneDrive - Queen Mary, University of London\project\merged_movies_with_trends.csv"

ANNOUNCEMENT_DAYS = 871
GEO_DEFAULT = args.geo        # try US first
SLEEP_SEC = 1.2           # pause between movies (avoid blocking)
TRY_WORLD_FALLBACK = True # if GB fails, try worldwide

TRENDS_MIN_DATE = pd.Timestamp("2004-01-01")  # Google Trends earliest reliable date

# ---------------- CLEANING ----------------
def clean_for_trends(s: str) -> str:
    """Make a search term more likely to match Google Trends."""
    if s is None or pd.isna(s):
        return ""
    s = str(s).strip()

    # remove trailing (...) or [...]
    s = re.sub(r"\s*\(.*?\)\s*$", "", s).strip()
    s = re.sub(r"\s*\[.*?\]\s*$", "", s).strip()

    # remove accents (léon -> leon)
    s = "".join(
        ch for ch in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(ch)
    )

    # collapse spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s


def trend_term_candidates(title_raw, title_clean):
    """Generate ordered candidates to try on Google Trends."""
    candidates = []

    t1 = clean_for_trends(title_raw)
    if t1:
        candidates.append(t1)

    t2 = clean_for_trends(title_clean)
    if t2 and t2 not in candidates:
        candidates.append(t2)

    # If title is very short/ambiguous, add disambiguation
    if t1 and len(t1) <= 4:
        candidates.append(f"{t1} movie")
        candidates.append(f"{t1} film")

    return candidates


# ---------------- TRENDS CLIENT ----------------
pytrends = TrendReq(hl="en-US", tz=0)


def get_trends_series(term: str, timeframe: str, geo: str):
    """Fetch trends time series for a term. Returns df or None."""
    try:
        pytrends.build_payload([term], timeframe=timeframe, geo=geo)
        df_t = pytrends.interest_over_time()
        if df_t is None or df_t.empty or term not in df_t.columns:
            return None
        return df_t
    except Exception as e:
        if args.verbose:
            print(f"[pytrends ERROR] term={term!r} timeframe={timeframe} geo={geo} -> {e}")
        return None


def trends_announce_to_release_multi(title_raw, title_clean, release_date):
    """
    Return (sum, mean, peak, term_used, geo_used)
    Using first term+geo that returns data.
    For movies before 2004, clamp to 2004 but mark as pre-2004.
    """
    if pd.isna(release_date):
        return None, None, None, None, None

    # If release date is before Google Trends reliable start, clamp to 2004
    original_release_date = release_date
    if release_date < TRENDS_MIN_DATE:
        release_date = TRENDS_MIN_DATE
        if args.verbose:
            print(f"[CLAMP] original_release_date {original_release_date.date()} < TRENDS_MIN_DATE; clamping to {TRENDS_MIN_DATE.date()}")

    # compute announcement proxy
    ann_date = release_date - pd.Timedelta(days=ANNOUNCEMENT_DAYS)

    # clamp to Google Trends earliest
    if ann_date < TRENDS_MIN_DATE:
        ann_date = TRENDS_MIN_DATE

    # timeframe string
    timeframe = f"{ann_date.date()} {release_date.date()}"
    if args.verbose:
        print(f"Trying timeframe: {timeframe}")
    # geos to try
    geos_to_try = [GEO_DEFAULT]
    if TRY_WORLD_FALLBACK:
        geos_to_try.append("")  # "" means worldwide

    for geo_try in geos_to_try:
        geo_label = geo_try if geo_try else "WORLD"

        for term in trend_term_candidates(title_raw, title_clean):
            # try term + disambiguation variants
            variants = [term, f"{term} movie", f"{term} film"]

            for v in variants:
                df_t = get_trends_series(v, timeframe, geo_try)
                if df_t is not None:
                    if args.verbose:
                        print(f"[FOUND] term={v!r} geo={geo_label} sum={float(df_t[v].sum())} peak={float(df_t[v].max())}")
                    return (
                        float(df_t[v].sum()),
                        float(df_t[v].mean()),
                        float(df_t[v].max()),
                        v,
                        geo_label
                    )

                time.sleep(0.2)  # tiny pause between attempts
            # Try next term candidate
            if args.verbose:
                print(f"No data for candidates based on {term!r} (geo={geo_label})")

    return None, None, None, None, None


# ---------------- LOAD DATA ----------------
df = pd.read_csv(INPUT_PATH)
print("Loaded rows:", len(df))

# parse release_date
df["release_date"] = pd.to_datetime(df.get("release_date"), errors="coerce")

# ensure title_clean exists
if "title_clean" not in df.columns:
    # create a simple one if missing
    df["title_clean"] = df["Title"].astype(str).str.lower()

# ---------------- RUN TRENDS ----------------
trend_sum = []
trend_mean = []
trend_peak = []
term_used = []
geo_used = []

success_count = 0
fail_count = 0
found_samples = []

for i, row in df.iterrows():
    title_raw = row.get("Title")
    title_clean = row.get("title_clean")
    release_date = row.get("release_date")

    s, m, p, term, geo = trends_announce_to_release_multi(title_raw, title_clean, release_date)

    trend_sum.append(s)
    trend_mean.append(m)
    trend_peak.append(p)
    term_used.append(term)
    geo_used.append(geo)

    if term is not None:
        success_count += 1
        if len(found_samples) < 10:
            found_samples.append((title_raw, term, geo, s, p))
    else:
        fail_count += 1

    time.sleep(SLEEP_SEC)

    if (i + 1) % 25 == 0:
        print(f"Processed {i+1}/{len(df)} (success {success_count}, fail {fail_count})")

df["trend_pre_release_sum"] = trend_sum
df["trend_pre_release_mean"] = trend_mean
df["trend_pre_release_peak"] = trend_peak
df["trend_term_used"] = term_used
df["trend_geo_used"] = geo_used

# Apply adjustment: for movies before 2004, use average of 2004 trends
df["original_release_year"] = pd.to_datetime(df.get("release_date"), errors="coerce").dt.year
year_2004_mask = df["original_release_year"] == 2004

if year_2004_mask.sum() > 0:
    # Calculate average trend values for 2004 movies
    avg_2004_sum = df[year_2004_mask]["trend_pre_release_sum"].mean()
    avg_2004_mean = df[year_2004_mask]["trend_pre_release_mean"].mean()
    avg_2004_peak = df[year_2004_mask]["trend_pre_release_peak"].mean()
    
    # Find movies before 2004 that got trends data
    pre_2004_with_data = (df["original_release_year"] < 2004) & (df["trend_pre_release_sum"].notna())
    
    # Replace with 2004 average for pre-2004 movies
    if pre_2004_with_data.sum() > 0:
        df.loc[pre_2004_with_data, "trend_pre_release_sum"] = avg_2004_sum
        df.loc[pre_2004_with_data, "trend_pre_release_mean"] = avg_2004_mean
        df.loc[pre_2004_with_data, "trend_pre_release_peak"] = avg_2004_peak
        print(f"Applied 2004 average adjustment to {pre_2004_with_data.sum()} pre-2004 movies")
        print(f"  2004 avg sum: {avg_2004_sum:.2f}, avg mean: {avg_2004_mean:.2f}, avg peak: {avg_2004_peak:.2f}")

# Apply adjustment: for movies before 2004, use average of 2004 trends
df["original_release_year"] = pd.to_datetime(df.get("release_date"), errors="coerce").dt.year
year_2004_mask = df["original_release_year"] == 2004

if year_2004_mask.sum() > 0:
    # Calculate average trend values for 2004 movies
    avg_2004_sum = df[year_2004_mask]["trend_pre_release_sum"].mean()
    avg_2004_mean = df[year_2004_mask]["trend_pre_release_mean"].mean()
    avg_2004_peak = df[year_2004_mask]["trend_pre_release_peak"].mean()
    
    # Find movies before 2004 that got trends data
    pre_2004_with_data = (df["original_release_year"] < 2004) & (df["trend_pre_release_sum"].notna())
    
    # Replace with 2004 average for pre-2004 movies
    if pre_2004_with_data.sum() > 0:
        df.loc[pre_2004_with_data, "trend_pre_release_sum"] = avg_2004_sum
        df.loc[pre_2004_with_data, "trend_pre_release_mean"] = avg_2004_mean
        df.loc[pre_2004_with_data, "trend_pre_release_peak"] = avg_2004_peak
        print(f"Applied 2004 average adjustment to {pre_2004_with_data.sum()} pre-2004 movies")
        print(f"  2004 avg sum: {avg_2004_sum:.2f}, avg mean: {avg_2004_mean:.2f}, avg peak: {avg_2004_peak:.2f}")

if args.verbose:
    print("Sample found items:")
    for sitem in found_samples:
        print(sitem)

# ---------------- SAVE ----------------
df.to_csv(OUTPUT_PATH, index=False)
print("Saved:", OUTPUT_PATH)

# ---------------- OPTIONAL: show failures ----------------
failed = df[df["trend_term_used"].isna()][["Title", "title_clean", "release_date"]].head(20)
print("\nFirst failed rows (if any):")
print(failed)