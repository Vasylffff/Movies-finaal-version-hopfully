"""
Actor -> Movies (Wikidata) | Fully workable script (FILMS ONLY)
- No API keys needed
- Input: a CSV/Excel file with actor names (or a manual list)
- Output:
  1) long table: Actor | Film | Year | FilmQID
  2) wide table: Actor | Movie1 | Movie2 | ...

NEW: filters out TV series / miniseries / TV episodes / web series / etc.
(You can also choose to exclude TV FILMS.)

Requires: pip install pandas openpyxl requests
"""

import time
import re
import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
# --- Revenue lookup ---
import os
import sys
sys.path.append(os.path.dirname(__file__))
try:
    from revenue_serch import tmdb_get_revenue_budget
except ImportError:
    tmdb_get_revenue_budget = None

session = requests.Session()
retry = Retry(
    total=6,
    backoff_factor=2,                 # 2s, 4s, 8s, ...
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
    raise_on_status=False
)
adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
session.mount("https://", adapter)
session.mount("http://", adapter)

# ---------- CONFIG ----------
INPUT_EXCEL = r"C:\Users\Basyl\OneDrive - Queen Mary, University of London\project\merged_movies4.csv"  # <-- change
ACTOR_COLUMN = "cast"                                      # <-- change (or set None to use MANUAL_ACTORS)
OUTPUT_LONG = r"C:\Users\Basyl\OneDrive - Queen Mary, University of London\project\actor_filmography2_long.xlsx"
OUTPUT_WIDE = r"C:\Users\Basyl\OneDrive - Queen Mary, University of London\project\actor_filmography2_wide.xlsx"

# Control size + politeness
MAX_FILMS_PER_ACTOR = 100      # max films to pull from Wikidata per actor
MAX_MOVIES_WIDE = 50           # max Movie columns in wide table
SLEEP_BETWEEN_ACTORS = 0.5     # seconds (be polite)

# NEW: If False, television films (TV movies) will be excluded too
KEEP_TELEVISION_FILMS = True
# ----------------------------

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
SPARQL_URL = "https://query.wikidata.org/sparql"

HEADERS = {
    "Accept": "application/sparql-results+json",
    "User-Agent": "qmul-filmography-project/1.1 (contact: student)"
}

def split_actor_names(cell):
    """
    Takes a cell like:
      "Tom Hanks, Meg Ryan; Bill Paxton"
    and returns a cleaned list of names.
    Adjust if your dataset uses a different separator.
    """
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    s = str(cell).strip()
    if not s:
        return []

    # split on commas/semicolons/pipes/slashes
    parts = re.split(r"[;,|/]+", s)
    names = []
    for p in parts:
        name = p.strip()
        if name:
            names.append(name)
    return names

def wikidata_search_person(name, limit=10):
    params = {
        "action": "wbsearchentities",
        "search": name,
        "language": "en",
        "format": "json",
        "type": "item",
        "limit": limit
    }
    r = session.get(WIKIDATA_API, params=params, headers=HEADERS, timeout=(10,40))
    r.raise_for_status()
    return r.json().get("search", [])

def get_actor_qid(actor_name):
    """
    Returns best-match QID for an actor.
    Tries to pick a result whose description suggests acting.
    """
    hits = wikidata_search_person(actor_name, limit=10)
    if not hits:
        return None

    for hit in hits:
        desc = (hit.get("description") or "").lower()
        if any(k in desc for k in ["actor", "actress", "film actor", "television actor"]):
            return hit["id"]

    # fallback: first result
    return hits[0]["id"]

def sparql(query: str):
    r = session.get(SPARQL_URL, params={"query": query, "format": "json"}, headers=HEADERS, timeout=(10,90))
    r.raise_for_status()
    return r.json()

def get_filmography(actor_qid: str, limit=MAX_FILMS_PER_ACTOR):
    """
    Pulls FILMS (not TV series) where the actor is listed as cast member (P161).
    Returns list of dicts: Film, Year, FilmQID.
    """

    # ---- Film inclusion ----
    # film: Q11424
    # We include anything that is instance/subclass of film.
    # ---- TV exclusions ----
    # television series: Q5398426
    # miniseries: Q1253544
    # television episode: Q21191270
    # web series: Q117467246  (not always present, but harmless)
    # TV program: Q15416
    # (optional) television film: Q506240
    #
    # Note: excluding TV program/series/episode is the key part.
    # Keeping TV films is controlled by KEEP_TELEVISION_FILMS.

    tv_film_minus = ""
    if not KEEP_TELEVISION_FILMS:
        tv_film_minus = """
          MINUS { ?film wdt:P31/wdt:P279* wd:Q506240. }  # television film
        """

    query = f"""
    SELECT ?film ?filmLabel ?year WHERE {{
      # must be a film (instance/subclass of film)
      ?film wdt:P31/wdt:P279* wd:Q11424.
      # actor is cast member
      ?film wdt:P161 wd:{actor_qid}.

      # EXCLUDE TV series / episodes / programs / miniseries / web series
      MINUS {{ ?film wdt:P31/wdt:P279* wd:Q5398426. }}   # television series
      MINUS {{ ?film wdt:P31/wdt:P279* wd:Q1253544. }}   # miniseries
      MINUS {{ ?film wdt:P31/wdt:P279* wd:Q21191270. }}  # television episode
      MINUS {{ ?film wdt:P31/wdt:P279* wd:Q15416. }}     # television program
      MINUS {{ ?film wdt:P31/wdt:P279* wd:Q117467246. }} # web series (if present)
      {tv_film_minus}

      OPTIONAL {{
        ?film wdt:P577 ?date.
        BIND(YEAR(?date) AS ?year)
      }}

      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    ORDER BY ?year ?filmLabel
    LIMIT {int(limit)}
    """

    data = sparql(query)
    rows = []
    for b in data.get("results", {}).get("bindings", []):
        film_label = b["filmLabel"]["value"]
        year = b.get("year", {}).get("value")
        film_uri = b["film"]["value"]
        film_qid = film_uri.rsplit("/", 1)[-1]
        rows.append({"Film": film_label, "Year": int(year) if year else None, "FilmQID": film_qid})
    return rows

def actor_to_movies_table(actor_name: str):
    """
    Returns a DataFrame: Actor | ActorQID | Film | Year | FilmQID | Error
    """
    qid = get_actor_qid(actor_name)
    if not qid:
        return pd.DataFrame([{"Actor": actor_name, "ActorQID": None, "Film": None, "Year": None, "FilmQID": None, "Error": "Actor not found"}])

    try:
        films = get_filmography(qid)
        if not films:
            return pd.DataFrame([{"Actor": actor_name, "ActorQID": qid, "Film": None, "Year": None, "FilmQID": None, "Revenue": None, "Error": f"No FILMS found (TV excluded) for {qid}"}])

        df = pd.DataFrame(films)
        df.insert(0, "Actor", actor_name)
        df.insert(1, "ActorQID", qid)
        df["Error"] = None

        # Add revenue lookup if available
        if tmdb_get_revenue_budget:
            revenues = []
            for _, row in df.iterrows():
                rev = None
                try:
                    rev, _ = tmdb_get_revenue_budget(row["Film"], row["Year"])
                except Exception:
                    rev = None
                revenues.append(rev)
                time.sleep(0.2)
            df["Revenue"] = revenues
        else:
            df["Revenue"] = None
        return df

    except Exception as e:
        return pd.DataFrame([{"Actor": actor_name, "ActorQID": qid, "Film": None, "Year": None, "FilmQID": None, "Revenue": None, "Error": str(e)}])

def make_wide_actor_table(df_long: pd.DataFrame, max_movies=50):
    """
    df_long columns: Actor, Film, Year (optional)
    Returns: Actor, Movie1..MovieN (up to max_movies)
    """
    # Sort so movie order is stable (by year then title)
    if "Year" in df_long.columns:
        df_long = df_long.sort_values(["Actor", "Year", "Film"], na_position="last")
    else:
        df_long = df_long.sort_values(["Actor", "Film"])

    grouped = df_long.groupby("Actor")["Film"].apply(list).reset_index()
    grouped["Film"] = grouped["Film"].apply(lambda xs: xs[:max_movies])

    movies_cols = grouped["Film"].apply(pd.Series)
    movies_cols.columns = [f"Movie{i+1}" for i in range(movies_cols.shape[1])]

    wide = pd.concat([grouped[["Actor"]], movies_cols], axis=1)
    return wide

def get_actor_list_from_excel(path, actor_col):
    # Your file is .csv currently, so keep read_csv
    df = pd.read_csv(path)
    if actor_col not in df.columns:
        raise ValueError(f"Column '{actor_col}' not found. Columns are: {list(df.columns)}")

    names = []
    for cell in df[actor_col]:
        names.extend(split_actor_names(cell))

    # dedupe preserving order
    seen = set()
    uniq = []
    for n in names:
        key = n.lower()
        if key not in seen:
            seen.add(key)
            uniq.append(n)
    return uniq

def main():
    MANUAL_ACTORS = []

    if ACTOR_COLUMN:
        actors = get_actor_list_from_excel(INPUT_EXCEL, ACTOR_COLUMN)
    else:
        actors = MANUAL_ACTORS[:]

    if not actors:
        raise ValueError("No actors found. Check ACTOR_COLUMN / MANUAL_ACTORS.")

    print(f"Found {len(actors)} unique actors.")
    print("Starting data fetch from Wikidata (FILMS ONLY; TV series excluded)...")
    print("KEEP_TELEVISION_FILMS =", KEEP_TELEVISION_FILMS)

    all_dfs = []
    for i, actor in enumerate(actors, start=1):
        print(f"[{i}/{len(actors)}] Fetching: {actor}")
        df_actor = actor_to_movies_table(actor)
        all_dfs.append(df_actor)
        time.sleep(SLEEP_BETWEEN_ACTORS)

        if i % 50 == 0:
            partial_path = OUTPUT_LONG.replace(".xlsx", f"_partial{i}.xlsx")
            pd.concat(all_dfs, ignore_index=True).to_excel(partial_path, index=False)
            print(f"Saved partial progress: {partial_path}")

    df_long = pd.concat(all_dfs, ignore_index=True)

    # save long table
    df_long.to_excel(OUTPUT_LONG, index=False)
    print(f"Saved long table to: {OUTPUT_LONG}")

    # build + save wide table (ignore rows with no Film)
    df_long_ok = df_long[df_long["Film"].notna()].copy()
    df_wide = make_wide_actor_table(df_long_ok[["Actor", "Film", "Year"]], max_movies=MAX_MOVIES_WIDE)
    df_wide.to_excel(OUTPUT_WIDE, index=False)
    print(f"Saved wide table to: {OUTPUT_WIDE}")

    print("Done.")

if __name__ == "__main__":
    main()