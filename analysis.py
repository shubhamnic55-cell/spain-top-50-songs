"""Core data preparation and lifecycle features for Spain Top 50 project."""
from pathlib import Path
import pandas as pd
import numpy as np

DATA_PATH = Path(__file__).resolve().parent / "Atlantic_Spain.csv"
REQUIRED = ["date", "position", "song", "artist", "popularity", "duration_ms", "album_type", "total_tracks", "is_explicit"]

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["position"] = pd.to_numeric(df["position"], errors="coerce")
    for c in ["popularity", "duration_ms", "total_tracks"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["song"] = df["song"].fillna("Unknown song").astype(str).str.strip()
    df["artist"] = df["artist"].fillna("Unknown artist").astype(str).str.strip()
    df["album_type"] = df["album_type"].fillna("unknown").astype(str).str.strip().str.lower()
    df["is_explicit"] = df["is_explicit"].map(lambda x: str(x).strip().lower() in {"true", "1", "yes", "y"} if pd.notna(x) else False)
    df = df.dropna(subset=["date", "position"])
    # Stable song key across snapshots; title + artist helps distinguish remakes/collaborations.
    df["song_key"] = (df["song"].str.lower().str.replace(r"[^\w\s]", "", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
                       + "||" + df["artist"].str.lower().str.replace(r"[^\w\s]", "", regex=True).str.replace(r"\s+", " ", regex=True).str.strip())
    return df.sort_values(["date", "position"]).reset_index(drop=True)

def daily_validation(df):
    counts = df.groupby("date").size().rename("row_count").reset_index()
    counts["is_exactly_50"] = counts["row_count"].eq(50)
    return counts

def build_lifecycle(df):
    dates = df["date"].dropna()
    if dates.empty:
        return pd.DataFrame()
    end_of_data = dates.max()
    rows = []
    for key, g in df.groupby("song_key", sort=False):
        g = g.sort_values("date")
        first = g.iloc[0]
        last = g.iloc[-1]
        # Consecutive calendar days observed in Top 50; gaps can mean temporary exits/re-entries.
        observed_days = g["date"].nunique()
        span_days = int((last["date"] - first["date"]).days + 1)
        best_idx = g["position"].idxmin()
        best = g.loc[best_idx]
        # Daily rank movement: lower position means improvement.
        rank_series = g.drop_duplicates("date").sort_values("date")["position"]
        improving = len(rank_series) > 1 and rank_series.iloc[-1] < rank_series.iloc[0]
        rows.append({
            "song_key": key, "song": first["song"], "artist": first["artist"],
            "entry_date": first["date"], "last_seen_date": last["date"],
            "days_on_playlist": observed_days, "calendar_span_days": span_days,
            "peak_position": float(g["position"].min()), "time_to_peak_days": int((best["date"] - first["date"]).days),
            "entry_position": float(first["position"]), "last_position": float(last["position"]),
            "popularity": float(g["popularity"].median()) if g["popularity"].notna().any() else np.nan,
            "duration_ms": float(g["duration_ms"].median()) if g["duration_ms"].notna().any() else np.nan,
            "album_type": first["album_type"], "total_tracks": float(g["total_tracks"].median()) if g["total_tracks"].notna().any() else np.nan,
            "is_explicit": bool(g["is_explicit"].mode().iloc[0]) if not g["is_explicit"].mode().empty else False,
            "still_on_playlist_at_data_end": bool(last["date"] == end_of_data),
            "rank_improved_over_observation": bool(improving),
        })
    life = pd.DataFrame(rows)
    # A track last seen on the final date may have an unobserved future exit (right-censored).
    life["lifecycle_stage"] = np.select(
        [life["days_on_playlist"] <= 7, life["peak_position"] <= 10,
         life["rank_improved_over_observation"], life["last_position"] >= 35],
        ["New Entry", "Peak Phase", "Growth Phase", "Decline Phase"], default="Mature Phase")
    return life

def daily_churn(df):
    dates = sorted(df["date"].dropna().unique())
    daily_sets = {d: set(df.loc[df.date.eq(d), "song_key"]) for d in dates}
    rows = []
    for i, d in enumerate(dates):
        current = daily_sets[d]
        previous = daily_sets[dates[i-1]] if i else set()
        entries = len(current - previous) if i else np.nan
        exits = len(previous - current) if i else np.nan
        rows.append({"date": pd.Timestamp(d), "entries": entries, "exits": exits,
                     "churn_rate_pct": ((entries + exits) / 2 / max(len(current), 1) * 100) if i else np.nan,
                     "unique_songs": len(current)})
    return pd.DataFrame(rows)
