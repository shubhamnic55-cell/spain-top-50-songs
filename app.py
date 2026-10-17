from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from analysis import load_data, daily_validation, build_lifecycle, daily_churn

st.set_page_config(page_title="Spain Top 50 Lifecycle Intelligence", page_icon="🎵", layout="wide")
st.title("🎵 Spain Top 50 — Content Maturity & Playlist Rotation")
st.caption("Atlantic Recording Corporation | Data analytics and machine-learning prototype")

@st.cache_data
def get_data():
    df = load_data()
    return df, daily_validation(df), build_lifecycle(df), daily_churn(df)

try:
    df, validation, life, churn = get_data()
except Exception as e:
    st.error(f"Could not load dataset: {e}")
    st.stop()

with st.sidebar:
    st.header("Filters")
    min_date, max_date = df.date.min().date(), df.date.max().date()
    selected_dates = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
        start_date, end_date = pd.Timestamp(selected_dates[0]), pd.Timestamp(selected_dates[1])
    else:
        start_date, end_date = pd.Timestamp(min_date), pd.Timestamp(max_date)
    stage_options = ["New Entry", "Growth Phase", "Peak Phase", "Mature Phase", "Decline Phase"]
    stages = st.multiselect("Lifecycle stage", stage_options, default=stage_options)
    explicit_filter = st.selectbox("Explicit content", ["All", "Explicit only", "Clean only"])
    album_filter = st.multiselect("Album type", sorted(life.album_type.dropna().unique()), default=sorted(life.album_type.dropna().unique()))

filtered_df = df[df.date.between(start_date, end_date)]
filtered_life = life[(life.entry_date <= end_date) & (life.last_seen_date >= start_date)].copy()
if stages:
    filtered_life = filtered_life[filtered_life.lifecycle_stage.isin(stages)]
if explicit_filter == "Explicit only": filtered_life = filtered_life[filtered_life.is_explicit]
elif explicit_filter == "Clean only": filtered_life = filtered_life[~filtered_life.is_explicit]
if album_filter: filtered_life = filtered_life[filtered_life.album_type.isin(album_filter)]

# KPIs
c1,c2,c3,c4 = st.columns(4)
c1.metric("Daily snapshots", f"{filtered_df.date.nunique():,}")
c2.metric("Songs in lifecycle view", f"{len(filtered_life):,}")
c3.metric("Mean observed playlist days", f"{filtered_life.days_on_playlist.mean():.1f}" if len(filtered_life) else "—")
c4.metric("Mean daily churn", f"{churn[churn.date.between(start_date,end_date)].churn_rate_pct.mean():.1f}%" if churn.date.between(start_date,end_date).any() else "—")

st.warning("Interpretation note: tracks still present on the dataset's final date have right-censored lifetimes. 'Days on playlist' counts observed Top 50 appearances, while 'calendar span' includes gaps. The ML model estimates observed lifetime from available metadata; it is not a causal model.")

tab1,tab2,tab3,tab4,tab5 = st.tabs(["Overview", "Lifecycle", "Rotation", "Content comparison", "Machine learning"])
with tab1:
    st.subheader("Dataset quality")
    v1,v2,v3 = st.columns(3)
    v1.metric("Rows", f"{len(df):,}")
    v2.metric("Date range", f"{df.date.min():%d %b %Y} – {df.date.max():%d %b %Y}")
    v3.metric("Days with exactly 50 rows", f"{validation.is_exactly_50.sum()} / {len(validation)}")
    st.write("Daily snapshots that do not contain exactly 50 rows")
    st.dataframe(validation[~validation.is_exactly_50], use_container_width=True, hide_index=True)
    st.subheader("Top songs by observed playlist days")
    st.dataframe(filtered_life.sort_values("days_on_playlist", ascending=False)[["song","artist","days_on_playlist","calendar_span_days","peak_position","album_type","is_explicit"]].head(20), use_container_width=True, hide_index=True)
with tab2:
    st.subheader("Lifecycle stage distribution")
    if len(filtered_life):
        fig, ax = plt.subplots(); filtered_life.lifecycle_stage.value_counts().reindex(stage_options).dropna().plot(kind="bar", ax=ax); ax.set_ylabel("Songs"); ax.set_xlabel(""); plt.xticks(rotation=25, ha="right"); st.pyplot(fig); plt.close(fig)
        st.subheader("Song lifecycle timeline (selected top 30 by observed duration)")
        plot_life = filtered_life.sort_values("days_on_playlist", ascending=False).head(30).sort_values("entry_date")
        fig, ax = plt.subplots(figsize=(10, max(4, len(plot_life)*0.25)))
        for y, (_, r) in enumerate(plot_life.iterrows()):
            ax.plot([r.entry_date, r.last_seen_date], [y,y], linewidth=4)
        ax.set_yticks(range(len(plot_life))); ax.set_yticklabels((plot_life.song + " — " + plot_life.artist).tolist(), fontsize=7); ax.set_xlabel("Date"); ax.set_title("First to last observed appearance (not necessarily continuous)"); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    else: st.info("No songs match these filters.")
with tab3:
    st.subheader("Entries, exits and churn over time")
    cc = churn[churn.date.between(start_date, end_date)]
    fig, ax = plt.subplots(figsize=(10,4)); ax.plot(cc.date, cc.entries, label="Entries"); ax.plot(cc.date, cc.exits, label="Exits"); ax.set_ylabel("Songs"); ax.legend(); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10,3)); ax.plot(cc.date, cc.churn_rate_pct); ax.set_ylabel("Churn (%)"); ax.set_xlabel("Date"); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.caption("Churn rate = average of entries and exits divided by current playlist size. The first snapshot has no prior day for comparison.")
with tab4:
    st.subheader("Explicit vs clean content")
    if len(filtered_life):
        fig, ax = plt.subplots(); sns.boxplot(data=filtered_life, x="is_explicit", y="days_on_playlist", ax=ax); ax.set_xlabel("Explicit content (False = clean)"); ax.set_ylabel("Observed days on playlist"); st.pyplot(fig); plt.close(fig)
        st.subheader("Single vs album tracks")
        summary = filtered_life.groupby("album_type").agg(song_count=("song_key","count"), avg_days=("days_on_playlist","mean"), median_days=("days_on_playlist","median"), avg_peak_position=("peak_position","mean")).reset_index()
        st.dataframe(summary.round(2), use_container_width=True, hide_index=True)
        fig, ax = plt.subplots(); sns.scatterplot(data=filtered_life, x="duration_ms", y="days_on_playlist", hue="album_type", alpha=.6, ax=ax); ax.set_xlabel("Duration (ms)"); ax.set_ylabel("Observed days"); st.pyplot(fig); plt.close(fig)
        st.subheader("Album size and longevity")
        fig, ax = plt.subplots(); sns.scatterplot(data=filtered_life, x="total_tracks", y="days_on_playlist", hue="album_type", alpha=.6, ax=ax); ax.set_xlabel("Tracks on album/release"); ax.set_ylabel("Observed days"); st.pyplot(fig); plt.close(fig)
    else: st.info("No songs match these filters.")
with tab5:
    st.subheader("Machine-learning model: predict observed playlist lifetime")
    st.write("Random Forest regression using song metadata. The model target is `days_on_playlist`; identifiers and rank-derived lifecycle fields are excluded to reduce leakage.")
    model_df = life.dropna(subset=["days_on_playlist"]).copy()
    feature_cols = ["popularity", "duration_ms", "total_tracks", "album_type", "is_explicit"]
    if len(model_df) >= 10 and model_df.days_on_playlist.nunique() > 1:
        X, y = model_df[feature_cols], model_df["days_on_playlist"]
        numeric = ["popularity", "duration_ms", "total_tracks"]
        categorical = ["album_type", "is_explicit"]
        pre = ColumnTransformer([("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
                                 ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)])
        model = Pipeline([("preprocess", pre), ("model", RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1))])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
        model.fit(X_train, y_train); pred = model.predict(X_test)
        m1,m2,m3 = st.columns(3); m1.metric("MAE (days)", f"{mean_absolute_error(y_test,pred):.2f}"); m2.metric("RMSE (days)", f"{np.sqrt(mean_squared_error(y_test,pred)):.2f}"); m3.metric("R²", f"{r2_score(y_test,pred):.3f}")
        st.caption("Random train/test split is a baseline. For stronger validation, use a time-based split and account for tracks whose observed lifetime is incomplete at the dataset boundary.")
        st.subheader("Try a song profile")
        a,b,c = st.columns(3)
        with a: popularity = st.slider("Popularity", 0, 100, 65); duration = st.number_input("Duration (seconds)", 60, 900, 200)
        with b: tracks = st.number_input("Total tracks", 1, 100, 1); album_type = st.selectbox("Album type", sorted(model_df.album_type.dropna().unique()))
        with c: explicit = st.selectbox("Explicit", [False, True], format_func=lambda x: "Explicit" if x else "Clean")
        if st.button("Estimate playlist lifetime"):
            sample = pd.DataFrame([{"popularity": popularity, "duration_ms": duration*1000, "total_tracks": tracks, "album_type": album_type, "is_explicit": explicit}])
            st.success(f"Estimated observed playlist lifetime: {max(0, model.predict(sample)[0]):.1f} days")
    else: st.info("Not enough variation or rows to train the model.")

st.divider()
st.subheader("Download analysis tables")
st.download_button("Download song lifecycle CSV", filtered_life.to_csv(index=False).encode("utf-8"), "song_lifecycle.csv", "text/csv")
st.download_button("Download daily churn CSV", churn.to_csv(index=False).encode("utf-8"), "daily_churn.csv", "text/csv")
st.caption("For research use: validate duplicate snapshots, missing dates, re-entries, and right-censoring before drawing strategic conclusions.")
