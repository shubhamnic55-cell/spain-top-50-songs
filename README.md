# Spain Top 50 Content Maturity & Playlist Rotation Analysis

## Features
- Daily snapshot validation (expected 50 rows/day)
- Song lifecycle construction: first/last appearance, observed days, calendar span, peak rank and time-to-peak
- Daily entry/exit counts and churn rate
- Explicit/clean, single/album, duration and album-size comparisons
- Random Forest regression baseline to estimate observed playlist lifetime from available metadata
- Interactive Streamlit dashboard and CSV exports

## Run on Windows
1. Install Python 3.10 or newer.
2. Open Command Prompt or VS Code terminal in this folder.
3. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
4. Install packages:
   ```bash
   pip install -r requirements.txt
   ```
5. Launch the dashboard:
   ```bash
   streamlit run app.py
   ```
6. Streamlit will print a local URL, usually `http://localhost:8501`.

## Files
- `Atlantic_Spain.csv`: supplied dataset
- `analysis.py`: cleaning, validation, lifecycle and churn features
- `app.py`: dashboard and model

## Methodology cautions
- A song is identified by normalized title + artist; alternate spellings or changing artist credits may still split one song into multiple keys.
- `days_on_playlist` is the number of distinct observed dates in the playlist, not necessarily a continuous run. `calendar_span_days` includes gaps.
- Tracks last seen on the final dataset date are right-censored: their true eventual lifetime may be longer.
- The dataset has at least one daily snapshot with 100 rows, so the validation panel reports deviations rather than silently forcing each date to 50 rows.
- Popularity is a metadata score, not a daily stream count. Model predictions are exploratory, not causal evidence.
- For a stronger model, use a chronological holdout and survival analysis / censor-aware methods.
