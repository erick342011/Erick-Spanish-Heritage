# Hispanic Heritage Historical Data Explorer

A Streamlit classroom project that retrieves historical indicator data from the official World Bank API, fits a centered-time polynomial, calculates single and batch estimates, compares countries, exports reports, and includes a symbolic function/calculus lab.

## Files

- `app.py` — self-contained Streamlit application
- `requirements.txt` — minimal unpinned dependencies for deployment
- `sources.md` — source URLs, indicator definitions, and notes

## Run locally

Use Python 3.10+ in a virtual environment if possible.

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt`, `README.md`, and `sources.md` to the repository root.
3. Push/commit the files.
4. In Streamlit Community Cloud, create an app from the repository, select the branch, and set the main file path to `app.py`.
5. Deploy. No API key is needed for the public World Bank API.

## Deploy to Render

Create a **Python Web Service** connected to the repository.

- Build command: `pip install -r requirements.txt`
- Start command: `python -m streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT`

## Data and mathematical notes

The requested data period is the current year minus 70 through the previous year, inclusive. World Bank series often contain missing years, so the app displays actual coverage and observation count. It does not fill missing observations with random or fabricated values.

The polynomial model uses centered time `t = year - base year`. R-squared and MAE are labeled in-sample. A polynomial curve can fit historical values without being a reliable forecast, especially outside the observed interval. MAE is not a prediction interval. The function lab is educational and context descriptions do not validate the model scientifically.

## Quick verification

- Compare selected source values with the linked World Bank API response.
- Edit a row and verify that the app labels it a user-modified scenario; use Reset to source data to restore original values.
- Test duplicate years, non-integer years, invalid values, and too few rows for the selected degree.
- Check that single-year and batch estimates agree for the same year.
- Change sample increment, forward horizon, comparison countries, degree, and course mode.
- Open CSV, Excel, and PDF exports and verify their formats and contents.
- After deployment, inspect application logs if a dependency or API error occurs.

## Limitations

This is a teaching prototype. It provides numerical/symbolic analysis where feasible, but general-purpose root finding can miss difficult roots, and interval extrema depend on finding all relevant critical points and respecting the function domain. The playback is a lightweight rerun-driven simulation, not a real-time animation. Medical, engineering, and other visual examples are illustrative and not professional advice.
