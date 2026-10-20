"""
Hispanic Heritage Historical Data Explorer
A self-contained Streamlit app for World Bank historical data and function analysis.
"""
from __future__ import annotations

import io
import math
import re
from datetime import date, datetime
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import sympy as sp
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle

st.set_page_config(page_title="Hispanic Heritage Historical Data Explorer", page_icon="🌎", layout="wide")

COUNTRIES = {
    "Mexico": "MEX", "Brazil": "BRA", "Argentina": "ARG", "Colombia": "COL",
    "Chile": "CHL", "Peru": "PER", "Venezuela": "VEN", "Ecuador": "ECU", "Uruguay": "URY"
}
INDICATORS = {
    "Life expectancy at birth": ("SP.DYN.LE00.IN", "years", "Average number of years a newborn would live if current age-specific mortality patterns continued."),
    "Unemployment, national estimate": ("SL.UEM.TOTL.NE.ZS", "% of labor force", "Share of the labor force without work but available for and seeking employment, using national estimates."),
    "GDP per capita": ("NY.GDP.PCAP.CD", "current US$ per person", "Gross domestic product divided by midyear population. It is not average wealth or household income."),
    "Crude birth rate": ("SP.DYN.CBRT.IN", "births per 1,000 people per year", "Annual live births per 1,000 people; this is a rate, not a birth count."),
    "Net migration": ("SM.POP.NETM", "people", "Net number of migrants during the period; it is not an emigration count."),
    "Intentional homicide rate": ("VC.IHR.PSRC.P5", "per 100,000 people", "Intentional homicides per 100,000 population."),
    "Consumer price inflation": ("FP.CPI.TOTL.ZG", "% annual change", "Annual percentage change in the consumer price index."),
    "Central government debt": ("GC.DOD.TOTL.GD.ZS", "% of GDP", "Central government debt as a percentage of GDP."),
    "Total reserves including gold": ("FI.RES.TOTL.CD", "current US$", "Total reserves including gold, in current US dollars."),
    "External balance on goods/services": ("NE.RSB.GNFS.ZS", "% of GDP", "External balance on goods and services as a percentage of GDP.")
}
WB_BASE = "https://api.worldbank.org/v2"
SAFE_FUNCS = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp, "log": sp.log,
    "sqrt": sp.sqrt, "abs": sp.Abs, "pi": sp.pi, "E": sp.E, "asin": sp.asin,
    "acos": sp.acos, "atan": sp.atan
}
CONTEXTS = {
    "Historical data": ("Year", "Indicator value", "year", "indicator units", "Use the fitted historical-data curve to explore trends. Polynomial fits are descriptive, not causal."),
    "Business": ("Time", "Revenue", "months", "currency/month", "Revenue growth is the first derivative; changing growth is the second derivative."),
    "Engineering": ("Time", "Position", "seconds", "meters", "For position, the first derivative is velocity and the second derivative is acceleration."),
    "Medical": ("Time", "Concentration", "hours", "mg/L", "A concentration curve can represent a hypothetical concentration profile; it is not medical advice."),
    "Sports science": ("Time", "Distance", "seconds", "meters", "Distance change per second is a rate; use a physically meaningful domain."),
    "Biology": ("Time", "Population", "days", "organisms", "Growth and changing growth can be explored, but real populations have constraints."),
    "Bioengineering": ("Time", "Signal", "seconds", "signal units", "A signal and its derivatives describe change and changing change."),
    "Economic forecasting": ("Year", "Economic indicator", "years", "indicator units", "Forecasts are uncertain and should not be treated as facts."),
    "Weather": ("Time", "Temperature", "hours", "°C", "The first derivative describes warming/cooling rate; the second describes how that rate changes."),
    "Custom": ("x", "f(x)", "x-units", "y-units", "Edit the context and units to match your own example.")
}
SUGGESTIONS = {
    "Business": "0.15*x**2 + 2*x + 10",
    "Engineering": "0.5*x**2 + 3*x",
    "Medical": "10*exp(-0.3*x)",
    "Sports science": "-0.5*x**2 + 8*x",
    "Biology": "2*exp(0.15*x)",
    "Bioengineering": "sin(x) + 0.2*x",
    "Economic forecasting": "0.1*x**2 + 2*x + 100",
    "Weather": "5*sin(2*pi*x/24) + 20",
    "Custom": "x**3 - 3*x",
    "Historical data": "Use fitted data curve"
}

def world_bank_data(country_code: str, indicator_code: str, start_year: int, end_year: int):
    url = f"{WB_BASE}/country/{country_code}/indicator/{indicator_code}"
    params = {"date": f"{start_year}:{end_year}", "format": "json", "per_page": 20000}
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
            return pd.DataFrame(columns=["year", "value"]), f"Unexpected/empty API response from {url}"
        rows = [{"year": int(item["date"]), "value": item["value"]}
                for item in payload[1] if item.get("date") and item.get("value") is not None]
        df = pd.DataFrame(rows, columns=["year", "value"])
        if not df.empty:
            df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna().astype({"year": int}).sort_values("year").reset_index(drop=True)
        return df, None if not df.empty else f"No non-null observations returned for {start_year}–{end_year}."
    except (requests.RequestException, ValueError, TypeError, KeyError) as exc:
        return pd.DataFrame(columns=["year", "value"]), f"Could not retrieve World Bank data: {exc}"

@st.cache_data(ttl=3600, show_spinner=False)
def cached_world_bank_data(country_code: str, indicator_code: str, start_year: int, end_year: int):
    return world_bank_data(country_code, indicator_code, start_year, end_year)

def validate_data(df: pd.DataFrame, indicator_name: str):
    errors = []
    if df.empty:
        return ["There are no rows to fit."]
    if not {"year", "value"}.issubset(df.columns):
        return ["The table must have year and value columns."]
    years = pd.to_numeric(df["year"], errors="coerce")
    values = pd.to_numeric(df["value"], errors="coerce")
    if years.isna().any() or ((years % 1) != 0).any():
        errors.append("Years must be whole-number integers.")
    if years.duplicated().any():
        errors.append("Each year must be unique.")
    if values.isna().any() or not np.isfinite(values.dropna()).all():
        errors.append("Values must be numeric and finite.")
    if indicator_name in ("Life expectancy at birth", "Unemployment, national estimate", "Crude birth rate", "Intentional homicide rate", "Total reserves including gold") and (values.dropna() < 0).any():
        errors.append("This indicator cannot contain negative values.")
    if indicator_name == "Life expectancy at birth" and (values.dropna() > 125).any():
        errors.append("Life expectancy values above 125 years need checking.")
    if indicator_name == "Unemployment, national estimate" and (values.dropna() > 100).any():
        errors.append("Unemployment percentage cannot exceed 100.")
    return errors

def safe_expression(source: str, x: sp.Symbol):
    """Parse a user expression while allowing only common mathematical operations/functions."""
    normalized = source.replace("^", "**")
    if not re.fullmatch(r"[0-9a-zA-Z_+\-*/().,\s*]+", normalized):
        raise ValueError("Use standard math notation, numbers, x, operators, and supported functions only.")
    try:
        expr = sp.sympify(normalized, locals={"x": x, **SAFE_FUNCS}, evaluate=True)
    except Exception as exc:
        raise ValueError(f"Could not parse expression: {exc}") from exc
    allowed = {x, *SAFE_FUNCS.values()}
    if not expr.free_symbols.issubset({x}):
        raise ValueError("Only x may be used as a variable.")
    if expr.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        raise ValueError("Expression contains an undefined or infinite constant.")
    return expr

def numeric_function(expr, xs):
    fn = sp.lambdify(sp.Symbol("x"), expr, modules=["numpy"])
    with np.errstate(all="ignore"):
        ys = np.asarray(fn(xs))
    if ys.shape == ():
        ys = np.full(np.asarray(xs).shape, float(ys))
    return np.asarray(ys, dtype=float)

def fmt_num(value, digits=6):
    try:
        if value is None or not np.isfinite(float(value)):
            return "undefined"
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return str(value)

def derivative_text(expr):
    return sp.sstr(sp.simplify(expr)).replace("**", "^")

def calculate_calculus(expr, x, interval):
    d1, d2 = sp.diff(expr, x), sp.diff(expr, x, 2)
    lo, hi = sorted(interval)
    critical = []
    try:
        for sol in sp.solve(sp.Eq(d1, 0), x):
            if sol.is_real is True and sol.is_number:
                val = float(sol)
                if lo <= val <= hi and np.isfinite(val):
                    critical.append(val)
    except Exception:
        pass
    # Include derivative singularities as candidates, but do not classify them as extrema automatically.
    candidates = sorted(set([lo, hi] + critical))
    extrema = []
    d2f = sp.lambdify(x, d2, "numpy")
    d1f = sp.lambdify(x, d1, "numpy")
    f = sp.lambdify(x, expr, "numpy")
    for c in critical:
        try:
            second = float(d2f(c))
            if second > 1e-7:
                label = "local minimum (second derivative > 0)"
            elif second < -1e-7:
                label = "local maximum (second derivative < 0)"
            else:
                left, right = max(lo, c - 1e-4), min(hi, c + 1e-4)
                dl, dr = float(d1f(left)), float(d1f(right))
                label = "local minimum (first derivative changes − to +)" if dl < 0 < dr else (
                    "local maximum (first derivative changes + to −)" if dl > 0 > dr else "stationary point; no extremum confirmed")
            extrema.append((c, label, float(f(c))))
        except Exception:
            extrema.append((c, "critical point; classification unavailable", float("nan")))
    # Numerically test candidate inflection points where f''=0 and concavity changes.
    inflections = []
    try:
        for sol in sp.solve(sp.Eq(d2, 0), x):
            if sol.is_real is True and sol.is_number:
                c = float(sol)
                if lo < c < hi:
                    eps = max(1e-5, (hi - lo) * 1e-5)
                    try:
                        lval, rval = float(d2f(c-eps)), float(d2f(c+eps))
                        if np.isfinite(lval) and np.isfinite(rval) and lval * rval < 0:
                            inflections.append((c, float(f(c))))
                    except Exception:
                        pass
    except Exception:
        pass
    return d1, d2, critical, extrema, inflections

def make_chart(df, x_symbol, expr, fit_expr, prediction_points, x_min, x_max, sample_increment, show_original, show_first, show_second, current_x, secant_a, secant_b, single_estimate=None):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    grid = np.linspace(x_min, x_max, 800)
    if show_original and not df.empty:
        ax.scatter(df["year"], df["value"], label="Observed source data", s=24, zorder=4)
    if fit_expr is not None:
        ys = numeric_function(fit_expr, grid)
        good = np.isfinite(ys)
        # Break lines at non-finite values so curves do not connect across singularities.
        ax.plot(grid, np.where(good, ys, np.nan), label="Historical polynomial fit", linewidth=2)
    elif expr is not None:
        ys = numeric_function(expr, grid)
        good = np.isfinite(ys)
        ax.plot(grid, np.where(good, ys, np.nan), label="Selected function", linewidth=2)
        if show_first:
            d1 = sp.diff(expr, x_symbol)
            ax.plot(grid, numeric_function(d1, grid), label="First derivative", linestyle="--")
        if show_second:
            d2 = sp.diff(expr, x_symbol, 2)
            ax.plot(grid, numeric_function(d2, grid), label="Second derivative", linestyle=":")
        try:
            fx = float(sp.N(expr.subs(x_symbol, current_x)))
            ax.scatter([current_x], [fx], label=f"Current point ({current_x:g}, {fx:.4g})", s=55, zorder=5)
            d1val = float(sp.N(sp.diff(expr, x_symbol).subs(x_symbol, current_x)))
            tangent = fx + d1val * (grid - current_x)
            ax.plot(grid, tangent, label="Tangent at current point", alpha=0.65, linestyle="-.")
            if secant_a != secant_b:
                fa = float(sp.N(expr.subs(x_symbol, secant_a)))
                fb = float(sp.N(expr.subs(x_symbol, secant_b)))
                slope = (fb-fa)/(secant_b-secant_a)
                ax.plot(grid, fa+slope*(grid-secant_a), label="Secant over selected interval", alpha=0.55, linestyle="--")
        except Exception:
            pass
    if prediction_points:
        px, py = zip(*prediction_points)
        ax.scatter(px, py, marker="x", s=70, label="Calculated estimates", zorder=6)
    if single_estimate is not None:
        ax.scatter([single_estimate[0]], [single_estimate[1]], marker="D", s=60, label="Single-year estimate", zorder=7)
    ax.set_xlabel("Year" if fit_expr is not None else "Independent variable")
    ax.set_ylabel("Indicator value" if fit_expr is not None else "Function value / derivative units")
    ax.grid(True, alpha=.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig

def pdf_export(title, metadata, equation, metrics, findings, preview, results, chart_png):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter), rightMargin=30, leftMargin=30, topMargin=28, bottomMargin=28)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Spacer(1, 8), Paragraph(metadata, styles["BodyText"]), Spacer(1, 8),
             Paragraph(f"<b>Function / equation:</b> {equation}", styles["BodyText"]), Spacer(1, 6),
             Paragraph(f"<b>Metrics:</b> {metrics}", styles["BodyText"]), Spacer(1, 6),
             Paragraph(f"<b>Findings and limitations:</b> {findings}", styles["BodyText"]), Spacer(1, 8)]
    if chart_png:
        story.append(RLImage(io.BytesIO(chart_png), width=650, height=350))
        story.append(Spacer(1, 8))
    for heading, frame in [("Data preview", preview), ("Calculated-year results", results)]:
        story.append(Paragraph(f"<b>{heading}</b>", styles["Heading2"]))
        if frame is None or frame.empty:
            story.append(Paragraph("No rows available.", styles["BodyText"]))
        else:
            temp = frame.head(18).copy().fillna("")
            data = [list(temp.columns)] + temp.astype(str).values.tolist()
            table = Table(data, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.lightgrey), ("GRID", (0,0), (-1,-1), .3, colors.grey),
                ("FONTSIZE", (0,0), (-1,-1), 7), ("VALIGN", (0,0), (-1,-1), "TOP"),
            ]))
            story.append(table)
        story.append(Spacer(1, 8))
    doc.build(story)
    return buffer.getvalue()

st.title("🌎 Hispanic Heritage Historical Data Explorer")
st.caption("Explore authentic World Bank indicators, fit historical trends, and analyze functions by course level.")
st.info("Brazil is included as a Latin American comparison country. World Bank observations may be missing or revised; the app displays actual coverage rather than assuming 70 observations.")

now_year = date.today().year
start_year, end_year = now_year - 70, now_year - 1
with st.sidebar:
    st.header("Data and model controls")
    country_name = st.selectbox("Country", list(COUNTRIES.keys()), index=0)
    indicator_name = st.selectbox("World Bank indicator", list(INDICATORS.keys()), index=0)
    degree = st.slider("Polynomial degree (2–8)", 2, 8, 3)
    sample_increment = st.slider("Sample marker/table increment (years)", 1, 10, 5)
    horizon = st.slider("Forward display horizon (years)", 0, 50, 10)
    course = st.selectbox("Mathematics pathway", ["Honors Algebra 2", "Precalculus", "Calculus"])
    comparison_name = st.selectbox("Comparison country", ["None"] + [c for c in COUNTRIES if c != country_name])
    printer_mode = st.checkbox("Printer-friendly report view", False)

indicator_code, units, definition = INDICATORS[indicator_name]
country_code = COUNTRIES[country_name]
source_url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}?date={start_year}:{end_year}&format=json"
with st.spinner("Retrieving official World Bank data..."):
    source_df, api_error = cached_world_bank_data(country_code, indicator_code, start_year, end_year)
if api_error:
    st.warning(api_error)
source_df = source_df.copy()
if "source_data" not in st.session_state:
    st.session_state.source_data = source_df.copy()
if st.session_state.get("source_key") != (country_code, indicator_code, start_year, end_year):
    st.session_state.source_key = (country_code, indicator_code, start_year, end_year)
    st.session_state.source_data = source_df.copy()
    st.session_state.working_data = source_df.copy()
    st.session_state.edited_flag = False
    st.session_state.single_result = None
    st.session_state.batch_result = pd.DataFrame()
if "working_data" not in st.session_state:
    st.session_state.working_data = source_df.copy()
    st.session_state.edited_flag = False
if st.session_state.get("model_key") != (country_code, indicator_code, degree):
    st.session_state.model_key = (country_code, indicator_code, degree)
    st.session_state.single_result = None
    st.session_state.batch_result = pd.DataFrame()

retrieved_date = date.today().isoformat()
actual_years = st.session_state.source_data["year"] if not st.session_state.source_data.empty else pd.Series(dtype=int)
coverage = f"{int(actual_years.min())}–{int(actual_years.max())}" if len(actual_years) else "no observations"
col1, col2, col3, col4 = st.columns(4)
col1.metric("Available observations", f"{len(st.session_state.source_data)}")
col2.metric("Actual year coverage", coverage)
col3.metric("Requested years", f"{start_year}–{end_year}")
col4.metric("Retrieved", retrieved_date)
st.markdown(f"**Definition:** {definition}  \n**Units:** {units}  \n**Indicator code:** `{indicator_code}`  \n**Source:** [World Bank API]({source_url})")
st.caption("Historical data source: World Bank API. Retrieval date is shown above. Missing years are not filled with invented values.")

tabs = st.tabs(["Historical data & fit", "Function lab & calculus", "Exports & verification"])
with tabs[0]:
    st.subheader("Source data and editable working table")
    st.write("Original source data is preserved separately. Edit the working table to create a user-modified scenario.")
    if st.button("Reset working table to source"):
        st.session_state.working_data = st.session_state.source_data.copy()
        st.session_state.edited_flag = False
        st.session_state.single_result = None
        st.session_state.batch_result = pd.DataFrame()
        st.rerun()
    working = st.data_editor(
        st.session_state.working_data,
        num_rows="dynamic",
        use_container_width=True,
        key=f"editor_{country_code}_{indicator_code}",
        column_config={"year": st.column_config.NumberColumn("Year", step=1, format="%d"), "value": st.column_config.NumberColumn("Value", format="%.8g")}
    )
    if not working.equals(st.session_state.working_data):
        st.session_state.edited_flag = True
        st.session_state.working_data = working.copy()
        st.session_state.single_result = None
        st.session_state.batch_result = pd.DataFrame()
    if st.session_state.edited_flag:
        st.warning("User-modified scenario: exports and model use the edited working table, not only the source data.")
    errors = validate_data(st.session_state.working_data, indicator_name)
    if errors:
        for err in errors:
            st.error(err)
    clean = st.session_state.working_data.copy()
    if not clean.empty and not errors:
        clean["year"] = pd.to_numeric(clean["year"]).astype(int)
        clean["value"] = pd.to_numeric(clean["value"]).astype(float)
        clean = clean.sort_values("year").reset_index(drop=True)
    fit_expr = None
    fit_metrics = {}
    if not errors and len(clean) >= degree + 1 and clean["year"].nunique() >= degree + 1:
        base_year = int(clean["year"].min())
        t = clean["year"].to_numpy(dtype=float) - base_year
        vals = clean["value"].to_numpy(dtype=float)
        try:
            coeff = np.polynomial.polynomial.polyfit(t, vals, degree)
            tx = sp.Symbol("x", real=True)
            fit_expr = sp.Add(*[sp.Float(float(c), 14) * (tx-base_year)**i for i, c in enumerate(coeff)])
            fitted = np.polynomial.polynomial.polyval(t, coeff)
            residuals = vals - fitted
            ss_res = float(np.sum(residuals**2))
            ss_tot = float(np.sum((vals - vals.mean())**2))
            r2 = 1 - ss_res/ss_tot if ss_tot > 0 else float("nan")
            mae = float(np.mean(np.abs(residuals)))
            fit_metrics = {"base_year": base_year, "r2": r2, "mae": mae, "mean": float(np.mean(vals)),
                           "residuals": residuals, "fitted": fitted, "coeff": coeff, "n": len(clean)}
        except Exception as exc:
            st.error(f"Polynomial fit failed: {exc}")
    elif not errors:
        st.warning(f"Degree {degree} needs at least {degree+1} distinct years; currently there are {clean['year'].nunique()}.")
    if fit_expr is not None:
        st.markdown("**Fitted polynomial** (centered time; \(t = year - base\ year\))")
        st.latex(sp.latex(sp.expand(fit_expr)).replace("x", "year"))
        a,b,c,d = st.columns(4)
        a.metric("R² (in-sample)", fmt_num(fit_metrics["r2"]))
        b.metric("MAE (in-sample)", fmt_num(fit_metrics["mae"]))
        c.metric("Mean observed value", fmt_num(fit_metrics["mean"]))
        d.metric("Fit observations", str(fit_metrics["n"]))
        st.caption(f"Base year: {fit_metrics['base_year']}. Historical range: {int(clean.year.min())}–{int(clean.year.max())}. Residual = observed − fitted. Metrics describe the fitted data, not future accuracy.")
        residual_frame = pd.DataFrame({"year": clean["year"], "observed": vals, "fitted": fit_metrics["fitted"], "residual": fit_metrics["residuals"]})
        with st.expander("Residuals and fitted values"):
            st.dataframe(residual_frame, use_container_width=True)
        if len(clean) >= max(degree + 3, 8):
            holdout_n = max(1, int(round(len(clean)*0.2)))
            train_n = len(clean)-holdout_n
            if train_n >= degree+1:
                tr = clean.iloc[:train_n]
                te = clean.iloc[train_n:]
                hc = np.polynomial.polynomial.polyfit(tr.year.to_numpy(float)-fit_metrics["base_year"], tr.value.to_numpy(float), degree)
                hp = np.polynomial.polynomial.polyval(te.year.to_numpy(float)-fit_metrics["base_year"], hc)
                holdout_mae = float(np.mean(np.abs(te.value.to_numpy(float)-hp)))
                st.write(f"Chronological holdout MAE (last {len(te)} observations, trained on earlier years): **{fmt_num(holdout_mae)}**. This score is separate from in-sample metrics.")
        st.subheader("Single-year and batch estimates")
        min_year, max_year = 1950, 2100
        single_year = st.number_input("Single year to estimate", min_value=min_year, max_value=max_year, value=min(max(int(clean.year.max()), min_year), max_year), step=1)
        if st.button("Calculate Value", key="single_calc"):
            est = float(np.polynomial.polynomial.polyval(float(single_year)-fit_metrics["base_year"], fit_metrics["coeff"]))
            st.session_state.single_result = {"year": int(single_year), "estimate": est, "within": int(clean.year.min()) <= single_year <= int(clean.year.max()), "key": (country_code, indicator_code, degree, tuple(clean.year), tuple(clean.value))}
        result = st.session_state.single_result
        if result and result.get("key") == (country_code, indicator_code, degree, tuple(clean.year), tuple(clean.value)):
            classification = "within observed fitted range" if result["within"] else "outside observed fitted range (extrapolation)"
            st.write(f"**{result['year']} estimate:** {fmt_num(result['estimate'])} {units}; {classification}. This is a model estimate, not an observed value.")
            if not result["within"]:
                st.warning("Polynomial extrapolations can become implausible. The app does not clip the estimate. Interpret with caution.")
        batch_mode = st.radio("Batch year input", ["Year range", "Custom years"], horizontal=True)
        batch_text = ""
        if batch_mode == "Year range":
            c1,c2,c3 = st.columns(3)
            yr_start = c1.number_input("Start year", 1950, 2100, int(clean.year.min()), key="batch_start")
            yr_end = c2.number_input("End year", 1950, 2100, int(clean.year.max()), key="batch_end")
            step = c3.slider("Range step", 1, 10, 1, key="batch_step")
            years = list(range(int(yr_start), int(yr_end)+1, int(step))) if yr_start <= yr_end else []
            if yr_start > yr_end: st.error("Start year must be less than or equal to end year.")
        else:
            batch_text = st.text_input("Comma-separated years (1950–2100)", "2000, 2010, 2020")
            years = []
            invalid_tokens = []
            for token in batch_text.split(","):
                token = token.strip()
                if not token: continue
                try:
                    y = int(token)
                    if not 1950 <= y <= 2100: invalid_tokens.append(token)
                    else: years.append(y)
                except ValueError:
                    invalid_tokens.append(token)
            if invalid_tokens: st.error("Invalid year entries ignored: " + ", ".join(invalid_tokens))
            years = list(dict.fromkeys(years))
        if st.button("Calculate batch results"):
            if not years:
                st.error("Enter at least one valid year.")
            else:
                vals_pred = [float(np.polynomial.polynomial.polyval(y-fit_metrics["base_year"], fit_metrics["coeff"])) for y in years]
                st.session_state.batch_result = pd.DataFrame({"year": years, "estimate": vals_pred, "units": units,
                    "classification": ["within observed fitted range" if clean.year.min() <= y <= clean.year.max() else "outside observed fitted range" for y in years]})
                st.session_state.batch_key = (country_code, indicator_code, degree, tuple(clean.year), tuple(clean.value))
        batch_df = st.session_state.get("batch_result", pd.DataFrame())
        if not batch_df.empty and st.session_state.get("batch_key") == (country_code, indicator_code, degree, tuple(clean.year), tuple(clean.value)):
            st.dataframe(batch_df, use_container_width=True)
        st.subheader("Historical fit and estimates")
        plot_min = min(int(clean.year.min()), int(single_year))
        plot_max = max(int(clean.year.max()), int(single_year)) + horizon
        prediction_points = []
        if not batch_df.empty:
            prediction_points = [(float(r.year), float(r.estimate)) for r in batch_df.itertuples()]
        single_point = (result["year"], result["estimate"]) if result and result.get("key") == (country_code, indicator_code, degree, tuple(clean.year), tuple(clean.value)) else None
        sample = clean.iloc[::sample_increment]
        st.caption(f"Sample table uses every {sample_increment} year(s); the smooth fitted curve is plotted separately.")
        st.dataframe(sample, use_container_width=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        grid = np.linspace(plot_min, plot_max, 800)
        yy = np.polynomial.polynomial.polyval(grid-fit_metrics["base_year"], fit_metrics["coeff"])
        hist_mask = grid <= int(clean.year.max())
        ax.scatter(clean.year, clean.value, s=25, label="Observed data")
        ax.plot(grid[hist_mask], yy[hist_mask], label="Fit over historical range", linewidth=2)
        if horizon > 0:
            ax.plot(grid[~hist_mask], yy[~hist_mask], linestyle="--", label="Extrapolated segment")
            ax.axvline(int(clean.year.max()), linestyle=":", label="Historical boundary")
        if prediction_points:
            px, py = zip(*prediction_points); ax.scatter(px, py, marker="x", s=65, label="Batch estimates")
        if single_point: ax.scatter([single_point[0]], [single_point[1]], marker="D", label="Single estimate")
        ax.set_xlabel("Year"); ax.set_ylabel(units); ax.grid(True, alpha=.25); ax.legend(); fig.tight_layout()
        if printer_mode:
            st.markdown("### Printable summary")
            st.write(f"{country_name} — {indicator_name}; {coverage}; degree {degree}; R² {fmt_num(fit_metrics['r2'])}; MAE {fmt_num(fit_metrics['mae'])}.")
        st.pyplot(fig, use_container_width=True)
        st.session_state.current_chart_png = None
        pngbuf = io.BytesIO(); fig.savefig(pngbuf, format="png", dpi=140, bbox_inches="tight"); st.session_state.current_chart_png = pngbuf.getvalue()
        plt.close(fig)
    else:
        st.info("A fit and estimates will appear after the working table passes validation and contains enough distinct years.")
    if comparison_name != "None":
        st.subheader(f"Country comparison: {country_name} and {comparison_name}")
        comp_code = COUNTRIES[comparison_name]
        comp_df, comp_error = cached_world_bank_data(comp_code, indicator_code, start_year, end_year)
        if comp_error: st.warning(f"{comparison_name}: {comp_error}")
        else:
            joined = pd.merge(st.session_state.source_data.rename(columns={"value": country_name}),
                              comp_df.rename(columns={"value": comparison_name}), on="year", how="outer").sort_values("year")
            st.line_chart(joined.set_index("year")[[country_name, comparison_name]])
            st.write(f"Coverage — {country_name}: {coverage}; {comparison_name}: " +
                     (f"{int(comp_df.year.min())}–{int(comp_df.year.max())} ({len(comp_df)} observations)." if not comp_df.empty else "no observations."))
            st.caption("Gaps reflect missing source observations; lines should not be interpreted as evidence of causes.")

with tabs[1]:
    st.subheader("Function lab")
    context = st.selectbox("Context", list(CONTEXTS.keys()))
    default_x, default_y, default_xu, default_yu, context_description = CONTEXTS[context]
    if context == "Historical data":
        st.write("The fitted historical polynomial is available in the first tab. For a standalone function lab, choose another context or enter a custom function.")
    context_description = st.text_area("Context description", context_description)
    v1,v2,v3,v4 = st.columns(4)
    x_name = v1.text_input("Independent variable", default_x)
    y_name = v2.text_input("Dependent variable", default_y)
    x_units = v3.text_input("Independent units", default_xu)
    y_units = v4.text_input("Dependent units", default_yu)
    function_mode = st.radio("Function source", ["Suggested function", "Enter my own function", "Use historical fitted curve"], horizontal=True)
    default_expr = SUGGESTIONS.get(context, "x**2")
    if function_mode == "Suggested function":
        expr_text = st.text_input("Suggested function (editable)", default_expr)
    elif function_mode == "Enter my own function":
        expr_text = st.text_input("Function in x (use ** or ^ for powers)", "x**3 - 3*x")
    else:
        expr_text = ""
    custom_min, custom_max = st.columns(2)
    domain_min = custom_min.number_input("Analysis interval minimum", value=0.0, step=1.0)
    domain_max = custom_max.number_input("Analysis interval maximum", value=10.0, step=1.0)
    if domain_min >= domain_max:
        st.error("Analysis interval minimum must be less than maximum.")
    x = sp.Symbol("x", real=True)
    expr = None
    if function_mode == "Use historical fitted curve":
        if "fit_expr" in locals() and fit_expr is not None:
            expr = fit_expr
        else:
            st.warning("A valid historical fit is not available. Choose a suggested or custom function.")
    else:
        try:
            expr = safe_expression(expr_text, x)
        except ValueError as exc:
            st.error(str(exc))
    if expr is not None and domain_min < domain_max:
        st.latex(f"f(x) = {sp.latex(expr)}")
        d1, d2 = sp.diff(expr, x), sp.diff(expr, x, 2)
        st.markdown(f"**First derivative:** \(f'(x) = {sp.latex(d1)}\)  \n**Second derivative:** \(f''(x) = {sp.latex(d2)}\)")
        c1,c2,c3 = st.columns(3)
        with c1:
            current_x = st.slider("Current input x", float(domain_min), float(domain_max), float((domain_min+domain_max)/2), step=max((domain_max-domain_min)/100, .001))
        with c2:
            secant_a = st.slider("Secant interval start a", float(domain_min), float(domain_max), float(domain_min))
        with c3:
            secant_b = st.slider("Secant interval end b", float(domain_min), float(domain_max), float(domain_max))
        play_col, speed_col, reset_col = st.columns(3)
        with play_col:
            play = st.checkbox("Auto-play simulation", False)
        with speed_col:
            speed = st.select_slider("Playback speed", options=["Slow", "Normal", "Fast"], value="Normal")
        with reset_col:
            if st.button("Reset simulation"):
                st.session_state.sim_x = float((domain_min+domain_max)/2)
                st.rerun()
        if play:
            st.info(f"Auto-play is enabled ({speed.lower()}). Streamlit reruns on interaction; move the input slider to advance the synchronized point.")
        try:
            fx = float(sp.N(expr.subs(x, current_x)))
            dfx = float(sp.N(d1.subs(x, current_x)))
            d2fx = float(sp.N(d2.subs(x, current_x)))
            st.write(f"At x = {current_x:g}, f(x) = {fmt_num(fx)} {y_units}; f′(x) = {fmt_num(dfx)} {y_units}/{x_units}; f″(x) = {fmt_num(d2fx)} {y_units}/{x_units}².")
            st.write(f"**Context interpretation:** {context_description} At this point, the first derivative means the instantaneous rate of change of {y_name} with respect to {x_name}; the second derivative describes how that rate changes.")
            if secant_a != secant_b:
                fa, fb = float(sp.N(expr.subs(x, secant_a))), float(sp.N(expr.subs(x, secant_b)))
                avg_rate = (fb-fa)/(secant_b-secant_a)
                st.write(f"**Average rate of change** from {secant_a:g} to {secant_b:g}: {fmt_num(avg_rate)} {y_units}/{x_units}.")
            else:
                st.write("Average rate of change is undefined when both interval endpoints are equal.")
        except Exception as exc:
            st.warning(f"The function is undefined at the selected point or interval: {exc}")
        try:
            d1, d2, critical, extrema, inflections = calculate_calculus(expr, x, (domain_min, domain_max))
            if course == "Calculus":
                st.markdown("### Calculus analysis")
                st.write("**Critical points** (numerically evaluated where solvable): " + (", ".join(fmt_num(v) for v in critical) if critical else "none found analytically in this interval"))
                st.write("**Local extrema checks:**")
                if extrema:
                    for val, label, yval in extrema:
                        st.write(f"- x ≈ {fmt_num(val)}, f(x) ≈ {fmt_num(yval)} — {label}.")
                else: st.write("- No stationary extrema confirmed in the interval.")
                st.write("**Absolute extrema on the closed interval:** compare function values at both endpoints and all interior critical points where the function is defined.")
                endpoint_values = []
                for v in [domain_min, domain_max] + critical:
                    try:
                        fv = float(sp.N(expr.subs(x, v)))
                        if np.isfinite(fv): endpoint_values.append((v, fv))
                    except Exception: pass
                if endpoint_values:
                    st.write(f"Candidates: {', '.join(f'x={fmt_num(a)} → {fmt_num(b)}' for a,b in endpoint_values)}.")
                    st.write(f"Numerical minimum candidate: x={fmt_num(min(endpoint_values, key=lambda p:p[1])[0])}; maximum candidate: x={fmt_num(max(endpoint_values, key=lambda p:p[1])[0])}.")
                st.write("**Inflection points with a numerical concavity-change check:** " +
                         (", ".join(f"x≈{fmt_num(a)}, f(x)≈{fmt_num(b)}" for a,b in inflections) if inflections else "none confirmed in this interval"))
                # Domain/singularity indicators based on sampled finite values and denominator zeros.
                denom = sp.denom(sp.together(expr))
                try:
                    discontinuities = [float(s) for s in sp.solve(sp.Eq(denom, 0), x) if s.is_real is True and s.is_number and domain_min <= float(s) <= domain_max] if denom != 1 else []
                except Exception: discontinuities = []
                st.write("**Domain restrictions / possible discontinuities:** " + (", ".join(fmt_num(v) for v in discontinuities) if discontinuities else "no real denominator zeros found in the selected interval; other function restrictions may still apply"))
                # Numeric monotonicity/concavity intervals, intentionally labeled approximate.
                grid = np.linspace(domain_min, domain_max, 2001)
                deriv_vals = numeric_function(d1, grid)
                second_vals = numeric_function(d2, grid)
                def runs(signs, xs):
                    out, start = [], 0
                    last = signs[0]
                    for i in range(1, len(signs)):
                        if signs[i] != last:
                            if last != 0: out.append((xs[start], xs[i-1], last))
                            start, last = i, signs[i]
                    if last != 0: out.append((xs[start], xs[-1], last))
                    return out
                inc = runs(np.sign(np.where(np.isfinite(deriv_vals), deriv_vals, 0)), grid)
                conc = runs(np.sign(np.where(np.isfinite(second_vals), second_vals, 0)), grid)
                st.write("**Approximate increasing/decreasing intervals:** " + "; ".join(f"{a:.3g}–{b:.3g} {'increasing' if s>0 else 'decreasing'}" for a,b,s in inc[:8]))
                st.write("**Approximate concavity intervals:** " + "; ".join(f"{a:.3g}–{b:.3g} {'concave up' if s>0 else 'concave down'}" for a,b,s in conc[:8]))
            elif course == "Precalculus":
                st.markdown("### Precalculus interpretation")
                st.write(f"Degree/coefficient changes alter end behavior and turning points. On the selected interval, f′(x) = {derivative_text(d1)} and f″(x) = {derivative_text(d2)}; changing average rates help reveal concavity.")
                st.write("For a polynomial, end behavior is controlled by the leading term. For non-polynomial functions, domain and asymptotes also matter.")
            else:
                st.markdown("### Honors Algebra 2 interpretation")
                st.write("Domain: the selected interval, subject to the function's mathematical restrictions. Range: approximate values on the plotted interval. Compare average rates to describe trends; polynomial end behavior depends on degree and leading coefficient.")
        except Exception as exc:
            st.warning(f"Some symbolic analysis could not be completed: {exc}")
        show_original = st.checkbox("Show observed historical data (where available)", True)
        show_first = st.checkbox("Show first derivative curve", course == "Calculus")
        show_second = st.checkbox("Show second derivative curve", course == "Calculus")
        fig = make_chart(pd.DataFrame(), x, expr, None, [], domain_min, domain_max, 1, show_original, show_first, show_second, current_x, secant_a, secant_b)
        st.pyplot(fig, use_container_width=True)
        pngbuf = io.BytesIO(); fig.savefig(pngbuf, format="png", dpi=140, bbox_inches="tight")
        st.session_state.function_chart_png = pngbuf.getvalue()
        plt.close(fig)
        st.caption("Numerical approximations depend on sampling and symbolic solver support. Curves are not connected through non-finite samples, but some complicated discontinuities may require closer inspection.")

with tabs[2]:
    st.subheader("Download current results")
    export_df = st.session_state.get("working_data", pd.DataFrame()).copy()
    current_batch = st.session_state.get("batch_result", pd.DataFrame())
    model_is_ready = "fit_metrics" in locals() and bool(fit_metrics) and fit_expr is not None
    metadata = f"Country: {country_name}; indicator: {indicator_name} ({indicator_code}); units: {units}; retrieval date: {retrieved_date}; edited scenario: {st.session_state.get('edited_flag', False)}"
    equation = derivative_text(fit_expr) if model_is_ready else "No valid historical polynomial fit"
    metrics_text = (f"R² (in-sample)={fmt_num(fit_metrics['r2'])}; MAE (in-sample)={fmt_num(fit_metrics['mae'])}; n={fit_metrics['n']}; range={coverage}" if model_is_ready else "Not available")
    findings = "Polynomial fits describe associations in the selected data; they do not establish causes. Extrapolations can be unstable. Missing observations are not invented."
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download working data CSV", csv_bytes, file_name="working_data.csv", mime="text/csv")
    xlsx_buf = io.BytesIO()
    with pd.ExcelWriter(xlsx_buf, engine="xlsxwriter") as writer:
        pd.DataFrame([{"country": country_name, "indicator": indicator_name, "indicator_code": indicator_code, "units": units,
                       "retrieved_date": retrieved_date, "edited_scenario": st.session_state.get("edited_flag", False),
                       "coverage": coverage, "equation": equation, "metrics": metrics_text}]).to_excel(writer, sheet_name="Summary", index=False)
        export_df.to_excel(writer, sheet_name="Working Data", index=False)
        current_batch.to_excel(writer, sheet_name="Calculated Years", index=False)
    st.download_button("Download Excel workbook (.xlsx)", xlsx_buf.getvalue(), file_name="hispanic_heritage_explorer.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    chart_png = st.session_state.get("current_chart_png") or st.session_state.get("function_chart_png")
    pdf_bytes = pdf_export("Hispanic Heritage Historical Data Explorer", metadata, equation, metrics_text, findings,
                           export_df, current_batch, chart_png)
    st.download_button("Download PDF report", pdf_bytes, file_name="hispanic_heritage_report.pdf", mime="application/pdf")
    st.markdown("### Verification checklist")
    checks = [
        "Source: compare indicator code, country, year, and values with the World Bank API link above.",
        "Editing/reset: change a working-table value, confirm the edited-scenario warning, then reset to source.",
        "Validation: try duplicate years, decimal years, text values, and insufficient distinct years.",
        "Model: compare displayed equation, R², MAE, residuals, and historical range.",
        "Single/batch: calculate the same year in both tools and compare estimates.",
        "Inputs/API: test malformed custom expressions, reversed intervals, empty API results, and offline/network failure.",
        "Controls: change country, indicator, degree, sample increment, horizon, comparison country, course, and printer mode.",
        "Exports: open CSV, inspect all Excel sheets, and open the PDF to verify its chart and tables.",
        "Deployment: verify app startup and logs on Streamlit Community Cloud or Render."
    ]
    for item in checks:
        st.markdown(f"- [ ] {item}")
    st.markdown("### Important limitations")
    st.write("World Bank coverage varies by country and indicator. A degree-d polynomial requires at least d+1 distinct years and can overfit. R² and MAE are in-sample unless explicitly labeled holdout. Polynomial extrapolation is not a validated forecast interval. Custom-function critical points and intervals may be approximate or incomplete for complicated expressions. Context examples are educational, not medical, financial, or engineering advice.")
