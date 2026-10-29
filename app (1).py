"""
Hispanic Heritage Historical Data Explorer
A classroom Streamlit app for World Bank data, polynomial regression, and function analysis.
Run: python -m streamlit run app.py
"""
from __future__ import annotations

from io import BytesIO
from datetime import date, datetime
import math
import re
import warnings

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import sympy as sp
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
import xlsxwriter

st.set_page_config(page_title="Hispanic Heritage Data Lab", page_icon="📊", layout="wide")

COUNTRIES = {
    "Mexico (MX)": "MEX", "Brazil (BR)": "BRA", "Argentina (AR)": "ARG",
    "Colombia (CO)": "COL", "Chile (CL)": "CHL", "Peru (PE)": "PER",
    "Venezuela (VE)": "VEN", "Ecuador (EC)": "ECU", "Uruguay (UY)": "URY",
}
INDICATORS = {
    "Life expectancy at birth": ("SP.DYN.LE00.IN", "years", "Average number of years a newborn is expected to live under current mortality patterns.", None),
    "Unemployment, national estimate": ("SL.UEM.TOTL.NE.ZS", "% of labor force", "Share of the labor force that is unemployed, based on national estimates.", (0, 100)),
    "GDP per capita": ("NY.GDP.PCAP.CD", "current US dollars per person", "Gross domestic product divided by midyear population. It is not average wealth or household income.", (0, None)),
    "Crude birth rate": ("SP.DYN.CBRT.IN", "births per 1,000 people per year", "Annual live births per 1,000 people, not a total birth count.", (0, None)),
    "Net migration": ("SM.POP.NETM", "people over the indicator's period", "Net number of migrants: immigration minus emigration. It is not an emigration count.", None),
    "Intentional homicide rate": ("VC.IHR.PSRC.P5", "intentional homicides per 100,000 people", "Intentional homicides per 100,000 population.", (0, None)),
    "Consumer price inflation": ("FP.CPI.TOTL.ZG", "% annual change", "Annual percentage change in consumer prices.", None),
    "Central government debt": ("GC.DOD.TOTL.GD.ZS", "% of GDP", "Central government debt as a percentage of GDP.", None),
    "Total reserves including gold": ("FI.RES.TOTL.CD", "current US dollars", "Total reserves including gold, valued in current US dollars.", (0, None)),
    "External balance on goods and services": ("NE.RSB.GNFS.ZS", "% of GDP", "External balance on goods and services as a share of GDP.", None),
}
WB_ROOT = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"
CONTEXTS = {
    "Business": ("Revenue or sales over time", "time", "year", "revenue", "dollars", "Polynomial: a*x^2 + b*x + c", "a*x**2 + b*x + c"),
    "Engineering": ("Position of a moving object", "time", "seconds", "position", "meters", "Quadratic position model", "a*x**2 + b*x + c"),
    "Medical": ("Illustrative concentration over time (not medical advice)", "time", "hours", "concentration", "mg/L", "Exponential decay", "a*exp(-b*x)"),
    "Sports science": ("Position or performance measure over time", "time", "seconds", "position", "meters", "Quadratic motion model", "a*x**2 + b*x + c"),
    "Biology": ("Population or cell colony size", "time", "days", "population", "cells", "Exponential growth", "a*exp(b*x)"),
    "Bioengineering": ("Illustrative system response", "time", "seconds", "response", "units", "Damped-style sinusoid (simplified)", "a*sin(b*x) + c"),
    "Economic forecasting": ("Economic indicator over time", "time", "years", "indicator", "indicator units", "Quadratic trend", "a*x**2 + b*x + c"),
    "Weather": ("Temperature over time", "time", "hours", "temperature", "°C", "Sinusoidal daily pattern", "a*sin(b*x) + c"),
    "Custom": ("Describe your own model and context.", "x", "units", "f(x)", "units", "Polynomial", "a*x**2 + b*x + c"),
}
VISUALS = ["Walking person with prosthetic leg (illustrative)", "Moving object / projectile", "Tank / concentration indicator",
           "Population / cell colony", "Thermometer / weather indicator", "Business / economic indicator"]

def wb_fetch(country: str, indicator: str, start_year: int, end_year: int) -> tuple[pd.DataFrame, str | None]:
    """Fetch annual observations from the official World Bank API."""
    url = WB_ROOT.format(country=country, indicator=indicator)
    params = {"format": "json", "date": f"{start_year}:{end_year}", "per_page": 20000}
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or len(payload) < 2 or not payload[1]:
            return pd.DataFrame(columns=["year", "value"]), "The API returned no observations for this selection."
        rows = [{"year": int(item["date"]), "value": item.get("value")}
                for item in payload[1] if item.get("date") and item.get("value") is not None]
        frame = pd.DataFrame(rows, columns=["year", "value"])
        if frame.empty:
            return frame, "The API response contained no non-null observations for this period."
        frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        frame = frame.dropna().astype({"year": int}).sort_values("year").reset_index(drop=True)
        return frame, None
    except requests.Timeout:
        return pd.DataFrame(columns=["year", "value"]), "Request timed out. Try again later."
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        return pd.DataFrame(columns=["year", "value"]), f"Could not retrieve data: {exc}"

def validate_table(frame: pd.DataFrame, bounds) -> tuple[pd.DataFrame, list[str]]:
    issues = []
    if not {"year", "value"}.issubset(frame.columns):
        return pd.DataFrame(columns=["year", "value"]), ["Table must contain year and value columns."]
    data = frame[["year", "value"]].copy()
    year_numeric = pd.to_numeric(data["year"], errors="coerce")
    value_numeric = pd.to_numeric(data["value"], errors="coerce")
    if year_numeric.isna().any() or ((year_numeric.dropna() % 1) != 0).any():
        issues.append("Every year must be an integer.")
    if value_numeric.isna().any():
        issues.append("Every value must be numeric; blank or invalid values are not allowed.")
    data["year"] = year_numeric
    data["value"] = value_numeric
    data = data.dropna()
    if data["year"].duplicated().any():
        issues.append("Years must be unique. Remove duplicate years before fitting.")
    if bounds:
        lo, hi = bounds
        if lo is not None and (data["value"] < lo).any():
            issues.append(f"Values for this indicator cannot be below {lo}.")
        if hi is not None and (data["value"] > hi).any():
            issues.append(f"Values for this indicator cannot be above {hi}.")
    if data.empty:
        issues.append("At least one valid row is required.")
    return data.astype({"year": int}).sort_values("year").reset_index(drop=True), issues

def fit_polynomial(data: pd.DataFrame, degree: int):
    if data["year"].nunique() < degree + 1:
        raise ValueError(f"Degree {degree} requires at least {degree + 1} distinct valid years; only {data['year'].nunique()} are available.")
    base_year = int(data["year"].min())
    t = (data["year"].to_numpy(dtype=float) - base_year).reshape(-1, 1)
    y = data["value"].to_numpy(dtype=float)
    model = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), LinearRegression())
    model.fit(t, y)
    pred = model.predict(t)
    r2 = float(r2_score(y, pred)) if len(y) > 1 else float("nan")
    mae = float(mean_absolute_error(y, pred))
    # PolynomialFeatures + LinearRegression; produce readable polynomial coefficients in t.
    lr = model.named_steps["linearregression"]
    powers = model.named_steps["polynomialfeatures"].powers_.flatten()
    coef = {int(p): float(c) for p, c in zip(powers, lr.coef_)}
    intercept = float(lr.intercept_)
    return model, base_year, pred, r2, mae, intercept, coef

def model_predict(model, base_year, years):
    arr = np.asarray(years, dtype=float)
    return model.predict((arr - base_year).reshape(-1, 1))

def fmt_num(v, digits=6):
    if pd.isna(v): return "not available"
    return f"{float(v):,.{digits}g}"

def equation_text(intercept, coef, degree, base_year, y_unit):
    terms = [f"{intercept:.8g}"]
    for power in range(1, degree + 1):
        c = coef.get(power, 0.0)
        terms.append(f"{c:+.8g}·t" + (f"^{power}" if power > 1 else ""))
    return f"y = {' '.join(terms)}, where t = year − {base_year} ({y_unit})"

def safe_symbolic_expr(source: str, x):
    """Parse common math notation while rejecting arbitrary Python."""
    text = source.strip().replace("^", "**").replace("π", "pi")
    allowed_names = {"x": x, "a": sp.Symbol("a", real=True), "b": sp.Symbol("b", real=True),
                     "c": sp.Symbol("c", real=True), "pi": sp.pi, "e": sp.E}
    funcs = {"sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp, "log": sp.log,
             "ln": sp.log, "sqrt": sp.sqrt, "abs": sp.Abs}
    try:
        expr = sp.sympify(text, locals={**allowed_names, **funcs}, evaluate=True)
        if expr.free_symbols - set(allowed_names.values()):
            raise ValueError("Use x and the parameters a, b, c only.")
        return expr
    except Exception as exc:
        raise ValueError("Could not parse the function. Use x, a, b, c and operators + - * / **, with functions sin, cos, exp, log, sqrt.") from exc

def numerical_function(expr, x_symbol, substitutions):
    e = sp.simplify(expr.subs(substitutions))
    f = sp.lambdify(x_symbol, e, modules=["numpy"])
    def wrapped(values):
        arr = np.asarray(values, dtype=float)
        with np.errstate(all="ignore"):
            result = f(arr)
        result = np.asarray(result, dtype=float)
        if result.ndim == 0:
            result = np.full(arr.shape, float(result))
        return result
    return e, wrapped

def make_chart(data, model, base_year, degree, sample_step, horizon, single_result, batch_result,
               custom_expr=None, x_symbol=None, substitutions=None, show_f=True, show_d1=False, show_d2=False,
               x_min=None, x_max=None, tangent=None, secant=None):
    fig, ax = plt.subplots(figsize=(10, 5.3))
    if custom_expr is not None:
        f = numerical_function(custom_expr, x_symbol, substitutions)[1]
        lo = float(x_min); hi = float(x_max)
        xs = np.linspace(lo, hi, 700)
        ys = f(xs)
        ys = np.where(np.isfinite(ys), ys, np.nan)
        if show_f: ax.plot(xs, ys, label="f(x)")
        if show_d1 or show_d2:
            d1 = sp.diff(custom_expr, x_symbol)
            d2 = sp.diff(d1, x_symbol)
            if show_d1:
                ax.plot(xs, numerical_function(d1, x_symbol, substitutions)[1](xs), label="f′(x)")
            if show_d2:
                ax.plot(xs, numerical_function(d2, x_symbol, substitutions)[1](xs), label="f″(x)")
        if tangent is not None:
            x0, y0, slope = tangent
            ax.plot([x0-(hi-lo)*.12, x0+(hi-lo)*.12], [y0-slope*(hi-lo)*.12, y0+slope*(hi-lo)*.12], "--", label="Tangent line")
            ax.scatter([x0], [y0], zorder=5, label="Current point")
        if secant is not None:
            x1, y1, x2, y2 = secant
            ax.plot([x1, x2], [y1, y2], "--", label="Secant line")
        ax.set_xlabel("x"); ax.set_ylabel("f(x)")
    else:
        hist_min, hist_max = int(data.year.min()), int(data.year.max())
        xs = np.linspace(hist_min, hist_max + horizon, max(400, (hist_max + horizon-hist_min)*4+1))
        ys = model_predict(model, base_year, xs)
        ax.plot(xs[xs <= hist_max], ys[xs <= hist_max], label="Polynomial fit (historical range)")
        if horizon > 0:
            ax.plot(xs[xs > hist_max], ys[xs > hist_max], "--", label="Extrapolated segment")
        sample_years = np.arange(hist_min, hist_max + 1, sample_step)
        ax.scatter(sample_years, model_predict(model, base_year, sample_years), s=20, marker="x", label=f"Model markers every {sample_step} year(s)")
        ax.scatter(data.year, data.value, s=30, label="Observed / working data", zorder=3)
        ax.axvline(hist_max, linestyle=":", label="Historical boundary")
        if single_result is not None:
            ax.scatter([single_result[0]], [single_result[1]], marker="D", s=65, label="Single-year estimate", zorder=5)
        if batch_result is not None and not batch_result.empty:
            ax.scatter(batch_result["year"], batch_result["estimate"], marker="^", s=40, label="Batch estimates", zorder=5)
        ax.set_xlabel("Year"); ax.set_ylabel(y_unit_label if (y_unit_label := st.session_state.get("unit_label", "Indicator value")) else "Indicator value")
    ax.grid(True, alpha=.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig

def build_excel(summary: dict, data: pd.DataFrame, results: pd.DataFrame) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        pd.DataFrame([summary]).to_excel(writer, sheet_name="Summary", index=False)
        data.to_excel(writer, sheet_name="Working data", index=False)
        results.to_excel(writer, sheet_name="Calculated years", index=False)
        for ws in writer.sheets.values():
            ws.freeze_panes(1, 0)
            ws.set_column(0, 10, 22)
    return buf.getvalue()

def build_pdf(summary: dict, data: pd.DataFrame, results: pd.DataFrame, chart_bytes: bytes | None) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(letter), rightMargin=28, leftMargin=28, topMargin=28, bottomMargin=28)
    styles = getSampleStyleSheet()
    story = [Paragraph("Hispanic Heritage Historical Data Explorer", styles["Title"]), Spacer(1, 8)]
    for key, value in summary.items():
        story.append(Paragraph(f"<b>{key}:</b> {str(value).replace('&','&amp;').replace('<','&lt;')}", styles["BodyText"]))
    story.append(Spacer(1, 8))
    if chart_bytes:
        story.append(RLImage(BytesIO(chart_bytes), width=650, height=330))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Working data preview", styles["Heading2"]))
    preview = data.head(18).copy()
    table_data = [list(preview.columns)] + [[str(v) for v in row] for row in preview.itertuples(index=False, name=None)]
    if table_data:
        tbl = Table(table_data, repeatRows=1)
        tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#dbeafe")),
                                 ("GRID",(0,0),(-1,-1),.3,colors.grey),("FONTSIZE",(0,0),(-1,-1),7)]))
        story.append(tbl)
    if not results.empty:
        story += [Spacer(1, 8), Paragraph("Calculated-year results", styles["Heading2"])]
        rt = [list(results.columns)] + [[str(v) for v in row] for row in results.head(40).itertuples(index=False, name=None)]
        tbl2 = Table(rt, repeatRows=1)
        tbl2.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#dbeafe")),
                                  ("GRID",(0,0),(-1,-1),.3,colors.grey),("FONTSIZE",(0,0),(-1,-1),7)]))
        story.append(tbl2)
    doc.build(story)
    return buf.getvalue()

st.title("Hispanic Heritage Historical Data Explorer")
st.caption("HISPANIC HERITAGE DATA LAB · Historical evidence, mathematical models, and interactive function analysis")
st.info("Brazil is included as a Latin American comparison country. The API may have gaps; the app reports the observations actually returned and never fills gaps with invented data.")

with st.sidebar:
    st.header("Data and model")
    country_label = st.selectbox("Featured country", list(COUNTRIES))
    indicator_label = st.selectbox("World Bank indicator", list(INDICATORS))
    degree = st.slider("Polynomial degree", 3, 8, 3)
    sample_step = st.slider("Graph sample increment (years)", 1, 10, 5)
    horizon = st.slider("Forward display horizon (years)", 0, 50, 10)
    comparison_labels = st.multiselect("Compare same indicator with", [c for c in COUNTRIES if c != country_label], default=[])
    printer_mode = st.checkbox("Printer-friendly report view", value=False)
    course = st.selectbox("Mathematics pathway", ["Honors Algebra 2", "Precalculus", "Calculus"], index=2)

country_code = COUNTRIES[country_label]
indicator_code, unit_label, definition, bounds = INDICATORS[indicator_label]
st.session_state["unit_label"] = unit_label
current_year = date.today().year
start_year, end_year = current_year - 70, current_year - 1
st.caption(f"Requested period: {start_year}–{end_year} inclusive · Retrieval date: {date.today().isoformat()} · Source: World Bank API")
st.markdown(f"**Definition:** {definition}  \n**Units:** {unit_label}  \n**Indicator code:** `{indicator_code}`  \n**Source:** [World Bank indicator page](https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}?format=json)")

cache_key = f"source::{country_code}::{indicator_code}::{start_year}::{end_year}"
if cache_key not in st.session_state:
    source_data, source_error = wb_fetch(country_code, indicator_code, start_year, end_year)
    st.session_state[cache_key] = (source_data, source_error)
source_data, source_error = st.session_state[cache_key]
if source_error:
    st.warning(source_error)
if source_data.empty:
    st.error("No usable source observations are available for this selection. Try another country/indicator or reload later.")
    st.stop()

edit_key = f"edit::{cache_key}"
if edit_key not in st.session_state:
    st.session_state[edit_key] = source_data.copy()
st.subheader("Historical data")
st.write(f"Actual source coverage: **{int(source_data.year.min())}–{int(source_data.year.max())}**, **{len(source_data)} non-null observations** (not necessarily 70).")
edited_input = st.data_editor(st.session_state[edit_key], num_rows="dynamic", use_container_width=True, key=f"editor::{cache_key}")
col_reset, col_status = st.columns([1, 3])
with col_reset:
    if st.button("Reset to source data"):
        st.session_state[edit_key] = source_data.copy()
        st.rerun()
with col_status:
    edited = not edited_input.reset_index(drop=True).equals(source_data.reset_index(drop=True))
    st.write("**Status:** User-modified scenario" if edited else "**Status:** Source data (unmodified)")
working_data, validation_issues = validate_table(edited_input, bounds)
for issue in validation_issues:
    st.error(issue)
if validation_issues:
    st.stop()
if len(working_data) < degree + 1:
    st.error(f"Degree {degree} requires at least {degree + 1} distinct valid years; currently {len(working_data)} rows are valid.")
    st.stop()

try:
    model, base_year, fitted, r2, mae, intercept, coef = fit_polynomial(working_data, degree)
except Exception as exc:
    st.error(str(exc)); st.stop()
working_data = working_data.copy()
working_data["fitted_value"] = fitted
working_data["residual"] = working_data["value"] - fitted
eq = equation_text(intercept, coef, degree, base_year, unit_label)
hist_min, hist_max = int(working_data.year.min()), int(working_data.year.max())
summary = {
    "Country": country_label, "Country code": country_code, "Indicator": indicator_label,
    "Indicator code": indicator_code, "Units": unit_label, "Definition": definition,
    "Retrieved": date.today().isoformat(), "Requested years": f"{start_year}–{end_year}",
    "Actual coverage": f"{hist_min}–{hist_max}", "Observation count": len(working_data),
    "Data status": "User-modified scenario" if edited else "Source data",
    "Degree": degree, "Base year": base_year, "Equation": eq,
    "In-sample R-squared": f"{r2:.6g}", "In-sample MAE": f"{mae:.6g}",
    "Mean observed value": f"{working_data.value.mean():.6g}",
    "Limitations": "Polynomial extrapolation is uncertain; MAE describes in-sample error and is not a prediction interval."
}
st.subheader("Polynomial model")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Valid observations", len(working_data))
m2.metric("In-sample R²", fmt_num(r2))
m3.metric("In-sample MAE", f"{fmt_num(mae)} {unit_label}")
m4.metric("Observed mean", f"{fmt_num(working_data.value.mean())} {unit_label}")
st.code(eq)
st.caption(f"Centered time: t = year − {base_year}. Fit metrics are in-sample and should not be interpreted as proof of forecasting accuracy.")
with st.expander("Residuals and chronological holdout"):
    st.dataframe(working_data[["year", "value", "fitted_value", "residual"]], use_container_width=True)
    if len(working_data) >= max(degree + 4, 12):
        split = max(degree + 1, int(len(working_data) * .8))
        train, test = working_data.iloc[:split], working_data.iloc[split:]
        if len(test) >= 2:
            try:
                hold_model, hold_base, *_ = fit_polynomial(train[["year", "value"]], degree)
                hold_pred = model_predict(hold_model, hold_base, test.year.to_numpy())
                st.write(f"Chronological holdout MAE (last {len(test)} observations, trained only on earlier rows): **{mean_absolute_error(test.value, hold_pred):.6g}**. This is separate from in-sample MAE.")
            except Exception as exc:
                st.write(f"Holdout score unavailable: {exc}")
    else:
        st.write("Not enough observations for the app's simple chronological holdout display.")

st.subheader("Calculate values")
c1, c2 = st.columns(2)
with c1:
    calc_year = st.number_input("Single year (1950–2100)", min_value=1950, max_value=2100, value=min(max(hist_max + 1, 1950), 2100), step=1)
    calc_clicked = st.button("Calculate Value", type="primary")
with c2:
    batch_mode = st.radio("Batch input", ["Year range", "Custom years"], horizontal=True)
    if batch_mode == "Year range":
        bcol1, bcol2, bcol3 = st.columns(3)
        with bcol1: batch_start = st.number_input("Start year", min_value=1950, max_value=2100, value=max(1950, hist_min), key="batch_start")
        with bcol2: batch_end = st.number_input("End year", min_value=1950, max_value=2100, value=min(2100, hist_max + 5), key="batch_end")
        with bcol3: batch_step = st.selectbox("Range step", list(range(1, 11)), index=0)
        custom_year_text = ""
    else:
        custom_year_text = st.text_input("Comma-separated years", value=f"{hist_max + 1}, {hist_max + 5}")
        batch_start, batch_end, batch_step = 1950, 1950, 1
    batch_clicked = st.button("Calculate batch")

# Store results with a signature so stale values cannot survive changes to data/model.
data_signature = (country_code, indicator_code, degree, tuple(working_data.year), tuple(np.round(working_data.value, 10)))
if st.session_state.get("result_signature") != data_signature:
    st.session_state["single_result"] = None
    st.session_state["batch_result"] = pd.DataFrame(columns=["year", "estimate", "classification"])
    st.session_state["result_signature"] = data_signature

if calc_clicked:
    yr = int(calc_year)
    est = float(model_predict(model, base_year, [yr])[0])
    actual_row = working_data.loc[working_data.year == yr, "value"]
    classification = "within fitted range" if hist_min <= yr <= hist_max else "outside fitted range (extrapolation)"
    st.session_state["single_result"] = (yr, est, classification, None if actual_row.empty else float(actual_row.iloc[0]))
if batch_clicked:
    try:
        if batch_mode == "Year range":
            if batch_end < batch_start: raise ValueError("End year must be greater than or equal to start year.")
            years = list(range(int(batch_start), int(batch_end) + 1, int(batch_step)))
        else:
            parts = [p.strip() for p in custom_year_text.split(",") if p.strip()]
            if not parts: raise ValueError("Enter at least one year.")
            if any(not re.fullmatch(r"\d{4}", p) for p in parts): raise ValueError("Each custom year must be a four-digit integer.")
            years = [int(p) for p in parts]
        invalid = [y for y in years if y < 1950 or y > 2100]
        if invalid: raise ValueError(f"Years must be between 1950 and 2100. Invalid: {invalid}")
        years = list(dict.fromkeys(years))
        estimates = model_predict(model, base_year, years)
        st.session_state["batch_result"] = pd.DataFrame({
            "year": years, "estimate": estimates,
            "classification": ["within fitted range" if hist_min <= y <= hist_max else "outside fitted range (extrapolation)" for y in years]
        }).sort_values("year").reset_index(drop=True)
    except Exception as exc:
        st.error(f"Batch input error: {exc}")

single_result = st.session_state.get("single_result")
batch_result = st.session_state.get("batch_result", pd.DataFrame(columns=["year", "estimate", "classification"]))
if single_result:
    yr, est, cls, actual = single_result
    st.success(f"{yr}: estimated {fmt_num(est)} {unit_label} — {cls}.")
    if actual is not None: st.write(f"Actual observed value for {yr}: **{fmt_num(actual)} {unit_label}**. The fitted estimate is not the same as the observation.")
    if not np.isfinite(est) or abs(est) > max(1e6, abs(working_data.value).max() * 1000):
        st.warning("This estimate is non-finite or far outside the scale of observed values. Treat it as an implausible polynomial extrapolation.")
if not batch_result.empty:
    st.write("Batch results")
    st.dataframe(batch_result, use_container_width=True)
    if len(batch_result):
        st.caption("Batch estimates use the same model transformation as the single-year calculation.")

st.subheader("Historical graph")
fig, ax = plt.subplots(figsize=(10, 5))
curve_years = np.linspace(hist_min, hist_max + horizon, max(400, (hist_max + horizon-hist_min)*4+1))
curve_vals = model_predict(model, base_year, curve_years)
ax.plot(curve_years[curve_years <= hist_max], curve_vals[curve_years <= hist_max], label="Polynomial fit (historical range)")
if horizon > 0:
    ax.plot(curve_years[curve_years > hist_max], curve_vals[curve_years > hist_max], "--", label="Extrapolated segment")
sample_years = np.arange(hist_min, hist_max + 1, sample_step)
ax.scatter(sample_years, model_predict(model, base_year, sample_years), marker="x", s=22, label=f"Model markers every {sample_step} year(s)")
ax.scatter(working_data.year, working_data.value, s=28, label="Observed / working data", zorder=3)
ax.axvline(hist_max, linestyle=":", label="Historical boundary")
if single_result:
    ax.scatter([single_result[0]], [single_result[1]], marker="D", s=65, label="Single-year estimate", zorder=5)
if not batch_result.empty:
    ax.scatter(batch_result.year, batch_result.estimate, marker="^", s=40, label="Batch estimates", zorder=5)
ax.set_xlabel("Year"); ax.set_ylabel(unit_label); ax.grid(True, alpha=.25); ax.legend(fontsize=8); fig.tight_layout()
st.pyplot(fig, use_container_width=True)

if comparison_labels:
    st.subheader("Same-indicator country comparison")
    compare_fig, compare_ax = plt.subplots(figsize=(10, 4.8))
    compare_ax.plot(working_data.year, working_data.value, marker="o", markersize=3, label=country_label)
    for label in comparison_labels:
        code = COUNTRIES[label]
        comp_key = f"source::{code}::{indicator_code}::{start_year}::{end_year}"
        if comp_key not in st.session_state:
            st.session_state[comp_key] = wb_fetch(code, indicator_code, start_year, end_year)
        comp, err = st.session_state[comp_key]
        if err or comp.empty:
            st.warning(f"{label}: {err or 'no usable observations'}")
        else:
            compare_ax.plot(comp.year, comp.value, marker=".", markersize=3, label=label)
            st.caption(f"{label}: actual coverage {int(comp.year.min())}–{int(comp.year.max())}, {len(comp)} observations.")
    compare_ax.set_xlabel("Year"); compare_ax.set_ylabel(unit_label); compare_ax.grid(True, alpha=.25); compare_ax.legend(fontsize=8); compare_fig.tight_layout()
    st.pyplot(compare_fig, use_container_width=True)
    st.caption("Lines may cover different years because countries have different reporting histories and missing values; gaps are not filled.")

st.subheader("Function lab: contexts, parameters, and calculus")
st.write("This lab uses an independent illustrative function unless you choose the generated historical best-fit model. A context label does not prove that the model is scientifically valid.")
context = st.selectbox("Context", list(CONTEXTS))
default_desc, default_x, default_xunit, default_y, default_yunit, suggestion_label, suggestion_expr = CONTEXTS[context]
context_desc = st.text_input("Context description", default_desc)
vcol1, vcol2, vcol3, vcol4 = st.columns(4)
with vcol1: x_name = st.text_input("Independent variable", default_x)
with vcol2: x_unit = st.text_input("Independent unit", default_xunit)
with vcol3: y_name = st.text_input("Dependent variable", default_y)
with vcol4: y_unit = st.text_input("Dependent unit", default_yunit)
function_mode = st.radio("Function source", ["Generated historical best fit", "Suggested function", "Parameterized function", "My own function"], horizontal=True)
x_symbol = sp.Symbol("x", real=True)
a_val = st.number_input("Parameter a", value=1.0, step=0.5)
b_val = st.number_input("Parameter b", value=0.2, step=0.1)
c_val = st.number_input("Parameter c", value=0.0, step=0.5)
if function_mode == "Generated historical best fit":
    # Use centered polynomial in year-like x: y(x) = model(x) and derive from that same transformation.
    expr = sp.Float(intercept)
    for power in range(1, degree + 1):
        expr += sp.Float(coef.get(power, 0.0)) * (x_symbol - base_year) ** power
    substitutions = {}
    function_domain_default = (float(hist_min), float(hist_max + horizon))
elif function_mode == "Suggested function":
    expr = safe_symbolic_expr(suggestion_expr, x_symbol)
    substitutions = {sp.Symbol("a", real=True): a_val, sp.Symbol("b", real=True): b_val, sp.Symbol("c", real=True): c_val}
    function_domain_default = (0.0, 10.0)
elif function_mode == "Parameterized function":
    ftype = st.selectbox("Function family", ["Polynomial", "Rational", "Exponential", "Logarithmic", "Trigonometric"])
    family_map = {
        "Polynomial": "a*x**2 + b*x + c", "Rational": "(a*x+b)/(x-c)",
        "Exponential": "a*exp(b*x)+c", "Logarithmic": "a*log(x)+b",
        "Trigonometric": "a*sin(b*x)+c",
    }
    expr = safe_symbolic_expr(family_map[ftype], x_symbol)
    substitutions = {sp.Symbol("a", real=True): a_val, sp.Symbol("b", real=True): b_val, sp.Symbol("c", real=True): c_val}
    function_domain_default = (0.1, 10.0) if ftype == "Logarithmic" else (0.0, 10.0)
else:
    user_expr = st.text_input("Enter f(x)", value="x**3 - 3*x")
    try:
        expr = safe_symbolic_expr(user_expr, x_symbol)
        substitutions = {sp.Symbol("a", real=True): a_val, sp.Symbol("b", real=True): b_val, sp.Symbol("c", real=True): c_val}
        function_domain_default = (-5.0, 5.0)
    except ValueError as exc:
        st.error(str(exc)); st.stop()

xmin_default, xmax_default = function_domain_default
interval_col1, interval_col2 = st.columns(2)
with interval_col1: x_min = st.number_input("Analysis interval start", value=float(xmin_default))
with interval_col2: x_max = st.number_input("Analysis interval end", value=float(xmax_default))
if x_max <= x_min:
    st.error("Analysis interval end must be greater than its start."); st.stop()

try:
    expr_sub, f = numerical_function(expr, x_symbol, substitutions)
    d1_expr = sp.simplify(sp.diff(expr_sub, x_symbol))
    d2_expr = sp.simplify(sp.diff(d1_expr, x_symbol))
    st.markdown(f"**Context:** {context_desc}")
    st.latex("f(x) = " + sp.latex(expr_sub))
    st.latex("f'(x) = " + sp.latex(d1_expr))
    st.latex("f''(x) = " + sp.latex(d2_expr))
except Exception as exc:
    st.error(f"Function evaluation error: {exc}"); st.stop()

x_lo, x_hi = float(x_min), float(x_max)
x_grid = np.linspace(x_lo, x_hi, 1000)
f_grid = f(x_grid)
finite = np.isfinite(f_grid)
valid_vals = f_grid[finite]
if len(valid_vals) == 0:
    st.error("The function is undefined throughout this interval."); st.stop()

if course == "Calculus":
    st.markdown("#### Calculus analysis")
elif course == "Precalculus":
    st.markdown("#### Precalculus analysis")
else:
    st.markdown("#### Honors Algebra 2 analysis")
show_col1, show_col2, show_col3 = st.columns(3)
with show_col1: show_function = st.checkbox("Show f(x)", value=True)
with show_col2: show_first = st.checkbox("Show f′(x)", value=(course == "Calculus"))
with show_col3: show_second = st.checkbox("Show f″(x)", value=(course == "Calculus"))
point_default = min(max((x_lo + x_hi) / 2, x_lo), x_hi)
point_x = st.slider("Current input value", min_value=float(x_lo), max_value=float(x_hi), value=float(point_default))
left_col, right_col = st.columns(2)
with left_col:
    sec_a = st.slider("Average-rate interval start", min_value=float(x_lo), max_value=float(x_hi), value=float(x_lo))
with right_col:
    sec_b = st.slider("Average-rate interval end", min_value=float(x_lo), max_value=float(x_hi), value=float(x_hi))
if sec_a == sec_b:
    st.warning("Choose different endpoints to calculate an average rate of change.")
f0 = float(f(np.array([point_x]))[0])
d1f = sp.lambdify(x_symbol, d1_expr, modules=["numpy"])
d2f = sp.lambdify(x_symbol, d2_expr, modules=["numpy"])
try:
    slope0 = float(np.asarray(d1f(point_x)))
except Exception:
    slope0 = float("nan")
try:
    second0 = float(np.asarray(d2f(point_x)))
except Exception:
    second0 = float("nan")
if sec_a != sec_b:
    ya, yb = float(f(np.array([sec_a]))[0]), float(f(np.array([sec_b]))[0])
    avg_rate = (yb - ya) / (sec_b - sec_a)
else:
    ya = yb = avg_rate = float("nan")
metric_cols = st.columns(4)
metric_cols[0].metric(f"f({point_x:.4g})", fmt_num(f0), y_unit)
metric_cols[1].metric("Instantaneous rate f′(x)", fmt_num(slope0), f"{y_unit}/{x_unit}")
metric_cols[2].metric("Second derivative f″(x)", fmt_num(second0), f"{y_unit}/{x_unit}²")
metric_cols[3].metric("Average rate", fmt_num(avg_rate), f"{y_unit}/{x_unit}")
if context in ["Engineering", "Sports science"] and y_name.lower() in ["position", "distance"]:
    interpretation = f"At x={point_x:.4g} {x_unit}, the model gives position {fmt_num(f0)} {y_unit}; the first derivative is velocity ({fmt_num(slope0)} {y_unit}/{x_unit}) and the second derivative is acceleration ({fmt_num(second0)} {y_unit}/{x_unit}²)."
elif context in ["Business", "Economic forecasting"]:
    direction = "increasing" if slope0 > 0 else ("decreasing" if slope0 < 0 else "locally flat")
    change = "increasing" if second0 > 0 else ("decreasing" if second0 < 0 else "roughly unchanged")
    interpretation = f"At x={point_x:.4g} {x_unit}, {y_name} is {fmt_num(f0)} {y_unit}. Its rate is {fmt_num(slope0)} {y_unit}/{x_unit}, so it is locally {direction}; the rate itself is {change} (second derivative {fmt_num(second0)} {y_unit}/{x_unit}²)."
else:
    interpretation = f"At x={point_x:.4g} {x_unit}, the model value is {fmt_num(f0)} {y_unit}. The first derivative describes the local rate of change ({fmt_num(slope0)} {y_unit}/{x_unit}); the second derivative describes how that rate changes ({fmt_num(second0)} {y_unit}/{x_unit}²)."
st.write(interpretation)

# Numeric calculus analysis, explicitly labelled approximations.
if course == "Calculus":
    with st.expander("Critical points, extrema, concavity, and domain notes", expanded=True):
        crit_candidates = []
        inflect_candidates = []
        try:
            crit_expr = sp.together(d1_expr)
            crit_roots = sp.solve(sp.Eq(crit_expr, 0), x_symbol)
            for root in crit_roots:
                if root.is_real is True:
                    rv = float(root)
                    if x_lo <= rv <= x_hi: crit_candidates.append(rv)
        except Exception:
            pass
        try:
            roots2 = sp.solve(sp.Eq(sp.together(d2_expr), 0), x_symbol)
            for root in roots2:
                if root.is_real is True:
                    rv = float(root)
                    if x_lo <= rv <= x_hi: inflect_candidates.append(rv)
        except Exception:
            pass
        def sign_at(func, value):
            try:
                z = float(np.asarray(func(value)))
                return 1 if z > 1e-7 else (-1 if z < -1e-7 else 0)
            except Exception: return 0
        rows = []
        for root in sorted(set(round(v, 8) for v in crit_candidates)):
            delta = max((x_hi-x_lo)*1e-5, 1e-5)
            left_sign = sign_at(d1f, root-delta) if root-delta >= x_lo else 0
            right_sign = sign_at(d1f, root+delta) if root+delta <= x_hi else 0
            kind = "local minimum (f′: − to +)" if left_sign < 0 and right_sign > 0 else ("local maximum (f′: + to −)" if left_sign > 0 and right_sign < 0 else "stationary; no extremum confirmed")
            rows.append({"x (approx/exact root)": root, "f(x)": float(f(np.array([root]))[0]), "classification": kind})
        st.write("Critical points (symbolic roots with numerical sign-test classification):")
        st.dataframe(pd.DataFrame(rows) if rows else pd.DataFrame(columns=["x (approx/exact root)", "f(x)", "classification"]), use_container_width=True)
        # Interval absolute extrema include endpoints, if all evaluations are finite.
        endpoints = [x_lo, x_hi] + [r for r in crit_candidates if x_lo <= r <= x_hi]
        candidates = [(x, float(f(np.array([x]))[0])) for x in endpoints]
        candidates = [(x, y) for x, y in candidates if np.isfinite(y)]
        if candidates:
            mn, mx = min(candidates, key=lambda z:z[1]), max(candidates, key=lambda z:z[1])
            st.write(f"Absolute minimum candidate on the closed interval: x={mn[0]:.6g}, f(x)={mn[1]:.6g}. Absolute maximum candidate: x={mx[0]:.6g}, f(x)={mx[1]:.6g}. These are reliable only when the function is continuous on the interval and all relevant critical points were found.")
        inflect_rows = []
        for root in sorted(set(round(v, 8) for v in inflect_candidates)):
            delta = max((x_hi-x_lo)*1e-5, 1e-5)
            ls = sign_at(d2f, root-delta) if root-delta >= x_lo else 0
            rs = sign_at(d2f, root+delta) if root+delta <= x_hi else 0
            if ls * rs < 0:
                inflect_rows.append({"x": root, "f(x)": float(f(np.array([root]))[0]), "confirmation": "concavity changes"})
            else:
                inflect_rows.append({"x": root, "f(x)": float(f(np.array([root]))[0]), "confirmation": "not confirmed: no sign change detected"})
        st.write("Possible inflection points (requires a concavity change):")
        st.dataframe(pd.DataFrame(inflect_rows) if inflect_rows else pd.DataFrame(columns=["x", "f(x)", "confirmation"]), use_container_width=True)
        st.caption("Root finding can miss complicated or non-algebraic roots. Values are numerical approximations unless shown as symbolic expressions above. Domain/discontinuities should be checked from the expression; plotted undefined values are omitted.")
elif course == "Precalculus":
    st.write("Degree and coefficients affect end behavior, steepness, and turning points. A changing average rate can suggest concavity, but this numerical view does not prove a historical cause.")
else:
    st.write(f"Domain: the selected interval [{x_lo:g}, {x_hi:g}] where the function is defined. Range over the plotted interval is approximately [{np.nanmin(valid_vals):.6g}, {np.nanmax(valid_vals):.6g}]. End behavior depends on the function family and leading terms.")

# Function graph with NaNs at undefined points so lines break at discontinuities.
plot_fig, plot_ax = plt.subplots(figsize=(10, 5))
if show_function:
    plot_ax.plot(x_grid, np.where(np.isfinite(f_grid), f_grid, np.nan), label="f(x)")
if show_first:
    d1_vals = np.asarray(d1f(x_grid), dtype=float)
    plot_ax.plot(x_grid, np.where(np.isfinite(d1_vals), d1_vals, np.nan), label="f′(x)")
if show_second:
    d2_vals = np.asarray(d2f(x_grid), dtype=float)
    plot_ax.plot(x_grid, np.where(np.isfinite(d2_vals), d2_vals, np.nan), label="f″(x)")
if np.isfinite(f0):
    plot_ax.scatter([point_x], [f0], s=55, label="Current point", zorder=5)
    if np.isfinite(slope0):
        tx = np.linspace(max(x_lo, point_x-(x_hi-x_lo)*.12), min(x_hi, point_x+(x_hi-x_lo)*.12), 50)
        plot_ax.plot(tx, f0 + slope0*(tx-point_x), "--", label="Tangent line")
if sec_a != sec_b and np.isfinite(ya) and np.isfinite(yb) and np.isfinite(avg_rate):
    sx = np.linspace(min(sec_a, sec_b), max(sec_a, sec_b), 50)
    plot_ax.plot(sx, ya + avg_rate*(sx-sec_a), ":", label="Secant line")
plot_ax.set_xlabel(f"{x_name} ({x_unit})"); plot_ax.set_ylabel(f"{y_name} ({y_unit})")
plot_ax.grid(True, alpha=.25); plot_ax.legend(fontsize=8); plot_fig.tight_layout()
st.pyplot(plot_fig, use_container_width=True)

visual = st.selectbox("Contextual visual simulation", VISUALS)
play_col1, play_col2, play_col3 = st.columns(3)
with play_col1: play = st.checkbox("Play animation", value=False)
with play_col2: speed = st.select_slider("Playback speed", options=["Slow", "Normal", "Fast"], value="Normal")
with play_col3:
    if st.button("Reset simulation"):
        st.session_state["sim_x"] = float(x_lo)
        st.rerun()
if play:
    # Streamlit reruns on interaction; use a deterministic step so playback never requires a timer thread.
    step = {"Slow": .01, "Normal": .03, "Fast": .08}[speed] * (x_hi-x_lo)
    st.session_state["sim_x"] = min(x_hi, float(st.session_state.get("sim_x", point_x)) + step)
    st.caption("Playback advances one step per rerun/interaction; this is a lightweight Streamlit simulation, not a continuous real-time animation.")
sim_x = float(st.session_state.get("sim_x", point_x))
sim_x = min(max(sim_x, x_lo), x_hi)
sim_y = float(f(np.array([sim_x]))[0])
st.markdown(f"**{visual}** — simulation point: {sim_x:.3g} {x_unit}; model output: {fmt_num(sim_y)} {y_unit}.")
if "prosthetic" in visual.lower():
    st.write("Illustrative gait indicator: the moving point represents a changing modeled gait measure. It is not a medical device simulation and does not evaluate a real person's gait.")
elif "projectile" in visual.lower():
    st.write("Illustrative motion indicator: horizontal position is scaled to the analysis interval and vertical height follows the function; negative or nonphysical values remain visible rather than being clipped.")
elif "tank" in visual.lower():
    st.write("Illustrative concentration gauge: the fill indicator is normalized to the finite plotted output range. This visual scaling is for demonstration only.")
elif "population" in visual.lower():
    st.write("Illustrative colony indicator: the circle size reflects the normalized model output range, not a measured count unless the chosen units support that interpretation.")
elif "thermometer" in visual.lower():
    st.write("Illustrative thermometer: the marker is scaled to the plotted output range; this is not a weather forecast.")
else:
    st.write("Illustrative business/economic indicator: bar height follows the model output on a normalized display scale, not a claim about actual future performance.")
lo_v, hi_v = float(np.nanmin(valid_vals)), float(np.nanmax(valid_vals))
normalized = 0.5 if hi_v == lo_v else min(1.0, max(0.0, (sim_y-lo_v)/(hi_v-lo_v))) if np.isfinite(sim_y) else 0.5
if "thermometer" in visual.lower():
    st.progress(normalized, text="Normalized output scale")
elif "tank" in visual.lower():
    st.progress(normalized, text="Illustrative fill / concentration scale")
elif "population" in visual.lower():
    st.progress(normalized, text="Normalized population / colony scale")
elif "business" in visual.lower():
    st.progress(normalized, text="Normalized indicator scale")
else:
    st.progress(normalized, text="Normalized simulation position / height")

st.subheader("Interpretation and limitations")
st.write(f"For the selected context, f(x) represents {y_name} in {y_unit}; f′(x) is its rate of change in {y_unit} per {x_unit}; f″(x) measures how that rate changes in {y_unit} per {x_unit} squared. The function is illustrative unless its assumptions are supported by domain evidence.")
st.warning("Polynomial extrapolations can become implausible outside observed years. In-sample MAE and R² are not validated prediction intervals. The app does not silently clip values or invent missing data.")
st.caption("World Bank data are subject to revisions, reporting gaps, and country-specific coverage. A fitted curve describes a mathematical pattern, not a proven explanation for why a historical event occurred.")

# Downloads reflect current selection, data status, model, and calculated results.
results_for_export = batch_result.copy()
if single_result:
    results_for_export = pd.concat([results_for_export, pd.DataFrame([{
        "year": single_result[0], "estimate": single_result[1], "classification": single_result[2],
        "actual_observation": single_result[3]
    }])], ignore_index=True)
csv_bytes = working_data[["year", "value", "fitted_value", "residual"]].to_csv(index=False).encode("utf-8")
xlsx_bytes = build_excel(summary, working_data[["year", "value", "fitted_value", "residual"]], results_for_export)
pdf_chart_buf = BytesIO()
fig.savefig(pdf_chart_buf, format="png", dpi=130, bbox_inches="tight")
pdf_bytes = build_pdf(summary, working_data[["year", "value", "fitted_value", "residual"]], results_for_export, pdf_chart_buf.getvalue())
st.subheader("Export current report")
d1, d2, d3 = st.columns(3)
with d1: st.download_button("Download CSV", csv_bytes, file_name="historical_data.csv", mime="text/csv")
with d2: st.download_button("Download Excel workbook", xlsx_bytes, file_name="hispanic_heritage_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
with d3: st.download_button("Download PDF report", pdf_bytes, file_name="hispanic_heritage_report.pdf", mime="application/pdf")

if printer_mode:
    st.markdown("---")
    st.header("Printer-friendly summary")
    st.write(f"**{country_label} · {indicator_label}**")
    st.write(f"Coverage: {hist_min}–{hist_max}; n={len(working_data)}; data status: {'user-modified scenario' if edited else 'source data'}.")
    st.write(f"Model: {eq}")
    st.write(f"In-sample R²={r2:.5g}; in-sample MAE={mae:.5g} {unit_label}.")
    st.write("Interpretation: the curve summarizes observed values; extrapolations are uncertain and should not be treated as validated forecasts.")
    if single_result: st.write(f"Calculated year {single_result[0]}: {single_result[1]:.6g} {unit_label} ({single_result[2]}).")
    st.dataframe(working_data[["year", "value", "fitted_value", "residual"]].tail(15), use_container_width=True)

with st.expander("Deployment and verification checklist"):
    st.markdown("""
**Run locally**
```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

**Streamlit Community Cloud**
1. Put `app.py` and `requirements.txt` in the root of a GitHub repository.
2. Push/commit the files to GitHub.
3. In Streamlit Community Cloud, create an app, select the repository/branch, and set the main file path to `app.py`.

**Render**
- Create a **Python Web Service** connected to the GitHub repository.
- Build command: `pip install -r requirements.txt`
- Start command: `python -m streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT`

**Verification**
- Confirm API links and codes match the chosen indicator; compare a few values against the World Bank indicator page.
- Edit a value, confirm the status changes, then press **Reset to source data**.
- Try duplicate years, non-integer years, blank/nonnumeric values, and values outside indicator-specific bounds.
- Select a degree with too few rows; the app should explain the minimum distinct-year requirement.
- Confirm the equation uses centered time, and compare in-sample R²/MAE with residuals.
- Calculate the same year in single-year and batch modes; estimates should match.
- Test reversed batch ranges, invalid custom years, API timeout/empty data, and countries with missing coverage.
- Change each sidebar control and verify the relevant graph/model/comparison changes.
- Open all three downloads and verify the CSV is comma-separated text, Excel is a real workbook, and PDF opens as a report.
- After deployment, inspect Streamlit/Render logs for dependency errors, API failures, or port configuration issues.

**Common errors**
- `ModuleNotFoundError`: check that the dependency is in `requirements.txt` and redeploy.
- Missing requirements file: ensure the filename is exactly `requirements.txt` at repository root.
- API errors: check network access and the World Bank API status; the app reports failures instead of generating substitute data.
- Render port errors: use the supplied start command with `$PORT`.
""")
