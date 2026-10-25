"""
Collections & Delinquency Portfolio Dashboard
==============================================
Interactive Streamlit + Plotly dashboard for `Collections_data.csv`
(a loan collections / delinquency portfolio snapshot).

Run with:  streamlit run dashboard.py

Design goals: unified theme, account-weighted accuracy, story-driven tabs,
a metric selector, and rule-based auto-insight commentary.
"""

import os
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Collections Portfolio",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Theme, palette & global CSS
# ---------------------------------------------------------------------------
THEME = "plotly_white"
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
           "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

st.markdown(
    """
    <style>
      div.block-container{padding-top:1.2rem; padding-bottom:2rem;}
      .kpi-card{
        background:linear-gradient(135deg,#f8fafc 0%,#eef2f7 100%);
        border:1px solid #e2e8f0; border-radius:12px;
        padding:12px 14px; height:100%;
      }
      .kpi-label{font-size:.78rem;color:#64748b;font-weight:600;
                 letter-spacing:.03em;text-transform:uppercase;}
      .kpi-value{font-size:1.35rem;color:#0f172a;font-weight:700;line-height:1.25;}
      .kpi-sub{font-size:.72rem;color:#94a3b8;}
      .section-note{color:#475569;font-size:.9rem;margin:-.4rem 0 .6rem 0;}
      .exec-box{
        background:#0f172a;color:#f8fafc;border-radius:12px;
        padding:16px 18px;margin-bottom:.5rem;
      }
      .exec-box h4{margin:0 0 .4rem 0;color:#38bdf8;}
      .exec-box ul{margin:.2rem 0 0 1rem;padding:0;}
      .exec-box li{margin:.15rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(DATA_DIR, "Collections_data.csv")
# Parquet cache next to the source CSV: ~10x smaller & loads in a few seconds.
PARQUET_FILE = os.path.join(DATA_DIR, "collections_cache.parquet")

DATE_COLS = ["STATUS_DATE", "RESOLUTION_TILL_DATE_LABEL", "LCS_LAST_UPDATED", "LAST_MOD"]
NUM_COLS = ["NO_ACCOUNTS", "POS_CRS", "AUM_ON_BOOK", "AUM_OFF_BOOK", "WRITE_OFF_AMT"]
FLAG_COLS = ["BOUNCE_FLAG", "WRITE_OFF_FLAG", "IS_ACTIVE", "POCI_FLAG", "BKT6_BELOW_FLAG"]
CAT_COLS = [
    "BUCKET", "FIN_TYPE", "SOURCE", "IS_MIGRATED", "PCHFL_RESOLUTIONSTATUS",
    "PCHFL_RESOLUTIONTYPE", "PCHFL_REPAYMENTFREQUENCY", "MONTH_NAME", "PRODUCT_LINE",
    "FINAL_PRODUCT_CATEGORY", "PAYMENT_MODE", "PAYMENT_MODE_FLAG", "ENTITY",
    "TENURE_TYPE", "FINAL_TAG", "SOR_TYPE", "INVESTORNAME", "DISTRICT", "STATE",
    "ZONE", "MOB_FILTER", "DISPOSITION_TAG", "DISPOSITION_TAG_CATG", "AGENCY_NAME",
    "BUSINESS_ALLOC_TYPE", "CITY_TIER", "PRODUCT_CATEGORY_FINAL", "X_SELL_TYPE_AGG",
    "GEOGRAPHY",
]
JUNK_VALUES = ["#REF!#REF!", "#N/A", "N/A", "nan", ""]

BUCKET_ORDER = ["1.B1", "2.B2", "3.B3", "4.NPA", "4.Opening_NPA", "4.Risk_NPA"]
BUCKET_COLORS = {
    "1.B1": "#2ca02c", "2.B2": "#9edae5", "3.B3": "#ffbb78",
    "4.NPA": "#d62728", "4.Opening_NPA": "#ff9896", "4.Risk_NPA": "#8c564b",
}
NPA_BUCKETS = ["4.NPA", "4.Opening_NPA", "4.Risk_NPA"]

POS, AUM_ON, AUM_OFF, ACCTS, WROFF = (
    "POS_CRS", "AUM_ON_BOOK", "AUM_OFF_BOOK", "NO_ACCOUNTS", "WRITE_OFF_AMT",
)
# ---------------------------------------------------------------------------
# Data loading & formatting helpers
# ---------------------------------------------------------------------------
def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names, dtypes and known dirty values."""
    df.columns = [str(c).strip().upper() for c in df.columns]

    for c in CAT_COLS:
        if c in df.columns:
            df[c] = (
                df[c].astype("string").str.strip()
                .replace({v: "Unknown" for v in JUNK_VALUES})
                .fillna("Unknown")
            )

    for c in DATE_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    for c in NUM_COLS + FLAG_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Down-cast for memory / speed.
    if ACCTS in df.columns:
        df[ACCTS] = df[ACCTS].fillna(0).astype("int32")
    for c in [POS, AUM_ON, AUM_OFF, WROFF]:
        if c in df.columns:
            df[c] = df[c].astype("float32")

    # Derived helpers.
    if "BUCKET" in df.columns:
        df["BUCKET"] = pd.Categorical(df["BUCKET"], categories=BUCKET_ORDER, ordered=True)
        df["IS_NPA"] = df["BUCKET"].isin(NPA_BUCKETS)
    if "MONTH_NAME" in df.columns:
        df["MONTH_ORDER"] = pd.to_datetime(df["MONTH_NAME"], format="%b-%y", errors="coerce")

    return df


def _read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="ISO-8859-1", low_memory=False)


@st.cache_data(show_spinner="Loading collections data ...")
def load_from_path(path: str) -> pd.DataFrame:
    """Load the CSV, using (and building) a Parquet cache for speed.

    Parquet is columnar: the clean-once frame is stored so subsequent runs skip
    the slow ISO-8859-1 CSV parse of the ~259 MB source. The cache is rebuilt
    whenever the source CSV is newer than the cache.
    """
    try:
        import pyarrow  # noqa: F401
    except Exception:
        return _clean(_read_csv(path))  # graceful fallback

    try:
        cache_fresh = (
            os.path.exists(PARQUET_FILE)
            and os.path.getmtime(PARQUET_FILE) >= os.path.getmtime(path)
        )
        if cache_fresh:
            return pd.read_parquet(PARQUET_FILE)
        df = _clean(_read_csv(path))
        df.to_parquet(PARQUET_FILE, index=False)
        return df
    except Exception:
        # Any Parquet problem -> fall back to the raw CSV path.
        return _clean(_read_csv(path))


@st.cache_data(show_spinner="Loading collections data ...")
def load_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    import io
    return _clean(pd.read_csv(io.BytesIO(file_bytes), encoding="ISO-8859-1", low_memory=False))


def cr(value) -> str:
    """Format a value as crores."""
    try:
        return f"{value:,.2f} Cr"
    except (TypeError, ValueError):
        return "-"


def num(value) -> str:
    """Format an integer-ish value with separators."""
    try:
        return f"{value:,.0f}"
    except (TypeError, ValueError):
        return "-"


def pct(value, dp: int = 1) -> str:
    try:
        return f"{value:,.{dp}f}%"
    except (TypeError, ValueError):
        return "-"


# background_gradient needs matplotlib, which may be absent; fall back safely.
try:
    import matplotlib  # noqa: F401
    HAS_MPL = True
except Exception:
    HAS_MPL = False


def show_table(frame: pd.DataFrame, cmap: str = "Blues"):
    """Render a styled dataframe when possible, else a plain one."""
    if HAS_MPL:
        try:
            st.write(frame.style.background_gradient(cmap=cmap))
            return
        except Exception:
            pass
    st.write(frame)


def ordered_unique(series) -> list:
    """Unique non-null values, preserving category order when present."""
    if isinstance(series.dtype, pd.CategoricalDtype):
        cats = [c for c in series.cat.categories if c in set(series.dropna())]
        extra = sorted({str(v) for v in series.dropna() if v not in cats})
        return cats + extra
    return sorted(series.dropna().astype(str).unique().tolist())


def df_download(frame: pd.DataFrame, label: str, fname: str):
    st.download_button(label, data=frame.to_csv(index=False).encode("utf-8"),
                       file_name=fname, mime="text/csv")


# ---------------------------------------------------------------------------
# Aggregation helpers (account-weighted where rates are involved)
# ---------------------------------------------------------------------------
METRIC_OPTIONS = {
    "POS (Cr)": POS,
    "Accounts": ACCTS,
    "AUM On-Book (Cr)": AUM_ON,
    "Write-Off Amt (Cr)": WROFF,
}


def weighted_rate(df: pd.DataFrame, mask: pd.Series, weight: str = ACCTS) -> float:
    """Account-weighted percentage of rows satisfying ``mask``."""
    w = df[weight].fillna(0)
    total = w.sum()
    if total <= 0:
        return float("nan")
    return float((w[mask]).sum() / total * 100)


def group_weighted_rate(df: pd.DataFrame, group: str, flag: str) -> pd.Series:
    """Vectorised account-weighted percentage of ``flag`` within each ``group``."""
    tmp = pd.DataFrame({
        "_w": df[ACCTS].fillna(0).to_numpy(),
        "_n": (df[flag].fillna(False).astype(bool).astype("int64").to_numpy()
               * df[ACCTS].fillna(0).to_numpy()),
    }, index=df.index)
    tmp[group] = df[group]
    g = tmp.groupby(group, observed=True)[["_n", "_w"]].sum()
    return (g["_n"] / g["_w"].replace(0, np.nan) * 100)


def kpi_block(df: pd.DataFrame) -> dict:
    """Compute the headline metrics for a filtered frame."""
    accounts = float(df[ACCTS].sum())
    npa_w = weighted_rate(df, df["IS_NPA"]) if "IS_NPA" in df else float("nan")
    write_off = float(df.loc[df["WRITE_OFF_FLAG"] == 1, WROFF].fillna(0).sum()) \
        if "WRITE_OFF_FLAG" in df else float("nan")
    pos_total = float(df[POS].sum())
    return {
        "accounts": accounts,
        "pos": pos_total,
        "aum_on": float(df[AUM_ON].sum()),
        "aum_off": float(df[AUM_OFF].sum()) if AUM_OFF in df else 0.0,
        "write_off": write_off,
        "active_pct": weighted_rate(df, df["IS_ACTIVE"] > 0) if "IS_ACTIVE" in df else float("nan"),
        "bounce_pct": weighted_rate(df, df["BOUNCE_FLAG"] > 0) if "BOUNCE_FLAG" in df else float("nan"),
        "npa_pct": npa_w,
        "write_off_pct": (write_off / pos_total * 100) if pos_total else float("nan"),
    }


def kpi_card(label: str, value: str, sub: str = ""):
    st.markdown(
        f"<div class='kpi-card'><div class='kpi-label'>{label}</div>"
        f"<div class='kpi-value'>{value}</div>"
        f"<div class='kpi-sub'>{sub}</div></div>",
        unsafe_allow_html=True,
    )


def agg_by(df: pd.DataFrame, col: str, metric: str, top: int | None = None) -> pd.DataFrame:
    """Group ``df`` by ``col`` and aggregate ``metric`` into a ``value`` column.

    ``metric`` may be a display label from ``METRIC_OPTIONS`` or a raw column
    name. Returns columns ``[col, "value"]`` sorted descending.
    """
    metric = METRIC_OPTIONS.get(metric, metric)
    if metric not in df.columns:
        return pd.DataFrame({col: [], "value": []})
    out = df.groupby(col, observed=True, as_index=False)[metric].sum()
    out = out.rename(columns={metric: "value"}).sort_values("value", ascending=False)
    if top:
        out = out.head(top)
    return out


def share_by(df: pd.DataFrame, col: str, metric: str = POS, top: int | None = None) -> pd.DataFrame:
    """Like ``agg_by`` but safe for pie/share charts.

    Negative balances (reversals/adjustments in the source) cannot be shown as a
    slice, so they are floored at zero; the effective total is reported by
    ``neg_share_note``.
    """
    out = agg_by(df, col, metric, top)
    if not out.empty:
        out["value"] = out["value"].clip(lower=0)
        out = out[out["value"] > 0]
    return out


def neg_share_note(df: pd.DataFrame, metric: str = POS) -> str:
    """Return a caption noting any negatives excluded from share charts."""
    if metric not in df.columns:
        return ""
    neg = int((df[metric] < 0).sum())
    if neg == 0:
        return ""
    return (f"{num(neg)} record(s) with negative {metric} were excluded from "
            f"share charts (floored at zero).")


# ---------------------------------------------------------------------------
# Click-to-drill-down (Plotly chart selections drive the filters)
# ---------------------------------------------------------------------------
def _selected_value(event, field: str):
    """Extract the clicked point's ``field`` value from a Plotly selection."""
    if event is None:
        return None
    sel = event.get("selection") if isinstance(event, dict) else getattr(event, "selection", None)
    if sel is None:
        return None
    points = sel.get("points") if isinstance(sel, dict) else getattr(sel, "points", None)
    if not points:
        return None
    p = points[0]
    return p.get(field) if isinstance(p, dict) else getattr(p, field, None)


def drillable_chart(fig, field_col: str, chart_key: str, axis: str = "y"):
    """Render a chart whose clicked bar drills the dashboard into ``field_col``.

    Clicking a bar sets the drill filter; clicking it again (deselecting) clears
    it. Drills are stored in ``st.session_state['drill']`` and applied with the
    sidebar filters on the next run.
    """
    event = st.plotly_chart(
        fig, width="stretch", key=chart_key,
        on_select="rerun", selection_mode="points",
    )
    val = _selected_value(event, axis)
    drill = st.session_state.get("drill", {})
    if val is not None:
        if drill.get(field_col) != val:
            st.session_state["drill"] = {field_col: val}
            st.session_state["drill_src"] = chart_key
            st.rerun()
    elif st.session_state.get("drill_src") == chart_key and field_col in drill:
        # Selection cleared on the owning chart -> drop the drill.
        st.session_state["drill"] = {}
        st.session_state["drill_src"] = None
        st.rerun()
    return event


def active_drill_labels() -> list:
    """Human-readable labels for the currently active drill-downs."""
    return [f"{col} = {val}" for col, val in st.session_state.get("drill", {}).items()]


def monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly snapshot aggregates, chronologically ordered."""
    if "MONTH_NAME" not in df.columns:
        return pd.DataFrame()
    out = (
        df.groupby(["MONTH_NAME", "MONTH_ORDER"], as_index=False)
        .agg(**{"POS": (POS, "sum"), "Accounts": (ACCTS, "sum"),
                "AUM": (AUM_ON, "sum")})
        .sort_values("MONTH_ORDER")
    )
    npa = group_weighted_rate(df, "MONTH_NAME", "IS_NPA").rename("NPA_pct").reset_index()
    return out.merge(npa, on="MONTH_NAME", how="left")


# ---------------------------------------------------------------------------
# Rule-based auto-insight narrative (no external calls)
# ---------------------------------------------------------------------------
def month_keys(df: pd.DataFrame) -> list:
    """Distinct months present, ordered chronologically."""
    if "MONTH_ORDER" not in df.columns:
        return []
    m = df[["MONTH_NAME", "MONTH_ORDER"]].dropna().drop_duplicates()
    return m.sort_values("MONTH_ORDER")["MONTH_NAME"].tolist()


def build_insights(df: pd.DataFrame) -> list:
    """Return a list of plain-English insight bullets for the selection."""
    tips = []
    k = kpi_block(df)
    tips.append(
        f"Selection covers **{num(k['accounts'])} accounts** across "
        f"{num(len(df))} records with **{cr(k['pos'])}** POS."
    )

    # NPA movement across monthly snapshots.
    months = month_keys(df)
    if len(months) >= 2:
        first, last = months[0], months[-1]
        a = weighted_rate(df[df["MONTH_NAME"] == first], df.loc[df["MONTH_NAME"] == first, "IS_NPA"])
        b = weighted_rate(df[df["MONTH_NAME"] == last], df.loc[df["MONTH_NAME"] == last, "IS_NPA"])
        if not (np.isnan(a) or np.isnan(b)):
            arrow = "▲" if b >= a else "▼"
            tips.append(
                f"NPA share is {pct(b)} ({arrow} {abs(b - a):.1f} pp vs {first})."
            )

    # Concentration by state.
    if "STATE" in df.columns:
        s = agg_by(df, "STATE", POS)
        total = s["value"].sum()
        if total:
            top3 = s.head(3)
            tips.append(
                f"Top 3 states ({', '.join(top3['STATE'])}) contribute "
                f"**{top3['value'].sum() / total * 100:.0f}%** of POS."
            )

    # Operational gap: unallocated disposition.
    if "DISPOSITION_TAG_CATG" in df.columns:
        unalloc_n = df.loc[df["DISPOSITION_TAG_CATG"] == "unallocated", ACCTS].sum()
        if k["accounts"]:
            tips.append(
                f"**{unalloc_n / k['accounts'] * 100:.0f}%** of accounts are still "
                f"unallocated to any collections channel."
            )

    # Anomaly flag: districts with unusually high NPA share.
    if {"DISTRICT", "IS_NPA"}.issubset(df.columns):
        grp = group_weighted_rate(df, "DISTRICT", "IS_NPA")
        counts = df.groupby("DISTRICT", observed=True)[ACCTS].sum()
        grp = grp[(counts > 500) & grp.notna()]
        if len(grp) > 5 and grp.std() > 0:
            z = (grp - grp.mean()) / grp.std()
            outliers = z[z > 2].sort_values(ascending=False)
            if len(outliers):
                top_d = outliers.index[0]
                tips.append(
                    f"⚠️ {len(outliers)} district(s) show NPA share >2σ above average "
                    f"(highest: {top_d}, {grp[top_d]:.1f}%)."
                )
    return tips


# ===========================================================================
# MAIN APP
# ===========================================================================
st.title(":bar_chart: Collections & Delinquency Portfolio Dashboard")
st.caption(
    "A single, filterable view of the PCHFL-style collections book: portfolio "
    "size, delinquency mix, collections operations and geographic concentration."
)

# --- Data source -----------------------------------------------------------
with st.sidebar:
    st.header("Data")
    fl = st.file_uploader("Upload a file", type=["csv", "txt", "xlsx", "xls"])

if fl is not None:
    df = load_from_bytes(fl.getvalue())
    st.sidebar.success(f"Loaded: {fl.name}")
else:
    df = load_from_path(DATA_FILE)

if df.empty:
    st.warning("No data available. Please upload a file to continue.")
    st.stop()

# --- Session state ---------------------------------------------------------
st.session_state.setdefault("drill", {})
st.session_state.setdefault("drill_src", None)
st.session_state.setdefault("metric", list(METRIC_OPTIONS.keys())[0])

# --- Sidebar filters -------------------------------------------------------
with st.sidebar:
    st.header("Choose your filter")

    date_filter = None
    if "STATUS_DATE" in df.columns and df["STATUS_DATE"].notna().any():
        st.subheader("Status Date range")
        d1 = st.date_input("Start Date", df["STATUS_DATE"].min().date())
        d2 = st.date_input("End Date", df["STATUS_DATE"].max().date())
        date_filter = (pd.to_datetime(d1), pd.to_datetime(d2))

    FILTER_COLUMNS = [
        ("ZONE", "Zone"),
        ("STATE", "State"),
        ("BUCKET", "Delinquency Bucket"),
        ("PRODUCT_CATEGORY_FINAL", "Product Category"),
        ("TENURE_TYPE", "Tenure Type"),
        ("DISPOSITION_TAG_CATG", "Disposition Category"),
    ]
    selections = {}
    for col, label in FILTER_COLUMNS:
        if col in df.columns:
            selections[col] = st.multiselect(label, ordered_unique(df[col]))

    st.divider()
    metric = st.radio(
        "Primary metric",
        list(METRIC_OPTIONS.keys()),
        help="Drives the adaptable charts across the dashboard.",
    )
    st.session_state["metric"] = metric

    st.divider()
    st.caption("💡 Tip: click any bar in the Bucket, Product, State or Agency "
               "charts to drill the whole dashboard into it.")
    if st.session_state.get("drill"):
        if st.button("✖ Reset drill-down", width="stretch"):
            st.session_state["drill"] = {}
            st.session_state["drill_src"] = None
            st.rerun()

# --- Apply filters ---------------------------------------------------------
drill = st.session_state.get("drill", {})
filtered_df = df.copy()
if date_filter is not None:
    filtered_df = filtered_df[
        (filtered_df["STATUS_DATE"] >= date_filter[0])
        & (filtered_df["STATUS_DATE"] <= date_filter[1])
    ]
for col, chosen in selections.items():
    if chosen:
        filtered_df = filtered_df[filtered_df[col].isin(chosen)]
for col, val in drill.items():
    if col in filtered_df.columns:
        filtered_df = filtered_df[filtered_df[col].astype(str) == str(val)]

if filtered_df.empty:
    st.warning("No records match the current filters. Please widen your selection.")
    st.stop()

# --- Active drill breadcrumb + clear --------------------------------------
_drills = active_drill_labels()
if _drills:
    bc1, bc2 = st.columns((5, 1))
    with bc1:
        st.info("🔎 **Drilled into:** " + " → ".join(_drills)
                + "  ·  click the same chart element again, or clear below, to reset.")
    with bc2:
        if st.button("✖ Clear drill", width="stretch"):
            st.session_state["drill"] = {}
            st.session_state["drill_src"] = None
            st.rerun()

# --- Executive summary + KPIs ---------------------------------------------
k = kpi_block(filtered_df)
insights = build_insights(filtered_df)

st.markdown(
    "<div class='exec-box'><h4>Executive summary</h4><ul>"
    + "".join(f"<li>{t}</li>" for t in insights)
    + "</ul></div>",
    unsafe_allow_html=True,
)

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    kpi_card("Accounts", num(k["accounts"]), f"{num(len(filtered_df))} records")
with c2:
    kpi_card("Total POS", cr(k["pos"]), "principal outstanding")
with c3:
    kpi_card("AUM On-Book", cr(k["aum_on"]), f"Off-book {cr(k['aum_off'])}")
with c4:
    kpi_card("Write-Off", cr(k["write_off"]), f"{pct(k['write_off_pct'])} of POS")
with c5:
    kpi_card("NPA Share", pct(k["npa_pct"]), "account-weighted")
with c6:
    kpi_card("Bounce / Active", f"{pct(k['bounce_pct'])} / {pct(k['active_pct'])}",
             "account-weighted")
st.write("")

# ===========================================================================
# STORY-DRIVEN TABS
# ===========================================================================
tab_overview, tab_delinq, tab_ops, tab_geo, tab_data = st.tabs(
    ["📊 Overview", "⚠️ Delinquency", "🏦 Collections Ops", "🗺️ Geography", "📄 Data"]
)

# ---------------------------------------------------------------------------
# TAB 1 - Overview
# ---------------------------------------------------------------------------
with tab_overview:
    st.markdown("<div class='section-note'>Portfolio composition and movement "
                "across the available monthly snapshots.</div>", unsafe_allow_html=True)

    o1, o2 = st.columns((3, 2))
    with o1:
        st.subheader(f"Trend by month — {metric}")
        ts = monthly(filtered_df)
        if not ts.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ts["MONTH_NAME"], y=ts["POS"], name="POS (Cr)",
                                     mode="lines+markers", yaxis="y1",
                                     line=dict(color=PALETTE[0], width=3)))
            fig.add_trace(go.Scatter(x=ts["MONTH_NAME"], y=ts["Accounts"], name="Accounts",
                                     mode="lines+markers", yaxis="y2",
                                     line=dict(color=PALETTE[1], width=3, dash="dot")))
            fig.update_layout(
                template=THEME, height=420,
                yaxis=dict(title="POS (Cr)"),
                yaxis2=dict(title="Accounts", overlaying="y", side="right", showgrid=False),
                legend=dict(orientation="h", y=1.12, x=0),
                margin=dict(t=40, l=10, r=10, b=10),
            )
            st.plotly_chart(fig, width="stretch")
            st.caption("POS (solid, left axis) vs account count (dotted, right axis) — "
                       "separate scales avoid flattening the POS line.")
        else:
            st.info("No month data available.")

    with o2:
        st.subheader(f"{metric} by Delinquency Bucket")
        st.caption("Click a bar to drill the whole dashboard into that bucket.")
        bdf = agg_by(filtered_df, "BUCKET", metric)
        fig = px.bar(bdf, x="value", y="BUCKET", orientation="h", color="BUCKET",
                     color_discrete_map=BUCKET_COLORS, template=THEME,
                     custom_data=["BUCKET"],
                     text=bdf["value"].map(lambda v: f"{v:,.0f}"))
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, height=420, yaxis_title="",
                          xaxis_title=metric, margin=dict(t=40, l=10, r=10, b=10))
        drillable_chart(fig, "BUCKET", "ov_bucket", axis="y")

    st.divider()
    p1, p2 = st.columns(2)
    with p1:
        st.subheader("Account mix by bucket")
        mix = (
            filtered_df.groupby("BUCKET", observed=True, as_index=False)[ACCTS].sum()
        )
        fig = px.pie(mix, values=ACCTS, names="BUCKET", hole=0.55,
                     color="BUCKET", color_discrete_map=BUCKET_COLORS, template=THEME)
        fig.update_traces(textinfo="label+percent")
        fig.update_layout(height=400, margin=dict(t=30, l=10, r=10, b=10))
        st.plotly_chart(fig, width="stretch")

    with p2:
        st.subheader(f"Product Category — {metric}")
        st.caption("Click a bar to drill into that product category.")
        cdf = agg_by(filtered_df, "PRODUCT_CATEGORY_FINAL", metric).head(10).sort_values("value")
        fig = px.bar(cdf, x="value", y="PRODUCT_CATEGORY_FINAL", orientation="h",
                     color="value", color_continuous_scale="Blues", template=THEME,
                     labels={"value": metric})
        fig.update_layout(height=400, yaxis_title="", coloraxis_showscale=False,
                          margin=dict(t=30, l=10, r=10, b=10))
        drillable_chart(fig, "PRODUCT_CATEGORY_FINAL", "ov_product", axis="y")


# ---------------------------------------------------------------------------
# TAB 2 - Delinquency deep-dive
# ---------------------------------------------------------------------------
with tab_delinq:
    d1, d2 = st.columns(2)
    with d1:
        st.subheader("Bucket mix over snapshots")
        st.markdown("<div class='section-note'>Accounts per bucket; rising NPA bands "
                    "signal deteriorating credit quality.</div>", unsafe_allow_html=True)
        if "MONTH_NAME" in filtered_df.columns:
            mix = (
                filtered_df.groupby(["MONTH_NAME", "MONTH_ORDER", "BUCKET"],
                                    observed=True, as_index=False)[ACCTS].sum()
                .sort_values("MONTH_ORDER")
            )
            fig = px.bar(mix, x="MONTH_NAME", y=ACCTS, color="BUCKET",
                         color_discrete_map=BUCKET_COLORS, template=THEME,
                         labels={ACCTS: "Accounts", "MONTH_NAME": ""})
            fig.update_layout(height=420, margin=dict(t=30, l=10, r=10, b=10))
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No month column available.")

    with d2:
        st.subheader("NPA share by tenure type")
        st.markdown("<div class='section-note'>Account-weighted NPA % across loan "
                    "tenure buckets.</div>", unsafe_allow_html=True)
        if "TENURE_TYPE" in filtered_df.columns:
            t = (
                filtered_df.groupby("TENURE_TYPE", observed=True)
                .agg(**{"Accounts": (ACCTS, "sum")})
                .reset_index()
            )
            t["NPA_pct"] = t["TENURE_TYPE"].map(
                group_weighted_rate(filtered_df, "TENURE_TYPE", "IS_NPA")
            )
            fig = px.bar(t, x="TENURE_TYPE", y="NPA_pct", text_auto=".1f",
                         color="TENURE_TYPE", template=THEME,
                         labels={"NPA_pct": "NPA %", "TENURE_TYPE": ""})
            fig.update_traces(textposition="outside")
            fig.update_layout(showlegend=False, height=420,
                              margin=dict(t=30, l=10, r=10, b=10))
            st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader("Concentration risk: Bucket × Month")
    st.markdown("<div class='section-note'>POS held in each bucket per snapshot — "
                "watch the NPA rows widen over time.</div>", unsafe_allow_html=True)
    if {"BUCKET", "MONTH_NAME"}.issubset(filtered_df.columns):
        pivot = pd.pivot_table(filtered_df, values=POS, index="BUCKET",
                               columns="MONTH_NAME", aggfunc="sum", observed=True)
        show_table(pivot, "Reds")
        df_download(pivot.reset_index(), "Download bucket-month pivot", "Bucket_Month.csv")

    st.divider()
    st.subheader("Risk Index — districts needing attention")
    st.markdown("<div class='section-note'>Composite of NPA share, write-off and bounce, "
                "sized by accounts. Larger, further-right districts carry the most "
                "combined risk.</div>", unsafe_allow_html=True)
    risk = (
        filtered_df.groupby("DISTRICT", observed=True)
        .agg(**{"Accounts": (ACCTS, "sum"), "POS_cr": (POS, "sum")})
        .reset_index()
    )
    risk["NPA_pct"] = risk["DISTRICT"].map(
        group_weighted_rate(filtered_df, "DISTRICT", "IS_NPA"))
    if "BOUNCE_FLAG" in filtered_df.columns:
        risk["Bounce_pct"] = risk["DISTRICT"].map(
            group_weighted_rate(filtered_df, "DISTRICT", "BOUNCE_FLAG"))
    risk = risk[risk["Accounts"] > 300].nlargest(250, "Accounts")
    fig = px.scatter(risk, x="NPA_pct", y="Bounce_pct", size="Accounts",
                     color="POS_cr", color_continuous_scale="Reds", hover_name="DISTRICT",
                     template=THEME, labels={"POS_cr": "POS (Cr)"})
    fig.update_layout(height=480, margin=dict(t=30, l=10, r=10, b=10))
    st.plotly_chart(fig, width="stretch")
    with st.expander("Risk Index — data"):
        show_table(risk.sort_values("NPA_pct", ascending=False).head(50), "Reds")
        df_download(risk, "Download risk index", "Risk_Index.csv")


# ---------------------------------------------------------------------------
# TAB 3 - Collections operations
# ---------------------------------------------------------------------------
with tab_ops:
    st.markdown("<div class='section-note'>How the book is being worked: channel "
                "allocation, agency load, payment behaviour and bounce.</div>",
                unsafe_allow_html=True)

    o1, o2 = st.columns(2)
    with o1:
        st.subheader("Accounts by disposition channel")
        if "DISPOSITION_TAG_CATG" in filtered_df.columns:
            dd = (
                filtered_df.groupby("DISPOSITION_TAG_CATG", observed=True, as_index=False)[ACCTS]
                .sum().sort_values(ACCTS)
            )
            fig = px.bar(dd, x=ACCTS, y="DISPOSITION_TAG_CATG", orientation="h",
                         color=ACCTS, color_continuous_scale="Blues", template=THEME,
                         labels={ACCTS: "Accounts", "DISPOSITION_TAG_CATG": ""})
            fig.update_layout(height=420, coloraxis_showscale=False,
                              margin=dict(t=30, l=10, r=10, b=10))
            st.plotly_chart(fig, width="stretch")

    with o2:
        st.subheader("Bounce % by bucket")
        if "BOUNCE_FLAG" in filtered_df.columns:
            bb = (
                filtered_df.groupby("BUCKET", observed=True)
                .agg(**{"Accounts": (ACCTS, "sum")})
                .reset_index()
            )
            bb["Bounce_pct"] = bb["BUCKET"].map(
                group_weighted_rate(filtered_df, "BUCKET", "BOUNCE_FLAG"))
            fig = px.bar(bb, x="BUCKET", y="Bounce_pct", color="BUCKET",
                         color_discrete_map=BUCKET_COLORS, template=THEME,
                         text_auto=".1f", labels={"Bounce_pct": "Bounce %"})
            fig.update_traces(textposition="outside")
            fig.update_layout(showlegend=False, height=420, xaxis_title="",
                              margin=dict(t=30, l=10, r=10, b=10))
            st.plotly_chart(fig, width="stretch")

    st.divider()
    a1, a2 = st.columns((3, 2))
    with a1:
        st.subheader("Top 15 collection agencies by POS")
        st.caption("Click an agency bar to drill the dashboard into that agency.")
        if "AGENCY_NAME" in filtered_df.columns:
            ag = agg_by(filtered_df[filtered_df["AGENCY_NAME"] != "Unknown"],
                        "AGENCY_NAME", POS).head(15).sort_values("value")
            fig = px.bar(ag, x="value", y="AGENCY_NAME", orientation="h", color="value",
                         color_continuous_scale="Oranges", template=THEME,
                         labels={"value": "POS (Cr)", "AGENCY_NAME": ""})
            fig.update_layout(height=480, coloraxis_showscale=False,
                              margin=dict(t=30, l=10, r=10, b=10))
            drillable_chart(fig, "AGENCY_NAME", "ops_agency", axis="y")

    with a2:
        st.subheader("Payment mode mix (POS)")
        if "PAYMENT_MODE" in filtered_df.columns:
            pm = share_by(filtered_df, "PAYMENT_MODE", POS).head(8)
            fig = px.pie(pm, values="value", names="PAYMENT_MODE", hole=0.5, template=THEME,
                         color_discrete_sequence=PALETTE)
            fig.update_traces(textinfo="label+percent")
            fig.update_layout(height=480, margin=dict(t=30, l=10, r=10, b=10))
            st.plotly_chart(fig, width="stretch")
            note = neg_share_note(filtered_df, POS)
            if note:
                st.caption(note)

    with st.expander("Collections operations — data"):
        if "DISPOSITION_TAG_CATG" in filtered_df.columns:
            op_tbl = (
                filtered_df.groupby("DISPOSITION_TAG_CATG", observed=True)
                .agg(**{"Accounts": (ACCTS, "sum"), "POS": (POS, "sum")})
                .reset_index().sort_values("Accounts", ascending=False)
            )
            show_table(op_tbl, "Blues")
            df_download(op_tbl, "Download operations data", "Operations.csv")


# ---------------------------------------------------------------------------
# TAB 4 - Geography
# ---------------------------------------------------------------------------
with tab_geo:
    st.markdown("<div class='section-note'>Where the exposure sits — zone, state and "
                "city-tier concentration of the selected book.</div>", unsafe_allow_html=True)

    g1, g2 = st.columns((3, 2))
    with g1:
        st.subheader(f"Top 15 states by {metric}")
        st.caption("Click a state bar to drill the dashboard into that state.")
        tdf = agg_by(filtered_df, "STATE", metric).head(15).sort_values("value")
        fig = px.bar(tdf, x="value", y="STATE", orientation="h", color="value",
                     color_continuous_scale="Blues", template=THEME,
                     text=tdf["value"].map(lambda v: f"{v:,.0f}"),
                     labels={"value": metric, "STATE": ""})
        fig.update_traces(textposition="outside")
        fig.update_layout(height=520, coloraxis_showscale=False,
                          margin=dict(t=30, l=10, r=10, b=10))
        drillable_chart(fig, "STATE", "geo_state", axis="y")

    with g2:
        st.subheader("Zone share of POS")
        zdf = share_by(filtered_df, "ZONE", POS)
        fig = px.pie(zdf, values="value", names="ZONE", hole=0.5, template=THEME,
                     color_discrete_sequence=PALETTE)
        fig.update_traces(textinfo="label+percent")
        fig.update_layout(height=250, margin=dict(t=20, l=10, r=10, b=10))
        st.plotly_chart(fig, width="stretch")
        note = neg_share_note(filtered_df, POS)
        if note:
            st.caption(note)

        st.subheader("City tier mix")
        if "CITY_TIER" in filtered_df.columns:
            ct = agg_by(filtered_df, "CITY_TIER", ACCTS).sort_values("value")
            fig = px.bar(ct, x="value", y="CITY_TIER", orientation="h", template=THEME,
                         color_discrete_sequence=[PALETTE[2]], labels={"CITY_TIER": ""})
            fig.update_layout(height=240, margin=dict(t=20, l=10, r=10, b=10))
            st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader("Hierarchical POS: Zone → Product Category → State")
    path_cols = [c for c in ["ZONE", "PRODUCT_CATEGORY_FINAL", "STATE"]
                 if c in filtered_df.columns]
    if len(path_cols) >= 2:
        tree_df = (
            filtered_df.groupby(path_cols, observed=True, as_index=False)
            .agg(**{"value": (POS, "sum"), "Accounts": (ACCTS, "sum")})
        )
        tree_df = tree_df[tree_df["value"] > 0]
        fig = px.treemap(tree_df, path=path_cols, values="value", color="value",
                         color_continuous_scale="Reds", hover_data=["Accounts"])
        fig.update_layout(height=560, margin=dict(t=30, l=10, r=10, b=10))
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Not enough categorical columns for a treemap.")

    st.divider()
    st.subheader("Top states — detail")
    st_tbl = (filtered_df.groupby("STATE", observed=True)
              .agg(**{"Accounts": (ACCTS, "sum"), "POS": (POS, "sum"),
                      "AUM_On": (AUM_ON, "sum")})
              .reset_index().sort_values("POS", ascending=False).head(25))
    show_table(st_tbl, "Blues")
    df_download(st_tbl, "Download state table", "Top_States.csv")
# ---------------------------------------------------------------------------
# TAB 5 - Data & Download
# ---------------------------------------------------------------------------
with tab_data:
    st.markdown("<div class='section-note'>Underlying aggregates and the raw filtered "
                "records. All tables are downloadable.</div>", unsafe_allow_html=True)

    st.subheader("Selection summary")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        kpi_card("Records shown", num(len(filtered_df)),
                 f"of {num(len(df))} total")
    with s2:
        kpi_card("Accounts", num(k["accounts"]), "account-weighted basis")
    with s3:
        kpi_card("POS", cr(k["pos"]), "principal outstanding")
    with s4:
        kpi_card("NPA share", pct(k["npa_pct"]), "reported")

    st.divider()
    st.subheader("Bucket × Month pivot (POS)")
    if {"BUCKET", "MONTH_NAME"}.issubset(filtered_df.columns):
        pivot = pd.pivot_table(filtered_df, values=POS, index="BUCKET",
                               columns="MONTH_NAME", aggfunc="sum", observed=True)
        show_table(pivot, "Blues")
        df_download(pivot.reset_index(), "Download pivot", "Bucket_Month_Pivot.csv")

    st.divider()
    st.subheader("Sample records")
    sample_cols = [c for c in ["BUCKET", "ZONE", "STATE", "PRODUCT_CATEGORY_FINAL",
                               "TENURE_TYPE", "MOB_FILTER", ACCTS, POS, AUM_ON, AUM_OFF,
                               WROFF] if c in filtered_df.columns]
    st.plotly_chart(
        ff.create_table(filtered_df[sample_cols].head(5), colorscale="Cividis"),
        width="stretch",
    )

    with st.expander("View filtered records (first 1,000)"):
        show_table(filtered_df[sample_cols].head(1000), "Oranges")

    st.divider()
    st.download_button(
        "⬇️ Download filtered data",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="Collections_Filtered.csv",
        mime="text/csv",
    )
    st.caption(
        "Data note: this is a period snapshot (source last updated 2026-10-14); "
        "months represent reporting snapshots, not a transaction ledger. "
        "All rates are account-weighted using NO_ACCOUNTS."
    )
