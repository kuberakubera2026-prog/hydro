import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "hydro.csv"


BG, CARD = "#0b1120", "#111c31"
TEXT, MUTED, GRID = "#f8fafc", "#94a3b8", "#334155"

BLUE = [[0, "#172554"], [.5, "#2563eb"], [1, "#22c55e"]]
RED_GREEN = [[0, "#dc2626"], [.5, "#facc15"], [1, "#22c55e"]]
CYAN = [[0, "#172554"], [.5, "#06b6d4"], [1, "#22c55e"]]
PURPLE = [[0, "#312e81"], [.5, "#7c3aed"], [1, "#f97316"]]
NET_WORTH = [
    [0, "#4c1d95"], [.25, "#7c3aed"], [.5, "#c026d3"],
    [.75, "#ec4899"], [1, "#fb923c"]
]


# ============================================================
# LOAD DATA
# ============================================================

raw = pd.read_csv(CSV_FILE)

if raw.empty:
    raise ValueError("hydro.csv is empty.")

df = (
    raw.set_index(raw.columns[0])
       .T
       .rename_axis("Symbol")
       .reset_index()
)

df["Symbol"] = (
    df["Symbol"]
    .astype("string")
    .str.strip()
)

df = (
    df[
        df["Symbol"].notna()
        & df["Symbol"].ne("")
    ]
    .drop_duplicates("Symbol")
    .sort_values(
        "Symbol",
        key=lambda x: x.str.upper(),
        kind="stable"
    )
    .reset_index(drop=True)
)


# ============================================================
# NUMBER CONVERSION
# ============================================================

INVALID = {
    "", "-", "--", "na", "n/a",
    "nan", "none", "null"
}


def number(value):
    """Convert text/numeric value to float."""

    if pd.isna(value):
        return np.nan

    text = str(value).strip().lower()

    if text in INVALID:
        return np.nan

    text = text.replace(",", "")

    match = re.search(
        r"-?\d+(?:\.\d+)?",
        text
    )

    return float(match.group()) if match else np.nan


def crore(value):
    """Convert Cr / Ar / Lac values to Crore."""

    if pd.isna(value):
        return np.nan

    text = str(value).strip().lower()
    value = number(text)

    if pd.isna(value):
        return np.nan

    if "ar" in text:
        return value * 100

    if "lac" in text or "lakh" in text:
        return value / 100

    return value


# ============================================================
# NUMERIC COLUMNS
# ============================================================

NORMAL_COLUMNS = [
    "LTP",
    "Capacity (MW)",
    "EPS (Reported)",
    "P/B Ratio",
    "Net Worth",
]

for col in NORMAL_COLUMNS:
    df[f"{col}_num"] = (
        df[col].map(number)
        if col in df
        else np.nan
    )

df["Total Mkt Cap_num"] = (
    df["Total Mkt Cap"].map(crore)
    if "Total Mkt Cap" in df
    else np.nan
)


# ============================================================
# CALCULATED VALUES
# ============================================================

df["Capacity_LTP_num"] = (
    df["Capacity (MW)_num"]
    / df["Total Mkt Cap_num"]
    * df["LTP_num"]
)

df["BookValue_4_LTP_num"] = (
    df["Net Worth_num"] * 4
    - df["LTP_num"]
)

df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)


# ============================================================
# COMPANY LIST
# ============================================================

symbols = sorted(
    df["Symbol"].astype(str).unique(),
    key=str.upper
)

COMPANY_OPTIONS = [
    {"label": x, "value": x}
    for x in symbols
]


# ============================================================
# CHART CONFIGURATION
# ============================================================

CHARTS = {
    "ltp": {
        "column": "LTP_num",
        "title": "Last Traded Price",
        "y": "LTP (Rs.)",
        "colors": BLUE,
    },

    "eps": {
        "column": "EPS (Reported)_num",
        "title": "Reported EPS",
        "y": "EPS",
        "colors": RED_GREEN,
        "zero": True,
    },

    "capacity": {
        "column": "Capacity (MW)_num",
        "title": "Hydropower Capacity",
        "y": "Capacity (MW)",
        "colors": CYAN,
    },

    "net_worth": {
        "column": "Net Worth_num",
        "title": "Net Worth",
        "y": "Net Worth",
        "colors": NET_WORTH,
    },

    "capacity_ltp": {
        "column": "Capacity_LTP_num",
        "title": "Capacity / Market Cap × LTP",
        "y": "Calculated Value",
        "colors": PURPLE,
    },

    "book_value_diff": {
        "column": "BookValue_4_LTP_num",
        "title": "Difference Value",
        "y": "Difference (Rs.)",
        "colors": RED_GREEN,
        "zero": True,
    },

    "pb": {
        "column": "P/B Ratio_num",
        "title": "Price to Book Ratio",
        "y": "P/B Ratio",
        "colors": PURPLE,
    },
}


# ============================================================
# CHART HELPERS
# ============================================================

def layout(title):

    return dict(
        title=dict(
            text=title,
            font=dict(size=18, color=TEXT),
            x=0.5,
            xanchor="center"
        ),
        template="plotly_dark",
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font=dict(color=TEXT),
        margin=dict(l=60, r=25, t=65, b=80),
        xaxis=dict(
            gridcolor=GRID,
            tickangle=-45
        ),
        yaxis=dict(gridcolor=GRID),
        hovermode="closest",
        height=450
    )


def empty_chart(title):

    fig = go.Figure()

    fig.add_annotation(
        text="No data",
        x=.5,
        y=.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            size=18,
            color=MUTED
        )
    )

    fig.update_layout(**layout(title))

    return fig


def bar_chart(data, cfg):

    col = cfg["column"]

    if col not in data:
        return empty_chart(cfg["title"])

    d = (
        data[["Symbol", col]]
        .dropna()
        .sort_values(
            "Symbol",
            key=lambda x: x.str.upper(),
            kind="stable"
        )
    )

    if d.empty:
        return empty_chart(cfg["title"])

    fig = go.Figure(
        go.Bar(
            x=d["Symbol"],
            y=d[col],

            text=[
                f"{v:,.2f}"
                for v in d[col]
            ],

            textposition="outside",

            marker=dict(
                color=d[col],
                colorscale=cfg["colors"],
                showscale=True,
                colorbar=dict(
                    thickness=12,
                    title=cfg["y"]
                ),
                line=dict(
                    color="white",
                    width=1
                )
            ),

            hovertemplate=(
                "<b>%{x}</b><br>"
                f"{cfg['y']}: "
                "%{y:,.2f}"
                "<extra></extra>"
            )
        )
    )

    fig.update_layout(
        **layout(cfg["title"]),
        yaxis_title=cfg["y"],
    )

    if cfg.get("zero"):
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="white"
        )

    return fig


def ltp_eps_chart(data):

    cols = [
        "Symbol",
        "LTP_num",
        "EPS (Reported)_num"
    ]

    if not set(cols).issubset(data.columns):
        return empty_chart("LTP vs EPS")

    d = (
        data[cols]
        .dropna()
        .sort_values(
            "Symbol",
            key=lambda x: x.str.upper(),
            kind="stable"
        )
    )

    if d.empty:
        return empty_chart("LTP vs EPS")

    fig = go.Figure(
        go.Scatter(
            x=d["LTP_num"],
            y=d["EPS (Reported)_num"],
            mode="markers+text",
            text=d["Symbol"],
            textposition="top center",

            marker=dict(
                size=16,
                color=d["EPS (Reported)_num"],
                colorscale=RED_GREEN,
                showscale=True,
                colorbar=dict(title="EPS"),
                line=dict(
                    color="white",
                    width=1
                )
            ),

            hovertemplate=(
                "<b>%{text}</b><br>"
                "LTP: %{x:,.2f}<br>"
                "EPS: %{y:,.2f}"
                "<extra></extra>"
            )
        )
    )

    fig.update_layout(
        **layout("LTP vs EPS"),
        xaxis_title="LTP (Rs.)",
        yaxis_title="EPS"
    )

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="white"
    )

    return fig


# ============================================================
# KPI
# ============================================================

def kpi(title, value, color):

    if pd.isna(value):
        value = "N/A"
    elif isinstance(value, (int, np.integer)):
        value = f"{value:,}"
    else:
        value = f"{value:,.2f}"

    return html.Div(
        [
            html.Small(
                title,
                style={"color": MUTED}
            ),
            html.H2(
                value,
                style={
                    "margin": "5px 0",
                    "color": TEXT
                }
            )
        ],
        style={
            "backgroundColor": CARD,
            "padding": "15px",
            "borderRadius": "12px",
            "borderLeft": f"4px solid {color}"
        }
    )


# ============================================================
# DASH APP
# ============================================================

app = Dash(__name__)
app.title = "Hydropower Stock Dashboard"


# ============================================================
# LAYOUT
# ============================================================

app.layout = html.Div(
    [
        html.H1(
            "Hydropower Stock Dashboard",
            style={"marginBottom": "5px"}
        ),

        html.P(
            "Nepal Hydropower Market",
            style={"color": MUTED}
        ),

        dcc.Dropdown(
            id="companies",
            options=COMPANY_OPTIONS,
            multi=True,
            placeholder="Select companies...",
            style={
                "color": "#111827",
                "marginBottom": "15px"
            }
        ),

        html.Div(
            id="kpis",
            style={
                "display": "grid",
                "gridTemplateColumns":
                    "repeat(auto-fit, minmax(180px, 1fr))",
                "gap": "12px",
                "margin": "20px 0"
            }
        ),

        html.Div(
            [
                *[
                    dcc.Graph(id=name)
                    for name in CHARTS
                ],

                dcc.Graph(id="ltp_eps")
            ],
            style={
                "display": "grid",
                "gridTemplateColumns":
                    "repeat(auto-fit, minmax(450px, 1fr))",
                "gap": "15px"
            }
        )
    ],

    style={
        "backgroundColor": BG,
        "minHeight": "100vh",
        "padding": "25px",
        "fontFamily": "Arial, sans-serif",
        "color": TEXT
    }
)


# ============================================================
# CALLBACK
# ============================================================

@app.callback(
    [
        Output("kpis", "children"),
        *[
            Output(name, "figure")
            for name in CHARTS
        ],
        Output("ltp_eps", "figure")
    ],
    Input("companies", "value")
)
def update(selected):

    data = (
        df[df["Symbol"].isin(selected)].copy()
        if selected
        else df.copy()
    )

    # Always A → Z.
    data = data.sort_values(
        "Symbol",
        key=lambda x: x.str.upper(),
        kind="stable"
    )

    # --------------------------------------------------------
    # KPIs
    # --------------------------------------------------------

    kpis = [
        kpi(
            "Companies",
            len(data),
            "#38bdf8"
        ),

        kpi(
            "Average LTP",
            data["LTP_num"].mean(),
            "#22c55e"
        ),

        kpi(
            "Average EPS",
            data["EPS (Reported)_num"].mean(),
            "#facc15"
        ),

        kpi(
            "Average Price / Book",
            data["P/B Ratio_num"].mean(),
            "#a78bfa"
        )
    ]

    # --------------------------------------------------------
    # Charts
    # --------------------------------------------------------

    figures = [
        bar_chart(data, cfg)
        for cfg in CHARTS.values()
    ]

    figures.append(
        ltp_eps_chart(data)
    )

    return [kpis, *figures]


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    app.run(
        debug=False,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8050))
    )

