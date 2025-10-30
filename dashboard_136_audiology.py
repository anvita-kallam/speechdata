import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
from scipy import stats

import dash
from dash import Dash, html, dcc, Input, Output, State, callback_context
import dash_bootstrap_components as dbc

# Optional regression; handle if statsmodels is unavailable
try:
    import statsmodels.api as sm
    from statsmodels.tools.sm_exceptions import PerfectSeparationError
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False


# ---------------------------
# Configuration and constants
# ---------------------------
WORKSPACE_DIR = "/Users/anvitakallam/Documents/Speech Data"
OUTCOME_FILE = os.path.join(WORKSPACE_DIR, "ATL Speech - Correlation Workflow - 136 Outcome.csv")
PROGRAM_FILE = os.path.join(WORKSPACE_DIR, "ATL Speech - Correlation Workflow - Presence of Aud Program.csv")

# Columns we will standardize to
STANDARD_COLUMN_STATE = "State"
STANDARD_COLUMN_PROGRAM = "Audiology Program Presence"
METRIC_COLUMNS_CANONICAL = {
    "% Meeting 1": ["% Meeting 1", "Percent Meeting 1", "Meeting1", "Pct Meeting 1"],
    "% Meeting 3": ["% Meeting 3", "Percent Meeting 3", "Meeting3", "Pct Meeting 3"],
    "% Meeting 6": ["% Meeting 6", "Percent Meeting 6", "Meeting6", "Pct Meeting 6"],
}

# Map of full state name to USPS code (lower 48 + AK, HI, DC where possible)
STATE_TO_USPS = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "District of Columbia": "DC", "Florida": "FL",
    "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN",
    "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME",
    "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH",
    "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY", "North Carolina": "NC", "North Dakota": "ND",
    "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI",
    "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI",
    "Wyoming": "WY"
}


def read_csv_safely(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {c: c.strip() for c in df.columns}

    # Try to detect state column (common exact names)
    found_state_key = None
    for candidate in [
        "State", "state", "STATE", "State Name", "Jurisdiction", "State/Territory",
        "STATE_NAME", "Location", "Location Name", "State/Area"
    ]:
        if candidate in renamed.values():
            found_state_key = next(k for k, v in renamed.items() if v == candidate)
            renamed[found_state_key] = STANDARD_COLUMN_STATE
            break
    # Fallback heuristic: look for any column name containing 'state'
    if STANDARD_COLUMN_STATE not in renamed.values():
        for original, clean in renamed.items():
            low = clean.lower()
            if ("state" in low) or (low in {"jurisdiction", "location"}):
                renamed[original] = STANDARD_COLUMN_STATE
                found_state_key = original
                break

    # Normalize metric columns
    present_cols = set(renamed.values())
    inverse = {v: k for k, v in renamed.items()}
    for canonical, aliases in METRIC_COLUMNS_CANONICAL.items():
        if canonical in present_cols:
            continue
        found = None
        for alias in aliases:
            if alias in present_cols:
                found = alias
                break
        if found is not None:
            original = inverse[found]
            renamed[original] = canonical

    # Program presence detection
    for candidate in [
        "Audiology Program Presence", "Program", "Has Program", "Presence", "Audiology Program",
        "Audiology Presence", "Program Presence", "Has Audiology Program"
    ]:
        if candidate in renamed.values():
            original = next(k for k, v in renamed.items() if v == candidate)
            renamed[original] = STANDARD_COLUMN_PROGRAM
            break

    df2 = df.rename(columns=renamed)
    return df2


def coerce_program_presence(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    s = series.astype(str).str.strip().str.lower()
    true_values = {"yes", "true", "1", "y", "present", "with", "has", "have", "available"}
    false_values = {"no", "false", "0", "n", "absent", "without", "none", "not available"}
    result = s.map(lambda x: True if x in true_values else (False if x in false_values else np.nan))
    return result


def prepare_data(outcome_path: str, program_path: str) -> pd.DataFrame:
    outcome_df = standardize_columns(read_csv_safely(outcome_path))
    program_df = standardize_columns(read_csv_safely(program_path))

    # If program presence not present, attempt to derive from any yes/no like column
    if STANDARD_COLUMN_PROGRAM not in program_df.columns:
        # Heuristic: take last non-state column
        candidate_cols = [c for c in program_df.columns if c != STANDARD_COLUMN_STATE]
        if len(candidate_cols) > 0:
            program_df[STANDARD_COLUMN_PROGRAM] = coerce_program_presence(program_df[candidate_cols[-1]])

    # Merge on state
    # Ensure required columns exist before merge
    if STANDARD_COLUMN_STATE not in outcome_df.columns:
        raise KeyError(f"State column not found in outcome data. Columns: {list(outcome_df.columns)}")
    if STANDARD_COLUMN_STATE not in program_df.columns:
        raise KeyError(f"State column not found in program data. Columns: {list(program_df.columns)}")
    if STANDARD_COLUMN_PROGRAM not in program_df.columns:
        raise KeyError("Program presence column not found/derivable in program data.")

    merged = pd.merge(
        outcome_df,
        program_df[[STANDARD_COLUMN_STATE, STANDARD_COLUMN_PROGRAM]],
        on=STANDARD_COLUMN_STATE,
        how="inner",
        validate="m:1",
    )

    # Keep only needed columns
    keep_cols = [STANDARD_COLUMN_STATE, STANDARD_COLUMN_PROGRAM] + list(METRIC_COLUMNS_CANONICAL.keys())
    existing = [c for c in keep_cols if c in merged.columns]
    merged = merged[existing].copy()

    # Coerce numeric percentage columns
    for metric in METRIC_COLUMNS_CANONICAL.keys():
        if metric in merged.columns:
            merged[metric] = (
                merged[metric]
                .astype(str)
                .str.replace('%', '', regex=False)
                .str.replace(',', '', regex=False)
                .str.strip()
            )
            merged[metric] = pd.to_numeric(merged[metric], errors='coerce')

    # Ensure boolean program presence
    if STANDARD_COLUMN_PROGRAM in merged.columns:
        merged[STANDARD_COLUMN_PROGRAM] = coerce_program_presence(merged[STANDARD_COLUMN_PROGRAM])

    # Drop rows missing critical fields
    merged = merged.dropna(subset=[STANDARD_COLUMN_STATE, STANDARD_COLUMN_PROGRAM])
    # Standardize state names capitalization
    merged[STANDARD_COLUMN_STATE] = merged[STANDARD_COLUMN_STATE].astype(str).str.strip()

    # Add USPS code if available for mapping
    merged["State_Code"] = merged[STANDARD_COLUMN_STATE].map(STATE_TO_USPS)
    return merged


def compute_group_stats(df: pd.DataFrame, metric: str) -> dict:
    # Guard: metric must exist
    if metric not in df.columns:
        return {
            "n_with": 0, "n_without": 0,
            "mean_with": np.nan, "mean_without": np.nan,
            "sd_with": np.nan, "sd_without": np.nan,
            "median_with": np.nan, "median_without": np.nan,
            "t_stat": np.nan, "t_pvalue": np.nan,
            "mw_stat": np.nan, "mw_pvalue": np.nan,
            "cohens_d": np.nan, "reg_coef": np.nan,
            "reg_pvalue": np.nan, "reg_summary": None,
        }

    df_metric = df.dropna(subset=[metric])
    with_prog = df_metric[df_metric[STANDARD_COLUMN_PROGRAM] == True][metric]
    without_prog = df_metric[df_metric[STANDARD_COLUMN_PROGRAM] == False][metric]

    # Basic stats
    stats_dict = {
        "n_with": int(with_prog.shape[0]),
        "n_without": int(without_prog.shape[0]),
        "mean_with": float(with_prog.mean()) if with_prog.size > 0 else np.nan,
        "mean_without": float(without_prog.mean()) if without_prog.size > 0 else np.nan,
        "sd_with": float(with_prog.std(ddof=1)) if with_prog.size > 1 else np.nan,
        "sd_without": float(without_prog.std(ddof=1)) if without_prog.size > 1 else np.nan,
        "median_with": float(with_prog.median()) if with_prog.size > 0 else np.nan,
        "median_without": float(without_prog.median()) if without_prog.size > 0 else np.nan,
    }

    # Welch's t-test
    t_p = (np.nan, np.nan)
    if with_prog.size > 1 and without_prog.size > 1:
        t_res = stats.ttest_ind(with_prog, without_prog, equal_var=False, nan_policy='omit')
        t_p = (float(t_res.statistic), float(t_res.pvalue))

    # Mann-Whitney U
    mw_p = (np.nan, np.nan)
    if with_prog.size > 0 and without_prog.size > 0:
        try:
            mw_res = stats.mannwhitneyu(with_prog, without_prog, alternative='two-sided')
            mw_p = (float(mw_res.statistic), float(mw_res.pvalue))
        except ValueError:
            mw_p = (np.nan, np.nan)

    # Cohen's d (pooled, handles unequal n)
    cohens_d = np.nan
    if with_prog.size > 1 and without_prog.size > 1:
        s1 = with_prog.std(ddof=1)
        s2 = without_prog.std(ddof=1)
        n1 = with_prog.size
        n2 = without_prog.size
        # Pooled SD
        sp = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2)) if (n1 + n2 - 2) > 0 else np.nan
        if sp and sp > 0:
            cohens_d = (with_prog.mean() - without_prog.mean()) / sp

    # Optional regression: Outcome ~ ProgramPresence
    reg_summary = None
    reg_pvalue = np.nan
    reg_coef = np.nan
    if HAS_STATSMODELS and df_metric.shape[0] >= 3:
        try:
            X = df_metric[[STANDARD_COLUMN_PROGRAM]].astype(int)
            X = sm.add_constant(X)
            y = df_metric[metric]
            model = sm.OLS(y, X, missing='drop')
            res = model.fit()
            reg_coef = float(res.params.get(STANDARD_COLUMN_PROGRAM, np.nan))
            reg_pvalue = float(res.pvalues.get(STANDARD_COLUMN_PROGRAM, np.nan))
            reg_summary = res.summary().as_text()
        except (PerfectSeparationError, Exception):
            reg_summary = None

    stats_dict.update({
        "t_stat": t_p[0],
        "t_pvalue": t_p[1],
        "mw_stat": mw_p[0],
        "mw_pvalue": mw_p[1],
        "cohens_d": float(cohens_d) if not pd.isna(cohens_d) else np.nan,
        "reg_coef": reg_coef,
        "reg_pvalue": reg_pvalue,
        "reg_summary": reg_summary,
    })
    return stats_dict


def generate_conclusion(stats_dict: dict, metric: str, include_georgia: bool) -> str:
    # Prefer Welch t-test p-value; fallback to Mann-Whitney
    p_value = stats_dict.get("t_pvalue")
    if pd.isna(p_value):
        p_value = stats_dict.get("mw_pvalue")

    if pd.isna(p_value):
        return (
            f"Insufficient data to determine a statistical difference for {metric}. "
            f"Consider reviewing data completeness and quality."
        )

    significant = p_value < 0.05
    base = (
        f"Analysis for {metric}{' (Georgia excluded)' if not include_georgia else ''}: "
    )
    if significant:
        return (
            base
            + f"States with audiology programs show significantly higher {metric} (p = {p_value:.3g}). "
            + "Georgia would likely benefit from establishing a program."
        )
    else:
        return (
            base
            + f"No statistically significant difference found (p = {p_value:.3g}); more data may be needed."
        )


# ---------------------------
# Data load
# ---------------------------
try:
    data_df = prepare_data(OUTCOME_FILE, PROGRAM_FILE)
except Exception as e:
    # Create an empty fallback DataFrame with expected columns to keep app booting
    data_df = pd.DataFrame(columns=[STANDARD_COLUMN_STATE, STANDARD_COLUMN_PROGRAM] + list(METRIC_COLUMNS_CANONICAL.keys()))
    data_df[STANDARD_COLUMN_PROGRAM] = data_df[STANDARD_COLUMN_PROGRAM].astype('bool')
    print(f"Data loading error: {e}", file=sys.stderr)


# Determine available metrics from the merged data
AVAILABLE_METRICS = [m for m in METRIC_COLUMNS_CANONICAL.keys() if m in data_df.columns]
if not AVAILABLE_METRICS:
    AVAILABLE_METRICS = list(METRIC_COLUMNS_CANONICAL.keys())


# ---------------------------
# Dash App
# ---------------------------
app: Dash = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "Impact of Audiology Program Presence on 1-3-6 Outcomes"


def make_header() -> dbc.Row:
    return dbc.Row([
        dbc.Col(html.H3("Impact of Audiology Program Presence on 1-3-6 Outcomes"), width=12)
    ], className="mb-3")


def make_controls() -> dbc.Card:
    return dbc.Card([
        dbc.CardHeader("Filters"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Outcome Metric"),
                    dcc.Dropdown(
                        id="metric-dropdown",
                        options=[{"label": m, "value": m} for m in AVAILABLE_METRICS],
                        value=AVAILABLE_METRICS[0] if AVAILABLE_METRICS else None,
                        clearable=False,
                    ),
                ], md=6),
                dbc.Col([
                    dbc.Label("Options"),
                    dbc.Checklist(
                        id="options-checklist",
                        options=[{"label": "Include Georgia", "value": "include_ga"}],
                        value=["include_ga"],
                        inline=True,
                        switch=True,
                    ),
                ], md=6),
            ])
        ])
    ], className="mb-3")


def make_visuals_and_stats() -> dbc.Row:
    return dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("With vs Without Audiology Program"),
            dbc.CardBody([dcc.Graph(id="group-plot")])
        ]), md=6),
        dbc.Col(dbc.Card([
            dbc.CardHeader("Outcome vs Program Presence (Scatter)"),
            dbc.CardBody([dcc.Graph(id="scatter-plot")])
        ]), md=6),
        dbc.Col(dbc.Card([
            dbc.CardHeader("U.S. Choropleth (optional)"),
            dbc.CardBody([dcc.Graph(id="choropleth")])
        ]), md=12, className="mt-3"),
        dbc.Col(dbc.Card([
            dbc.CardHeader("Statistical Results"),
            dbc.CardBody([
                html.Div(id="stats-summary", style={"whiteSpace": "pre-wrap", "fontFamily": "monospace"})
            ])
        ]), md=12, className="mt-3"),
        dbc.Col(dbc.Card([
            dbc.CardHeader("Automated Conclusion"),
            dbc.CardBody([
                html.Div(id="insights-text")
            ])
        ]), md=12, className="mt-3"),
    ])


app.layout = dbc.Container([
    make_header(),
    make_controls(),
    make_visuals_and_stats(),
], fluid=True)


def filter_dataframe(include_ga: bool) -> pd.DataFrame:
    if include_ga:
        return data_df.copy()
    return data_df[data_df[STANDARD_COLUMN_STATE] != "Georgia"].copy()


def make_group_plot(df: pd.DataFrame, metric: str):
    if metric not in df.columns:
        return px.bar(title="Metric not found in data")
    tmp = df.dropna(subset=[metric]).copy()
    tmp["Program"] = np.where(tmp[STANDARD_COLUMN_PROGRAM], "With Program", "Without Program")

    # Box plot for distribution comparison
    fig = px.box(
        tmp,
        x="Program",
        y=metric,
        points="all",
        color="Program",
        title=f"{metric}: Distribution by Program Presence",
        hover_data={STANDARD_COLUMN_STATE: True, STANDARD_COLUMN_PROGRAM: True, metric: ":.2f"},
    )

    # Add mean and median annotations per group
    annotations = []
    for program_flag, group_name in [(True, "With Program"), (False, "Without Program")]:
        vals = tmp[tmp[STANDARD_COLUMN_PROGRAM] == program_flag][metric].dropna()
        if vals.empty:
            continue
        mean_val = vals.mean()
        median_val = vals.median()
        annotations.append(dict(
            x=group_name, y=mean_val, xref='x', yref='y', text=f"Mean: {mean_val:.1f}",
            showarrow=False, yshift=10, font=dict(color="#2c3e50")
        ))
        annotations.append(dict(
            x=group_name, y=median_val, xref='x', yref='y', text=f"Median: {median_val:.1f}",
            showarrow=False, yshift=-10, font=dict(color="#7f8c8d")
        ))
    fig.update_layout(annotations=annotations, showlegend=False, yaxis_title=f"{metric} (%)")
    return fig


def make_scatter_plot(df: pd.DataFrame, metric: str):
    if metric not in df.columns:
        return px.scatter(title="Metric not found in data")
    tmp = df.dropna(subset=[metric]).copy()
    tmp["Program"] = np.where(tmp[STANDARD_COLUMN_PROGRAM], 1, 0)
    fig = px.strip(
        tmp,
        x="Program",
        y=metric,
        color=tmp[STANDARD_COLUMN_PROGRAM].map({True: "With Program", False: "Without Program"}),
        hover_name=STANDARD_COLUMN_STATE,
        hover_data={STANDARD_COLUMN_PROGRAM: True, metric: ":.2f"},
        title=f"{metric} vs Program Presence (0/1)",
    )
    fig.update_layout(xaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=["Without", "With"]))
    fig.update_yaxes(title=f"{metric} (%)")
    fig.update_xaxes(title="Program Presence (Binary)")
    return fig


def make_choropleth(df: pd.DataFrame, metric: str):
    if (metric not in df.columns) or ("State_Code" not in df.columns):
        return px.choropleth(title="Choropleth unavailable: missing metric or state codes")
    tmp = df.dropna(subset=[metric]).copy()
    if tmp["State_Code"].isna().all():
        return px.choropleth(title="Choropleth unavailable: state code mapping failed")
    fig = px.choropleth(
        tmp,
        locations="State_Code",
        locationmode="USA-states",
        color=metric,
        color_continuous_scale="Blues",
        scope="usa",
        hover_name=STANDARD_COLUMN_STATE,
        hover_data={STANDARD_COLUMN_PROGRAM: True, metric: ":.2f"},
        title=f"{metric} by State",
    )
    fig.update_coloraxes(colorbar_title=f"{metric} (%)")
    return fig


@app.callback(
    Output("group-plot", "figure"),
    Output("scatter-plot", "figure"),
    Output("choropleth", "figure"),
    Output("stats-summary", "children"),
    Output("insights-text", "children"),
    Input("metric-dropdown", "value"),
    Input("options-checklist", "value"),
)
def update_dashboard(selected_metric: str, options_values):
    include_ga = isinstance(options_values, list) and ("include_ga" in options_values)
    filtered = filter_dataframe(include_ga)

    group_fig = make_group_plot(filtered, selected_metric)
    scatter_fig = make_scatter_plot(filtered, selected_metric)
    choropleth_fig = make_choropleth(filtered, selected_metric)

    stats_dict = compute_group_stats(filtered, selected_metric) if (selected_metric and selected_metric in filtered.columns) else {}
    stats_lines = []
    if stats_dict:
        stats_lines.append(f"Metric: {selected_metric}")
        stats_lines.append(f"N (With Program): {stats_dict['n_with']}")
        stats_lines.append(f"N (Without Program): {stats_dict['n_without']}")
        stats_lines.append(f"Mean +/- SD (With): {stats_dict['mean_with']:.2f} +/- {stats_dict['sd_with'] if not pd.isna(stats_dict['sd_with']) else np.nan:.2f}")
        stats_lines.append(f"Mean +/- SD (Without): {stats_dict['mean_without']:.2f} +/- {stats_dict['sd_without'] if not pd.isna(stats_dict['sd_without']) else np.nan:.2f}")
        stats_lines.append(f"Median (With): {stats_dict['median_with']:.2f}")
        stats_lines.append(f"Median (Without): {stats_dict['median_without']:.2f}")
        stats_lines.append(f"Welch t-test: t={stats_dict['t_stat'] if not pd.isna(stats_dict['t_stat']) else np.nan:.3f}, p={stats_dict['t_pvalue'] if not pd.isna(stats_dict['t_pvalue']) else np.nan:.3g}")
        stats_lines.append(f"Mann-Whitney U: U={stats_dict['mw_stat'] if not pd.isna(stats_dict['mw_stat']) else np.nan:.3f}, p={stats_dict['mw_pvalue'] if not pd.isna(stats_dict['mw_pvalue']) else np.nan:.3g}")
        stats_lines.append(f"Cohen's d: {stats_dict['cohens_d'] if not pd.isna(stats_dict['cohens_d']) else np.nan:.3f}")
        if HAS_STATSMODELS:
            stats_lines.append(f"Regression (Outcome ~ Program): coef={stats_dict['reg_coef'] if not pd.isna(stats_dict['reg_coef']) else np.nan:.3f}, p={stats_dict['reg_pvalue'] if not pd.isna(stats_dict['reg_pvalue']) else np.nan:.3g}")
        else:
            stats_lines.append("Regression: statsmodels not available")
    stats_text = "\n".join(stats_lines) if stats_lines else "No statistics available."

    insight_text = generate_conclusion(stats_dict, selected_metric, include_ga) if stats_dict else ""
    return group_fig, scatter_fig, choropleth_fig, stats_text, insight_text


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)


