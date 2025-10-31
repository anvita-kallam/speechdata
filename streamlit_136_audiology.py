import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
from scipy import stats

import streamlit as st

try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

import base64


# ---------------------------
# Configuration and constants
# ---------------------------
# Use the script directory so relative CSV paths work locally and in Streamlit Cloud
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTCOME_FILE = os.path.join(WORKSPACE_DIR, "ATL Speech - Correlation Workflow - 136 Outcome.csv")
PROGRAM_FILE = os.path.join(WORKSPACE_DIR, "ATL Speech - Correlation Workflow - Presence of Aud Program.csv")
BACKGROUND_IMAGE_FILE = os.path.join(WORKSPACE_DIR, "speechback.png")

STANDARD_COLUMN_STATE = "State"
STANDARD_COLUMN_PROGRAM = "Audiology Program Presence"
METRIC_COLUMNS_CANONICAL = {
    "% Meeting 1": [
        "% Meeting 1", "Percent Meeting 1", "Meeting1", "Pct Meeting 1",
        "% meeting 1", "percent meeting 1", "pct meeting 1"
    ],
    "% Meeting 3": [
        "% Meeting 3", "Percent Meeting 3", "Meeting3", "Pct Meeting 3",
        "% meeting 3", "percent meeting 3", "pct meeting 3"
    ],
    "% Meeting 6": [
        "% Meeting 6", "Percent Meeting 6", "Meeting6", "Pct Meeting 6",
        "% meeting 6", "percent meeting 6", "pct meeting 6"
    ],
}

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

# Fix typos found in the CSV files (e.g., "Noebraska" -> "Nebraska")
STATE_NAME_CORRECTIONS = {
    "noebraska": "nebraska",
    "noevada": "nevada",
    "noew hampshire": "new hampshire",
    "noew jersey": "new jersey",
    "noew mexico": "new mexico",
    "noew york": "new york",
}


def read_csv_safely(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {c: c.strip() for c in df.columns}

    found_state_key = None
    for candidate in [
        "State", "state", "STATE", "State Name", "Jurisdiction", "State/Territory",
        "STATE_NAME", "Location", "Location Name", "State/Area"
    ]:
        if candidate in renamed.values():
            found_state_key = next(k for k, v in renamed.items() if v == candidate)
            renamed[found_state_key] = STANDARD_COLUMN_STATE
            break
    if STANDARD_COLUMN_STATE not in renamed.values():
        for original, clean in renamed.items():
            low = clean.lower()
            if ("state" in low) or (low in {"jurisdiction", "location"}):
                renamed[original] = STANDARD_COLUMN_STATE
                found_state_key = original
                break

    present_cols = set(renamed.values()) | {v.lower() for v in renamed.values()}
    inverse = {v: k for k, v in renamed.items()}
    for canonical, aliases in METRIC_COLUMNS_CANONICAL.items():
        if canonical in present_cols:
            continue
        for alias in aliases:
            if alias in present_cols or alias.lower() in present_cols:
                original = inverse.get(alias, None)
                if original is None:
                    for k, v in renamed.items():
                        if v.lower() == alias.lower():
                            original = k
                            break
                if original is None:
                    continue
                renamed[original] = canonical
                break

    for candidate in [
        "Audiology Program Presence", "Presence of Audiology Program", "Program", "Has Program", "Presence", "Audiology Program",
        "Audiology Presence", "Program Presence", "Has Audiology Program"
    ]:
        if candidate in renamed.values():
            original = next(k for k, v in renamed.items() if v == candidate)
            renamed[original] = STANDARD_COLUMN_PROGRAM
            break

    return df.rename(columns=renamed)


def coerce_program_presence(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    s = series.astype(str).str.strip().str.lower()
    true_values = {"yes", "true", "1", "y", "present", "with", "has", "have", "available"}
    false_values = {"no", "false", "0", "n", "absent", "without", "none", "not available"}
    return s.map(lambda x: True if x in true_values else (False if x in false_values else np.nan))


@st.cache_data(show_spinner=False)
def prepare_data(outcome_path: str, program_path: str) -> pd.DataFrame:
    outcome_df = standardize_columns(read_csv_safely(outcome_path))
    program_df = standardize_columns(read_csv_safely(program_path))

    if STANDARD_COLUMN_PROGRAM not in program_df.columns:
        candidate_cols = [c for c in program_df.columns if c != STANDARD_COLUMN_STATE]
        if len(candidate_cols) > 0:
            program_df[STANDARD_COLUMN_PROGRAM] = coerce_program_presence(program_df[candidate_cols[-1]])

    if STANDARD_COLUMN_STATE not in outcome_df.columns:
        raise KeyError(f"State column not found in outcome data. Columns: {list(outcome_df.columns)}")
    if STANDARD_COLUMN_STATE not in program_df.columns:
        raise KeyError(f"State column not found in program data. Columns: {list(program_df.columns)}")
    if STANDARD_COLUMN_PROGRAM not in program_df.columns:
        raise KeyError("Program presence column not found/derivable in program data.")

    # Normalize state name for robust merging (case-insensitive, trimmed, and fix typos)
    def normalize_state_name(state_name: str) -> str:
        normalized = str(state_name).strip().lower()
        return STATE_NAME_CORRECTIONS.get(normalized, normalized)
    
    outcome_df["_state_key"] = outcome_df[STANDARD_COLUMN_STATE].apply(normalize_state_name)
    program_df["_state_key"] = program_df[STANDARD_COLUMN_STATE].apply(normalize_state_name)

    # Include audiologists per 100k population column
    audiologist_col = None
    for col in program_df.columns:
        if "audiologist" in col.lower() and "100k" in col.lower():
            audiologist_col = col
            break
    
    merge_cols = ["_state_key", STANDARD_COLUMN_PROGRAM]
    if audiologist_col:
        merge_cols.append(audiologist_col)
    
    merged = pd.merge(
        outcome_df,
        program_df[merge_cols],
        on="_state_key",
        how="inner",
        validate="m:1",
    )
    # Replace state column with the original display name and drop helper key
    merged[STANDARD_COLUMN_STATE] = outcome_df[STANDARD_COLUMN_STATE]
    merged = merged.drop(columns=["_state_key"])

    keep_cols = [STANDARD_COLUMN_STATE, STANDARD_COLUMN_PROGRAM] + list(METRIC_COLUMNS_CANONICAL.keys())
    # Preserve Year column if it exists (for aggregation later)
    if "Year" in merged.columns:
        keep_cols.append("Year")
    # Preserve audiologists per 100k column
    if audiologist_col and audiologist_col in merged.columns:
        keep_cols.append(audiologist_col)
        # Standardize column name
        merged = merged.rename(columns={audiologist_col: "Audiologists_per_100k"})
        keep_cols = [c if c != audiologist_col else "Audiologists_per_100k" for c in keep_cols]
        # Convert to numeric
        merged["Audiologists_per_100k"] = pd.to_numeric(merged["Audiologists_per_100k"], errors='coerce')
    existing = [c for c in keep_cols if c in merged.columns]
    merged = merged[existing].copy()

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

    if STANDARD_COLUMN_PROGRAM in merged.columns:
        merged[STANDARD_COLUMN_PROGRAM] = coerce_program_presence(merged[STANDARD_COLUMN_PROGRAM])

    merged = merged.dropna(subset=[STANDARD_COLUMN_STATE, STANDARD_COLUMN_PROGRAM])
    merged[STANDARD_COLUMN_STATE] = merged[STANDARD_COLUMN_STATE].astype(str).str.strip()
    merged["State_Code"] = merged[STANDARD_COLUMN_STATE].map(STATE_TO_USPS)
    return merged


def compute_group_stats(df: pd.DataFrame, metric: str) -> dict:
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

    t_stat, t_pval = np.nan, np.nan
    if with_prog.size > 1 and without_prog.size > 1:
        t_res = stats.ttest_ind(with_prog, without_prog, equal_var=False, nan_policy='omit')
        t_stat, t_pval = float(t_res.statistic), float(t_res.pvalue)

    mw_stat, mw_pval = np.nan, np.nan
    if with_prog.size > 0 and without_prog.size > 0:
        try:
            mw_res = stats.mannwhitneyu(with_prog, without_prog, alternative='two-sided')
            mw_stat, mw_pval = float(mw_res.statistic), float(mw_res.pvalue)
        except ValueError:
            pass

    cohens_d = np.nan
    if with_prog.size > 1 and without_prog.size > 1:
        s1 = with_prog.std(ddof=1)
        s2 = without_prog.std(ddof=1)
        n1 = with_prog.size
        n2 = without_prog.size
        sp = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2)) if (n1 + n2 - 2) > 0 else np.nan
        if sp and sp > 0:
            cohens_d = (with_prog.mean() - without_prog.mean()) / sp

    reg_coef, reg_pvalue, reg_summary = np.nan, np.nan, None
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
        except Exception:
            pass

    stats_dict.update({
        "t_stat": t_stat, "t_pvalue": t_pval,
        "mw_stat": mw_stat, "mw_pvalue": mw_pval,
        "cohens_d": float(cohens_d) if not pd.isna(cohens_d) else np.nan,
        "reg_coef": reg_coef, "reg_pvalue": reg_pvalue, "reg_summary": reg_summary,
    })
    return stats_dict


def generate_conclusion(stats_dict: dict, metric: str, include_georgia: bool) -> str:
    p_value = stats_dict.get("t_pvalue")
    if pd.isna(p_value):
        p_value = stats_dict.get("mw_pvalue")
    if pd.isna(p_value):
        return (
            f"Insufficient data to determine a statistical difference for {metric}. "
            f"Consider reviewing data completeness and quality."
        )
    significant = p_value < 0.05
    base = f"Analysis for {metric}{' (Georgia excluded)' if not include_georgia else ''}: "
    if significant:
        return (
            base
            + f"States with audiology programs show significantly higher {metric} (p = {p_value:.3g}). "
            + "Georgia would likely benefit from establishing a program."
        )
    return base + f"No statistically significant difference found (p = {p_value:.3g}); more data may be needed."


# ===========================
# Streamlit App
# ===========================
st.set_page_config(page_title="Impact of Audiology Program Presence on 1-3-6 Outcomes", layout="wide")

# Ensure no background image is applied
bg_style = ""

# Global CSS styling
st.markdown(
    """
    <style>
    :root {
      --brand: #3a7bd5;
      --brand-dark: #264b96;
      --muted: #6c757d;
      --bg: #ffffff; /* revert to white background */
      --card-bg: #ffffff;
      --accent: #22c55e;
    }
    .stApp {
      background-color: var(--bg);
    }
    h1, h2, h3 {
      color: var(--brand-dark) !important;
    }
    .page-title {
      font-size: 2rem;
      font-weight: 800;
      background: linear-gradient(90deg, var(--brand), var(--accent));
      -webkit-background-clip: text;
      background-clip: text;
      color: transparent;
      margin-bottom: 0.5rem;
    }
    .subtitle {
      color: var(--muted);
      margin-bottom: 1.25rem;
    }
    .section-header {
      padding: 0.6rem 0.9rem;
      border-left: 4px solid var(--brand);
      background: rgba(58, 123, 213, 0.08);
      border-radius: 6px;
      margin: 0.5rem 0 0.75rem 0;
      font-weight: 700;
    }
    .card {
      background: var(--card-bg);
      border-radius: 10px;
      padding: 0.75rem;
      box-shadow: 0 2px 10px rgba(0,0,0,0.05);
      border: 1px solid rgba(38,75,150,0.08);
    }
    .stat-block {
      background: var(--card-bg);
      border-radius: 10px;
      padding: 1rem;
      border: 1px dashed rgba(38,75,150,0.2);
    }
    .note {
      font-size: 0.9rem;
      color: var(--muted);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# No background image injection

st.markdown("<div class='page-title'>Impact of Audiology Program Presence on 1-3-6 Outcomes</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Explore whether audiology programs are associated with improved 1–3–6 outcomes</div>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Filters")
    try:
        data_df = prepare_data(OUTCOME_FILE, PROGRAM_FILE)
        available_metrics = [m for m in METRIC_COLUMNS_CANONICAL.keys() if m in data_df.columns]
        if not available_metrics:
            available_metrics = list(METRIC_COLUMNS_CANONICAL.keys())
    except Exception as e:
        st.error(f"Data loading error: {e}")
        data_df = pd.DataFrame(columns=[STANDARD_COLUMN_STATE, STANDARD_COLUMN_PROGRAM] + list(METRIC_COLUMNS_CANONICAL.keys()))
        available_metrics = [m for m in METRIC_COLUMNS_CANONICAL.keys() if m in data_df.columns]

    selected_metric = st.selectbox("Outcome Metric", options=available_metrics, index=0 if available_metrics else None)
    include_ga = st.checkbox("Include Georgia", value=True)
    with st.expander("Data summary", expanded=False):
        st.write({
            "rows": int(data_df.shape[0]) if not data_df.empty else 0,
            "columns": list(data_df.columns),
        })


def filter_dataframe(df: pd.DataFrame, include_ga: bool) -> pd.DataFrame:
    if include_ga:
        return df.copy()
    return df[df[STANDARD_COLUMN_STATE] != "Georgia"].copy()


if data_df.empty or not selected_metric:
    st.warning("No data available to display. Please verify CSVs and columns.")
    st.stop()

filtered = filter_dataframe(data_df, include_ga)

col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='section-header'>With vs Without Audiology Program</div>", unsafe_allow_html=True)
    if selected_metric in filtered.columns:
        tmp = filtered.dropna(subset=[selected_metric]).copy()
        tmp["Program"] = np.where(tmp[STANDARD_COLUMN_PROGRAM], "With Program", "Without Program")
        fig_box = px.box(
            tmp,
            x="Program",
            y=selected_metric,
            points="all",
            color="Program",
            color_discrete_sequence=px.colors.sequential.Viridis,
            hover_data={STANDARD_COLUMN_STATE: True, STANDARD_COLUMN_PROGRAM: True, selected_metric: ":.2f"},
        )
        fig_box.update_layout(margin=dict(t=10, b=20, l=10, r=10))
        annotations = []
        for flag, label in [(True, "With Program"), (False, "Without Program")]:
            vals = tmp[tmp[STANDARD_COLUMN_PROGRAM] == flag][selected_metric].dropna()
            if vals.empty:
                continue
            annotations.append(dict(x=label, y=vals.mean(), xref='x', yref='y', text=f"Mean: {vals.mean():.1f}", showarrow=False, yshift=10))
            annotations.append(dict(x=label, y=vals.median(), xref='x', yref='y', text=f"Median: {vals.median():.1f}", showarrow=False, yshift=-10))
        fig_box.update_layout(annotations=annotations, showlegend=False, yaxis_title=f"{selected_metric} (%)")
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.plotly_chart(fig_box, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Selected metric not found in data.")

with col2:
    st.markdown("<div class='section-header'>Outcome vs Program Presence (Binary)</div>", unsafe_allow_html=True)
    if selected_metric in filtered.columns:
        tmp = filtered.dropna(subset=[selected_metric]).copy()
        tmp["Program"] = np.where(tmp[STANDARD_COLUMN_PROGRAM], 1, 0)
        fig_strip = px.strip(
            tmp, x="Program", y=selected_metric,
            color=tmp[STANDARD_COLUMN_PROGRAM].map({True: "With Program", False: "Without Program"}),
            color_discrete_sequence=px.colors.sequential.Viridis,
            hover_name=STANDARD_COLUMN_STATE,
            hover_data={STANDARD_COLUMN_PROGRAM: True, selected_metric: ":.2f"},
        )
        fig_strip.update_layout(xaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=["Without", "With"]))
        fig_strip.update_yaxes(title=f"{selected_metric} (%)")
        fig_strip.update_xaxes(title="Program Presence (Binary)")
        fig_strip.update_layout(margin=dict(t=10, b=20, l=10, r=10))
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.plotly_chart(fig_strip, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Selected metric not found in data.")

st.markdown("<div class='section-header'>U.S. Choropleth</div>", unsafe_allow_html=True)
if (selected_metric in filtered.columns) and ("State_Code" in filtered.columns) and not filtered["State_Code"].isna().all():
    tmp = filtered.dropna(subset=[selected_metric, "State_Code"]).copy()
    # Aggregate by state: take the most recent value (or average if no Year column)
    if "Year" in tmp.columns:
        tmp_agg = tmp.sort_values("Year", ascending=False).groupby("State_Code").first().reset_index()
    else:
        # If no Year column, take mean per state
        tmp_agg = tmp.groupby("State_Code", as_index=False).agg({
            selected_metric: 'mean',
            STANDARD_COLUMN_STATE: 'first',
            STANDARD_COLUMN_PROGRAM: 'first',  # Should be same for all rows of same state
        })
    
    # Calculate global min/max across all metrics for consistent color scale
    global_min = np.inf
    global_max = -np.inf
    for metric in METRIC_COLUMNS_CANONICAL.keys():
        if metric in filtered.columns:
            metric_data = filtered.dropna(subset=[metric, "State_Code"])
            if "Year" in metric_data.columns:
                metric_agg = metric_data.sort_values("Year", ascending=False).groupby("State_Code").first()[metric]
            else:
                metric_agg = metric_data.groupby("State_Code")[metric].mean()
            if len(metric_agg) > 0:
                global_min = min(global_min, float(metric_agg.min()))
                global_max = max(global_max, float(metric_agg.max()))
    
    # Use consistent color range across all metrics
    if global_min < np.inf and global_max > -np.inf:
        color_range = [global_min, global_max]
    else:
        color_range = None
    
    fig_map = px.choropleth(
        tmp_agg,
        locations="State_Code",
        locationmode="USA-states",
        color=selected_metric,
        color_continuous_scale=px.colors.sequential.Viridis[::-1],
        range_color=color_range,
        scope="usa",
        hover_name=STANDARD_COLUMN_STATE,
        hover_data={STANDARD_COLUMN_PROGRAM: True, selected_metric: ":.2f"},
    )
    fig_map.update_coloraxes(colorbar_title=f"{selected_metric} (%)")
    fig_map.update_layout(margin=dict(t=10, b=20, l=10, r=10))
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.plotly_chart(fig_map, width='stretch')
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("Choropleth unavailable: missing state codes or metric.")

# Scattergram: Selected Outcome vs Audiologists per 100k
if "Audiologists_per_100k" in filtered.columns and selected_metric in filtered.columns:
    st.markdown("<div class='section-header'>Outcome vs Audiologists per 100k Population</div>", unsafe_allow_html=True)
    
    tmp = filtered.dropna(subset=[selected_metric, "Audiologists_per_100k"]).copy()
    if not tmp.empty:
        # Aggregate by state if multiple years exist (take most recent year)
        if "Year" in tmp.columns:
            tmp = tmp.sort_values("Year", ascending=False).groupby(STANDARD_COLUMN_STATE).agg({
                selected_metric: 'first',
                "Audiologists_per_100k": 'first',
                STANDARD_COLUMN_PROGRAM: 'first',
            }).reset_index()
        else:
            tmp = tmp.groupby(STANDARD_COLUMN_STATE, as_index=False).agg({
                selected_metric: 'mean',
                "Audiologists_per_100k": 'first',
                STANDARD_COLUMN_PROGRAM: 'first',
            })
        
        fig_scatter = px.scatter(
            tmp,
            x="Audiologists_per_100k",
            y=selected_metric,
            color=tmp[STANDARD_COLUMN_PROGRAM].map({True: "With Program", False: "Without Program"}),
            color_discrete_sequence=px.colors.sequential.Viridis,
            hover_name=STANDARD_COLUMN_STATE,
            hover_data={STANDARD_COLUMN_PROGRAM: True, selected_metric: ":.2f", "Audiologists_per_100k": ":.1f"},
        )
        
        # Remove the program presence legend items
        fig_scatter.update_traces(showlegend=False, selector=dict(type='scatter', mode='markers'))
        
        # Add single OLS trendline for all data (not separate by program)
        if tmp.shape[0] >= 3:
            x_vals = tmp["Audiologists_per_100k"].values
            y_vals = tmp[selected_metric].values
            # Simple linear regression
            x_mean = np.mean(x_vals)
            y_mean = np.mean(y_vals)
            numerator = np.sum((x_vals - x_mean) * (y_vals - y_mean))
            denominator = np.sum((x_vals - x_mean) ** 2)
            if denominator != 0:
                slope = numerator / denominator
                intercept = y_mean - slope * x_mean
                # Generate trendline points
                x_trend = np.linspace(x_vals.min(), x_vals.max(), 100)
                y_trend = intercept + slope * x_trend
                fig_scatter.add_scatter(
                    x=x_trend,
                    y=y_trend,
                    mode='lines',
                    name='OLS Trendline',
                    line=dict(color='rgba(68, 1, 84, 0.6)', width=2, dash='dash'),
                    showlegend=True,
                )
        
        fig_scatter.update_layout(
            margin=dict(t=10, b=20, l=10, r=10),
            xaxis_title="Audiologists per 100k Population",
            yaxis_title=f"{selected_metric} (%)",
        )
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.plotly_chart(fig_scatter, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info(f"No data available for {selected_metric} vs Audiologists per 100k.")
elif "Audiologists_per_100k" not in filtered.columns:
    st.info("Audiologists per 100k data not available.")


st.markdown("<div class='section-header'>Statistical Results</div>", unsafe_allow_html=True)
stats_dict = compute_group_stats(filtered, selected_metric)
stats_lines = []
stats_lines.append(f"Metric: {selected_metric}")
stats_lines.append(f"N (With Program): {stats_dict['n_with']}")
stats_lines.append(f"N (Without Program): {stats_dict['n_without']}")
stats_lines.append(f"Mean +/- SD (With): {stats_dict['mean_with'] if not pd.isna(stats_dict['mean_with']) else np.nan:.2f} +/- {stats_dict['sd_with'] if not pd.isna(stats_dict['sd_with']) else np.nan:.2f}")
stats_lines.append(f"Mean +/- SD (Without): {stats_dict['mean_without'] if not pd.isna(stats_dict['mean_without']) else np.nan:.2f} +/- {stats_dict['sd_without'] if not pd.isna(stats_dict['sd_without']) else np.nan:.2f}")
stats_lines.append(f"Median (With): {stats_dict['median_with'] if not pd.isna(stats_dict['median_with']) else np.nan:.2f}")
stats_lines.append(f"Median (Without): {stats_dict['median_without'] if not pd.isna(stats_dict['median_without']) else np.nan:.2f}")
stats_lines.append(f"Welch t-test: t={stats_dict['t_stat'] if not pd.isna(stats_dict['t_stat']) else np.nan:.3f}, p={stats_dict['t_pvalue'] if not pd.isna(stats_dict['t_pvalue']) else np.nan:.3g}")
stats_lines.append(f"Mann-Whitney U: U={stats_dict['mw_stat'] if not pd.isna(stats_dict['mw_stat']) else np.nan:.3f}, p={stats_dict['mw_pvalue'] if not pd.isna(stats_dict['mw_pvalue']) else np.nan:.3g}")
stats_lines.append(f"Cohen's d: {stats_dict['cohens_d'] if not pd.isna(stats_dict['cohens_d']) else np.nan:.3f}")
if HAS_STATSMODELS:
    stats_lines.append(f"Regression (Outcome ~ Program): coef={stats_dict['reg_coef'] if not pd.isna(stats_dict['reg_coef']) else np.nan:.3f}, p={stats_dict['reg_pvalue'] if not pd.isna(stats_dict['reg_pvalue']) else np.nan:.3g}")
else:
    stats_lines.append("Regression: statsmodels not available")

st.markdown("<div class='stat-block'>", unsafe_allow_html=True)
st.code("\n".join(stats_lines))
st.markdown("</div>", unsafe_allow_html=True)

# Explanations of statistical tests
st.markdown(
    """
    - **Welch t-test**: Compares group means allowing unequal variances. A small p-value (< 0.05) suggests a statistically significant difference in average outcomes between states with and without programs.
    - **Mann–Whitney U test**: Non-parametric test comparing the distributions (medians/ranks). Useful when data are not normally distributed or have outliers. A small p-value indicates a distributional shift between groups.
    - **Cohen’s d**: Standardized effect size of the mean difference. Rough guide: 0.2 small, 0.5 medium, 0.8 large.
    - **Regression (Outcome ~ Program)**: Linear model where the coefficient on Program estimates the average percentage-point difference associated with having a program, controlling for the intercept. The p-value tests whether that coefficient differs from zero.
    """
)

st.subheader("Conclusion")
st.write(generate_conclusion(stats_dict, selected_metric, include_ga))


