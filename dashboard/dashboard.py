"""
Stablecoin Route Optimization Dashboard - Modular Version

Clean, modular dashboard using actual data structure
Integrates Prometheus metrics (from API `/metrics`) into the existing layout.
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import os
from pathlib import Path
import requests
import re
import json
import sys

# Add project root to the Python path
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
from ai.ai_integration import get_transaction_insights

# Import our custom modules
# NOTE: added build_baseline_df for fallback baseline auto-detection
from metrics_calculator import MetricsCalculator, get_comparison_metrics, build_baseline_df
from data_loader import DataLoader, apply_filters, get_filter_options

# Page config
st.set_page_config(
    page_title="Stablecoin Route Optimization",
    page_icon="💱",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.big-metric {font-size: 2em; font-weight: bold; color: #1f77b4;}
.alert-green {background: #d4edda; padding: 15px; border-left: 4px solid #28a745; margin: 10px 0;}
.alert-red {background: #f8d7da; padding: 15px; border-left: 4px solid #dc3545; margin: 10px 0;}
.alert-blue {background: #d1ecf1; padding: 15px; border-left: 4px solid #0c5460; margin: 10px 0;}
</style>
""", unsafe_allow_html=True)

# ---------------- PROMETHEUS METRICS HELPERS ----------------
import re
import requests
import os

API_BASE = os.environ.get("API_BASE", "http://localhost:8000")
METRICS_URL = f"{API_BASE}/metrics"

_METRIC_RE = re.compile(r"^([a-zA-Z0-9_:]+)(?:\{[^}]*\})?\s+([0-9.eE+\-]+)$")

@st.cache_data(show_spinner=False, ttl=60)
def fetch_metrics_text(url: str = METRICS_URL, timeout: int = 20) -> str | None:
    """Fetch /metrics text from Prometheus-compatible endpoint."""
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            return r.text
    except Exception as e:
        # do not raise here; dashboard will note metrics not available
        return None
    return None


def parse_prometheus_text(text: str) -> dict:
    """
    Minimal parser to extract key Prometheus metrics we care about.
    """
    metrics = {}
    if not text:
        return metrics

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _METRIC_RE.match(line)
        if not m:
            continue
        name, val = m.groups()
        try:
            metrics[name] = float(val)
        except ValueError:
            continue

    out = {}
    if "batches_started_total" in metrics:
        out["batches_started_total"] = int(metrics["batches_started_total"])
    if "batches_in_progress" in metrics:
        out["batches_in_progress"] = int(metrics["batches_in_progress"])

    sum_k = "batch_processing_time_seconds_sum"
    cnt_k = "batch_processing_time_seconds_count"
    if sum_k in metrics and cnt_k in metrics and metrics[cnt_k] > 0:
        out["avg_processing_seconds"] = metrics[sum_k] / metrics[cnt_k]
        out["processed_batches_count"] = int(metrics[cnt_k])

    return out

# ----------------=== SIDEBAR ===----------------

# st.sidebar.title("📁 Configuration")

# # Auto-detect latest batch directory under ./config if present
def _latest_batch_dir(base: str = "./config") -> str:
    try:
        base_path = Path(base)
        if not base_path.exists():
            return base
        # find subdirs with optimization_results.csv
        candidates = []
        for p in base_path.iterdir():
            if p.is_dir():
                if (p / "optimization_results.csv").exists() or (p / "optimization_results_lp.csv").exists() or (p / "optimization_results_mip.csv").exists():
                    candidates.append((p.stat().st_mtime, str(p)))
        if not candidates:
            return base
        candidates.sort(reverse=True)
        return candidates[0][1]
    except Exception:
        return base

_default_dir = _latest_batch_dir("./config")

# Data directory
data_dir = st.sidebar.text_input(
    "Data Directory",
    value=_default_dir,
    help="Path to directory with CSV files"
)

# Prometheus metrics quick panel in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Prometheus Metrics")

# prom_metrics default so later code can safely read it
prom_metrics = {}
metrics_text = fetch_metrics_text()
if metrics_text:
    prom_metrics = parse_prometheus_text(metrics_text)
    if prom_metrics:
        st.sidebar.metric("Batches started", prom_metrics.get("batches_started_total", 0))
        st.sidebar.metric("Batches in progress", prom_metrics.get("batches_in_progress", 0))
        avg = prom_metrics.get("avg_processing_seconds")
        st.sidebar.metric("Avg processing (s)", f"{avg:.2f}" if avg else "—")
        # raw small snippet
        snippet = "\n".join(metrics_text.splitlines()[:20])
        st.sidebar.code(snippet)
    else:
        st.sidebar.info("Prometheus metrics not available.")
else:
    st.sidebar.info("Prometheus metrics not available. Ensure API is running.")

# Initialize data loader
loader = DataLoader(data_dir)

# Check file status
# st.sidebar.markdown("### 📊 Data Files")
# file_status = loader.check_files_exist()
# for name, exists in file_status.items():
#     icon = "✅" if exists else "❌"
#     st.sidebar.markdown(f"{icon} {name.title()}")

# Load data
optimized_df = loader.load_optimized_data()
# try loader method first, fallback to auto-built baseline using build_baseline_df
baseline_df = loader.load_baseline_data() if hasattr(loader, "load_baseline_data") else None
if baseline_df is None:
    try:
        # Build baseline automatically from provided data_dir
        baseline_guess = build_baseline_df(data_dir)
        if baseline_guess is not None:
            baseline_df = baseline_guess
    except Exception:
        baseline_df = None

if optimized_df is None:
    st.error("⚠️ No optimization results found! Please check your data directory.")
    st.stop()

# # Analysis mode
# st.sidebar.markdown("---")
# st.sidebar.title("🎯 Analysis Mode")

# if baseline_df is not None:
#     analysis_mode = st.sidebar.radio(
#         "Mode",
#         ["Current Performance", "Optimization Comparison"],
#         help="Compare before/after or view current results"
#     )
# else:
#     analysis_mode = "Current Performance"
#     st.sidebar.info("Only optimized data available")

# Filters
st.sidebar.markdown("---")
st.sidebar.title("🎛️ Filters")

filter_options = get_filter_options(optimized_df)

# Business type filter
if 'business_types' in filter_options:
    selected_business_types = st.sidebar.multiselect(
        "Business Type",
        filter_options['business_types'],
        default=['All']
    )
else:
    selected_business_types = ['All']

# Region filter
if 'regions' in filter_options:
    selected_regions = st.sidebar.multiselect(
        "Region",
        filter_options['regions'],
        default=['All']
    )
else:
    selected_regions = ['All']

# Urgency filter
if 'urgency_levels' in filter_options:
    selected_urgency = st.sidebar.multiselect(
        "Urgency Level",
        filter_options['urgency_levels'],
        default=['All']
    )
else:
    selected_urgency = ['All']

# User tier filter
if 'user_tiers' in filter_options:
    selected_tiers = st.sidebar.multiselect(
        "User Tier",
        filter_options['user_tiers'],
        default=['All']
    )
else:
    selected_tiers = ['All']

# Apply filters
filtered_df = apply_filters(
    optimized_df,
    business_types=selected_business_types if 'All' not in selected_business_types else None,
    regions=selected_regions if 'All' not in selected_regions else None,
    urgency_levels=selected_urgency if 'All' not in selected_urgency else None,
    user_tiers=selected_tiers if 'All' not in selected_tiers else None
)

if baseline_df is not None:
    filtered_baseline = apply_filters(
        baseline_df,
        business_types=selected_business_types if 'All' not in selected_business_types else None,
        regions=selected_regions if 'All' not in selected_regions else None,
        urgency_levels=selected_urgency if 'All' not in selected_urgency else None,
        user_tiers=selected_tiers if 'All' not in selected_tiers else None
    )
else:
    filtered_baseline = None

# ==================== MAIN DASHBOARD ====================

st.title("💱 Stablecoin Route Optimization Dashboard")
st.markdown(f"**Transactions:** {len(filtered_df):,} | **Mode:** {analysis_mode}")
st.markdown("---")

# Create calculator
calc = MetricsCalculator(filtered_df, filtered_baseline)
metrics = calc.get_summary_metrics()

# Show a compact metrics row that includes Prometheus-derived numbers if available
col_prom1, col_prom2, col_prom3, col_prom4 = st.columns(4)
with col_prom1:
    st.metric("Total Volume", f"${metrics['total_volume_usd']/1e6:.2f}M")
with col_prom2:
    st.metric("Avg Cost", f"{metrics['avg_cost_bps']:.2f} BPS")
with col_prom3:
    st.metric("Success Rate", f"{metrics['success_rate_pct']:.1f}%")
with col_prom4:
    # show batches in progress from Prometheus if available to correlate dashboard with backend
    bip = prom_metrics.get("batches_in_progress")
    if bip is not None:
        st.metric("Batches In Progress (backend)", f"{bip}")
    else:
        st.metric("Transactions", f"{metrics['total_transactions']:,}")

# ==================== TABS ====================

if analysis_mode == "Optimization Comparison" and filtered_baseline is not None:
    tabs = st.tabs([
        "🎯 Optimization Impact",
        "📊 Overview",
        "💰 Cost Analysis",
        "⚡ Performance",
        "📈 Segmentation"
    ])
else:
    tabs = st.tabs([
        "📊 Overview",
        "💰 Cost Analysis",
        "⚡ Performance",
        "📈 Segmentation"
    ])

# ==================== TAB 1: OPTIMIZATION IMPACT ====================

if analysis_mode == "Optimization Comparison" and filtered_baseline is not None:
    with tabs[0]:
        st.header("🎯 Optimization Impact")
        
        # Use get_comparison_metrics helper to compute before/after metrics safely
        try:
            comparison = get_comparison_metrics(filtered_baseline, filtered_df) or {}
        except Exception as e:
            st.warning(f"Could not compute comparison metrics: {e}")
            comparison = {}

        # Extract before/after metrics; fallback to older calculations if missing
        metrics_baseline = comparison.get('before', {})
        metrics_after = comparison.get('after', {})

        # If get_comparison_metrics returned nothing, fall back to previous approach for baseline & after
        if not metrics_baseline:
            try:
                calc_baseline = MetricsCalculator(filtered_baseline)
                metrics_baseline = calc_baseline.get_summary_metrics()
            except Exception:
                metrics_baseline = {}

        if not metrics_after:
            try:
                calc_after = MetricsCalculator(filtered_df)
                metrics_after = calc_after.get_summary_metrics()
            except Exception:
                metrics_after = {}

        # For backward compatibility keep `metrics` (overall) as the optimized summary used elsewhere
        # but prefer metrics_after for values shown in this tab when available
        optimized_metrics_local = metrics_after if metrics_after else metrics

        # Key improvements
        st.markdown("### 💡 Key Improvements")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # cost improvement: prefer comparison improvements if present, else use avg improvement field
            cost_improvement_bps = comparison.get('improvements', {}).get('cost_bps')
            if cost_improvement_bps is None:
                cost_improvement_bps = optimized_metrics_local.get('avg_cost_improvement_bps', 0.0)
            base_cost = metrics_baseline.get('avg_cost_bps', 0.0)
            pct = (cost_improvement_bps / base_cost * 100) if base_cost else 0.0
            st.metric("Cost Reduction", f"{cost_improvement_bps:.1f} BPS", f"{pct:.1f}%")
        
        with col2:
            savings = optimized_metrics_local.get('total_savings_usd', 0.0)
            st.metric("Total Savings", f"${savings:,.2f}", "Saved")
        
        with col3:
            success_rate_after = optimized_metrics_local.get('success_rate_pct', metrics.get('success_rate_pct', 0.0))
            success_rate_before = metrics_baseline.get('success_rate_pct', 0.0)
            st.metric("Success Rate", f"{success_rate_after:.1f}%", f"{success_rate_after - success_rate_before:.1f}pp")
        
        with col4:
            avg_routes = optimized_metrics_local.get('avg_routes', metrics.get('avg_routes', 0.0))
            st.metric("Avg Routes", f"{avg_routes:.1f}", "Optimized")
        
        # Comparison charts
        st.markdown("---")
        st.markdown("### 📊 Before vs After")
        
        col1, col2 = st.columns(2)
        
        # Helper function to safely read value
        def _safe_val(d, key, fallback=0.0):
            try:
                v = d.get(key, fallback)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return fallback
                return float(v)
            except Exception:
                return float(fallback)

        with col1:
            # Cost comparison
            before_cost = _safe_val(metrics_baseline, 'avg_cost_bps', 0.0)
            after_cost = _safe_val(optimized_metrics_local, 'avg_cost_bps', 0.0)
            comparison_data = pd.DataFrame({
                'Metric': ['Before', 'After'],
                'Cost (BPS)': [before_cost, after_cost]
            })
            fig = px.bar(
                comparison_data, x='Metric', y='Cost (BPS)',
                title='Average Cost Comparison',
                color='Metric',
                color_discrete_map={'Before': '#ff7f0e', 'After': '#2ca02c'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Time comparison (minutes)
            before_time_min = _safe_val(metrics_baseline, 'avg_settlement_time_sec', 0.0) / 60.0
            after_time_min = _safe_val(optimized_metrics_local, 'avg_settlement_time_sec', 0.0) / 60.0
            comparison_time = pd.DataFrame({
                'Metric': ['Before', 'After'],
                'Time (min)': [before_time_min, after_time_min]
            })
            fig = px.bar(
                comparison_time, x='Metric', y='Time (min)',
                title='Average Settlement Time Comparison',
                color='Metric',
                color_discrete_map={'Before': '#ff7f0e', 'After': '#2ca02c'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    overview_tab = tabs[1]
else:
    overview_tab = tabs[0]

# ==================== OVERVIEW TAB ====================

with overview_tab:
    st.header("📊 Performance Overview")
    
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Volume",
            f"${metrics['total_volume_usd']/1e6:.2f}M"
        )
    
    with col2:
        st.metric(
            "Avg Cost",
            f"{metrics['avg_cost_bps']:.2f} BPS"
        )
    
    with col3:
        st.metric(
            "Success Rate",
            f"{metrics['success_rate_pct']:.1f}%"
        )
    
    with col4:
        st.metric(
            "Total Fees",
            f"${metrics['total_fees_usd']:,.2f}"
        )
    
    # Second row
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric(
            "Transactions",
            f"{metrics['total_transactions']:,}"
        )
    
    with col6:
        st.metric(
            "Avg Time",
            f"{metrics['avg_settlement_time_sec']/60:.1f} min"
        )
    
    with col7:
        st.metric(
            "Avg Routes",
            f"{metrics['avg_routes']:.1f}"
        )
    
    with col8:
        st.metric(
            "Constraints Met",
            f"{metrics['constraints_satisfied_pct']:.1f}%"
        )
    
    # Scores
    st.markdown("---")
    st.markdown("### 📈 Optimization Scores")
    
    col1, col2, col3, col4 = st.columns(4)
    scores = metrics['scores']
    
    with col1:
        st.metric("Cost Score", f"{scores.get('cost_score', 0):.1f}")
    with col2:
        st.metric("Speed Score", f"{scores.get('speed_score', 0):.1f}")
    with col3:
        st.metric("Risk Score", f"{scores.get('risk_score', 0):.1f}")
    with col4:
        st.metric("Total Score", f"{scores.get('total_score', 0):.1f}")

# ==================== COST ANALYSIS TAB ====================

cost_tab_idx = 2 if analysis_mode == "Optimization Comparison" and filtered_baseline is not None else 1

with tabs[cost_tab_idx]:
    st.header("💰 Cost Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Cost distribution
        if 'total_cost_bps' in filtered_df.columns:
            fig = px.histogram(filtered_df, x='total_cost_bps',
                             title='Cost Distribution (BPS)',
                             nbins=30)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Cost by business type
        if 'original_type' in filtered_df.columns:
            cost_by_type = filtered_df.groupby('original_type')['total_cost_bps'].mean().sort_values()
            fig = px.bar(x=cost_by_type.values, y=cost_by_type.index,
                        orientation='h',
                        title='Average Cost by Business Type',
                        labels={'x': 'Avg Cost (BPS)', 'y': 'Business Type'})
            st.plotly_chart(fig, use_container_width=True)
    
    # Cost improvement
    if 'cost_improvement_bps' in filtered_df.columns:
        st.markdown("---")
        st.markdown("### 💰 Cost Improvement Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(filtered_df, x='cost_improvement_bps',
                             title='Cost Improvement Distribution',
                             nbins=30)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Top improved transactions
            top_improved = filtered_df.nlargest(10, 'cost_improvement_bps')[
                ['transfer_id', 'original_type', 'cost_improvement_bps', 'total_amount_usd']
            ]
            st.markdown("#### Top 10 Cost Reductions")
            st.dataframe(top_improved, use_container_width=True)

# ==================== PERFORMANCE TAB ====================

perf_tab_idx = 3 if analysis_mode == "Optimization Comparison" and filtered_baseline is not None else 2

with tabs[perf_tab_idx]:
    st.header("⚡ Performance Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Time distribution
        if 'total_time_sec' in filtered_df.columns:
            fig = px.box(filtered_df, y='total_time_sec',
                        title='Settlement Time Distribution')
            fig.update_yaxes(title='Time (seconds)')
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Routes distribution
        if 'num_routes' in filtered_df.columns:
            route_dist = filtered_df['num_routes'].value_counts().sort_index()
            fig = px.bar(x=route_dist.index, y=route_dist.values,
                        title='Number of Routes Distribution',
                        labels={'x': 'Number of Routes', 'y': 'Count'})
            st.plotly_chart(fig, use_container_width=True)
    
    # Status breakdown
    if 'status' in filtered_df.columns:
        st.markdown("---")
        st.markdown("### ✅ Optimization Status")
        
        status_counts = filtered_df['status'].value_counts()
        fig = px.pie(values=status_counts.values, names=status_counts.index,
                    title='Optimization Status Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    # Venue performance
    if 'route_1_venue' in filtered_df.columns:
        st.markdown("---")
        st.markdown("### 🏦 Top Venues")
        
        venue_counts = filtered_df['route_1_venue'].value_counts().head(10)
        fig = px.bar(x=venue_counts.values, y=venue_counts.index,
                    orientation='h',
                    title='Top 10 Venues by Usage',
                    labels={'x': 'Transaction Count', 'y': 'Venue'})
        st.plotly_chart(fig, use_container_width=True)

# ==================== SEGMENTATION TAB ====================

seg_tab_idx = 4 if analysis_mode == "Optimization Comparison" and filtered_baseline is not None else 3

with tabs[seg_tab_idx]:
    st.header("📈 Segmentation Analysis")
    
    # By business type
    if 'original_type' in filtered_df.columns:
        st.markdown("### 💼 By Business Type")
        metrics_by_type = calc.get_metrics_by_business_type()
        st.dataframe(metrics_by_type, use_container_width=True)
    
    # By region
    if 'region' in filtered_df.columns:
        st.markdown("---")
        st.markdown("### 🌍 By Region")
        metrics_by_region = calc.get_metrics_by_region()
        st.dataframe(metrics_by_region, use_container_width=True)
        
        # Regional volume chart
        fig = px.bar(metrics_by_region, x='Region', y='Volume ($)',
                    title='Transaction Volume by Region')
        st.plotly_chart(fig, use_container_width=True)
    
    # By urgency
    if 'urgency_level' in filtered_df.columns:
        st.markdown("---")
        st.markdown("### ⚡ By Urgency Level")
        metrics_by_urgency = calc.get_metrics_by_urgency()
        st.dataframe(metrics_by_urgency, use_container_width=True)
    
    # By user tier
    if 'user_tier' in filtered_df.columns:
        st.markdown("---")
        st.markdown("### 👥 By User Tier")
        metrics_by_tier = calc.get_metrics_by_user_tier()
        st.dataframe(metrics_by_tier, use_container_width=True)
# Run insights
if st.button("🔍 Generate AI Insights"):
    with st.spinner("Analyzing transactions..."):
        try:
            insights = get_transaction_insights()
            st.success("✅ Insights generated successfully!")

            # Layout columns
            col1, col2, col3 = st.columns(3)

            # Summary
            with col1:
                st.markdown("### 📋 Summary")
                st.write(insights.get("summary", "No summary available."))

            # Anomalies
            with col2:
                st.markdown("### ⚠️ Anomalies")
                anomalies = insights.get("anomalies", [])
                if anomalies:
                    for a in anomalies:
                        st.markdown(f"- {a}")
                else:
                    st.write("No anomalies detected.")

            # Recommendations
            with col3:
                st.markdown("### 💡 Recommendations")
                recs = insights.get("recommendations", [])
                if recs:
                    for r in recs:
                        st.markdown(f"- {r}")
                else:
                    st.write("No recommendations provided.")

        except Exception as e:
            st.error(f"Error generating insights: {e}")

# Footer
st.markdown("---")
st.markdown(f"*Dashboard v3.0 - Modular Architecture | Transactions: {len(filtered_df):,}*")
