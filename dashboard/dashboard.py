"""
Stablecoin Route Optimization Dashboard - Optimized Version

Key improvements:
- Centralized initialization flow
- Better error handling and fallbacks
- Cleaner separation of concerns
- Improved caching strategy
- Backward compatible
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
import importlib
import traceback
import subprocess
import time
from io import StringIO
import logging
import numpy as np

# ==================== CONFIGURATION ====================

# Logger setup
logger = logging.getLogger("dashboard")
if not logger.handlers:
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s: %(message)s"
    )

# Path configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
CONFIG_DIR = os.path.join(ROOT_DIR, "config")
CONFIG_DIR = os.path.abspath(CONFIG_DIR)

# Add to Python path
sys.path.insert(0, ROOT_DIR)

# API configuration
API_BASE = os.environ.get("API_BASE", "https://stablecoin-optimizer.onrender.com")
METRICS_URL = f"{API_BASE}/metrics"

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

# ==================== UTILITY FUNCTIONS ====================

def _latest_batch_dir(base: str = "./config") -> str:
    """Auto-detect latest batch directory with optimization results."""
    try:
        base_path = Path(base)
        if not base_path.exists():
            return base
        
        candidates = []
        for p in base_path.iterdir():
            if p.is_dir():
                result_files = [
                    "optimization_results.csv",
                    "optimization_results_lp.csv",
                    "optimization_results_mip.csv"
                ]
                if any((p / f).exists() for f in result_files):
                    candidates.append((p.stat().st_mtime, str(p)))
        
        if not candidates:
            return base
        
        candidates.sort(reverse=True)
        return candidates[0][1]
    except Exception as e:
        logger.warning(f"Error detecting latest batch dir: {e}")
        return base

def _show_log_text(log_text: str):
    """Display log text in expandable section."""
    with st.expander("📋 Pipeline logs (click to expand)", expanded=False):
        st.code(log_text, language="text")

def _safe_val(d: dict, key: str, fallback=0.0) -> float:
    """Safely extract numeric value from dict."""
    try:
        v = d.get(key, fallback)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return fallback
        return float(v)
    except Exception:
        return float(fallback)

# ==================== PROMETHEUS METRICS ====================

_METRIC_RE = re.compile(r"^([a-zA-Z0-9_:]+)(?:\{[^}]*\})?\s+([0-9.eE+\-]+)$")

@st.cache_data(show_spinner=False, ttl=60)
def fetch_metrics_text(url: str = METRICS_URL, timeout: int = 20) -> str | None:
    """Fetch /metrics text from Prometheus-compatible endpoint."""
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            return r.text
        logger.warning(f"Metrics endpoint returned status {r.status_code}")
    except requests.Timeout:
        logger.warning("Metrics request timed out")
    except Exception as e:
        logger.warning(f"Error fetching metrics: {e}")
    return None

def parse_prometheus_text(text: str) -> dict:
    """Parse Prometheus metrics text format."""
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

# ==================== PIPELINE EXECUTION ====================

@st.cache_data(show_spinner=False, ttl=300)
def run_pipeline_cached(n_transfers: int = 100) -> dict:
    """Cached wrapper for pipeline execution."""
    return run_pipeline_direct(n_transfers)

def run_pipeline_direct(n_transfers: int = 100, show_preview: bool = True) -> dict:
    """
    Run pipeline in-process: generate -> normalize -> optimize.
    Returns dict with status and paths to generated files.
    """
    log_buffer = StringIO()
    
    def log_print(*args, **kwargs):
        log_buffer.write(" ".join(map(str, args)) + "\n")
    
    try:
        # Import pipeline components
        from normalizer.normalizer_integration import generate_transfers, normalize_transfers
        from stablecoin_router.optimizer import TransactionReader, UnifiedOptimizer, ResultExporter
        from stablecoin_router.catalog import VenueCatalog

        log_print(f"Pipeline started at {time.asctime()}")
        log_print(f"Config directory: {CONFIG_DIR}")
        
        # Ensure config directory exists
        os.makedirs(CONFIG_DIR, exist_ok=True)
        
        # Step 1: Generate transfers
        log_print(f"Generating {n_transfers} transfers...")
        transfers_df = generate_transfers(
            n_transfers=n_transfers,
            output_dir=CONFIG_DIR,
            save_csv=True,
            save_json=True
        )
        log_print(f"Generated transfers shape: {transfers_df.shape if hasattr(transfers_df, 'shape') else 'unknown'}")

        # Step 2: Normalize transactions
        log_print("Normalizing transactions...")
        normalized_transactions = normalize_transfers(
            transfers_df=transfers_df,
            output_dir=CONFIG_DIR,
            save_results=True
        )
        log_print(f"Normalized transactions shape: {normalized_transactions.shape if hasattr(normalized_transactions, 'shape') else 'unknown'}")

        # Define file paths
        INPUT_CSV = os.path.join(CONFIG_DIR, "normalized_transactions.csv")
        OUTPUT_CSV = os.path.join(CONFIG_DIR, "optimization_results.csv")
        OUTPUT_CSV_MIP = os.path.join(CONFIG_DIR, "optimization_results_mip.csv")

        # Step 3: Optimize with LP
        log_print("Running LP optimization...")
        reader = TransactionReader()
        transactions = reader.read_from_csv(INPUT_CSV)
        log_print(f"Read {len(transactions)} transactions from {INPUT_CSV}")

        catalog = VenueCatalog()
        optimizer = UnifiedOptimizer(catalog, top_k=None)

        results_lp = optimizer.optimize_batch(transactions)
        log_print(f"LP optimization completed: {len(results_lp)} results")

        # Step 4: Attempt MIP optimization (optional)
        results_mip = []
        try:
            log_print("Attempting MIP optimization...")
            liquidity_map = {}
            for v in catalog.get_all_venues():
                asset = getattr(v, "settlement_asset", None) or v.venue_id
                liquidity_map[asset] = liquidity_map.get(asset, 0.0) + float(
                    getattr(v, "available_liquidity_usd", 0.0) or 0.0
                )

            mip_res = optimizer.optimize_batch_mip(
                transactions,
                liquidity_available=liquidity_map,
                alpha=1.0,
                beta=0.15,
                gamma=0.05,
                top_k=4,
                time_limit_sec=20
            )
            
            if mip_res.get("feasible", False):
                log_print(f"MIP optimization succeeded: objective={mip_res.get('objective_value')}")
                # Note: MIP result conversion requires additional helper function
                # If not available, results_mip remains empty
            else:
                log_print("MIP returned infeasible solution")
        except Exception as e:
            log_print(f"MIP optimization skipped: {str(e)}")

        # Step 5: Export results
        log_print("Exporting results...")
        exporter = ResultExporter()
        exporter.export_results(results_lp, OUTPUT_CSV)
        log_print(f"Exported LP results to {OUTPUT_CSV}")

        if results_mip:
            exporter.export_results(results_mip, OUTPUT_CSV_MIP)
            log_print(f"Exported MIP results to {OUTPUT_CSV_MIP}")

        log_text = log_buffer.getvalue()
        return {
            "ok": True,
            "log": log_text,
            "generated": {
                "normalized": INPUT_CSV,
                "lp": OUTPUT_CSV,
                "mip": OUTPUT_CSV_MIP if results_mip else None
            }
        }

    except Exception as e:
        tb = traceback.format_exc()
        log_buffer.write(f"\nERROR:\n{tb}")
        return {
            "ok": False,
            "log": log_buffer.getvalue(),
            "error": str(e)
        }

# ==================== INITIALIZATION ====================

@st.cache_resource(show_spinner=False)
def initialize_dashboard():
    """
    Initialize dashboard: ensure data exists, fetch metrics.
    This runs once per session and is cached.
    """
    init_status = {
        "data_ready": False,
        "pipeline_run": False,
        "prom_metrics": {},
        "data_dir": CONFIG_DIR,
        "error": None
    }
    
    try:
        # Check if data files exist
        data_dir = _latest_batch_dir(CONFIG_DIR)
        required_files = [
            "optimization_results.csv",
            "optimization_results_lp.csv",
            "normalized_transactions.csv"
        ]
        
        files_exist = any(
            os.path.exists(os.path.join(data_dir, f)) for f in required_files
        )
        
        if not files_exist:
            logger.info("No data files found, running pipeline...")
            result = run_pipeline_direct(n_transfers=100, show_preview=False)
            
            if result["ok"]:
                init_status["pipeline_run"] = True
                init_status["data_ready"] = True
                logger.info("Pipeline completed successfully")
            else:
                init_status["error"] = result.get("error", "Pipeline failed")
                logger.error(f"Pipeline failed: {init_status['error']}")
        else:
            init_status["data_ready"] = True
            logger.info(f"Data files found in {data_dir}")
        
        # Fetch Prometheus metrics
        metrics_text = fetch_metrics_text()
        if metrics_text:
            init_status["prom_metrics"] = parse_prometheus_text(metrics_text)
        
        init_status["data_dir"] = data_dir
        
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        init_status["error"] = str(e)
    
    return init_status

# ==================== MAIN DASHBOARD ====================

def main():
    """Main dashboard function."""
    
    # Initialize dashboard
    with st.spinner("Initializing dashboard..."):
        init_status = initialize_dashboard()
    
    # Import modules after initialization
    try:
        from metrics_calculator import MetricsCalculator, get_comparison_metrics, build_baseline_df
        from data_loader import DataLoader, apply_filters, get_filter_options
        from ai.ai_integration import get_transaction_insights
    except ImportError as e:
        st.error(f"Failed to import required modules: {e}")
        st.stop()
    
    # Check initialization status
    if init_status.get("error"):
        st.error(f"⚠️ Initialization error: {init_status['error']}")
        st.info("Try running the pipeline manually below.")
    
    if not init_status["data_ready"]:
        st.warning("⚠️ No data available. Please run the pipeline.")
    
    # ==================== SIDEBAR ====================
    
    st.sidebar.title("📁 Configuration")
    
    # Data directory
    data_dir = st.sidebar.text_input(
        "Data Directory",
        value=init_status["data_dir"],
        help="Path to directory with CSV files"
    )
    
    # Prometheus metrics panel
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Prometheus Metrics")
    
    prom_metrics = init_status.get("prom_metrics", {})
    if prom_metrics:
        st.sidebar.metric("Batches started", prom_metrics.get("batches_started_total", 0))
        st.sidebar.metric("Batches in progress", prom_metrics.get("batches_in_progress", 0))
        avg = prom_metrics.get("avg_processing_seconds")
        st.sidebar.metric("Avg processing (s)", f"{avg:.2f}" if avg else "—")
    else:
        st.sidebar.info("Metrics unavailable. Ensure API is running.")
    
    # Load data
    loader = DataLoader(data_dir)
    optimized_df = loader.load_optimized_data()
    
    if optimized_df is None:
        st.error("⚠️ No optimization results found!")
        render_pipeline_controls(data_dir)
        st.stop()
    
    # Load or build baseline
    baseline_df = loader.load_baseline_data() if hasattr(loader, "load_baseline_data") else None
    if baseline_df is None:
        try:
            baseline_df = build_baseline_df(data_dir)
        except Exception:
            baseline_df = None
    
    # Analysis mode
    st.sidebar.markdown("---")
    st.sidebar.title("🎯 Analysis Mode")
    
    if baseline_df is not None:
        analysis_mode = st.sidebar.radio(
            "Mode",
            ["Current Performance", "Optimization Comparison"],
            index=1,
            help="Compare before/after or view current results"
        )
    else:
        analysis_mode = "Current Performance"
        st.sidebar.info("Only optimized data available")
    
    # Filters
    st.sidebar.markdown("---")
    st.sidebar.title("🎛️ Filters")
    
    filter_options = get_filter_options(optimized_df)
    
    selected_business_types = st.sidebar.multiselect(
        "Business Type",
        filter_options.get('business_types', ['All']),
        default=['All']
    )
    
    selected_regions = st.sidebar.multiselect(
        "Region",
        filter_options.get('regions', ['All']),
        default=['All']
    )
    
    selected_urgency = st.sidebar.multiselect(
        "Urgency Level",
        filter_options.get('urgency_levels', ['All']),
        default=['All']
    )
    
    selected_tiers = st.sidebar.multiselect(
        "User Tier",
        filter_options.get('user_tiers', ['All']),
        default=['All']
    )
    
    # Apply filters
    filtered_df = apply_filters(
        optimized_df,
        business_types=selected_business_types if 'All' not in selected_business_types else None,
        regions=selected_regions if 'All' not in selected_regions else None,
        urgency_levels=selected_urgency if 'All' not in selected_urgency else None,
        user_tiers=selected_tiers if 'All' not in selected_tiers else None
    )
    
    filtered_baseline = None
    if baseline_df is not None:
        filtered_baseline = apply_filters(
            baseline_df,
            business_types=selected_business_types if 'All' not in selected_business_types else None,
            regions=selected_regions if 'All' not in selected_regions else None,
            urgency_levels=selected_urgency if 'All' not in selected_urgency else None,
            user_tiers=selected_tiers if 'All' not in selected_tiers else None
        )
    
    # ==================== HEADER ====================
    
    st.title("💱 Stablecoin Route Optimization Dashboard")
    st.markdown(f"**Transactions:** {len(filtered_df):,} | **Mode:** {analysis_mode}")
    # st.markdown("---")
    
    # ==================== PIPELINE CONTROLS ====================
    
    render_pipeline_controls(data_dir)
    
    # ==================== METRICS SUMMARY ====================
    
    calc = MetricsCalculator(filtered_df, filtered_baseline)
    metrics = calc.get_summary_metrics()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Volume", f"${metrics['total_volume_usd']/1e6:.2f}M")
    with col2:
        st.metric("Avg Cost", f"{metrics['avg_cost_bps']:.2f} BPS")
    with col3:
        st.metric("Success Rate", f"{metrics['success_rate_pct']:.1f}%")
    with col4:
        bip = prom_metrics.get("batches_in_progress")
        if bip is not None:
            st.metric("Batches In Progress", f"{bip}")
        else:
            st.metric("Transactions", f"{metrics['total_transactions']:,}")
    
    # ==================== TABS ====================
    
    if analysis_mode == "Optimization Comparison" and filtered_baseline is not None:
        tabs = st.tabs([
            "🎯 Optimization Impact",
            "📊 Overview",
            "💰 Cost Analysis",
            "⚡ Performance",
            "📈 Segmentation",
            "🤖 AI Insights"
        ])
        render_optimization_tab(tabs[0], filtered_baseline, filtered_df, calc, metrics)
        overview_tab_idx = 1
    else:
        tabs = st.tabs([
            "📊 Overview",
            "💰 Cost Analysis",
            "⚡ Performance",
            "📈 Segmentation",
            "🤖 AI Insights"
        ])
        overview_tab_idx = 0
    
    render_overview_tab(tabs[overview_tab_idx], metrics)
    render_cost_tab(tabs[overview_tab_idx + 1], filtered_df)
    render_performance_tab(tabs[overview_tab_idx + 2], filtered_df)
    render_segmentation_tab(tabs[overview_tab_idx + 3], filtered_df, calc)
    render_ai_insights_tab(tabs[overview_tab_idx + 4])
    
    # Footer
    st.markdown("---")
    st.markdown(f"*Dashboard v3.1 - Optimized Architecture | Transactions: {len(filtered_df):,}*")

# ==================== TAB RENDERERS ====================

def render_pipeline_controls(data_dir: str):
    """Render pipeline execution controls."""

    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown("## 🛠️ Transfer generation engine")
    
    with col2:
        # Push button down slightly to align with title
        st.write("")  # Small spacer
        run_direct = st.button(
            "▶️ Optimize",
            help="Generate and optimize transfers",
            type="primary",
            use_container_width=True
        )
    st.markdown("---")
    # st.caption(f"📁 Output: `{data_dir}`")
    
    if run_direct:
        with st.spinner("Running pipeline... this may take a minute"):
            res = run_pipeline_direct(n_transfers=100)
        
        _show_log_text(res.get("log", ""))
        
        if res.get("ok"):
            st.success("✅ Pipeline completed successfully!")
            for k, v in res.get("generated", {}).items():
                if v and os.path.exists(v):
                    st.download_button(
                        f"📥 Download {os.path.basename(v)}",
                        data=open(v, "rb").read(),
                        file_name=os.path.basename(v),
                        key=f"download_{k}"
                    )
            st.rerun()
        else:
            st.error(f"❌ Pipeline failed: {res.get('error', 'Unknown error')}")

def render_optimization_tab(tab, baseline_df, optimized_df, calc, metrics):
    """Render optimization impact tab."""
    with tab:
        st.header("🎯 Optimization Impact")
        
        try:
            comparison = get_comparison_metrics(baseline_df, optimized_df) or {}
        except Exception as e:
            st.warning(f"Could not compute comparison metrics: {e}")
            comparison = {}
        
        metrics_baseline = comparison.get('before', {})
        metrics_after = comparison.get('after', {})
        
        if not metrics_baseline:
            try:
                from metrics_calculator import MetricsCalculator
                calc_baseline = MetricsCalculator(baseline_df)
                metrics_baseline = calc_baseline.get_summary_metrics()
            except Exception:
                metrics_baseline = {}
        
        if not metrics_after:
            metrics_after = metrics
        
        # Key improvements
        st.markdown("### 💡 Key Improvements")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            cost_improvement = comparison.get('improvements', {}).get('cost_bps')
            if cost_improvement is None:
                cost_improvement = metrics_after.get('avg_cost_improvement_bps', 0.0)
            base_cost = metrics_baseline.get('avg_cost_bps', 0.0)
            pct = (cost_improvement / base_cost * 100) if base_cost else 0.0
            st.metric("Cost Reduction", f"{cost_improvement:.1f} BPS", f"{pct:.1f}%")
        
        with col2:
            savings = metrics_after.get('total_savings_usd', 0.0)
            st.metric("Total Savings", f"${savings:,.2f}", "Saved")
        
        with col3:
            success_after = metrics_after.get('success_rate_pct', 0.0)
            success_before = metrics_baseline.get('success_rate_pct', 0.0)
            st.metric("Success Rate", f"{success_after:.1f}%", f"{success_after - success_before:.1f}pp")
        
        with col4:
            avg_routes = metrics_after.get('avg_routes', 0.0)
            st.metric("Avg Routes", f"{avg_routes:.1f}", "Optimized")
        
        # Comparison charts
        st.markdown("---")
        st.markdown("### 📊 Before vs After")
        
        col1, col2 = st.columns(2)
        
        with col1:
            before_cost = _safe_val(metrics_baseline, 'avg_cost_bps', 0.0)
            after_cost = _safe_val(metrics_after, 'avg_cost_bps', 0.0)
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
            before_time = _safe_val(metrics_baseline, 'avg_settlement_time_sec', 0.0) / 60.0
            after_time = _safe_val(metrics_after, 'avg_settlement_time_sec', 0.0) / 60.0
            comparison_time = pd.DataFrame({
                'Metric': ['Before', 'After'],
                'Time (min)': [before_time, after_time]
            })
            fig = px.bar(
                comparison_time, x='Metric', y='Time (min)',
                title='Average Settlement Time Comparison',
                color='Metric',
                color_discrete_map={'Before': '#ff7f0e', 'After': '#2ca02c'}
            )
            st.plotly_chart(fig, use_container_width=True)

def render_overview_tab(tab, metrics):
    """Render overview tab."""
    with tab:
        st.header("📊 Performance Overview")
        
        # Top metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Volume", f"${metrics['total_volume_usd']/1e6:.2f}M")
        with col2:
            st.metric("Avg Cost", f"{metrics['avg_cost_bps']:.2f} BPS")
        with col3:
            st.metric("Success Rate", f"{metrics['success_rate_pct']:.1f}%")
        with col4:
            st.metric("Total Fees", f"${metrics['total_fees_usd']:,.2f}")
        
        # Second row
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            st.metric("Transactions", f"{metrics['total_transactions']:,}")
        with col6:
            st.metric("Avg Time", f"{metrics['avg_settlement_time_sec']/60:.1f} min")
        with col7:
            st.metric("Avg Routes", f"{metrics['avg_routes']:.1f}")
        with col8:
            st.metric("Constraints Met", f"{metrics['constraints_satisfied_pct']:.1f}%")
        
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

def render_cost_tab(tab, filtered_df):
    """Render cost analysis tab."""
    with tab:
        st.header("💰 Cost Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'total_cost_bps' in filtered_df.columns:
                fig = px.histogram(
                    filtered_df, x='total_cost_bps',
                    title='Cost Distribution (BPS)',
                    nbins=30
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'original_type' in filtered_df.columns:
                cost_by_type = filtered_df.groupby('original_type')['total_cost_bps'].mean().sort_values()
                fig = px.bar(
                    x=cost_by_type.values, y=cost_by_type.index,
                    orientation='h',
                    title='Average Cost by Business Type',
                    labels={'x': 'Avg Cost (BPS)', 'y': 'Business Type'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        if 'cost_improvement_bps' in filtered_df.columns:
            st.markdown("---")
            st.markdown("### 💰 Cost Improvement Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(
                    filtered_df, x='cost_improvement_bps',
                    title='Cost Improvement Distribution',
                    nbins=30
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                top_improved = filtered_df.nlargest(10, 'cost_improvement_bps')[
                    ['transfer_id', 'original_type', 'cost_improvement_bps', 'total_amount_usd']
                ]
                st.markdown("#### Top 10 Cost Reductions")
                st.dataframe(top_improved, use_container_width=True)

def render_performance_tab(tab, filtered_df):
    """Render performance metrics tab."""
    with tab:
        st.header("⚡ Performance Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'total_time_sec' in filtered_df.columns:
                fig = px.box(filtered_df, y='total_time_sec', title='Settlement Time Distribution')
                fig.update_yaxes(title='Time (seconds)')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'num_routes' in filtered_df.columns:
                route_dist = filtered_df['num_routes'].value_counts().sort_index()
                fig = px.bar(
                    x=route_dist.index, y=route_dist.values,
                    title='Number of Routes Distribution',
                    labels={'x': 'Number of Routes', 'y': 'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        if 'status' in filtered_df.columns:
            st.markdown("---")
            st.markdown("### ✅ Optimization Status")
            
            status_counts = filtered_df['status'].value_counts()
            fig = px.pie(
                values=status_counts.values, names=status_counts.index,
                title='Optimization Status Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        if 'route_1_venue' in filtered_df.columns:
            st.markdown("---")
            st.markdown("### 🏦 Top Venues")
            
            venue_counts = filtered_df['route_1_venue'].value_counts().head(10)
            fig = px.bar(
                x=venue_counts.values, y=venue_counts.index,
                orientation='h',
                title='Top 10 Venues by Usage',
                labels={'x': 'Transaction Count', 'y': 'Venue'}
            )
            st.plotly_chart(fig, use_container_width=True)

def render_segmentation_tab(tab, filtered_df, calc):
    """Render segmentation analysis tab."""
    with tab:
        st.header("📈 Segmentation Analysis")
        
        if 'original_type' in filtered_df.columns:
            st.markdown("### 💼 By Business Type")
            metrics_by_type = calc.get_metrics_by_business_type()
            st.dataframe(metrics_by_type, use_container_width=True)
        
        if 'region' in filtered_df.columns:
            st.markdown("---")
            st.markdown("### 🌍 By Region")
            metrics_by_region = calc.get_metrics_by_region()
            st.dataframe(metrics_by_region, use_container_width=True)
            
            fig = px.bar(
                metrics_by_region, x='Region', y='Volume ($)',
                title='Transaction Volume by Region'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        if 'urgency_level' in filtered_df.columns:
            st.markdown("---")
            st.markdown("### ⚡ By Urgency Level")
            metrics_by_urgency = calc.get_metrics_by_urgency()
            st.dataframe(metrics_by_urgency, use_container_width=True)
        
        if 'user_tier' in filtered_df.columns:
            st.markdown("---")
            st.markdown("### 👥 By User Tier")
            metrics_by_tier = calc.get_metrics_by_user_tier()
            st.dataframe(metrics_by_tier, use_container_width=True)

def render_ai_insights_tab(tab):
    """Render AI insights tab."""
    with tab:
        st.header("🤖 AI-Powered Insights")
        
        if st.button("🔍 Generate AI Insights", type="primary"):
            with st.spinner("Analyzing transactions..."):
                try:
                    from ai.ai_integration import get_transaction_insights
                    insights = get_transaction_insights()
                    st.success("✅ Insights generated successfully!")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("### 📋 Summary")
                        st.write(insights.get("summary", "No summary available."))
                    
                    with col2:
                        st.markdown("### ⚠️ Anomalies")
                        anomalies = insights.get("anomalies", [])
                        if anomalies:
                            for a in anomalies:
                                st.markdown(f"- {a}")
                        else:
                            st.write("No anomalies detected.")
                    
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
        else:
            st.info("Click the button above to generate AI-powered insights from your transaction data.")

# ==================== ENTRY POINT ====================

if __name__ == "__main__":
    main()