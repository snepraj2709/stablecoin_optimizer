import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path
import sys

# ---------- Defensive helpers (paste near imports) ----------
def safe_mean(series):
    """Return numeric mean or 0.0 if series empty/NaN/uncoercible."""
    try:
        if series is None or len(series) == 0:
            return 0.0
        # coerce to numeric to protect against strings
        s = pd.to_numeric(series, errors='coerce')
        m = s.mean()
        return float(m) if pd.notna(m) else 0.0
    except Exception:
        return 0.0

def safe_sum(series):
    try:
        if series is None or len(series) == 0:
            return 0.0
        s = pd.to_numeric(series, errors='coerce').sum()
        return float(s) if pd.notna(s) else 0.0
    except Exception:
        return 0.0

def safe_count(df, col=None):
    if df is None:
        return 0
    if col is None:
        return len(df)
    return int(df[col].count())

def format_val_or_na(val, fmt="{:.2f}"):
    """Format numeric value or return 'N/A' for NaN/None types."""
    try:
        if val is None:
            return "N/A"
        if isinstance(val, float) and np.isnan(val):
            return "N/A"
        # for ints etc.
        return fmt.format(val)
    except Exception:
        try:
            return fmt.format(float(val))
        except Exception:
            return "N/A"

def calculate_improvement(before_val, after_val, inverse=False):
    """Calculate percentage improvement (safe). Returns float."""
    try:
        b = float(before_val) if pd.notna(before_val) else 0.0
    except Exception:
        b = 0.0
    try:
        a = float(after_val) if pd.notna(after_val) else 0.0
    except Exception:
        a = 0.0

    if b == 0:
        return 0.0
    improvement = ((b - a) / b) * 100
    return -improvement if inverse else improvement
# -----------------------------------------------------------

# Try to import AI insights module
try:
    from ai_insights import (
        AIInsightsEngine, 
        prepare_daily_summary_data,
        identify_exceptions,
        identify_cost_anomalies,
        prepare_route_optimization_data
    )
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    st.warning("⚠️ AI Insights module not found. AI features will be disabled.")

# Page configuration
st.set_page_config(
    page_title="Stablecoin Route Optimization Dashboard",
    page_icon="💱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .big-font {
        font-size:50px !important;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0 0;
        padding: 10px 20px;
    }
    .alert-red {
        background-color: #ffebee;
        padding: 10px;
        border-left: 4px solid #f44336;
        margin: 10px 0;
    }
    .alert-green {
        background-color: #e8f5e9;
        padding: 10px;
        border-left: 4px solid #4caf50;
        margin: 10px 0;
    }
    .alert-blue {
        background-color: #e3f2fd;
        padding: 10px;
        border-left: 4px solid #2196f3;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# File path configuration
st.sidebar.title("📁 Data Configuration")

# Allow users to specify data directory
data_dir = st.sidebar.text_input(
    "Data Directory Path",
    value="./config",
    help="Path to directory containing your CSV files"
)

# Define file paths
ALL_TRANSFER_CSV = os.path.join(data_dir, "generated_transfers.csv")
ALL_NORMALISED_CSV = os.path.join(data_dir, "normalized_transactions.csv")
OPTIMISED_TRANSFER_CSV = os.path.join(data_dir, "optimization_results.csv")

# Load data function
@st.cache_data
def load_data(file_path):
    """Load data from CSV file"""
    try:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            return df
        else:
            return None
    except Exception as e:
        st.error(f"Error loading {file_path}: {str(e)}")
        return None

# Check file availability
def check_files():
    """Check which data files are available"""
    files_status = {
        'Original Transfers': os.path.exists(ALL_TRANSFER_CSV),
        'Normalized Data': os.path.exists(ALL_NORMALISED_CSV),
        'Optimized Results': os.path.exists(OPTIMISED_TRANSFER_CSV)
    }
    return files_status

# Display file status
files_status = check_files()
st.sidebar.markdown("### 📊 Data Files Status")
for file_name, status in files_status.items():
    status_icon = "✅" if status else "❌"
    st.sidebar.markdown(f"{status_icon} {file_name}")

# Data source selection
st.sidebar.markdown("---")
st.sidebar.title("🎯 Analysis Mode")

analysis_mode = st.sidebar.radio(
    "Select Analysis Mode",
    ["Current State", "Optimization Comparison"],
    help="Choose between analyzing current data or comparing pre/post optimization"
)

# Load appropriate data based on mode
df = None
df_before = None
df_after = None
data_source = "N/A"

if analysis_mode == "Current State":
    # Use normalized data as primary source, fall back to optimized if available
    if files_status['Normalized Data']:
        df = load_data(ALL_NORMALISED_CSV)
        data_source = "Normalized Transactions"
    elif files_status['Optimized Results']:
        df = load_data(OPTIMISED_TRANSFER_CSV)
        data_source = "Optimized Results"
    elif files_status['Original Transfers']:
        df = load_data(ALL_TRANSFER_CSV)
        data_source = "Original Transfers"
    else:
        st.error("⚠️ No data files found! Please check your data directory path.")
        st.stop()
    
    st.sidebar.info(f"📌 Using: **{data_source}**")
    
else:  # Optimization Comparison
    if files_status['Original Transfers'] and files_status['Optimized Results']:
        df_before = load_data(ALL_TRANSFER_CSV)
        df_after = load_data(OPTIMISED_TRANSFER_CSV)
        # Use optimized as primary for filtering
        df = df_after if df_after is not None else df_before
        data_source = "Comparison Mode"
        st.sidebar.success("✅ Comparison mode enabled")
    else:
        st.error("⚠️ Both original and optimized data files are required for comparison mode!")
        st.stop()

# Sidebar filters
st.sidebar.markdown("---")
st.sidebar.title("🎛️ Filters")

# Date range filter
if df is not None and 'timestamp' in df.columns and df['timestamp'].notna().any():
    # Ensure we have at least one non-NaT timestamp
    min_date = df['timestamp'].dropna().min().date()
    max_date = df['timestamp'].dropna().max().date()
    try:
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
    except Exception:
        # Fallback if Streamlit complains about values
        date_range = (min_date, max_date)
else:
    date_range = None

# Business type filter
if df is not None and 'business_type' in df.columns:
    business_types = ['All'] + sorted(df['business_type'].dropna().unique().tolist())
    selected_business_type = st.sidebar.multiselect(
        "Business Type",
        business_types,
        default=['All']
    )
else:
    selected_business_type = ['All']

# Region filter
if df is not None and 'region' in df.columns:
    regions = ['All'] + sorted(df['region'].dropna().unique().tolist())
    selected_region = st.sidebar.multiselect(
        "Region",
        regions,
        default=['All']
    )
else:
    selected_region = ['All']

# Urgency level filter
if df is not None and 'urgency_level' in df.columns:
    urgency_levels = ['All'] + sorted(df['urgency_level'].dropna().unique().tolist())
    selected_urgency = st.sidebar.multiselect(
        "Urgency Level",
        urgency_levels,
        default=['All']
    )
else:
    selected_urgency = ['All']

# User tier filter
if df is not None and 'user_tier' in df.columns:
    user_tiers = ['All'] + sorted(df['user_tier'].dropna().unique().tolist())
    selected_tier = st.sidebar.multiselect(
        "User Tier",
        user_tiers,
        default=['All']
    )
else:
    selected_tier = ['All']

# AI Configuration Section
st.sidebar.markdown("---")
st.sidebar.title("🤖 AI Insights (Beta)")

ai_enabled = False
ai_engine = None

if AI_AVAILABLE:
    openai_api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key to enable AI-powered insights"
    )
    
    if openai_api_key:
        try:
            ai_model = st.sidebar.selectbox(
                "AI Model",
                ["gpt-4", "gpt-4-turbo-preview", "gpt-3.5-turbo"],
                help="GPT-4 provides better analysis but is more expensive"
            )
            ai_engine = AIInsightsEngine(openai_api_key, model=ai_model)
            ai_enabled = True
            st.sidebar.success("✅ AI Insights Enabled")
        except Exception as e:
            st.sidebar.error(f"❌ AI Error: {str(e)}")
    else:
        st.sidebar.info("💡 Enter OpenAI API key to enable AI insights")
else:
    st.sidebar.warning("⚠️ Install openai package:\n`pip install openai`")

# Apply filters function
def apply_filters(dataframe):
    """Apply selected filters to dataframe"""
    if dataframe is None:
        return None
    
    filtered = dataframe.copy()
    
    # Date filter
    if date_range is not None and isinstance(date_range, (list, tuple)) and len(date_range) == 2 and 'timestamp' in filtered.columns:
        try:
            start_date, end_date = date_range
            mask = (filtered['timestamp'].dt.date >= start_date) & (filtered['timestamp'].dt.date <= end_date)
            filtered = filtered[mask]
        except Exception:
            pass
    
    # Business type filter
    if 'All' not in selected_business_type and 'business_type' in filtered.columns:
        filtered = filtered[filtered['business_type'].isin(selected_business_type)]
    
    # Region filter
    if 'All' not in selected_region and 'region' in filtered.columns:
        filtered = filtered[filtered['region'].isin(selected_region)]
    
    # Urgency filter
    if 'All' not in selected_urgency and 'urgency_level' in filtered.columns:
        filtered = filtered[filtered['urgency_level'].isin(selected_urgency)]
    
    # User tier filter
    if 'All' not in selected_tier and 'user_tier' in filtered.columns:
        filtered = filtered[filtered['user_tier'].isin(selected_tier)]
    
    return filtered

# Apply filters
if analysis_mode == "Optimization Comparison":
    filtered_df_before = apply_filters(df_before)
    filtered_df_after = apply_filters(df_after)
    filtered_df = filtered_df_after if filtered_df_after is not None else filtered_df_before
else:
    filtered_df = apply_filters(df)

# Check if we have data after filtering
if filtered_df is None or len(filtered_df) == 0:
    st.warning("⚠️ No data available with current filters. Please adjust your filters.")
    st.stop()

# Ensure some numeric columns are coerced to numeric to avoid surprises downstream
numeric_cols = [
    'total_cost_bps', 'settlement_time_sec', 'total_fees_usd', 'gas_cost_usd',
    'lp_fee_usd', 'bridge_cost_usd', 'slippage_cost_usd', 'amount_source',
    'routing_hops'
]
for col in numeric_cols:
    if col in filtered_df.columns:
        filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce')

if analysis_mode == "Optimization Comparison":
    if filtered_df_before is not None:
        for col in numeric_cols:
            if col in filtered_df_before.columns:
                filtered_df_before[col] = pd.to_numeric(filtered_df_before[col], errors='coerce')
    if filtered_df_after is not None:
        for col in numeric_cols:
            if col in filtered_df_after.columns:
                filtered_df_after[col] = pd.to_numeric(filtered_df_after[col], errors='coerce')

# Main content
st.title("💱 Stablecoin Route Optimization Dashboard")
st.markdown(f"**Data Source:** {data_source} | **Transactions:** {len(filtered_df):,}")
st.markdown("---")

# Create tabs for different dashboard views
if analysis_mode == "Optimization Comparison":
    if ai_enabled:
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "🎯 Optimization Impact",
            "🤖 AI Insights",
            "📊 Executive Summary",
            "💰 Cost Analysis",
            "⚡ Performance Metrics",
            "🗺️ Route Intelligence",
            "🌍 Regional & Compliance"
        ])
    else:
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🎯 Optimization Impact",
            "📊 Executive Summary",
            "💰 Cost Analysis",
            "⚡ Performance Metrics",
            "🗺️ Route Intelligence",
            "🌍 Regional & Compliance"
        ])
else:
    if ai_enabled:
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🤖 AI Insights",
            "📊 Executive Summary",
            "💰 Cost Analysis",
            "⚡ Performance Metrics",
            "🗺️ Route Intelligence",
            "🌍 Regional & Compliance"
        ])
    else:
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Executive Summary",
            "💰 Cost Analysis",
            "⚡ Performance Metrics",
            "🗺️ Route Intelligence",
            "🌍 Regional & Compliance"
        ])

# TAB: Optimization Impact (only in comparison mode)
if analysis_mode == "Optimization Comparison":
    with tab1:
        st.header("🎯 Optimization Impact Analysis")
        st.markdown("### 📈 Bottom Line: Key Improvements")
        
        # Calculate key metrics for both datasets
        metrics_comparison = {}
        cost_improvement = 0.0
        time_improvement = 0.0
        success_improvement = 0.0
        
        # Cost metrics
        if filtered_df_before is not None and filtered_df_after is not None and \
           'total_cost_bps' in filtered_df_before.columns and 'total_cost_bps' in filtered_df_after.columns:
            before_cost = safe_mean(filtered_df_before['total_cost_bps'])
            after_cost = safe_mean(filtered_df_after['total_cost_bps'])
            cost_improvement = calculate_improvement(before_cost, after_cost)
            metrics_comparison['cost'] = (before_cost, after_cost, cost_improvement)
        
        # Time metrics
        if filtered_df_before is not None and filtered_df_after is not None and \
           'settlement_time_sec' in filtered_df_before.columns and 'settlement_time_sec' in filtered_df_after.columns:
            before_time = safe_mean(filtered_df_before['settlement_time_sec'])
            after_time = safe_mean(filtered_df_after['settlement_time_sec'])
            time_improvement = calculate_improvement(before_time, after_time)
            metrics_comparison['time'] = (before_time, after_time, time_improvement)
        else:
            time_improvement = 0.0
        
        # Success rate
        if filtered_df_before is not None and filtered_df_after is not None and \
           'settlement_status' in filtered_df_before.columns and 'settlement_status' in filtered_df_after.columns:
            before_success = ((filtered_df_before['settlement_status'] == 'completed').sum() / len(filtered_df_before) * 100) if len(filtered_df_before) > 0 else 0.0
            after_success = ((filtered_df_after['settlement_status'] == 'completed').sum() / len(filtered_df_after) * 100) if len(filtered_df_after) > 0 else 0.0
            success_improvement = after_success - before_success
            metrics_comparison['success'] = (before_success, after_success, success_improvement)
        
        # Display improvement metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'cost' in metrics_comparison:
                before, after, improvement = metrics_comparison['cost']
                st.metric(
                    "Avg Cost (BPS)",
                    format_val_or_na(after, "{:.2f}"),
                    f"{improvement:.1f}%",
                    delta_color="inverse"
                )
                st.caption(f"Before: {format_val_or_na(before, '{:.2f}')} BPS")
            else:
                st.metric("Avg Cost (BPS)", "N/A")
        
        with col2:
            if 'time' in metrics_comparison:
                before, after, improvement = metrics_comparison['time']
                st.metric(
                    "Avg Settlement Time",
                    format_val_or_na(after/60 if after is not None else None, "{:.1f}") + " min" if after not in (None, 0, np.nan) else "N/A",
                    f"{improvement:.1f}%",
                    delta_color="inverse"
                )
                st.caption(f"Before: {format_val_or_na(before/60 if before is not None else None, '{:.1f}')} min")
            else:
                st.metric("Avg Settlement Time", "N/A")
        
        with col3:
            if 'success' in metrics_comparison:
                before, after, improvement = metrics_comparison['success']
                st.metric(
                    "Success Rate",
                    format_val_or_na(after, "{:.1f}") + "%" if after not in (None, np.nan) else "N/A",
                    f"+{improvement:.1f}%",
                    delta_color="normal"
                )
                st.caption(f"Before: {format_val_or_na(before, '{:.1f}')}%")
            else:
                st.metric("Success Rate", "N/A")
        
        with col4:
            # Calculate cost savings
            if filtered_df_before is not None and filtered_df_after is not None and \
               'total_fees_usd' in filtered_df_before.columns and 'total_fees_usd' in filtered_df_after.columns:
                total_savings = safe_sum(filtered_df_before['total_fees_usd']) - safe_sum(filtered_df_after['total_fees_usd'])
                st.metric(
                    "Total Cost Savings",
                    f"${total_savings:,.2f}",
                    "Saved",
                    delta_color="normal"
                )
            else:
                st.metric("Total Cost Savings", "N/A")
        
        st.markdown("---")
        
        # ROI Calculation
        st.markdown("### 💰 Return on Investment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if filtered_df_before is not None and filtered_df_after is not None and \
               'total_fees_usd' in filtered_df_before.columns and 'total_fees_usd' in filtered_df_after.columns:
                total_before = safe_sum(filtered_df_before['total_fees_usd'])
                total_after = safe_sum(filtered_df_after['total_fees_usd'])
                total_savings = total_before - total_after
                
                # Calculate annualized savings (assuming data represents a period)
                if 'timestamp' in filtered_df_after.columns and filtered_df_after['timestamp'].notna().any():
                    min_ts = filtered_df_after['timestamp'].dropna().min()
                    max_ts = filtered_df_after['timestamp'].dropna().max()
                    if pd.notna(min_ts) and pd.notna(max_ts):
                        days_in_data = (max_ts - min_ts).days
                    else:
                        days_in_data = 0
                    if days_in_data > 0:
                        annualized_savings = (total_savings / days_in_data) * 365
                    else:
                        annualized_savings = total_savings
                else:
                    annualized_savings = total_savings
                
                savings_rate = (total_savings / total_before * 100) if total_before > 0 else 0.0
                
                st.markdown(f"""
                <div class="alert-green">
                    <h4>💵 Cost Reduction Impact</h4>
                    <p><strong>Total Fees Before Optimization:</strong> ${total_before:,.2f}</p>
                    <p><strong>Total Fees After Optimization:</strong> ${total_after:,.2f}</p>
                    <p><strong>Total Savings:</strong> ${total_savings:,.2f}</p>
                    <p><strong>Savings Rate:</strong> {savings_rate:.1f}%</p>
                    <p><strong>Projected Annual Savings:</strong> ${annualized_savings:,.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Cost data not sufficient for ROI calculation")
        
        with col2:
            # Transaction efficiency improvement
            if filtered_df_before is not None and filtered_df_after is not None:
                txn_count_before = len(filtered_df_before)
                txn_count_after = len(filtered_df_after)
                
                st.markdown(f"""
                <div class="alert-blue">
                    <h4>📊 Transaction Efficiency</h4>
                    <p><strong>Transactions Analyzed:</strong> {txn_count_after:,}</p>
                    <p><strong>Avg Cost Reduction:</strong> {format_val_or_na(cost_improvement, '{:.1f}')}%</p>
                    <p><strong>Avg Time Reduction:</strong> {format_val_or_na(time_improvement, '{:.1f}')}%</p>
                    <p><strong>Success Rate Improvement:</strong> +{format_val_or_na(success_improvement, '{:.1f}')}pp</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Transaction data not sufficient for efficiency metrics")
        
        st.markdown("---")
        
        # Side-by-side comparison charts
        st.markdown("### 📊 Before vs After Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Cost distribution comparison
            if filtered_df_before is not None and filtered_df_after is not None and \
               'total_cost_bps' in filtered_df_before.columns and 'total_cost_bps' in filtered_df_after.columns:
                fig = go.Figure()
                fig.add_trace(go.Box(y=filtered_df_before['total_cost_bps'].dropna(), name='Before'))
                fig.add_trace(go.Box(y=filtered_df_after['total_cost_bps'].dropna(), name='After'))
                fig.update_layout(title='Cost Distribution Comparison (BPS)', 
                                 yaxis_title='Total Cost (BPS)', height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Cost distribution data not available for comparison")
        
        with col2:
            # Settlement time comparison
            if filtered_df_before is not None and filtered_df_after is not None and \
               'settlement_time_sec' in filtered_df_before.columns and 'settlement_time_sec' in filtered_df_after.columns:
                fig = go.Figure()
                fig.add_trace(go.Box(y=(filtered_df_before['settlement_time_sec'].dropna()/60), name='Before'))
                fig.add_trace(go.Box(y=(filtered_df_after['settlement_time_sec'].dropna()/60), name='After'))
                fig.update_layout(title='Settlement Time Comparison (minutes)', 
                                 yaxis_title='Settlement Time (min)', height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Settlement time data not available for comparison")
        
        # Cost component breakdown comparison
        st.markdown("### 💵 Cost Component Analysis")
        
        def sum_col_or_zero(df_obj, col):
            return safe_sum(df_obj[col]) if df_obj is not None and col in df_obj.columns else 0.0
        
        cost_components_before = {
            'Gas': sum_col_or_zero(filtered_df_before, 'gas_cost_usd'),
            'LP Fees': sum_col_or_zero(filtered_df_before, 'lp_fee_usd'),
            'Bridge': sum_col_or_zero(filtered_df_before, 'bridge_cost_usd'),
            'Slippage': sum_col_or_zero(filtered_df_before, 'slippage_cost_usd')
        }
        
        cost_components_after = {
            'Gas': sum_col_or_zero(filtered_df_after, 'gas_cost_usd'),
            'LP Fees': sum_col_or_zero(filtered_df_after, 'lp_fee_usd'),
            'Bridge': sum_col_or_zero(filtered_df_after, 'bridge_cost_usd'),
            'Slippage': sum_col_or_zero(filtered_df_after, 'slippage_cost_usd')
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure(data=[
                go.Bar(name='Before', x=list(cost_components_before.keys()), 
                      y=list(cost_components_before.values())),
                go.Bar(name='After', x=list(cost_components_after.keys()), 
                      y=list(cost_components_after.values()))
            ])
            fig.update_layout(title='Cost Components: Before vs After', 
                            yaxis_title='Cost ($)', barmode='group', height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Calculate savings per component
            savings_data = {
                'Component': list(cost_components_before.keys()),
                'Savings ($)': [cost_components_before[k] - cost_components_after[k] 
                               for k in cost_components_before.keys()],
                'Savings (%)': [
                    ((cost_components_before[k] - cost_components_after[k]) / cost_components_before[k] * 100) 
                    if cost_components_before[k] > 0 else 0.0
                    for k in cost_components_before.keys()
                ]
            }
            savings_df = pd.DataFrame(savings_data)
            
            if not savings_df.empty:
                fig = px.bar(savings_df, x='Component', y='Savings ($)', 
                            title='Savings by Cost Component',
                            text='Savings (%)',
                            color='Savings ($)',
                            color_continuous_scale='Greens')
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No savings data to display")
        
        # Routing improvement
        st.markdown("### 🗺️ Routing Optimization")
        
        if filtered_df_before is not None and filtered_df_after is not None and \
           'routing_hops' in filtered_df_before.columns and 'routing_hops' in filtered_df_after.columns:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                before_hops = safe_mean(filtered_df_before['routing_hops'])
                after_hops = safe_mean(filtered_df_after['routing_hops'])
                hop_improvement = calculate_improvement(before_hops, after_hops)
                st.metric("Avg Routing Hops", format_val_or_na(after_hops, "{:.2f}"), f"{hop_improvement:.1f}%", delta_color="inverse")
                st.caption(f"Before: {format_val_or_na(before_hops, '{:.2f}')}")
            
            with col2:
                before_single = (filtered_df_before['routing_hops'] == 1).sum() / len(filtered_df_before) * 100 if len(filtered_df_before) > 0 else 0.0
                after_single = (filtered_df_after['routing_hops'] == 1).sum() / len(filtered_df_after) * 100 if len(filtered_df_after) > 0 else 0.0
                st.metric("Single-Hop Routes", format_val_or_na(after_single, "{:.1f}") + "%", f"+{(after_single-before_single):.1f}pp")
                st.caption(f"Before: {format_val_or_na(before_single, '{:.1f}')}%")
            
            with col3:
                before_multi = (filtered_df_before['routing_hops'] >= 3).sum() / len(filtered_df_before) * 100 if len(filtered_df_before) > 0 else 0.0
                after_multi = (filtered_df_after['routing_hops'] >= 3).sum() / len(filtered_df_after) * 100 if len(filtered_df_after) > 0 else 0.0
                st.metric("Multi-Hop (3+) Routes", format_val_or_na(after_multi, "{:.1f}") + "%", f"{(after_multi-before_multi):.1f}pp", delta_color="inverse")
                st.caption(f"Before: {format_val_or_na(before_multi, '{:.1f}')}%")
        else:
            st.info("Routing hops data not available for comparison")

# TAB: AI Insights (when enabled)
if ai_enabled:
    if analysis_mode == "Optimization Comparison":
        ai_tab = tab2
        first_data_tab = tab3
    else:
        ai_tab = tab1
        first_data_tab = tab2
    
    with ai_tab:
        st.header("🤖 AI-Powered Insights & Recommendations")
        st.markdown("""
        <div class="alert-blue">
        <strong>🎯 AI-Generated Analysis</strong><br>
        Powered by OpenAI, these insights are generated based on your transaction data to provide:
        <ul>
            <li>Daily treasury summaries</li>
            <li>Exception remediation suggestions</li>
            <li>Route optimization recommendations</li>
            <li>Cost anomaly analysis</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Section selector
        ai_section = st.selectbox(
            "Select Analysis Type",
            [
                "📋 Daily Treasury Summary",
                "⚠️ Exception Analysis & Remediation",
                "🗺️ Route Optimization Recommendations",
                "💰 Cost Anomaly Analysis",
                "📊 Executive Insights"
            ]
        )
        
        st.markdown("---")
        
        # Daily Treasury Summary
        if ai_section == "📋 Daily Treasury Summary":
            st.markdown("### 📋 Daily Treasury Summary")
            
            col1, col2 = st.columns([3, 1])
            
            with col2:
                if st.button("🔄 Generate Summary", type="primary"):
                    with st.spinner("Generating AI summary..."):
                        # Prepare data
                        if 'timestamp' in filtered_df.columns and filtered_df['timestamp'].notna().any():
                            # Get yesterday's data for comparison
                            max_date = filtered_df['timestamp'].dropna().max()
                            yesterday = max_date - timedelta(days=1)
                            
                            today_df = filtered_df[filtered_df['timestamp'].dt.date == max_date.date()]
                            yesterday_df = filtered_df[filtered_df['timestamp'].dt.date == yesterday.date()]
                            
                            if len(today_df) > 0:
                                summary_data = prepare_daily_summary_data(today_df, yesterday_df if len(yesterday_df) > 0 else None)
                                summary = ai_engine.generate_daily_summary(summary_data)
                                
                                st.session_state['daily_summary'] = summary
                                st.session_state['summary_timestamp'] = datetime.now()
                            else:
                                st.warning("No data available for today")
                        else:
                            summary_data = prepare_daily_summary_data(filtered_df)
                            summary = ai_engine.generate_daily_summary(summary_data)
                            
                            st.session_state['daily_summary'] = summary
                            st.session_state['summary_timestamp'] = datetime.now()
            
            with col1:
                if 'daily_summary' in st.session_state:
                    st.markdown("#### AI-Generated Summary")
                    st.info(f"*Generated: {st.session_state['summary_timestamp'].strftime('%Y-%m-%d %H:%M:%S')}*")
                    st.markdown(st.session_state['daily_summary'])
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Summary",
                        data=st.session_state['daily_summary'],
                        file_name=f"treasury_summary_{datetime.now().strftime('%Y%m%d')}.txt",
                        mime="text/plain"
                    )
                else:
                    st.info("👆 Click 'Generate Summary' to create an AI-powered daily treasury summary")
        
        # Exception Analysis
        elif ai_section == "⚠️ Exception Analysis & Remediation":
            st.markdown("### ⚠️ Exception Analysis & Remediation")
            
            col1, col2 = st.columns([3, 1])
            
            with col2:
                if st.button("🔍 Analyze Exceptions", type="primary"):
                    with st.spinner("Analyzing exceptions..."):
                        exceptions = identify_exceptions(filtered_df)
                        
                        if exceptions:
                            analysis = ai_engine.analyze_exceptions(exceptions)
                            st.session_state['exception_analysis'] = analysis
                            st.session_state['exceptions_data'] = exceptions
                            st.session_state['exception_timestamp'] = datetime.now()
                        else:
                            st.session_state['exception_analysis'] = "✅ No exceptions found! All transactions are performing within normal parameters."
                            st.session_state['exceptions_data'] = []
            
            with col1:
                if st.session_state.get('exceptions_data') is not None:
                    # Show exception summary
                    st.markdown("#### Exception Summary")
                    
                    if st.session_state['exceptions_data']:
                        exception_df = pd.DataFrame(st.session_state['exceptions_data'])
                        
                        # Display metrics (guard against many exceptions)
                        cols = st.columns(min(len(st.session_state['exceptions_data']), 4))
                        for idx, exc in enumerate(st.session_state['exceptions_data'][: len(cols)]):
                            with cols[idx]:
                                severity_color = {
                                    'High': '🔴',
                                    'Medium': '🟡',
                                    'Low': '🟢'
                                }
                                count = exc.get('count', 0)
                                st.metric(
                                    f"{severity_color.get(exc.get('severity'), '⚪')} {exc.get('type', 'Exception')}", 
                                    f"{count}",
                                    delta=str(exc.get('severity', ''))
                                )
                        
                        st.markdown("---")
                    
                    if 'exception_analysis' in st.session_state:
                        st.markdown("#### AI Remediation Recommendations")
                        st.info(f"*Generated: {st.session_state.get('exception_timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S')}*")
                        st.markdown(st.session_state['exception_analysis'])
                        
                        # Download button
                        st.download_button(
                            label="📥 Download Remediation Plan",
                            data=st.session_state['exception_analysis'],
                            file_name=f"exception_remediation_{datetime.now().strftime('%Y%m%d')}.txt",
                            mime="text/plain"
                        )
                else:
                    st.info("👆 Click 'Analyze Exceptions' to identify issues and get AI-powered remediation suggestions")
        
        # Route Optimization
        elif ai_section == "🗺️ Route Optimization Recommendations":
            st.markdown("### 🗺️ Route Optimization Recommendations")
            
            col1, col2 = st.columns([3, 1])
            
            with col2:
                if st.button("🎯 Get Recommendations", type="primary"):
                    with st.spinner("Analyzing routes..."):
                        route_data = prepare_route_optimization_data(filtered_df)
                        
                        if route_data:
                            recommendations = ai_engine.generate_optimization_recommendations(route_data)
                            st.session_state['route_recommendations'] = recommendations
                            st.session_state['route_timestamp'] = datetime.now()
                        else:
                            st.warning("Insufficient route data for analysis")
            
            with col1:
                if 'route_recommendations' in st.session_state:
                    st.markdown("#### AI-Generated Recommendations")
                    st.info(f"*Generated: {st.session_state.get('route_timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S')}*")
                    st.markdown(st.session_state['route_recommendations'])
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Recommendations",
                        data=st.session_state['route_recommendations'],
                        file_name=f"route_optimization_{datetime.now().strftime('%Y%m%d')}.txt",
                        mime="text/plain"
                    )
                else:
                    st.info("👆 Click 'Get Recommendations' to receive AI-powered route optimization suggestions")
        
        # Cost Anomaly Analysis
        elif ai_section == "💰 Cost Anomaly Analysis":
            st.markdown("### 💰 Cost Anomaly Analysis")
            
            col1, col2 = st.columns([3, 1])
            
            with col2:
                percentile = st.slider("Anomaly Threshold", 90, 99, 95, help="Percentile threshold for identifying high-cost transactions")
                
                if st.button("🔍 Analyze Anomalies", type="primary"):
                    with st.spinner("Identifying cost anomalies..."):
                        anomalies = identify_cost_anomalies(filtered_df, percentile)
                        
                        if anomalies:
                            analysis = ai_engine.analyze_cost_anomalies(anomalies)
                            st.session_state['anomaly_analysis'] = analysis
                            st.session_state['anomalies_data'] = anomalies
                            st.session_state['anomaly_timestamp'] = datetime.now()
                        else:
                            st.session_state['anomaly_analysis'] = "No significant cost anomalies detected."
                            st.session_state['anomalies_data'] = []
            
            with col1:
                if st.session_state.get('anomalies_data'):
                    # Show anomaly summary
                    st.markdown("#### High-Cost Transactions")
                    
                    anomaly_df = pd.DataFrame(st.session_state['anomalies_data'])
                    st.dataframe(anomaly_df, use_container_width=True)
                    
                    st.markdown("---")
                
                if 'anomaly_analysis' in st.session_state:
                    st.markdown("#### AI Analysis")
                    st.info(f"*Generated: {st.session_state.get('anomaly_timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S')}*")
                    st.markdown(st.session_state['anomaly_analysis'])
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Analysis",
                        data=st.session_state['anomaly_analysis'],
                        file_name=f"cost_anomaly_analysis_{datetime.now().strftime('%Y%m%d')}.txt",
                        mime="text/plain"
                    )
                else:
                    st.info("👆 Click 'Analyze Anomalies' to identify and understand high-cost transactions")
        
        # Executive Insights
        elif ai_section == "📊 Executive Insights":
            st.markdown("### 📊 Executive-Level Insights")
            
            col1, col2 = st.columns([3, 1])
            
            with col2:
                if st.button("💼 Generate Insights", type="primary"):
                    with st.spinner("Generating executive insights..."):
                        # Prepare executive summary data
                        if 'timestamp' in filtered_df.columns and filtered_df['timestamp'].notna().any():
                            period = f"{filtered_df['timestamp'].dropna().min().date()} to {filtered_df['timestamp'].dropna().max().date()}"
                        else:
                            period = "Current period"
                        
                        exec_data = {
                            'period': period,
                            'total_volume_usd': float(safe_sum(filtered_df['amount_source'])) if 'amount_source' in filtered_df.columns else 0.0,
                            'total_transactions': len(filtered_df),
                            'avg_cost_bps': float(safe_mean(filtered_df['total_cost_bps'])) if 'total_cost_bps' in filtered_df.columns else 0.0,
                            'success_rate': float(((filtered_df['settlement_status'] == 'completed').sum() / len(filtered_df) * 100)) if 'settlement_status' in filtered_df.columns and len(filtered_df) > 0 else 0.0,
                            'total_fees_usd': float(safe_sum(filtered_df['total_fees_usd'])) if 'total_fees_usd' in filtered_df.columns else 0.0,
                            'avg_settlement_time_min': float(safe_mean(filtered_df['settlement_time_sec']) / 60) if 'settlement_time_sec' in filtered_df.columns else 0.0,
                            'compliance_rate': float(((filtered_df['compliance_passed'] == True).sum() / len(filtered_df) * 100)) if 'compliance_passed' in filtered_df.columns and len(filtered_df) > 0 else 100.0
                        }
                        
                        insights = ai_engine.generate_executive_insights(exec_data)
                        st.session_state['executive_insights'] = insights
                        st.session_state['exec_timestamp'] = datetime.now()
            
            with col1:
                if 'executive_insights' in st.session_state:
                    st.markdown("#### Strategic Insights for Leadership")
                    st.info(f"*Generated: {st.session_state.get('exec_timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S')}*")
                    
                    st.markdown("""
                    <div class="alert-green">
                    """ + st.session_state['executive_insights'] + """
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Executive Summary",
                        data=st.session_state['executive_insights'],
                        file_name=f"executive_insights_{datetime.now().strftime('%Y%m%d')}.txt",
                        mime="text/plain"
                    )
                else:
                    st.info("👆 Click 'Generate Insights' to create an executive-level summary suitable for C-suite")
        
        # Usage tips
        st.markdown("---")
        st.markdown("""
        ### 💡 Tips for Using AI Insights
        
        - **Daily Summary**: Run every morning for a quick overview of treasury performance
        - **Exception Analysis**: Use when you have failures or anomalies to investigate
        - **Route Optimization**: Review weekly to identify improvement opportunities
        - **Cost Anomalies**: Investigate unexpected high-cost transactions
        - **Executive Insights**: Generate before leadership meetings or board presentations
        
        All AI-generated content can be downloaded and shared with your team.
        """)
else:
    # AI tab not available, adjust tab references
    if analysis_mode == "Optimization Comparison":
        first_data_tab = tab2
    else:
        first_data_tab = tab1

# Determine which tab to start with based on mode
first_data_tab = tab2 if analysis_mode == "Optimization Comparison" else tab1

# TAB 1: Executive Summary
with first_data_tab:
    st.header("Executive Summary")
    
    # Critical Alerts
    if 'settlement_status' in filtered_df.columns:
        failed_txns = filtered_df[filtered_df['settlement_status'] == 'failed']
    else:
        failed_txns = pd.DataFrame()
    
    if 'compliance_passed' in filtered_df.columns:
        compliance_fails = filtered_df[filtered_df['compliance_passed'] == False]
    else:
        compliance_fails = pd.DataFrame()
    
    if 'total_cost_bps' in filtered_df.columns and 'max_acceptable_fee_bps' in filtered_df.columns:
        high_cost_txns = filtered_df[filtered_df['total_cost_bps'] > filtered_df['max_acceptable_fee_bps']]
    else:
        high_cost_txns = pd.DataFrame()
    
    if len(failed_txns) > 0 or len(compliance_fails) > 0 or len(high_cost_txns) > 0:
        st.markdown("### 🚨 Critical Alerts")
        
        col_alert1, col_alert2, col_alert3 = st.columns(3)
        
        with col_alert1:
            if len(failed_txns) > 0:
                st.markdown(f"""
                <div class="alert-red">
                    <strong>⚠️ Failed Transactions: {len(failed_txns)}</strong><br>
                    {(len(failed_txns)/len(filtered_df)*100):.1f}% failure rate
                </div>
                """, unsafe_allow_html=True)
        
        with col_alert2:
            if len(compliance_fails) > 0:
                st.markdown(f"""
                <div class="alert-red">
                    <strong>🔒 Compliance Failures: {len(compliance_fails)}</strong><br>
                    {(len(compliance_fails)/len(filtered_df)*100):.1f}% non-compliant
                </div>
                """, unsafe_allow_html=True)
        
        with col_alert3:
            if len(high_cost_txns) > 0:
                st.markdown(f"""
                <div class="alert-red">
                    <strong>💸 Fee Limit Exceeded: {len(high_cost_txns)}</strong><br>
                    {(len(high_cost_txns)/len(filtered_df)*100):.1f}% over limit
                </div>
                """, unsafe_allow_html=True)
    
    # Top-line KPIs
    st.markdown("### 📈 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'total_cost_bps' in filtered_df.columns:
            avg_cost = safe_mean(filtered_df['total_cost_bps'])
            if analysis_mode == "Optimization Comparison" and filtered_df_before is not None and 'total_cost_bps' in filtered_df_before.columns:
                before_cost = safe_mean(filtered_df_before['total_cost_bps'])
                delta = f"{calculate_improvement(before_cost, avg_cost):.1f}%"
            else:
                delta = None
            st.metric("Avg Cost (BPS)", format_val_or_na(avg_cost, "{:.2f}"), delta, delta_color="inverse")
        else:
            st.metric("Avg Cost (BPS)", "N/A")
    
    with col2:
        if 'settlement_status' in filtered_df.columns:
            success_rate = ((filtered_df['settlement_status'] == 'completed').sum() / len(filtered_df) * 100) if len(filtered_df) > 0 else 0.0
            if analysis_mode == "Optimization Comparison" and filtered_df_before is not None and 'settlement_status' in filtered_df_before.columns:
                before_success = ((filtered_df_before['settlement_status'] == 'completed').sum() / len(filtered_df_before) * 100) if len(filtered_df_before) > 0 else 0.0
                delta = f"+{success_rate - before_success:.1f}%"
            else:
                delta = None
            st.metric("Success Rate", format_val_or_na(success_rate, "{:.1f}") + "%" if success_rate else "N/A", delta)
        else:
            st.metric("Success Rate", "N/A")
    
    with col3:
        if 'settlement_time_sec' in filtered_df.columns:
            avg_time = safe_mean(filtered_df['settlement_time_sec']) / 60
            if analysis_mode == "Optimization Comparison" and filtered_df_before is not None and 'settlement_time_sec' in filtered_df_before.columns:
                before_time = safe_mean(filtered_df_before['settlement_time_sec']) / 60
                delta = f"{calculate_improvement(before_time, avg_time):.1f}%"
            else:
                delta = None
            st.metric("Avg Settlement Time", format_val_or_na(avg_time, "{:.1f}") + " min" if avg_time else "N/A", delta, delta_color="inverse")
        else:
            st.metric("Avg Settlement Time", "N/A")
    
    with col4:
        if 'amount_source' in filtered_df.columns:
            total_volume = safe_sum(filtered_df['amount_source']) / 1_000_000
            st.metric("Total Volume", f"${total_volume:.2f}M" if total_volume else "N/A")
        else:
            st.metric("Total Volume", "N/A")
    
    # Additional metrics row
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        txn_count = len(filtered_df)
        st.metric("Transaction Count", f"{txn_count:,}")
    
    with col6:
        if 'routing_hops' in filtered_df.columns:
            avg_hops = safe_mean(filtered_df['routing_hops'])
            st.metric("Avg Routing Hops", format_val_or_na(avg_hops, "{:.2f}") if avg_hops else "N/A")
        else:
            st.metric("Avg Routing Hops", "N/A")
    
    with col7:
        if 'compliance_passed' in filtered_df.columns:
            compliance_rate = ((filtered_df['compliance_passed'] == True).sum() / len(filtered_df) * 100) if len(filtered_df) > 0 else 0.0
            st.metric("Compliance Rate", format_val_or_na(compliance_rate, "{:.1f}") + "%" if compliance_rate else "N/A")
        else:
            st.metric("Compliance Rate", "N/A")
    
    with col8:
        if 'total_fees_usd' in filtered_df.columns:
            total_fees = safe_sum(filtered_df['total_fees_usd']) / 1000
            st.metric("Total Fees", f"${total_fees:.1f}K" if total_fees else "N/A")
        else:
            st.metric("Total Fees", "N/A")
    
    st.markdown("---")
    
    # Transaction volume over time
    if 'timestamp' in filtered_df.columns and filtered_df['timestamp'].notna().any():
        st.markdown("### 📊 Transaction Volume Trends")
        
        # Build dynamic agg_map
        agg_map = {}
        # count
        id_col = None
        for possible_id in ['transfer_id', 'transaction_id', 'txn_id', 'id']:
            if possible_id in filtered_df.columns:
                id_col = possible_id
                break
        if id_col is not None:
            agg_map[id_col] = 'count'
        else:
            # fallback to counting rows via any column
            agg_map[filtered_df.columns[0]] = 'count'
        
        if 'amount_source' in filtered_df.columns:
            agg_map['amount_source'] = 'sum'
        if 'total_cost_bps' in filtered_df.columns:
            agg_map['total_cost_bps'] = 'mean'
        if 'settlement_time_sec' in filtered_df.columns:
            agg_map['settlement_time_sec'] = 'mean'
        
        # safe groupby
        try:
            daily_data = filtered_df.groupby(filtered_df['timestamp'].dt.date).agg(agg_map).reset_index()
        except Exception:
            daily_data = pd.DataFrame()
        
        # Rename columns dynamically
        if not daily_data.empty:
            col_map = {}
            col_map[daily_data.columns[0]] = 'date'
            idx = 1
            # assign names in the order inserted into agg_map (dict preserves insertion order in py3.7+)
            for key in list(agg_map.keys()):
                if idx < len(daily_data.columns):
                    if key == id_col or key == filtered_df.columns[0]:
                        col_map[daily_data.columns[idx]] = 'count'
                    elif key == 'amount_source':
                        col_map[daily_data.columns[idx]] = 'volume'
                    elif key == 'total_cost_bps':
                        col_map[daily_data.columns[idx]] = 'avg_cost_bps'
                    elif key == 'settlement_time_sec':
                        col_map[daily_data.columns[idx]] = 'avg_time_sec'
                    idx += 1
            daily_data = daily_data.rename(columns=col_map)
            
            # Fill missing expected columns with zeros to avoid plotting errors
            for expected_col in ['count', 'volume', 'avg_cost_bps', 'avg_time_sec']:
                if expected_col not in daily_data.columns:
                    daily_data[expected_col] = 0
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Daily Transaction Count', 'Daily Volume ($)', 
                               'Avg Cost (BPS)', 'Avg Settlement Time (min)'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            fig.add_trace(
                go.Scatter(x=daily_data['date'], y=daily_data['count'], 
                          name='Txn Count', fill='tozeroy'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=daily_data['date'], y=daily_data['volume'], 
                          name='Volume', fill='tozeroy'),
                row=1, col=2
            )
            
            if 'avg_cost_bps' in daily_data.columns:
                fig.add_trace(
                    go.Scatter(x=daily_data['date'], y=daily_data['avg_cost_bps'], 
                              name='Avg Cost'),
                    row=2, col=1
                )
            
            if 'avg_time_sec' in daily_data.columns:
                fig.add_trace(
                    go.Scatter(x=daily_data['date'], y=daily_data['avg_time_sec']/60, 
                              name='Avg Time'),
                    row=2, col=2
                )
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough timestamped data to build daily trends.")
    else:
        st.info("Timestamp data not available for time series charts.")

# TAB: Cost Analysis
if ai_enabled:
    cost_tab = tab4 if analysis_mode == "Optimization Comparison" else tab3
else:
    cost_tab = tab3 if analysis_mode == "Optimization Comparison" else tab2

with cost_tab:
    st.header("Cost Analysis Dashboard")
    
    # Cost breakdown metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'gas_cost_usd' in filtered_df.columns:
            total_gas = safe_sum(filtered_df['gas_cost_usd'])
            st.metric("Total Gas Costs", f"${total_gas:,.2f}" if total_gas else "N/A")
        else:
            st.metric("Total Gas Costs", "N/A")
    
    with col2:
        if 'lp_fee_usd' in filtered_df.columns:
            total_lp = safe_sum(filtered_df['lp_fee_usd'])
            st.metric("Total LP Fees", f"${total_lp:,.2f}" if total_lp else "N/A")
        else:
            st.metric("Total LP Fees", "N/A")
    
    with col3:
        if 'bridge_cost_usd' in filtered_df.columns:
            total_bridge = safe_sum(filtered_df['bridge_cost_usd'])
            st.metric("Total Bridge Costs", f"${total_bridge:,.2f}" if total_bridge else "N/A")
        else:
            st.metric("Total Bridge Costs", "N/A")
    
    with col4:
        if 'slippage_cost_usd' in filtered_df.columns:
            total_slippage = safe_sum(filtered_df['slippage_cost_usd'])
            st.metric("Total Slippage", f"${total_slippage:,.2f}" if total_slippage else "N/A")
        else:
            st.metric("Total Slippage", "N/A")
    
    st.markdown("---")
    
    # Cost breakdown by component
    st.markdown("### 💵 Cost Component Breakdown")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart of cost components
        if all(col in filtered_df.columns for col in ['gas_cost_usd', 'lp_fee_usd', 'bridge_cost_usd', 'slippage_cost_usd']):
            cost_components = pd.DataFrame({
                'Component': ['Gas', 'LP Fees', 'Bridge', 'Slippage'],
                'Amount': [
                    safe_sum(filtered_df['gas_cost_usd']),
                    safe_sum(filtered_df['lp_fee_usd']),
                    safe_sum(filtered_df['bridge_cost_usd']),
                    safe_sum(filtered_df['slippage_cost_usd'])
                ]
            })
            
            if cost_components['Amount'].sum() > 0:
                fig = px.pie(cost_components, values='Amount', names='Component',
                            title='Cost Distribution by Component')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Cost component totals are zero.")
        else:
            st.info("Cost component data not available")
    
    with col2:
        # Box plot of costs by business type
        if 'total_cost_bps' in filtered_df.columns and 'business_type' in filtered_df.columns:
            plot_df = filtered_df[['business_type', 'total_cost_bps']].dropna()
            if not plot_df.empty:
                fig = px.box(plot_df, x='business_type', y='total_cost_bps',
                            title='Cost Distribution by Business Type',
                            color='business_type')
                fig.update_xaxis(tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough data to plot cost by business type")
        else:
            st.info("Cost by business type data not available")
    
    # Cost efficiency table
    if 'business_type' in filtered_df.columns and 'total_cost_bps' in filtered_df.columns:
        st.markdown("### 📋 Cost Efficiency by Transaction Type")
        
        agg_dict = {'transfer_id': 'count'} if 'transfer_id' in filtered_df.columns else {filtered_df.columns[0]: 'count'}
        if 'total_cost_bps' in filtered_df.columns:
            agg_dict['total_cost_bps'] = ['mean', 'median', 'std']
        if 'total_fees_usd' in filtered_df.columns:
            agg_dict['total_fees_usd'] = 'sum'
        if 'amount_source' in filtered_df.columns:
            agg_dict['amount_source'] = 'sum'
        
        try:
            cost_summary = filtered_df.groupby('business_type').agg(agg_dict).round(2)
            # Flatten column names
            cost_summary.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col 
                                   for col in cost_summary.columns.values]
            cost_summary = cost_summary.reset_index()
            st.dataframe(cost_summary, use_container_width=True)
        except Exception:
            st.info("Unable to compute cost efficiency table with available data.")

# TAB: Performance Metrics  
if ai_enabled:
    perf_tab = tab5 if analysis_mode == "Optimization Comparison" else tab4
else:
    perf_tab = tab4 if analysis_mode == "Optimization Comparison" else tab3

with perf_tab:
    st.header("Performance Metrics Dashboard")
    
    # Key performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'settlement_status' in filtered_df.columns:
            completed = (filtered_df['settlement_status'] == 'completed').sum()
            completion_rate = (completed / len(filtered_df) * 100) if len(filtered_df) > 0 else 0.0
            st.metric("Completion Rate", format_val_or_na(completion_rate, "{:.1f}") + "%" if completion_rate else "N/A")
        else:
            st.metric("Completion Rate", "N/A")
    
    with col2:
        if 'settlement_status' in filtered_df.columns:
            failed = (filtered_df['settlement_status'] == 'failed').sum()
            failure_rate = (failed / len(filtered_df) * 100) if len(filtered_df) > 0 else 0.0
            st.metric("Failure Rate", format_val_or_na(failure_rate, "{:.1f}") + "%" if failure_rate else "N/A")
        else:
            st.metric("Failure Rate", "N/A")
    
    with col3:
        if 'settlement_status' in filtered_df.columns:
            pending = (filtered_df['settlement_status'] == 'pending').sum()
            pending_rate = (pending / len(filtered_df) * 100) if len(filtered_df) > 0 else 0.0
            st.metric("Pending Rate", format_val_or_na(pending_rate, "{:.1f}") + "%" if pending_rate else "N/A")
        else:
            st.metric("Pending Rate", "N/A")
    
    with col4:
        if 'settlement_time_sec' in filtered_df.columns:
            avg_settlement = safe_mean(filtered_df['settlement_time_sec'])
            st.metric("Avg Settlement Time", format_val_or_na(avg_settlement/60 if avg_settlement else None, "{:.1f}") + " min" if avg_settlement else "N/A")
        else:
            st.metric("Avg Settlement Time", "N/A")
    
    st.markdown("---")
    
    # Settlement time analysis
    st.markdown("### ⏱️ Settlement Time Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Settlement time by urgency
        if 'urgency_level' in filtered_df.columns and 'settlement_time_sec' in filtered_df.columns:
            plot_df = filtered_df[['urgency_level', 'settlement_time_sec']].dropna()
            if not plot_df.empty:
                # Ensure category orders exists safely
                category_orders = {'urgency_level': ['urgent', 'standard', 'low']}
                fig = px.box(plot_df, x='urgency_level', y='settlement_time_sec',
                            title='Settlement Time by Urgency Level',
                            color='urgency_level',
                            category_orders=category_orders)
                fig.update_yaxis(title='Settlement Time (seconds)')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough data for settlement time by urgency")
        else:
            st.info("Settlement time by urgency data not available")
    
    with col2:
        # Settlement time by routing hops
        if 'routing_hops' in filtered_df.columns and 'settlement_time_sec' in filtered_df.columns:
            hop_time = filtered_df.groupby('routing_hops').agg({
                'settlement_time_sec': ['mean', 'count']
            }).reset_index()
            hop_time.columns = ['Hops', 'Avg Time (sec)', 'Count']
            if not hop_time.empty:
                fig = px.bar(hop_time, x='Hops', y='Avg Time (sec)',
                            title='Avg Settlement Time by Routing Hops',
                            text='Count', color='Avg Time (sec)',
                            color_continuous_scale='Reds')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough data to plot settlement time by hops")
        else:
            st.info("Settlement time by routing hops data not available")
    
    # Success rate analysis
    st.markdown("### 📊 Success Rate Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Success by user tier
        if 'user_tier' in filtered_df.columns and 'settlement_status' in filtered_df.columns:
            try:
                success_by_tier = filtered_df.groupby('user_tier')['settlement_status'].apply(
                    lambda x: (x == 'completed').sum() / len(x) * 100 if len(x) > 0 else 0.0
                ).reset_index()
                success_by_tier.columns = ['user_tier', 'success_rate']
                if not success_by_tier.empty:
                    fig = px.bar(success_by_tier, x='user_tier', y='success_rate',
                                title='Success Rate by User Tier',
                                text='success_rate', color='success_rate',
                                color_continuous_scale='RdYlGn', range_color=[0, 100])
                    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No success-by-tier data to display")
            except Exception:
                st.info("Could not compute success rate by user tier.")
        else:
            st.info("Success rate by user tier data not available")
    
    with col2:
        # Success by liquidity availability
        if 'liquidity_available' in filtered_df.columns and 'settlement_status' in filtered_df.columns:
            try:
                success_by_liquidity = filtered_df.groupby('liquidity_available')['settlement_status'].apply(
                    lambda x: (x == 'completed').sum() / len(x) * 100 if len(x) > 0 else 0.0
                ).reset_index()
                success_by_liquidity.columns = ['liquidity_available', 'success_rate']
                success_by_liquidity['liquidity_available'] = success_by_liquidity['liquidity_available'].map(
                    {True: 'Available', False: 'Not Available'}
                ).fillna(success_by_liquidity['liquidity_available'].astype(str))
                
                if not success_by_liquidity.empty:
                    fig = px.bar(success_by_liquidity, x='liquidity_available', y='success_rate',
                                title='Success Rate by Liquidity Availability',
                                text='success_rate', color='success_rate',
                                color_continuous_scale='RdYlGn', range_color=[0, 100])
                    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No success-by-liquidity data to display")
            except Exception:
                st.info("Could not compute success rate by liquidity.")
        else:
            st.info("Success rate by liquidity data not available")

# TAB: Route Intelligence
if ai_enabled:
    route_tab = tab6 if analysis_mode == "Optimization Comparison" else tab5
else:
    route_tab = tab5 if analysis_mode == "Optimization Comparison" else tab4

with route_tab:
    st.header("Route Intelligence Dashboard")
    
    # Route metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'source_chain' in filtered_df.columns and 'dest_chain' in filtered_df.columns:
            unique_routes = filtered_df.groupby(['source_chain', 'dest_chain']).size().shape[0]
            st.metric("Unique Route Pairs", unique_routes)
        else:
            st.metric("Unique Route Pairs", "N/A")
    
    with col2:
        if 'routing_hops' in filtered_df.columns:
            avg_hops = safe_mean(filtered_df['routing_hops'])
            st.metric("Avg Routing Hops", format_val_or_na(avg_hops, "{:.2f}") if avg_hops else "N/A")
        else:
            st.metric("Avg Routing Hops", "N/A")
    
    with col3:
        if 'routing_hops' in filtered_df.columns:
            single_hop = (filtered_df['routing_hops'] == 1).sum()
            single_hop_pct = (single_hop / len(filtered_df) * 100) if len(filtered_df) > 0 else 0.0
            st.metric("Single-Hop Routes", format_val_or_na(single_hop_pct, "{:.1f}") + "%" if single_hop_pct else "N/A")
        else:
            st.metric("Single-Hop Routes", "N/A")
    
    with col4:
        if 'routing_hops' in filtered_df.columns:
            multi_hop = (filtered_df['routing_hops'] >= 3).sum()
            multi_hop_pct = (multi_hop / len(filtered_df) * 100) if len(filtered_df) > 0 else 0.0
            st.metric("Multi-Hop (3+) Routes", format_val_or_na(multi_hop_pct, "{:.1f}") + "%" if multi_hop_pct else "N/A")
        else:
            st.metric("Multi-Hop (3+) Routes", "N/A")
    
    st.markdown("---")
    
    # Chain pair analysis
    if all(col in filtered_df.columns for col in ['source_chain', 'dest_chain', 'total_cost_bps']):
        st.markdown("### 🗺️ Route Performance by Chain Pair")
        
        agg_map = {
            'total_cost_bps': 'mean',
            'transfer_id': 'count' if 'transfer_id' in filtered_df.columns else filtered_df.columns[0],
        }
        if 'settlement_time_sec' in filtered_df.columns:
            agg_map['settlement_time_sec'] = 'mean'
        
        try:
            route_data = filtered_df.groupby(['source_chain', 'dest_chain']).agg(agg_map).reset_index()
            # normalize column names
            col_names = ['Source Chain', 'Dest Chain', 'Avg Cost (BPS)', 'Count']
            if 'settlement_time_sec' in agg_map:
                col_names.append('Avg Time (sec)')
            # ensure we don't mismatch columns
            while len(col_names) < len(route_data.columns):
                col_names.append(route_data.columns[len(col_names)])
            route_data.columns = col_names[:len(route_data.columns)]
            
            # Display top routes
            st.markdown("#### Top 10 Routes by Volume")
            top_routes = route_data.nlargest(10, 'Count')[
                ['Source Chain', 'Dest Chain', 'Count', 'Avg Cost (BPS)'] + (['Avg Time (sec)'] if 'Avg Time (sec)' in route_data.columns else [])
            ]
            st.dataframe(top_routes, use_container_width=True)
            
            # Scatter plot of routes
            if 'Avg Time (sec)' in route_data.columns:
                fig = px.scatter(route_data, x='Avg Cost (BPS)', y='Avg Time (sec)',
                                size='Count', color='Count',
                                hover_data=['Source Chain', 'Dest Chain'],
                                title='Route Performance: Cost vs Speed')
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = px.scatter(route_data, x='Avg Cost (BPS)', y='Count',
                                size='Count', color='Count',
                                hover_data=['Source Chain', 'Dest Chain'],
                                title='Route Performance: Cost vs Volume')
                st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.info("Could not compute route performance with available data.")
    
    # Venue performance
    if 'venues_used' in filtered_df.columns:
        st.markdown("### 🏦 Venue Performance Analysis")
        
        # Extract and count venue usage robustly
        all_venues = []
        for venues in filtered_df['venues_used'].dropna():
            items = [v.strip() for v in str(venues).split(',') if v.strip()]
            all_venues.extend(items)
        
        if all_venues:
            venue_counts = pd.Series(all_venues).value_counts().reset_index()
            venue_counts.columns = ['Venue', 'Count']
            
            # Calculate average cost per venue (top 10)
            venue_stats = []
            for venue in venue_counts['Venue'][:10]:
                mask = filtered_df['venues_used'].fillna('').str.contains(venue, na=False)
                venue_txns = filtered_df[mask]
                if len(venue_txns) > 0:
                    venue_stat = {
                        'Venue': venue,
                        'Count': len(venue_txns)
                    }
                    
                    if 'total_cost_bps' in venue_txns.columns:
                        venue_stat['Avg Cost (BPS)'] = safe_mean(venue_txns['total_cost_bps'])
                    if 'settlement_time_sec' in venue_txns.columns:
                        venue_stat['Avg Time (min)'] = safe_mean(venue_txns['settlement_time_sec']) / 60
                    if 'settlement_status' in venue_txns.columns:
                        venue_stat['Success Rate (%)'] = (venue_txns['settlement_status'] == 'completed').sum() / len(venue_txns) * 100
                    if 'amount_source' in venue_txns.columns:
                        venue_stat['Total Volume ($)'] = safe_sum(venue_txns['amount_source'])
                    
                    venue_stats.append(venue_stat)
            
            if venue_stats:
                venue_df = pd.DataFrame(venue_stats).round(2)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.bar(venue_df.sort_values('Count', ascending=False), 
                                x='Venue', y='Count',
                                title='Transaction Volume by Venue',
                                color='Count')
                    fig.update_xaxis(tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if 'Avg Cost (BPS)' in venue_df.columns and 'Avg Time (min)' in venue_df.columns:
                        fig = px.scatter(venue_df, x='Avg Cost (BPS)', y='Avg Time (min)',
                                        size='Count', 
                                        color='Success Rate (%)' if 'Success Rate (%)' in venue_df.columns else 'Count',
                                        hover_data=['Venue'],
                                        title='Venue Performance: Cost vs Speed')
                        st.plotly_chart(fig, use_container_width=True)
                
                # Venue performance table
                st.markdown("### 📊 Detailed Venue Statistics")
                st.dataframe(venue_df.sort_values('Count', ascending=False), use_container_width=True)
            else:
                st.info("No venue statistics available")
        else:
            st.info("No venues used data available")

# TAB: Regional & Compliance
if ai_enabled:
    regional_tab = tab7 if analysis_mode == "Optimization Comparison" else tab6
else:
    regional_tab = tab6 if analysis_mode == "Optimization Comparison" else tab5

with regional_tab:
    st.header("Regional & Compliance Dashboard")
    
    # Regional metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'region' in filtered_df.columns:
            unique_regions = filtered_df['region'].nunique()
            st.metric("Active Regions", unique_regions)
        else:
            st.metric("Active Regions", "N/A")
    
    with col2:
        if 'beneficiary_country' in filtered_df.columns:
            unique_countries = filtered_df['beneficiary_country'].nunique()
            st.metric("Beneficiary Countries", unique_countries)
        else:
            st.metric("Beneficiary Countries", "N/A")
    
    with col3:
        if 'compliance_passed' in filtered_df.columns:
            compliance_rate = ((filtered_df['compliance_passed'] == True).sum() / len(filtered_df) * 100) if len(filtered_df) > 0 else 0.0
            st.metric("Compliance Rate", format_val_or_na(compliance_rate, "{:.1f}") + "%" if compliance_rate else "N/A")
        else:
            st.metric("Compliance Rate", "N/A")
    
    with col4:
        if 'kyc_status' in filtered_df.columns:
            kyc_verified = ((filtered_df['kyc_status'] == 'verified').sum() / len(filtered_df) * 100) if len(filtered_df) > 0 else 0.0
            st.metric("KYC Verified", format_val_or_na(kyc_verified, "{:.1f}") + "%" if kyc_verified else "N/A")
        else:
            st.metric("KYC Verified", "N/A")
    
    st.markdown("---")
    
    # Regional performance
    if 'region' in filtered_df.columns:
        st.markdown("### 🌍 Performance by Region")
        
        # Build aggregation dictionary with safe column checking
        agg_dict = {}
        
        # Use any available ID column for counting, or first column as fallback
        count_col = None
        for possible_id in ['transfer_id', 'transaction_id', 'txn_id', 'id']:
            if possible_id in filtered_df.columns:
                count_col = possible_id
                break
        
        if count_col is None:
            # Use first column as fallback
            count_col = filtered_df.columns[0]
        
        agg_dict[count_col] = 'count'
        
        if 'amount_source' in filtered_df.columns:
            agg_dict['amount_source'] = 'sum'
        if 'total_cost_bps' in filtered_df.columns:
            agg_dict['total_cost_bps'] = 'mean'
        if 'settlement_time_sec' in filtered_df.columns:
            agg_dict['settlement_time_sec'] = 'mean'
        if 'settlement_status' in filtered_df.columns:
            agg_dict['settlement_status'] = lambda x: (x == 'completed').sum() / len(x) * 100 if len(x) > 0 else 0.0
        if 'compliance_passed' in filtered_df.columns:
            agg_dict['compliance_passed'] = lambda x: (x == True).sum() / len(x) * 100 if len(x) > 0 else 0.0
        
        try:
            regional_stats = filtered_df.groupby('region').agg(agg_dict).reset_index()
        except Exception:
            regional_stats = pd.DataFrame()
    
        # Rename columns properly
        if not regional_stats.empty:
            new_cols = ['Region', 'Count']
            col_idx = 2
            if 'amount_source' in filtered_df.columns:
                new_cols.append('Volume ($)')
                col_idx += 1
            if 'total_cost_bps' in filtered_df.columns:
                new_cols.append('Avg Cost (BPS)')
                col_idx += 1
            if 'settlement_time_sec' in filtered_df.columns:
                new_cols.append('Avg Time (sec)')
                col_idx += 1
            if 'settlement_status' in filtered_df.columns:
                new_cols.append('Success Rate (%)')
                col_idx += 1
            if 'compliance_passed' in filtered_df.columns:
                new_cols.append('Compliance Rate (%)')
                col_idx += 1
            # Ensure length match
            new_cols = new_cols[:len(regional_stats.columns)]
            regional_stats.columns = new_cols
            st.dataframe(regional_stats.sort_values('Count', ascending=False), use_container_width=True)
        else:
            st.info("Not enough regional data to display aggregated stats")
    
    # Beneficiary country analysis
    if 'beneficiary_country' in filtered_df.columns:
        st.markdown("### 🗺️ Top Beneficiary Countries")
        
        country_valid = filtered_df[filtered_df['beneficiary_country'].notna()]
        
        # Find count column
        count_col = None
        for possible_id in ['transfer_id', 'transaction_id', 'txn_id', 'id']:
            if possible_id in country_valid.columns:
                count_col = possible_id
                break
        if count_col is None:
            count_col = country_valid.columns[0]
        
        country_agg = {count_col: 'count'}
        if 'amount_source' in country_valid.columns:
            country_agg['amount_source'] = 'sum'
        if 'total_cost_bps' in country_valid.columns:
            country_agg['total_cost_bps'] = 'mean'
        if 'settlement_status' in country_valid.columns:
            country_agg['settlement_status'] = lambda x: (x == 'completed').sum() / len(x) * 100 if len(x) > 0 else 0.0
        
        try:
            country_data = country_valid.groupby('beneficiary_country').agg(country_agg).reset_index()
            # Normalize columns to expected names safely
            # Build the output columns list depending on what aggregated columns exist
            out_cols = ['Country']
            # We don't strictly know order; map them by inference
            # Build a mapping by position
            # The groupby output columns will be: [beneficiary_country, <agg1>, <agg2>, ...]
            # We'll create reasonable column names depending on which keys were in country_agg
            col_names = ['Country']
            for k in list(country_agg.keys()):
                if k == count_col:
                    col_names.append('Count')
                elif k == 'amount_source':
                    col_names.append('Volume ($)')
                elif k == 'total_cost_bps':
                    col_names.append('Avg Cost (BPS)')
                elif k == 'settlement_status':
                    col_names.append('Success Rate (%)')
            country_data.columns = col_names[:len(country_data.columns)]
            
            # Ensure we have a consistent set of columns for plotting
            expected_cols = ['Country', 'Count', 'Volume ($)', 'Avg Cost (BPS)', 'Success Rate (%)']
            # Add missing columns if needed
            for c in expected_cols:
                if c not in country_data.columns:
                    country_data[c] = 0
            country_data = country_data[expected_cols].sort_values('Count', ascending=False).head(15)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(country_data, x='Country', y='Count',
                            title='Top 15 Beneficiary Countries by Transaction Count',
                            color='Success Rate (%)')
                fig.update_xaxis(tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if country_data['Volume ($)'].sum() > 0:
                    fig = px.treemap(country_data, path=['Country'], values='Volume ($)',
                                    title='Transaction Volume Distribution by Country',
                                    color='Avg Cost (BPS)')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No volume data to display in treemap")
        except Exception:
            st.info("Could not compute beneficiary country aggregates.")
    
    # Compliance analysis
    if 'compliance_passed' in filtered_df.columns:
        st.markdown("### 🔒 Compliance Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Compliance by region
            if 'region' in filtered_df.columns:
                try:
                    compliance_by_region = filtered_df.groupby('region')['compliance_passed'].apply(
                        lambda x: (x == True).sum() / len(x) * 100 if len(x) > 0 else 0.0
                    ).reset_index()
                    compliance_by_region.columns = ['Region', 'Compliance Rate (%)']
                    
                    fig = px.bar(compliance_by_region, x='Region', y='Compliance Rate (%)',
                                title='Compliance Rate by Region',
                                color='Compliance Rate (%)', text='Compliance Rate (%)')
                    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    st.info("Could not compute compliance by region.")
            else:
                st.info("Region data not available for compliance breakdown")
        
        with col2:
            # KYC status distribution
            if 'kyc_status' in filtered_df.columns:
                kyc_dist = filtered_df['kyc_status'].value_counts()
                if not kyc_dist.empty:
                    fig = px.pie(values=kyc_dist.values, names=kyc_dist.index,
                                title='KYC Status Distribution')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No KYC data available")
            else:
                st.info("KYC status data not available")

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666;'>
    <p>Stablecoin Route Optimization Dashboard v2.0 | Data refreshed: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    <p>💡 Use filters in the sidebar to drill down into specific segments</p>
    <p>📁 Data files loaded from: {data_dir}</p>
</div>
""", unsafe_allow_html=True)
