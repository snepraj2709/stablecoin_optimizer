"""
Stablecoin Route Optimization Dashboard - Modular Version

Clean, modular dashboard using actual data structure
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import os

# Import our custom modules
from metrics_calculator import MetricsCalculator, get_comparison_metrics
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

# ==================== SIDEBAR ====================

st.sidebar.title("📁 Configuration")

# Data directory
data_dir = st.sidebar.text_input(
    "Data Directory",
    value="./config",
    help="Path to directory with CSV files"
)

# Initialize data loader
loader = DataLoader(data_dir)

# Check file status
st.sidebar.markdown("### 📊 Data Files")
file_status = loader.check_files_exist()
for name, exists in file_status.items():
    icon = "✅" if exists else "❌"
    st.sidebar.markdown(f"{icon} {name.title()}")

# Load data
optimized_df = loader.load_optimized_data()
baseline_df = loader.load_baseline_data()

if optimized_df is None:
    st.error("⚠️ No optimization results found! Please check your data directory.")
    st.stop()

# Analysis mode
st.sidebar.markdown("---")
st.sidebar.title("🎯 Analysis Mode")

if baseline_df is not None:
    analysis_mode = st.sidebar.radio(
        "Mode",
        ["Current Performance", "Optimization Comparison"],
        help="Compare before/after or view current results"
    )
else:
    analysis_mode = "Current Performance"
    st.sidebar.info("Only optimized data available")

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
        
        # Calculate comparison
        calc_baseline = MetricsCalculator(filtered_baseline)
        metrics_baseline = calc_baseline.get_summary_metrics()
        
        # Key improvements
        st.markdown("### 💡 Key Improvements")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            cost_improvement = metrics['avg_cost_improvement_bps']
            base = metrics_baseline.get('avg_cost_bps', 0)
            pct = (cost_improvement / base * 100) if base else 0
            st.metric("Cost Reduction", f"{cost_improvement:.1f} BPS", f"{pct:.1f}%")

        
        with col2:
            savings = metrics['total_savings_usd']
            st.metric(
                "Total Savings",
                f"${savings:,.2f}",
                "Saved"
            )
        
        with col3:
            success_rate = metrics['success_rate_pct']
            st.metric(
                "Success Rate",
                f"{success_rate:.1f}%",
                f"{success_rate - metrics_baseline.get('success_rate_pct', 100):.1f}pp"
            )
        
        with col4:
            avg_routes = metrics['avg_routes']
            st.metric(
                "Avg Routes",
                f"{avg_routes:.1f}",
                "Optimized"
            )
        
        # Comparison charts
        st.markdown("---")
        st.markdown("### 📊 Before vs After")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Cost comparison
            comparison_data = pd.DataFrame({
                'Metric': ['Before', 'After'],
                'Cost (BPS)': [metrics_baseline['avg_cost_bps'], metrics['avg_cost_bps']]
            })
            fig = px.bar(comparison_data, x='Metric', y='Cost (BPS)',
                        title='Average Cost Comparison',
                        color='Metric',
                        color_discrete_map={'Before': '#ff7f0e', 'After': '#2ca02c'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Time comparison
            comparison_time = pd.DataFrame({
                'Metric': ['Before', 'After'],
                'Time (min)': [
                    metrics_baseline.get('avg_settlement_time_sec', 0)/60,
                    metrics['avg_settlement_time_sec']/60
                ]
            })
            fig = px.bar(comparison_time, x='Metric', y='Time (min)',
                        title='Average Settlement Time Comparison',
                        color='Metric',
                        color_discrete_map={'Before': '#ff7f0e', 'After': '#2ca02c'})
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

# Footer
st.markdown("---")
st.markdown(f"*Dashboard v3.0 - Modular Architecture | Transactions: {len(filtered_df):,}*")
