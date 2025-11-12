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
                df['timestamp'] = pd.to_datetime(df['timestamp'])
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
        df = df_after  # Use optimized as primary for filtering
        data_source = "Comparison Mode"
        st.sidebar.success("✅ Comparison mode enabled")
    else:
        st.error("⚠️ Both original and optimized data files are required for comparison mode!")
        st.stop()

# Sidebar filters
st.sidebar.markdown("---")
st.sidebar.title("🎛️ Filters")

# Date range filter
if df is not None and 'timestamp' in df.columns:
    min_date = df['timestamp'].min().date()
    max_date = df['timestamp'].max().date()
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
else:
    date_range = None

# Business type filter
if df is not None and 'business_type' in df.columns:
    business_types = ['All'] + sorted(df['business_type'].unique().tolist())
    selected_business_type = st.sidebar.multiselect(
        "Business Type",
        business_types,
        default=['All']
    )
else:
    selected_business_type = ['All']

# Region filter
if df is not None and 'region' in df.columns:
    regions = ['All'] + sorted(df['region'].unique().tolist())
    selected_region = st.sidebar.multiselect(
        "Region",
        regions,
        default=['All']
    )
else:
    selected_region = ['All']

# Urgency level filter
if df is not None and 'urgency_level' in df.columns:
    urgency_levels = ['All'] + sorted(df['urgency_level'].unique().tolist())
    selected_urgency = st.sidebar.multiselect(
        "Urgency Level",
        urgency_levels,
        default=['All']
    )
else:
    selected_urgency = ['All']

# User tier filter
if df is not None and 'user_tier' in df.columns:
    user_tiers = ['All'] + sorted(df['user_tier'].unique().tolist())
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
    if date_range is not None and len(date_range) == 2 and 'timestamp' in filtered.columns:
        mask = (filtered['timestamp'].dt.date >= date_range[0]) & (filtered['timestamp'].dt.date <= date_range[1])
        filtered = filtered[mask]
    
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
    filtered_df = filtered_df_after
else:
    filtered_df = apply_filters(df)

# Check if we have data after filtering
if filtered_df is None or len(filtered_df) == 0:
    st.warning("⚠️ No data available with current filters. Please adjust your filters.")
    st.stop()

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

# Helper function to calculate improvement
def calculate_improvement(before_val, after_val, inverse=False):
    """Calculate percentage improvement"""
    if before_val == 0:
        return 0
    improvement = ((before_val - after_val) / before_val) * 100
    if inverse:
        improvement = -improvement
    return improvement

# TAB: Optimization Impact (only in comparison mode)
if analysis_mode == "Optimization Comparison":
    with tab1:
        st.header("🎯 Optimization Impact Analysis")
        st.markdown("### 📈 Bottom Line: Key Improvements")
        
        # Calculate key metrics for both datasets
        metrics_comparison = {}
        
        # Cost metrics
        if 'total_cost_bps' in filtered_df_before.columns and 'total_cost_bps' in filtered_df_after.columns:
            before_cost = filtered_df_before['total_cost_bps'].mean()
            after_cost = filtered_df_after['total_cost_bps'].mean()
            cost_improvement = calculate_improvement(before_cost, after_cost)
            metrics_comparison['cost'] = (before_cost, after_cost, cost_improvement)
        
        # Time metrics
        if 'settlement_time_sec' in filtered_df_before.columns and 'settlement_time_sec' in filtered_df_after.columns:
            before_time = filtered_df_before['settlement_time_sec'].mean()
            after_time = filtered_df_after['settlement_time_sec'].mean()
            time_improvement = calculate_improvement(before_time, after_time)
            metrics_comparison['time'] = (before_time, after_time, time_improvement)
        
        # Success rate
        if 'settlement_status' in filtered_df_before.columns and 'settlement_status' in filtered_df_after.columns:
            before_success = (filtered_df_before['settlement_status'] == 'completed').sum() / len(filtered_df_before) * 100
            after_success = (filtered_df_after['settlement_status'] == 'completed').sum() / len(filtered_df_after) * 100
            success_improvement = after_success - before_success
            metrics_comparison['success'] = (before_success, after_success, success_improvement)
        
        # Display improvement metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'cost' in metrics_comparison:
                before, after, improvement = metrics_comparison['cost']
                st.metric(
                    "Avg Cost (BPS)",
                    f"{after:.2f}",
                    f"{improvement:.1f}%",
                    delta_color="inverse"
                )
                st.caption(f"Before: {before:.2f} BPS")
        
        with col2:
            if 'time' in metrics_comparison:
                before, after, improvement = metrics_comparison['time']
                st.metric(
                    "Avg Settlement Time",
                    f"{after/60:.1f} min",
                    f"{improvement:.1f}%",
                    delta_color="inverse"
                )
                st.caption(f"Before: {before/60:.1f} min")
        
        with col3:
            if 'success' in metrics_comparison:
                before, after, improvement = metrics_comparison['success']
                st.metric(
                    "Success Rate",
                    f"{after:.1f}%",
                    f"+{improvement:.1f}%",
                    delta_color="normal"
                )
                st.caption(f"Before: {before:.1f}%")
        
        with col4:
            # Calculate cost savings
            if 'total_fees_usd' in filtered_df_before.columns and 'total_fees_usd' in filtered_df_after.columns:
                total_savings = filtered_df_before['total_fees_usd'].sum() - filtered_df_after['total_fees_usd'].sum()
                st.metric(
                    "Total Cost Savings",
                    f"${total_savings:,.2f}",
                    f"Saved",
                    delta_color="normal"
                )
        
        st.markdown("---")
        
        # ROI Calculation
        st.markdown("### 💰 Return on Investment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'total_fees_usd' in filtered_df_before.columns and 'total_fees_usd' in filtered_df_after.columns:
                total_before = filtered_df_before['total_fees_usd'].sum()
                total_after = filtered_df_after['total_fees_usd'].sum()
                total_savings = total_before - total_after
                
                # Calculate annualized savings (assuming data represents a period)
                if 'timestamp' in filtered_df_after.columns:
                    days_in_data = (filtered_df_after['timestamp'].max() - filtered_df_after['timestamp'].min()).days
                    if days_in_data > 0:
                        annualized_savings = (total_savings / days_in_data) * 365
                    else:
                        annualized_savings = total_savings
                else:
                    annualized_savings = total_savings
                
                st.markdown(f"""
                <div class="alert-green">
                    <h4>💵 Cost Reduction Impact</h4>
                    <p><strong>Total Fees Before Optimization:</strong> ${total_before:,.2f}</p>
                    <p><strong>Total Fees After Optimization:</strong> ${total_after:,.2f}</p>
                    <p><strong>Total Savings:</strong> ${total_savings:,.2f}</p>
                    <p><strong>Savings Rate:</strong> {(total_savings/total_before*100):.1f}%</p>
                    <p><strong>Projected Annual Savings:</strong> ${annualized_savings:,.2f}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Transaction efficiency improvement
            txn_count_before = len(filtered_df_before)
            txn_count_after = len(filtered_df_after)
            
            st.markdown(f"""
            <div class="alert-blue">
                <h4>📊 Transaction Efficiency</h4>
                <p><strong>Transactions Analyzed:</strong> {txn_count_after:,}</p>
                <p><strong>Avg Cost Reduction:</strong> {cost_improvement:.1f}%</p>
                <p><strong>Avg Time Reduction:</strong> {time_improvement:.1f}%</p>
                <p><strong>Success Rate Improvement:</strong> +{success_improvement:.1f}pp</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Side-by-side comparison charts
        st.markdown("### 📊 Before vs After Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Cost distribution comparison
            if 'total_cost_bps' in filtered_df_before.columns and 'total_cost_bps' in filtered_df_after.columns:
                fig = go.Figure()
                fig.add_trace(go.Box(y=filtered_df_before['total_cost_bps'], name='Before', 
                                    marker_color='#ff7f0e'))
                fig.add_trace(go.Box(y=filtered_df_after['total_cost_bps'], name='After', 
                                    marker_color='#2ca02c'))
                fig.update_layout(title='Cost Distribution Comparison (BPS)', 
                                 yaxis_title='Total Cost (BPS)', height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Settlement time comparison
            if 'settlement_time_sec' in filtered_df_before.columns and 'settlement_time_sec' in filtered_df_after.columns:
                fig = go.Figure()
                fig.add_trace(go.Box(y=filtered_df_before['settlement_time_sec']/60, name='Before', 
                                    marker_color='#ff7f0e'))
                fig.add_trace(go.Box(y=filtered_df_after['settlement_time_sec']/60, name='After', 
                                    marker_color='#2ca02c'))
                fig.update_layout(title='Settlement Time Comparison (minutes)', 
                                 yaxis_title='Settlement Time (min)', height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Cost component breakdown comparison
        st.markdown("### 💵 Cost Component Analysis")
        
        cost_components_before = {
            'Gas': filtered_df_before['gas_cost_usd'].sum() if 'gas_cost_usd' in filtered_df_before.columns else 0,
            'LP Fees': filtered_df_before['lp_fee_usd'].sum() if 'lp_fee_usd' in filtered_df_before.columns else 0,
            'Bridge': filtered_df_before['bridge_cost_usd'].sum() if 'bridge_cost_usd' in filtered_df_before.columns else 0,
            'Slippage': filtered_df_before['slippage_cost_usd'].sum() if 'slippage_cost_usd' in filtered_df_before.columns else 0
        }
        
        cost_components_after = {
            'Gas': filtered_df_after['gas_cost_usd'].sum() if 'gas_cost_usd' in filtered_df_after.columns else 0,
            'LP Fees': filtered_df_after['lp_fee_usd'].sum() if 'lp_fee_usd' in filtered_df_after.columns else 0,
            'Bridge': filtered_df_after['bridge_cost_usd'].sum() if 'bridge_cost_usd' in filtered_df_after.columns else 0,
            'Slippage': filtered_df_after['slippage_cost_usd'].sum() if 'slippage_cost_usd' in filtered_df_after.columns else 0
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure(data=[
                go.Bar(name='Before', x=list(cost_components_before.keys()), 
                      y=list(cost_components_before.values()), marker_color='#ff7f0e'),
                go.Bar(name='After', x=list(cost_components_after.keys()), 
                      y=list(cost_components_after.values()), marker_color='#2ca02c')
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
                'Savings (%)': [(cost_components_before[k] - cost_components_after[k]) / cost_components_before[k] * 100 
                               if cost_components_before[k] > 0 else 0 
                               for k in cost_components_before.keys()]
            }
            savings_df = pd.DataFrame(savings_data)
            
            fig = px.bar(savings_df, x='Component', y='Savings ($)', 
                        title='Savings by Cost Component',
                        text='Savings (%)', color='Savings ($)',
                        color_continuous_scale='Greens')
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Routing improvement
        st.markdown("### 🗺️ Routing Optimization")
        
        if 'routing_hops' in filtered_df_before.columns and 'routing_hops' in filtered_df_after.columns:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                before_hops = filtered_df_before['routing_hops'].mean()
                after_hops = filtered_df_after['routing_hops'].mean()
                hop_improvement = calculate_improvement(before_hops, after_hops)
                st.metric("Avg Routing Hops", f"{after_hops:.2f}", f"{hop_improvement:.1f}%", 
                         delta_color="inverse")
                st.caption(f"Before: {before_hops:.2f}")
            
            with col2:
                # Single hop percentage
                before_single = (filtered_df_before['routing_hops'] == 1).sum() / len(filtered_df_before) * 100
                after_single = (filtered_df_after['routing_hops'] == 1).sum() / len(filtered_df_after) * 100
                st.metric("Single-Hop Routes", f"{after_single:.1f}%", f"+{after_single-before_single:.1f}pp")
                st.caption(f"Before: {before_single:.1f}%")
            
            with col3:
                # Multi-hop (3+) percentage
                before_multi = (filtered_df_before['routing_hops'] >= 3).sum() / len(filtered_df_before) * 100
                after_multi = (filtered_df_after['routing_hops'] >= 3).sum() / len(filtered_df_after) * 100
                st.metric("Multi-Hop (3+) Routes", f"{after_multi:.1f}%", f"{after_multi-before_multi:.1f}pp",
                         delta_color="inverse")
                st.caption(f"Before: {before_multi:.1f}%")

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
                        if 'timestamp' in filtered_df.columns:
                            # Get yesterday's data for comparison
                            max_date = filtered_df['timestamp'].max()
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
                if 'exceptions_data' in st.session_state:
                    # Show exception summary
                    st.markdown("#### Exception Summary")
                    
                    if st.session_state['exceptions_data']:
                        exception_df = pd.DataFrame(st.session_state['exceptions_data'])
                        
                        # Display metrics
                        cols = st.columns(len(st.session_state['exceptions_data']))
                        for idx, exc in enumerate(st.session_state['exceptions_data']):
                            with cols[idx]:
                                severity_color = {
                                    'High': '🔴',
                                    'Medium': '🟡',
                                    'Low': '🟢'
                                }
                                st.metric(
                                    f"{severity_color.get(exc['severity'], '⚪')} {exc['type']}", 
                                    exc['count'],
                                    delta=exc['severity']
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
                if 'anomalies_data' in st.session_state and st.session_state['anomalies_data']:
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
                        exec_data = {
                            'period': f"{filtered_df['timestamp'].min().date()} to {filtered_df['timestamp'].max().date()}" if 'timestamp' in filtered_df.columns else "Current period",
                            'total_volume_usd': float(filtered_df['amount_source'].sum()) if 'amount_source' in filtered_df.columns else 0,
                            'total_transactions': len(filtered_df),
                            'avg_cost_bps': float(filtered_df['total_cost_bps'].mean()) if 'total_cost_bps' in filtered_df.columns else 0,
                            'success_rate': float((filtered_df['settlement_status'] == 'completed').sum() / len(filtered_df) * 100) if 'settlement_status' in filtered_df.columns else 0,
                            'total_fees_usd': float(filtered_df['total_fees_usd'].sum()) if 'total_fees_usd' in filtered_df.columns else 0,
                            'avg_settlement_time_min': float(filtered_df['settlement_time_sec'].mean() / 60) if 'settlement_time_sec' in filtered_df.columns else 0,
                            'compliance_rate': float((filtered_df['compliance_passed'] == True).sum() / len(filtered_df) * 100) if 'compliance_passed' in filtered_df.columns else 100
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
            avg_cost = filtered_df['total_cost_bps'].mean()
            if analysis_mode == "Optimization Comparison" and 'total_cost_bps' in filtered_df_before.columns:
                before_cost = filtered_df_before['total_cost_bps'].mean()
                delta = f"{calculate_improvement(before_cost, avg_cost):.1f}%"
            else:
                delta = None
            st.metric("Avg Cost (BPS)", f"{avg_cost:.2f}", delta, delta_color="inverse")
        else:
            st.metric("Avg Cost (BPS)", "N/A")
    
    with col2:
        if 'settlement_status' in filtered_df.columns:
            success_rate = (filtered_df['settlement_status'] == 'completed').sum() / len(filtered_df) * 100
            if analysis_mode == "Optimization Comparison" and 'settlement_status' in filtered_df_before.columns:
                before_success = (filtered_df_before['settlement_status'] == 'completed').sum() / len(filtered_df_before) * 100
                delta = f"+{success_rate - before_success:.1f}%"
            else:
                delta = None
            st.metric("Success Rate", f"{success_rate:.1f}%", delta)
        else:
            st.metric("Success Rate", "N/A")
    
    with col3:
        if 'settlement_time_sec' in filtered_df.columns:
            avg_time = filtered_df['settlement_time_sec'].mean() / 60
            if analysis_mode == "Optimization Comparison" and 'settlement_time_sec' in filtered_df_before.columns:
                before_time = filtered_df_before['settlement_time_sec'].mean() / 60
                delta = f"{calculate_improvement(before_time, avg_time):.1f}%"
            else:
                delta = None
            st.metric("Avg Settlement Time", f"{avg_time:.1f} min", delta, delta_color="inverse")
        else:
            st.metric("Avg Settlement Time", "N/A")
    
    with col4:
        if 'amount_source' in filtered_df.columns:
            total_volume = filtered_df['amount_source'].sum() / 1_000_000
            st.metric("Total Volume", f"${total_volume:.2f}M")
        else:
            st.metric("Total Volume", "N/A")
    
    # Additional metrics row
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        txn_count = len(filtered_df)
        st.metric("Transaction Count", f"{txn_count:,}")
    
    with col6:
        if 'routing_hops' in filtered_df.columns:
            avg_hops = filtered_df['routing_hops'].mean()
            st.metric("Avg Routing Hops", f"{avg_hops:.2f}")
        else:
            st.metric("Avg Routing Hops", "N/A")
    
    with col7:
        if 'compliance_passed' in filtered_df.columns:
            compliance_rate = (filtered_df['compliance_passed'] == True).sum() / len(filtered_df) * 100
            st.metric("Compliance Rate", f"{compliance_rate:.1f}%")
        else:
            st.metric("Compliance Rate", "N/A")
    
    with col8:
        if 'total_fees_usd' in filtered_df.columns:
            total_fees = filtered_df['total_fees_usd'].sum() / 1000
            st.metric("Total Fees", f"${total_fees:.1f}K")
        else:
            st.metric("Total Fees", "N/A")
    
    st.markdown("---")
    
    # Transaction volume over time
    if 'timestamp' in filtered_df.columns:
        st.markdown("### 📊 Transaction Volume Trends")
        
        daily_data = filtered_df.groupby(filtered_df['timestamp'].dt.date).agg({
            'transfer_id': 'count',
            'amount_source': 'sum' if 'amount_source' in filtered_df.columns else 'count',
            'total_cost_bps': 'mean' if 'total_cost_bps' in filtered_df.columns else 'count',
            'settlement_time_sec': 'mean' if 'settlement_time_sec' in filtered_df.columns else 'count'
        }).reset_index()
        
        daily_data.columns = ['date', 'count', 'volume', 'avg_cost_bps', 'avg_time_sec']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Daily Transaction Count', 'Daily Volume ($)', 
                           'Avg Cost (BPS)', 'Avg Settlement Time (min)'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        fig.add_trace(
            go.Scatter(x=daily_data['date'], y=daily_data['count'], 
                      name='Txn Count', fill='tozeroy', line=dict(color='#1f77b4')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=daily_data['date'], y=daily_data['volume'], 
                      name='Volume', fill='tozeroy', line=dict(color='#2ca02c')),
            row=1, col=2
        )
        
        if 'total_cost_bps' in filtered_df.columns:
            fig.add_trace(
                go.Scatter(x=daily_data['date'], y=daily_data['avg_cost_bps'], 
                          name='Avg Cost', line=dict(color='#ff7f0e')),
                row=2, col=1
            )
        
        if 'settlement_time_sec' in filtered_df.columns:
            fig.add_trace(
                go.Scatter(x=daily_data['date'], y=daily_data['avg_time_sec']/60, 
                          name='Avg Time', line=dict(color='#d62728')),
                row=2, col=2
            )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

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
            total_gas = filtered_df['gas_cost_usd'].sum()
            st.metric("Total Gas Costs", f"${total_gas:,.2f}")
        else:
            st.metric("Total Gas Costs", "N/A")
    
    with col2:
        if 'lp_fee_usd' in filtered_df.columns:
            total_lp = filtered_df['lp_fee_usd'].sum()
            st.metric("Total LP Fees", f"${total_lp:,.2f}")
        else:
            st.metric("Total LP Fees", "N/A")
    
    with col3:
        if 'bridge_cost_usd' in filtered_df.columns:
            total_bridge = filtered_df['bridge_cost_usd'].sum()
            st.metric("Total Bridge Costs", f"${total_bridge:,.2f}")
        else:
            st.metric("Total Bridge Costs", "N/A")
    
    with col4:
        if 'slippage_cost_usd' in filtered_df.columns:
            total_slippage = filtered_df['slippage_cost_usd'].sum()
            st.metric("Total Slippage", f"${total_slippage:,.2f}")
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
                    filtered_df['gas_cost_usd'].sum(),
                    filtered_df['lp_fee_usd'].sum(),
                    filtered_df['bridge_cost_usd'].sum(),
                    filtered_df['slippage_cost_usd'].sum()
                ]
            })
            
            fig = px.pie(cost_components, values='Amount', names='Component',
                        title='Cost Distribution by Component',
                        color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Cost component data not available")
    
    with col2:
        # Box plot of costs by business type
        if 'total_cost_bps' in filtered_df.columns and 'business_type' in filtered_df.columns:
            fig = px.box(filtered_df, x='business_type', y='total_cost_bps',
                        title='Cost Distribution by Business Type',
                        color='business_type')
            fig.update_xaxis(tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Cost by business type data not available")
    
    # Cost efficiency table
    if 'business_type' in filtered_df.columns and 'total_cost_bps' in filtered_df.columns:
        st.markdown("### 📋 Cost Efficiency by Transaction Type")
        
        agg_dict = {'transfer_id': 'count'}
        if 'total_cost_bps' in filtered_df.columns:
            agg_dict['total_cost_bps'] = ['mean', 'median', 'std']
        if 'total_fees_usd' in filtered_df.columns:
            agg_dict['total_fees_usd'] = 'sum'
        if 'amount_source' in filtered_df.columns:
            agg_dict['amount_source'] = 'sum'
        
        cost_summary = filtered_df.groupby('business_type').agg(agg_dict).round(2)
        
        # Flatten column names
        cost_summary.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col 
                               for col in cost_summary.columns.values]
        cost_summary = cost_summary.reset_index()
        
        st.dataframe(cost_summary, use_container_width=True)

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
            completion_rate = completed / len(filtered_df) * 100
            st.metric("Completion Rate", f"{completion_rate:.1f}%")
        else:
            st.metric("Completion Rate", "N/A")
    
    with col2:
        if 'settlement_status' in filtered_df.columns:
            failed = (filtered_df['settlement_status'] == 'failed').sum()
            failure_rate = failed / len(filtered_df) * 100
            st.metric("Failure Rate", f"{failure_rate:.1f}%")
        else:
            st.metric("Failure Rate", "N/A")
    
    with col3:
        if 'settlement_status' in filtered_df.columns:
            pending = (filtered_df['settlement_status'] == 'pending').sum()
            pending_rate = pending / len(filtered_df) * 100
            st.metric("Pending Rate", f"{pending_rate:.1f}%")
        else:
            st.metric("Pending Rate", "N/A")
    
    with col4:
        if 'settlement_time_sec' in filtered_df.columns:
            avg_settlement = filtered_df['settlement_time_sec'].mean()
            st.metric("Avg Settlement Time", f"{avg_settlement/60:.1f} min")
        else:
            st.metric("Avg Settlement Time", "N/A")
    
    st.markdown("---")
    
    # Settlement time analysis
    st.markdown("### ⏱️ Settlement Time Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Settlement time by urgency
        if 'urgency_level' in filtered_df.columns and 'settlement_time_sec' in filtered_df.columns:
            fig = px.box(filtered_df, x='urgency_level', y='settlement_time_sec',
                        title='Settlement Time by Urgency Level',
                        color='urgency_level',
                        category_orders={'urgency_level': ['urgent', 'standard', 'low']})
            fig.update_yaxis(title='Settlement Time (seconds)')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Settlement time by urgency data not available")
    
    with col2:
        # Settlement time by routing hops
        if 'routing_hops' in filtered_df.columns and 'settlement_time_sec' in filtered_df.columns:
            hop_time = filtered_df.groupby('routing_hops').agg({
                'settlement_time_sec': ['mean', 'count']
            }).reset_index()
            hop_time.columns = ['Hops', 'Avg Time (sec)', 'Count']
            
            fig = px.bar(hop_time, x='Hops', y='Avg Time (sec)',
                        title='Avg Settlement Time by Routing Hops',
                        text='Count', color='Avg Time (sec)',
                        color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Settlement time by routing hops data not available")
    
    # Success rate analysis
    st.markdown("### 📊 Success Rate Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Success by user tier
        if 'user_tier' in filtered_df.columns and 'settlement_status' in filtered_df.columns:
            success_by_tier = filtered_df.groupby('user_tier')['settlement_status'].apply(
                lambda x: (x == 'completed').sum() / len(x) * 100
            ).reset_index()
            success_by_tier.columns = ['user_tier', 'success_rate']
            
            fig = px.bar(success_by_tier, x='user_tier', y='success_rate',
                        title='Success Rate by User Tier',
                        text='success_rate', color='success_rate',
                        color_continuous_scale='RdYlGn', range_color=[0, 100])
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Success rate by user tier data not available")
    
    with col2:
        # Success by liquidity availability
        if 'liquidity_available' in filtered_df.columns and 'settlement_status' in filtered_df.columns:
            success_by_liquidity = filtered_df.groupby('liquidity_available')['settlement_status'].apply(
                lambda x: (x == 'completed').sum() / len(x) * 100
            ).reset_index()
            success_by_liquidity.columns = ['liquidity_available', 'success_rate']
            success_by_liquidity['liquidity_available'] = success_by_liquidity['liquidity_available'].map(
                {True: 'Available', False: 'Not Available'}
            )
            
            fig = px.bar(success_by_liquidity, x='liquidity_available', y='success_rate',
                        title='Success Rate by Liquidity Availability',
                        text='success_rate', color='success_rate',
                        color_continuous_scale='RdYlGn', range_color=[0, 100])
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
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
            avg_hops = filtered_df['routing_hops'].mean()
            st.metric("Avg Routing Hops", f"{avg_hops:.2f}")
        else:
            st.metric("Avg Routing Hops", "N/A")
    
    with col3:
        if 'routing_hops' in filtered_df.columns:
            single_hop = (filtered_df['routing_hops'] == 1).sum()
            single_hop_pct = single_hop / len(filtered_df) * 100
            st.metric("Single-Hop Routes", f"{single_hop_pct:.1f}%")
        else:
            st.metric("Single-Hop Routes", "N/A")
    
    with col4:
        if 'routing_hops' in filtered_df.columns:
            multi_hop = (filtered_df['routing_hops'] >= 3).sum()
            multi_hop_pct = multi_hop / len(filtered_df) * 100
            st.metric("Multi-Hop (3+) Routes", f"{multi_hop_pct:.1f}%")
        else:
            st.metric("Multi-Hop (3+) Routes", "N/A")
    
    st.markdown("---")
    
    # Chain pair analysis
    if all(col in filtered_df.columns for col in ['source_chain', 'dest_chain', 'total_cost_bps']):
        st.markdown("### 🗺️ Route Performance by Chain Pair")
        
        route_data = filtered_df.groupby(['source_chain', 'dest_chain']).agg({
            'total_cost_bps': 'mean',
            'transfer_id': 'count',
            'settlement_time_sec': 'mean' if 'settlement_time_sec' in filtered_df.columns else 'count'
        }).reset_index()
        
        route_data.columns = ['Source Chain', 'Dest Chain', 'Avg Cost (BPS)', 'Count', 'Avg Time (sec)']
        
        # Display top routes
        st.markdown("#### Top 10 Routes by Volume")
        top_routes = route_data.nlargest(10, 'Count')[
            ['Source Chain', 'Dest Chain', 'Count', 'Avg Cost (BPS)', 'Avg Time (sec)']
        ]
        st.dataframe(top_routes, use_container_width=True)
        
        # Scatter plot of routes
        fig = px.scatter(route_data, x='Avg Cost (BPS)', y='Avg Time (sec)',
                        size='Count', color='Count',
                        hover_data=['Source Chain', 'Dest Chain'],
                        title='Route Performance: Cost vs Speed',
                        color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
    
    # Venue performance
    if 'venues_used' in filtered_df.columns:
        st.markdown("### 🏦 Venue Performance Analysis")
        
        # Extract and count venue usage
        all_venues = []
        for venues in filtered_df['venues_used']:
            if pd.notna(venues):
                all_venues.extend(str(venues).split(','))
        
        if all_venues:
            venue_counts = pd.Series(all_venues).value_counts().reset_index()
            venue_counts.columns = ['Venue', 'Count']
            
            # Calculate average cost per venue
            venue_stats = []
            for venue in venue_counts['Venue'][:10]:  # Top 10 venues
                venue_txns = filtered_df[filtered_df['venues_used'].str.contains(venue, na=False)]
                if len(venue_txns) > 0:
                    venue_stat = {
                        'Venue': venue,
                        'Count': len(venue_txns)
                    }
                    
                    if 'total_cost_bps' in venue_txns.columns:
                        venue_stat['Avg Cost (BPS)'] = venue_txns['total_cost_bps'].mean()
                    if 'settlement_time_sec' in venue_txns.columns:
                        venue_stat['Avg Time (min)'] = venue_txns['settlement_time_sec'].mean() / 60
                    if 'settlement_status' in venue_txns.columns:
                        venue_stat['Success Rate (%)'] = (venue_txns['settlement_status'] == 'completed').sum() / len(venue_txns) * 100
                    if 'amount_source' in venue_txns.columns:
                        venue_stat['Total Volume ($)'] = venue_txns['amount_source'].sum()
                    
                    venue_stats.append(venue_stat)
            
            if venue_stats:
                venue_df = pd.DataFrame(venue_stats).round(2)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.bar(venue_df.sort_values('Count', ascending=False), 
                                x='Venue', y='Count',
                                title='Transaction Volume by Venue',
                                color='Count', color_continuous_scale='Blues')
                    fig.update_xaxis(tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if 'Avg Cost (BPS)' in venue_df.columns and 'Avg Time (min)' in venue_df.columns:
                        fig = px.scatter(venue_df, x='Avg Cost (BPS)', y='Avg Time (min)',
                                        size='Count', 
                                        color='Success Rate (%)' if 'Success Rate (%)' in venue_df.columns else 'Count',
                                        hover_data=['Venue'],
                                        title='Venue Performance: Cost vs Speed',
                                        color_continuous_scale='RdYlGn',
                                        range_color=[90, 100] if 'Success Rate (%)' in venue_df.columns else None)
                        st.plotly_chart(fig, use_container_width=True)
                
                # Venue performance table
                st.markdown("### 📊 Detailed Venue Statistics")
                st.dataframe(venue_df.sort_values('Count', ascending=False), use_container_width=True)

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
            compliance_rate = (filtered_df['compliance_passed'] == True).sum() / len(filtered_df) * 100
            st.metric("Compliance Rate", f"{compliance_rate:.1f}%")
        else:
            st.metric("Compliance Rate", "N/A")
    
    with col4:
        if 'kyc_status' in filtered_df.columns:
            kyc_verified = (filtered_df['kyc_status'] == 'verified').sum() / len(filtered_df) * 100
            st.metric("KYC Verified", f"{kyc_verified:.1f}%")
        else:
            st.metric("KYC Verified", "N/A")
    
    st.markdown("---")
    
    # Regional performance
    if 'region' in filtered_df.columns:
        st.markdown("### 🌍 Performance by Region")
        
        agg_dict = {'transfer_id': 'count'}
        if 'amount_source' in filtered_df.columns:
            agg_dict['amount_source'] = 'sum'
        if 'total_cost_bps' in filtered_df.columns:
            agg_dict['total_cost_bps'] = 'mean'
        if 'settlement_time_sec' in filtered_df.columns:
            agg_dict['settlement_time_sec'] = 'mean'
        if 'settlement_status' in filtered_df.columns:
            agg_dict['settlement_status'] = lambda x: (x == 'completed').sum() / len(x) * 100
        if 'compliance_passed' in filtered_df.columns:
            agg_dict['compliance_passed'] = lambda x: (x == True).sum() / len(x) * 100
        
        regional_stats = filtered_df.groupby('region').agg(agg_dict).reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'amount_source' in filtered_df.columns:
                fig = px.bar(regional_stats, x='region', y='amount_source',
                            title='Transaction Volume by Region',
                            color='amount_source', color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(regional_stats, x='region', y='transfer_id',
                        title='Transaction Count by Region',
                        color='transfer_id', color_continuous_scale='Greens')
            st.plotly_chart(fig, use_container_width=True)
        
        # Regional performance table
        st.dataframe(regional_stats, use_container_width=True)
    
    # Beneficiary country analysis
    if 'beneficiary_country' in filtered_df.columns:
        st.markdown("### 🗺️ Top Beneficiary Countries")
        
        country_data = filtered_df[filtered_df['beneficiary_country'].notna()].groupby('beneficiary_country').agg({
            'transfer_id': 'count',
            'amount_source': 'sum' if 'amount_source' in filtered_df.columns else 'count',
            'total_cost_bps': 'mean' if 'total_cost_bps' in filtered_df.columns else 'count',
            'settlement_status': lambda x: (x == 'completed').sum() / len(x) * 100 if 'settlement_status' in filtered_df.columns else 0
        }).reset_index()
        
        country_data.columns = ['Country', 'Count', 'Volume ($)', 'Avg Cost (BPS)', 'Success Rate (%)']
        country_data = country_data.sort_values('Count', ascending=False).head(15)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(country_data, x='Country', y='Count',
                        title='Top 15 Beneficiary Countries by Transaction Count',
                        color='Success Rate (%)', color_continuous_scale='RdYlGn',
                        range_color=[90, 100])
            fig.update_xaxis(tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.treemap(country_data, path=['Country'], values='Volume ($)',
                            title='Transaction Volume Distribution by Country',
                            color='Avg Cost (BPS)', color_continuous_scale='RdYlGn_r')
            st.plotly_chart(fig, use_container_width=True)
    
    # Compliance analysis
    if 'compliance_passed' in filtered_df.columns:
        st.markdown("### 🔒 Compliance Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Compliance by region
            if 'region' in filtered_df.columns:
                compliance_by_region = filtered_df.groupby('region')['compliance_passed'].apply(
                    lambda x: (x == True).sum() / len(x) * 100
                ).reset_index()
                compliance_by_region.columns = ['Region', 'Compliance Rate (%)']
                
                fig = px.bar(compliance_by_region, x='Region', y='Compliance Rate (%)',
                            title='Compliance Rate by Region',
                            color='Compliance Rate (%)', color_continuous_scale='RdYlGn',
                            range_color=[0, 100], text='Compliance Rate (%)')
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # KYC status distribution
            if 'kyc_status' in filtered_df.columns:
                kyc_dist = filtered_df['kyc_status'].value_counts()
                fig = px.pie(values=kyc_dist.values, names=kyc_dist.index,
                            title='KYC Status Distribution',
                            color_discrete_sequence=px.colors.qualitative.Set2)
                st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666;'>
    <p>Stablecoin Route Optimization Dashboard v2.0 | Data refreshed: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    <p>💡 Use filters in the sidebar to drill down into specific segments</p>
    <p>📁 Data files loaded from: {data_dir}</p>
</div>
""", unsafe_allow_html=True)