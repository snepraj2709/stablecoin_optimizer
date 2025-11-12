"""
AI-Powered Insights Module for Stablecoin Route Optimization Dashboard

This module adds OpenAI integration for:
1. Daily treasury summaries
2. Exception remediation suggestions
3. Trend analysis and recommendations
"""

import openai
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple

class AIInsightsEngine:
    """Handle all AI-powered analysis using OpenAI API"""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        """
        Initialize the AI insights engine
        
        Args:
            api_key: OpenAI API key
            model: Model to use (gpt-4, gpt-3.5-turbo, etc.)
        """
        openai.api_key = api_key
        self.model = model
    
    def generate_daily_summary(self, data_dict: Dict) -> str:
        prompt = f"""You are a treasury operations analyst. Generate a concise daily summary report based on the following transaction data.

        Data Summary:
        - Total Transactions: {data_dict.get('total_transactions', 0):,}
        - Total Volume: ${data_dict.get('total_volume', 0):,.2f}
        - Average Cost (BPS): {data_dict.get('avg_cost_bps', 0):.2f}
        - Success Rate: {data_dict.get('success_rate', 0):.1f}%
        - Failed Transactions: {data_dict.get('failed_count', 0)}
        - Average Settlement Time: {data_dict.get('avg_settlement_time', 0):.1f} minutes
        - Total Fees: ${data_dict.get('total_fees', 0):,.2f}
        - Compliance Rate: {data_dict.get('compliance_rate', 0):.1f}%

        Comparison to Previous Period:
        - Volume Change: {data_dict.get('volume_change', 0):+.1f}%
        - Cost Change: {data_dict.get('cost_change', 0):+.1f}%
        - Success Rate Change: {data_dict.get('success_change', 0):+.1f}pp

        Top Business Types:
        {data_dict.get('top_business_types', 'N/A')}

        Top Routes:
        {data_dict.get('top_routes', 'N/A')}

        Generate a professional daily summary that:
        1. Highlights key performance metrics
        2. Identifies notable trends or changes
        3. Flags any concerns
        4. Provides a brief outlook

        Keep it concise (200-300 words) and actionable."""

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert treasury operations analyst specializing in stablecoin transactions and route optimization."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def analyze_exceptions(self, exceptions_data: List[Dict]) -> str:
        """
        Analyze exceptions and provide remediation suggestions
        
        Args:
            exceptions_data: List of exception details
            
        Returns:
            AI-generated remediation suggestions
        """
        exceptions_summary = json.dumps(exceptions_data, indent=2)
        
        prompt = f"""You are a treasury operations expert. Analyze the following transaction exceptions and provide specific remediation suggestions.

Exceptions Data:
{exceptions_summary}

For each exception category, provide:
1. Root cause analysis
2. Immediate remediation steps
3. Long-term prevention strategies
4. Priority level (High/Medium/Low)

Format your response with clear sections and actionable recommendations."""

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in treasury operations, blockchain transactions, and operational risk management."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            return f"Error analyzing exceptions: {str(e)}"
    
    def generate_optimization_recommendations(self, route_data: Dict) -> str:
        """
        Generate route optimization recommendations
        
        Args:
            route_data: Dictionary containing route performance data
            
        Returns:
            AI-generated recommendations
        """
        prompt = f"""As a route optimization specialist, analyze the following routing data and provide actionable recommendations.

Route Performance Data:
{json.dumps(route_data, indent=2)}

Provide:
1. Top 3 optimization opportunities
2. Specific actions to reduce costs
3. Actions to improve settlement times
4. Venue selection recommendations
5. Risk considerations

Be specific and quantify potential improvements where possible."""

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in stablecoin routing, DeFi protocols, and transaction optimization."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6,
                max_tokens=800
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            return f"Error generating recommendations: {str(e)}"
    
    def analyze_cost_anomalies(self, anomaly_data: List[Dict]) -> str:
        """
        Analyze cost anomalies and provide insights
        
        Args:
            anomaly_data: List of transactions with unusual costs
            
        Returns:
            AI-generated analysis
        """
        prompt = f"""Analyze the following cost anomalies in stablecoin transactions and explain potential causes.

Anomalous Transactions:
{json.dumps(anomaly_data[:10], indent=2)}  # Limit to 10 for context

Provide:
1. Likely reasons for high costs
2. Whether these are expected or concerning
3. Recommended investigation steps
4. Preventive measures

Be specific and technical where appropriate."""

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in blockchain transaction costs, gas fees, and DeFi protocols."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=600
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            return f"Error analyzing anomalies: {str(e)}"
    
    def generate_executive_insights(self, summary_data: Dict) -> str:
        """
        Generate executive-level insights for leadership
        
        Args:
            summary_data: High-level summary data
            
        Returns:
            Executive summary with strategic insights
        """
        prompt = f"""As a CFO advisor, provide executive-level insights based on this treasury data.

Summary Metrics:
{json.dumps(summary_data, indent=2)}

Provide a brief executive summary covering:
1. Key takeaways (3-4 bullet points)
2. Strategic implications
3. Risk assessment
4. Recommended focus areas

Keep it strategic and suitable for C-level audience (150-200 words)."""

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a CFO advisor specializing in digital treasury and blockchain operations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=400
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            return f"Error generating insights: {str(e)}"
    
    def compare_periods(self, current_data: Dict, previous_data: Dict) -> str:
        """
        Compare two periods and provide insights on changes
        
        Args:
            current_data: Current period metrics
            previous_data: Previous period metrics
            
        Returns:
            Comparative analysis
        """
        prompt = f"""Compare these two periods and explain significant changes.

Current Period:
{json.dumps(current_data, indent=2)}

Previous Period:
{json.dumps(previous_data, indent=2)}

Analyze:
1. Most significant changes and their implications
2. Improving vs. declining metrics
3. Potential causes for major shifts
4. Recommended actions based on trends

Be analytical and provide context for changes."""

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a treasury analyst specializing in period-over-period analysis and trend identification."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6,
                max_tokens=600
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            return f"Error comparing periods: {str(e)}"

# Helper functions for data preparation

def prepare_daily_summary_data(df, previous_df=None):
    """
    Prepare data dictionary for daily summary generation
    
    Args:
        df: Current period dataframe
        previous_df: Previous period dataframe for comparison
        
    Returns:
        Dictionary with summary metrics
    """
    data = {
        'total_transactions': len(df),
        'total_volume': df['amount_source'].sum() if 'amount_source' in df.columns else 0,
        'avg_cost_bps': df['total_cost_bps'].mean() if 'total_cost_bps' in df.columns else 0,
        'success_rate': (df['settlement_status'] == 'completed').sum() / len(df) * 100 if 'settlement_status' in df.columns else 0,
        'failed_count': (df['settlement_status'] == 'failed').sum() if 'settlement_status' in df.columns else 0,
        'avg_settlement_time': df['settlement_time_sec'].mean() / 60 if 'settlement_time_sec' in df.columns else 0,
        'total_fees': df['total_fees_usd'].sum() if 'total_fees_usd' in df.columns else 0,
        'compliance_rate': (df['compliance_passed'] == True).sum() / len(df) * 100 if 'compliance_passed' in df.columns else 100,
    }
    
    # Add comparisons if previous data available
    if previous_df is not None and len(previous_df) > 0:
        prev_volume = previous_df['amount_source'].sum() if 'amount_source' in previous_df.columns else 0
        if prev_volume > 0:
            data['volume_change'] = ((data['total_volume'] - prev_volume) / prev_volume) * 100
        else:
            data['volume_change'] = 0
            
        prev_cost = previous_df['total_cost_bps'].mean() if 'total_cost_bps' in previous_df.columns else 0
        if prev_cost > 0:
            data['cost_change'] = ((data['avg_cost_bps'] - prev_cost) / prev_cost) * 100
        else:
            data['cost_change'] = 0
            
        prev_success = (previous_df['settlement_status'] == 'completed').sum() / len(previous_df) * 100 if 'settlement_status' in previous_df.columns else 0
        data['success_change'] = data['success_rate'] - prev_success
    else:
        data['volume_change'] = 0
        data['cost_change'] = 0
        data['success_change'] = 0
    
    # Top business types
    if 'business_type' in df.columns:
        top_types = df['business_type'].value_counts().head(3)
        data['top_business_types'] = '\n'.join([f"- {k}: {v} transactions" for k, v in top_types.items()])
    else:
        data['top_business_types'] = 'N/A'
    
    # Top routes
    if 'source_chain' in df.columns and 'dest_chain' in df.columns:
        df['route'] = df['source_chain'].astype(str) + ' → ' + df['dest_chain'].astype(str)
        top_routes = df['route'].value_counts().head(3)
        data['top_routes'] = '\n'.join([f"- {k}: {v} transactions" for k, v in top_routes.items()])
    else:
        data['top_routes'] = 'N/A'
    
    return data

def identify_exceptions(df):
    """
    Identify exceptional transactions that need attention
    
    Args:
        df: Transaction dataframe
        
    Returns:
        List of exception dictionaries
    """
    exceptions = []
    
    # Failed transactions
    if 'settlement_status' in df.columns:
        failed = df[df['settlement_status'] == 'failed']
        if len(failed) > 0:
            exceptions.append({
                'type': 'Failed Transactions',
                'count': len(failed),
                'severity': 'High',
                'details': f"{len(failed)} transactions failed",
                'sample_ids': failed['transfer_id'].head(5).tolist() if 'transfer_id' in failed.columns else []
            })
    
    # Compliance failures
    if 'compliance_passed' in df.columns:
        compliance_fail = df[df['compliance_passed'] == False]
        if len(compliance_fail) > 0:
            exceptions.append({
                'type': 'Compliance Failures',
                'count': len(compliance_fail),
                'severity': 'High',
                'details': f"{len(compliance_fail)} transactions failed compliance",
                'sample_ids': compliance_fail['transfer_id'].head(5).tolist() if 'transfer_id' in compliance_fail.columns else []
            })
    
    # High cost transactions (exceeding max acceptable fee)
    if 'total_cost_bps' in df.columns and 'max_acceptable_fee_bps' in df.columns:
        high_cost = df[df['total_cost_bps'] > df['max_acceptable_fee_bps']]
        if len(high_cost) > 0:
            avg_excess = (high_cost['total_cost_bps'] - high_cost['max_acceptable_fee_bps']).mean()
            exceptions.append({
                'type': 'Excessive Costs',
                'count': len(high_cost),
                'severity': 'Medium',
                'details': f"{len(high_cost)} transactions exceeded fee limits by avg {avg_excess:.1f} BPS",
                'sample_ids': high_cost['transfer_id'].head(5).tolist() if 'transfer_id' in high_cost.columns else []
            })
    
    # Slow transactions (significantly above average)
    if 'settlement_time_sec' in df.columns:
        avg_time = df['settlement_time_sec'].mean()
        std_time = df['settlement_time_sec'].std()
        slow_txns = df[df['settlement_time_sec'] > (avg_time + 2 * std_time)]
        if len(slow_txns) > 0:
            exceptions.append({
                'type': 'Slow Settlements',
                'count': len(slow_txns),
                'severity': 'Medium',
                'details': f"{len(slow_txns)} transactions took >2σ longer than average",
                'avg_time': slow_txns['settlement_time_sec'].mean() / 60,
                'sample_ids': slow_txns['transfer_id'].head(5).tolist() if 'transfer_id' in slow_txns.columns else []
            })
    
    # Pending transactions (stuck)
    if 'settlement_status' in df.columns and 'timestamp' in df.columns:
        pending = df[df['settlement_status'] == 'pending']
        if len(pending) > 0:
            # Check if any are old (>1 hour)
            now = df['timestamp'].max()
            old_pending = pending[pending['timestamp'] < (now - timedelta(hours=1))]
            if len(old_pending) > 0:
                exceptions.append({
                    'type': 'Stuck Pending Transactions',
                    'count': len(old_pending),
                    'severity': 'High',
                    'details': f"{len(old_pending)} transactions pending for >1 hour",
                    'sample_ids': old_pending['transfer_id'].head(5).tolist() if 'transfer_id' in old_pending.columns else []
                })
    
    return exceptions

def identify_cost_anomalies(df, threshold_percentile=95):
    """
    Identify transactions with anomalously high costs
    
    Args:
        df: Transaction dataframe
        threshold_percentile: Percentile threshold for anomaly detection
        
    Returns:
        List of anomalous transaction dictionaries
    """
    if 'total_cost_bps' not in df.columns:
        return []
    
    threshold = df['total_cost_bps'].quantile(threshold_percentile / 100)
    anomalies = df[df['total_cost_bps'] > threshold]
    
    anomaly_list = []
    for _, row in anomalies.head(20).iterrows():  # Limit to 20
        anomaly = {
            'transfer_id': row.get('transfer_id', 'N/A'),
            'cost_bps': row.get('total_cost_bps', 0),
            'amount': row.get('amount_source', 0),
            'business_type': row.get('business_type', 'N/A'),
            'route': f"{row.get('source_chain', 'N/A')} → {row.get('dest_chain', 'N/A')}",
            'routing_hops': row.get('routing_hops', 0),
            'gas_cost': row.get('gas_cost_usd', 0),
            'slippage': row.get('slippage_cost_usd', 0)
        }
        anomaly_list.append(anomaly)
    
    return anomaly_list

def prepare_route_optimization_data(df):
    """
    Prepare route performance data for optimization recommendations
    
    Args:
        df: Transaction dataframe
        
    Returns:
        Dictionary with route performance metrics
    """
    if not all(col in df.columns for col in ['source_chain', 'dest_chain']):
        return {}
    
    route_data = df.groupby(['source_chain', 'dest_chain']).agg({
        'transfer_id': 'count',
        'total_cost_bps': 'mean' if 'total_cost_bps' in df.columns else 'count',
        'settlement_time_sec': 'mean' if 'settlement_time_sec' in df.columns else 'count',
        'settlement_status': lambda x: (x == 'completed').sum() / len(x) * 100 if 'settlement_status' in df.columns else 100,
        'routing_hops': 'mean' if 'routing_hops' in df.columns else 'count'
    }).reset_index()
    
    route_data.columns = ['source', 'dest', 'volume', 'avg_cost_bps', 'avg_time_sec', 'success_rate', 'avg_hops']
    
    # Convert to dict format
    routes = []
    for _, row in route_data.head(10).iterrows():
        routes.append({
            'route': f"{row['source']} → {row['dest']}",
            'volume': int(row['volume']),
            'avg_cost_bps': float(row['avg_cost_bps']),
            'avg_time_min': float(row['avg_time_sec']) / 60,
            'success_rate': float(row['success_rate']),
            'avg_hops': float(row['avg_hops'])
        })
    
    return {
        'top_routes': routes,
        'total_routes': len(route_data),
        'avg_cost_all': float(df['total_cost_bps'].mean()) if 'total_cost_bps' in df.columns else 0,
        'avg_time_all': float(df['settlement_time_sec'].mean() / 60) if 'settlement_time_sec' in df.columns else 0
    }