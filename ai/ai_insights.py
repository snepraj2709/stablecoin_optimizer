"""
ai_insights.py

AI-powered insights for the Stablecoin Optimizer.

Features:
- Uses OpenAI if `OPENAI_API_KEY` is present and `openai` is importable.
- Otherwise, falls back to deterministic heuristics.
- Adds a tiny in-memory caching decorator (TTL) to cache dashboard calls for 5-10 minutes.
- Exposes:
    - AIInsightsEngine (class) with the same public methods you provided.
    - generate_insights(input_obj, model=...) top-level function used by the dashboard.
"""
from __future__ import annotations
import os
import json
import math
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from functools import wraps
from copy import deepcopy
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

# from api.metrics_registry import metrics_registry

OPENAI_AVAILABLE = True
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
_metrics_cache = None

# Small deterministic RNG for fallback recommendations (keeps outputs stable across runs)
def _deterministic_hash(s: str) -> int:
    h = 2166136261
    for ch in s:
        h = (h ^ ord(ch)) * 16777619 & 0xFFFFFFFF
    return h

# ----------------- Tiny in-memory TTL cache decorator -----------------
def cache_result(ttl_seconds: int = 600):
    """
    Simple thread-safe in-memory cache decorator with TTL.
    Attaches `.cache_store` dict and `.cache_hits` counter (dict) to the wrapped function
    so unit tests or callers can inspect usage if needed.
    """
    def decorator(fn):
        cache_store: Dict[str, Tuple[float, Any]] = {}
        cache_lock = threading.Lock()
        cache_hits = {"hits": 0}

        @wraps(fn)
        def wrapped(*args, **kwargs):
            # Build cache key from args/kwargs (JSON-serializable)
            try:
                key_obj = {"args": args, "kwargs": kwargs}
                # we must convert non-serializable to str
                key = json.dumps(key_obj, default=str, sort_keys=True)
            except Exception:
                # fallback simple key
                key = str((args, tuple(sorted(kwargs.items()))))

            now = time.time()
            with cache_lock:
                ent = cache_store.get(key)
                if ent:
                    ts, val = ent
                    if now - ts < ttl_seconds:
                        cache_hits["hits"] += 1
                        # return deep copy to avoid accidental mutation by callers
                        return deepcopy(val)
                    else:
                        # expired
                        del cache_store[key]
            # compute
            result = fn(*args, **kwargs)
            with cache_lock:
                cache_store[key] = (now, deepcopy(result))
            return result

        # attach introspection helpers
        wrapped.cache_store = cache_store  # type: ignore
        wrapped.cache_hits = cache_hits    # type: ignore
        wrapped.cache_lock = cache_lock    # type: ignore
        wrapped.cache_ttl = ttl_seconds    # type: ignore
        return wrapped
    return decorator

# ----------------- AIInsightsEngine (preserves your original methods) -----------------
class AIInsightsEngine:
    """
    Wrapper engine for AI-powered insights. Uses OpenAI when available,
    otherwise falls back to deterministic heuristics.
    """
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        """
        Initialize the engine. If api_key is None, will attempt to use OPENAI_API_KEY env var.
        If no key is present, engine will operate in heuristic fallback mode.
        """
        self.model = model
        self.api_key = api_key or OPENAI_API_KEY
        self.use_openai = OPENAI_AVAILABLE and bool(self.api_key)
        if self.use_openai:
            self.client = OpenAI(api_key=self.api_key) # type: ignore

    # ----------------- Helpers -----------------
    def _call_openai_chat(self, messages: List[Dict[str, str]], temperature: float = 0.0, max_tokens: int = 400) -> str:
        """
        Safe wrapper for OpenAI ChatCompletion.
        Returns string result or explanatory message on failure.
        """
        if not self.use_openai:
            raise RuntimeError("OpenAI not configured; falling back to local heuristics.")
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return resp.choices[0].message["content"].strip()
        except Exception as e:
            return f"[OpenAI error: {type(e).__name__}] {str(e)}"

    # ----------------- Public methods -----------------
    def generate_daily_summary(self, data_dict: Dict) -> str:
        if self.use_openai:
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

Keep it concise (150-250 words) and actionable."""
            messages = [
                {"role": "system", "content": "You are an expert treasury operations analyst specializing in stablecoin transactions and route optimization."},
                {"role": "user", "content": prompt}
            ]
            return self._call_openai_chat(messages, temperature=0.7, max_tokens=500)
        else:
            total = data_dict.get("total_transactions", 0)
            vol = data_dict.get("total_volume", 0.0)
            avg_cost = data_dict.get("avg_cost_bps", 0.0)
            success = data_dict.get("success_rate", 0.0)
            failed = data_dict.get("failed_count", 0)
            lines = []
            lines.append(f"Daily summary: {total} transactions totaling ${vol:,.2f}.")
            lines.append(f"Average cost was {avg_cost:.2f} BPS with success rate {success:.1f}%.")
            if failed:
                lines.append(f"{failed} transactions failed — investigate failure cluster.")
            vol_chg = data_dict.get("volume_change", 0)
            if vol_chg > 5:
                lines.append(f"Volume up {vol_chg:.1f}% vs previous period — higher throughput.")
            elif vol_chg < -5:
                lines.append(f"Volume down {vol_chg:.1f}% vs previous period — check traffic sources.")
            if data_dict.get("compliance_rate", 100) < 95:
                lines.append("Compliance rate below 95% — priority to remediate.")
            lines.append("Outlook: monitor cost and failed transactions; consider routing adjustments if costs remain elevated.")
            return " ".join(lines)

    def analyze_exceptions(self, exceptions_data: List[Dict]) -> str:
        if self.use_openai:
            prompt = f"""You are a treasury operations expert. Analyze the following transaction exceptions and provide specific remediation suggestions.

Exceptions Data:
{json.dumps(exceptions_data, indent=2)}

For each exception category, provide:
1. Root cause analysis
2. Immediate remediation steps
3. Long-term prevention strategies
4. Priority level (High/Medium/Low)

Format your response with clear sections and actionable recommendations."""
            messages = [
                {"role": "system", "content": "You are an expert in treasury operations, blockchain transactions, and operational risk management."},
                {"role": "user", "content": prompt}
            ]
            return self._call_openai_chat(messages, temperature=0.6, max_tokens=1000)
        else:
            if not exceptions_data:
                return "No exceptions detected."
            parts = []
            for ex in exceptions_data:
                t = ex.get("type", "Unknown")
                cnt = ex.get("count", 0)
                sev = ex.get("severity", "Medium")
                details = ex.get("details", "")
                parts.append(f"Exception: {t} (count={cnt}, severity={sev}). {details}")
                if t.lower().startswith("failed"):
                    parts.append("Remediation: retry logic with exponential backoff; capture transfer logs and inspect venue failures.")
                elif "compliance" in t.lower():
                    parts.append("Remediation: quarantine affected transfers, flag merchant, and require manual review.")
                elif "cost" in t.lower():
                    parts.append("Remediation: investigate venue cost spikes, consider rerouting to lower-cost venues.")
                else:
                    parts.append("Remediation: investigate logs and categorize root cause.")
            return "\n\n".join(parts)

    def generate_optimization_recommendations(self, route_data: Dict) -> str:
        if self.use_openai:
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
            messages = [
                {"role": "system", "content": "You are an expert in stablecoin routing, DeFi protocols, and transaction optimization."},
                {"role": "user", "content": prompt}
            ]
            return self._call_openai_chat(messages, temperature=0.6, max_tokens=800)
        else:
            routes = route_data.get("top_routes", []) or []
            recs = []
            if not routes:
                return "No route data available to recommend optimization."
            sorted_routes = sorted(routes, key=lambda r: r.get("avg_cost_bps", 0), reverse=True)
            for i, r in enumerate(sorted_routes[:3]):
                recs.append(f"{i+1}. Route {r.get('route')}: avg cost {r.get('avg_cost_bps', 0):.2f} BPS, volume {r.get('volume',0)}. Suggest: evaluate alternative venues, cap slippage, and rebalance liquidity.")
            recs.append("Consider enabling low-cost venues for smaller volumes and using batching to reduce per-transfer overhead.")
            return "\n".join(recs)

    def analyze_cost_anomalies(self, anomaly_data: List[Dict]) -> str:
        if self.use_openai:
            prompt = f"""Analyze the following cost anomalies in stablecoin transactions and explain potential causes.

Anomalous Transactions:
{json.dumps(anomaly_data[:10], indent=2)}

Provide:
1. Likely reasons for high costs
2. Whether these are expected or concerning
3. Recommended investigation steps
4. Preventive measures"""
            messages = [
                {"role": "system", "content": "You are an expert in blockchain transaction costs, gas fees, and DeFi protocols."},
                {"role": "user", "content": prompt}
            ]
            return self._call_openai_chat(messages, temperature=0.5, max_tokens=600)
        else:
            if not anomaly_data:
                return "No cost anomalies detected."
            reasons = []
            for a in anomaly_data[:5]:
                reason = "Possible venue liquidity issue"
                if a.get("gas_cost", 0) > 1.0:
                    reason = "High gas costs"
                if a.get("slippage", 0) > (0.01 * (a.get("amount", 1) or 1)):
                    reason = "High slippage due to illiquid route"
                reasons.append(f"- Transfer {a.get('transfer_id','N/A')}: {reason}.")
            return "Potential reasons:\n" + "\n".join(reasons)

    def generate_executive_insights(self, summary_data: Dict) -> str:
        if self.use_openai:
            prompt = f"""As a CFO advisor, provide executive-level insights based on this treasury data.

Summary Metrics:
{json.dumps(summary_data, indent=2)}

Provide a brief executive summary covering:
1. Key takeaways (3-4 bullet points)
2. Strategic implications
3. Risk assessment
4. Recommended focus areas

Keep it strategic and suitable for C-level audience (150-200 words)."""
            messages = [
                {"role": "system", "content": "You are a CFO advisor specializing in digital treasury and blockchain operations."},
                {"role": "user", "content": prompt}
            ]
            return self._call_openai_chat(messages, temperature=0.7, max_tokens=400)
        else:
            top = []
            tv = summary_data.get("total_volume", 0)
            br = summary_data.get("avg_cost_bps", 0)
            top.append(f"- Volume: ${tv:,.0f}")
            top.append(f"- Avg cost: {br:.2f} BPS")
            if summary_data.get("success_rate", 100) < 98:
                top.append("- Some settlement friction exists; review failed transactions.")
            top.append("- Recommendation: prioritize cost reduction opportunities and ensure compliance coverage.")
            return "\n".join(top)

    def compare_periods(self, current_data: Dict, previous_data: Dict) -> str:
        if self.use_openai:
            prompt = f"""Compare these two periods and explain significant changes.
Current Period:
{json.dumps(current_data, indent=2)}

Previous Period:
{json.dumps(previous_data, indent=2)}

Analyze:
1. Most significant changes and their implications
2. Improving vs. declining metrics
3. Potential causes for major shifts
4. Recommended actions based on trends"""
            messages = [
                {"role": "system", "content": "You are a treasury analyst specializing in period-over-period analysis and trend identification."},
                {"role": "user", "content": prompt}
            ]
            return self._call_openai_chat(messages, temperature=0.6, max_tokens=600)
        else:
            out = []
            for k in ("total_volume", "avg_cost_bps", "success_rate"):
                c = current_data.get(k)
                p = previous_data.get(k)
                if c is None or p is None:
                    continue
                try:
                    delta = (c - p)
                    pct = (delta / p * 100) if p else 0.0
                except Exception:
                    pct = 0.0
                out.append(f"{k}: current={c}, previous={p}, delta={delta:+.2f}, pct_change={pct:+.1f}%")
            if not out:
                return "No comparable metrics found between periods."
            return "\n".join(out)

# ----------------- Heuristic utilities for dashboard generate_insights -----------------
def _heuristic_summarize_metrics(metrics: Dict[str, Any], recent_batches: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build a minimal structured summary (deterministic) used by fallback generate_insights.

    Prefer numeric metrics passed in via `metrics` (from dashboard / tests).
    If `metrics` is empty or missing keys, fall back to defaults (do not import registry by default).
    """
    # safe defaults
    default_batches_started = 0
    default_batches_in_progress = 0
    default_avg_processing_seconds = 0.0

    # Prefer values provided in the `metrics` dict (which Tests pass in).
    try:
        bs = metrics.get("batches_started_total", None) if isinstance(metrics, dict) else None
        bip = metrics.get("batches_in_progress", None) if isinstance(metrics, dict) else None
        avg = metrics.get("avg_processing_seconds", None) if isinstance(metrics, dict) else None
    except Exception:
        bs = bip = avg = None

    # If any are missing, fall back to defaults (avoid importing metric objects)
    try:
        batches_started_total = int(bs) if bs is not None else default_batches_started
    except Exception:
        batches_started_total = default_batches_started

    try:
        batches_in_progress = int(bip) if bip is not None else default_batches_in_progress
    except Exception:
        batches_in_progress = default_batches_in_progress

    try:
        avg_processing_seconds = float(avg) if avg is not None else default_avg_processing_seconds
    except Exception:
        avg_processing_seconds = default_avg_processing_seconds

    summary = {}
    summary["batches_started_total"] = batches_started_total
    summary["batches_in_progress"] = batches_in_progress
    summary["avg_processing_seconds"] = avg_processing_seconds

    # detect failures in recent batches
    failures = [b for b in recent_batches if b.get("status") == "FAILED"]
    summary["recent_failures"] = len(failures)

    # gather durations from per-batch kpis (if present)
    kpis = [b.get("kpis", {}) for b in recent_batches]
    durations = [k.get("duration_seconds", 0) for k in kpis if isinstance(k, dict) and "duration_seconds" in k]
    if durations:
        # median-ish
        durations_sorted = sorted(durations)
        summary["median_duration"] = float(durations_sorted[len(durations_sorted)//2])
    else:
        summary["median_duration"] = summary["avg_processing_seconds"]

    return summary

def _heuristic_generate_insights_obj(input_obj: Dict[str, Any]) -> Dict[str, Any]:
    metrics = input_obj.get("metrics", {}) or {}
    recent_batches = input_obj.get("recent_batches", []) or []

    summary_struct = _heuristic_summarize_metrics(metrics, recent_batches)
    summary_lines = []
    summary_lines.append(f"{summary_struct['batches_started_total']} batches started; {summary_struct['batches_in_progress']} in progress.")
    summary_lines.append(f"Median processing time ~ {summary_struct['median_duration']:.2f}s.")
    if summary_struct["recent_failures"] > 0:
        summary_lines.append(f"{summary_struct['recent_failures']} recent failures detected — inspect worker logs and Redis connectivity.")
    else:
        summary_lines.append("No recent failures detected.")

    anomalies = []
    if summary_struct["batches_in_progress"] > max(10, 2):
        anomalies.append({"severity": "MEDIUM", "description": "High concurrency: many batches in progress."})
    if summary_struct["median_duration"] > 5.0:
        anomalies.append({"severity": "MEDIUM", "description": f"Slower processing: median duration {summary_struct['median_duration']:.2f}s."})
    if summary_struct["recent_failures"] > 0:
        anomalies.append({"severity": "HIGH", "description": f"{summary_struct['recent_failures']} failed batches in recent history."})

    seed = _deterministic_hash(str(summary_struct["batches_started_total"]))
    recs = []
    if summary_struct["recent_failures"] > 0:
        recs.append({"asset_from": "N/A", "asset_to": "N/A", "amount": 0, "rationale": "Investigate failure root causes; retry failed transfers."})
    if summary_struct["median_duration"] > 5.0:
        recs.append({"asset_from": "N/A", "asset_to": "N/A", "amount": 0, "rationale": "Consider scaling workers or reducing per-transfer compute."})
    recs.append({"asset_from": "USDC", "asset_to": "USDC", "amount": 0, "rationale": "No rebalancing suggested; review routing policies regularly."})

    return {
        "summary": " ".join(summary_lines),
        "anomalies": anomalies,
        "recommendations": recs
    }

# ----------------- Top-level helper used by dashboard -----------------
@cache_result(ttl_seconds=600)  # default cache TTL 10 minutes
def generate_insights(input_obj: Dict[str, Any], model: str = "gpt-4o-mini") -> Dict[str, Any]:
    if OPENAI_AVAILABLE and OPENAI_API_KEY:
        PROMPT_SUMMARY = """
You are an AI assistant for a payments treasury system.
Input: a JSON object with KPI timeseries and recent abnormal transfers.
Task: Produce:
1) A concise summary paragraph (2-3 sentences).
2) List of anomalies with severity (LOW/MEDIUM/HIGH).
3) Suggested rebalancing moves (structured: asset_from, asset_to, amount, rationale).
Output must be JSON with keys: summary, anomalies, recommendations.
Here is the input:
{input_json}
"""
        prompt = PROMPT_SUMMARY.format(input_json=json.dumps(input_obj))
        messages = [
            {"role": "system", "content": "You are an assistant that outputs JSON with keys: summary, anomalies, recommendations."},
            {"role": "user", "content": prompt}
        ]

        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=600
            )
            text = resp.choices[0].message["content"].strip()
            try:
                return json.loads(text)
            except Exception:
                import re
                m = re.search(r"(\{[\s\S]*\})", text)
                if m:
                    try:
                        return json.loads(m.group(1))
                    except Exception:
                        pass
                return {"summary": text, "anomalies": [], "recommendations": []}
        except Exception as e:
            base = _heuristic_generate_insights_obj(input_obj)
            base["summary"] = f"[AI error: {type(e).__name__}] {str(e)}\n\n" + base["summary"]
            return base
    else:
        return _heuristic_generate_insights_obj(input_obj)
