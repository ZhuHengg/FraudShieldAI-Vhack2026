"""
LLM-Powered Investigation Assistant (V4 Enhancement #9)
========================================================
Integrates Google Gemini to provide natural-language explanations
of fraud decisions and support analyst investigations.

Synthesizes SHAP values, mule flags, quarantine status, behavioral
reasons, and feature snapshots into human-readable narratives.
"""
import os
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Gemini client (lazy-initialized)
_client = None
_MODEL_ID = "gemini-2.0-flash"


def _get_client():
    """Lazy-initialize the Gemini client."""
    global _client
    if _client is not None:
        return _client

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set — LLM features disabled")
        return None

    try:
        from google import genai
        _client = genai.Client(api_key=api_key)
        logger.info("Gemini client initialized successfully")
        return _client
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT — defines the LLM's role and output format
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are FraudShield AI, an expert fraud investigation assistant embedded in a real-time fraud detection system for digital wallets in Southeast Asia (ASEAN).

Your role is to analyze transaction risk scores, model outputs, and feature data to provide clear, actionable explanations for fraud analysts.

## Context about the system:
- The system uses a 3-layer ensemble: LightGBM (supervised), Isolation Forest (unsupervised), and Behavioral Rules
- Scores are 0-100 scale. Thresholds: <29.3 = APPROVE, 29.3-64.7 = FLAG, >64.7 = BLOCK
- The system has an Anti-Mule Layer that detects money mule networks (10+ unique senders to same recipient in 60min)
- A Quarantine system prevents model poisoning by validating suspicious labels before they enter retraining

## Guidelines:
1. Be concise but thorough — analysts are busy
2. Always explain the "why" behind the risk score
3. Highlight the most suspicious signals first
4. If a transaction looks like a specific fraud pattern (account takeover, mule network, phishing), name the pattern
5. Suggest concrete next steps for the analyst
6. Use bullet points and clear formatting
7. Never reveal model weights or internal thresholds in your explanation
8. If the transaction is low-risk, briefly confirm why it's safe and mention any minor flags to watch
"""


def build_investigation_context(
    risk_response: dict,
    shap_data: Optional[dict] = None,
    transaction_history: Optional[list] = None,
    quarantine_info: Optional[dict] = None,
) -> str:
    """
    Build a structured context string from all available data sources.
    This becomes the user message to Gemini.
    """
    parts = []

    # ── Risk Score Summary ────────────────────────────────────────────
    parts.append("## Transaction Risk Assessment")
    parts.append(f"- **Transaction ID**: {risk_response.get('transaction_id', 'N/A')}")
    parts.append(f"- **Final Risk Score**: {risk_response.get('risk_score', 0):.1f}/100")
    parts.append(f"- **Decision**: {risk_response.get('risk_level', 'N/A')}")
    parts.append(f"- **Engine Mode**: {risk_response.get('engine_mode', 'full')}")
    parts.append(f"- **Active Models**: {', '.join(risk_response.get('active_models', ['lgb','iso','beh']))}")
    parts.append(f"- **Mule Flag**: {'YES ⚠️' if risk_response.get('mule_flag') else 'No'}")

    # ── Model Scores ─────────────────────────────────────────────────
    parts.append("\n## Model Scores (0-1 scale)")
    parts.append(f"- LightGBM (supervised): {risk_response.get('supervised_score', 0):.4f}")
    parts.append(f"- Isolation Forest (anomaly): {risk_response.get('unsupervised_score', 0):.4f}")
    parts.append(f"- Behavioral Rules: {risk_response.get('behavioral_score', 0):.4f}")

    # ── Behavioral Reasons ───────────────────────────────────────────
    reasons = risk_response.get('reasons', [])
    if reasons:
        parts.append("\n## Triggered Risk Reasons")
        for r in reasons:
            parts.append(f"- {r}")

    # ── Rule Breakdown ───────────────────────────────────────────────
    rb = risk_response.get('rule_breakdown')
    if rb:
        parts.append("\n## Behavioral Rule Breakdown (weighted scores)")
        parts.append(f"- Drain to Unknown: {rb.get('drain_score', 0):.4f}")
        parts.append(f"- Amount Deviation: {rb.get('deviation_score', 0):.4f}")
        parts.append(f"- Context Risk: {rb.get('context_score', 0):.4f}")
        parts.append(f"- Velocity/Urgency: {rb.get('velocity_score', 0):.4f}")

    # ── Feature Snapshot ─────────────────────────────────────────────
    fs = risk_response.get('feature_snapshot')
    if fs:
        parts.append("\n## Key Features at Time of Transaction")
        parts.append(f"- Amount vs Average Ratio: {fs.get('amount_vs_avg_ratio', 1.0):.2f}x")
        parts.append(f"- IP Risk Score: {fs.get('ip_risk_score', 0):.2f}")
        parts.append(f"- Transactions in Last 24h: {fs.get('tx_count_24h', 0)}")
        parts.append(f"- Session Duration: {fs.get('session_duration_seconds', 0):.0f}s")
        parts.append(f"- New Device: {'Yes' if fs.get('is_new_device') else 'No'}")
        parts.append(f"- Country Mismatch: {'Yes' if fs.get('country_mismatch') else 'No'}")
        parts.append(f"- Account Fully Drained: {'Yes' if fs.get('sender_fully_drained') else 'No'}")
        parts.append(f"- New Recipient: {'Yes' if fs.get('is_new_recipient') else 'No'}")
        parts.append(f"- Account Age: {fs.get('account_age_days', 0):.0f} days")
        parts.append(f"- Proxy IP: {'Yes' if fs.get('is_proxy_ip') else 'No'}")

    # ── SHAP Explanations ────────────────────────────────────────────
    if shap_data:
        parts.append("\n## SHAP Feature Contributions (top factors)")
        top = shap_data.get('top_features', [])
        for feat in top:
            direction = "↑ increases" if feat.get('contribution', 0) > 0 else "↓ decreases"
            parts.append(f"- **{feat['feature']}**: {feat['contribution']:.4f} ({direction} risk)")

    # ── Quarantine Info ──────────────────────────────────────────────
    if quarantine_info:
        parts.append("\n## Quarantine Status")
        parts.append(f"- Status: {quarantine_info.get('quarantine_status', 'None')}")
        parts.append(f"- Reason: {quarantine_info.get('quarantine_reason', 'N/A')}")

    # ── Transaction History ──────────────────────────────────────────
    if transaction_history:
        parts.append(f"\n## Recent Transaction History ({len(transaction_history)} recent)")
        for txn in transaction_history[:5]:
            parts.append(
                f"- ID={txn.get('transaction_id', '?')}, "
                f"Amount={txn.get('amount', 0):.2f}, "
                f"Risk={txn.get('ml_risk_score', 0):.1f}%, "
                f"Action={txn.get('action_taken', '?')}"
            )

    return "\n".join(parts)


async def investigate_transaction(
    query: str,
    risk_response: dict,
    shap_data: Optional[dict] = None,
    transaction_history: Optional[list] = None,
    quarantine_info: Optional[dict] = None,
) -> dict:
    """
    Send the transaction context + analyst query to Gemini for investigation.

    Returns:
        {
            "response": str,           # The LLM's analysis
            "model_used": str,         # Which model was used
            "tokens_used": int | None, # Token count if available
            "status": str,             # "success" | "error" | "unavailable"
        }
    """
    client = _get_client()
    if client is None:
        return {
            "response": _generate_fallback_explanation(risk_response),
            "model_used": "rule-based-fallback",
            "tokens_used": None,
            "status": "unavailable",
        }

    # Build context
    context = build_investigation_context(
        risk_response=risk_response,
        shap_data=shap_data,
        transaction_history=transaction_history,
        quarantine_info=quarantine_info,
    )

    # Construct the message
    user_message = f"""{context}

---

## Analyst Question
{query}

Please provide a detailed investigation analysis based on the data above."""

    try:
        from google.genai import types
        import asyncio

        def _call_gemini():
            return client.models.generate_content(
                model=_MODEL_ID,
                contents=user_message,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.3,
                    max_output_tokens=1024,
                ),
            )

        # Run synchronous Gemini call in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, _call_gemini)

        tokens = None
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            tokens = getattr(response.usage_metadata, 'total_token_count', None)

        return {
            "response": response.text,
            "model_used": _MODEL_ID,
            "tokens_used": tokens,
            "status": "success",
        }

    except Exception as e:
        logger.error(f"Gemini API call failed: {e}", exc_info=True)
        return {
            "response": _generate_fallback_explanation(risk_response),
            "model_used": "rule-based-fallback",
            "tokens_used": None,
            "status": "error",
        }


def _generate_fallback_explanation(risk_response: dict) -> str:
    """
    Generate a structured explanation without LLM when Gemini is unavailable.
    Uses the same data but via deterministic templates.
    """
    score = risk_response.get('risk_score', 0)
    level = risk_response.get('risk_level', 'UNKNOWN')
    reasons = risk_response.get('reasons', [])
    fs = risk_response.get('feature_snapshot', {})
    mule = risk_response.get('mule_flag', False)

    lines = []
    lines.append(f"## Risk Assessment: {level} ({score:.1f}/100)\n")

    if mule:
        lines.append("🚨 **MULE NETWORK DETECTED** — This recipient wallet has received funds from 10+ unique senders in the past 60 minutes. This is a strong indicator of a money mule operation.\n")
        lines.append("**Recommended Action:** Immediately freeze the recipient account and escalate to the AML team.\n")
        return "\n".join(lines)

    if level == "HIGH":
        lines.append("⚠️ **HIGH RISK** — This transaction exhibits multiple fraud indicators:\n")
    elif level == "MEDIUM":
        lines.append("🟡 **FLAGGED** — This transaction has some suspicious signals:\n")
    else:
        lines.append("✅ **LOW RISK** — This transaction appears normal.\n")

    if reasons:
        lines.append("**Triggered Signals:**")
        for r in reasons:
            lines.append(f"- {r}")

    # Feature-based insights
    insights = []
    if fs.get('amount_vs_avg_ratio', 1.0) > 3.0:
        insights.append(f"Transaction amount is {fs['amount_vs_avg_ratio']:.1f}x the user's average")
    if fs.get('is_new_device'):
        insights.append("First time this device has been used")
    if fs.get('sender_fully_drained'):
        insights.append("Sender's account was fully drained")
    if fs.get('is_proxy_ip'):
        insights.append("Connection is via a proxy/VPN")
    if fs.get('session_duration_seconds', 999) < 60:
        insights.append(f"Very short session ({fs['session_duration_seconds']:.0f}s) — possible automation")

    if insights:
        lines.append("\n**Key Observations:**")
        for i in insights:
            lines.append(f"- {i}")

    return "\n".join(lines)


async def explain_risk_score(risk_response: dict) -> dict:
    """
    Quick explanation of a risk score — no analyst query needed.
    Uses a focused prompt for fast, concise output.
    """
    return await investigate_transaction(
        query="Explain why this transaction received this risk score. Be concise — 3-4 bullet points max.",
        risk_response=risk_response,
    )
