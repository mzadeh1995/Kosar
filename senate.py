# ==============================================================================
# 🏛️ senate.py
# ------------------------------------------------------------------------------
# Multi-provider AI routing, deliberative senate, and senator weight logic.
# v49.1: Added "Calibration" and Patch
# Senate deliberation/weighting behavior preserved.
# ==============================================================================

from __future__ import annotations

import os
import json
import asyncio
from typing import Any, Dict, Optional, Tuple

import aiohttp
import pandas as pd

# AI SDKs
import openai
import cohere  # kept (not used for v2 HTTP), but fine
from google.cloud import aiplatform

from config import (
    CONFIG,
    TZ,
    SCHEMA_VERSION,
    MASTER_ANALYSIS_PROMPT,
    MASTER_DEBATE_PROMPT_TEMPLATE,
    SENATORS,
)

from portfolio import pm

from telemetry_utils import (
    logger,
    wrap_prompt,
    extract_clean_json,
    validate_analysis_json,
    get_api_key,
    get_gcp_auth_token,
    build_vertex_openapi_base_url,
    safe_symbol,
    write_json,
    append_jsonl,
    assert_feature_contract_event,
    assert_feature_contract_transcript,
)

from datetime import datetime, timedelta

# 🧠 Weights (unchanged)
# -------------------------
def _safe_fromiso(s: str):
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = TZ.localize(dt)
        return dt
    except Exception:
        return None

def calculate_senator_weights():
    logger.info("⚖️ Checking conditions for senator weight recalculation...")
    history = pm.get_trade_history()
    if not history:
        logger.warning("No trade history found. Using default weights.")
        return {name: 1.0 for name in SENATORS.keys()}

    first_trade_close_time = _safe_fromiso(history[0].get("close_time", ""))
    if not first_trade_close_time or (datetime.now(TZ) - first_trade_close_time) < timedelta(days=30):
        logger.info("Less than 30 days since first valid trade. Deferring. Using default weights.")
        return {name: 1.0 for name in SENATORS.keys()}

    logger.info("✅ Conditions met. Recalculating weights based on last 30 days of performance...")
    thirty_days_ago = datetime.now(TZ) - timedelta(days=30)
    scores = {name: 0.0 for name in SENATORS.keys()}

    recent_trades = []
    for t in history:
        ct = _safe_fromiso(t.get("close_time", ""))
        if ct and ct > thirty_days_ago:
            recent_trades.append(t)

    for trade in recent_trades:
        graded_votes = trade.get("senator_deliberations", {}).get("final_analyses_graded", {})
        for senator, vote_details in graded_votes.items():
            if senator not in scores:
                continue
            is_correct = vote_details.get("correct")
            realized_r = trade.get("realized_r", 0.0)
            if abs(realized_r) < 0.11:
                continue
            if is_correct is True:
                scores[senator] += abs(realized_r)
            elif is_correct is False:
                scores[senator] -= abs(realized_r)

    max_abs_score = max(abs(s) for s in scores.values()) if scores else 1.0
    if max_abs_score == 0:
        max_abs_score = 1.0

    weights = {}
    for name, score in scores.items():
        normalized_score = score / max_abs_score
        weight = 1.0 + normalized_score
        weights[name] = max(0.2, min(2.0, weight))

    pm._write_json(pm.weights_file, {"last_updated": datetime.now(TZ).isoformat(), "weights": weights})
    logger.info("✅ New senator weights calculated and saved.")
    return weights

def load_or_recalculate_weights():
    weight_data = pm._read_json(pm.weights_file, {})
    last_updated_str = weight_data.get("last_updated")
    if last_updated_str:
        dt = _safe_fromiso(last_updated_str)
        if dt and (datetime.now(TZ) - dt) < timedelta(days=30):
            logger.info("Using existing senator weights (less than 30 days old).")
            return weight_data.get("weights", {name: 1.0 for name in SENATORS.keys()})
    return calculate_senator_weights()


# -------------------------

# -------------------------
# 🔗 COHERE v2 (/v2/chat)
# -------------------------
async def call_cohere_v2(session: aiohttp.ClientSession, prompt: str, model_id: str) -> Optional[str]:
    api_key = get_api_key("Cohere", ["COHERE_API_KEY"])
    url = "https://api.cohere.com/v2/chat"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        "temperature": CONFIG["API_TEMPERATURE"],
        "max_tokens": CONFIG["MAX_TOKENS"],
    }

    try:
        async with session.post(url, headers=headers, json=payload, timeout=120) as resp:
            data = await resp.json(content_type=None)
            if resp.status < 200 or resp.status >= 300:
                return None

            msg = data.get("message") or {}
            content = msg.get("content")
            if isinstance(content, list):
                texts = []
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str):
                        texts.append(b["text"])
                if texts:
                    return "\n".join(texts).strip()

            if isinstance(data.get("text"), str):
                return data["text"].strip()

            return None
    except Exception:
        return None



# -------------------------
# 🧠 MISTRAL rawPredict (doc-compliant payload)
# -------------------------
async def call_mistral_rawpredict(
    session: aiohttp.ClientSession,
    project_id: str,
    location: str,
    model_id: str,
    token: str,
    sys_prompt: str,
    user_prompt: str,
) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """
    Returns: (assistant_text, http_status, raw_body_preview)
    """
    url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/mistralai/models/{model_id}:rawPredict"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=utf-8"}

    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "max_tokens": CONFIG["MAX_TOKENS"],
        "temperature": CONFIG["API_TEMPERATURE"],
    }

    try:
        async with session.post(url, headers=headers, json=payload, timeout=120) as resp:
            raw = await resp.text()
            preview = raw[:2000] if isinstance(raw, str) else None
            if 200 <= resp.status < 300:
                try:
                    data = json.loads(raw)
                    content = data.get("choices", [{}])[0].get("message", {}).get("content")
                    return content, resp.status, preview
                except Exception:
                    return None, resp.status, preview
            return None, resp.status, preview
    except Exception as e:
        return None, None, str(e)[:2000]

# -------------------------
# 🧠 Provider routing (OpenAI SDK)
# -------------------------
def _provider_client(model_type: str) -> Tuple[str, Optional[str]]:
    if model_type == "openai_direct":
        return get_api_key("OpenAI", ["OPENAI_API_KEY"]), None
    if model_type == "xai_direct":
        return get_api_key("xAI", ["XAI_API_KEY", "GROK_API_KEY"]), "https://api.x.ai/v1"
    if model_type == "perplexity_direct":
        return get_api_key("Perplexity", ["PERPLEXITY_API_KEY", "PPLX_API_KEY"]), "https://api.perplexity.ai"
    if model_type == "openrouter_direct":
        return get_api_key("OpenRouter", ["OPENROUTER_API_KEY"]), "https://openrouter.ai/api/v1"
    raise ValueError("Unknown provider client type")


def _perplexity_response_format(debate: bool) -> dict:
    if debate:
        schema = {
            "name": "final_vote_schema",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "final_vote": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
                    "final_confidence": {"type": "integer", "minimum": 0, "maximum": 100},
                    "changed_opinion": {"type": "boolean"},
                    "reason_for_final_decision": {"type": "string"},
                },
                "required": ["final_vote", "final_confidence", "changed_opinion", "reason_for_final_decision"],
            },
            "strict": True,
        }
    else:
        schema = {
            "name": "initial_vote_schema",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "vote": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
                    "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
                    "reason": {"type": "string"},
                },
                "required": ["vote", "confidence", "reason"],
            },
            "strict": True,
        }

    return {"type": "json_schema", "json_schema": schema}


# -------------------------

async def ask_senator(
    session: aiohttp.ClientSession,
    cycle_id: str,
    symbol: str,
    phase: int,
    senator_name: str,
    cfg: Dict[str, Any],
    sys_prompt: str,
    user_prompt_content: str,
    debate_round: bool,
) -> Tuple[Optional[dict], Optional[str], dict]:
    """
    Returns: (validated_json, raw_text, event_meta)
    """
    model_type = cfg["type"]
    model_id = cfg["id"]
    location = cfg.get("location", CONFIG["GCP_LOCATION"])

    meta_base = {
        "schema_version": SCHEMA_VERSION,
        "ts": datetime.now(TZ).isoformat(),
        "cycle_id": cycle_id,
        "symbol": symbol,
        "phase": phase,
        "debate_round": debate_round,
        "senator": senator_name,
        "provider": cfg.get("provider"),
        "model_type": model_type,
        "model_id": model_id,
        "location": location,
    }

    logger.info(f"🧠 Calling {senator_name} | type={model_type} | phase={phase} | debate={debate_round}")

    raw_text = None

    for attempt in range(1, 4):
        fail_stage = None
        err_msg = None
        http_status = None
        response_preview = None
        parsed = None
        validated = None

        try:
            max_tokens = int(CONFIG["MAX_TOKENS"])
            temp = float(CONFIG["API_TEMPERATURE"])

            if model_type == "google_native":
                await asyncio.to_thread(aiplatform.init, project=CONFIG["GCP_PROJECT_ID"], location=location)
                from vertexai.generative_models import GenerativeModel
                model = GenerativeModel(model_id)
                final_prompt = wrap_prompt(sys_prompt, user_prompt_content)
                resp = await asyncio.to_thread(
                    model.generate_content,
                    final_prompt,
                    generation_config={"temperature": temp, "max_output_tokens": max_tokens},
                )
                raw_text = getattr(resp, "text", None)

            elif model_type == "vertex_openai_compat":
                token = await asyncio.to_thread(get_gcp_auth_token)
                api_version = cfg.get("api_version", CONFIG["DEFAULT_VERTEX_OPENAI_COMPAT_VERSION"])
                base_url = build_vertex_openapi_base_url(CONFIG["GCP_PROJECT_ID"], location, api_version)
                client = openai.AsyncOpenAI(api_key=token, base_url=base_url)
                resp = await client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt_content}],
                    temperature=temp,
                    max_tokens=max_tokens,
                )
                raw_text = resp.choices[0].message.content

            elif model_type == "mistral_rawpredict":
                token = await asyncio.to_thread(get_gcp_auth_token)
                raw_text, http_status, response_preview = await call_mistral_rawpredict(
                    session,
                    CONFIG["GCP_PROJECT_ID"],
                    location,
                    model_id,
                    token,
                    sys_prompt,
                    user_prompt_content,
                )

            elif model_type in ["openai_direct", "xai_direct", "perplexity_direct", "openrouter_direct"]:
                api_key, base_url = _provider_client(model_type)
                client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

                kwargs = dict(
                    model=model_id,
                    messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt_content}],
                    temperature=temp,
                )
                if model_type == "openai_direct" and str(model_id).startswith("gpt-5"):
                    kwargs["max_completion_tokens"] = max_tokens
                else:
                    kwargs["max_tokens"] = max_tokens

                if model_type == "perplexity_direct":
                    kwargs["response_format"] = _perplexity_response_format(debate_round)
                    kwargs["temperature"] = 0.0

                resp = await client.chat.completions.create(**kwargs)
                raw_text = resp.choices[0].message.content

            elif model_type == "cohere_v2":
                final_prompt = wrap_prompt(sys_prompt, user_prompt_content)
                raw_text = await call_cohere_v2(session, final_prompt, model_id)

            if raw_text:
                response_preview = (raw_text[:400] + "…") if len(raw_text) > 400 else raw_text
                parsed = extract_clean_json(raw_text)
                validated = validate_analysis_json(parsed, debate_round) if parsed else None

                # Perplexity fallback: if it returned <think> w/out JSON, re-ask once
                if (model_type == "perplexity_direct") and (validated is None) and attempt == 1:
                    fail_stage = "json_parse_or_validation"
                    hard_sys = "Return ONLY the JSON object. No <think>, no markdown, no extra text."
                    hard_user = user_prompt_content
                    api_key, base_url = _provider_client(model_type)
                    client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
                    resp2 = await client.chat.completions.create(
                        model=model_id,
                        messages=[{"role": "system", "content": hard_sys}, {"role": "user", "content": hard_user}],
                        temperature=0.0,
                        max_tokens=512,
                        response_format=_perplexity_response_format(debate_round),
                    )
                    raw_text = resp2.choices[0].message.content
                    response_preview = (raw_text[:400] + "…") if len(raw_text) > 400 else raw_text
                    parsed = extract_clean_json(raw_text)
                    validated = validate_analysis_json(parsed, debate_round) if parsed else None

            if validated:
                ev = {
                    **meta_base,
                    "attempt": attempt,
                    "ok": True,
                    "fail_stage": None,
                    "error": None,
                    "http_status": http_status,
                    "response_preview": response_preview,
                    "parsed": validated,
                }
                assert_feature_contract_event(ev)
                if CONFIG["TELEMETRY_SENATE_EVENTS"]:
                    append_jsonl(CONFIG["SENATE_EVENTS_FILE"], ev)
                return validated, raw_text, ev

            fail_stage = fail_stage or ("empty_response" if not raw_text else "validation_fail")
            ev = {
                **meta_base,
                "attempt": attempt,
                "ok": False,
                "fail_stage": fail_stage,
                "error": err_msg,
                "http_status": http_status,
                "response_preview": response_preview,
                "parsed": parsed,
            }
            assert_feature_contract_event(ev)
            if CONFIG["TELEMETRY_SENATE_EVENTS"]:
                append_jsonl(CONFIG["SENATE_EVENTS_FILE"], ev)

        except Exception as e:
            fail_stage = "api_exception"
            err_msg = f"{type(e).__name__}: {e}"
            ev = {
                **meta_base,
                "attempt": attempt,
                "ok": False,
                "fail_stage": fail_stage,
                "error": err_msg,
                "http_status": http_status,
                "response_preview": response_preview,
                "parsed": None,
            }
            try:
                assert_feature_contract_event(ev)
            except Exception as _:
                pass
            if CONFIG["TELEMETRY_SENATE_EVENTS"]:
                append_jsonl(CONFIG["SENATE_EVENTS_FILE"], ev)
            logger.error(f"❌ API call failed [{senator_name}] attempt {attempt}: {err_msg}")

        if attempt < 3:
            await asyncio.sleep(2 * attempt)

    return None, raw_text, {"ok": False, **meta_base}



# -------------------------
# 🏛️ SENATE (FULL TRANSCRIPTS)
# -------------------------
async def convene_deliberative_senate(
    session: aiohttp.ClientSession,
    cycle_id: str,
    symbol: str,
    df: pd.DataFrame,
    market_data: Dict[str, Any],
    weights: Dict[str, float],
) -> Tuple[str, dict]:

    logger.info(f"🏛️ CONVENING DELIBERATIVE SENATE for {symbol}...")

    def _fmt_dashboard_value(value: Any, decimals: int = 6) -> str:
        if value is None:
            return "N/A"
        try:
            numeric_value = float(value)
            if pd.isna(numeric_value):
                return "N/A"
            return f"{numeric_value:.{decimals}f}"
        except Exception:
            return str(value)

    def _fmt_flag(value: Any) -> str:
        if value is None:
            return "N/A"
        if isinstance(value, bool):
            return "True" if value else "False"
        return str(value)

    raw_feature_cols = market_data.get("hmm_feature_columns", [])
    if isinstance(raw_feature_cols, list):
        hmm_feature_cols_display = ", ".join(str(col) for col in raw_feature_cols) if raw_feature_cols else "N/A"
    else:
        hmm_feature_cols_display = "N/A"

    technical_dashboard = (
        "TECHNICAL DASHBOARD\n"
        f"Symbol: {symbol}\n"
        f"- RSI(14): {_fmt_dashboard_value(market_data.get('rsi'), 2)}\n"
        f"- ADX(14): {_fmt_dashboard_value(market_data.get('adx'), 2)}\n"
        f"- ATR(14): {_fmt_dashboard_value(market_data.get('atr'), 6)}\n"
        f"- RVOL(20): {_fmt_dashboard_value(market_data.get('rvol'), 2)}\n"
        f"- EMA50: {_fmt_dashboard_value(market_data.get('ema50'), 6)}\n"
        f"- EMA200: {_fmt_dashboard_value(market_data.get('ema200'), 6)}\n"
        f"- HMM Regime: {market_data.get('hmm_regime', 'unknown')}\n"
        f"- HMM Confidence: {_fmt_dashboard_value(market_data.get('hmm_confidence'), 4)}\n"
        f"- HMM Filtered Confidence: {_fmt_dashboard_value(market_data.get('hmm_filtered_confidence'), 4)}\n"
        f"- HMM Bull Prob: {_fmt_dashboard_value(market_data.get('hmm_bull_prob'), 4)}\n"
        f"- HMM Neutral Prob: {_fmt_dashboard_value(market_data.get('hmm_neutral_prob'), 4)}\n"
        f"- HMM Bear Prob: {_fmt_dashboard_value(market_data.get('hmm_bear_prob'), 4)}\n"
        f"- HMM Policy: {market_data.get('hmm_policy', 'caution')}\n"
        f"- HMM Persistence: {_fmt_dashboard_value(market_data.get('hmm_persistence'), 4)}\n"
        f"- HMM Previous Regime: {market_data.get('hmm_prev_regime', 'N/A')}\n"
        f"- HMM Regime Changed: {market_data.get('hmm_regime_changed', False)}\n"
        f"- HMM Regime Age Bars: {market_data.get('hmm_regime_age_bars', 0)}\n"
        f"- HMM Switch Margin: {_fmt_dashboard_value(market_data.get('hmm_switch_margin'), 4)}\n"
        f"- HMM Feature Mode: {market_data.get('hmm_feature_mode', 'unknown')}\n"
        f"- HMM Feature Rows: {market_data.get('hmm_feature_rows', 'N/A')}\n"
        f"- HMM Feature Columns: {hmm_feature_cols_display}\n"
        f"- HMM Fractional Diff d: {_fmt_dashboard_value(market_data.get('hmm_frac_diff_d', market_data.get('hmm_fracdiff_d_used')), 4)}\n"
        f"- HMM Fractional Diff Threshold: {_fmt_dashboard_value(market_data.get('hmm_frac_diff_threshold', market_data.get('hmm_fracdiff_threshold')), 8)}\n\n"
        "MICROSTRUCTURE DASHBOARD\n"
        f"- Microstructure OK: {_fmt_flag(market_data.get('microstructure_ok'))}\n"
        f"- OFI OK: {_fmt_flag(market_data.get('ofi_ok'))}\n"
        f"- OFI L1: {_fmt_dashboard_value(market_data.get('ofi_l1'), 6)}\n"
        f"- OFI L1 Norm: {_fmt_dashboard_value(market_data.get('ofi_l1_norm'), 8)}\n"
        f"- OFI L1 Z: {_fmt_dashboard_value(market_data.get('ofi_l1_z'), 6)}\n"
        f"- OFI Depth Imbalance: {_fmt_dashboard_value(market_data.get('ofi_depth_imbalance'), 6)}\n"
        f"- OFI Spread bps: {_fmt_dashboard_value(market_data.get('ofi_spread_bps'), 6)}\n"
        f"- OFI Liquidity USD: {_fmt_dashboard_value(market_data.get('ofi_liquidity_usd'), 2)}\n"
        f"- VPIN OK: {_fmt_flag(market_data.get('vpin_ok'))}\n"
        f"- VPIN: {_fmt_dashboard_value(market_data.get('vpin'), 6)}\n"
        f"- VPIN Z: {_fmt_dashboard_value(market_data.get('vpin_z'), 6)}\n"
        f"- VPIN Buckets: {_fmt_dashboard_value(market_data.get('vpin_bucket_count'), 0)}\n"
        f"- VPIN Buy Volume: {_fmt_dashboard_value(market_data.get('vpin_buy_volume'), 2)}\n"
        f"- VPIN Sell Volume: {_fmt_dashboard_value(market_data.get('vpin_sell_volume'), 2)}\n"
        f"- VPIN Total Volume: {_fmt_dashboard_value(market_data.get('vpin_total_volume'), 2)}"
    )
    regime_instruction = (
        "REGIME NOTE: Give meaningful weight to the inferred probabilistic market regime, "
        "but do not treat it as a guaranteed directional signal. Treat freshly switched or low-persistence regimes with extra caution."
    )
    microstructure_instruction = (
        "MICROSTRUCTURE NOTE: OFI and VPIN are early warning features for order-flow imbalance, liquidity withdrawal, "
        "toxic flow, and news-driven shock behavior. Treat them as risk/context features only. "
        "They are not standalone buy/sell signals and must not override the rest of the system by themselves."
    )

    price_data_txt = df.tail(50)[["Open", "High", "Low", "Close", "Volume"]].to_string()
    data_txt = (
        f"{technical_dashboard}\n{regime_instruction}\n{microstructure_instruction}\n\n"
        f"RECENT PRICE DATA (LAST 50 CANDLES)\n{price_data_txt}"
    )

    # Phase 1
    logger.info("Phase 1: Gathering initial analyses...")
    phase1_sys = MASTER_ANALYSIS_PROMPT
    phase1_user = data_txt

    initial_tasks = [
        ask_senator(session, cycle_id, symbol, 1, name, cfg, phase1_sys, phase1_user, debate_round=False)
        for name, cfg in SENATORS.items()
    ]
    initial_results = await asyncio.gather(*initial_tasks)

    initial_analyses = {}
    initial_raw = {}
    for name, (validated, raw_text, _meta) in zip(SENATORS.keys(), initial_results):
        if validated:
            initial_analyses[name] = validated
            initial_raw[name] = raw_text

    if len(initial_analyses) < max(1, int(len(SENATORS) * 0.5)):
        logger.warning("Not enough senators responded in Phase 1. Aborting.")
        transcript = {
            "schema_version": SCHEMA_VERSION,
            "cycle_id": cycle_id,
            "symbol": symbol,
            "phase1": {
                "sys_prompt": phase1_sys,
                "user_prompt": phase1_user,
                "responses_parsed": initial_analyses,
                "responses_raw": initial_raw,
            },
            "phase2": {
                "debate_payloads": {},
                "sys_prompt_per_senator": {},
                "user_prompt": data_txt,
                "responses_parsed": {},
                "responses_raw": {},
            },
            "weights_used": weights,
            "consensus": {"buy": 0.0, "sell": 0.0},
            "conviction_scores": {"BUY": 0.0, "SELL": 0.0},
            "decision": "HOLD_ABORT_PHASE1",
        }
        assert_feature_contract_transcript(transcript)
        if CONFIG["TELEMETRY_SENATE_TRANSCRIPTS"]:
            outp = os.path.join(CONFIG["SENATE_TRANSCRIPTS_DIR"], f"{cycle_id}__{safe_symbol(symbol)}.json")
            write_json(outp, transcript)
        return "HOLD", {"initial_analyses": initial_analyses, "final_analyses": {}}

    # Phase 2 debate
    logger.info("Phase 2: Initiating debate round...")
    final_tasks = []
    debate_payloads: Dict[str, Any] = {}
    responding_names = []
    phase2_sys_map = {}

    for name, cfg in SENATORS.items():
        if name not in initial_analyses:
            continue

        mine = initial_analyses[name]
        others_dict = {k: v for k, v in initial_analyses.items() if k != name}

        debate_payloads[name] = {"mine": mine, "others": others_dict}

        colleagues_text = "\n\n".join(
            [f"--- Senator: {k} ---\n{json.dumps(v, ensure_ascii=False, indent=2)}" for k, v in others_dict.items()]
        )
        phase2_sys = MASTER_DEBATE_PROMPT_TEMPLATE.format(
            my_initial_analysis=json.dumps(mine, ensure_ascii=False, indent=2),
            colleagues_analyses=colleagues_text,
        )
        phase2_sys_map[name] = phase2_sys
        phase2_user = data_txt

        responding_names.append(name)
        final_tasks.append(ask_senator(session, cycle_id, symbol, 2, name, cfg, phase2_sys, phase2_user, debate_round=True))

    final_results = await asyncio.gather(*final_tasks)

    final_analyses = {}
    final_raw = {}
    for name, (validated, raw_text, _meta) in zip(responding_names, final_results):
        if validated:
            final_analyses[name] = validated
            final_raw[name] = raw_text

    if len(final_analyses) < max(1, int(len(responding_names) * 0.5)):
        logger.warning("Not enough senators responded in Phase 2. Aborting.")
        transcript = {
            "schema_version": SCHEMA_VERSION,
            "cycle_id": cycle_id,
            "symbol": symbol,
            "phase1": {
                "sys_prompt": phase1_sys,
                "user_prompt": phase1_user,
                "responses_parsed": initial_analyses,
                "responses_raw": initial_raw,
            },
            "phase2": {
                "debate_payloads": debate_payloads,
                "sys_prompt_per_senator": phase2_sys_map,
                "user_prompt": data_txt,
                "responses_parsed": final_analyses,
                "responses_raw": final_raw,
            },
            "weights_used": weights,
            "consensus": {"buy": 0.0, "sell": 0.0},
            "conviction_scores": {"BUY": 0.0, "SELL": 0.0},
            "decision": "HOLD_ABORT_PHASE2",
        }
        assert_feature_contract_transcript(transcript)
        if CONFIG["TELEMETRY_SENATE_TRANSCRIPTS"]:
            outp = os.path.join(CONFIG["SENATE_TRANSCRIPTS_DIR"], f"{cycle_id}__{safe_symbol(symbol)}.json")
            write_json(outp, transcript)
        return "HOLD", {"initial_analyses": initial_analyses, "final_analyses": final_analyses}

    # Weighted consensus (confidence 0..100)
    conviction_scores = {"BUY": 0.0, "SELL": 0.0}
    total_possible_score = 0.0

    for name, data in final_analyses.items():
        vote = data.get("final_vote")
        conf = int(data.get("final_confidence", 0))
        w = float(weights.get(name, 1.0))
        total_possible_score += (100.0 * w)

        if vote in ["BUY", "SELL"] and conf >= CONFIG["MIN_VOTE_CONFIDENCE"]:
            conviction_scores[vote] += (w * conf)

    buy_consensus = conviction_scores["BUY"] / total_possible_score if total_possible_score > 0 else 0.0
    sell_consensus = conviction_scores["SELL"] / total_possible_score if total_possible_score > 0 else 0.0

    logger.info(f"🗳️ FINAL Deliberative Results {symbol}: BUY: {buy_consensus:.1%} | SELL: {sell_consensus:.1%}")

    decision = "HOLD"
    if buy_consensus >= CONFIG["CONSENSUS_THRESHOLD"]:
        decision = "BUY"
    elif sell_consensus >= CONFIG["CONSENSUS_THRESHOLD"]:
        decision = "SELL"

    full_deliberation_data = {"initial_analyses": initial_analyses, "final_analyses": final_analyses}

    transcript = {
        "schema_version": SCHEMA_VERSION,
        "cycle_id": cycle_id,
        "symbol": symbol,
        "phase1": {
            "sys_prompt": phase1_sys,
            "user_prompt": phase1_user,
            "responses_parsed": initial_analyses,
            "responses_raw": initial_raw,
        },
        "phase2": {
            "debate_payloads": debate_payloads,
            "sys_prompt_per_senator": phase2_sys_map,
            "user_prompt": data_txt,
            "responses_parsed": final_analyses,
            "responses_raw": final_raw,
        },
        "weights_used": weights,
        "consensus": {"buy": buy_consensus, "sell": sell_consensus},
        "conviction_scores": conviction_scores,
        "decision": decision,
    }

    assert_feature_contract_transcript(transcript)
    if CONFIG["TELEMETRY_SENATE_TRANSCRIPTS"]:
        outp = os.path.join(CONFIG["SENATE_TRANSCRIPTS_DIR"], f"{cycle_id}__{safe_symbol(symbol)}.json")
        write_json(outp, transcript)

    return decision, full_deliberation_data
