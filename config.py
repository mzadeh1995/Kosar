# ==============================================================================
# 🧠 Kosar (v47.5)
# ------------------------------------------------------------------------------
# ✅ This is a faithful file-split of the original monolithic bot.
# v49.1: Added "Calibration" and Patch
# while preserving HMM/Senate/gating/portfolio/risk behavior.
# ==============================================================================

from __future__ import annotations

from dotenv import load_dotenv
import os
from typing import Any, Dict

import pytz

# -------------------------
# 🔒 PATH LOCK (IMPORTANT)
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOTENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=DOTENV_PATH, override=False)

# -------------------------
# ⚙️ CONFIG
# -------------------------
VERSION = "47.5"
SCHEMA_VERSION = 2  # bump when transcript/event schemas change

CONFIG: Dict[str, Any] = {
    # Universe (internal symbols; mapped to Binance Spot symbols)
    "UNIVERSE": [
        "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD", "AVAX-USD", "LINK-USD", "DOT-USD",
        "POL28321-USD",
        "LTC-USD", "NEAR-USD", "ATOM-USD", "ALGO-USD", "TRX-USD", "ICP-USD", "INJ-USD", "OP-USD", "ARB-USD",
        "STX4847-USD",
        "FIL-USD",
        "GRT6719-USD",
        "VET-USD",
        "UNI7083-USD",
        "AAVE-USD", "BCH-USD", "ETC-USD",
        "RENDER-USD",
    ],

    "TIMEFRAME": "1h",
    "TOP_CANDIDATES_COUNT": 3,

    # Market data provider (Stage 1: Binance Spot public REST)
    "MARKET_DATA_PROVIDER": "binance_spot",
    "BINANCE_PUBLIC_BASE_URL": "https://data-api.binance.vision",
    "BINANCE_TIMEOUT_SECONDS": 10,
    "BINANCE_RETRY_COUNT": 2,
    "BINANCE_EXCHANGE_INFO_TTL_SECONDS": 3600,
    "BINANCE_MAX_CONCURRENT_REQUESTS": 4,
    "BINANCE_MIN_REQUEST_SPACING_SECONDS": 0.12,
    "BINANCE_BACKOFF_BASE_SECONDS": 0.75,
    "BINANCE_BACKOFF_MAX_SECONDS": 8.0,
    "BINANCE_RESPECT_RETRY_AFTER": True,
    "BINANCE_BOOK_TICKER_TTL_SECONDS": 2.0,
    "BINANCE_PRICE_TTL_SECONDS": 2.0,
    "BINANCE_SYMBOL_MAP": {
        "BTC-USD": "BTCUSDT",
        "ETH-USD": "ETHUSDT",
        "BNB-USD": "BNBUSDT",
        "SOL-USD": "SOLUSDT",
        "XRP-USD": "XRPUSDT",
        "ADA-USD": "ADAUSDT",
        "AVAX-USD": "AVAXUSDT",
        "LINK-USD": "LINKUSDT",
        "DOT-USD": "DOTUSDT",
        "POL28321-USD": "POLUSDT",
        "LTC-USD": "LTCUSDT",
        "NEAR-USD": "NEARUSDT",
        "ATOM-USD": "ATOMUSDT",
        "ALGO-USD": "ALGOUSDT",
        "TRX-USD": "TRXUSDT",
        "ICP-USD": "ICPUSDT",
        "INJ-USD": "INJUSDT",
        "OP-USD": "OPUSDT",
        "ARB-USD": "ARBUSDT",
        "STX4847-USD": "STXUSDT",
        "FIL-USD": "FILUSDT",
        "GRT6719-USD": "GRTUSDT",
        "VET-USD": "VETUSDT",
        "UNI7083-USD": "UNIUSDT",
        "AAVE-USD": "AAVEUSDT",
        "MKR-USD": "MKRUSDT",
        "BCH-USD": "BCHUSDT",
        "ETC-USD": "ETCUSDT",
        "XMR-USD": "XMRUSDT",
        "RENDER-USD": "RENDERUSDT",
    },

    # Senate thresholds
    # Temporary caretaker engine; revert to "senate" to restore LLM voting.
    "DECISION_ENGINE": "sarparast",
    "CONSENSUS_THRESHOLD": 0.61,      # ✅ changed (was 0.70)
    "MIN_VOTE_CONFIDENCE": 65,        # confidence is canonical 0-100

    # Portfolio
    "INITIAL_BALANCE_USD": 10000.0,
    "RISK_PER_TRADE_PCT": 0.02,
    "MAX_OPEN_TRADES": 3,
    "CORRELATION_THRESHOLD": 0.8,

    # Capital allocation rule (No trade can lock > 33% of total equity)
    "CAPITAL_ALLOCATION_MAX_PCT": 0.33,

    # Optional: minimum order margin to avoid dust spam
    "MIN_TRADE_MARGIN_USD": 10.0,

    # Risk / circuit breaker
    "SWING_LOOKBACK_PERIOD": 12,
    "MAX_DRAWDOWN_PCT": 0.10,
    "CIRCUIT_BREAKER_COOLDOWN_HOURS": 24,

    # API
    "API_TEMPERATURE": 0.4,
    "MAX_TOKENS": 4096,  # 🔒 restored (do NOT reduce silently)

    "TIMEZONE": "Europe/Rome",

    # Force mode
    "TRADING_MODE": "spot",           # ✅ default set to spot (SELL ignored)
    "FUTURES_LEVERAGE": 1.0,          # ✅ ALWAYS 1 (your rule) when futures is used

    # Sell Filter (spot-only)
    # Purpose: stop calling the Senate for clearly bearish/downtrend setups in spot (saves spend).
    "SELL_FILTER_ENABLED": True,
    "SELL_FILTER_EMA_LENGTH": 50,
    "SELL_FILTER_SLOPE_LOOKBACK_BARS": 12,

    # HMM regime layer (regime gate only; not a trade decision engine)
    "HMM_ENABLED": True,
    "HMM_BACKEND": "pomegranate",
    "HMM_EMISSION_BACKEND": "pomegranate_normal",
    "HMM_N_STATES": 3,
    "HMM_LOOKBACK_PERIOD": "1y",
    "HMM_BLOCK_BEAR_IN_SPOT": True,
    "HMM_GATING_ENABLED": True,
    "HMM_SEPARATE_DOWNLOAD": True,
    "HMM_MIN_FEATURE_ROWS": 300,
    "HMM_MAX_FEATURE_ROWS": 2000,
    "HMM_MAX_ITER": 50,
    "HMM_TOL": 1e-3,
    "HMM_GMM_COMPONENTS": 2,
    "HMM_GMM_MIN_COMPONENT_ROWS": 30,
    "HMM_GMM_MIN_COV": 1e-3,
    "HMM_GMM_MAX_ITER": 30,
    "HMM_GMM_TOL": 1e-3,
    "HMM_GMM_FALLBACK_TO_NORMAL": True,
    "HMM_GMM_SPLIT_COL": "fd_volatility",
    "HMM_MAP_MATCH_MAX_DIST": 2.0,
    "HMM_MAP_REBASE_MIN_SPREAD": 0.5,
    "HMM_MIN_CONFIDENCE": 0.50,
    "HMM_MIN_REGIME_PROB_MARGIN": 0.05,
    "HMM_ALLOW_CONFIDENCE": 0.58,
    "HMM_BLOCK_CONFIDENCE": 0.60,
    "HMM_STICKY_SELF_TRANSITION": 0.88,
    "HMM_SWITCH_CONFIRM_BARS": 2,
    "HMM_SWITCH_BLOCK_FRESH_BARS": 2,
    "HMM_FAIL_POLICY": "caution",
    "HMM_VOL_WINDOW": 24,
    "HMM_TREND_WINDOW": 8,
    "HMM_RET_EPS": 1e-12,
    "HMM_FEATURE_MODE": "fractional_diff",  # options: "fractional_diff", "legacy"
    "HMM_FRAC_DIFF_D": 0.45,
    "HMM_FRAC_DIFF_THRESHOLD": 1e-5,
    "HMM_FRAC_DIFF_MAX_SIZE": 10000,
    "HMM_FRAC_DIFF_USE_ADF": False,
    "HMM_FRAC_DIFF_ADF_PVALUE": 0.05,
    "HMM_FRAC_DIFF_D_GRID": [0.10, 0.20, 0.30, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00],
    "HMM_FRAC_DIFF_MIN_CORR": 0.30,
    "HMM_REGIME_TREND_COL": "fd_trend",
    "HMM_REGIME_VOL_COL": "fd_volatility",
    "HMM_REGIME_RANGE_COL": "fd_log_range",
    "HMM_ANCHOR_METHOD": "return_trend_volatility",

    # Microstructure features: OFI + VPIN
    "MICROSTRUCTURE_FEATURES_ENABLED": True,
    "MICROSTRUCTURE_FAIL_OPEN": True,
    "OFI_ENABLED": True,
    "OFI_DEPTH_LIMIT": 100,
    "OFI_LEVELS": 10,
    "OFI_SNAPSHOT_COUNT": 3,
    "OFI_SNAPSHOT_INTERVAL_SECONDS": 0.35,
    "OFI_ZSCORE_WINDOW": 50,
    "OFI_MIN_LIQUIDITY_USD": 0.0,
    "OFI_BYPASS_DEPTH_CACHE": True,
    "BINANCE_DEPTH_TTL_SECONDS": 1.0,
    "VPIN_ENABLED": True,
    "VPIN_LOOKBACK_MINUTES": 60,
    "VPIN_AGGTRADES_LIMIT": 1000,
    "VPIN_BUCKET_COUNT": 20,
    "VPIN_BUCKET_VOLUME_MODE": "dynamic_quote",  # options: dynamic_quote, fixed_quote, fixed_base
    "VPIN_FIXED_BUCKET_VOLUME_QUOTE": 100000.0,
    "VPIN_FIXED_BUCKET_VOLUME_BASE": 10.0,
    "VPIN_MIN_BUCKETS": 5,
    "VPIN_ZSCORE_WINDOW": 50,
    "BINANCE_AGGTRADES_TTL_SECONDS": 2.0,
    "FEATURE_ISOLATION_ENABLED": True,
    "FEATURE_ISOLATION_HORIZON_BARS": [1, 3, 6, 12],
    "FEATURE_ISOLATION_OUTPUT_DIR": os.path.join(BASE_DIR, "log", "feature_isolation"),
    "MICROSTRUCTURE_HISTORY_FILE": os.path.join(BASE_DIR, "log", "microstructure_history.jsonl"),
    "RESEARCH_DEFAULT_PERIOD": "1mo",
    "RESEARCH_MIN_ROWS": 200,
    "DATASET_DATA_DIR": os.path.join(BASE_DIR, "data", "binance_vision"),
    "DATASET_OUTPUT_DIR": os.path.join(BASE_DIR, "data", "datasets"),
    "DATASET_DEFAULT_TIMEFRAME": "1h",
    "DATASET_DEFAULT_MONTHS": 24,
    "TRIPLE_BARRIER_HORIZON": 24,
    "TRIPLE_BARRIER_PROFIT_MULT": 4.0,
    "TRIPLE_BARRIER_LOSS_MULT": 2.0,
    "TRIPLE_BARRIER_VOL_WINDOW": 24,
    "TRIPLE_BARRIER_EMBARGO_BARS": 24,
    "TRIPLE_BARRIER_AMBIGUOUS_POLICY": "stop",
    "PRIMARY_MODEL_DIR": os.path.join(BASE_DIR, "data", "models"),
    # Tree models split only by value order: return duplicates log_return as a
    # monotonic transform, and taker_buy_ratio duplicates flow_imb as 2x - 1.
    # hmm_* columns are intentionally excluded from the primary X feature set.
    "PRIMARY_FEATURE_COLUMNS": [
        "log_return", "volatility", "log_range", "volume_change",
        "trend", "flow_imb", "flow_imb_z", "quote_vol_z", "trades_z",
        "ret_autocorr_24", "dist_from_max_168", "dist_from_min_168",
        "hour_of_day", "day_of_week",
        "fd_log_close", "fd_return", "fd_volatility", "fd_log_volume",
        "fd_log_range",
        "close_z_4h", "vol_4h", "vol_ratio_4h_1d", "trend_4h",
        "trend_strength_4h", "trend_1d", "dist_from_max_1d",
        "btc_trend_24", "btc_volatility", "btc_dist_from_max_168",
        "rel_strength",
    ],
    "PRIMARY_CV_SPLITS": 5,
    "PRIMARY_RANDOM_SEED": 41,
    "PRIMARY_RF_PARAMS": {
        "n_estimators": 400,
        "max_depth": None,
        "min_samples_leaf": 100,
        "max_features": "sqrt",
        "class_weight": "balanced_subsample",
        "n_jobs": -1,
    },
    "PRIMARY_FS_VAL_FRACTION": 0.2,
    "PRIMARY_FS_PERM_REPEATS": 5,
    "PRIMARY_FS_KEEP_RATIO": 0.5,
    "PRIMARY_FEE_BPS_PER_SIDE": 10.0,
    "PRIMARY_SLIPPAGE_BPS_PER_SIDE": 5.0,
    "PRIMARY_PROB_THRESHOLDS": [0.30, 0.35, 0.40, 0.45, 0.50, 0.55],
    "PRIMARY_XGB_PARAMS": {
        "n_estimators": 400,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "objective": "binary:logistic",
        "n_jobs": -1,
    },

    # Feature flags for ALL audit/telemetry (default ON)
    "TELEMETRY_SENATE_EVENTS": True,
    "TELEMETRY_SENATE_TRANSCRIPTS": True,
    "TELEMETRY_SCREENER_DETAILS": True,
    "TELEMETRY_SCREENER_EVENTS": True,
    "TELEMETRY_EQUITY_CURVE": True,

    # If you intentionally want telemetry OFF, set env ALLOW_TELEMETRY_DISABLE=1
    # (otherwise the bot will refuse to start with telemetry disabled)

    # Vertex
    "GCP_PROJECT_ID": "shora-478718",
    "GCP_LOCATION": "us-central1",
    "DEFAULT_VERTEX_OPENAI_COMPAT_VERSION": "v1beta1",

    # Anti-throttle for market-data REST requests
    "DOWNLOAD_DELAY_SECONDS": 0.15,

    # PATHS (locked)
    "DATA_DIR": os.path.join(BASE_DIR, "data"),
    "LOG_DIR": os.path.join(BASE_DIR, "log"),
    "SENATE_TRANSCRIPTS_DIR": os.path.join(BASE_DIR, "log", "senate_transcripts"),
    "SCREENER_RUNS_DIR": os.path.join(BASE_DIR, "log", "screener_runs"),
    "SENATE_EVENTS_DIR": os.path.join(BASE_DIR, "log", "senate"),
    "SCREENER_EVENTS_DIR": os.path.join(BASE_DIR, "log", "screener"),

    # Files
    "MAIN_LOG_FILE": os.path.join(BASE_DIR, "log", "main.log"),
    "VERSION_LOG_FILE": os.path.join(BASE_DIR, "log", f"main_v{VERSION.replace('.', '_')}.log"),
    "SCREENER_LATEST_FILE": os.path.join(BASE_DIR, "log", "screener_details.json"),

    # JSONL telemetry
    "SENATE_EVENTS_FILE": os.path.join(BASE_DIR, "log", "senate", "senate_events.jsonl"),
    "SCREENER_EVENTS_FILE": os.path.join(BASE_DIR, "log", "screener", "screener_events.jsonl"),
}

TZ = pytz.timezone(CONFIG["TIMEZONE"])

# Safety: futures leverage must be 1
if CONFIG["TRADING_MODE"] == "futures":
    try:
        lev = float(CONFIG.get("FUTURES_LEVERAGE", 1.0))
        if abs(lev - 1.0) > 1e-9:
            raise RuntimeError("FUTURES_LEVERAGE must be 1.0 (your rule).")
    except Exception:
        raise RuntimeError("Invalid FUTURES_LEVERAGE. Set FUTURES_LEVERAGE=1.0")

# Ensure dirs exist
os.makedirs(CONFIG["DATA_DIR"], exist_ok=True)
os.makedirs(CONFIG["LOG_DIR"], exist_ok=True)
os.makedirs(CONFIG["SENATE_TRANSCRIPTS_DIR"], exist_ok=True)
os.makedirs(CONFIG["SCREENER_RUNS_DIR"], exist_ok=True)
os.makedirs(CONFIG["SENATE_EVENTS_DIR"], exist_ok=True)
os.makedirs(CONFIG["SCREENER_EVENTS_DIR"], exist_ok=True)
os.makedirs(CONFIG["FEATURE_ISOLATION_OUTPUT_DIR"], exist_ok=True)

MASTER_ANALYSIS_PROMPT = (
    "You are an elite, multi-disciplinary financial analyst. Provide a comprehensive, 360-degree analysis "
    "(Technical, Psychological, Risk). Conclude with a balanced recommendation.\n"
    "Return ONLY pure JSON (no markdown): "
    '{"vote":"BUY"|"SELL"|"HOLD","confidence":0,"reason":"Your comprehensive summary."}\n'
    "IMPORTANT: confidence must be an integer 0..100."
)

MASTER_DEBATE_PROMPT_TEMPLATE = """You are an expert financial analyst in a deliberative council.
Your Initial Analysis was: {my_initial_analysis}
Your Colleagues' Analyses are: {colleagues_analyses}
Your Task: Critically evaluate all perspectives, re-evaluate your position, and cast your final, binding vote.
Return ONLY pure JSON (no markdown): {{"final_vote":"BUY"|"SELL"|"HOLD","final_confidence":0,"changed_opinion":true|false,"reason_for_final_decision":"Your conclusive reasoning."}}
IMPORTANT: final_confidence must be an integer 0..100.
"""


# -------------------------
# 🎭 SENATORS
# -------------------------

SENATORS: Dict[str, Dict[str, Any]] = {
    # "Gemini-2.5-Pro": {"provider": "vertex_ai", "type": "google_native", "id": "gemini-2.5-pro", "location": "us-central1"},  # Temporarily disabled: OpenAI-only mode.
    # "Mistral-Medium-3": {"provider": "vertex_ai", "type": "mistral_rawpredict", "id": "mistral-medium-3", "location": "us-central1"},  # Temporarily disabled at user's request.
    # "Llama-3.3-70B": {"provider": "vertex_ai", "type": "vertex_openai_compat", "id": "meta/llama-3.3-70b-instruct-maas", "location": "us-central1"},  # Temporarily disabled at user's request.
    # "Llama-4-Maverick": {"provider": "vertex_ai", "type": "vertex_openai_compat", "id": "meta/llama-4-maverick-17b-128e-instruct-maas", "location": "us-east5"},  # Temporarily disabled at user's request.
    # "DeepSeek-V3.2": {"provider": "vertex_ai", "type": "vertex_openai_compat", "id": "deepseek-ai/deepseek-v3.2-maas", "location": "global"},  # Temporarily disabled at user's request.

    "GPT-5.4": {"provider": "openai", "type": "openai_direct", "id": "gpt-5.4"},
    # "Grok-4": {"provider": "xai", "type": "xai_direct", "id": "grok-4-0709"},  # Temporarily disabled: OpenAI-only mode.
    # "Perplexity-Sonar": {"provider": "perplexity", "type": "perplexity_direct", "id": "sonar-pro"},  # Temporarily disabled at user's request.

    # "Cohere-Command-A-Reasoning": {"provider": "cohere", "type": "cohere_v2", "id": "command-a-reasoning"},  # Temporarily disabled at user's request.

    # "Qwen-3-Max": {"provider": "openrouter", "type": "openrouter_direct", "id": "qwen/qwen3-max"},  # Temporarily disabled at user's request.
    # "Amazon-Nova-Premier": {"provider": "openrouter", "type": "openrouter_direct", "id": "amazon/nova-premier-v1"},  # Temporarily disabled at user's request.
    # "Nous-Hermes-4-405B": {"provider": "openrouter", "type": "openrouter_direct", "id": "nousresearch/hermes-4-405b"},  # Temporarily disabled at user's request.
}
