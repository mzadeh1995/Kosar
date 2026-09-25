# ==============================================================================
# 📡 telemetry_utils.py
# ------------------------------------------------------------------------------
# Shared telemetry, logging, JSON helpers, API-key helpers, and feature-contract guards.
# v49.1: Added "Calibration" and Patch
# telemetry contracts/guards preserved.
# ==============================================================================

from __future__ import annotations

import os
import sys
import json
import logging
import re
from typing import Any, Dict, Optional, Tuple, List
from datetime import datetime, timedelta

from google.auth import default, exceptions as google_auth_exceptions
from google.auth.transport.requests import Request

from config import CONFIG, TZ, SCHEMA_VERSION, SENATORS

# -------------------------
NON_NEGOTIABLE_TELEMETRY_FLAGS = [
    "TELEMETRY_SENATE_EVENTS",
    "TELEMETRY_SENATE_TRANSCRIPTS",
    "TELEMETRY_SCREENER_DETAILS",
    "TELEMETRY_SCREENER_EVENTS",
    "TELEMETRY_EQUITY_CURVE",
]

REQUIRED_TRANSCRIPT_DOTPATHS = [
    "schema_version",
    "cycle_id",
    "symbol",
    "phase1.sys_prompt",
    "phase1.user_prompt",
    "phase1.responses_parsed",
    "phase1.responses_raw",
    "phase2.debate_payloads",
    "phase2.sys_prompt_per_senator",
    "phase2.user_prompt",
    "phase2.responses_parsed",
    "phase2.responses_raw",
    "decision",
    "consensus.buy",
    "consensus.sell",
]

REQUIRED_SENATE_EVENT_KEYS = [
    "schema_version", "ts", "cycle_id", "symbol", "phase", "debate_round",
    "senator", "provider", "model_type", "model_id", "location",
    "attempt", "ok", "fail_stage", "error", "http_status", "response_preview", "parsed"
]


def _truthy_env(name: str) -> bool:
    v = os.environ.get(name, "")
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def enforce_non_negotiables_or_exit() -> None:
    """
    Prevent running with telemetry silently disabled.
    """
    allow_disable = _truthy_env("ALLOW_TELEMETRY_DISABLE")
    disabled = [k for k in NON_NEGOTIABLE_TELEMETRY_FLAGS if not bool(CONFIG.get(k, True))]
    if disabled and not allow_disable:
        msg = (
            "FATAL: Telemetry feature flags are disabled without explicit permission.\n"
            f"Disabled: {', '.join(disabled)}\n"
            "To intentionally disable telemetry, set env: ALLOW_TELEMETRY_DISABLE=1"
        )
        logger.critical(msg)
        sys.exit(1)


def _get_dotpath(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def assert_feature_contract_transcript(transcript: dict) -> None:
    missing = [p for p in REQUIRED_TRANSCRIPT_DOTPATHS if _get_dotpath(transcript, p) is None]
    if missing:
        raise RuntimeError(f"Feature Contract violated: transcript missing keys: {missing}")


def assert_feature_contract_event(ev: dict) -> None:
    missing = [k for k in REQUIRED_SENATE_EVENT_KEYS if k not in ev]
    if missing:
        raise RuntimeError(f"Feature Contract violated: senate_event missing keys: {missing}")


# -------------------------
# 🧾 LOGGING (to ./log)
# -------------------------
logger = logging.getLogger("cognitive_bot")
logger.setLevel(logging.INFO)
logger.handlers.clear()

_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

file_main = logging.FileHandler(CONFIG["MAIN_LOG_FILE"], encoding="utf-8")
file_main.setFormatter(_fmt)
file_main.setLevel(logging.INFO)

file_ver = logging.FileHandler(CONFIG["VERSION_LOG_FILE"], encoding="utf-8")
file_ver.setFormatter(_fmt)
file_ver.setLevel(logging.INFO)

console = logging.StreamHandler()
console.setFormatter(_fmt)
console.setLevel(logging.INFO)

logger.addHandler(file_main)
logger.addHandler(file_ver)
logger.addHandler(console)

# Optional: reduce 3rd-party noise
for noisy in ["urllib3", "httpx", "openai"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)


# -------------------------
# 🧾 JSON helpers
# -------------------------
def now_ts() -> str:
    return datetime.now(TZ).strftime("%Y%m%d_%H%M%S")

def safe_symbol(sym: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in sym)

def write_json(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def append_json_array(path: str, item: Any, max_len: Optional[int] = None) -> None:
    data = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                data = []
        except Exception:
            data = []
    data.append(item)
    if max_len and len(data) > max_len:
        data = data[-max_len:]
    write_json(path, data)

def append_jsonl(path: str, item: dict) -> None:
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error(f"❌ Failed writing jsonl to {path}: {e}")

def log_stats(title: str, stats: dict):
    logger.info(f"📊 {title} | {json.dumps(stats, ensure_ascii=False)}")



# -------------------------
# 🔐 API key helpers
# -------------------------
def get_api_key(key_name: str, env_vars: list) -> str:
    for var in env_vars:
        val = os.environ.get(var)
        if val and val.strip():
            return val.strip()
    raise RuntimeError(f"{key_name} not found in .env. Add one of: {', '.join(env_vars)}")

def check_api_keys() -> None:
    logger.info("Performing mandatory API key check...")

    required_providers = {s["provider"] for s in SENATORS.values()}

    missing: List[str] = []
    key_map = {
        "openai": ["OPENAI_API_KEY"],
        "xai": ["XAI_API_KEY", "GROK_API_KEY"],
        "perplexity": ["PERPLEXITY_API_KEY", "PPLX_API_KEY"],
        "cohere": ["COHERE_API_KEY"],
        "openrouter": ["OPENROUTER_API_KEY"],
    }

    # Vertex credentials
    if "vertex_ai" in required_providers:
        try:
            creds, _ = default()
            if not creds:
                raise google_auth_exceptions.DefaultCredentialsError()
            logger.info("✅ Google Cloud credentials found successfully.")
        except google_auth_exceptions.DefaultCredentialsError:
            missing.append("VERTEX_AI (GCP Credentials)")

    for provider, env_vars in key_map.items():
        if provider in required_providers:
            try:
                _ = get_api_key(provider.upper(), env_vars)
                logger.info(f"✅ {provider.upper()} API key found.")
            except RuntimeError:
                missing.append(provider.upper())

    if missing:
        logger.critical(f"FATAL ERROR: Missing required API keys/credentials for: {', '.join(missing)}.")
        sys.exit(1)

    logger.info("✅ All required API keys/credentials are present.")

def get_gcp_auth_token() -> str:
    creds, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(Request())
    return creds.token

def build_vertex_openapi_base_url(project_id: str, location: str, api_version: str) -> str:
    host = f"{location}-aiplatform.googleapis.com" if location != "global" else "aiplatform.googleapis.com"
    return f"https://{host}/{api_version}/projects/{project_id}/locations/{location}/endpoints/openapi"

# -------------------------
# 🧠 Prompt helpers + JSON extraction
# -------------------------
def wrap_prompt(sys_prompt: str, user_content: str) -> str:
    return f"{sys_prompt}\n\n--- MARKET DATA ---\n{user_content}"

_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)

def _strip_think_blocks(text: str) -> str:
    if not text:
        return ""
    return re.sub(_THINK_RE, "", text).strip()

def _extract_codeblock_json(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None

def extract_clean_json(text: str) -> Optional[dict]:
    """
    Robust JSON extraction:
    - strips <think> blocks
    - extracts ```json {..} ``` if present
    - otherwise finds first JSON object in text
    """
    if not text:
        return None

    text = text.strip()
    text = _strip_think_blocks(text)

    cb = _extract_codeblock_json(text)
    if cb:
        try:
            obj = json.loads(cb)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

    # direct load
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # scan for first dict
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            obj, _end = decoder.raw_decode(text[start:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            start = text.find("{", start + 1)
            continue
        start = text.find("{", start + 1)
    return None


# -------------------------
# ✅ Confidence normalization: CANONICAL = 0..100 (integer)
# -------------------------
def normalize_confidence_pct(x: Any) -> Optional[int]:
    """
    Accepts:
      - 0..1 floats (e.g., 0.8)   -> 80
      - 1..100 numbers (e.g., 75) -> 75
    Clamps to 0..100.
    """
    try:
        v = float(x)
    except Exception:
        return None

    if v < 0:
        v = 0.0
    if 0.0 <= v <= 1.0:
        v = v * 100.0
    if v > 100.0:
        v = 100.0
    return int(round(v))


def validate_analysis_json(d: dict, debate: bool) -> Optional[dict]:
    if not isinstance(d, dict):
        return None

    if debate:
        if d.get("final_vote") in ["BUY", "SELL", "HOLD"]:
            c = normalize_confidence_pct(d.get("final_confidence"))
            if c is None:
                return None
            d["final_confidence"] = c
            d["changed_opinion"] = bool(d.get("changed_opinion", False))
            return d
        return None

    if d.get("vote") in ["BUY", "SELL", "HOLD"]:
        c = normalize_confidence_pct(d.get("confidence"))
        if c is None:
            return None
        d["confidence"] = c
        return d

    return None


# -------------------------
# 📌 PROMPTS (confidence is 0..100)
