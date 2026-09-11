"""
Shared settings. Secrets are read from unit4/.env so nothing is duplicated.
"""

import os

from dotenv import load_dotenv
from poke_env import ServerConfiguration

HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(HERE, "..", "unit4", ".env"))

XAI_API_KEY = os.environ.get("XAI_API_KEY", "")
XAI_BASE_URL = "https://api.x.ai/v1"

# grok-4.3 is the fast, cheap one (about $0.15 per battle). grok-4.6 thinks harder and costs 2x.
DEFAULT_MODEL = os.environ.get("POKE_MODEL", "grok-4.3")
PRICE_PER_M = {"grok-4.3": (1.25, 2.50), "grok-4.6": (2.00, 6.00), "grok-4.5": (2.00, 6.00)}

LOG_DIR = os.path.join(HERE, "logs")
BATTLE_FORMAT = "gen9randombattle"

# The course's public server (the one that is still alive). Only used by challenge_online.py.
COURSE_SERVER = ServerConfiguration(
    "wss://deploy-lingering-grove-7264.fly.dev/showdown/websocket",
    "https://play.pokemonshowdown.com/action.php?",
)
COURSE_WEB_CLIENT = "https://deploy-lingering-grove-7264.fly.dev/"

# Browser client for the LOCAL server: the official test client pointed at localhost.
LOCAL_WEB_CLIENT = "http://localhost:8000  (it redirects to https://localhost.psim.us, the official client pointed at your server)"


def require_xai_key() -> str:
    if not XAI_API_KEY:
        raise SystemExit("XAI_API_KEY missing in unit4/.env")
    return XAI_API_KEY
