"""Paths, model names and API keys. Keys come from the environment or a .env file."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
INDEX_DIR = DATA / "index"

CORPUS = DATA / "corpus.jsonl"
FORMULARY = DATA / "formulary.jsonl"
PLAN_COSTS = DATA / "plan_costs.json"

# Models (change here to try others)
REWRITE_MODEL = "claude-haiku-5-5"     # fast: language detection + English search queries
ANSWER_MODEL = "claude-sonnet-5-5"     # writes the cited answer, calls the tools
JUDGE_MODEL = "claude-sonnet-5-5"      # eval: grades answers against the test set
EMBED_MODEL = "voyage-3.5"             # multilingual embeddings

TOP_K = 8                # chunks given to the answer model
RRF_K = 60               # reciprocal rank fusion constant


def _load_dotenv():
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

# When to call Voyage for vector search: "auto" (only non-English questions), "always", or "off"
VECTOR_MODE = os.environ.get("VECTOR_MODE", "auto").lower()


def key(name: str) -> str:
    v = os.environ.get(name, "")
    if not v:
        raise RuntimeError(f"{name} is not set. Copy .env.example to .env and add it.")
    return v
