"""Azure OpenAI client built from ~/.secrets, per the user's instruction to use
Azure OpenAI rather than standard OpenAI.

Endpoint and key come from ~/.secrets (azure_openai_endpoint / azure_openai_api_key,
falling back to the AZURE_OPENAI_* env vars). The deployment name and api-version
are NOT stored in secrets, so they are read from env with placeholders that must
be set before any call:

  export AZURE_CHAT_DEPLOYMENT=<your gpt-5.4-mini deployment name>
  export AZURE_API_VERSION=2024-10-21
"""
from __future__ import annotations

import os
from pathlib import Path

from openai import AzureOpenAI

SECRETS = Path.home() / ".secrets"


def _read(name: str, env: str) -> str:
    f = SECRETS / name
    if f.exists():
        return f.read_text().strip()
    return os.environ.get(env, "").strip()


AZURE_ENDPOINT = _read("azure_openai_endpoint", "AZURE_OPENAI_ENDPOINT")
AZURE_KEY = _read("azure_openai_api_key", "AZURE_OPENAI_API_KEY")
API_VERSION = os.environ.get("AZURE_API_VERSION", "2024-10-21")

# Deployment names (Azure calls models by deployment, not model id). Verified on
# kaf-internal-dev: the deployment is named the same as the model.
CHAT_DEPLOYMENT = os.environ.get("AZURE_CHAT_DEPLOYMENT", "gpt-5.4-mini")
JUDGE_DEPLOYMENT = os.environ.get("AZURE_JUDGE_DEPLOYMENT", CHAT_DEPLOYMENT)

# gpt-5.x deployments reject max_tokens; they require max_completion_tokens.
MAX_COMPLETION_TOKENS = int(os.environ.get("AZURE_MAX_COMPLETION_TOKENS", "600"))


def client() -> AzureOpenAI:
    if not AZURE_ENDPOINT or not AZURE_KEY:
        raise SystemExit("missing Azure endpoint/key in ~/.secrets")
    return AzureOpenAI(azure_endpoint=AZURE_ENDPOINT, api_key=AZURE_KEY,
                       api_version=API_VERSION)


def require_deployment(name: str) -> str:
    if not name:
        raise SystemExit(
            "set AZURE_CHAT_DEPLOYMENT (and optionally AZURE_JUDGE_DEPLOYMENT) "
            "to your Azure deployment name(s) before running")
    return name
