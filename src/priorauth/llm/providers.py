"""Where Claude is served from: the Claude API or Amazon Bedrock. Nothing else changes.

Bedrock serves Claude through an Anthropic-compatible endpoint, so the same SDK client, pointed at
Bedrock's URL with a Bedrock API key, sends the same requests. Requests keep the model's own name
(claude-sonnet-5-5), so recordings, fingerprints and prices do not depend on the provider; the Bedrock
client swaps in Bedrock's model ID only when it calls.
"""

import os
from enum import StrEnum

from priorauth.llm.base import ModelClient


class Provider(StrEnum):
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"


# Global cross-region inference: Bedrock routes to any region with capacity, with no price premium.
BEDROCK_MODEL_IDS = {
    "claude-sonnet-5-5": "global.anthropic.claude-sonnet-5-5",
    "claude-opus-5-5": "global.anthropic.claude-opus-5-5",
}
BEDROCK_REGION = "us-east-1"


def bedrock_url(region: str) -> str:
    return f"https://bedrock-runtime.{region}.amazonaws.com/anthropic"


def bedrock_token(region: str) -> str:
    """A Bedrock API key from AWS_BEARER_TOKEN_BEDROCK, or a short-lived one made from your AWS credentials."""
    if token := os.environ.get("AWS_BEARER_TOKEN_BEDROCK", "").strip():
        return token
    try:
        from aws_bedrock_token_generator import provide_token
    except ImportError as e:
        raise RuntimeError(
            "Bedrock needs AWS_BEARER_TOKEN_BEDROCK, or the bedrock extra to make a short-lived token "
            "from your AWS credentials"
        ) from e
    return provide_token(region=region)


def make_client(provider: Provider, region: str | None = None) -> ModelClient:
    """A live model client for the provider. Needs the agent extra, and a key for that provider."""
    from priorauth.llm.claude import ClaudeClient, anthropic_sdk  # the only module that imports the SDK

    if provider is Provider.ANTHROPIC:
        return ClaudeClient()
    region = region or os.environ.get("AWS_REGION") or BEDROCK_REGION
    sdk = anthropic_sdk(base_url=bedrock_url(region), api_key=bedrock_token(region))
    return ClaudeClient(sdk, model_ids=BEDROCK_MODEL_IDS)
