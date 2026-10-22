"""The chat model behind the workflow and the agent, chosen by SCANNER_LLM:

  bedrock:<model id>     Amazon Bedrock in the project Region. No key: the IAM role grants access.
  openrouter:<model>     OpenRouter: the key from OPENROUTER_API_KEY or, on AWS, from the Secrets Manager
                         secret named in OPENROUTER_SECRET.
  openai:<model>         Any OpenAI-compatible endpoint (OPENAI_BASE_URL, OPENAI_API_KEY), for self-hosting.
"""
import os
from functools import cache

from langchain_core.messages import SystemMessage

DEFAULT = "openrouter:anthropic/claude-haiku-5.5"


def name():
    return os.environ.get("SCANNER_LLM", DEFAULT)


@cache
def _openrouter_key():
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    import boto3
    sm = boto3.client("secretsmanager", region_name=os.environ.get("AWS_REGION", "eu-north-1"))
    return sm.get_secret_value(SecretId=os.environ["OPENROUTER_SECRET"])["SecretString"].strip()


def chat_model(spec=None, max_tokens=2000):
    provider, _, model = (spec or name()).partition(":")
    if provider == "bedrock":
        from botocore.config import Config
        from langchain_aws import ChatBedrockConverse
        return ChatBedrockConverse(model_id=model, region_name=os.environ.get("AWS_REGION", "eu-north-1"),
                                   temperature=0, max_tokens=max_tokens,
                                   config=Config(retries={"max_attempts": 10, "mode": "adaptive"}))
    from langchain_openai import ChatOpenAI
    if provider == "openrouter":
        return ChatOpenAI(model=model, base_url="https://openrouter.ai/api/v1",
                          api_key=_openrouter_key(), temperature=0, max_tokens=max_tokens)
    if provider == "openai":
        return ChatOpenAI(model=model, temperature=0, max_tokens=max_tokens)
    raise ValueError(f"unknown provider in SCANNER_LLM={spec or name()!r}")


def system(text, spec=None):
    """A system message the provider caches between calls, where caching must be asked for."""
    provider, _, model = (spec or name()).partition(":")
    if provider == "openrouter" and model.startswith("anthropic/"):
        return SystemMessage([{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}])
    if provider == "bedrock" and ("anthropic" in model or "amazon.nova" in model):
        return SystemMessage([{"type": "text", "text": text}, {"cachePoint": {"type": "default"}}])
    return SystemMessage(text)
