"""Pick the chat model for the examples from environment variables.

Azure OpenAI when AZURE_OPENAI_API_KEY is set (AZURE_OPENAI_ENDPOINT,
AZURE_OPENAI_API_VERSION, AZURE_OPENAI_DEPLOYMENT), otherwise OpenAI
(OPENAI_API_KEY, OPENAI_MODEL).
"""

import os

from langchain_core.language_models import BaseChatModel


def model_name() -> str:
    if os.getenv("AZURE_OPENAI_API_KEY"):
        return os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4-nano")
    return os.getenv("OPENAI_MODEL", "gpt-5-mini")


def make_model() -> BaseChatModel:
    if os.getenv("AZURE_OPENAI_API_KEY"):
        from langchain_openai import AzureChatOpenAI

        return AzureChatOpenAI(
            azure_deployment=model_name(),
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model_name())
