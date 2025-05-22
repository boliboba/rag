import os
from typing import Optional

from langchain_community.chat_models import ChatOpenAI

from core.config import OPENROUTER_API_KEY, OPENROUTER_API_URL


class ChatOpenRouter(ChatOpenAI):
    """
    Адаптер для использования OpenRouter API с интерфейсом ChatOpenAI.
    """
    def __init__(self,
                 model_name: str,
                 openai_api_key: Optional[str] = None,
                 openai_api_base: str = "https://openrouter.ai/api/v1",
                 **kwargs):
        openai_api_key = openai_api_key or os.getenv('OPENROUTER_API_KEY') or OPENROUTER_API_KEY
        openai_api_base = openai_api_base or OPENROUTER_API_URL
        super().__init__(openai_api_base=openai_api_base,
                         openai_api_key=openai_api_key,
                         model_name=model_name, **kwargs)