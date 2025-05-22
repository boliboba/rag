from core.config import MODEL_NAME
from core.llm.models import ChatOpenRouter
from core.utils.singletons import lazy_singleton


@lazy_singleton
def get_llm(model_name=None, temperature=0.0, max_tokens=4000):
    """LLM для использования в проекте"""
    if model_name is None:
        model_name = MODEL_NAME
    
    return ChatOpenRouter(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )
