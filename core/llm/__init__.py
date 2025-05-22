import torch
from core.config import MODEL_NAME
from core.llm.models import ChatOpenRouter

# Кэш для LLM моделей с ключом по model_name + temperature + max_tokens
_llm_cache = {}

def get_llm(model_name=None, temperature=0.0, max_tokens=4000):
    """LLM для использования в проекте с кэшированием по модели и параметрам"""
    if model_name is None:
        model_name = MODEL_NAME
    
    # Создаем уникальный ключ для кэша
    cache_key = f"{model_name}_{temperature}_{max_tokens}"
    
    if cache_key not in _llm_cache:
        print(f"Инициализация LLM модели: {model_name}")
        try:
            _llm_cache[cache_key] = ChatOpenRouter(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
            print(f"LLM модель {model_name} успешно инициализирована")
        except Exception as e:
            print(f"Ошибка при инициализации LLM {model_name}: {e}")
            # Очистка GPU памяти при ошибке
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
    
    return _llm_cache[cache_key]

def reset_llm_cache():
    """Очищает кэш LLM моделей и GPU память"""
    global _llm_cache
    _llm_cache.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("LLM кэш очищен")
