import torch
import gc
from contextlib import contextmanager
from FlagEmbedding import FlagLLMReranker

from core.config import RERANKER_MODEL, USE_RERANKER, RERANKER_TOP_K
from core.utils.singletons import lazy_singleton

@contextmanager
def gpu_memory_manager():
    """Контекстный менеджер для управления GPU памятью"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()



@lazy_singleton
def get_reranker():
    if not USE_RERANKER:
        return None
    
    with gpu_memory_manager():
        # Пытаемся использовать GPU 1 для реранкера, чтобы избежать конфликтов с основными моделями
        device = 'cpu'  # По умолчанию CPU
        gpu_device_id = None
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"🔍 Доступно GPU: {gpu_count}")
            
            if gpu_count > 1:
                # Если есть несколько GPU, используем вторую для реранкера
                device = 'cuda:1'
                gpu_device_id = 1
                print(f"🚀 Реранкер загружается на GPU 1")
            else:
                # Если только одна GPU, используем её
                device = 'cuda:0'
                gpu_device_id = 0
                print(f"⚠️ Только одна GPU доступна, реранкер загружается на GPU 0")
        
        use_fp16 = torch.cuda.is_available()
        
        # Устанавливаем максимальную память для модели на выбранной GPU
        if gpu_device_id is not None:
            torch.cuda.set_per_process_memory_fraction(0.7, device=gpu_device_id)
        
        reranker = FlagLLMReranker(
            RERANKER_MODEL, 
            use_fp16=use_fp16,
            device=device
        )
        
        print(f"✅ Реранкер успешно загружен на {device}")
        return reranker

def rerank_documents(query, docs, reranker=None, top_k=None):
    if not docs:
        return docs
    
    if reranker is None:
        reranker = get_reranker()
        
    if reranker is None:
        return docs[:top_k] if top_k else docs
    
    if top_k is None:
        top_k = RERANKER_TOP_K
    
    with gpu_memory_manager():
        print(f"🔄 Начинаем реранжирование {len(docs)} документов...")
        
        # Ограничиваем длину контента документов для стабильности
        max_content_length = 512  # Безопасная длина для реранкера
        pairs = []
        for i, doc in enumerate(docs):
            content = doc.page_content[:max_content_length]
            pairs.append([query, content])
            if i < 3:  # Логируем первые 3 документа
                print(f"  Документ {i+1}: {len(content)} символов")
        
        print(f"🚀 Отправляем {len(pairs)} пар на реранжирование...")
        scores = reranker.compute_score(pairs)
        print(f"✅ Получены скоры для {len(scores)} документов")
        
        scored_docs = list(zip(docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        reranked_docs = [doc for doc, _ in scored_docs[:top_k]]
        print(f"📊 Реранжирование завершено: отобрано {len(reranked_docs)} из {len(docs)} документов")
        return reranked_docs

def cleanup_reranker():
    """Очищает ресурсы реранкера"""
    get_reranker.reset()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()