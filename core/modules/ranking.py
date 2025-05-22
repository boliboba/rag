import torch
import gc
from FlagEmbedding import FlagLLMReranker

from core.config import RERANKER_MODEL, USE_RERANKER, RERANKER_TOP_K
from core.utils.singletons import lazy_singleton

@lazy_singleton
def get_reranker():
    if not USE_RERANKER:
        return None
    
    try:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        use_fp16 = torch.cuda.is_available()
        
        reranker = FlagLLMReranker(
            RERANKER_MODEL, 
            use_fp16=use_fp16,
            device=device
        )
        
        return reranker
            
    except Exception as e:
        print(f"⚠️ Ошибка загрузки реранкера: {e}")
        return None

def rerank_documents(query, docs, reranker=None, top_k=None):
    if not docs:
        return docs
    
    if reranker is None:
        reranker = get_reranker()
        
    if reranker is None:
        return docs[:top_k] if top_k else docs
    
    if top_k is None:
        top_k = RERANKER_TOP_K
    
    try:
        # Простой и быстрый реранкинг для ~100 документов
        pairs = [[query, doc.page_content[:400]] for doc in docs]
        scores = reranker.compute_score(pairs)
        
        scored_docs = list(zip(docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in scored_docs[:top_k]]
        
    except Exception as e:
        print(f"⚠️ Ошибка реранжирования: {e}")
        return docs[:top_k] if top_k else docs

def cleanup_reranker():
    """Очищает ресурсы реранкера"""
    get_reranker.reset()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()