import torch
from FlagEmbedding import FlagLLMReranker

from core.config import RERANKER_MODEL, USE_RERANKER, RERANKER_TOP_K
from core.utils.singletons import lazy_singleton

@lazy_singleton
def get_reranker():
    if not USE_RERANKER:
        return None
    
    try:
        use_fp16 = torch.cuda.is_available()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        return FlagLLMReranker(
            RERANKER_MODEL, 
            use_fp16=use_fp16,
            device=device
        )
    except Exception:
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
    
    pairs = [[query, doc.page_content[:400]] for doc in docs] # TODO: убрать ограничение на длину документа
    
    scores = reranker.compute_score(pairs)
        
    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    reranked_docs = [doc for doc, _ in scored_docs[:top_k]]
    
    return reranked_docs