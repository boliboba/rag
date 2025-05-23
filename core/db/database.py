import torch
import os
from tqdm import tqdm

# Добавляем импорты для работы с TPU
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    HAS_TPU = True
except ImportError:
    HAS_TPU = False

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from core.config import INDEX_NAME, DATA_PATH, SEARCH_TOP_K, USE_RERANKER, RERANKER_TOP_K, EMBEDDING_MODEL
from core.utils.singletons import lazy_singleton
from core.modules.ranking import get_reranker, rerank_documents

# Функция для определения доступного устройства
def get_device():
    if HAS_TPU:
        return 'xla'  # Специальное значение для HuggingFaceEmbeddings
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'

# Функции для работы с векторной базой данных
@lazy_singleton
def get_vector_store(index_path=None):
    if index_path is None:
        index_path = os.path.join(DATA_PATH, f"{INDEX_NAME}_faiss")
    
    if os.path.exists(index_path):
        return load_vectorstore(index_path)
    else:
        return FAISS.from_documents([], embedding=get_embedding_model())

@lazy_singleton
def get_embedding_model(model_name=EMBEDDING_MODEL):
    device = get_device()
    
    # Выводим информацию об используемом устройстве
    device_type = "TPU" if device == 'xla' else device.upper()
    print(f"🔧 Инициализация модели эмбеддингов на {device_type}")
    
    return HuggingFaceEmbeddings(
        model_name=model_name, 
        model_kwargs={'device': device, 
                      'trust_remote_code': True}
    )

def create_vectorstore(documents, embedding_model=None):
    if embedding_model is None:
        embedding_model = get_embedding_model()
    
    if len(documents) <= 1:
        return FAISS.from_documents(documents, embedding=embedding_model)
    
    db = None
    with tqdm(total=len(documents), desc="Индексация документов") as pbar:
        for d in documents:
            if db:
                db.add_documents([d])
            else:
                db = FAISS.from_documents([d], embedding=embedding_model)
            pbar.update(1)
            
    return db

def create_and_save_vectorstore(documents, output_path=None, model_name=EMBEDDING_MODEL, device=None):
    # Если устройство не указано, определяем его автоматически
    if device is None:
        device = get_device()
        
    embedding_model = get_embedding_model(model_name)
    
    if len(documents) <= 1:
        db = FAISS.from_documents(documents, embedding=embedding_model)
    else:      
        db = FAISS.from_documents([documents[0]], embedding=embedding_model)
        with tqdm(total=len(documents)-1, desc="Индексация документов") as pbar:
            for doc in documents[1:]:
                db.add_documents([doc])
                pbar.update(1)
    
    if output_path is not None:
        save_vectorstore(db, output_path)
    
    return db

def save_vectorstore(vectorstore, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vectorstore.save_local(path)
    
def load_vectorstore(path, embedding_model=None):
    if embedding_model is None:
        embedding_model = get_embedding_model()
    
    return FAISS.load_local(path, embedding_model, allow_dangerous_deserialization=True)

def get_top_k(query, vectorstore, k=None):
    if k is None:
        k = SEARCH_TOP_K
    return vectorstore.similarity_search(query=query, k=k)

def stop():
    """Сбрасывает все синглтоны для освобождения ресурсов"""
    get_vector_store.reset()
    get_embedding_model.reset()
    
    # Очистка памяти в зависимости от устройства
    if HAS_TPU:
        xm.mark_step()
        print("🧹 Очистка памяти TPU выполнена")
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("🧹 Очистка памяти GPU выполнена")
        
    import gc
    gc.collect()

# Функции для извлечения и ранжирования
def retrieve(query, query_transformer=None, top_k=None):
    transformed_query = query_transformer(query) if query_transformer else query
    
    vectorstore = get_vector_store()
    
    if top_k is None:
        top_k = SEARCH_TOP_K
    
    return get_top_k(transformed_query, vectorstore, k=top_k)

def rerank(query, docs, top_k=None):
    reranker = get_reranker()
    if reranker:
        return_k = RERANKER_TOP_K if top_k is None else top_k
        return rerank_documents(query, docs, reranker, top_k=return_k)
    return docs 