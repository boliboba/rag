import streamlit as st

# Импортируем компоненты из нашего проекта
from core.llm import get_llm
from core.config import OPENROUTER_API_KEY, MODEL_NAME, PROMPTS
from core.db import get_vector_store, retrieve, rerank
from core.llm.chains import get_qa_chain, format_docs
from core.modules.ranking import get_reranker

# Константы для моделей
DEFAULT_MODEL = MODEL_NAME
# Восстанавливаем список моделей
AVAILABLE_MODELS_LIST = [
    "qwen/qwen3-235b-a22b", 
    "qwen/qwen3-32b", 
    "qwen/qwen3-14b",
    "google/gemini-2.5-flash-preview",
    "google/gemini-2.0-flash-001",
    "google/gemma-3-27b-it",
    "google/gemma-3-12b-it",
    "meta-llama/llama-4-maverick",
    "meta-llama/llama-3.1-405b-instruct",
    "meta-llama/llama-3.3-70b-instruct"
]

def main():
    st.set_page_config(page_title="RAG Demo", page_icon="🤖", layout="wide")
    
    st.title("🔍 RAG Демонстрация")
    st.markdown("""
    Это демонстрация работы пайплайна Retrieval Augmented Generation (RAG) с использованием:
    - **FAISS** векторного хранилища для эффективного поиска
    - **Reranker** для улучшения релевантности найденных документов
    - **OpenRouter** для доступа к различным языковым моделям
    """)
    
    if not OPENROUTER_API_KEY:
        st.error("OpenRouter API Key не найден! Проверьте переменную OPENROUTER_API_KEY в .env файле.")
        return
    
    with st.sidebar:
        st.header("Настройки")
        
        # Восстанавливаем st.selectbox для выбора модели
        selected_model = st.selectbox("Выберите модель", AVAILABLE_MODELS_LIST, 
                                      index=AVAILABLE_MODELS_LIST.index(DEFAULT_MODEL) if DEFAULT_MODEL in AVAILABLE_MODELS_LIST else 0)
        temperature = st.slider("Температура", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
        
        with st.expander("Настройки поиска и переранжирования"):
            use_reranker = st.checkbox("Использовать reranker", value=True)
            k_retriever = st.slider("Количество начальных результатов", min_value=10, max_value=200, value=100, step=10)
            k_reranker = st.slider("Количество результатов после переранжирования", min_value=1, max_value=20, value=5, step=1)
        
        try:
            vector_store = get_vector_store()
            st.success("Векторное хранилище успешно загружено!")
            if hasattr(vector_store, 'index') and hasattr(vector_store.index, 'ntotal'):
                st.write(f"Количество векторов: {vector_store.index.ntotal}")
        except Exception as e:
            st.error(f"Ошибка при загрузке векторного хранилища: {str(e)}")
        
        if use_reranker:
            try:
                reranker = get_reranker()
                if reranker:
                    st.success("Reranker успешно загружен!")
                else:
                    st.warning("Reranker не настроен. Будет использоваться только начальный поиск.")
            except Exception as e:
                st.error(f"Ошибка при загрузке reranker: {str(e)}")
    
    query = st.text_input("Введите ваш запрос", placeholder="Например: Какие документы есть в базе?")
    
    col1, col2 = st.columns([2, 3])
    
    if query:
        final_docs_for_display_and_context = []
        # --- Шаг 1: Однократный Retrieve и Rerank ---
        with st.spinner("Поиск релевантных документов..."):
            try:
                # Используем параметры из UI (k_retriever)
                retrieved_docs = retrieve(query, top_k=k_retriever)
                st.info(f"Найдено {len(retrieved_docs)} документов в начальном поиске.")
                
                final_docs_for_display_and_context = retrieved_docs
                if use_reranker:
                    # Используем параметры из UI (k_reranker)
                    final_docs_for_display_and_context = rerank(query, retrieved_docs, top_k=k_reranker)
                    st.success(f"Отобрано {len(final_docs_for_display_and_context)} наиболее релевантных документов после переранжирования.")
                
            except Exception as e:
                st.error(f"Ошибка при поиске документов: {str(e)}")
                final_docs_for_display_and_context = [] # Очищаем, если ошибка

        # --- Шаг 2: Отображение документов в col1 ---
        with col1:
            st.subheader("Найденные документы")
            if not final_docs_for_display_and_context and query: # Если после попытки поиска документов нет
                 st.warning("Документы не найдены или произошла ошибка при поиске.")
            elif final_docs_for_display_and_context:
                for i, doc in enumerate(final_docs_for_display_and_context):
                    with st.expander(f"Документ {i+1}"):
                        st.write("**Метаданные:**")
                        for key, value in doc.metadata.items():
                            st.write(f"- **{key}:** {value}")
                        st.write("**Содержимое:**")
                        st.write(doc.page_content)
        
        # --- Шаг 3: Генерация ответа в col2 ---
        with col2:
            st.subheader("Ответ модели")
            if final_docs_for_display_and_context: # Генерируем ответ только если есть документы
                with st.spinner(f"Генерация ответа с использованием {selected_model}..."): # Используем selected_model
                    try:
                        # Настраиваем LLM (синглтон обновится)
                        # Используем selected_model и температуру из слайдера
                        get_llm(model_name=selected_model, temperature=temperature)
                        
                        # Получаем цепочку qa_chain
                        qa_chain = get_qa_chain()
                        
                        # Форматируем контекст из ОДИНАКОВЫХ документов
                        formatted_context = format_docs(final_docs_for_display_and_context)
                        
                        response = qa_chain.invoke({
                            "context": formatted_context,
                            "question": query
                        })
                        st.markdown(response)
                        
                    except Exception as e:
                        st.error(f"Ошибка при генерации ответа: {str(e)}")
            elif query: # Если был запрос, но нет документов для ответа
                st.warning("Нет документов для генерации ответа.")

if __name__ == "__main__":
    main() 