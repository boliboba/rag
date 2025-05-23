#!/usr/bin/env python
import sys
import os
# Добавление корневой директории проекта в PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import asyncio
import json
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# Добавляем импорты для работы с TPU
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp
    HAS_TPU = True
    print("🚀 TPU доступен и будет использован")
except ImportError:
    HAS_TPU = False
    print("⚠️ TPU не доступен, будет использована CPU/GPU")

from core.llm import get_llm
from core.db import get_vector_store
from core.llm.chains import get_retrieval_chain, format_docs, retrieve, rerank
from core.config import MODEL_NAME as DEFAULT_MODEL_NAME, PROMPTS

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from evals.evaluation import (
    evaluate_dataset,
    evaluate_dataset_async,
    generate_report,
    save_results,
    stop as stop_evaluation
)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluate_models.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Конфигурация для оценки - список моделей
MODELS_TO_EVALUATE = [
    "google/gemini-2.5-flash-preview-05-20",
    "qwen/qwen3-235b-a22b", 
    "qwen/qwen3-32b", 
    "qwen/qwen3-14b",
    "google/gemma-3-27b-it",
    "google/gemma-3-12b-it",
    "meta-llama/llama-3.1-405b-instruct",
    "meta-llama/llama-3.3-70b-instruct",
    "meta-llama/llama-4-maverick",
    "meta-llama/llama-4-scout"
]

# Фиксированная температура
TEMPERATURE = 0.0

# Модель для проведения оценки (независимо от оцениваемой модели)
EVAL_MODEL_NAME = "google/gemini-2.5-flash-preview-05-20"

# Путь к тестовому датасету
TEST_DATASET_PATH = "data/filtered_evaluated_dataset.csv"

# Количество примеров для оценки (None для всего датасета)
LIMIT = 200

# Максимальное количество одновременно оцениваемых моделей
MAX_CONCURRENCY = 6

# Глобальный кэш для предпосчитанных документов
_document_cache = {}

# Функция для определения доступного устройства
def get_device():
    if HAS_TPU:
        return xm.xla_device()
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def precompute_documents_for_all_questions(dataset, limit=None):
    """Предпосчитывает релевантные документы для всех вопросов один раз"""
    if limit is not None and limit < len(dataset):
        dataset = dataset.sample(limit, random_state=42).reset_index(drop=True)
    
    global _document_cache
    _document_cache.clear()
    
    print(f"🔍 Предпосчитываем документы для {len(dataset)} вопросов...")
    
    # Определяем устройство
    device = get_device()
    device_type = "TPU" if HAS_TPU else ("GPU" if torch.cuda.is_available() else "CPU")
    print(f"🔧 Используется {device_type} для вычислений")
    
    for idx, (_, row) in enumerate(tqdm(dataset.iterrows(), total=len(dataset), desc="Предпосчёт документов")):
        question = row["question"]
        
        # Получаем документы из векторной базы
        docs = retrieve(question)
        print(f"Найдено чанков: {len(docs)}")
        
        # Проверяем настройку реранкера
        from core.config import USE_RERANKER
        if USE_RERANKER:
            # ВКЛЮЧАЕМ РЕРАНКЕР для качественного отбора документов
            reranked_docs = rerank(question, docs)
            print(f"Реранжировано чанков: {len(reranked_docs)}")
            final_docs = reranked_docs
        else:
            # Реранкер отключен - берём топ-5 без реранжирования
            final_docs = docs[:5] if len(docs) > 5 else docs
            print(f"Реранкер отключен - взято {len(final_docs)} документов без реранжирования")
        
        # Форматируем контекст
        formatted_context = format_docs(final_docs)
        
        # Кэшируем результат
        _document_cache[question] = formatted_context
    
    print(f"✅ Предпосчёт завершён! Сохранено {len(_document_cache)} контекстов")
    return dataset

def create_model_specific_chain(model_name):
    """Создает специальную цепочку для конкретной модели с использованием предпосчитанных документов"""
    prompt = ChatPromptTemplate.from_template(PROMPTS["qa"])
    
    # Создаем новый экземпляр LLM для этой конкретной модели
    llm = get_llm(model_name=model_name, temperature=TEMPERATURE)
    
    def get_cached_context(query):
        """Получает предпосчитанный контекст из кэша"""
        global _document_cache
        return _document_cache.get(query, "Контекст недоступен")
    
    rag_chain = (
        {"context": get_cached_context, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

async def generate_system_responses_async(dataset, model_name, limit=None, max_concurrent_questions=6):
    """Асинхронно генерирует ответы системы для заданной модели с параллельной обработкой вопросов"""
    if limit is not None and limit < len(dataset):
        # Используем тот же random_state=42 для согласованности с create_test_cases
        dataset = dataset.sample(limit, random_state=42).reset_index(drop=True)
    
    # Создаем специальную цепочку для этой модели
    retrieval_chain = create_model_specific_chain(model_name)
    
    # Семафор для ограничения одновременных запросов к одной модели
    semaphore = asyncio.Semaphore(max_concurrent_questions)
    
    async def process_question(question, golden_answer, question_idx):
        """Обрабатывает один вопрос"""
        async with semaphore:
            print(f"\n[{model_name}] Вопрос {question_idx + 1}: {question}")
            # Добавляем /no_think для моделей Qwen чтобы отключить режим thinking
            modified_question = question
            if "qwen" in model_name.lower():
                modified_question = f"{question} /no_think"
                print(f"[{model_name}] Добавлен флаг /no_think для отключения режима thinking")
            
            result = await retrieval_chain.ainvoke(modified_question)
            answer = result
            print(f"[{model_name}] Ответ {question_idx + 1}: {answer[:100]}..." if len(answer) > 100 else f"[{model_name}] Ответ {question_idx + 1}: {answer}")
            
            return {
                "question": question,
                "system_answer": answer,
                "golden_answer": golden_answer
            }
    
    # Создаем задачи для всех вопросов
    tasks = []
    for idx, (_, row) in enumerate(dataset.iterrows()):
        task = process_question(row["question"], row["answer"], idx)
        tasks.append(task)
    
    # Выполняем все задачи параллельно с прогресс-баром
    print(f"\n🚀 Запускаем параллельную генерацию {len(tasks)} ответов для {model_name} (конкурентность: {max_concurrent_questions})")
    responses = await asyncio.gather(*tasks)
    
    return pd.DataFrame(responses)

async def evaluate_model(model_name, dataset, output_dir, limit=None):
    """Оценивает одну модель"""
    # Имя модели для директории и логов
    model_dir_name = model_name.replace('/', '_')
    model_dir = output_dir / model_dir_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Начинаем оценку модели: {model_name}")
    start_time = time.time()
    
    # Генерируем ответы системы
    system_responses = await generate_system_responses_async(
        dataset=dataset,
        model_name=model_name,
        limit=limit,
        max_concurrent_questions=6  # Параллельная обработка 6 вопросов на модель
    )
    
    # Сохраняем сырые ответы
    system_responses_path = model_dir / "system_responses.csv"
    system_responses.to_csv(system_responses_path, index=False)
    logger.info(f"Ответы системы сохранены в {system_responses_path}")
    
    # Оцениваем результаты
    # Используем асинхронную версию оценки для ускорения
    # Используем EVAL_MODEL_NAME для оценки всех моделей
    logger.info(f"Используем {EVAL_MODEL_NAME} для оценки ответов модели {model_name}")
    evaluation_df = await evaluate_dataset_async(
        dataset=dataset,
        system_responses=system_responses,
        model_name=EVAL_MODEL_NAME,  # Используем одну и ту же модель для оценки
        temperature=0.6,
        limit=limit,
        max_concurrency=6  # Лимит одновременных запросов к LLM для метрик
    )
    
    # Сохраняем сырые данные оценки
    raw_evaluation_path = model_dir / "raw_evaluation.csv"
    evaluation_df.to_csv(raw_evaluation_path, index=False)
    logger.info(f"Сырые данные оценки сохранены в {raw_evaluation_path}")
    
    # Генерируем отчет
    report = generate_report(evaluation_df)
    
    # Сохраняем результаты и отчет
    results_path = model_dir / "evaluation_results.csv"
    save_results(evaluation_df, str(results_path), report)
    
    # Освобождаем ресурсы
    stop_evaluation()
    
    # Очистка памяти в зависимости от используемого устройства
    import gc
    gc.collect()
    
    if HAS_TPU:
        # Очистка TPU памяти
        xm.mark_step()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    elapsed_time = time.time() - start_time
    logger.info(f"Оценка модели {model_name} завершена за {elapsed_time:.2f} секунд")
    
    return {
        "model": model_name,
        "metrics": report,
        "elapsed_time": elapsed_time,
        "examples_count": len(dataset) if limit is None else min(len(dataset), limit)
    }

async def run_evaluations(models, dataset, output_dir, limit, concurrency):
    """Запускает оценку нескольких моделей с ограничением параллелизма"""
    semaphore = asyncio.Semaphore(concurrency)
    
    async def run_with_semaphore(model):
        async with semaphore:
            return await evaluate_model(
                model_name=model,
                dataset=dataset,
                output_dir=output_dir,
                limit=limit
            )
    
    tasks = [run_with_semaphore(model) for model in models]
    return await asyncio.gather(*tasks)

async def main():
    # Создаем директорию для результатов
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("./results") / f"eval_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Определяем тип устройства для логов
    device_type = "TPU" if HAS_TPU else ("GPU" if torch.cuda.is_available() else "CPU")
    
    # Сохраняем параметры запуска
    with open(output_dir / "config.json", "w") as f:
        json.dump({
            "models": MODELS_TO_EVALUATE,
            "eval_model": EVAL_MODEL_NAME,
            "temperature": TEMPERATURE,
            "dataset": TEST_DATASET_PATH,
            "limit": LIMIT,
            "device": device_type,
            "optimization": "precomputed_docs_tpu_optimized" if HAS_TPU else "precomputed_docs_gpu_optimized"
        }, f, indent=2)
    
    # Загружаем датасет
    dataset = pd.read_csv(TEST_DATASET_PATH)
    logger.info(f"Загружен датасет с {len(dataset)} примерами")
    
    # 🚀 КРИТИЧЕСКАЯ ОПТИМИЗАЦИЯ: Предпосчитываем документы для всех вопросов
    logger.info("🔍 Начинаем предпосчёт релевантных документов для всех вопросов...")
    dataset = precompute_documents_for_all_questions(dataset, limit=LIMIT)
    
    logger.info(f"🚀 Будет оценено {len(MODELS_TO_EVALUATE)} моделей с предпосчитанными документами")
    
    # Запускаем оценку всех моделей
    results = await run_evaluations(
        models=MODELS_TO_EVALUATE,
        dataset=dataset,
        output_dir=output_dir,
        limit=LIMIT,
        concurrency=MAX_CONCURRENCY
    )
    
    # Сохраняем сводные результаты
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Создаем сводную таблицу с результатами
    summary_data = []
    for result in results:
        row = {
            "model": result["model"],
            "examples_count": result["examples_count"],
            "elapsed_time": result["elapsed_time"]
        }
        
        # Добавляем основные метрики
        for metric_name, metric_values in result.get("metrics", {}).items():
            row[f"{metric_name}_mean"] = metric_values.get("mean", np.nan)
        
        summary_data.append(row)
    
    # Сохраняем сводную таблицу
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / "summary_table.csv", index=False)
    
    # Финальная очистка всех ресурсов
    global _document_cache
    _document_cache.clear()
    
    # Очищаем память и реранкер в зависимости от устройства
    from core.modules.ranking import cleanup_reranker
    cleanup_reranker()
    
    # Финальная очистка
    stop_evaluation()
    import gc
    gc.collect()
    
    if HAS_TPU:
        # Очистка TPU памяти
        xm.mark_step()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info(f"🎉 Оценка завершена! Результаты сохранены в {output_dir}")
    logger.info(f"📊 Обработано {len(MODELS_TO_EVALUATE)} моделей с оптимизированным алгоритмом для {device_type}")

if __name__ == "__main__":
    # Избегаем устаревшего предупреждения
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main()) 