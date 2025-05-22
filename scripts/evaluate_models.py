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
    "google/gemma-3-12b-it",
    "qwen/qwen3-32b"
]

# Фиксированная температура
TEMPERATURE = 0.0

# Модель для проведения оценки (независимо от оцениваемой модели)
EVAL_MODEL_NAME = "google/gemini-2.5-flash-preview-05-20"

# Путь к тестовому датасету
TEST_DATASET_PATH = "data/filtered_evaluated_dataset.csv"

# Количество примеров для оценки (None для всего датасета)
LIMIT = 3

# Максимальное количество одновременно оцениваемых моделей
MAX_CONCURRENCY = 6

def create_model_specific_chain(model_name):
    """Создает специальную цепочку для конкретной модели без изменения глобальных настроек"""
    prompt = ChatPromptTemplate.from_template(PROMPTS["qa"])
    
    # Создаем новый экземпляр LLM для этой конкретной модели
    llm = get_llm(model_name=model_name, temperature=TEMPERATURE)
    
    def retrieve_and_rerank(query):
        docs = retrieve(query)
        print(f"Найдено чанков: {len(docs)}")
        # Реранкер временно отключен для экономии памяти
        # reranked_docs = rerank(query, docs)
        # print(f"Реранжировано чанков: {len(reranked_docs)}")
        
        # Выбираем 5 случайных документов для экономии памяти
        import random
        if len(docs) > 5:
            selected_docs = random.sample(docs, 5)
            print(f"Выбрано 5 случайных документов из {len(docs)}")
        else:
            selected_docs = docs
            print(f"Используются все {len(docs)} документов")
        
        print("Реранкер отключен - используются исходные документы")
        return selected_docs
    
    retrieval_fn = lambda query: format_docs(
        retrieve_and_rerank(query)
    )
    
    rag_chain = (
        {"context": retrieval_fn, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

async def generate_system_responses_async(dataset, model_name, limit=None):
    """Асинхронно генерирует ответы системы для заданной модели"""
    if limit is not None and limit < len(dataset):
        # Используем тот же random_state=42 для согласованности с create_test_cases
        dataset = dataset.sample(limit, random_state=42).reset_index(drop=True)
    
    # Создаем специальную цепочку для этой модели
    retrieval_chain = create_model_specific_chain(model_name)
    
    responses = []
    for _, row in tqdm(dataset.iterrows(), total=len(dataset), 
                      desc=f"Генерация ответов для {model_name}"):
        question = row["question"]
        golden_answer = row["answer"]
        
        # Используем стандартный retrieval_chain с асинхронным вызовом
        try:
            # retrieval_chain извлекает контекст и генерирует ответ
            print(f"\nВопрос: {question}")
            result = await retrieval_chain.ainvoke(question)
            answer = result
            print(f"Ответ модели: {answer[:100]}..." if len(answer) > 100 else f"Ответ модели: {answer}")
            
            responses.append({
                "question": question,
                "system_answer": answer,
                "golden_answer": golden_answer
            })
        except Exception as e:
            logger.error(f"Ошибка при генерации ответа: {e}")
            responses.append({
                "question": question,
                "system_answer": "Произошла ошибка при генерации ответа",
                "golden_answer": golden_answer
            })
    
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
        limit=limit
    )
    
    # Сохраняем сырые ответы
    system_responses_path = model_dir / "system_responses.csv"
    system_responses.to_csv(system_responses_path, index=False)
    logger.info(f"Ответы системы сохранены в {system_responses_path}")
    
    # Оцениваем результаты
    try:
        # Используем асинхронную версию оценки для ускорения
        # Используем EVAL_MODEL_NAME для оценки всех моделей
        logger.info(f"Используем {EVAL_MODEL_NAME} для оценки ответов модели {model_name}")
        evaluation_df = await evaluate_dataset_async(
            dataset=dataset,
            system_responses=system_responses,
            model_name=EVAL_MODEL_NAME,  # Используем одну и ту же модель для оценки
            temperature=TEMPERATURE,
            limit=limit,
            max_concurrency=6  # Лимит одновременных запросов к LLM для метрик
        )
        
        # Генерируем отчет
        report = generate_report(evaluation_df)
        
        # Сохраняем результаты и отчет
        results_path = model_dir / "evaluation_results.csv"
        save_results(evaluation_df, str(results_path), report)
        
        # Освобождаем ресурсы
        stop_evaluation()
        
        elapsed_time = time.time() - start_time
        logger.info(f"Оценка модели {model_name} завершена за {elapsed_time:.2f} секунд")
        

        
        return {
            "model": model_name,
            "metrics": report,
            "elapsed_time": elapsed_time,
            "examples_count": len(dataset) if limit is None else min(len(dataset), limit)
        }
    except Exception as e:
        logger.error(f"Ошибка при оценке результатов: {e}")
        return {
            "model": model_name,
            "metrics": {},
            "elapsed_time": time.time() - start_time,
            "examples_count": len(dataset) if limit is None else min(len(dataset), limit),
            "error": str(e)
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
    
    # Сохраняем параметры запуска
    with open(output_dir / "config.json", "w") as f:
        json.dump({
            "models": MODELS_TO_EVALUATE,
            "eval_model": EVAL_MODEL_NAME,
            "temperature": TEMPERATURE,
            "dataset": TEST_DATASET_PATH,
            "limit": LIMIT
        }, f, indent=2)
    
    # Загружаем датасет
    try:
        dataset = pd.read_csv(TEST_DATASET_PATH)
        logger.info(f"Загружен датасет с {len(dataset)} примерами")
    except Exception as e:
        logger.error(f"Ошибка при загрузке датасета: {e}")
        return
    
    logger.info(f"Будет оценено {len(MODELS_TO_EVALUATE)} моделей")
    
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
    
    logger.info(f"Оценка завершена. Результаты сохранены в {output_dir}")

if __name__ == "__main__":
    # Избегаем устаревшего предупреждения
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main()) 