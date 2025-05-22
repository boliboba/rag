#!/usr/bin/env python3
import sys
import os
# Добавление корневой директории проекта в PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import asyncio
import time
from tqdm import tqdm

from core.llm import get_llm
from core.config import MODEL_NAME
from core.llm.models import ChatOpenRouter

async def test_ainvoke_with_models(models):
    """Тестирует асинхронный вызов для разных моделей"""
    results = {}
    
    for model_name in tqdm(models, desc="Тестирование моделей"):
        # Получаем экземпляр LLM для конкретной модели
        llm = get_llm(model_name=model_name)
        
        # Тестовый промпт
        prompt = f"Расскажи короткий анекдот про программиста. Подпишись в конце как {model_name}."
        
        try:
            # Асинхронный вызов
            start_time = time.time()
            response = await llm.ainvoke(prompt)
            elapsed = time.time() - start_time
            
            results[model_name] = {
                "success": True,
                "response": response.content if hasattr(response, 'content') else str(response),
                "time": elapsed
            }
        except Exception as e:
            results[model_name] = {
                "success": False,
                "error": str(e),
                "time": 0
            }
    
    return results

async def test_concurrent_ainvoke(models, concurrency=2):
    """Тестирует одновременный асинхронный вызов для разных моделей с ограничением параллелизма"""
    semaphore = asyncio.Semaphore(concurrency)
    
    async def call_with_semaphore(model):
        async with semaphore:
            # Получаем экземпляр LLM для конкретной модели
            llm = get_llm(model_name=model)
            
            # Тестовый промпт
            prompt = f"Расскажи короткий анекдот про программиста. Подпишись в конце как {model}."
            
            try:
                # Асинхронный вызов
                start_time = time.time()
                response = await llm.ainvoke(prompt)
                elapsed = time.time() - start_time
                
                return {
                    "model": model,
                    "success": True,
                    "response": response.content if hasattr(response, 'content') else str(response),
                    "time": elapsed
                }
            except Exception as e:
                return {
                    "model": model,
                    "success": False,
                    "error": str(e),
                    "time": 0
                }
    
    # Создаем список задач
    tasks = [call_with_semaphore(model) for model in models]
    
    # Запускаем задачи конкурентно
    return await asyncio.gather(*tasks)

async def main():
    # Список моделей для тестирования
    models_to_test = [
        "google/gemini-1.5-pro",
        "anthropic/claude-3-haiku-20240307",
        "meta-llama/llama-3-8b-instruct"
    ]
    
    print("\n--- Тест 1: Последовательные вызовы ---")
    sequential_results = await test_ainvoke_with_models(models_to_test)
    
    for model, result in sequential_results.items():
        print(f"\nМодель: {model}")
        if result["success"]:
            print(f"Время: {result['time']:.2f} сек")
            print(f"Ответ: {result['response'][:100]}...")
        else:
            print(f"Ошибка: {result['error']}")
    
    print("\n--- Тест 2: Параллельные вызовы ---")
    concurrent_results = await test_concurrent_ainvoke(models_to_test, concurrency=2)
    
    for result in concurrent_results:
        print(f"\nМодель: {result['model']}")
        if result["success"]:
            print(f"Время: {result['time']:.2f} сек")
            print(f"Ответ: {result['response'][:100]}...")
        else:
            print(f"Ошибка: {result['error']}")

if __name__ == "__main__":
    # Создаем и запускаем новый цикл событий
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main()) 