import json
import os
import pandas as pd
import numpy as np
import torch
import asyncio
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer

from core.llm import get_llm
from core.config import MODEL_NAME, EMBEDDING_MODEL
from core.llm.deepeval_adapter import OpenRouterDeepEvalAdapter
from core.llm.chains import split_docs
from core.db import get_embedding_model
from core.utils.singletons import lazy_singleton

from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    GEval
)

@lazy_singleton
def get_bleurt_model():
    """Возвращает модель BLEURT для оценки текстовой генерации"""
    model = BleurtForSequenceClassification.from_pretrained('lucadiliello/BLEURT-20')
    
    # Переводим модель в режим оценки
    model.eval()
    
    # Перемещаем на GPU, если доступно
    if torch.cuda.is_available():
        model = model.cuda()
        
    return model

@lazy_singleton
def get_bleurt_tokenizer():
    """Возвращает токенизатор для модели BLEURT"""
    return BleurtTokenizer.from_pretrained('lucadiliello/BLEURT-20')

def calculate_bleurt_score(references, candidates):
    """Рассчитывает BLEURT оценку между эталонными и кандидатными текстами"""
    model = get_bleurt_model()
    tokenizer = get_bleurt_tokenizer()
    
    with torch.no_grad():
        inputs = tokenizer(references, candidates, padding='longest', return_tensors='pt')
        
        # Перемещаем входные данные на GPU, если доступно
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
        scores = model(**inputs).logits.flatten().cpu().tolist()
    
    return scores

def calculate_cosine_similarity(texts1, texts2):
    """Рассчитывает косинусную близость между двумя наборами текстов, используя эмбеддер из конфигурации"""
    embedding_model = get_embedding_model(model_name=EMBEDDING_MODEL)
    
    # Получаем эмбеддинги для обоих наборов текстов
    embeddings1 = [embedding_model.embed_query(text) for text in texts1]
    embeddings2 = [embedding_model.embed_query(text) for text in texts2]
    
    # Рассчитываем косинусную близость между соответствующими эмбеддингами
    similarities = [cosine_similarity([emb1], [emb2])[0][0] for emb1, emb2 in zip(embeddings1, embeddings2)]
    
    return similarities


def create_test_cases(dataset, system_responses=None, limit=None):
    if limit is not None and limit < len(dataset):
        if system_responses is not None:
            # Сохраняем консистентность индексов при выборке
            selected_indices = dataset.sample(limit).index
            dataset = dataset.loc[selected_indices]
            system_responses = system_responses.loc[selected_indices]
        else:
            dataset = dataset.sample(limit)
    
    test_cases = []
    for i, (_, row) in enumerate(tqdm(dataset.iterrows(), total=len(dataset), desc="Создание тестовых случаев")):
        # Если есть ответы системы, используем их для actual_output
        if system_responses is not None:
            actual_output = system_responses.iloc[i]["answer"] if "answer" in system_responses.columns else system_responses.iloc[i]
        else:
            # Иначе используем ответ из датасета и для actual_output, и для expected_output
            actual_output = row["answer"]
        
        # Создаем тестовый случай
        test_case = LLMTestCase(
            input=row["question"],
            actual_output=actual_output,
            expected_output=row["answer"],
            retrieval_context=split_docs(row["context"]) if isinstance(row["context"], str) else row["context"]
        )
        test_cases.append(test_case)
    
    return test_cases

def evaluate_dataset(dataset, system_responses=None, model_name=MODEL_NAME, temperature=0.0, limit=None):
    # Создаем тестовые случаи в зависимости от наличия ответов системы
    test_cases = create_test_cases(dataset, system_responses, limit)
    eval_model = OpenRouterDeepEvalAdapter(model_name=model_name, temperature=temperature)
    
    # RAG Triad и Correctness
    metrics = [
        FaithfulnessMetric(threshold=0.5, model=eval_model),
        AnswerRelevancyMetric(threshold=0.5, model=eval_model),
        ContextualRelevancyMetric(threshold=0.5, model=eval_model),
        GEval(
            name="Correctness",
            criteria="Определите, является ли 'фактический вывод' правильным на основе 'ожидаемого вывода'.",
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
            threshold=0.5,
            model=eval_model
        )
    ]
    
    # Подготовка данных для BLEURT и косинусных метрик
    references = [test_case.expected_output for test_case in test_cases]
    candidates = [test_case.actual_output for test_case in test_cases]
    
    # Расчет BLEURT оценок
    bleurt_scores = calculate_bleurt_score(references, candidates)
    
    # Расчет косинусных оценок
    cosine_scores = calculate_cosine_similarity(references, candidates)
    
    results = []
    for i, test_case in enumerate(tqdm(test_cases, desc="Оценка примеров")):
        metric_scores = {}
        for metric in metrics:
            metric.measure(test_case)
            metric_name = metric.__class__.__name__.replace("Metric", "")
            metric_scores[metric_name] = metric.score

        # Добавляем BLEURT и косинусную оценки
        metric_scores["BLEURT"] = bleurt_scores[i]
        metric_scores["CosineSimilarity"] = cosine_scores[i]

        # В зависимости от наличия ответов системы формируем результат
        if system_responses is not None:
            result_item = {
                "question": test_case.input,
                "system_answer": test_case.actual_output,
                "golden_answer": test_case.expected_output,
                **metric_scores,
                "avg_score": np.mean(list(metric_scores.values())),
            }
        else:
            result_item = {
                **dataset.iloc[i].to_dict(),
                **metric_scores,
                "avg_score": np.mean(list(metric_scores.values())),
            }
            
        results.append(result_item)
    
    return pd.DataFrame(results)

def generate_report(evaluated_df):
    # Определяем колонки с метриками (не входящие в исключаемый список)
    exclude_cols = ["question", "answer", "system_answer", "golden_answer", 
                   "context", "chunk_ids", "avg_score"]
    metric_columns = [col for col in evaluated_df.columns if col not in exclude_cols]
    
    stats = {}
    # Для каждой метрики вычисляем статистики
    for metric in metric_columns:
        scores = evaluated_df[metric].dropna().tolist()
        if scores:
            stats[metric] = {
                "mean": float(np.mean(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "median": float(np.median(scores))
            }
    
    # Добавляем статистику по средней оценке
    stats["avg_score"] = {
        "mean": float(np.mean(evaluated_df["avg_score"].dropna())),
        "min": float(np.min(evaluated_df["avg_score"].dropna())),
        "max": float(np.max(evaluated_df["avg_score"].dropna())),
        "median": float(np.median(evaluated_df["avg_score"].dropna()))
    }
    
    return stats

def filter_best_examples(evaluated_df, threshold=0.5):
    best_df = evaluated_df[evaluated_df["avg_score"] >= threshold].copy()
    print(f"Отобрано {len(best_df)} примеров из {len(evaluated_df)} (порог: {threshold})")
    return best_df

def save_results(dataset, output_path, report=None):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dataset.to_csv(output_path, index=False)
    
    if report:
        report_path = os.path.join(
            os.path.dirname(output_path), 
            f"{os.path.splitext(os.path.basename(output_path))[0]}_report.json"
        )
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2) 

def stop():
    """Сбрасывает все синглтоны для освобождения ресурсов"""
    get_bleurt_model.reset()
    get_bleurt_tokenizer.reset() 

async def calculate_bleurt_score_async(references, candidates):
    """Асинхронно рассчитывает BLEURT оценку между эталонными и кандидатными текстами"""
    # Поскольку BLEURT работает с GPU и вычисления уже оптимизированы,
    # просто вызываем синхронную функцию
    return calculate_bleurt_score(references, candidates)

async def calculate_cosine_similarity_async(texts1, texts2):
    """Асинхронно рассчитывает косинусную близость между двумя наборами текстов"""
    # Поскольку вычисление эмбеддингов - затратная операция,
    # просто вызываем синхронную функцию
    return calculate_cosine_similarity(texts1, texts2)

async def evaluate_dataset_async(dataset, system_responses=None, model_name=MODEL_NAME, temperature=0.0, limit=None, max_concurrency=5):
    """Асинхронная версия функции evaluate_dataset, использующая a_measure для метрик"""
    # Создаем тестовые случаи в зависимости от наличия ответов системы
    test_cases = create_test_cases(dataset, system_responses, limit)
    eval_model = OpenRouterDeepEvalAdapter(model_name=model_name, temperature=temperature)
    
    # RAG Triad и Correctness
    metrics = [
        FaithfulnessMetric(threshold=0.5, model=eval_model),
        AnswerRelevancyMetric(threshold=0.5, model=eval_model),
        ContextualRelevancyMetric(threshold=0.5, model=eval_model),
        GEval(
            name="Correctness",
            criteria="Определите, является ли 'фактический вывод' правильным на основе 'ожидаемого вывода'.",
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
            threshold=0.5,
            model=eval_model
        )
    ]
    
    # Подготовка данных для BLEURT и косинусных метрик
    references = [test_case.expected_output for test_case in test_cases]
    candidates = [test_case.actual_output for test_case in test_cases]
    
    # Асинхронный расчет BLEURT и косинусных оценок
    bleurt_task = asyncio.create_task(calculate_bleurt_score_async(references, candidates))
    cosine_task = asyncio.create_task(calculate_cosine_similarity_async(references, candidates))
    
    # Ограничиваем количество одновременных запросов к LLM
    semaphore = asyncio.Semaphore(max_concurrency)
    
    async def process_test_case(i, test_case):
        """Обрабатывает один тестовый случай со всеми метриками"""
        async with semaphore:
            metric_scores = {}
            
            # Асинхронно вычисляем метрики
            measure_tasks = []
            for metric in metrics:
                # Проверяем наличие асинхронного метода a_measure
                if hasattr(metric, 'a_measure'):
                    task = asyncio.create_task(metric.a_measure(test_case))
                else:
                    # Если асинхронного метода нет, запускаем синхронный в другом потоке
                    loop = asyncio.get_event_loop()
                    task = loop.run_in_executor(None, lambda: metric.measure(test_case))
                measure_tasks.append((metric, task))
            
            # Ждем завершения всех задач по метрикам
            for metric, task in measure_tasks:
                await task
                metric_name = metric.__class__.__name__.replace("Metric", "")
                metric_scores[metric_name] = metric.score
            
            return i, metric_scores
    
    # Запускаем обработку всех тестовых случаев параллельно
    tasks = [process_test_case(i, test_case) for i, test_case in enumerate(test_cases)]
    metric_results = await asyncio.gather(*tasks)
    
    # Получаем результаты других асинхронных задач
    bleurt_scores = await bleurt_task
    cosine_scores = await cosine_task
    
    # Формируем итоговые результаты
    results = []
    for i, metric_scores in sorted(metric_results):
        test_case = test_cases[i]
        
        # Добавляем BLEURT и косинусную оценки
        metric_scores["BLEURT"] = bleurt_scores[i]
        metric_scores["CosineSimilarity"] = cosine_scores[i]
        
        # В зависимости от наличия ответов системы формируем результат
        if system_responses is not None:
            result_item = {
                "question": test_case.input,
                "system_answer": test_case.actual_output,
                "golden_answer": test_case.expected_output,
                **metric_scores,
                "avg_score": np.mean(list(metric_scores.values())),
            }
        else:
            result_item = {
                **dataset.iloc[i].to_dict(),
                **metric_scores,
                "avg_score": np.mean(list(metric_scores.values())),
            }
            
        results.append(result_item)
    
    return pd.DataFrame(results) 