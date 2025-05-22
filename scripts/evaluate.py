import sys
import os
# Добавление корневой директории проекта в PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import argparse
import pandas as pd
import os
from tqdm import tqdm

from core.llm import get_llm
from core.config import MODEL_NAME, PROMPTS, DOCS_SEPARATOR
from core.db import retrieve, rerank
from core.llm.chains import get_retrieval_chain, format_docs, split_docs
from langchain_core.prompts import ChatPromptTemplate

from evals import (
    evaluate_dataset, filter_best_examples, 
    generate_report, save_results
)

def parse_args():
    parser = argparse.ArgumentParser(description="Универсальный скрипт для оценки")
    # Общие параметры
    parser.add_argument("--mode", type=str, choices=["synthetic", "system"], required=True, 
                        help="Режим оценки: synthetic - оценка синтетических данных, system - оценка системы")
    parser.add_argument("--output-file", type=str, required=True, 
                        help="Путь для сохранения результатов оценки")
    parser.add_argument("--limit", type=int, default=None, 
                        help="Лимит примеров для оценки")
    parser.add_argument("--model", type=str, default=MODEL_NAME, 
                        help="Модель для оценки")
    parser.add_argument("--temperature", type=float, default=0.0, 
                        help="Температура модели")
    
    # Параметры для режима synthetic
    parser.add_argument("--input-file", type=str, 
                        help="Путь к CSV файлу с синтетическими данными (для режима synthetic)")
    parser.add_argument("--threshold", type=float, default=0.5, 
                        help="Порог качества для фильтрации (0.0-1.0) (для режима synthetic)")
    
    # Параметры для режима system
    parser.add_argument("--golden-file", type=str, 
                        help="Путь к CSV файлу с эталонными данными (для режима system)")
    parser.add_argument("--system-file", type=str, default=None, 
                        help="Путь к CSV файлу с ответами системы (если не указан, ответы будут сгенерированы) (для режима system)")
    parser.add_argument("--top-k", type=int, default=5, 
                        help="Количество документов для извлечения при генерации ответов (для режима system)")
    
    return parser.parse_args()

def generate_system_responses(dataset, top_k=3, limit=None):
    if limit is not None and limit < len(dataset):
        dataset = dataset.sample(limit)
    
    # Получаем retrieval_chain, который выполнит и извлечение контекста, и генерацию ответа
    retrieval_chain = get_retrieval_chain()
    
    responses = []
    for _, row in tqdm(dataset.iterrows(), total=len(dataset), desc="Генерация ответов"):
        question = row["question"]

        
        # Генерируем ответ с использованием retrieval_chain
        try:
            # Если контекст предоставлен, мы все равно используем retrieval_chain,
            # но в этом случае нам пришлось бы модифицировать chain для принятия контекста
            # Для простоты всегда используем автоматическое извлечение
            answer = retrieval_chain.invoke(question)
            
            # Если контекста не было, извлекаем его для сохранения в ответах
            if not context:
                docs = retrieve(question, top_k=top_k)
                reranked_docs = rerank(question, docs)
                context = format_docs(reranked_docs)
            
            responses.append({
                "question": question,
                "answer": answer,
                "context": context
            })
        except Exception as e:
            print(f"Ошибка при генерации ответа: {str(e)}")
            # В случае ошибки добавляем пустой ответ
            responses.append({
                "question": question,
                "answer": "",
                "context": context if context else ""
            })
    
    return pd.DataFrame(responses)

def evaluate_synthetic_data(args):
    # Инициализация модели
    get_llm(model_name=args.model, temperature=args.temperature)
    
    # Загрузка датасета
    dataset = pd.read_csv(args.input_file)
    
    # Оценка датасета
    print(f"Оценка синтетических данных из {args.input_file}...")
    evaluated_dataset = evaluate_dataset(
        dataset, 
        model_name=args.model, 
        temperature=args.temperature, 
        limit=args.limit
    )
    
    # Фильтрация лучших примеров
    filtered_dataset = filter_best_examples(evaluated_dataset, args.threshold)
    
    # Генерация отчета
    report = generate_report(evaluated_dataset)
    
    # Вывод общей оценки
    avg_score = report["avg_score"]["mean"]
    print(f"Средняя оценка датасета: {avg_score:.4f}")
    
    # Сохранение результатов
    save_results(filtered_dataset, args.output_file, report)
    print(f"Результаты сохранены в {args.output_file}")

def evaluate_system(args):
    # Инициализация модели
    get_llm(model_name=args.model, temperature=args.temperature)
    
    # Загрузка эталонного датасета
    golden_dataset = pd.read_csv(args.golden_file)
    
    # Получение ответов системы
    if args.system_file:
        # Если ответы системы предоставлены, загружаем их
        system_responses = pd.read_csv(args.system_file)
    else:
        # Иначе генерируем ответы системы
        print("Генерация ответов системы...")
        system_responses = generate_system_responses(
            golden_dataset,
            top_k=args.top_k,
            limit=args.limit
        )
        
        # Сохраняем сгенерированные ответы
        system_output_path = args.output_file.replace(".csv", "_system_responses.csv")
        save_results(system_responses, system_output_path)
        print(f"Ответы системы сохранены в {system_output_path}")
    
    # Оценка ответов системы относительно эталонных ответов
    print("Оценка качества ответов системы...")
    evaluation_results = evaluate_dataset(
        golden_dataset,
        system_responses=system_responses,
        model_name=args.model,
        temperature=args.temperature,
        limit=args.limit
    )
    
    # Генерация отчета
    report = generate_report(evaluation_results)
    
    # Вывод общей оценки
    avg_score = report["avg_score"]["mean"]
    print(f"Средняя оценка системы: {avg_score:.4f}")
    
    # Сохранение результатов
    save_results(evaluation_results, args.output_file, report)
    print(f"Результаты оценки сохранены в {args.output_file}")

def main():
    args = parse_args()
    
    # Проверяем наличие необходимых аргументов в зависимости от режима
    if args.mode == "synthetic" and not args.input_file:
        raise ValueError("Для режима 'synthetic' требуется указать --input-file")
    
    if args.mode == "system" and not args.golden_file:
        raise ValueError("Для режима 'system' требуется указать --golden-file")
    
    # Вызываем соответствующую функцию в зависимости от режима
    if args.mode == "synthetic":
        evaluate_synthetic_data(args)
    else:  # args.mode == "system"
        evaluate_system(args)

if __name__ == "__main__":
    main() 