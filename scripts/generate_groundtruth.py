import sys
import os
# Добавление корневой директории проекта в PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import argparse
import pandas as pd
from tqdm import tqdm
import os

from core.llm.chains import get_qa_chain, format_docs
from core.config import DOCS_SEPARATOR


def parse_args():
    parser = argparse.ArgumentParser(description="Генерация синтетических ответов")
    parser.add_argument("--questions-csv", type=str, required=True, help="Путь к CSV файлу с вопросами")
    parser.add_argument("--output-file", type=str, default="data/synthetic_qa.csv", help="Путь к выходному CSV файлу")
    parser.add_argument("--num-examples", type=int, default=100, help="Количество примеров для генерации")
    parser.add_argument("--num-chunks", type=int, default=5, help="Количество чанков для каждого вопроса")
    return parser.parse_args()


def get_random_chunks_for_query(eval_df, query, n=5):
    matching_rows = eval_df[(eval_df['query'] == query) & (eval_df['match'] == 1)]
    if len(matching_rows) >= n:
        return matching_rows.sample(n).to_dict('records')
    
    additional_rows = eval_df[(eval_df['query'] == query) & (eval_df['match'] == 0)].sample(n - len(matching_rows))
    return pd.concat([matching_rows, additional_rows]).to_dict('records')


def create_synthetic_dataset(questions_df, num_examples=100, num_chunks=5):
    qa_chain = get_qa_chain()
    
    dataset = []
    
    unique_questions = questions_df[questions_df['match'] == 1]['query'].unique()
    questions_to_use = pd.Series(unique_questions).sample(min(len(unique_questions), num_examples)).tolist()
    
    for question in tqdm(questions_to_use, total=len(questions_to_use), desc="Генерация датасета"):
        selected_chunks = get_random_chunks_for_query(questions_df, question, num_chunks)
        context = DOCS_SEPARATOR.join([chunk["candidate"] for chunk in selected_chunks])
        
        answer = qa_chain.invoke({
            "question": question,
            "context": context
        })
        
        dataset.append({
            "question": question,
            "answer": answer,
            "context": context,
            "chunk_ids": [chunk.get("chunk_id", i) for i, chunk in enumerate(selected_chunks)]
        })
    
    return dataset

def main():
    args = parse_args()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    questions_df = pd.read_csv(args.questions_csv, sep=";")
    
    # Фильтрация: исключаем "Материалы из сообщества"
    if 'rubrics' in questions_df.columns:
        initial_count = len(questions_df)
        questions_df = questions_df[questions_df['rubrics'] != 'Материалы из сообщества']
        filtered_count = len(questions_df)
        print(f"Исключены записи с рубрикой 'Материалы из сообщества': {initial_count} -> {filtered_count} ({initial_count - filtered_count} удалено)")
    else:
        print("Предупреждение: поле 'rubrics' не найдено в данных")

    dataset = create_synthetic_dataset(
        questions_df, 
        num_examples=args.num_examples,
        num_chunks=args.num_chunks
    )

    df_output = pd.DataFrame(dataset)
    df_output.to_csv(args.output_file, index=False)

if __name__ == "__main__":
    main()