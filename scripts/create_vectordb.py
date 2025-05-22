import sys
import os
# Добавление корневой директории проекта в PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import argparse
import pandas as pd
from core.modules import preprocess_article, process_data_to_documents
from core.db import create_vectorstore, save_vectorstore, get_embedding_model
from core.config import EMBEDDING_MODEL
from langchain.text_splitter import MarkdownTextSplitter


def parse_args():
    parser = argparse.ArgumentParser(description="Создание векторной базы данных")
    parser.add_argument("--input", type=str, required=True, help="Путь к входному CSV файлу с данными")
    parser.add_argument("--output", type=str, required=True, help="Путь к выходной директории для векторной БД")
    parser.add_argument("--embedding-model", type=str, default=EMBEDDING_MODEL, help="Модель для создания эмбеддингов")
    parser.add_argument("--chunk-size", type=int, default=1024, help="Размер чанка при разбиении текста")
    parser.add_argument("--chunk-overlap", type=int, default=20, help="Перекрытие чанков при разбиении текста")
    return parser.parse_args()

def main():
    args = parse_args()
    
    df = pd.read_csv(args.input, sep=';')
    df = df.dropna(subset=["content_raw"])
    df["content_raw"] = df["content_raw"].apply(preprocess_article)   

    text_splitter = MarkdownTextSplitter(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    documents = process_data_to_documents(df, text_splitter=text_splitter)
    
    embedding_model = get_embedding_model(model_name=args.embedding_model)
    vectorstore = create_vectorstore(documents, embedding_model)
    save_vectorstore(vectorstore, args.output)

if __name__ == "__main__":
    main() 