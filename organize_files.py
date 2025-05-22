#!/usr/bin/env python3
import os
import shutil
import glob
from pathlib import Path

def create_directory_structure(base_path):
    """Создает структуру директорий проекта"""
    directories = [
        "core/db",
        "core/llm",
        "core/modules",
        "core/utils",
        "data/tbank_faiss",
        "data/csv",  # Добавляем директорию для CSV файлов
        "evals",
        "scripts",
        ".deepeval"
    ]
    
    for directory in directories:
        os.makedirs(os.path.join(base_path, directory), exist_ok=True)
        # Создаем __init__.py в python-пакетах
        if not directory.startswith(".") and not directory.startswith("data"):
            init_file = os.path.join(base_path, directory, "__init__.py")
            if not os.path.exists(init_file):
                with open(init_file, "w") as f:
                    f.write("# Инициализация пакета\n")

def map_files_to_destinations(base_path, current_dir):
    """Создает маппинг файлов для перемещения в нужные директории"""
    # Маппинг имен файлов к их целевым директориям
    file_mapping = {
        # Векторная база данных и индексы (2 файла)
        "1ZeidBZOEZQayR3WhKlIXB2RkFzW1UiTD": os.path.join(base_path, "data/tbank_faiss/index.faiss"),
        "1AH4j1GJ23aMF7ZG25ftTMcoN2i9CBqCM": os.path.join(base_path, "data/tbank_faiss/index.pkl"),
        
        # Файл example.env (1 файл)
        "1PJIMSkQSKltYC4BkY35wH8bmMJIXgN86": os.path.join(base_path, "example.env"),
        
        # CSV файлы (4 файла)
        "1MwyH3fkupzm1Qkv-Jyo5j1AHnGN7h2mu": os.path.join(base_path, "data/csv/dataset1.csv"),
        "1ZEe2Ib1bS2O7v_PnCIphlnSLzyGHQcsf": os.path.join(base_path, "data/csv/dataset2.csv"),
        "11_F_VW5zZnR9jrilAegkZdx1LSPX8eHu": os.path.join(base_path, "data/csv/dataset3.csv"),
        "1dFFUhBVXSJqdEp1IBseMLY5rm7A7oHFx": os.path.join(base_path, "data/csv/dataset4.csv"),
    }
    
    # Находим загруженные файлы в текущей директории
    downloaded_files = []
    for file_id in file_mapping.keys():
        matches = glob.glob(os.path.join(current_dir, file_id + "*"))
        if matches:
            downloaded_files.append((matches[0], file_mapping[file_id]))
    
    return downloaded_files

def identify_file_types(file_mappings, base_path):
    """Определяет типы файлов на основе их содержимого и переименовывает их соответственно"""
    updated_mappings = []
    
    for source, destination in file_mappings:
        file_content = ""
        try:
            with open(source, 'r', encoding='utf-8', errors='ignore') as f:
                file_content = f.read(1024)  # Читаем первые 1024 байта для анализа
        except:
            pass  # Если не можем прочитать как текст, возможно это бинарный файл
        
        # Определяем тип файла
        if source.endswith('.faiss') or destination.endswith('.faiss'):
            # Это уже правильно определено как FAISS файл
            updated_mappings.append((source, destination))
        elif source.endswith('.pkl') or destination.endswith('.pkl'):
            # Это уже правильно определено как PKL файл
            updated_mappings.append((source, destination))
        elif "OPENROUTER_API_KEY" in file_content or "MODEL_NAME" in file_content:
            # Это файл конфигурации .env
            updated_mappings.append((source, os.path.join(base_path, "example.env")))
        elif file_content and (file_content.count(',') > 5 or file_content.count('\t') > 5):
            # Вероятно, это CSV файл
            # Ищем первое доступное имя файла
            csv_base = os.path.join(base_path, "data/csv")
            i = 1
            while os.path.exists(os.path.join(csv_base, f"dataset{i}.csv")):
                i += 1
            updated_mappings.append((source, os.path.join(csv_base, f"dataset{i}.csv")))
        else:
            # По умолчанию оставляем как есть
            updated_mappings.append((source, destination))
    
    return updated_mappings

def copy_files(file_mappings):
    """Копирует файлы в нужные директории"""
    for source, destination in file_mappings:
        # Создаем директорию назначения, если она не существует
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        # Копируем файл
        shutil.copy2(source, destination)
        print(f"Скопирован: {source} -> {destination}")

def main():
    # Путь к клонированному репозиторию
    repo_path = os.path.join(os.getcwd(), "rag")
    
    # Текущая директория, где находятся загруженные файлы
    current_dir = os.getcwd()
    
    # Создаем структуру директорий
    create_directory_structure(repo_path)
    
    # Получаем маппинг файлов
    initial_file_mappings = map_files_to_destinations(repo_path, current_dir)
    
    # Определяем типы файлов и корректируем пути назначения
    file_mappings = identify_file_types(initial_file_mappings, repo_path)
    
    # Копируем файлы
    copy_files(file_mappings)
    
    print(f"\nФайлы успешно организованы в директории {repo_path}")
    print("Теперь вы можете перейти в эту директорию и запустить проект.")

if __name__ == "__main__":
    main() 