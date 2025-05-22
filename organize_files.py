#!/usr/bin/env python3
import os
import shutil
import glob
from pathlib import Path

def create_directory_structure(base_path):
    """Создает структуру директорий проекта"""
    directories = [
        "data/tbank_faiss",
        "data/csv"
    ]
    
    for directory in directories:
        os.makedirs(os.path.join(base_path, directory), exist_ok=True)

def organize_files(repo_path, current_dir):
    """Организует файлы в нужные директории"""
    # Получаем список всех файлов, скачанных через gdown
    file_ids = [
        "1ZeidBZOEZQayR3WhKlIXB2RkFzW1UiTD", 
        "1AH4j1GJ23aMF7ZG25ftTMcoN2i9CBqCM",
        "1PJIMSkQSKltYC4BkY35wH8bmMJIXgN86", 
        "1MwyH3fkupzm1Qkv-Jyo5j1AHnGN7h2mu",
        "1ZEe2Ib1bS2O7v_PnCIphlnSLzyGHQcsf", 
        "11_F_VW5zZnR9jrilAegkZdx1LSPX8eHu", 
        "1dFFUhBVXSJqdEp1IBseMLY5rm7A7oHFx"
    ]
    
    all_files = []
    for file_id in file_ids:
        matches = glob.glob(os.path.join(current_dir, file_id + "*"))
        if matches:
            all_files.append(matches[0])
    
    # Два первых файла в списке должны быть FAISS индексами
    # Остальные файлы - CSV и example.env
    processed_files = []
    
    # Сначала обрабатываем FAISS индексы (первые два файла в списке)
    if len(all_files) >= 2:
        # Первый файл -> index.faiss
        faiss_file = all_files[0]
        dest_path = os.path.join(repo_path, "data/tbank_faiss/index.faiss")
        shutil.copy2(faiss_file, dest_path)
        print(f"Скопирован FAISS индекс: {faiss_file} -> {dest_path}")
        processed_files.append(faiss_file)
        
        # Второй файл -> index.pkl
        pkl_file = all_files[1]
        dest_path = os.path.join(repo_path, "data/tbank_faiss/index.pkl")
        shutil.copy2(pkl_file, dest_path)
        print(f"Скопирован PKL индекс: {pkl_file} -> {dest_path}")
        processed_files.append(pkl_file)
    
    # Обрабатываем example.env (третий файл)
    if len(all_files) >= 3:
        env_file = all_files[2]
        dest_path = os.path.join(repo_path, "example.env")
        shutil.copy2(env_file, dest_path)
        print(f"Скопирован example.env: {env_file} -> {dest_path}")
        processed_files.append(env_file)
    
    # Остальные файлы считаем CSV и копируем в data/csv
    for i, file in enumerate(all_files):
        if file not in processed_files:
            dest_path = os.path.join(repo_path, f"data/csv/dataset{i+1}.csv")
            shutil.copy2(file, dest_path)
            print(f"Скопирован CSV файл: {file} -> {dest_path}")
    
    return len(all_files)

def main():
    # Путь к клонированному репозиторию
    repo_path = os.path.join(os.getcwd(), "rag")
    
    # Текущая директория, где находятся загруженные файлы
    current_dir = os.getcwd()
    
    # Создаем структуру директорий
    create_directory_structure(repo_path)
    
    # Организуем файлы
    num_files = organize_files(repo_path, current_dir)
    
    print(f"\nУспешно организовано {num_files} файлов в директории {repo_path}")
    print("Теперь вы можете перейти в эту директорию и запустить проект.")

if __name__ == "__main__":
    main() 