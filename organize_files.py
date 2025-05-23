#!/usr/bin/env python3
import os
import shutil
import glob
from pathlib import Path

def create_directory_structure(base_path):
    """Создает структуру директорий проекта"""
    directories = [
        "data/tbank_faiss",
        "data"
    ]
    
    for directory in directories:
        os.makedirs(os.path.join(base_path, directory), exist_ok=True)

def organize_files(repo_path, current_dir):
    """Организует файлы в нужные директории на основе их имен"""
    # Ищем файлы в текущей директории
    
    # Перемещаем FAISS индексы
    faiss_index_files = glob.glob(os.path.join(current_dir, "index.*"))
    for file in faiss_index_files:
        file_name = os.path.basename(file)
        dest_path = os.path.join(repo_path, "data/tbank_faiss", file_name)
        shutil.copy2(file, dest_path)
        print(f"Скопирован индекс: {file} -> {dest_path}")
    
    # Перемещаем example.env как .env
    env_files = glob.glob(os.path.join(current_dir, "*.env"))
    for file in env_files:
        if "example" in file.lower() or "env.example" in file.lower():
            dest_path = os.path.join(repo_path, ".env")
            shutil.copy2(file, dest_path)
            print(f"Скопирован env файл: {file} -> {dest_path}")
    
    # Перемещаем CSV файлы
    csv_files = glob.glob(os.path.join(current_dir, "*.csv"))
    for file in csv_files:
        file_name = os.path.basename(file)
        dest_path = os.path.join(repo_path, "data", file_name)
        shutil.copy2(file, dest_path)
        print(f"Скопирован CSV файл: {file} -> {dest_path}")

def main():
    # Путь к клонированному репозиторию
    repo_path = os.path.join(os.getcwd(), "rag")
    
    # Текущая директория, где находятся загруженные файлы
    current_dir = os.getcwd()
    
    # Создаем структуру директорий
    create_directory_structure(repo_path)
    
    # Организуем файлы
    organize_files(repo_path, current_dir)

if __name__ == "__main__":
    main() 