import json
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from collections import Counter
import sys

# --- Настройки ---
RESULTS_FOLDER = 'results'
REPORT_FILENAME = 'benchmark_analysis_report.txt'
TOP_N_MODELS = 10 # Увеличим для более детального лидерборда

# Имена файлов для графиков
PIE_CHART_FILENAME = 'benchmark_pie_chart.png'
PERFORMANCE_SCATTER_FILENAME = 'benchmark_performance_scatter.png'
FAILURE_STAGE_BAR_FILENAME = 'benchmark_failure_stages.png'
PROVIDER_BAR_FILENAME = 'benchmark_provider_success.png'
# ---

def find_latest_results_file(folder: str) -> str | None:
    """Находит последний JSON-файл с результатами в папке."""
    if not os.path.isdir(folder):
        print(f"Ошибка: Папка с результатами '{folder}' не найдена.", file=sys.stderr)
        print("Пожалуйста, сначала запустите universal_tester.py для генерации результатов.", file=sys.stderr)
        return None
        
    list_of_files = glob.glob(os.path.join(folder, 'final_results_*.json'))
    if not list_of_files:
        print(f"Файлы 'final_results_*.json' не найдены в '{folder}'.", file=sys.stderr)
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"Найден файл с результатами: {latest_file}")
    return latest_file

def load_and_parse_data(filepath: str) -> pd.DataFrame:
    """
    Загружает JSON-файл и парсит его в DataFrame.
    Извлекает данные о финальном тесте, этапах сбоя и успешных провайдерах.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Ошибка: Не удалось прочитать или декодировать файл {filepath}. {e}")
        return pd.DataFrame()

    parsed_data = []
    for model_name, results in data.items():
        final_test = results.get('final_test', {})
        success = final_test.get('success', False)
        summary = final_test.get('summary', {})
        iterations = results.get('iterations', [])
        
        avg_time = float('inf')
        max_mem = None
        issue = "Unknown Error"
        failure_stage = "N/A"
        successful_providers = []

        if success:
            # Для успешных моделей получаем реальные метрики
            avg_time = summary.get('avg_time', float('inf'))
            max_mem = summary.get('max_memory_kb')
            issue = "Success"
            failure_stage = "N/A (Success)"
        else:
            # Для неуспешных ищем причину и этап сбоя
            issue = final_test.get('issue', 'Unknown Error')
            
            # Сначала ищем ошибку LLM (например, "No response")
            llm_error_found = False
            for iter_data in iterations:
                if iter_data.get('error'): # Ошибка LLM
                    failure_stage = iter_data.get('stage', 'Unknown Stage')