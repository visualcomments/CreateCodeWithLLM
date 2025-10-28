import json
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from collections import Counter

# --- Настройки ---
RESULTS_FOLDER = 'промежуточные результаты'
REPORT_FILENAME = 'benchmark_analysis_report.txt'
PIE_CHART_FILENAME = 'benchmark_pie_chart.png'
BAR_CHART_FILENAME = 'benchmark_top5_performance.png'
TOP_N_MODELS = 5
# ---

def find_latest_results_file(folder: str) -> str | None:
    """Находит последний по времени JSON-файл с результатами в папке."""
    list_of_files = glob.glob(os.path.join(folder, 'final_results_*.json'))
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def load_and_parse_data(filepath: str) -> pd.DataFrame:
    """
    Загружает JSON-файл и парсит его в DataFrame.
    Структура JSON: { 'model_name': {'final_test': {...}, ...} }
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
        
        if success:
            # Для успешных моделей берем реальные метрики
            avg_time = summary.get('avg_time', float('inf'))
            max_mem = summary.get('max_memory_kb')
            issue = "Success"
        else:
            # Для провальных ставим 'inf' и берем причину
            avg_time = float('inf')
            max_mem = None
            issue = final_test.get('issue', 'Unknown Error')
            
            # Упрощаем длинные ошибки (часто это JSON с тестами)
            if isinstance(issue, str) and issue.startswith('{'):
                try:
                    # Попробуем взять первую причину сбоя
                    issue_json = json.loads(issue)
                    first_fail = issue_json.get('failing_cases', [{}])[0]
                    issue = first_fail.get('error', 'Test mismatch')
                except json.JSONDecodeError:
                    issue = "Test mismatch (JSON output)" # Ошибка парсинга JSON

        parsed_data.append({
            'Model': model_name,
            'Success': success,
            'Avg_Time (s)': avg_time,
            'Max_Memory (KB)': max_mem,
            'Result_Details': issue
        })
        
    return pd.DataFrame(parsed_data)

def generate_report(df: pd.DataFrame):
    """
    Генерирует текстовый отчет, таблицы и сохраняет графики.
    """
    if df.empty:
        print("Нет данных для анализа.")
        return

    # --- 1. Подготовка данных ---
    total_models = len(df)
    successful_df = df[df['Success'] == True].sort_values(by='Avg_Time (s)')
    failed_df = df[df['Success'] == False]
    
    success_count = len(successful_df)
    fail_count = len(failed_df)
    
    # Статистика по ошибкам
    error_counts = Counter(failed_df['Result_Details'])
    
    # --- 2. Генерация текстового отчета (и в файл, и в консоль) ---
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("                ОТЧЕТ ПО БЕНЧМАРКУ РЕАЛИЗАЦИЙ АЛГОРИТМА LRX")
    report_lines.append("="*80)
    report_lines.append("\n## 1. Критерии ранжирования моделей\n")
    report_lines.append("Лучшая модель определяется по следующим приоритетным критериям:")
    report_lines.append("  1. **Корректность (Success = True):** Самый важный критерий. Модель должна")
    report_lines.append("     пройти *все* тесты, включая пограничные случаи (пустые списки, n=1, n=2).")
    report_lines.append("  2. **Среднее время выполнения (Avg_Time):** Среди *корректных* моделей,")
    report_lines.append("     предпочтение отдается тем, чей код выполняется быстрее.")
    report_lines.append("  3. **Потребление памяти (Max_Memory):** Дополнительный критерий для оценки")
    report_lines.append("     эффективности (в данном тесте менее показателен, чем время).\n")

    report_lines.append("---")
    report_lines.append("## 2. Общая статистика\n")
    
    summary_table = pd.DataFrame({
        'Метрика': ['Всего протестировано моделей', 'Успешно прошли тест', 'Провалили тест'],
        'Значение': [total_models, success_count, fail_count]
    })
    report_lines.append(summary_table.to_string(index=False))
    report_lines.append("\n")

    # --- 3. Выбор лучшей модели ---
    report_lines.append("---")
    report_lines.append("## 3. Выбор лучшей модели\n")
    if not successful_df.empty:
        best_model = successful_df.iloc[0]
        report_lines.append(f"🏆 **Лучшая модель: {best_model['Model']}** 🏆\n")
        report_lines.append("Она показала наилучшее среднее время выполнения среди всех корректных реализаций.\n")
        report_lines.append(f"  - Среднее время: {best_model['Avg_Time (s)']:.6f} сек.")
        report_lines.append(f"  - Макс. память: {best_model['Max_Memory (KB)']} KB\n")
    else:
        report_lines.append("❌ **Лучшая модель не найдена.**\n")
        report_lines.append("Ни одна из протестированных моделей не смогла пройти полный набор тестов.\n")

    # --- 4. Топ N успешных моделей (таблица) ---
    if not successful_df.empty:
        report_lines.append("---")
        report_lines.append(f"## 4. Топ-{TOP_N_MODELS} успешных моделей (по времени выполнения)\n")
        
        # Форматируем для вывода
        top_n_df = successful_df.head(TOP_N_MODELS).copy()
        top_n_df['Avg_Time (s)'] = top_n_df['Avg_Time (s)'].map('{:,.6f}'.format)
        top_n_df['Max_Memory (KB)'] = top_n_df['Max_Memory (KB)'].map('{:,.0f}'.format)
        top_n_df.drop(columns=['Success', 'Result_Details'], inplace=True)
        top_n_df.reset_index(drop=True, inplace=True)
        top_n_df.index = top_n_df.index + 1
        top_n_df.index.name = "Rank"
        
        report_lines.append(top_n_df.to_string())
        report_lines.append("\n")
    
    # --- 5. Анализ ошибок (таблица) ---
    if not failed_df.empty:
        report_lines.append("---")
        report_lines.append("## 5. Анализ основных причин сбоев\n")
        
        error_df = pd.DataFrame(error_counts.items(), columns=['Причина сбоя', 'Кол-во моделей'])
        error_df = error_df.sort_values(by='Кол-во моделей', ascending=False)
        error_df.reset_index(drop=True, inplace=True)
        
        report_lines.append(error_df.to_string(index=False))
        report_lines.append("\n")

    report_lines.append("="*80)
    report_lines.append(f"Полный отчет сохранен в: {REPORT_FILENAME}")
    report_lines.append(f"Графики сохранены в: {PIE_CHART_FILENAME}, {BAR_CHART_FILENAME}")
    report_lines.append("="*80)

    # --- 6. Вывод в консоль и сохранение в файл ---
    report_text = "\n".join(report_lines)
    print(report_text)
    
    try:
        with open(REPORT_FILENAME, 'w', encoding='utf-8') as f:
            f.write(report_text)
    except Exception as e:
        print(f"Не удалось сохранить текстовый отчет: {e}")

    # --- 7. Генерация графиков ---
    
    # График 1: Круговая диаграмма (Success vs Fail)
    try:
        plt.figure(figsize=(8, 6))