import json
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from collections import Counter
import sys

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
            
            # Улучшение: убираем лишние переносы строк из ошибок
            if isinstance(issue, str):
                issue = issue.split('\n')[0]


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
        top_n_df['Max_Memory (KB)'] = top_n_df['Max_Memory (KB)'].map(lambda x: '{:,.0f}'.format(x) if pd.notnull(x) else 'N/A')
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
        
        report_lines.append(error_df.to_string(index=False, max_colwidth=80)) # Ограничим ширину колонки
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
        sizes = [success_count, fail_count]
        labels = [f'Успешно ({success_count})', f'С ошибкой ({fail_count})']
        colors = ['#4CAF50', '#F44336']
        explode = (0.1, 0) if success_count > 0 else (0, 0)

        plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=140)
        
        plt.title('Общие результаты бенчмарка (Успех vs. Ошибка)')
        plt.axis('equal') # Гарантирует, что диаграмма круглая
        
        plt.savefig(PIE_CHART_FILENAME)
        plt.close()
        print(f"Круговая диаграмма сохранена в {PIE_CHART_FILENAME}")
    except Exception as e:
        print(f"Не удалось создать круговую диаграмму: {e}")

    # График 2: Столбчатая диаграмма (Топ N по времени)
    try:
        if not successful_df.empty:
            top_n_plot = successful_df.head(TOP_N_MODELS).sort_values(by='Avg_Time (s)', ascending=True)
            
            plt.figure(figsize=(10, 7))
            
            # Используем .astype(str) для моделей, чтобы избежать проблем с категориальными данными
            models = top_n_plot['Model'].astype(str)
            times = top_n_plot['Avg_Time (s)']
            
            bars = plt.bar(models, times, color='skyblue')
            
            plt.xlabel('Модель')
            plt.ylabel('Среднее время (сек)')
            plt.title(f'Топ-{TOP_N_MODELS} моделей по производительности (Меньше = Лучше)')
            plt.xticks(rotation=45, ha='right')
            
            # Добавляем значения над столбцами
            plt.bar_label(bars, fmt='%.5f', padding=3)
            
            plt.tight_layout() # Подгоняем метки, чтобы они не вылезали
            
            plt.savefig(BAR_CHART_FILENAME)
            plt.close()
            print(f"Столбчатая диаграмма сохранена в {BAR_CHART_FILENAME}")
        else:
            print("Пропуск графика Top-N: нет успешных моделей.")
    except Exception as e:
        print(f"Не удалось создать столбчатую диаграмму: {e}")


def main():
    """
    Главная функция для запуска анализа.
    """
    print(f"Поиск файлов с результатами в папке: '{RESULTS_FOLDER}'...")
    
    # Проверка, которая приводила к ошибке в логах
    if not os.path.exists(RESULTS_FOLDER):
        print(f"Ошибка: Папка с результатами '{RESULTS_FOLDER}' не найдена.")
        print("Пожалуйста, сначала запустите 'test_code_LRX.py' для генерации результатов.")
        sys.exit(1) # Выход с кодом ошибки

    latest_file = find_latest_results_file(RESULTS_FOLDER)
    
    if not latest_file:
        print(f"Ошибка: Не найдены файлы 'final_results_*.json' в папке '{RESULTS_FOLDER}'.")
        sys.exit(1)

    print(f"Анализ последнего файла: {latest_file}")
    
    df = load_and_parse_data(latest_file)
    
    if not df.empty:
        generate_report(df)
    else:
        print("Данные не загружены (DataFrame пуст), генерация отчета пропущена.")

if __name__ == "__main__":
    main()