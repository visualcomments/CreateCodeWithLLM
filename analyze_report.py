import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from tabulate import tabulate

def analyze_results(file_path):
    """
    Загружает и анализирует JSON-файл с результатами, 
    собирая данные для отчета и статистику по ошибкам.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Ошибка: Файл '{file_path}' не найден.")
        return None, None
    except json.JSONDecodeError:
        print(f"Ошибка: Не удалось декодировать JSON из файла '{file_path}'.")
        return None, None

    results = data.get('results', [])
    if not results:
        print("В файле не найдено поле 'results'.")
        return None, None

    report_data = []
    issue_counter = Counter()

    for result in results:
        model_name = result.get('model', 'N/A')
        final_test = result.get('final_test', {})
        success = final_test.get('success', False)
        
        if success:
            issue = "Success"
        else:
            issue = final_test.get('issue', 'Unknown Error')
            
        issue_counter[issue] += 1
        
        report_data.append({
            "Model": model_name,
            "Success": "Да" if success else "Нет",
            "Details": final_test.get('summary') if success else issue
        })

    return report_data, issue_counter

def save_summary_table(report_data, save_path='model_results_summary.txt'):
    """
    Сохраняет данные отчета в виде таблицы в файл.
    """
    if not report_data:
        print("Нет данных для отображения в таблице.")
        return
    
    # Создаем DataFrame для удобной работы
    df = pd.DataFrame(report_data)
    
    # Используем tabulate для красивого вывода
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("### Сводный отчет по результатам тестирования моделей ###\n")
        f.write(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
    print(f"Сводный отчет сохранен в файл: {save_path}")

def save_issue_summary(issue_counter, save_path='model_results_summary.png'):
    """
    Создает и сохраняет круговую диаграмму по причинам сбоев/успехов.
    """
    if not issue_counter:
        print("Нет данных для построения графика.")
        return

    labels = issue_counter.keys()
    sizes = issue_counter.values()

    # Создание графика
    plt.figure(figsize=(10, 7))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, 
            textprops={'fontsize': 10})
    
    plt.title('Распределение результатов тестирования (Успех / Причины сбоев)')
    plt.axis('equal')  # Делает диаграмму круглой
    
    # Сохранение файла
    plt.savefig(save_path)
    print(f"График с результатами сохранен в файл: {save_path}")

def main():
    """
    Главная функция скрипта.
    """
    file_path = 'final_results.json'  # Убедитесь, что файл в той же папке
    
    report_data, issue_counter = analyze_results(file_path)
    
    if report_data and issue_counter:
        # 1. Сохранить таблицу
        save_summary_table(report_data)
        
        # 2. Сохранить график
        save_issue_summary(issue_counter)

if __name__ == '__main__':
    main()
