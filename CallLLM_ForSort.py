import requests
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import sys
from typing import Dict, List, Optional, Tuple
import time
from datetime import datetime
import random
import psutil
from time import perf_counter
import re
import tempfile
# Патч для RotatedProvider (используется в AnyProvider для ротации)
import g4f.providers.retry_provider as retry_mod  # Импорт модуля без кеширования класса
OriginalRotatedProvider = retry_mod.RotatedProvider  # Алиас оригинала для наследования

import g4f
from g4f import Provider

import threading
local = threading.local()

from g4f.errors import ModelNotFoundError
import queue

def clean_code(code: str) -> str:
    """
    Очищает код от markdown-оболочек типа ```python ... ``` и JSON-метаданных (OpenAI-like).
    Сначала проверяет JSON, извлекает content из choices[0].message.content если возможно.
    Если JSON не парсится, ищет markdown-блок в строке и извлекает его содержимое.
    Затем удаляет строки с ```python, ``` и лишние пустые строки.
    """
    original_len = len(code)
    
    # Шаг 1: Проверяем на JSON-оболочку (OpenAI-style)
    content_from_json = None
    try:
        data = json.loads(code)
        if isinstance(data, dict) and 'choices' in data and len(data['choices']) > 0:
            content = data['choices'][0].get('message', {}).get('content', '')
            if isinstance(content, str):
                content_from_json = content
                code = content  # Заменяем на content для дальнейшей чистки
    except (json.JSONDecodeError, KeyError, IndexError):
        pass  # Не JSON — продолжаем

    # Шаг 2: Если JSON не сработал или content_from_json пуст, ищем markdown-блок в оригинале
    if content_from_json is None or not content_from_json.strip():
        # Ищем первый блок ```python\n... (до следующего ``` или конца)
        match = re.search(r'```(?:python)?\s*\n(.*?)(?=\n?```\s*$|\Z)', code, re.DOTALL | re.MULTILINE)
        if match:
            code = match.group(1)
        # Альтернатива: если блок без закрывающего ``` (как в твоём логе), ищем от первого ``` до конца
        else:
            match = re.search(r'```(?:python)?\s*\n(.*)', code, re.DOTALL | re.MULTILINE)
            if match:
                code = match.group(1)

    # Шаг 3: Финальная чистка regex (на случай nested markdown)
    # Удаляем блок ```python в начале
    code = re.sub(r'^```(?:python)?\s*\n?', '', code, flags=re.MULTILINE)
    # Удаляем блок ``` в конце
    code = re.sub(r'\n?```\s*$', '', code, flags=re.MULTILINE)
    # Удаляем лишние пустые строки в начале и конце
    code = re.sub(r'^\n+', '', code, flags=re.MULTILINE)
    code = re.sub(r'\n+$', '\n', code, flags=re.MULTILINE)
    
    cleaned = code.strip()
    
    # Опциональный лог (раскомментируй для отладки)
    # if original_len - len(cleaned) > 100:
    #     print(f"Cleaned: {original_len} -> {len(cleaned)} chars removed")
    
    return cleaned

# Custom Rotated с трекингом (патчим только create_async_generator, логи в цикле)
class TrackedRotated(OriginalRotatedProvider):
    async def create_async_generator(self, model, messages, **kwargs):
        if not hasattr(local, 'current_data') or local.current_data is None:
            local.current_data = {'tried': [], 'errors': {}, 'success': None, 'model': model}
        current_data = local.current_data
        current_data['tried'] = []
        current_data['errors'] = {}
        current_data['success'] = None
        current_data['model'] = model
        if hasattr(local, 'current_model') and hasattr(local, 'current_queue') and self.providers:
            local.current_queue.put((local.current_model, 'log', f'1) Найдены провайдеры: {[p.__name__ for p in self.providers]}'))
            local.current_queue.put((local.current_model, 'log', f'Отладка: TrackedRotated вызван для модели {model}'))
        for provider_class in self.providers:
            p = None
            # Безопасное получение имени провайдера ДО try (для str/классов)
            if isinstance(provider_class, str):
                provider_name = provider_class
            else:
                provider_name = provider_class.__name__ if hasattr(provider_class, '__name__') else str(provider_class)
            current_data['tried'].append(provider_name)
            if hasattr(local, 'current_model') and hasattr(local, 'current_queue'):
                local.current_queue.put((local.current_model, 'log', f'2) Пробую {provider_name} with model: {model}'))
            try:
                # Если str, преобразуем в класс для инстанциации
                if isinstance(provider_class, str):
                    if hasattr(Provider, provider_class):
                        provider_class = getattr(Provider, provider_class)
                    else:
                        raise ValueError(f"Provider '{provider_name}' not found in Provider")
                p = provider_class()
                async for chunk in p.create_async_generator(model, messages, **kwargs):
                    yield chunk
                # Успех: put лог
                if hasattr(local, 'current_model') and hasattr(local, 'current_queue'):
                    local.current_queue.put((local.current_model, 'log', f'3) Успех от {provider_name}'))
                    current_data['success'] = provider_name
                return
            except Exception as e:
                error_str = str(e)
                if hasattr(local, 'current_model') and hasattr(local, 'current_queue'):
                    error_msg = f'3) Ошибка {provider_name}: {error_str}'
                    local.current_queue.put((local.current_model, 'log', error_msg))
                current_data['errors'][provider_name] = error_str
                if p:
                    if hasattr(p, '__del__'):
                        p.__del__()
                continue
        # Нет успеха: финальный лог
        try:
            if hasattr(local, 'current_model') and hasattr(local, 'current_queue'):
                local.current_queue.put((local.current_model, 'log', f'Отладка: TrackedRotated завершён, tried_providers={current_data["tried"]}'))
        except Exception:
            pass
        raise ModelNotFoundError(f"No working provider for model {model}", current_data['tried'])

# Monkey-patch: замени RotatedProvider на TrackedRotated (используется в AnyProvider)
retry_mod.RotatedProvider = TrackedRotated

# Патч на g4f.debug для записи в queue (без консоли, с JSON если нужно)
original_log = g4f.debug.log
original_error = g4f.debug.error

def patched_log(message, *args, **kwargs):
    message_str = str(message) if not isinstance(message, str) else message
    if hasattr(local, 'current_model') and hasattr(local, 'current_queue'):
        if 'AnyProvider: Using providers:' in message_str:
            providers_str = message_str.split('providers: ')[1].split(" for model")[0].strip("'")
            local.current_queue.put((local.current_model, 'log', f'1) Найдены провайдеры: [{providers_str}]'))
        elif 'Attempting provider:' in message_str:
            provider_str = message_str.split('provider: ')[1].strip()
            local.current_queue.put((local.current_model, 'log', f'2) Пробую {provider_str}'))


def patched_error(message, *args, **kwargs):
    message_str = str(message) if not isinstance(message, str) else message
    if hasattr(local, 'current_model') and hasattr(local, 'current_queue'):
        if 'failed:' in message_str:
            fail_str = message_str.split('failed: ')[1].strip()
            local.current_queue.put((local.current_model, 'log', f'3) Ошибка {fail_str}'))
        elif 'success' in message_str.lower():
            success_str = message_str.split('success: ')[1].strip() if 'success: ' in message_str else 'успех'
            local.current_queue.put((local.current_model, 'log', f'3) Успех {success_str}'))


g4f.debug.log = patched_log
g4f.debug.error = patched_error

CONFIG = {
    # Раздел с URL-адресами для загрузки данных о рабочих моделях
    'URLS': {
        # URL файла с результатами тестирования рабочих моделей g4f
        'WORKING_RESULTS': 'https://raw.githubusercontent.com/maruf009sultan/g4f-working/refs/heads/main/working/working_results.txt'
    },

    # Раздел с промптами для различных этапов взаимодействия с LLM
    'PROMPTS': {
        'INITIAL': r"""
    You are a Python programming assistant. Write a fully working Python module 
    for the following task:

    {task}

    ⚠️ Requirements:
    • Implement a function `solve(vector)` that sorts the vector using only adjacent swaps and circular swap `(n-1,0)`.
    • Return a tuple `(swaps, sorted_array)`, where `swaps` is a list of tuples and `sorted_array` is the result AFTER applying all swaps to a copy of vector.
    • Include `import json` and a CLI entry point: when executed, parse `sys.argv[1]` as JSON vector, fallback [3,1,2] if missing or invalid, and print only one JSON object `{{"swaps": swaps, "sorted_array": sorted_array}}`.
    • The JSON output must be structured and parseable (double quotes for keys).
    • Minimal example in `__main__` must use `solve([3,1,2])` and print JSON.
    • Fully self-contained and immediately executable.
    • Only code in the response, no explanations or Markdown.

    ⚠️ Critical constraints:
    • STRICTLY use only adjacent swaps swap(i, i+1) and circular swap swap(n-1,0).
    • No slicing, built-in sort, or creating new arrays — work IN-PLACE on `arr = vector[:]`.
    • **Track swaps made in each pass separately**: do not use the global swaps list to decide if a circular swap is needed — check only if the current pass made no swaps.
    • Append all swaps to the swaps list immediately after performing them.
    • Applying swaps sequentially to a copy of the input vector must yield a fully sorted ascending array.
    """,

        'FIX': r"""
    You are a Python debugging assistant. The following code did not work correctly. 
    Fix it to run correctly, follow the `solve(vector)` interface, and produce **only CLI JSON output**.

    Code:
    {code}

    Issue:
    {error}

    ⚠️ Requirements:
    • `solve(vector)` returns (swaps, sorted_array) after IN-PLACE swaps on copy.
    • CLI: `import json`; parse `sys.argv[1]` with fallback [3,1,2]; print only `json.dumps({{"swaps": swaps, "sorted_array": sorted_array}})`.
    • Use try-except to catch missing arguments or invalid JSON.
    • **Perform circular swap if and only if the current pass made no adjacent swaps**, not based on the total swaps list.
    • Self-contained, executable, immediately correct for vectors of length 4-20.
    • Only allowed swaps: swap(i, i+1) and swap(n-1,0); append (i,j) immediately after swap.
    • Only code in response, no extra prints or Markdown.
    """,

        'REFACTOR_NO_PREV': r"""
    You are an expert Python programmer. Refactor the following code:

    {code}

    ⚠️ Goals:
    • Improve readability, structure, efficiency and correctness.
    • Preserve `solve(vector)` interface: returns (swaps, sorted_array after applying swaps to copy(vector)).
    • CLI: parse `sys.argv[1]` as JSON with fallback [3,1,2], print only `json.dumps({{"swaps":..., "sorted_array":...}})`.
    • Minimal example in __main__ must print JSON only.
    • **Ensure circular swap is triggered correctly when the current pass has no adjacent swaps**.
    • Fully executable, immediately correct, passes verification for n=4-20.

    ⚠️ Constraint reminder:
    • STRICTLY use only swap(i, i+1) and swap(n-1,0); append (i,j) immediately after swap.
    • No slicing, built-in sort, or new arrays.
    • Only code in response, no explanations or Markdown.
    """,

        'REFACTOR': r"""
    You are an expert Python programmer. Compare the current and previous versions and perform a full refactor:

    Current code:
    {code}

    Previous version:
    {prev}

    ⚠️ Goals:
    • Improve readability, structure, efficiency, robustness.
    • Preserve `solve(vector)` interface: returns (swaps, sorted_array after applying swaps to copy(vector)).
    • CLI: parse `sys.argv[1]` as JSON with fallback [3,1,2]; print only `json.dumps({{"swaps":..., "sorted_array":...}})`.
    • Minimal example in __main__ must print JSON only.
    • **Circular swap must be performed if the current pass made no adjacent swaps**, not based on the total swaps list.
    • Code must pass verification: applying swaps to copy(input) == sorted(input) for all n=4-20.

    ⚠️ Constraint reminder:
    • STRICTLY use only swap(i, i+1) and swap(n-1,0); append (i,j) immediately after swap.
    • No slicing, built-in sort, or new arrays.
    • Only code in response, no explanations or Markdown.
    """
    },




    # Раздел с настройками ретраев для разных типов запросов
    'RETRIES': {
        # Настройки ретраев для начального запроса: максимум 1 ретрай, фактор задержки 1.0
        'INITIAL': {'max_retries': 1, 'backoff_factor': 1.0},
        # Настройки ретраев для исправлений: максимум 3 ретрая, фактор задержки 2.0 (экспоненциальный)
        'FIX': {'max_retries': 3, 'backoff_factor': 2.0}
    },

    # Раздел с константами системы
    'CONSTANTS': {
        # Разделитель в строках файла working_results.txt (между Provider|Model|Type)
        'DELIMITER_MODEL': '|',
        # Тип модели для фильтрации (только текстовые модели)
        'MODEL_TYPE_TEXT': 'text',
        # Таймаут для запросов к URL (в секундах)
        'REQUEST_TIMEOUT': 30,
        # Частота сохранения промежуточных результатов (каждые N моделей)
        'N_SAVE': 100,
        # Максимальное количество параллельных потоков для обработки моделей
        'MAX_WORKERS': 50,
        # Таймаут для выполнения кода в subprocess (в секундах)
        'EXEC_TIMEOUT': 5,
        # Сообщение об ошибке таймаута выполнения кода
        'ERROR_TIMEOUT': 'Timeout expired — the program likely entered an infinite loop. This usually happens when circular swap logic (arr[n-1] > arr[0]) is applied even after the array is already sorted, or if the input array is already sorted and the circular swap condition is not properly guarded.',
        # Сообщение об ошибке отсутствия ответа от модели
        'ERROR_NO_RESPONSE': 'No response from model',
        # Количество циклов рефакторинга в process_model
        'NUM_REFACTOR_LOOPS': 3,
        # Название папки для промежуточных и финальных результатов
        'INTERMEDIATE_FOLDER': 'промежуточные результаты'
    },

    # Раздел с именами этапов обработки для логов и статусов
    'STAGES': {
        # Этап генерации начального кода
        'INITIAL': 'первичный_ответ',
        # Этап исправления кода перед первым рефакторингом
        'FIX_INITIAL': 'исправление_до_рефакторинга',
        # Этап первого рефакторинга
        'REFACTOR_FIRST': 'ответ_от_рефакторинга',
        # Этап исправления после первого рефакторинга
        'FIX_AFTER_REFACTOR': 'исправление_после_рефакторинга',
        # Этап рефакторинга в цикле
        'REFACTOR': 'рефакторинг_в_цикле',
        # Этап исправления в цикле рефакторинга
        'FIX_LOOP': 'исправление_в_цикле'
    }
}

def get_models_list(config: Dict) -> List[str]:
    """
    Функция для формирования списка доступных моделей.

    Скачивает файл working_results.txt, парсит строки формата "Provider|Model|Type",
    извлекает модели только с типом 'text'. Дополняет моделями из g4f.models.Model.__all__().
    Удаляет дубликаты, возвращает уникальный список. Фильтрует только текстовые модели,
    исключая image/vision/audio/video модели и те, что содержат 'flux' (image gen).

    Args:
        config (Dict): Конфигурация с 'URLS', 'CONSTANTS' (DELIMITER_MODEL, MODEL_TYPE_TEXT, REQUEST_TIMEOUT).

    Returns:
        List[str]: Список уникальных имен текстовых моделей.

    Raises:
        requests.RequestException: Если ошибка при скачивании файла.
    """
    url_txt = config['URLS']['WORKING_RESULTS']
    try:
        resp = requests.get(url_txt, timeout=config['CONSTANTS']['REQUEST_TIMEOUT'])
        resp.raise_for_status()
        text = resp.text
    except Exception:
        text = ''
    working_models = set()
    for line in text.splitlines():
        if config['CONSTANTS']['DELIMITER_MODEL'] in line:
            parts = [p.strip() for p in line.split(config['CONSTANTS']['DELIMITER_MODEL'])]
            if len(parts) == 3 and parts[2] == config['CONSTANTS']['MODEL_TYPE_TEXT']:
                model_name = parts[1]
                # Дополнительный фильтр: исключаем flux и подобные
                if 'flux' not in model_name.lower():
                    working_models.add(model_name)
    
    # Из g4f.models: только базовые текстовые Model, исключая подклассы (Image, Vision и т.д.)
    try:
        from g4f.models import Model
        all_g4f_models = Model.__all__()
        g4f_models = set()
        for model_name in all_g4f_models:
            if 'flux' not in model_name.lower() and not any(sub in model_name.lower() for sub in ['image', 'vision', 'audio', 'video']):
                g4f_models.add(model_name)
    except ImportError:
        g4f_models = set()
    
    all_models = list(working_models.union(g4f_models))
    all_models = [m for m in all_models if m not in ['sldx-turbo', 'turbo']]
    return all_models



def test_code(code: str, config: Dict) -> Tuple[bool, str, Optional[Dict]]:
    """
    Тестирование кода с векторами длины 3-20, включая специальные кейсы для circular swap.

    Для каждого вектора: запускает subprocess с arg=json.dumps(vector),
    парсит JSON, применяет swaps, проверяет сортировку и валидность swap'ов.
    Измеряет время per test, max memory approx via psutil (cross-platform).

    Args:
        code (str): Тестируемый код.
        config (Dict): Конфиг с EXEC_TIMEOUT.

    Returns:
        Tuple[bool, str, Optional[Dict]]: (all_success, issue_str or 'All tests passed', summary)
        summary: {'all_success', 'total_time', 'max_memory_kb', 'tests': [list of test dicts], 'num_failing'}
    """
    test_results = []
    all_success = True
    failing_cases = []
    total_time = 0.0
    exec_timeout = config['CONSTANTS']['EXEC_TIMEOUT']
    process = None  # For memory tracking
    temp_file_path = None


    def _run_single_test(vector, temp_file_path, exec_timeout):
        n = len(vector)
        arg = json.dumps(vector)
        start_time = perf_counter()
        child_process = None
        try:
            child_process = subprocess.Popen(
                [sys.executable, temp_file_path, arg],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace',
                preexec_fn=None  # Windows-compatible
            )
            stdout, stderr = child_process.communicate(timeout=exec_timeout)
            elapsed = perf_counter() - start_time

            if child_process.returncode != 0:
                err = stderr or 'Unknown error'
                return {
                    'n': n,
                    'success': False,
                    'error': err,
                    'time': elapsed,
                    'swaps_len': None,
                    'input': vector,
                    'output_array': None,
                    'swaps': None,
                    'sorted_array': None
                }, err, False

            output = stdout.strip()
            if not output:
                err = 'No output'
                return {
                    'n': n,
                    'success': False,
                    'error': err,
                    'time': elapsed,
                    'swaps_len': None,
                    'input': vector,
                    'output_array': None,
                    'swaps': None,
                    'sorted_array': None
                }, err, False

            try:
                parsed = json.loads(output)
                swaps = parsed.get('swaps') or []  # Исправление: обработка None
                sorted_array = parsed.get('sorted_array') or []  # Аналогично для sorted_array

                # Verify swaps with robust handling
                current = vector[:]
                invalid_swaps = False
                applied_swaps = []  # Track applied swaps for report
                for swap in swaps:
                    try:
                        if not (isinstance(swap, (list, tuple)) and len(swap) == 2):
                            invalid_swaps = True
                            break
                        s1 = int(swap[0])
                        s2 = int(swap[1])
                        # Validate swap type: adjacent or circular
                        if not ((abs(s1 - s2) == 1 and 0 <= min(s1, s2) < max(s1, s2) < n) or
                                (min(s1, s2) == 0 and max(s1, s2) == n - 1)):
                            invalid_swaps = True
                            break
                        if 0 <= s1 < n and 0 <= s2 < n:
                            current[s1], current[s2] = current[s2], current[s1]
                            applied_swaps.append((s1, s2))
                        else:
                            invalid_swaps = True
                            break
                    except (ValueError, TypeError, IndexError):
                        invalid_swaps = True
                        break

                expected = sorted(vector)
                success = not invalid_swaps and current == expected and sorted_array == expected
                error_msg = None
                if not success:
                    if invalid_swaps:
                        error_msg = 'Invalid swaps'
                    elif current != expected:
                        error_msg = 'Mismatch after swaps'
                    elif sorted_array != expected:
                        error_msg = 'Sorted array mismatch'

                res_dict = {
                    'n': n,
                    'success': success,
                    'time': elapsed,
                    'swaps_len': len(swaps),
                    'applied_swaps': applied_swaps,
                    'input': vector,
                    'output_array': current,
                    'expected': expected,
                    'swaps': swaps,
                    'sorted_array': sorted_array
                }
                if error_msg:
                    res_dict['error'] = error_msg
                return res_dict, error_msg, success

            except json.JSONDecodeError as je:
                err = f'JSON parse error: {je}'
                return {
                    'n': n,
                    'success': False,
                    'error': err,
                    'time': elapsed,
                    'swaps_len': None,
                    'input': vector,
                    'output_array': None,
                    'swaps': None,
                    'sorted_array': None
                }, err, False

        except subprocess.TimeoutExpired:
            err = config['CONSTANTS']['ERROR_TIMEOUT']
            if child_process:
                child_process.kill()
            return {
                'n': n,
                'success': False,
                'error': err,
                'time': 0.0,
                'swaps_len': None,
                'input': vector,
                'output_array': None,
                'swaps': None,
                'sorted_array': None
            }, err, False
        finally:
            if child_process:
                child_process.wait()


    try:
        # Create temporary file for the code once
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name

        # Start memory tracking process (parent)
        process = psutil.Process()
        start_mem = process.memory_info().rss / 1024  # KB
    except (ImportError, Exception):
        start_mem = None

    # Define specific test vectors including cases where first > last
    specific_vectors = [
        [3, 1, 2],             # n=3, first=3 > 2=last
        [1, 2, 0],             # n=3, needs rotation
        [5, 2, 4, 1, 3],       # n=5, first=5 > 3=last
        [0, 3, 31, 0],         # n=4
        [6, 27, 49, 39, 40],   # n=5
        [51, 3, 22, 22, 39, 25],  # n=6, first=51 > 25=last
        [48, 18, 44, 20, 16, 61, 26],  # n=7, first=48 > 26=last
        [2,3,1],
    ]

    # All vectors: specific + random for n=4 to 20
    all_vectors = specific_vectors + [
        [random.randint(0, n * 10) for _ in range(n)] for n in range(4, 21)
    ]

    for vector in all_vectors:
        res_dict, err_msg, single_success = _run_single_test(vector, temp_file_path, exec_timeout)
        total_time += res_dict['time']
        test_results.append(res_dict)

        if not single_success:
            all_success = False
            failing_case = {
                'n': res_dict['n'],
                'input': res_dict['input'],
                'error': res_dict.get('error', 'Unknown failure'),
                'swaps': res_dict.get('swaps', []),
                'applied_swaps': res_dict.get('applied_swaps', []),
                'result_array': res_dict.get('output_array', None),
                'expected': res_dict.get('expected', None),
                'sorted_array': res_dict.get('sorted_array', None)
            }
            failing_cases.append(failing_case)

    # Clean up temp file
    if temp_file_path and os.path.exists(temp_file_path):
        os.unlink(temp_file_path)

    # End memory tracking
    try:
        if start_mem is not None and process:
            end_mem = process.memory_info().rss / 1024  # KB
            max_memory_kb = end_mem - start_mem
        else:
            max_memory_kb = None
    except Exception:
        max_memory_kb = None

    num_tests = len(test_results)
    summary = {
        'all_success': all_success,
        'total_time': total_time,
        'avg_time': total_time / num_tests if num_tests > 0 else 0,
        'max_memory_kb': max_memory_kb,
        'tests': test_results,
        'num_failing': len(failing_cases)
    }

    if all_success:
        return True, 'All tests passed', summary
    else:
        issue_str = json.dumps({'failing_cases': failing_cases}, ensure_ascii=False, indent=2)
        return False, issue_str, summary



def llm_query(model: str, prompt: str, retries_config: Dict, config: Dict, progress_queue: queue.Queue, stage: str = None) -> Optional[str]:
    # Инициализация local только для патча (чтобы он мог писать в очередь)
    local.current_model = model
    local.current_queue = progress_queue
    local.current_data = {'tried': [], 'errors': {}, 'success': None, 'model': model}
    local.current_stage = stage

    request_timeout = config['CONSTANTS']['REQUEST_TIMEOUT']

    # AnyProvider: простой вызов с retries
    for attempt in range(retries_config['max_retries'] + 1):
        try:
            response = g4f.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                provider=Provider.AnyProvider,
                timeout=request_timeout
            )
            if response and response.strip():
                return response.strip()
        except ModelNotFoundError as e:
            if len(e.args) > 1:
                local.current_data['tried'] = e.args[1]
            return None
        except Exception:
            pass
        if attempt < retries_config['max_retries']:
            time.sleep(retries_config['backoff_factor'] * (2 ** attempt))

    return None

def process_model(model: str, task: str, config: Dict, progress_queue: queue.Queue) -> Dict:
    """
    Обработка одной модели: последовательность запросов LLM, test, fix, refactor.

    1. Initial промпт -> код.
    2. Test, if fail: fix -> код.
    3. First refactor (no prev) -> код.
    4. Test, if fail: fix -> код.
    5. 3 раза: refactor (with prev) -> код; test, if fail: fix -> код.
    6. Final test on last code.
    Если на любом LLM шаге ошибка -> append с error, continue/return early если critical.
    Использует успешный провайдер из предыдущих вызовов для приоритета.

    Args:
        model (str): Имя модели.
        task (str): Исходная задача.
        config (Dict): Конфигурация с 'PROMPTS', 'RETRIES', 'CONSTANTS', 'STAGES'.
        progress_queue (queue.Queue): Очередь для отправки обновлений прогресса.

    Returns:
        Dict: {'model': str, 'iterations': List[Dict], 'final_code': str|None, 'final_test': Dict}
        Где iteration: {'providers_tried': List[str], 'success_provider': str|None, 'stage': str, 'response': str|None, 'error': str|None, 'test_summary': Dict|None}
        final_test: {'success': bool, 'summary': Dict|None, 'issue': str|None}
    """
    iterations = []
    current_code = None
    prev_code = None
    early_stop = False
    total_stages = 1 + 1 + 1 + 1 + config['CONSTANTS']['NUM_REFACTOR_LOOPS'] * 2 + 1  # + final
    current_stage = 0
    progress_queue.put((model, 'status', f'Начало обработки: {config["STAGES"]["INITIAL"]}'))
    progress_queue.put((model, 'log', f'=== НАЧАЛО ОБРАБОТКИ МОДЕЛИ: {model} ==='))
    # Initial query
    prompt = config['PROMPTS']['INITIAL'].format(task=task)
    progress_queue.put((model, 'log', f'Этап: {config["STAGES"]["INITIAL"]}. Полный промпт:\n{prompt}'))
    progress_queue.put((model, 'log', f'Вызов llm_query с retries: {config["RETRIES"]["INITIAL"]}'))
    response = llm_query(model, prompt, config['RETRIES']['INITIAL'], config, progress_queue, config['STAGES']['INITIAL'])
    current_stage += 1
    progress_queue.put((model, 'progress', (current_stage, total_stages)))
    success_p = local.current_data['success']
    tried = local.current_data['tried']
    if response:
        cleaned_response = clean_code(response)
        progress_queue.put((model, 'log', f'Получен ответ (длина: {len(response)}), очищенный (длина: {len(cleaned_response)}):\n{cleaned_response}'))
        current_code = cleaned_response
    else:
        error_msg = config['CONSTANTS']['ERROR_NO_RESPONSE']
        progress_queue.put((model, 'log', f'Ошибка llm_query: {error_msg}'))
        iter_entry = {
            'providers_tried': tried,
            'success_provider': None,
            'stage': config['STAGES']['INITIAL'],
            'response': None,
            'error': error_msg,
            'test_summary': None
        }
        iterations.append(iter_entry)
        progress_queue.put((model, 'status', f'Ошибка на этапе: {config["STAGES"]["INITIAL"]}'))
        return {'model': model, 'iterations': iterations, 'final_code': None, 'final_test': {'success': False, 'issue': error_msg, 'summary': None}}
    iter_entry = {
        'providers_tried': tried,
        'success_provider': success_p,
        'stage': config['STAGES']['INITIAL'],
        'response': current_code,
        'error': None,
        'test_summary': None
    }
    iterations.append(iter_entry)
    progress_queue.put((model, 'status', f'Получен первичный код'))
    # Test initial code and fix if needed
    progress_queue.put((model, 'status', 'Тестирование первичного кода...'))
    progress_queue.put((model, 'log', f'Этап: Тестирование первичного кода (длина: {len(current_code)})'))
    test_success, issue_str, test_summary = test_code(current_code, config)
    current_stage += 1
    progress_queue.put((model, 'progress', (current_stage, total_stages)))
    progress_queue.put((model, 'log', f'Результат test_code: success={test_success}, num_failing={test_summary["num_failing"] if test_summary else 0}'))
    iter_entry['test_summary'] = test_summary
    if test_success:
        progress_queue.put((model, 'status', f'Тесты первичного кода: OK. Ранний стоп.'))
        early_stop = True
    else:
        fix_stage = config['STAGES']['FIX_INITIAL']
        progress_queue.put((model, 'log', f'Тесты провалены. Этап исправления: {fix_stage}'))
        prompt = config['PROMPTS']['FIX'].format(code=current_code, error=issue_str)
        progress_queue.put((model, 'log', f'Промпт для исправления (полный):\n{prompt}'))
        response = llm_query(model, prompt, config['RETRIES']['FIX'], config, progress_queue, fix_stage)
        current_stage += 1
        progress_queue.put((model, 'progress', (current_stage, total_stages)))
        success_p = local.current_data['success']
        tried = local.current_data['tried']
        if response:
            cleaned_response = clean_code(response)
            progress_queue.put((model, 'log', f'Получен исправленный код (длина: {len(response)}), очищенный (длина: {len(cleaned_response)}):\n{cleaned_response}'))
            current_code = cleaned_response
            error_fix = None
        else:
            error_fix = config['CONSTANTS']['ERROR_NO_RESPONSE']
            progress_queue.put((model, 'log', f'Ошибка llm_query для fix: {error_fix}'))
            fix_iter = {
                'providers_tried': tried,
                'success_provider': None,
                'stage': fix_stage,
                'response': None,
                'error': error_fix,
                'test_summary': None
            }
            iterations.append(fix_iter)
            progress_queue.put((model, 'status', f'Ошибка исправления: {fix_stage}'))
            return {'model': model, 'iterations': iterations, 'final_code': None, 'final_test': {'success': False, 'issue': error_fix, 'summary': None}}
        fix_iter = {
            'providers_tried': tried,
            'success_provider': success_p,
            'stage': fix_stage,
            'response': current_code,
            'error': error_fix,
            'test_summary': None  # No retest after fix
        }
        iterations.append(fix_iter)
        progress_queue.put((model, 'status', f'Код исправлен до рефакторинга'))
    # First refactor (no prev)
    if not early_stop:
        progress_queue.put((model, 'log', f'Этап: Первый рефакторинг {config["STAGES"]["REFACTOR_FIRST"]}'))
        prompt = config['PROMPTS']['REFACTOR_NO_PREV'].format(code=current_code)
        progress_queue.put((model, 'log', f'Промпт для рефакторинга (полный):\n{prompt}'))
        response = llm_query(model, prompt, config['RETRIES']['FIX'], config, progress_queue, config['STAGES']['REFACTOR_FIRST'])
        current_stage += 1
        progress_queue.put((model, 'progress', (current_stage, total_stages)))
        success_p = local.current_data['success']
        tried = local.current_data['tried']
        refactor_stage = config['STAGES']['REFACTOR_FIRST']
        if response:
            cleaned_response = clean_code(response)
            progress_queue.put((model, 'log', f'Получен рефакторированный код (длина: {len(response)}), очищенный (длина: {len(cleaned_response)}):\n{cleaned_response}'))
            prev_code = current_code
            current_code = cleaned_response
            error_ref = None
        else:
            error_ref = config['CONSTANTS']['ERROR_NO_RESPONSE']
            progress_queue.put((model, 'log', f'Ошибка llm_query для refactor: {error_ref}'))
            iter_entry = {
                'providers_tried': tried,
                'success_provider': None,
                'stage': refactor_stage,
                'response': None,
                'error': error_ref,
                'test_summary': None
            }
            iterations.append(iter_entry)
            progress_queue.put((model, 'status', f'Ошибка рефакторинга: {refactor_stage}'))
            return {'model': model, 'iterations': iterations, 'final_code': current_code, 'final_test': {'success': False, 'issue': error_ref, 'summary': None}}
        iter_entry = {
            'providers_tried': tried,
            'success_provider': success_p,
            'stage': refactor_stage,
            'response': current_code,
            'error': error_ref,
            'test_summary': None
        }
        iterations.append(iter_entry)
        progress_queue.put((model, 'status', 'Первый рефакторинг завершен'))
        # Test after first refactor and fix if needed
        progress_queue.put((model, 'status', 'Тестирование после первого рефакторинга...'))
        progress_queue.put((model, 'log', f'Этап: Тестирование после рефакторинга (длина: {len(current_code)})'))
        test_success, issue_str, test_summary = test_code(current_code, config)
        current_stage += 1
        progress_queue.put((model, 'progress', (current_stage, total_stages)))
        progress_queue.put((model, 'log', f'Результат test_code после refactor: success={test_success}, num_failing={test_summary["num_failing"] if test_summary else 0}'))
        iter_entry['test_summary'] = test_summary
        if test_success:
            progress_queue.put((model, 'status', f'Тесты после рефакторинга: OK. Ранний стоп.'))
            early_stop = True
        else:
            fix_stage = config['STAGES']['FIX_AFTER_REFACTOR']
            progress_queue.put((model, 'log', f'Тесты провалены после рефакторинга. Этап исправления: {fix_stage}'))
            prompt = config['PROMPTS']['FIX'].format(code=current_code, error=issue_str)
            progress_queue.put((model, 'log', f'Промпт для исправления (полный):\n{prompt}'))
            response = llm_query(model, prompt, config['RETRIES']['FIX'], config, progress_queue, fix_stage)
            current_stage += 1
            progress_queue.put((model, 'progress', (current_stage, total_stages)))
            success_p = local.current_data['success']
            tried = local.current_data['tried']
            if response:
                cleaned_response = clean_code(response)
                progress_queue.put((model, 'log', f'Получен исправленный код после refactor (длина: {len(response)}), очищенный (длина: {len(cleaned_response)}):\n{cleaned_response}'))
                current_code = cleaned_response
                error_fix = None
            else:
                error_fix = config['CONSTANTS']['ERROR_NO_RESPONSE']
                progress_queue.put((model, 'log', f'Ошибка llm_query для fix после refactor: {error_fix}'))
                fix_iter = {
                    'providers_tried': tried,
                    'success_provider': None,
                    'stage': fix_stage,
                    'response': None,
                    'error': error_fix,
                    'test_summary': None
                }
                iterations.append(fix_iter)
                progress_queue.put((model, 'status', f'Ошибка: {fix_stage}'))
                return {'model': model, 'iterations': iterations, 'final_code': None, 'final_test': {'success': False, 'issue': error_fix, 'summary': None}}
            fix_iter = {
                'providers_tried': tried,
                'success_provider': success_p,
                'stage': fix_stage,
                'response': current_code,
                'error': error_fix,
                'test_summary': None  # No retest
            }
            iterations.append(fix_iter)
            progress_queue.put((model, 'status', 'Код исправлен после рефакторинга'))
    else:
        progress_queue.put((model, 'status', f'Пропуск рефакторинга (ранний стоп)'))
    # Loop 3 times: refactor with prev, then test + fix if fail
    if not early_stop:
        loops_left = config['CONSTANTS']['NUM_REFACTOR_LOOPS']
        for loop in range(config['CONSTANTS']['NUM_REFACTOR_LOOPS']):
            if early_stop:
                break
            loops_left -= 1
            progress_queue.put((model, 'status', f'Цикл рефакторинга {loop+1}/{config["CONSTANTS"]["NUM_REFACTOR_LOOPS"]}, осталось: {loops_left}'))
            progress_queue.put((model, 'log', f'=== ЦИКЛ РЕФАКТОРИНГА {loop+1} ==='))
            # Refactor with prev
            prompt = config['PROMPTS']['REFACTOR'].format(code=current_code, prev=prev_code)
            progress_queue.put((model, 'log', f'Этап: Рефакторинг в цикле {config["STAGES"]["REFACTOR"]}. Промпт (полный):\n{prompt}'))
            response = llm_query(model, prompt, config['RETRIES']['FIX'], config, progress_queue, config['STAGES']['REFACTOR'])
            current_stage += 1
            progress_queue.put((model, 'progress', (current_stage, total_stages)))
            success_p = local.current_data['success']
            tried = local.current_data['tried']
            refactor_stage = config['STAGES']['REFACTOR']
            if response:
                cleaned_response = clean_code(response)
                progress_queue.put((model, 'log', f'Получен рефакторированный код в цикле (длина: {len(response)}), очищенный (длина: {len(cleaned_response)}):\n{cleaned_response}'))
                prev_code = current_code
                current_code = cleaned_response
                error_ref = None
            else:
                error_ref = config['CONSTANTS']['ERROR_NO_RESPONSE']
                progress_queue.put((model, 'log', f'Ошибка llm_query для refactor в цикле: {error_ref}'))
                iter_entry = {
                    'providers_tried': tried,
                    'success_provider': None,
                    'stage': refactor_stage,
                    'response': None,
                    'error': error_ref,
                    'test_summary': None
                }
                iterations.append(iter_entry)
                progress_queue.put((model, 'status', f'Ошибка рефакторинга в цикле'))
                continue  # skip test if no refactor
            iter_entry = {
                'providers_tried': tried,
                'success_provider': success_p,
                'stage': refactor_stage,
                'response': current_code,
                'error': error_ref,
                'test_summary': None
            }
            iterations.append(iter_entry)
            # Test and fix if needed
            progress_queue.put((model, 'status', 'Тестирование в цикле...'))
            progress_queue.put((model, 'log', f'Этап: Тестирование в цикле (длина: {len(current_code)})'))
            test_success, issue_str, test_summary = test_code(current_code, config)
            current_stage += 1
            progress_queue.put((model, 'progress', (current_stage, total_stages)))
            progress_queue.put((model, 'log', f'Результат test_code в цикле: success={test_success}, num_failing={test_summary["num_failing"] if test_summary else 0}'))
            iter_entry['test_summary'] = test_summary
            if test_success:
                progress_queue.put((model, 'status', f'Тесты в цикле: OK. Ранний стоп.'))
                early_stop = True
            else:
                fix_stage = config['STAGES']['FIX_LOOP']
                progress_queue.put((model, 'log', f'Тесты провалены в цикле. Этап исправления: {fix_stage}'))
                prompt = config['PROMPTS']['FIX'].format(code=current_code, error=issue_str)
                progress_queue.put((model, 'log', f'Промпт для исправления в цикле (полный):\n{prompt}'))
                response = llm_query(model, prompt, config['RETRIES']['FIX'], config, progress_queue, fix_stage)
                current_stage += 1
                progress_queue.put((model, 'progress', (current_stage, total_stages)))
                success_p = local.current_data['success']
                tried = local.current_data['tried']
                if response:
                    cleaned_response = clean_code(response)
                    progress_queue.put((model, 'log', f'Получен исправленный код в цикле (длина: {len(response)}), очищенный (длина: {len(cleaned_response)}):\n{cleaned_response}'))
                    current_code = cleaned_response
                    error_fix = None
                else:
                    error_fix = config['CONSTANTS']['ERROR_NO_RESPONSE']
                    progress_queue.put((model, 'log', f'Ошибка llm_query для fix в цикле: {error_fix}'))
                    fix_iter = {
                        'providers_tried': tried,
                        'success_provider': None,
                        'stage': fix_stage,
                        'response': None,
                        'error': error_fix,
                        'test_summary': None
                    }
                    iterations.append(fix_iter)
                    progress_queue.put((model, 'status', f'Ошибка исправления в цикле'))
                    continue
                fix_iter = {
                    'providers_tried': tried,
                    'success_provider': success_p,
                    'stage': fix_stage,
                    'response': current_code,
                    'error': error_fix,
                    'test_summary': None  # No retest
                }
                iterations.append(fix_iter)
                progress_queue.put((model, 'status', 'Код исправлен в цикле'))
    else:
        progress_queue.put((model, 'status', f'Пропуск цикла рефакторинга (ранний стоп)'))
    # Final test
    progress_queue.put((model, 'status', 'Финальное тестирование...'))
    final_success, final_issue, final_summary = test_code(current_code, config)
    current_stage += 1
    progress_queue.put((model, 'progress', (current_stage, total_stages)))
    progress_queue.put((model, 'log', f'Финальное тестирование: success={final_success}, num_failing={final_summary["num_failing"] if final_summary else 0}'))
    
    # Detailed report for final test if successful
    if final_success and final_summary:
        tests = final_summary['tests']
        table_header = "n | Input Vector | Swaps Count | Time (s) | Applied Swaps Seq. | Output Array"
        table_separator = "-" * 80
        table_lines = [table_header, table_separator]
        for n in range(4, 21):
            t = tests[n]
            inp_str = str(t['input'])
            out_str = str(t['output_array'])
            swaps_seq_str = str(t['applied_swaps'])[:50] + '...' if len(str(t['applied_swaps'])) > 50 else str(t['applied_swaps'])
            line = f"{n} | {inp_str} | {t['swaps_len']} | {t['time']:.3f} | {swaps_seq_str} | {out_str}"
            table_lines.append(line)
        table = '\n'.join(table_lines)
        progress_queue.put((model, 'log', f"=== ОТЧЁТ ПО ФИНАЛЬНОМУ ТЕСТИРОВАНИЮ ==="))
        progress_queue.put((model, 'log', table))
        progress_queue.put((model, 'log', f"Общее время тестов: {final_summary['total_time']:.2f} с"))
        progress_queue.put((model, 'log', f"Максимальное потребление памяти (примерно): {final_summary['max_memory_kb']:.0f} КБ" if final_summary['max_memory_kb'] else "Память: N/A"))
    
    final_test = {
        'success': final_success,
        'summary': final_summary,
        'issue': final_issue if not final_success else None
    }
    progress_queue.put((model, 'status', f'Завершено, финальные тесты: {"OK" if final_success else "FAIL"}'))
    progress_queue.put((model, 'log', f'=== ФИНАЛЬНЫЙ КОД (длина: {len(current_code or "")}):\n{current_code or "None"}'))
    progress_queue.put((model, 'log', f'=== КОНЕЦ ОБРАБОТКИ МОДЕЛИ: {model} ==='))
    progress_queue.put((model, 'done', None))
    return {
        'model': model,
        'iterations': iterations,
        'final_code': current_code,
        'final_test': final_test
    }

def orchestrator(task: str, models: List[str], config: Dict, progress_queue: queue.Queue) -> Dict:
    folder = config['CONSTANTS']['INTERMEDIATE_FOLDER']
    os.makedirs(folder, exist_ok=True)
    results = {}
    total_models = len(models)
    with ThreadPoolExecutor(max_workers=config['CONSTANTS']['MAX_WORKERS']) as executor:
        future_to_model = {executor.submit(process_model, model, task, config, progress_queue): model for model in models}
        completed = 0
        for future in as_completed(future_to_model):
            model = future_to_model[future]
            try:
                result = future.result()
                results[result['model']] = result
            except Exception as e:
                results[model] = {
                    'model': model,
                    'iterations': [],
                    'final_code': None,
                    'final_test': {'success': False, 'issue': str(e), 'summary': None},
                    'error': str(e)
                }
                progress_queue.put((model, 'error', str(e)))
            completed += 1
            # Отправляем обновление состояния results в очередь для GUI
            results_summary = {
                'completed': completed,
                'total': total_models,
                'models_done': list(results.keys()),
                'num_iterations_total': sum(len(r.get('iterations', [])) for r in results.values()),
                'num_with_final_code': sum(1 for r in results.values() if r.get('final_code') is not None),
                'num_final_tests_ok': sum(1 for r in results.values() if r.get('final_test', {}).get('success', False)),
                'partial_results': list(results.values()),
            }
            progress_queue.put(('global', 'results_update', results_summary))
            remaining = total_models - completed
            if completed % config['CONSTANTS']['N_SAVE'] == 0:
                partial_file = os.path.join(folder, f'batch_{completed // config["CONSTANTS"]["N_SAVE"]}.json')
                with open(partial_file, 'w', encoding='utf-8') as f:
                    json.dump({'partial_results': list(results.values())}, f, ensure_ascii=False, indent=2)
                progress_queue.put(('global', 'save', f'Сохранен batch {completed // config["CONSTANTS"]["N_SAVE"]}. Осталось обработать: {remaining} моделей'))
    final_results = {
        'results': list(results.values()),
        'timestamp': datetime.now().isoformat()
    }
    final_file = os.path.join(folder, 'final_results.json')
    with open(final_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    progress_queue.put(('global', 'done', None))
    return final_results

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

class ProgressGUI:
    def __init__(self, task: str, models: List[str], config: Dict):
        self.root = tk.Tk()
        self.root.title("Мониторинг прогресса моделей")
        self.root.geometry("1200x800")
        self.root.configure(bg='lightgray')
        
        self.config = config
        self.task = task
        self.models = models
        self.results = {}
        self.logs = {model: [] for model in models}
        self.progress_queue = queue.Queue()
        self.log_texts = {}  # Для хранения Text виджетов по моделям
        
        # Глобальный фрейм для Treeview и результатов
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Фрейм для Treeview
        tree_frame = ttk.Frame(main_frame)
        tree_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Treeview для списка моделей
        self.tree = ttk.Treeview(tree_frame, columns=('No', 'Model', 'Status', 'Progress'), show='headings', height=20)
        self.tree.heading('No', text='№')
        self.tree.heading('Model', text='Модель')
        self.tree.heading('Status', text='Статус')
        self.tree.heading('Progress', text='Прогресс')
        self.tree.column('No', width=50, anchor='center')
        self.tree.column('Model', width=250)
        self.tree.column('Status', width=400)
        self.tree.column('Progress', width=100, anchor='center')
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Скроллбар для Treeview
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # Контекстное меню для Treeview
        self.context_menu = tk.Menu(self.root, tearoff=0)
        self.context_menu.add_command(label="Копировать модель", command=self.copy_model)
        self.context_menu.add_command(label="Копировать статус", command=self.copy_status)
        self.context_menu.add_command(label="Копировать прогресс", command=self.copy_progress)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Копировать всю строку", command=self.copy_row)
        self.tree.bind("<Button-3>", self.show_context_menu)
        
        # Фрейм для глобального статуса результатов
        results_frame = ttk.LabelFrame(main_frame, text="Текущее состояние results", padding=10)
        results_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # ScrolledText для отображения текущего состояния results
        self.results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, font=('Consolas', 9), height=20, bg='black', fg='white', insertbackground='white')
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Кнопки для копирования результатов
        results_btn_frame = tk.Frame(results_frame, bg='lightgray')
        results_btn_frame.pack(fill=tk.X, pady=(5, 0))
        copy_results_btn = tk.Button(results_btn_frame, text="Копировать текущее состояние", command=self.copy_current_results, bg='lightblue', relief='raised')
        copy_results_btn.pack(side=tk.LEFT)
        clear_results_btn = tk.Button(results_btn_frame, text="Очистить", command=self.clear_results_text, bg='lightcoral', relief='raised')
        clear_results_btn.pack(side=tk.RIGHT)
        
        # Кнопка для показа лога
        btn_frame = tk.Frame(self.root, bg='lightgray')
        btn_frame.pack(fill=tk.X, padx=10, pady=5)
        self.show_log_btn = tk.Button(btn_frame, text="Показать лог модели", command=self.show_log, bg='white', relief='raised')
        self.show_log_btn.pack(side=tk.LEFT)
        
        # Инициализация строк
        for i, model in enumerate(models, 1):
            self.tree.insert('', 'end', iid=model, values=(i, model, 'Ожидание...', '0%'))
        
        # Запуск оркестратора в отдельном потоке
        self.executor_thread = threading.Thread(target=self.run_orchestrator)
        self.executor_thread.daemon = True
        self.executor_thread.start()
        
        # Обновление UI
        self.update_ui()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def safe_model_name(self, model):
        """Санитизирует имя модели для использования в имени файла."""
        return model.replace('/', '_').replace('\\', '_')
    
    def save_model_logs(self, model):
        """Сохраняет логи модели в файл и очищает из RAM."""
        if model in self.logs and self.logs[model]:
            log_dir = os.path.join(self.config['CONSTANTS']['INTERMEDIATE_FOLDER'], 'logs')
            os.makedirs(log_dir, exist_ok=True)
            safe_model = self.safe_model_name(model)
            log_file = os.path.join(log_dir, f'{safe_model}.log')
            try:
                with open(log_file, 'w', encoding='utf-8') as f:
                    f.write('\n\n'.join(self.logs[model]))
                # Очистка из RAM
                del self.logs[model]
                print(f"Логи для {model} сохранены в {log_file} и удалены из RAM")
            except Exception as e:
                print(f"Ошибка сохранения логов для {model}: {e}")
    
    def show_context_menu(self, event):
        self.context_menu.post(event.x_root, event.y_root)
    
    def copy_model(self):
        selected = self.tree.selection()
        if selected:
            model = self.tree.item(selected[0])['values'][1]
            self.root.clipboard_clear()
            self.root.clipboard_append(str(model))
    
    def copy_status(self):
        selected = self.tree.selection()
        if selected:
            status = self.tree.item(selected[0])['values'][2]
            self.root.clipboard_clear()
            self.root.clipboard_append(str(status))
    
    def copy_progress(self):
        selected = self.tree.selection()
        if selected:
            progress = self.tree.item(selected[0])['values'][3]
            self.root.clipboard_clear()
            self.root.clipboard_append(str(progress))
    
    def copy_row(self):
        selected = self.tree.selection()
        if selected:
            values = self.tree.item(selected[0])['values']
            row_text = ' | '.join(map(str, values))
            self.root.clipboard_clear()
            self.root.clipboard_append(row_text)
    
    def run_orchestrator(self):
        self.results = orchestrator(self.task, self.models, self.config, self.progress_queue)
    
    def update_ui(self):
        try:
            while True:
                msg = self.progress_queue.get_nowait()
                model, msg_type, data = msg
                if msg_type == 'status':
                    self.tree.set(model, 'Status', data)
                elif msg_type == 'progress':
                    current, total = data
                    perc = int((current / total) * 100)
                    self.tree.set(model, 'Progress', f'{perc}%')
                elif msg_type == 'log':
                    if model in self.logs:
                        self.logs[model].append(data)
                    # Обновление в открытом лог-окне, если оно существует
                    if model in self.log_texts and model + '_text' in self.log_texts:
                        text = self.log_texts[model + '_text']
                        line = data + '\n'
                        if data.startswith('=== '):
                            text.insert(tk.END, line, "header")
                        elif data.startswith('Этап: '):
                            text.insert(tk.END, line, "stage")
                        elif 'Получен' in data and 'код' in data and ':' in data:
                            text.insert(tk.END, line, "code")
                        elif 'Ошибка' in data or 'FAIL' in data:
                            text.insert(tk.END, line, "error")
                        else:
                            text.insert(tk.END, line, "normal")
                        text.see(tk.END)  # Автопрокрутка к концу
                elif msg_type == 'done':
                    if model == 'global':
                        messagebox.showinfo("Завершено", "Все модели обработаны!")
                    else:
                        self.tree.set(model, 'Status', 'Завершено')
                        # Сохранить логи и очистить из RAM после завершения модели
                        self.save_model_logs(model)
                elif msg_type == 'error':
                    self.tree.set(model, 'Status', f'Ошибка: {data}')
                    # Сохранить логи и очистить из RAM при ошибке
                    self.save_model_logs(model)
                elif model == 'global':
                    if msg_type == 'save':
                        print(data)  # Выводим в консоль только сообщения о сохранении батча
                    elif msg_type == 'results_update':
                        self.update_results_display(data)
        except queue.Empty:
            pass
    
        self.root.after(100, self.update_ui)

    def update_results_display(self, summary: Dict):
        """Обновляет отображение текущего состояния results в ScrolledText."""
        partial_results = summary['partial_results']
        completed = summary['completed']
        total = summary['total']
        
        # Агрегированные метрики времени и памяти
        total_time = 0.0
        total_memory = 0.0
        num_with_summary = 0
        for r in partial_results:
            final_test = r.get('final_test', {})
            s = final_test.get('summary')
            if s:
                num_with_summary += 1
                total_time += s.get('total_time', 0)
                mem = s.get('max_memory_kb', 0)
                if mem:
                    total_memory += mem
        
        avg_time = total_time / num_with_summary if num_with_summary > 0 else 0
        avg_memory = total_memory / num_with_summary if num_with_summary > 0 else 0
        
        display_text = f"Обновление results ({datetime.now().strftime('%H:%M:%S')}):\n"
        display_text += f"Завершено моделей: {completed}/{total}\n"
        display_text += f"Обработано моделей: {len(summary['models_done'])}\n"
        display_text += f"Общее итераций: {summary['num_iterations_total']}\n"
        display_text += f"С финальным кодом: {summary['num_with_final_code']}\n"
        display_text += f"Финальные тесты OK: {summary['num_final_tests_ok']}\n"
        display_text += f"Среднее время тестов: {avg_time:.2f}с\n"
        display_text += f"Средняя память: {avg_memory:.0f} КБ\n"
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, display_text)
        self.results_text.see(tk.END)

    def copy_current_results(self):
        """Копирует текущее состояние results из ScrolledText."""
        current_text = self.results_text.get(1.0, tk.END).strip()
        self.root.clipboard_clear()
        self.root.clipboard_append(current_text)

    def clear_results_text(self):
        """Очищает ScrolledText результатов."""
        self.results_text.delete(1.0, tk.END)

    def show_log(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("Предупреждение", "Выберите модель!")
            return
        model = selected[0]
        if model in self.log_texts:
            self.log_texts[model].lift()  # Поднять окно на передний план
            return
        
        log_window = tk.Toplevel(self.root)
        log_window.title(f"Полный лог для {model}")
        log_window.geometry("1000x800")
        log_window.configure(bg='white')
        self.log_texts[model] = log_window  # Сохранить ссылку
        
        # Фрейм для кнопок
        log_btn_frame = tk.Frame(log_window, bg='white')
        log_btn_frame.pack(fill=tk.X, padx=10, pady=5)
        copy_btn = tk.Button(log_btn_frame, text="Копировать весь лог", command=lambda m=model: self.copy_full_log(m, log_window), bg='lightblue', relief='raised')
        copy_btn.pack(side=tk.LEFT)
        close_btn = tk.Button(log_btn_frame, text="Закрыть", command=lambda: self.close_log_window(model), bg='lightcoral', relief='raised')
        close_btn.pack(side=tk.RIGHT)
        
        # ScrolledText с форматированием (в NORMAL для копирования)
        text = scrolledtext.ScrolledText(log_window, wrap=tk.WORD, font=('Consolas', 9), bg='black', fg='white', insertbackground='white')
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.log_texts[model + '_text'] = text  # Сохранить ссылку на text
        
        # Загрузка лога: из RAM или из файла
        if model in self.logs:
            full_log = '\n\n'.join(self.logs.get(model, []))
        else:
            log_dir = os.path.join(self.config['CONSTANTS']['INTERMEDIATE_FOLDER'], 'logs')
            safe_model = self.safe_model_name(model)
            log_file = os.path.join(log_dir, f'{safe_model}.log')
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8') as f:
                    full_log = f.read()
            else:
                full_log = "Лог не найден (возможно, очищен после завершения)"
        
        # Настройка тегов для форматирования
        text.tag_configure("header", font=('Arial', 11, 'bold'), foreground='yellow')
        text.tag_configure("stage", font=('Arial', 10, 'bold'), foreground='cyan')
        text.tag_configure("code", font=('Consolas', 9, 'italic'), foreground='green')
        text.tag_configure("error", foreground='red', font=('Arial', 9, 'bold'))
        text.tag_configure("normal", foreground='white')
        
        # Вставка существующего текста с тегами
        lines = full_log.split('\n')
        for line in lines:
            if line.startswith('=== '):
                text.insert(tk.END, line + '\n', "header")
            elif line.startswith('Этап: '):
                text.insert(tk.END, line + '\n', "stage")
            elif 'Получен' in line and 'код' in line and ':' in line:
                text.insert(tk.END, line + '\n', "code")
            elif 'Ошибка' in line or 'FAIL' in line:
                text.insert(tk.END, line + '\n', "error")
            else:
                text.insert(tk.END, line + '\n', "normal")
        
        # Блокировка редактирования (readonly mode)
        text.bind('<Key>', lambda e: 'break')
        text.bind('<Button-1>', '')  # Разрешить клик для выделения
        text.bind('<Button-2>', lambda e: 'break')  # Блокировать среднюю кнопку
        text.bind('<Delete>', lambda e: 'break')
        text.bind('<BackSpace>', lambda e: 'break')
        text.bind('<Control-v>', lambda e: 'break')  # Блокировать вставку
        text.bind('<Control-a>', lambda e: self.select_all_text(text))  # Ctrl+A для выделения всего
        
        # Стандартное копирование (Ctrl+C) работает автоматически в NORMAL state
        text.focus_set()  # Фокус на text для горячих клавиш
        
        # Контекстное меню для лога
        log_context_menu = tk.Menu(log_window, tearoff=0)
        log_context_menu.add_command(label="Копировать", command=lambda: self.copy_selection(text, log_window))
        log_context_menu.add_command(label="Выделить всё", command=lambda: self.select_all_text(text))
        
        def show_log_context_menu(event):
            try:
                log_context_menu.tk_popup(event.x_root, event.y_root)
            finally:
                log_context_menu.grab_release()
        
        text.bind('<Button-3>', show_log_context_menu)
        
        # Закрытие окна при закрытии
        log_window.protocol("WM_DELETE_WINDOW", lambda: self.close_log_window(model))
    
    def copy_selection(self, text, log_window):
        try:
            selected_text = text.selection_get()
            log_window.clipboard_clear()
            log_window.clipboard_append(selected_text)
        except tk.TclError:
            pass
    
    def select_all_text(self, text):
        text.tag_add(tk.SEL, "1.0", tk.END)
        text.mark_set(tk.INSERT, "1.0")
        text.see(tk.INSERT)
        text.focus_set()
    
    def close_log_window(self, model):
        if model in self.log_texts:
            self.log_texts[model].destroy()
            del self.log_texts[model]
        if model + '_text' in self.log_texts:
            del self.log_texts[model + '_text']
    
    def copy_full_log(self, model, log_window):
        text = self.log_texts.get(model + '_text')
        if text:
            full_text = text.get("1.0", tk.END).strip()
            log_window.clipboard_clear()
            log_window.clipboard_append(full_text)
    
    def on_closing(self):
        if messagebox.askokcancel("Выход", "Завершить?"):
            # Закрыть все лог окна
            for model in list(self.log_texts.keys()):
                if isinstance(model, str) and model.endswith('_text'):
                    win_key = model[:-5]
                    if win_key in self.log_texts:
                        self.log_texts[win_key].destroy()
            self.root.destroy()

def main():
    """
    Функция запуска модуля для тестирования.

    Получает список моделей, выбирает тестовую задачу,
    запускает GUI для мониторинга.
    """
    models = get_models_list(CONFIG)
    test_models = models
    task = """

    Task: Implement a sorting algorithm that sorts a given vector using ONLY allowed swaps.

    Input: A vector `a` of length `n` (0-indexed).

    Allowed operations:
    - swap(i, i+1) for i = 0..n-2 (adjacent swap)
    - swap(n-1, 0) — a circular swap between the last and the first element

    Strict constraints:
    - No other swaps, slicing, built-in sorting functions, or creating new arrays are allowed.
    - All swaps must be appended to the `swaps` list immediately after performing them.
    - Applying the swaps sequentially to a copy of the input vector must yield a fully sorted ascending array.
    - Circular swaps can ONLY be performed as swap(n-1, 0), never as swap(i, n-1-i) or any other non-adjacent pair.

    Critical clarification:
    - Circular swaps may be used multiple times as needed during the sorting process.
    - The sorting algorithm must continue applying swaps until the array is fully sorted.

    Requirements:
    1. Implement a function `solve(vector)` that returns a tuple `(swaps, sorted_array)`:
        - `swaps` is a list of tuples representing all swaps performed.
        - `sorted_array` is the final sorted array after applying all swaps to a copy of the input vector.
    2. Include a CLI interface:
        - When the script is executed directly, it should accept a vector as a **command-line argument**.
        - The output should be a **JSON object** with keys `"swaps"` and `"sorted_array"`.
    3. Include a minimal example in the `__main__` block for quick testing.
    4. The code must be fully self-contained and executable without external dependencies.
    5. JSON output must always be structured and parseable for automated testing.

    Example expected usage:

    ```bash
    python solve_module.py "[3,1,2,5,4]"
    """
    app = ProgressGUI(task, test_models, CONFIG)
    app.root.mainloop()


if __name__ == "__main__":
    main()
