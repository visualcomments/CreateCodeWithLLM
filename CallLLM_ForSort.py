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
        'WORKING_RESULTS': '[https://raw.githubusercontent.com/maruf009sultan/g4f-working/refs/heads/main/working/working_results.txt](https://raw.githubusercontent.com/maruf009sultan/g4f-working/refs/heads/main/working/working_results.txt)'
    },

    # Раздел с промптами для различных этапов взаимодействия с LLM
    'PROMPTS': {
        'INITIAL': r"""
You are an AI assistant. Your task is to write code that implements the three fundamental transformations of the "LRX algorithm". These transformations operate on a permutation of $n$ elements.

Based on the definitions:
L (Left Cyclic Shift): [e_1, e_2, ..., e_n] becomes [e_2, ..., e_n, e_1]
R (Right Cyclic Shift): [e_1, e_2, ..., e_n] becomes [e_n, e_1, ..., e_{n-1}]
X (Transposition): [e_1, e_2, ..., e_n] becomes [e_2, e_1, ..., e_n]

⚠️ Requirements:
• Implement three distinct functions `L(vector)`, `R(vector)`, and `X(vector)`.
• Each function must accept a list/array and return the **new** list/array. The original list must not be modified.
• Include `import json` and a CLI entry point: when executed, parse `sys.argv[1]` as JSON vector, fallback [1, 2, 3, 4] if missing or invalid.
• In the `__main__` block, call L, R, and X on the input vector.
• Print only one JSON object: `{{"L_result": L(vector), "R_result": R(vector), "X_result": X(vector)}}`.
• The JSON output must be structured and parseable (double quotes for keys).
• Fully self-contained and immediately executable.
• Only code in the response, no explanations or Markdown.
""",

        'FIX': r"""
You are a Python debugging assistant. The following code did not work correctly.
Fix it to run correctly, follow the `L, R, X` function interfaces, and produce **only CLI JSON output**.

Code:
{code}

Issue:
{error}

⚠️ Requirements:
• `L(vector)`, `R(vector)`, `X(vector)` must return **new** lists, not modify the input.
• CLI: `import json`; parse `sys.argv[1]` with fallback [1, 2, 3, 4]; print only `json.dumps({{"L_result": L(vector), "R_result": R(vector), "X_result": X(vector)}})`.
• Use try-except to catch missing arguments or invalid JSON.
• Self-contained, executable, immediately correct.
• Only code in response, no extra prints or Markdown.
""",

        'REFACTOR_NO_PREV': r"""
You are an expert Python programmer. Refactor the following code:

{code}

⚠️ Goals:
• Improve readability, structure, and efficiency of `L`, `R`, `X` (e.g., using slicing for new lists).
• Ensure functions return **new** lists and do not modify the input.
• Preserve CLI: parse `sys.argv[1]` as JSON with fallback [1, 2, 3, 4], print only `json.dumps({{"L_result": L(vector), "R_result": R(vector), "X_result": X(vector)}})`.
• Minimal example in __main__ must print JSON only.
• Fully executable, immediately correct.
• Only code in response, no explanations or Markdown.
""",

        'REFACTOR': r"""
You are an expert Python programmer. Compare the current and previous versions and perform a full refactor:

Current code:
{code}

Previous version:
{prev}

⚠️ Goals:
• Improve readability, structure, and efficiency of `L`, `R`, `X`.
• Ensure functions return **new** lists and do not modify the input.
• Preserve CLI: parse `sys.argv[1]` as JSON with fallback [1, 2, 3, 4]; print only `json.dumps({{"L_result": L(vector), "R_result": R(vector), "X_result": X(vector)}})`.
• Minimal example in __main__ must print JSON only.
• Code must pass verification (e.g., L([1,2,3]) == [2,3,1], R([1,2,3]) == [3,1,2], X([1,2,3]) == [2,1,3]).
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
        'ERROR_TIMEOUT': 'Timeout expired — the program likely entered an infinite loop.',
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


# =============================================================================
# === ОБНОВЛЕННАЯ ФУНКЦИЯ test_code() ===
# =============================================================================

def test_code(code: str, config: Dict) -> Tuple[bool, str, Optional[Dict]]:
    """
    Тестирование кода (LRX) с различными векторами, включая пограничные случаи.

    Для каждого вектора: запускает subprocess с arg=json.dumps(vector),
    парсит JSON {L_result, R_result, X_result},
    сравнивает с эталонными результатами L, R, X.
    Измеряет время per test, max memory approx via psutil (cross-platform).

    Args:
        code (str): Тестируемый код.
        config (Dict): Конфиг с EXEC_TIMEOUT.

    Returns:
        Tuple[bool, str, Optional[Dict]]: (all_success, issue_str or 'All tests passed', summary)
        summary: {'all_success', 'total_time', 'max_memory_kb', 'tests': [list of test dicts], 'num_failing'}
    """
    
    # --- Эталонные (Ground Truth) реализации LRX для проверки ---
    def _expected_L(v):
        """Эталонный левый сдвиг"""
        if not v:
            return []
        return v[1:] + v[:1]

    def _expected_R(v):
        """Эталонный правый сдвиг"""
        if not v:
            return []
        return v[-1:] + v[:-1]

    def _expected_X(v):
        """Эталонная транспозиция"""
        if len(v) < 2:
            return v[:]
        # Создаем новый список, комбинируя поменяные 
        # первые два элемента и остаток списка
        return [v[1], v[0]] + v[2:]
    # ---

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
        
        # Ожидаемые результаты (вычисляем до запуска)
        exp_l = _expected_L(vector)
        exp_r = _expected_R(vector)
        exp_x = _expected_X(vector)
        
        # Словарь-заготовка для ошибки
        error_result_dict = {
            'n': n,
            'success': False,
            'time': 0.0,
            'input': vector,
            'L_result': None, 'expected_L': exp_l,
            'R_result': None, 'expected_R': exp_r,
            'X_result': None, 'expected_X': exp_x
        }

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
            error_result_dict['time'] = elapsed # Обновляем время даже в случае ошибки

            if child_process.returncode != 0:
                err = stderr or 'Unknown error'
                error_result_dict['error'] = err
                return error_result_dict, err, False

            output = stdout.strip()
            if not output:
                err = 'No output'
                error_result_dict['error'] = err
                return error_result_dict, err, False

            try:
                parsed = json.loads(output)
                
                # --- НОВЫЙ БЛОК ПРОВЕРКИ LRX ---
                l_res = parsed.get('L_result')
                r_res = parsed.get('R_result')
                x_res = parsed.get('X_result')

                success_l = (l_res == exp_l)
                success_r = (r_res == exp_r)
                success_x = (x_res == exp_x)
                success = success_l and success_r and success_x

                error_msg = None
                if not success:
                    errors = []
                    if not success_l: errors.append(f"L mismatch: got {l_res}, expected {exp_l}")
                    if not success_r: errors.append(f"R mismatch: got {r_res}, expected {exp_r}")
                    if not success_x: errors.append(f"X mismatch: got {x_res}, expected {exp_x}")
                    error_msg = "; ".join(errors)
                # --- КОНЕЦ НОВОГО БЛОКА ---

                res_dict = {
                    'n': n,
                    'success': success,
                    'time': elapsed,
                    'input': vector,
                    'L_result': l_res,
                    'expected_L': exp_l,
                    'R_result': r_res,
                    'expected_R': exp_r,
                    'X_result': x_res,
                    'expected_X': exp_x
                }
                if error_msg:
                    res_dict['error'] = error_msg
                return res_dict, error_msg, success

            except json.JSONDecodeError as je:
                err = f'JSON parse error: {je}. Output was: {output}'
                error_result_dict['error'] = err
                return error_result_dict, err, False
            except KeyError as ke: 
                err = f'KeyError: {ke}. JSON output missing expected keys. Output was: {output}'
                error_result_dict['error'] = err
                # Заполняем тем, что смогли спарсить
                error_result_dict['L_result'] = parsed.get('L_result')
                error_result_dict['R_result'] = parsed.get('R_result')
                error_result_dict['X_result'] = parsed.get('X_result')
                return error_result_dict, err, False

        except subprocess.TimeoutExpired:
            err = config['CONSTANTS']['ERROR_TIMEOUT']
            if child_process:
                child_process.kill()
            error_result_dict['error'] = err
            error_result_dict['time'] = exec_timeout # Время = таймаут
            return error_result_dict, err, False
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

    # --- ОБНОВЛЕННЫЕ ВЕКТОРЫ для LRX ---
    specific_vectors = [
        [],                      # n=0
        [1],                     # n=1
        [1, 2],                  # n=2
        [1, 2, 3],               # n=3
        [3, 1, 2],
        [5, 2, 4, 1, 3],
        [48, 18, 44, 20, 16, 61, 26],
        [1, 2, 3, 4], # Базовый вектор из промпта
    ]
    # ---

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
            # --- ОБНОВЛЕННЫЙ ОТЧЕТ ОБ ОШИБКЕ ---
            failing_case = {
                'n': res_dict['n'],
                'input': res_dict['input'],
                'error': res_dict.get('error', 'Unknown failure'),
                'L_result': res_dict.get('L_result'),
                'expected_L': res_dict.get('expected_L'),
                'R_result': res_dict.get('R_result'),
                'expected_R': res_dict.get('expected_R'),
                'X_result': res_dict.get('X_result'),
                'expected_X': res_dict.get('expected_X'),
            }
            # ---
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

# =============================================================================
# === КОНЕЦ ОБНОВЛЕННОЙ ФУНКЦИИ test_code() ===
# =============================================================================


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
        task (str): Исходная задача (используется только в INITIAL, если {task} там есть).
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
    
    # Считаем общее количество этапов для progress bar
    # 1 (Initial) + 1 (Test/Fix) + 1 (Refactor1) + 1 (Test/Fix) + N*(Refactor + Test/Fix) + 1 (Final Test)
    num_loops = config['CONSTANTS']['NUM_REFACTOR_LOOPS']
    total_stages = 1 + 1 + 1 + 1 + (num_loops * 2) + 1
    current_stage_count = 0

    def update_progress(stage_name):
        nonlocal current_stage_count
        current_stage_count += 1
        progress_queue.put((model, 'status', f'Этап: {stage_name} ({current_stage_count}/{total_stages})'))
        progress_queue.put((model, 'progress', (current_stage_count, total_stages)))

    def run_test(code_to_test, stage_name):
        """Внутренняя функция для тестирования и логирования."""
        if not code_to_test or not code_to_test.strip():
            progress_queue.put((model, 'log', f'Тест {stage_name}: Пропущен (нет кода).'))
            return False, "No code to test", None
            
        progress_queue.put((model, 'log', f'Тест {stage_name}: Запуск test_code...'))
        success, issue, summary = test_code(code_to_test, config)
        if success:
            progress_queue.put((model, 'log', f'Тест {stage_name}: УСПЕХ. {issue}'))
        else:
            progress_queue.put((model, 'log', f'Тест {stage_name}: ОШИБКА. Причина: {issue}'))
        return success, issue, summary

    def run_llm_query(prompt, stage_name, retries_key='FIX'):
        """Внутренняя функция для LLM-запроса и логирования."""
        progress_queue.put((model, 'log', f'Этап: {stage_name}. Промпт:\n{prompt}'))
        retries_cfg = config['RETRIES'][retries_key]
        progress_queue.put((model, 'log', f'Вызов llm_query с retries: {retries_cfg}'))
        
        response = llm_query(model, prompt, retries_cfg, config, progress_queue, stage_name)
        
        tried = local.current_data.get('tried', [])
        success_p = local.current_data.get('success', None)
        
        if response:
            cleaned = clean_code(response)
            progress_queue.put((model, 'log', f'Получен ответ (длина: {len(response)}), очищенный (длина: {len(cleaned)}):\n{cleaned}'))
            return cleaned, None, tried, success_p
        else:
            error_msg = config['CONSTANTS']['ERROR_NO_RESPONSE']
            progress_queue.put((model, 'log', f'Ошибка llm_query: {error_msg}'))
            return None, error_msg, tried, success_p

    def add_iteration(stage, response, error, test_summary, tried, success_p):
        """Добавляет запись в историю итераций."""
        iterations.append({
            'providers_tried': tried,
            'success_provider': success_p,
            'stage': stage,
            'response': response, # Сохраняем код (или None)
            'error': error, # Ошибка LLM или теста
            'test_summary': test_summary # Результат test_code
        })

    # ---
    # НАЧАЛО ПРОЦЕССА
    # ---
    progress_queue.put((model, 'log', f'=== НАЧАЛО ОБРАБОТКИ МОДЕЛИ: {model} ==='))
    
    # 1. Initial
    stage = config['STAGES']['INITIAL']
    update_progress(stage)
    # ПРИМЕЧАНИЕ: {task} больше не используется в новых промптах, но мы 
    # оставляем `task` в format() на случай, если он вернется в конфиг.
    prompt = config['PROMPTS']['INITIAL'].format(task=task) 
    
    current_code, llm_error, tried, s_provider = run_llm_query(prompt, stage, 'INITIAL')
    add_iteration(stage, current_code, llm_error, None, tried, s_provider)
    
    if llm_error:
        progress_queue.put((model, 'status', f'Ошибка на этапе: {stage}'))
        return {'model': model, 'iterations': iterations, 'final_code': None,
                'final_test': {'success': False, 'summary': None, 'issue': 'No initial response'}}

    # 2. Test & Fix (Initial)
    stage = config['STAGES']['FIX_INITIAL']
    update_progress(stage)
    success, issue, summary = run_test(current_code, stage)
    
    if not success:
        prompt = config['PROMPTS']['FIX'].format(code=current_code, error=issue)
        current_code, llm_error, tried, s_provider = run_llm_query(prompt, stage)
        add_iteration(stage, current_code, llm_error, summary, tried, s_provider) # Добавляем попытку FIX
        if llm_error:
            early_stop = True # Ошибка LLM при исправлении = стоп
    else:
        # Добавляем запись об успешном тесте, даже если FIX не требовался
        add_iteration(stage, current_code, None, summary, [], None) 

    if early_stop:
        progress_queue.put((model, 'status', f'Ошибка на этапе: {stage}'))
        return {'model': model, 'iterations': iterations, 'final_code': current_code,
                'final_test': {'success': False, 'summary': summary, 'issue': f'LLM error during {stage}'}}

    # 3. Refactor (First)
    prev_code = current_code
    stage = config['STAGES']['REFACTOR_FIRST']
    update_progress(stage)
    prompt = config['PROMPTS']['REFACTOR_NO_PREV'].format(code=current_code)
    
    current_code, llm_error, tried, s_provider = run_llm_query(prompt, stage, 'INITIAL') # Используем 'INITIAL' retries
    add_iteration(stage, current_code, llm_error, None, tried, s_provider)

    if llm_error:
        current_code = prev_code # Откатываемся, если рефакторинг не удался
        progress_queue.put((model, 'log', f'Ошибка {stage}, откат к предыдущей версии кода.'))
    
    # 4. Test & Fix (After Refactor 1)
    stage = config['STAGES']['FIX_AFTER_REFACTOR']
    update_progress(stage)
    success, issue, summary = run_test(current_code, stage)
    
    if not success:
        prompt = config['PROMPTS']['FIX'].format(code=current_code, error=issue)
        current_code, llm_error, tried, s_provider = run_llm_query(prompt, stage)
        add_iteration(stage, current_code, llm_error, summary, tried, s_provider)
        if llm_error:
            early_stop = True
    else:
        add_iteration(stage, current_code, None, summary, [], None)

    if early_stop:
        progress_queue.put((model, 'status', f'Ошибка на этапе: {stage}'))
        return {'model': model, 'iterations': iterations, 'final_code': current_code,
                'final_test': {'success': False, 'summary': summary, 'issue': f'LLM error during {stage}'}}

    # 5. Refactor Loops (N times)
    for i in range(config['CONSTANTS']['NUM_REFACTOR_LOOPS']):
        if not current_code or not current_code.strip():
            progress_queue.put((model, 'log', f'Пропуск цикла рефакторинга {i+1} (нет кода).'))
            update_progress(f'loop {i+1} refactor (skip)') # Пропускаем 2 этапа
            update_progress(f'loop {i+1} fix (skip)')
            continue
            
        # 5a. Refactor
        stage = f"{config['STAGES']['REFACTOR']}_{i+1}"
        update_progress(stage)
        
        prompt = config['PROMPTS']['REFACTOR'].format(code=current_code, prev=prev_code)
        prev_code = current_code # Сохраняем для следующего цикла
        
        current_code, llm_error, tried, s_provider = run_llm_query(prompt, stage, 'INITIAL')
        add_iteration(stage, current_code, llm_error, None, tried, s_provider)

        if llm_error:
            current_code = prev_code # Откат
            progress_queue.put((model, 'log', f'Ошибка {stage}, откат к предыдущей версии кода.'))
        
        # 5b. Test & Fix
        stage = f"{config['STAGES']['FIX_LOOP']}_{i+1}"
        update_progress(stage)
        success, issue, summary = run_test(current_code, stage)
        
        if not success:
            prompt = config['PROMPTS']['FIX'].format(code=current_code, error=issue)
            current_code, llm_error, tried, s_provider = run_llm_query(prompt, stage)
            add_iteration(stage, current_code, llm_error, summary, tried, s_provider)
            if llm_error:
                progress_queue.put((model, 'status', f'Ошибка на этапе: {stage}, ОСТАНОВКА ЦИКЛА'))
                break # Прерываем цикл рефакторинга, если LLM не смог исправить
        else:
            add_iteration(stage, current_code, None, summary, [], None)

    # 6. Final Test
    stage = 'final_test'
    update_progress(stage)
    success, issue, summary = run_test(current_code, stage)
    
    # Добавляем финальную запись (она дублирует final_test в ответе, но полезна для истории)
    add_iteration(stage, current_code, None if success else issue, summary, [], None)

    if success:
        progress_queue.put((model, 'log', f'ФИНАЛ: УСПЕХ. {issue}'))
        progress_queue.put((model, 'status', 'Успех (финальный тест)'))
    else:
        progress_queue.put((model, 'log', f'ФИНАЛ: ОШИБКА. Причина: {issue}'))
        progress_queue.put((model, 'status', 'Ошибка (финальный тест)'))

    return {
        'model': model,
        'iterations': iterations,
        'final_code': current_code,
        'final_test': {'success': success, 'summary': summary, 'issue': issue}
    }


def save_results(results, folder, filename):
    """Безопасное сохранение результатов в JSON."""
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except OSError as e:
            print(f"Ошибка создания папки {folder}: {e}")
            return
    path = os.path.join(folder, filename)
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Ошибка сохранения файла {path}: {e}")

def main():
    """Главная функция: загрузка моделей, запуск потоков, обработка результатов."""
    print("Загрузка списка моделей...")
    try:
        models = get_models_list(CONFIG)
        if not models:
            print("Не найдено ни одной модели. Проверьте URLS и g4f.models.")
            return
        print(f"Найдено {len(models)} уникальных моделей для тестирования.")
        # print(models) # Раскомментируйте для просмотра списка
    except Exception as e:
        print(f"Не удалось загрузить список моделей: {e}")
        return

    # Убедимся, что папка для результатов существует
    intermediate_folder = CONFIG['CONSTANTS']['INTERMEDIATE_FOLDER']
    if not os.path.exists(intermediate_folder):
        try:
            os.makedirs(intermediate_folder)
        except OSError as e:
            print(f"Не удалось создать папку {intermediate_folder}: {e}")
            return

    progress_queue = queue.Queue()
    all_results = {}
    
    # 'task' больше не используется в промптах LRX, но мы передаем ее
    # для совместимости с интерфейсом process_model
    task_description = "LRX Algorithm Implementation"

    # Ограничим количество моделей для примера (установите -1 для всех)
    MAX_MODELS_TO_TEST = -1 # -1 для всех, 10 для быстрого теста
    
    if MAX_MODELS_TO_TEST > 0:
        models_to_test = models[:MAX_MODELS_TO_TEST]
        print(f"--- НАЧАЛО ТЕСТИРОВАНИЯ (Ограничено до {len(models_to_test)} моделей) ---")
    else:
        models_to_test = models
        print(f"--- НАЧАЛО ТЕСТИРОВАНИЯ (Все {len(models_to_test)} моделей) ---")


    try:
        with ThreadPoolExecutor(max_workers=CONFIG['CONSTANTS']['MAX_WORKERS']) as executor:
            # Запускаем задачи
            futures = {executor.submit(process_model, model, task_description, CONFIG, progress_queue): model for model in models_to_test}
            
            completed_count = 0
            total_count = len(futures)
            
            start_time_main = perf_counter()

            while completed_count < total_count:
                # 1. Проверяем завершенные задачи
                done_futures = [f for f in futures if f.done()]
                
                for future in done_futures:
                    model = futures.pop(future)
                    completed_count += 1
                    try:
                        result = future.result()
                        all_results[model] = result
                        
                        final_success = result.get('final_test', {}).get('success', False)
                        status_str = "УСПЕХ" if final_success else "ОШИБКА"

                        # Сохраняем код, если он есть
                        if result.get('final_code'):
                            code_filename = f"{model.replace('/', '_')}_final.py"
                            code_path = os.path.join(intermediate_folder, code_filename)
                            try:
                                with open(code_path, 'w', encoding='utf-8') as f:
                                    f.write(result['final_code'])
                            except Exception as e:
                                print(f"Ошибка сохранения кода для {model}: {e}")
                                
                        print(f"--- ({completed_count}/{total_count}) ЗАВЕРШЕНО: {model} [Статус: {status_str}] ---")
                        
                    except Exception as e:
                        print(f"--- ({completed_count}/{total_count}) КРИТ. ОШИБКА (Executor): {model} -> {e} ---")
                        all_results[model] = {'error': str(e), 'iterations': [], 'final_code': None, 'final_test': {'success': False, 'summary': None, 'issue': str(e)}}
                    
                    # Промежуточное сохранение
                    if completed_count % CONFIG['CONSTANTS']['N_SAVE'] == 0 or completed_count == total_count:
                        save_results(all_results, intermediate_folder, f"intermediate_results_{completed_count}.json")

                # 2. Обрабатываем очередь логов (с таймаутом, чтобы не блокировать)
                try:
                    while not progress_queue.empty():
                        model, type, message = progress_queue.get_nowait()
                        # Можно раскомментировать для ОЧЕНЬ подробных логов
                        # if type == 'log':
                        #     print(f"LOG [{model}]: {message}")
                        # elif type == 'status':
                        #     print(f"STATUS [{model}]: {message}")
                
                except queue.Empty:
                    pass # Очередь пуста, это нормально
                
                time.sleep(0.2) # Небольшая пауза, чтобы не загружать ЦП

    except KeyboardInterrupt:
        print("\nОстановка по требованию пользователя... (Ожидание завершения текущих потоков)")
        # Примечание: ThreadPoolExecutor нелегко прервать, 
        # он завершит уже запущенные задачи.
    finally:
        end_time_main = perf_counter()
        total_time_main = end_time_main - start_time_main
        print("--- ТЕСТИРОВАНИЕ ЗАВЕРШЕНО ---")
        print(f"Общее время выполнения: {total_time_main:.2f} сек.")
        
        # Финальное сохранение
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_filename = f"final_results_{timestamp}.json"
        save_results(all_results, intermediate_folder, final_filename)
        print(f"Финальные результаты сохранены в: {os.path.join(intermediate_folder, final_filename)}")
        
        # Подведение итогов
        success_count = sum(1 for res in all_results.values() if res.get('final_test', {}).get('success', False))
        fail_count = len(all_results) - success_count
        print(f"Итоги: {success_count} Успешно, {fail_count} С ошибками.")


if __name__ == "__main__":
    # Этот блок __main__ предназначен для запуска самого скрипта тестирования (main()).
    # Код, который генерируют LLM, также имеет свой блок __main__, 
    # который выполняется, когда test_code() запускает его как subprocess.
    main()