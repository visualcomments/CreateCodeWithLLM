import requests
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import sys
from typing import Dict, List, Optional, Tuple
import time
from datetime import datetime

# Патч для RotatedProvider (используется в AnyProvider для ротации)
import g4f.providers.retry_provider as retry_mod  # Импорт модуля без кеширования класса
OriginalRotatedProvider = retry_mod.RotatedProvider  # Алиас оригинала для наследования

import g4f
from g4f import Provider

import threading
local = threading.local()

from g4f.errors import ModelNotFoundError
import queue

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
        # Промпт для генерации начального кода по задаче
        'INITIAL': (
            "Ты — помощник по программированию. Напиши корректный, рабочий и СРАЗУ ВЫПОЛНИМЫЙ Python-код для решения следующей задачи:\n\n"
            "{task}\n\n"
            "⚠️ В ответе должен быть только код, без пояснений и Markdown.\n"
            "Код должен быть самодостаточным, запускаться без ошибок и сразу выдавать результат.\n"
            "Все данные для примера должны быть **придуманы внутри кода**, без использования input() или внешних данных.\n\n"
            "Примеры правильного формата ответа:\n"
            "# Пример 1\n"
            "n = 5\n"
            "print(n * 2)\n\n"
            "# Пример 2\n"
            "def factorial(n):\n"
            "    res = 1\n"
            "    for i in range(2, n+1):\n"
            "        res *= i\n"
            "    return res\n\n"
            "print(factorial(5))\n"
        ),

        # Промпт для исправления ошибок в коде
        'FIX': (
            "Ты — помощник по отладке Python-кода. "
            "Ниже приведён код и ошибка, возникшая при его выполнении. "
            "Исправь код так, чтобы он успешно выполнялся, был самодостаточным и сразу выдавал результат.\n\n"
            "Код:\n{code}\n\n"
            "Ошибка:\n{error}\n\n"
            "⚠️ В ответе только исправленный код без пояснений и Markdown.\n"
            "Все данные для примера должны быть придуманы внутри кода, без input() или внешних источников.\n\n"
            "Примеры формата ответа:\n"
            "# Пример 1\n"
            "a, b = 2, 3\n"
            "print(a + b)\n\n"
            "# Пример 2\n"
            "def greet(name):\n"
            "    print('Hello,', name)\n\n"
            "greet('Alice')\n"
        ),

        # Промпт для первого рефакторинга без предыдущей версии
        'REFACTOR_NO_PREV': (
            "Ты — эксперт по Python. Проведи всесторонний рефакторинг кода:\n\n"
            "{code}\n\n"
            "Цели рефакторинга:\n"
            "• Улучшить читаемость и структуру.\n"
            "• Оптимизировать выполнение и алгоритмы.\n"
            "• Повысить качество и устойчивость к ошибкам.\n"
            "• Сохранить поведение и результат задачи.\n"
            "• Сделать код самодостаточным, полностью выполняемым и сразу выдающим результат.\n"
            "• Все данные должны быть придуманы внутри кода, без input() или внешних источников.\n\n"
            "⚠️ Только рефакторированный код без пояснений и Markdown.\n\n"
            "Примеры формата ответа:\n"
            "# Пример 1\n"
            "def square(x):\n"
            "    return x * x\n\n"
            "print(square(5))\n\n"
            "# Пример 2\n"
            "def is_prime(n):\n"
            "    if n < 2:\n"
            "        return False\n"
            "    if n == 2:\n"
            "        return True\n"
            "    if n % 2 == 0:\n"
            "        return False\n"
            "    i = 3\n"
            "    while i * i <= n:\n"
            "        if n % i == 0:\n"
            "            return False\n"
            "        i += 2\n"
            "    return True\n\n"
            "print(is_prime(7))\n"
        ),

        # Промпт для рефакторинга с сравнением предыдущей версии
        'REFACTOR': (
            "Ты — эксперт по Python. Сравни текущий и предыдущий варианты кода и проведи всесторонний рефакторинг:\n\n"
            "Текущий код:\n{code}\n\n"
            "Предыдущая версия:\n{prev}\n\n"
            "Цели рефакторинга:\n"
            "• Улучшить читаемость и структуру.\n"
            "• Оптимизировать алгоритмы и производительность.\n"
            "• Повысить качество, устойчивость к ошибкам и проверку корректности.\n"
            "• Исправить возможные логические ошибки.\n"
            "• Сохранить или улучшить правильность решения.\n"
            "• Сделать код самодостаточным, полностью выполняемым и сразу выдающим результат.\n"
            "• Все данные должны быть придуманы внутри кода, без input() или внешних источников.\n\n"
            "⚠️ Только рефакторированный код без пояснений и Markdown. Только самую новую исправленную версию присылай. Предыдущие версии не нужны!!!\n\n"
            "Примеры формата ответа:\n"
            "# Пример 1\n"
            "def fib(n):\n"
            "    a, b = 0, 1\n"
            "    for _ in range(n):\n"
            "        a, b = b, a+b\n"
            "    return a\n\n"
            "print(fib(10))\n\n"
            "# Пример 2\n"
            "def normalize_numbers(nums):\n"
            "    if not nums:\n"
            "        return []\n"
            "    total = sum(nums)\n"
            "    return [x / total for x in nums]\n\n"
            "nums = [1, 2, 3, 4]\n"
            "print(normalize_numbers(nums))\n"
        )
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
        'EXEC_TIMEOUT': 30,
        # Сообщение об ошибке таймаута выполнения кода
        'ERROR_TIMEOUT': 'Timeout expired',
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
    return all_models




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






def safe_execute(code: str, config: Dict) -> Tuple[bool, str]:
    """
    Безопасное выполнение Python кода через subprocess.

    Изолированное выполнение с захватом stdout/stderr, таймаутом.
    Возвращает успех и вывод/ошибку.

    Args:
        code (str): Код для выполнения.
        config (Dict): Конфигурация с 'CONSTANTS' (EXEC_TIMEOUT, ERROR_TIMEOUT).

    Returns:
        Tuple[bool, str]: (успех, вывод_или_трейсбек)

    Raises:
        subprocess.SubprocessError: Редкие системные ошибки.
    """
    try:
        result = subprocess.run(
            [sys.executable, '-c', code],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=config['CONSTANTS']['EXEC_TIMEOUT']
        )
        if result.returncode == 0:
            return True, result.stdout or ''
        else:
            return False, result.stderr or 'Unknown error'
    except subprocess.TimeoutExpired:
        return False, config['CONSTANTS']['ERROR_TIMEOUT']
    except Exception as e:
        return False, str(e)

def process_model(model: str, task: str, config: Dict, progress_queue: queue.Queue) -> Dict:
    """
    Обработка одной модели: последовательность запросов LLM, exec, fix, refactor.

    1. Initial промпт -> код, append.
    2. Exec, if error: fix промпт -> код, append.
    3. First refactor (no prev) -> код, append.
    4. Exec, if error: fix -> код, append.
    5. 3 раза: refactor (with prev) -> код, append; exec, if error: fix -> код, append.
    Если на любом LLM шаге ошибка -> append с error, continue/return early если critical.
    Использует успешный провайдер из предыдущих вызовов для приоритета.

    Args:
        model (str): Имя модели.
        task (str): Исходная задача.
        config (Dict): Конфигурация с 'PROMPTS', 'RETRIES', 'CONSTANTS', 'STAGES'.
        progress_queue (queue.Queue): Очередь для отправки обновлений прогресса.

    Returns:
        Dict: {'model': str, 'iterations': List[Dict], 'final_code': str|None}
        Где iteration: {'providers_tried': List[str], 'success_provider': str|None, 'stage': str, 'response': str|None, 'error': str|None}
    """
    iterations = []
    current_code = None
    prev_code = None
    total_stages = 1 + 1 + 1 + 1 + config['CONSTANTS']['NUM_REFACTOR_LOOPS'] * 2  # Примерное кол-во этапов
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
        progress_queue.put((model, 'log', f'Получен ответ (длина: {len(response)}):\n{response}'))
        current_code = response
        iter_entry = {
            'providers_tried': tried,
            'success_provider': success_p,
            'stage': config['STAGES']['INITIAL'],
            'response': response,
            'error': None
        }
    else:
        error_msg = config['CONSTANTS']['ERROR_NO_RESPONSE']
        progress_queue.put((model, 'log', f'Ошибка llm_query: {error_msg}'))
        iter_entry = {
            'providers_tried': tried,
            'success_provider': None,
            'stage': config['STAGES']['INITIAL'],
            'response': None,
            'error': error_msg
        }
        iterations.append(iter_entry)
        progress_queue.put((model, 'status', f'Ошибка на этапе: {config["STAGES"]["INITIAL"]}'))
        return {'model': model, 'iterations': iterations, 'final_code': None}
    iterations.append(iter_entry)
    progress_queue.put((model, 'status', f'Получен первичный код'))
    # Exec and fix if needed (before first refactor)
    progress_queue.put((model, 'status', 'Выполнение первичного кода...'))
    progress_queue.put((model, 'log', f'Этап: Выполнение первичного кода (длина: {len(current_code)}):\n{current_code}'))
    success_exec, output = safe_execute(current_code, config)
    current_stage += 1
    progress_queue.put((model, 'progress', (current_stage, total_stages)))
    progress_queue.put((model, 'log', f'Результат safe_execute: success={success_exec}, output:\n{output}'))
    fix_stage = config['STAGES']['FIX_INITIAL']
    if not success_exec or not output.strip():
        progress_queue.put((model, 'log', f'Обнаружена ошибка выполнения. Этап исправления: {fix_stage}'))
        prompt = config['PROMPTS']['FIX'].format(code=current_code, error=str(output))
        progress_queue.put((model, 'log', f'Промпт для исправления (полный):\n{prompt}'))
        response = llm_query(model, prompt, config['RETRIES']['FIX'], config, progress_queue, fix_stage)
        current_stage += 1
        progress_queue.put((model, 'progress', (current_stage, total_stages)))
        success_p = local.current_data['success']
        tried = local.current_data['tried']
        if response:
            progress_queue.put((model, 'log', f'Получен исправленный код (длина: {len(response)}):\n{response}'))
            current_code = response
            error_fix = None
        else:
            error_fix = config['CONSTANTS']['ERROR_NO_RESPONSE']
            progress_queue.put((model, 'log', f'Ошибка llm_query для fix: {error_fix}'))
            iter_entry = {
                'providers_tried': tried,
                'success_provider': None,
                'stage': fix_stage,
                'response': None,
                'error': error_fix
            }
            iterations.append(iter_entry)
            progress_queue.put((model, 'status', f'Ошибка исправления: {fix_stage}'))
            return {'model': model, 'iterations': iterations, 'final_code': None}
        iter_entry = {
            'providers_tried': tried,
            'success_provider': success_p,
            'stage': fix_stage,
            'response': current_code,
            'error': error_fix
        }
        iterations.append(iter_entry)
        progress_queue.put((model, 'status', f'Код исправлен до рефакторинга'))
    # First refactor (no prev)
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
        progress_queue.put((model, 'log', f'Получен рефакторированный код (длина: {len(response)}):\n{response}'))
        prev_code = current_code
        current_code = response
        error_ref = None
        iter_entry = {
            'providers_tried': tried,
            'success_provider': success_p,
            'stage': refactor_stage,
            'response': current_code,
            'error': error_ref
        }
        iterations.append(iter_entry)
        progress_queue.put((model, 'status', 'Первый рефакторинг завершен'))
    else:
        error_ref = config['CONSTANTS']['ERROR_NO_RESPONSE']
        progress_queue.put((model, 'log', f'Ошибка llm_query для refactor: {error_ref}'))
        iter_entry = {
            'providers_tried': tried,
            'success_provider': None,
            'stage': refactor_stage,
            'response': None,
            'error': error_ref
        }
        iterations.append(iter_entry)
        progress_queue.put((model, 'status', f'Ошибка рефакторинга: {refactor_stage}'))
        return {'model': model, 'iterations': iterations, 'final_code': current_code}
    # Exec after first refactor and fix if needed
    progress_queue.put((model, 'status', 'Выполнение после первого рефакторинга...'))
    progress_queue.put((model, 'log', f'Этап: Выполнение после рефакторинга (длина: {len(current_code)}):\n{current_code}'))
    success_exec, output = safe_execute(current_code, config)
    current_stage += 1
    progress_queue.put((model, 'progress', (current_stage, total_stages)))
    progress_queue.put((model, 'log', f'Результат safe_execute после refactor: success={success_exec}, output:\n{output}'))
    fix_stage = config['STAGES']['FIX_AFTER_REFACTOR']
    if not success_exec or not output.strip():
        progress_queue.put((model, 'log', f'Обнаружена ошибка после рефакторинга. Этап исправления: {fix_stage}'))
        prompt = config['PROMPTS']['FIX'].format(code=current_code, error=str(output))
        progress_queue.put((model, 'log', f'Промпт для исправления (полный):\n{prompt}'))
        response = llm_query(model, prompt, config['RETRIES']['FIX'], config, progress_queue, fix_stage)
        current_stage += 1
        progress_queue.put((model, 'progress', (current_stage, total_stages)))
        success_p = local.current_data['success']
        tried = local.current_data['tried']
        if response:
            progress_queue.put((model, 'log', f'Получен исправленный код после refactor (длина: {len(response)}):\n{response}'))
            current_code = response
            error_fix = None
        else:
            error_fix = config['CONSTANTS']['ERROR_NO_RESPONSE']
            progress_queue.put((model, 'log', f'Ошибка llm_query для fix после refactor: {error_fix}'))
            iter_entry = {
                'providers_tried': tried,
                'success_provider': None,
                'stage': fix_stage,
                'response': None,
                'error': error_fix
            }
            iterations.append(iter_entry)
            progress_queue.put((model, 'status', f'Ошибка: {fix_stage}'))
            return {'model': model, 'iterations': iterations, 'final_code': None}
        iter_entry = {
            'providers_tried': tried,
            'success_provider': success_p,
            'stage': fix_stage,
            'response': current_code,
            'error': error_fix
        }
        iterations.append(iter_entry)
        progress_queue.put((model, 'status', 'Код исправлен после рефакторинга'))
    # Loop 3 times: refactor with prev, then exec + fix if fail
    loops_left = config['CONSTANTS']['NUM_REFACTOR_LOOPS']
    for loop in range(config['CONSTANTS']['NUM_REFACTOR_LOOPS']):
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
            progress_queue.put((model, 'log', f'Получен рефакторированный код в цикле (длина: {len(response)}):\n{response}'))
            prev_code = current_code
            current_code = response
            error_ref = None
        else:
            error_ref = config['CONSTANTS']['ERROR_NO_RESPONSE']
            progress_queue.put((model, 'log', f'Ошибка llm_query для refactor в цикле: {error_ref}'))
            iter_entry = {
                'providers_tried': tried,
                'success_provider': None,
                'stage': refactor_stage,
                'response': None,
                'error': error_ref
            }
            iterations.append(iter_entry)
            progress_queue.put((model, 'status', f'Ошибка рефакторинга в цикле'))
            continue  # skip exec if no refactor
        iter_entry = {
            'providers_tried': tried,
            'success_provider': success_p,
            'stage': refactor_stage,
            'response': current_code,
            'error': error_ref
        }
        iterations.append(iter_entry)
        # Exec and fix if needed
        progress_queue.put((model, 'status', 'Выполнение в цикле...'))
        progress_queue.put((model, 'log', f'Этап: Выполнение в цикле (длина: {len(current_code)}):\n{current_code}'))
        success_exec, output = safe_execute(current_code, config)
        current_stage += 1
        progress_queue.put((model, 'progress', (current_stage, total_stages)))
        progress_queue.put((model, 'log', f'Результат safe_execute в цикле: success={success_exec}, output:\n{output}'))
        fix_stage = config['STAGES']['FIX_LOOP']
        if not success_exec or not output.strip():
            progress_queue.put((model, 'log', f'Обнаружена ошибка в цикле. Этап исправления: {fix_stage}'))
            prompt = config['PROMPTS']['FIX'].format(code=current_code, error=str(output))
            progress_queue.put((model, 'log', f'Промпт для исправления в цикле (полный):\n{prompt}'))
            response = llm_query(model, prompt, config['RETRIES']['FIX'], config, progress_queue, fix_stage)
            current_stage += 1
            progress_queue.put((model, 'progress', (current_stage, total_stages)))
            success_p = local.current_data['success']
            tried = local.current_data['tried']
            if response:
                progress_queue.put((model, 'log', f'Получен исправленный код в цикле (длина: {len(response)}):\n{response}'))
                current_code = response
                error_fix = None
            else:
                error_fix = config['CONSTANTS']['ERROR_NO_RESPONSE']
                progress_queue.put((model, 'log', f'Ошибка llm_query для fix в цикле: {error_fix}'))
                iter_entry = {
                    'providers_tried': tried,
                    'success_provider': None,
                    'stage': fix_stage,
                    'response': None,
                    'error': error_fix
                }
                iterations.append(iter_entry)
                progress_queue.put((model, 'status', f'Ошибка исправления в цикле'))
                continue
            iter_entry = {
                'providers_tried': tried,
                'success_provider': success_p,
                'stage': fix_stage,
                'response': current_code,
                'error': error_fix
            }
            iterations.append(iter_entry)
            progress_queue.put((model, 'status', 'Код исправлен в цикле'))
    progress_queue.put((model, 'status', 'Завершено'))
    progress_queue.put((model, 'log', f'=== ФИНАЛЬНЫЙ КОД (длина: {len(current_code or "")}):\n{current_code or "None"}'))
    progress_queue.put((model, 'log', f'=== КОНЕЦ ОБРАБОТКИ МОДЕЛИ: {model} ==='))
    progress_queue.put((model, 'done', None))
    return {'model': model, 'iterations': iterations, 'final_code': current_code}

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
        results_frame = ttk.LabelFrame(main_frame, text="Текущее состояние results (для JSON)", padding=10)
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
                    self.logs[model].append(data)
                    # Обновление в открытом лог-окне, если оно существует
                    if model in self.log_texts:
                        text = self.log_texts[model + '_text']
                        line = data + '\n'
                        if data.startswith('=== '):
                            text.insert(tk.END, line, "header")
                        elif data.startswith('Этап: '):
                            text.insert(tk.END, line, "stage")
                        elif 'Получен' in data and 'код' in data and ':' in data:
                            text.insert(tk.END, line, "code")
                        elif 'Ошибка' in data:
                            text.insert(tk.END, line, "error")
                        else:
                            text.insert(tk.END, line, "normal")
                        text.see(tk.END)  # Автопрокрутка к концу
                elif msg_type == 'done':
                    if model == 'global':
                        messagebox.showinfo("Завершено", "Все модели обработаны!")
                    else:
                        self.tree.set(model, 'Status', 'Завершено')
                elif msg_type == 'error':
                    self.tree.set(model, 'Status', f'Ошибка: {data}')
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
        display_text = f"Обновление results ({datetime.now().strftime('%H:%M:%S')}):\n"
        display_text += f"Завершено моделей: {summary['completed']}/{summary['total']}\n"
        display_text += f"Обработано моделей: {len(summary['models_done'])}\n"
        display_text += f"Общее итераций: {summary['num_iterations_total']}\n"
        display_text += f"С финальным кодом: {summary['num_with_final_code']}\n\n"
        display_text += "Текущее состояние results (JSON для копирования):\n"
        display_text += json.dumps({'partial_results': summary['partial_results']}, ensure_ascii=False, indent=2)
        
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
        
        # Настройка тегов для форматирования
        text.tag_configure("header", font=('Arial', 11, 'bold'), foreground='yellow')
        text.tag_configure("stage", font=('Arial', 10, 'bold'), foreground='cyan')
        text.tag_configure("code", font=('Consolas', 9, 'italic'), foreground='green')
        text.tag_configure("error", foreground='red', font=('Arial', 9, 'bold'))
        text.tag_configure("normal", foreground='white')
        
        # Вставка существующего текста с тегами
        full_log = '\n\n'.join(self.logs.get(model, []))
        lines = full_log.split('\n')
        for line in lines:
            if line.startswith('=== '):
                text.insert(tk.END, line + '\n', "header")
            elif line.startswith('Этап: '):
                text.insert(tk.END, line + '\n', "stage")
            elif 'Получен' in line and 'код' in line and ':' in line:
                text.insert(tk.END, line + '\n', "code")
            elif 'Ошибка' in line:
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
    task = "Напиши функцию на Python, которая вычисляет сумму элементов в списке."
    app = ProgressGUI(task, test_models, CONFIG)
    app.root.mainloop()

if __name__ == "__main__":
    main()
